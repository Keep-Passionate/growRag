"""Frozen-evaluation coordinator tests: synthetic files, no API/gold/shutdown."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments.representation_runner import fingerprint

PATH = Path(__file__).resolve().parents[1] / "scripts/run_operator_pipeline.py"
SPEC = importlib.util.spec_from_file_location("operator_pipeline_resume", PATH)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)

IDS = [f"{index:024x}" for index in range(500)]


def options(*extra, resume=True, start=300):
    argv = [
        "--manifest",
        "data/manifest.json",
        "--expected-manifest-sha256",
        "a" * 64,
        "--expected-execution-sha256",
        "b" * 64,
        "--output-dir",
        "runs/resume_pipeline",
    ]
    if resume:
        argv += [
            "--resume-after-freeze",
            "--evaluation-start",
            str(start),
            "--expected-freeze-sha256",
            "c" * 64,
        ]
    return pipeline.parser().parse_args([*argv, *extra])


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def forbidden(*args, **kwargs):
    pytest.fail("unexpected API, subprocess, gold, or shutdown boundary")


@pytest.mark.parametrize("start", [0, 300, 499])
def test_resume_stages_do_not_repeat_source_or_rebuild_freeze(start):
    assert tuple(pipeline.stage_names(options(start=start))) == (
        "evaluation500",
        "evaluation_preflight",
        "evaluation_score",
    )


def test_fully_terminal_500_only_preflights_then_scores():
    assert tuple(pipeline.stage_names(options(start=500))) == (
        "evaluation_preflight",
        "evaluation_score",
    )


@pytest.mark.parametrize(
    "extra",
    [
        ("--source-start", "300"),
        ("--source-resume-certificate", "runs/source.json"),
        ("--evaluation-start", "-1"),
        ("--evaluation-start", "501"),
        ("--expected-freeze-sha256", "not-a-sha"),
    ],
)
def test_resume_rejects_mixed_source_parameters_and_invalid_bounds(extra):
    with pytest.raises(ValueError):
        pipeline.stage_names(options(*extra))


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("--source-start", "500"),
        ("--source-start", "0", "--evaluation-start", "300"),
        ("--source-start", "0", "--evaluation-resume-certificate", "runs/old.json"),
        ("--source-start", "0", "--expected-freeze-sha256", "c" * 64),
    ],
)
def test_fresh_mode_requires_source_range_and_forbids_resume_parameters(extra):
    with pytest.raises(ValueError):
        pipeline.stage_names(options(*extra, resume=False))


def test_resume_dry_run_only_audits_and_never_creates_outputs(tmp_path, monkeypatch):
    for name in ("sha", "budget_snapshot", "execute", "request_shutdown"):
        monkeypatch.setattr(pipeline, name, forbidden)
    monkeypatch.setattr(
        pipeline, "resume_context", lambda *args: {"prefix": {"terminal_questions": 300}}
    )
    monkeypatch.setattr(pipeline.subprocess, "run", forbidden)
    config = options("--sync-github", "--shutdown-on-complete", "--shutdown-on-quota-stop")
    assert pipeline.run(config, tmp_path) == 0
    assert list(tmp_path.iterdir()) == []


def test_resume_commands_preserve_explicit_start_certificate_and_freeze(tmp_path):
    dump(tmp_path / pipeline.FREEZE, {})
    config = options("--evaluation-resume-certificate", "runs/untouched.json")
    argv = pipeline.stage_command(config, tmp_path, "evaluation500", tmp_path / "stop.flag")
    assert "scripts/run_operator_evaluation_remaining.py" in argv
    assert argv[argv.index("--start") + 1] == "300"
    assert argv[argv.index("--end") + 1] == "500"
    assert argv[argv.index("--expected-freeze-sha256") + 1] == "c" * 64
    assert Path(argv[argv.index("--resume-certificate") + 1]) == Path("runs/untouched.json")
    assert "scripts/run_operator_source_remaining.py" not in argv


def test_evaluation_preflight_is_the_original_scorer_without_gold_flag(tmp_path):
    config = options(start=500)
    dump(tmp_path / pipeline.FREEZE, {})
    preflight = pipeline.stage_command(config, tmp_path, "evaluation_preflight", tmp_path / "stop")
    score = pipeline.stage_command(config, tmp_path, "evaluation_score", tmp_path / "stop")
    assert "growrag.experiments.score_operator_evaluation" in preflight
    assert "--score-new" not in preflight
    assert "--allow-network" not in preflight
    assert preflight[preflight.index("--expected-certificate-sha256") + 1] == "c" * 64
    assert "--score-new" in score


def mock_run_environment(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir(exist_ok=True)
    calls = []
    monkeypatch.setattr(pipeline, "sha", lambda path: "a" * 64)
    monkeypatch.setattr(
        pipeline, "action_list_execution_signature", lambda path: {"sha256": "b" * 64}
    )
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="", returncode=0)
    )
    monkeypatch.setattr(pipeline, "resume_context", lambda *args: {"evaluation_ids": IDS})
    monkeypatch.setattr(pipeline, "budget_snapshot", lambda root: {"prior_reserved_cny": 12})
    monkeypatch.setattr(pipeline, "project_budget_stopped", lambda *args: False)
    monkeypatch.setattr(pipeline, "public_report", lambda *args: "Synthetic safe summary\n")
    monkeypatch.setattr(pipeline, "stage_command", lambda args, root, stage, stop: [stage])
    monkeypatch.setattr(pipeline, "sync_public", lambda *args: calls.append("sync"))
    monkeypatch.setattr(pipeline, "request_shutdown", lambda *args: calls.append("shutdown"))
    return calls


@pytest.mark.parametrize("outcome", ["completed", "technical_stop", "quota_stop"])
def test_resume_has_same_safe_sync_and_shutdown_contract(tmp_path, monkeypatch, outcome):
    calls = mock_run_environment(tmp_path, monkeypatch)

    def execute(root, output, stage, argv):
        calls.append(stage)
        if stage == "evaluation500" and outcome == "quota_stop":
            dump(output / "quota_stop.flag", {})
            dump(output / "stop_requested.flag", {})
            return 2
        return int(stage == "evaluation500" and outcome == "technical_stop")

    monkeypatch.setattr(pipeline, "execute", execute)
    config = options(
        "--allow-network", "--sync-github", "--shutdown-on-complete", "--shutdown-on-quota-stop"
    )
    assert pipeline.run(config, tmp_path) == (0 if outcome == "completed" else 1)
    result = json.loads((tmp_path / "runs/resume_pipeline/pipeline_result.json").read_text())
    assert result["status"] == outcome
    assert not any(stage in calls for stage in ("source500", "source_score", "banks", "freeze"))
    assert "sync" in calls
    assert ("shutdown" in calls) == (outcome != "technical_stop")
    if "shutdown" in calls:
        assert calls.index("sync") < calls.index("shutdown")
    if outcome == "completed":
        assert calls.index("evaluation_preflight") < calls.index("evaluation_score")
    else:
        assert "evaluation_preflight" not in calls and "evaluation_score" not in calls


def test_500_missing_coverage_stops_before_gold_sync_completion_or_shutdown(tmp_path, monkeypatch):
    calls = mock_run_environment(tmp_path, monkeypatch)

    def execute(root, output, stage, argv):
        calls.append(stage)
        return 1

    monkeypatch.setattr(pipeline, "execute", execute)
    config = options("--allow-network", "--shutdown-on-complete", start=500)
    assert pipeline.run(config, tmp_path) == 1
    assert calls == ["evaluation_preflight"]
    result = json.loads((tmp_path / "runs/resume_pipeline/pipeline_result.json").read_text())
    assert result["status"] == "technical_stop"


def test_existing_output_directory_is_not_overwritten(tmp_path, monkeypatch):
    mock_run_environment(tmp_path, monkeypatch)
    existing = tmp_path / "runs/resume_pipeline"
    existing.mkdir()
    sentinel = existing / "preserve.txt"
    sentinel.write_text("existing audit", encoding="utf-8")
    monkeypatch.setattr(pipeline, "execute", forbidden)
    with pytest.raises((ValueError, FileExistsError)):
        pipeline.run(options("--allow-network"), tmp_path)
    assert sentinel.read_text() == "existing audit"


def frozen_context():
    return {
        "evaluation_ids": IDS,
        "evaluation_order_sha256": fingerprint(IDS),
        "execution_signature": {"sha256": "b" * 64, "configuration": {"model": "synthetic"}},
        "banks": {
            str(size): {"fingerprint": "d" * 64, "file_sha256": "e" * 64}
            for size in (50, 100, 250, 500)
        },
    }


def existing_products(root):
    (root / pipeline.FEEDBACK).mkdir(parents=True)
    (root / pipeline.BANKS).mkdir(parents=True)
    dump(root / pipeline.FREEZE, {"synthetic": True})
    dump(root / "data/manifest.json", {"synthetic": True})


def sealed_interval(root, start, end, *, label="a", failed=False, planned_end=None):
    """Only ownership/coverage fixtures; original scorer validates full API provenance."""
    freeze = frozen_context()
    name = f"{pipeline.ACTION_LIST.prefix}evaluation_synthetic_{label}"
    directory = root / "runs" / name
    plan = {
        "run_id": name,
        "phase": "evaluation",
        "protocol": pipeline.ACTION_LIST.protocol,
        "profile": pipeline.PROFILE,
        "arms": list(pipeline.ARMS),
        "question_ids": IDS[start : planned_end if planned_end is not None else end],
        "manifest_sha256": "a" * 64,
        "execution_signature": freeze["execution_signature"],
        "model": "synthetic",
        "evaluation_freeze_sha256": "c" * 64,
        "evaluation_freeze_path": str(root / pipeline.FREEZE),
        "evaluation_order_sha256": freeze["evaluation_order_sha256"],
        "bank_sha256": {f"memory{size}": "d" * 64 for size in (50, 100, 250, 500)},
        "bank_file_sha256": {f"memory{size}": "e" * 64 for size in (50, 100, 250, 500)},
        "gold_loaded": False,
        "memory_updates": False,
    }
    dump(directory / "launch_plan.json", plan)
    dump(root / "runs" / f"{name}.claim.json", {**plan, "plan_sha256": fingerprint(plan)})
    rows = []
    for index in range(start, end):
        failed_question = failed and index == end - 1
        active = list(pipeline.ARMS[:2] if failed_question else pipeline.ARMS)
        row = {"question_id": IDS[index], "arms": {}}
        for arm in active:
            outcome = {
                "question_id": IDS[index],
                "arm": arm,
                "status": "failed" if failed_question and arm == active[-1] else "completed",
                "feedback": None,
                "memory_updated": False,
                "calls": [{"trace_id": f"{name}/{IDS[index]}/{arm}/synthetic"}],
            }
            row["arms"][arm] = outcome
            dump(directory / f"{IDS[index]}_{arm}.json", outcome)
        rows.append(row)
        dump(directory / f"checkpoint_{index - start:04d}.json", row)
    dump(directory / "predictions.json", rows)
    dump(
        directory / "predictions_frozen.json",
        {
            "question_ids": IDS[start:end],
            "sha256": fingerprint(rows),
            "phase": "evaluation",
            "gold_loaded": False,
            "status": "failed" if failed else "completed",
            "cleanup_errors": [],
        },
    )
    owned = [call for row in rows for arm in row["arms"].values() for call in arm["calls"]]
    dump(directory / "final_budget.json", {"calls": owned, "api_requests": len(owned)})
    dump(
        directory / "events.jsonl",
        {"kind": "exit", "status": "failed" if failed else "completed", "requests": len(owned)},
    )
    return directory


@pytest.mark.parametrize("missing", [pipeline.FEEDBACK, pipeline.BANKS, pipeline.FREEZE])
def test_resume_context_requires_existing_source_products_before_gate(
    tmp_path, monkeypatch, missing
):
    existing_products(tmp_path)
    target = tmp_path / missing
    target.unlink() if target.is_file() else target.rmdir()
    monkeypatch.setattr(pipeline, "_evaluation_gate", forbidden)
    with pytest.raises(ValueError):
        pipeline.resume_context(options(), tmp_path)


def test_existing_evaluation_feedback_blocks_resume_without_overwrite(tmp_path, monkeypatch):
    existing_products(tmp_path)
    target = tmp_path / pipeline.EVALUATION
    target.mkdir()
    dump(target / "preserved.json", {"scored": True})
    monkeypatch.setattr(pipeline, "_evaluation_gate", forbidden)
    with pytest.raises(ValueError):
        pipeline.resume_context(options(), tmp_path)
    assert json.loads((target / "preserved.json").read_text()) == {"scored": True}


def test_context_delegates_original_freeze_and_suffix_verification(tmp_path, monkeypatch):
    existing_products(tmp_path)
    certificate = tmp_path / "runs/untouched.json"
    dump(certificate, {"proof": {"question_ids": IDS[300:305]}})
    calls = []

    def original_gate(actual, root):
        calls.append(actual)
        assert root == tmp_path
        assert actual.profile == "action-list-v3"
        assert actual.phase == "evaluation"
        assert actual.arms == list(pipeline.ARMS)
        assert actual.expected_freeze_sha256 == "c" * 64
        assert actual.expected_manifest_sha256 == "a" * 64
        assert actual.expected_execution_sha256 == "b" * 64
        assert actual.start == 300 and actual.count == 5
        assert actual.resume_certificate == certificate
        assert actual.evaluation_freeze == tmp_path / pipeline.FREEZE
        return frozen_context()

    monkeypatch.setattr(pipeline, "_evaluation_gate", original_gate)
    monkeypatch.setattr(pipeline, "audit_resume_prefix", lambda *a: {"terminal_questions": 300})
    config = options("--evaluation-resume-certificate", str(certificate))
    result = pipeline.resume_context(config, tmp_path)
    assert len(calls) == 1
    assert result["prefix"]["terminal_questions"] == 300


def test_wrong_freeze_fails_core_gate_before_prefix_or_launch(tmp_path, monkeypatch):
    existing_products(tmp_path)

    def original_gate(actual, root):
        assert actual.expected_freeze_sha256 == "f" * 64
        raise ValueError("evaluation certificate SHA mismatch")

    monkeypatch.setattr(pipeline, "_evaluation_gate", original_gate)
    monkeypatch.setattr(pipeline, "audit_resume_prefix", forbidden)
    with pytest.raises(ValueError, match="SHA mismatch"):
        pipeline.resume_context(options("--expected-freeze-sha256", "f" * 64), tmp_path)


def test_boundary_500_does_not_create_an_empty_suffix_certificate():
    with pytest.raises(ValueError):
        pipeline.stage_names(
            options("--evaluation-resume-certificate", "runs/empty.json", start=500)
        )


@pytest.mark.parametrize("suffix", [[], IDS[:26], "not-an-id-list"])
def test_context_rejects_invalid_suffix_before_core_gate(tmp_path, monkeypatch, suffix):
    existing_products(tmp_path)
    certificate = tmp_path / "runs/untouched.json"
    dump(certificate, {"proof": {"question_ids": suffix}})
    monkeypatch.setattr(pipeline, "_evaluation_gate", forbidden)
    with pytest.raises(ValueError):
        pipeline.resume_context(
            options("--evaluation-resume-certificate", str(certificate)), tmp_path
        )


def test_completed_boundary_prefix_needs_no_resume_certificate(tmp_path):
    sealed_interval(tmp_path, 0, 3)
    result = pipeline.audit_resume_prefix(options(start=3), tmp_path, frozen_context())
    assert result["terminal_questions"] == 3
    assert result["technical_failures"] == 0


def test_prefix_retains_failed_question_without_counting_it_success(tmp_path):
    sealed_interval(tmp_path, 0, 3, failed=True)
    result = pipeline.audit_resume_prefix(options(start=3), tmp_path, frozen_context())
    assert result["terminal_questions"] == 3
    assert result["technical_failures"] == 1


def test_parent_and_child_planned_overlap_is_not_actual_replay(tmp_path):
    sealed_interval(tmp_path, 0, 2, failed=True, planned_end=5, label="parent")
    sealed_interval(tmp_path, 2, 5, label="child")
    result = pipeline.audit_resume_prefix(options(start=5), tmp_path, frozen_context())
    assert result["terminal_questions"] == 5
    assert result["technical_failures"] == 1


@pytest.mark.parametrize("case", ["hole", "replay", "future", "claim_only", "partial_success"])
def test_prefix_rejects_holes_replay_future_claims_and_unexplained_partial_arms(tmp_path, case):
    directory = sealed_interval(tmp_path, 0, 2)
    start = 3
    if case == "replay":
        sealed_interval(tmp_path, 0, 1, label="replay")
        start = 2
    elif case == "future":
        start = 1
    elif case == "claim_only":
        dump(
            tmp_path / "runs" / f"{pipeline.ACTION_LIST.prefix}evaluation_future.claim.json",
            {"phase": "evaluation", "protocol": pipeline.ACTION_LIST.protocol},
        )
        start = 2
    elif case == "partial_success":
        rows = json.loads((directory / "predictions.json").read_text())
        rows[-1]["arms"].pop("memory500")
        dump(directory / "predictions.json", rows)
        dump(directory / "checkpoint_0001.json", rows[-1])
        seal = json.loads((directory / "predictions_frozen.json").read_text())
        seal["sha256"] = fingerprint(rows)
        dump(directory / "predictions_frozen.json", seal)
        start = 2
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.audit_resume_prefix(options(start=start), tmp_path, frozen_context())


@pytest.mark.parametrize("change", ["checksum", "cleanup", "gold", "memory", "arm_file", "claim"])
def test_sealed_prefix_tampering_is_rejected(tmp_path, change):
    directory = sealed_interval(tmp_path, 0, 2)
    if change in {"checksum", "cleanup", "gold"}:
        path = directory / "predictions_frozen.json"
        seal = json.loads(path.read_text())
        if change == "checksum":
            seal["sha256"] = "0" * 64
        elif change == "cleanup":
            seal["cleanup_errors"] = ["OSError"]
        else:
            seal["gold_loaded"] = True
        dump(path, seal)
    elif change in {"memory", "arm_file"}:
        path = directory / f"{IDS[0]}_base.json"
        row = json.loads(path.read_text())
        row["memory_updated" if change == "memory" else "feedback"] = True
        dump(path, row)
    else:
        path = tmp_path / "runs" / f"{directory.name}.claim.json"
        claim = json.loads(path.read_text())
        claim["plan_sha256"] = "0" * 64
        dump(path, claim)
    with pytest.raises(ValueError):
        pipeline.audit_resume_prefix(options(start=2), tmp_path, frozen_context())


@pytest.mark.parametrize(
    "change",
    [
        "missing_events_file",
        "missing_exit",
        "wrong_exit_status",
        "exit_request_mismatch",
        "early_exit",
        "unowned_call",
        "missing_owned_call",
        "negative_api_count",
        "boolean_api_count",
    ],
)
def test_prefix_requires_reconciled_ledger_and_one_real_terminal_exit(tmp_path, change):
    directory = sealed_interval(tmp_path, 0, 2)
    budget_path = directory / "final_budget.json"
    events_path = directory / "events.jsonl"
    budget = json.loads(budget_path.read_text())
    event = json.loads(events_path.read_text())
    if change == "missing_events_file":
        events_path.unlink()
    elif change == "missing_exit":
        dump(events_path, {"kind": "start"})
    elif change == "wrong_exit_status":
        dump(events_path, {**event, "status": "failed"})
    elif change == "exit_request_mismatch":
        dump(events_path, {**event, "requests": budget["api_requests"] + 1})
    elif change == "early_exit":
        events_path.write_text(json.dumps(event) + "\n" + json.dumps(event), encoding="utf-8")
    elif change == "unowned_call":
        budget["calls"].append({"trace_id": "unowned/request"})
        dump(budget_path, budget)
    elif change == "missing_owned_call":
        budget["calls"].pop()
        dump(budget_path, budget)
    else:
        budget["api_requests"] = -1 if change == "negative_api_count" else True
        dump(budget_path, budget)
        dump(events_path, {**event, "requests": budget["api_requests"]})
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.audit_resume_prefix(options(start=2), tmp_path, frozen_context())
