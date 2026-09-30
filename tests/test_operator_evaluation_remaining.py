"""No real API/evaluation labels: seven-arm supervisor and final-failure audit fixtures."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments.operator_profiles import (
    ACTION_LIST_METHOD_FILES,
    ACTION_LIST_SCHEMA,
    action_list_execution_configuration,
)
from growrag.experiments.representation_runner import fingerprint

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "run_operator_source_remaining", SCRIPTS / "run_operator_source_remaining.py"
)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)
sys.modules["run_operator_source_remaining"] = shared
SPEC = importlib.util.spec_from_file_location(
    "evaluation_remaining", SCRIPTS / "run_operator_evaluation_remaining.py"
)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
IDS = [f"{n:024x}" for n in range(500)]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def options(tmp_path, *extra):
    return evaluation.parser().parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--banks",
            str(tmp_path / "banks"),
            "--evaluation-freeze",
            str(tmp_path / "freeze.json"),
            "--expected-manifest-sha256",
            "a" * 64,
            "--expected-execution-sha256",
            "b" * 64,
            "--expected-freeze-sha256",
            "c" * 64,
            "--end",
            "5",
            *extra,
        ]
    )


def interval(argv):
    start = int(argv[argv.index("--start") + 1])
    return start, start + int(argv[argv.index("--count") + 1])


def terminal_fixture(tmp_path, args, *, start=0, end=2, failed=True, phase="evaluation"):
    arms = evaluation.ARMS if phase == "evaluation" else shared.ARMS
    name = evaluation.run_name(start, end) if phase == "evaluation" else shared.run_name(start, end)
    directory = tmp_path / "runs" / name
    body = {
        "schema": ACTION_LIST_SCHEMA,
        "configuration": action_list_execution_configuration(),
        "files": {name: "b" * 64 for name in ACTION_LIST_METHOD_FILES},
    }
    signature = {**body, "sha256": fingerprint(body)}
    args.expected_execution_sha256 = signature["sha256"]
    launch = {
        "run_id": name,
        "protocol": shared.ACTION_LIST.protocol,
        "profile": shared.ACTION_LIST.name,
        "phase": phase,
        "arms": arms,
        "question_ids": IDS[start:end],
        "gold_loaded": False,
        "memory_updates": False,
        "manifest_sha256": args.expected_manifest_sha256,
        "execution_signature": signature,
        "model": signature["configuration"]["model"],
    }
    call = {
        "trace_id": f"{name}/{IDS[end - 1]}/fresh/reader",
        "status": "completed",
        "api_requests": 1,
    }
    reports, owned = [], []
    for index in range(start, end):
        active = arms[:2] if failed and index == end - 1 else arms
        row = {"question_id": IDS[index], "arms": {}}
        for arm in active:
            calls = [call] if index == end - 1 and arm == "fresh" else []
            result = {
                "question_id": IDS[index],
                "arm": arm,
                "status": "failed" if failed and calls else "completed",
                "error_type": "ValueError" if failed and calls else None,
                "feedback": None,
                "memory_updated": False,
                "calls": calls,
            }
            row["arms"][arm] = result
            owned.extend(calls)
            dump(directory / f"{IDS[index]}_{arm}.json", result)
        reports.append(row)
        dump(directory / f"checkpoint_{index - start:04d}.json", row)
    ledger = {"calls": owned, "api_requests": 1, "block_reason": None}
    dump(directory / "launch_plan.json", launch)
    dump(tmp_path / "runs" / f"{name}.claim.json", {**launch, "plan_sha256": fingerprint(launch)})
    dump(directory / "predictions.json", reports)
    status = "failed" if failed else "completed"
    dump(
        directory / "predictions_frozen.json",
        {
            "status": status,
            "phase": phase,
            "gold_loaded": False,
            "cleanup_errors": [],
            "sha256": fingerprint(reports),
            "question_ids": IDS[start:end],
        },
    )
    dump(directory / "final_budget.json", ledger)
    dump(directory / "request_journal/0000_intent.json", {"trace_id": call["trace_id"]})
    dump(directory / "request_journal/0000_after.json", ledger)
    dump(directory / "api_audit/request.json", {"trace_id": call["trace_id"]})
    (directory / "events.jsonl").write_text(
        json.dumps({"kind": "exit", "status": status, "requests": 1}), encoding="utf-8"
    )
    for filename in ("process.json", "source_snapshot.json", "cumulative_budget.json"):
        dump(directory / filename, {})
    (directory / "live.log").write_text("fixture", encoding="utf-8")
    return directory


def test_dryrun_has_seven_arms_freeze_and_budget_without_child(tmp_path, capsys):
    args = options(tmp_path)
    assert (
        evaluation.collect(
            args,
            tmp_path,
            {"evaluation_ids": IDS},
            execute=lambda *a, **kw: pytest.fail("no child"),
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "--phase evaluation" in out and "--budget-cny 5" in out
    assert "--arms " + " ".join(evaluation.ARMS) in out
    assert "--expected-freeze-sha256" in out and "--allow-network" not in out
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("phase", ["source", "evaluation"])
def test_terminal_last_failure_is_fully_audited_not_called_success(tmp_path, monkeypatch, phase):
    args = options(tmp_path)
    dump(args.evaluation_freeze, {})
    directory = terminal_fixture(tmp_path, args, phase=phase)
    monkeypatch.setattr(
        shared,
        "_evaluation_binding",
        lambda *a: {
            "freeze_path": str(args.evaluation_freeze),
            "freeze_sha256": args.expected_freeze_sha256,
        },
    )
    result = shared.terminal_interval(
        args,
        tmp_path,
        directory.name,
        IDS[:2],
        phase=phase,
        arms=evaluation.ARMS if phase == "evaluation" else shared.ARMS,
    )
    assert result == {"status": "terminal_failed", "completed_questions": 1}
    (directory / "request_journal/0000_after.json").unlink()
    with pytest.raises(ValueError, match="after"):
        shared.terminal_interval(
            args,
            tmp_path,
            directory.name,
            IDS[:2],
            phase=phase,
            arms=evaluation.ARMS if phase == "evaluation" else shared.ARMS,
        )


def test_terminal_interval_rejects_unclean_seal_and_incomplete_coverage(tmp_path):
    args = options(tmp_path)
    directory = terminal_fixture(tmp_path, args, phase="source")
    with pytest.raises(ValueError, match="completely and cleanly"):
        shared.terminal_interval(args, tmp_path, directory.name, IDS[:3])
    path = directory / "predictions_frozen.json"
    seal = json.loads(path.read_text())
    seal["cleanup_errors"] = ["ledger"]
    dump(path, seal)
    with pytest.raises(ValueError, match="completely and cleanly"):
        shared.terminal_interval(args, tmp_path, directory.name, IDS[:2])


def test_evaluation_last_failure_moves_next_batch_without_empty_certificate(tmp_path, monkeypatch):
    args = options(
        tmp_path,
        "--allow-network",
        "--continue-untouched-recorded-model-errors",
        "--batch-size",
        "2",
    )
    seen = []
    monkeypatch.setattr(shared, "continuation_progress", lambda *a: 0)
    monkeypatch.setattr(
        shared,
        "_evaluation_binding",
        lambda *a: {
            "freeze_path": str(args.evaluation_freeze),
            "freeze_sha256": args.expected_freeze_sha256,
        },
    )
    dump(args.evaluation_freeze, {})

    def execute(argv, **kw):
        start, end = interval(argv)
        seen.append((start, end))
        terminal_fixture(tmp_path, args, start=start, end=end, failed=start == 0)
        return SimpleNamespace(returncode=int(start == 0))

    assert (
        evaluation.collect(
            args,
            tmp_path,
            {"evaluation_ids": IDS},
            execute=execute,
            certifier=lambda *a, **kw: pytest.fail("no empty certificate"),
        )
        == 0
    )
    assert seen == [(0, 2), (2, 4), (4, 5)]


def test_stop_file_prevents_evaluation_child(tmp_path):
    flag = tmp_path / "stop.flag"
    flag.write_bytes(b"\xff")
    assert (
        evaluation.collect(
            options(tmp_path, "--stop-file", str(flag), "--allow-network"),
            tmp_path,
            {"evaluation_ids": IDS},
            execute=lambda *a, **kw: pytest.fail("stopped"),
        )
        == 2
    )


@pytest.mark.parametrize("bad_suffix", [False, True])
def test_evaluation_continues_only_complete_suffix_under_same_freeze(
    tmp_path, monkeypatch, bad_suffix
):
    args = options(tmp_path, "--allow-network", "--continue-untouched-transport")
    dump(args.evaluation_freeze, {})
    seen = []
    monkeypatch.setattr(shared, "continuation_progress", lambda *a: 0)
    monkeypatch.setattr(
        shared,
        "terminal_interval",
        lambda *a, **kw: {"status": "completed", "completed_questions": 4},
    )

    def execute(argv, **kw):
        start, end = interval(argv)
        seen.append((start, end))
        if start == 0:
            rows = [{"question_id": IDS[0], "arms": {"base": {"status": "failed"}}}]
            dump(tmp_path / "runs" / evaluation.run_name(start, end) / "predictions.json", rows)
            return SimpleNamespace(returncode=1)
        assert "--resume-certificate" in argv
        dump(
            tmp_path / "runs" / evaluation.run_name(start, end) / "predictions.json",
            [
                {"question_id": qid, "arms": {a: {"status": "completed"} for a in evaluation.ARMS}}
                for qid in IDS[start:end]
            ],
        )
        return SimpleNamespace(returncode=0)

    def certificate(runs, name, **kw):
        return {
            "proof": {
                "started_question_ids": IDS[:1],
                "question_ids": IDS[2:5] if bad_suffix else IDS[1:5],
                "phase": "evaluation",
                "parent_run_id": name,
                "arms": evaluation.ARMS,
                "manifest_sha256": "a" * 64,
                "source_execution_sha256": "b" * 64,
                "evaluation_binding": {
                    "freeze_sha256": "c" * 64,
                    "freeze_path": str(args.evaluation_freeze),
                },
            }
        }

    if bad_suffix:
        with pytest.raises(ValueError, match="exact frozen evaluation suffix"):
            evaluation.collect(
                args, tmp_path, {"evaluation_ids": IDS}, execute=execute, certifier=certificate
            )
        assert seen == [(0, 5)]
    else:
        assert (
            evaluation.collect(
                args, tmp_path, {"evaluation_ids": IDS}, execute=execute, certifier=certificate
            )
            == 0
        )
        assert seen == [(0, 5), (1, 5)]


def test_three_zero_complete_seven_arm_batches_stop_even_with_base_fresh_static_complete(
    tmp_path, monkeypatch
):
    args = options(
        tmp_path,
        "--allow-network",
        "--continue-untouched-recorded-model-errors",
        "--batch-size",
        "1",
    )
    seen = []
    monkeypatch.setattr(
        shared, "continuation_progress", lambda *a: 1
    )  # Its three-arm count MUST be ignored.
    monkeypatch.setattr(
        shared,
        "terminal_interval",
        lambda *a, **kw: {"status": "terminal_failed", "completed_questions": 0},
    )

    def execute(argv, **kw):
        start, end = interval(argv)
        seen.append((start, end))
        arms = {a: {"status": "completed"} for a in shared.ARMS}
        arms["memory50"] = {"status": "failed"}
        dump(
            tmp_path / "runs" / evaluation.run_name(start, end) / "predictions.json",
            [{"question_id": IDS[start], "arms": arms}],
        )
        return SimpleNamespace(returncode=1)

    assert evaluation.collect(args, tmp_path, {"evaluation_ids": IDS}, execute=execute) == 1
    assert seen == [(0, 1), (1, 2), (2, 3)]
