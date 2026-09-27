"""Offline tests for the bounded source-collection CLI."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import run_memory_source as runner
from growrag.experiments import source_collection_core as core


def args(tmp_path):
    return [
        "--manifest",
        "m.json",
        "--corpus-manifest",
        "c.json",
        "--index-path",
        "i.sqlite",
        "--upstream",
        "upstream",
        "--runs-root",
        str(tmp_path),
        "--start",
        "0",
        "--count",
        "8",
    ]


@pytest.fixture
def prepare(monkeypatch):
    qs = [core.RuntimeQuestion(f"q{i}", f"question {i}", "synthetic") for i in range(64)]
    monkeypatch.setattr(
        runner,
        "load_source_questions",
        lambda _: (qs, {"roles": {"source": [q.question_id for q in qs]}}),
    )
    monkeypatch.setattr(runner, "recheck_exposure", lambda *a: None)
    monkeypatch.setattr(runner, "reviewed_history", lambda _: {"prior_reserved_cny": 5.0})
    monkeypatch.setattr(runner, "source_snapshot", lambda _: {"sha256": "synthetic"})
    monkeypatch.setattr(runner, "execution_signature", lambda _: "synthetic-execution")
    monkeypatch.setattr(runner, "_git_state", lambda: {"commit": "abc", "worktree_dirty": True})
    monkeypatch.setattr(
        runner, "S2GAuthorAPI", lambda *a, **k: SimpleNamespace(provenance={"synthetic": True})
    )
    monkeypatch.setattr(
        runner, "read_local_bailian_settings", lambda _: pytest.fail("no key access")
    )
    monkeypatch.setattr(runner, "LiveChatClient", lambda *a, **k: pytest.fail("no network"))
    return qs


def test_dry_plan_no_runtime_key_claim_or_network(tmp_path, monkeypatch, prepare, capsys):
    monkeypatch.setattr(
        runner, "load_source_runtime", lambda *a: pytest.fail("no expensive runtime")
    )
    assert runner.main(args(tmp_path)) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["max_calls"] == 88 and plan["subcap_cny"] == 3
    assert plan["question_ids"] == [f"q{i}" for i in range(8)]
    assert plan["gold_policy"].startswith("none during collection")
    assert not list(tmp_path.iterdir())


def test_live_dirty_checkout_rejected_before_key(tmp_path, prepare):
    with pytest.raises(ValueError, match="clean committed"):
        runner.main(args(tmp_path) + ["--allow-network", "--api-config", "secret.md"])


def test_untracked_copy_cannot_be_used_for_live(tmp_path, monkeypatch, prepare):
    monkeypatch.setattr(runner, "__file__", str(tmp_path / "draft.py"))
    monkeypatch.setattr(runner, "_git_state", lambda: {"commit": "abc", "worktree_dirty": False})
    with pytest.raises(ValueError, match="ignored draft"):
        runner.main(args(tmp_path) + ["--allow-network", "--api-config", "secret.md"])


@pytest.mark.parametrize("project,series", [(50, 0), (5, 3)])
def test_series_and_project_budget_cap(tmp_path, monkeypatch, prepare, project, series):
    monkeypatch.setattr(runner, "reviewed_history", lambda _: {"prior_reserved_cny": project})
    monkeypatch.setattr(runner, "series_reserved", lambda _: series)
    with pytest.raises(ValueError, match="budget exhausted"):
        runner.main(args(tmp_path))


def test_offline_score_never_reads_api_key(tmp_path, monkeypatch, prepare, capsys):
    closed = []
    runtime = SimpleNamespace(questions=prepare, close=lambda: closed.append(True))
    monkeypatch.setattr(runner, "load_source_runtime", lambda *a: runtime)
    monkeypatch.setattr(
        runner,
        "score_collection",
        lambda *a: (
            [{"arms": {"BASE": {"feedback": {}}, "S2G": {"feedback": {}}}} for _ in range(63)]
            + [{"arms": {"BASE": {"feedback": None}, "S2G": {"feedback": None}}}]
        ),
    )
    assert (
        runner.main(
            args(tmp_path) + ["--score-only", "--raw-train", "raw", "--score-output", "scores"]
        )
        == 0
    )
    assert closed == [True] and json.loads(capsys.readouterr().out)["network_used"] is False


def test_claim_no_replay_even_without_outputs(tmp_path):
    claim = tmp_path / f"{runner.batch_identity(0, 8)}.claim.json"
    runner.write_json(
        claim,
        {"protocol": runner.PROTOCOL, "manifest_sha256": runner.PLAN_SHA, "question_ids": ["q0"]},
    )
    with pytest.raises(ValueError, match="already claimed"):
        runner.check_no_replay(tmp_path, ["q0"])
    runner.check_no_replay(tmp_path, ["q1"])
    with pytest.raises(ValueError, match="unfinished"):
        runner.series_reserved(tmp_path)


@pytest.mark.parametrize("reserved", [None, True, -1, float("nan"), float("inf")])
def test_uncertain_reservation_not_free(tmp_path, reserved):
    root = tmp_path / runner.batch_identity(0, 8)
    root.mkdir()
    runner.write_json(
        root / "launch_plan.json", {"protocol": runner.PROTOCOL, "model": runner.PILOT_MODEL}
    )
    (root / "final_budget.json").write_text(json.dumps({"reserved_cny": reserved}))
    with pytest.raises(ValueError, match="reservation"):
        runner.series_reserved(tmp_path)


@pytest.mark.parametrize("fail_collection", [False, True])
@pytest.mark.parametrize("previous", [None, "synthetic-previous-key"])
def test_simulated_live_exit_restores_key_and_closes_runtime(
    tmp_path,
    monkeypatch,
    prepare,
    fail_collection,
    previous,
):
    """No real transport: test lifecycle after in-memory fake client construction."""
    import os

    closed = []
    runtime = SimpleNamespace(
        questions=prepare, metadata={"synthetic": True}, close=lambda: closed.append(True)
    )
    monkeypatch.setattr(
        runner, "__file__", str(Path.cwd() / "src/growrag/experiments/run_memory_source.py")
    )
    monkeypatch.setattr(
        runner, "_git_state", lambda: {"commit": "synthetic", "worktree_dirty": False}
    )
    monkeypatch.setattr(runner, "load_source_runtime", lambda *a: runtime)
    monkeypatch.setattr(
        runner,
        "read_local_bailian_settings",
        lambda _: SimpleNamespace(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="synthetic-new-key",
        ),
    )
    monkeypatch.setattr(runner, "LiveChatClient", lambda *a, **k: "synthetic-transport")
    budget = SimpleNamespace(
        block_reason=None,
        attempts=0,
        reserved_cny=0,
        report=lambda: {"calls": [], "api_requests": 0, "reserved_cny": 0},
    )
    monkeypatch.setattr(runner, "SharedBudgetClient", lambda *a, **k: budget)
    monkeypatch.setattr(
        runner,
        "reviewed_history",
        lambda _: {"prior_reserved_cny": 5, "prior_unknown_cost_requests": 0},
    )

    def collect(*a, **k):
        assert os.environ[runner.KEY_VARIABLE] == "synthetic-new-key"
        if fail_collection:
            raise RuntimeError("synthetic-new-key must not enter exception logs")

    monkeypatch.setattr(runner, "collect_batch", collect)
    if previous is None:
        monkeypatch.delenv(runner.KEY_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(runner.KEY_VARIABLE, previous)
    assert runner.main(args(tmp_path) + ["--allow-network", "--api-config", "secret.md"]) == int(
        fail_collection
    )
    assert os.environ.get(runner.KEY_VARIABLE) == previous and closed == [True]
    root = tmp_path / runner.batch_identity(0, 8)
    assert (root / "final_budget.json").is_file()
    if fail_collection:
        assert json.loads((root / "interrupted.json").read_bytes())["error_type"] == "RuntimeError"
    for path in root.glob("*.json*"):
        assert "synthetic-new-key" not in path.read_text(encoding="utf-8")


def closed_failed_parent(tmp_path):
    """Synthetic closed run: q0 attempted/failed, q1 planned but never touched."""
    name = runner.batch_identity(0, 2)
    root = tmp_path / name
    root.mkdir()
    launch = {
        "run_id": name,
        "protocol": runner.PROTOCOL,
        "manifest_sha256": runner.PLAN_SHA,
        "model": runner.PILOT_MODEL,
        "generation_profile": runner.GENERATION_PROFILE,
        "backend_output_caps": runner.SHARED_OUTPUT_CAPS,
        "start": 0,
        "count": 2,
        "question_ids": ["q0", "q1"],
    }
    runner.write_json(root / "launch_plan.json", launch)
    runner.write_json(
        tmp_path / f"{name}.claim.json",
        {
            "protocol": runner.PROTOCOL,
            "manifest_sha256": runner.PLAN_SHA,
            "question_ids": launch["question_ids"],
            "plan_sha256": runner.fingerprint(launch),
        },
    )
    trace = f"{name}/q0/BASE1_AUTHOR_READER/s2g-author/01-answer"
    call = {"trace_id": trace, "api_requests": 1, "status": "failed"}
    outcomes = {
        "BASE1_AUTHOR_READER": {"status": "failed", "feedback": None, "calls": [call]},
        "S2G_AUTHOR_API4": {"status": "not_executed", "feedback": None, "calls": []},
    }
    report = {"question_id": "q0", "offset": 0, "arms": outcomes, "complete_pair": False}
    runner.write_json(root / "predictions.json", [report])
    runner.write_json(
        root / "predictions_frozen.json",
        {
            "run_id": name,
            "question_ids": ["q0"],
            "reports_sha256_before_scoring": runner.fingerprint([report]),
        },
    )
    qdir = root / "questions" / "0000"
    qdir.mkdir(parents=True)
    runner.write_json(qdir / "prediction_report.json", report)
    runner.write_json(qdir / "BASE1_AUTHOR_READER_execution.json", outcomes["BASE1_AUTHOR_READER"])
    budget = {"api_requests": 1, "calls": [call], "block_reason": "transport_failure"}
    runner.write_json(root / "final_budget.json", budget)
    journal, audits = root / "request_journal", root / "api_audit"
    journal.mkdir()
    audits.mkdir()
    runner.write_json(journal / "0000_intent.json", {"trace_id": trace})
    runner.write_json(journal / "0000_after.json", budget)
    runner.write_json(audits / "api.json", {"trace_id": trace, "api_requests": 1})
    events = [
        {"kind": "api_failure", "trace_id": trace},
        {"kind": "arm_failed", "question_id": "q0"},
        {"kind": "question_complete", "question_id": "q0", "status": "failed"},
        {"kind": "exit", "status": "failed"},
    ]
    (root / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return name, root


def test_closed_unstarted_claim_can_be_explicitly_transferred(tmp_path):
    name, root = closed_failed_parent(tmp_path)
    with pytest.raises(ValueError, match="already claimed"):
        runner.check_no_replay(tmp_path, ["q1"])
    proof = runner.check_no_replay(tmp_path, ["q1"], continuation_of=name)
    assert proof["continued_question_ids"] == proof["released_question_ids"] == ["q1"]
    assert proof["checked_global_offsets"] == [1]
    assert proof["prior_request_evidence"]["intent_after_pairs"] == 1
    assert proof["validated_prior_runs"] == [name]


@pytest.mark.parametrize("wanted", [["q0"], ["q2"], ["q1", "q1"], []])
def test_failed_outside_duplicate_or_empty_targets_rejected(tmp_path, wanted):
    name, _ = closed_failed_parent(tmp_path)
    with pytest.raises(ValueError):
        runner.prove_unstarted_continuation(tmp_path, name, wanted)


@pytest.mark.parametrize(
    "name", ["../escape", "bad", "2026-09-27_memory_source_v1_0000_0002/child"]
)
def test_unsafe_continuation_name_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="safe source"):
        runner.prove_unstarted_continuation(tmp_path, name, ["q1"])


@pytest.mark.parametrize(
    "tamper",
    [
        "target_dir",
        "target_event",
        "target_intent",
        "orphan_intent",
        "missing_after",
        "orphan_audit",
        "changed_after",
        "changed_claim",
        "changed_seal",
        "after_exit",
        "missing_exit",
        "unowned_ledger",
        "changed_prediction",
        "changed_model",
    ],
)
def test_unstarted_proof_rejects_uncertainty_without_any_api(tmp_path, tamper):
    name, root = closed_failed_parent(tmp_path)

    def replace_fixture(path, value):
        # Deliberately corrupt synthetic fixtures; production writes stay immutable.
        path.write_text(json.dumps(value), encoding="utf-8")

    if tamper == "target_dir":
        (root / "questions" / "0001").mkdir()
    elif tamper == "target_event":
        events = [json.loads(x) for x in (root / "events.jsonl").read_text().splitlines()]
        events.insert(0, {"kind": "arm_start", "question_id": "q1"})
        (root / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    elif tamper == "target_intent":
        replace_fixture(
            root / "request_journal/0000_intent.json",
            {"trace_id": f"{name}/q1/BASE1_AUTHOR_READER/request"},
        )
    elif tamper == "orphan_intent":
        runner.write_json(
            root / "request_journal/0001_intent.json",
            {"trace_id": f"{name}/q0/BASE1_AUTHOR_READER/request2"},
        )
    elif tamper == "missing_after":
        (root / "request_journal/0000_after.json").unlink()
    elif tamper == "orphan_audit":
        runner.write_json(
            root / "api_audit/other.json", {"trace_id": f"{name}/q0/BASE1_AUTHOR_READER/request2"}
        )
    elif tamper == "changed_after":
        replace_fixture(root / "request_journal/0000_after.json", {"calls": []})
    elif tamper == "changed_claim":
        replace_fixture(tmp_path / f"{name}.claim.json", {})
    elif tamper == "changed_seal":
        replace_fixture(root / "predictions_frozen.json", {})
    elif tamper == "after_exit":
        with (root / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"kind": "api_request"}) + "\n")
    elif tamper == "missing_exit":
        (root / "events.jsonl").write_text(json.dumps({"kind": "question_complete"}))
    elif tamper == "unowned_ledger":
        replace_fixture(root / "final_budget.json", {"calls": []})
    elif tamper == "changed_prediction":
        replace_fixture(root / "questions/0000/prediction_report.json", {})
    elif tamper == "changed_model":
        p = root / "launch_plan.json"
        changed = json.loads(p.read_bytes())
        changed["model"] = "different"
        replace_fixture(p, changed)
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.prove_unstarted_continuation(tmp_path, name, ["q1"])


def test_continuation_proof_does_not_override_another_new_claim(tmp_path):
    name, _ = closed_failed_parent(tmp_path)
    other = runner.batch_identity(1, 1)
    runner.write_json(
        tmp_path / f"{other}.claim.json",
        {
            "protocol": runner.PROTOCOL,
            "manifest_sha256": runner.PLAN_SHA,
            "question_ids": ["q1"],
        },
    )
    with pytest.raises(ValueError, match="already claimed"):
        runner.check_no_replay(tmp_path, ["q1"], continuation_of=name)
