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
    monkeypatch.setattr(runner, "score_collection", lambda *a: [None] * 64)
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
