"""Pilot orchestration contracts; fixtures are synthetic and never call a provider."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import run_dualrag_pilot as runner
from growrag.experiments.api_client import ChatResponse
from growrag.experiments.hotpot import HotpotExample
from growrag.experiments.protocol import Evidence, GoldRecord, RuntimeQuestion

Q = RuntimeQuestion("fixture", "When was Northbridge founded?", "synthetic")
E = Evidence("e0", "Northbridge", 0, "Northbridge was founded in 1901.")
GOLD = GoldRecord(Q.question_id, ("1901",), (("Northbridge", 0),))


class Client:
    transport_source = "mock"
    config = SimpleNamespace(
        base_url="https://test.invalid/v1",
        model="TEST",
        max_output_tokens=1024,
        json_schema_mode=False,
    )

    def __init__(self, responses):
        self.responses, self.requests, self.calls = list(responses), [], []
        self.block_reason = None

    @property
    def attempts(self):
        return len(self.requests)

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        self.calls.append(
            {
                "api_requests": 0,
                "input_tokens": None,
                "output_tokens": None,
                "estimated_actual_cny": 0,
            }
        )
        content = self.responses.pop(0)
        return ChatResponse(
            content if isinstance(content, str) else json.dumps(content),
            "TEST",
            "TEST",
            "response",
            "request",
            20,
            20,
            0,
            Path("offline.json"),
            "mock",
        )


def responses():
    answer = {"answer": "1901", "cited_evidence_ids": ["e0"]}
    return [
        answer,
        {"information_need": "Find founding year.", "need_retrieve": True},
        {"entities": [{"entity": "Northbridge", "queries": ["Northbridge founding year"]}]},
        {"summary": "Northbridge was founded in 1901.", "evidence_ids": ["e0"]},
        {"information_need": "The year is known.", "need_retrieve": False},
        answer,
    ]


def test_complete_pair_scores_only_after_all_api_paths_finish(tmp_path):
    client = Client(responses())

    class GuardedExample:
        question, candidate_context = Q, (E,)

        @property
        def gold(self):
            assert len(client.requests) == 6
            return GOLD

    report = runner.run_question(GuardedExample(), client, tmp_path / "question")
    assert report["complete_pair"]
    assert all(report["arms"][arm]["feedback"]["answer_em"] == 1 for arm in runner.ARMS)
    assert report["arms"]["DUALRAG2"]["all_retrieved_support_recall"] == 1
    assert not client.block_reason
    # Execution artifacts predate scores and must remain unscored.
    for arm in runner.ARMS:
        saved = json.loads((tmp_path / "question" / f"{arm}_execution.json").read_text())
        assert saved["feedback"] is None
    assert all("gold" not in str(messages).lower() for messages, _ in client.requests)


@pytest.mark.parametrize("fail_arm", ["BASE1", "DUALRAG2"])
def test_failure_preserves_requests_and_never_reads_gold_or_retries(tmp_path, fail_arm):
    client = Client(["invalid"] if fail_arm == "BASE1" else [responses()[0], "invalid"])

    class GuardedExample:
        question, candidate_context = Q, (E,)

        @property
        def gold(self):
            raise AssertionError("gold must not be read for an incomplete pair")

    report = runner.run_question(GuardedExample(), client, tmp_path / "question")
    assert not report["complete_pair"]
    assert report["arms"][fail_arm]["status"] == "failed"
    assert all(report["arms"][arm]["feedback"] is None for arm in runner.ARMS)
    assert client.block_reason
    assert len(client.requests) == (1 if fail_arm == "BASE1" else 2)
    if fail_arm == "BASE1":
        assert report["arms"]["DUALRAG2"]["status"] == "not_executed"
    assert (tmp_path / "question" / "report.json").exists()


def test_execute_runs_first_two_then_remaining_without_quality_tuning(tmp_path):
    client = Client(responses() * 8)
    examples = tuple(HotpotExample(Q, (E,), GOLD) for _ in range(8))
    summary = runner.execute(client, examples, tmp_path)
    assert summary["complete_scored_pairs"] == 8 and summary["started_questions"] == 8
    checkpoint = json.loads((tmp_path / "first_two_checkpoint.json").read_text())
    assert checkpoint["completed_pairs"] == 2
    assert checkpoint["continue_only_if_no_protocol_failure"]
    assert checkpoint["outcome_based_prompt_tuning"] is False
    assert client.attempts == 48
    assert summary["paired"]["mean_f1_delta"] == 0


def test_execute_stops_after_first_failure_and_counts_unstarted(tmp_path):
    client = Client(["bad"])
    examples = tuple(HotpotExample(Q, (E,), GOLD) for _ in range(8))
    summary = runner.execute(client, examples, tmp_path)
    assert summary["started_questions"] == 1 and summary["not_started_questions"] == 7
    assert summary["complete_scored_pairs"] == 0
    assert summary["arms"]["BASE1"]["answer_f1"] is None
    assert summary["arms"]["BASE1"]["cost_including_incomplete"]["recorded_attempts"] == 1
    assert not (tmp_path / "first_two_checkpoint.json").exists()
    assert (tmp_path / "summary.json").exists()


def test_unexpected_interruption_has_durable_partial_summary(monkeypatch, tmp_path):
    def interrupted(*args):
        raise RuntimeError("private failure text")

    monkeypatch.setattr(runner, "run_question", interrupted)
    with pytest.raises(RuntimeError):
        runner.execute(Client([]), [object()] * 8, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["started_questions"] == 1 and summary["reported_questions"] == 0
    assert "private failure" not in (tmp_path / "summary.json").read_text()


def test_source_snapshot_includes_source_not_private_configuration(tmp_path):
    (tmp_path / "src" / "growrag").mkdir(parents=True)
    (tmp_path / "src" / "growrag" / "test.py").write_text("# synthetic source\n")
    (tmp_path / "src" / "growrag" / "__pycache__").mkdir()
    (tmp_path / "src" / "growrag" / "__pycache__" / "test.pyc").write_bytes(b"ignored")
    (tmp_path / "qwenAPI.md").write_text("DO-NOT-READ-OR-SNAPSHOT")
    snapshot = runner.source_snapshot(tmp_path)
    assert set(snapshot["files"]) == {"src/growrag/test.py"}
    assert snapshot["files"]["src/growrag/test.py"]["text"] == (
        tmp_path / "src" / "growrag" / "test.py"
    ).read_bytes().decode("utf-8")
    assert "DO-NOT-READ" not in json.dumps(snapshot)
    assert len(snapshot["sha256"]) == 64


def mock_plan(monkeypatch):
    examples = tuple(HotpotExample(Q, (E,), GOLD) for _ in range(8))
    monkeypatch.setattr(runner, "load_debug", lambda _: (examples, {"role": "synthetic"}))
    ledgers = []

    def history(root, *, reviewed_extra_ledgers):
        ledgers.append(reviewed_extra_ledgers)
        return {"prior_reserved_cny": 0.9, "prior_unknown_cost_calls": 1}

    monkeypatch.setattr(runner, "reconcile_history", history)
    monkeypatch.setattr(
        runner, "_git_state", lambda: {"commit": "synthetic", "worktree_dirty": True}
    )
    return ledgers


def test_dry_plan_has_all_historical_roots_source_hashes_and_no_secret_read(monkeypatch, tmp_path):
    ledgers = mock_plan(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("plan must not read API config or contact provider")

    monkeypatch.setattr(runner, "read_local_bailian_settings", forbidden)
    monkeypatch.setattr(runner, "LiveChatClient", forbidden)
    output = tmp_path / "plan"
    assert (
        runner.main(["--manifest", "unused", "--runs-root", str(tmp_path), "--output", str(output)])
        == 0
    )
    plan = json.loads((output / "plan.json").read_text())
    assert ledgers == [runner.EXTRA_ROOTS] and len(runner.EXTRA_ROOTS) == 4
    assert plan["worst_case_model_calls"] == plan["max_api_calls"] == 80
    assert plan["model"] == "qwen3.7-flash-2026-07-15"
    assert plan["subcap_cny"] == 3
    assert not plan["official_dev_test_used"] and not plan["check24_used"]
    assert len(plan["prompts"]) == 5
    source = json.loads((output / "source_snapshot.json").read_text(encoding="utf-8"))
    assert source["sha256"] == plan["source_snapshot_sha256"]
    assert "src/growrag/experiments/dualrag_adapter.py" in source["files"]
    assert not (tmp_path / runner.CLAIM_NAME).exists()


@pytest.mark.parametrize("name", ["outside", "same", "exists", "wrong-live-name"])
def test_output_safety_precedes_data_and_configuration(monkeypatch, tmp_path, name):
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject path before reading data")

    monkeypatch.setattr(runner, "load_debug", forbidden)
    output = (
        tmp_path.parent / "outside"
        if name == "outside"
        else tmp_path
        if name == "same"
        else tmp_path / "inside"
    )
    if name == "exists":
        output.mkdir()
    args = ["--manifest", "unused", "--runs-root", str(tmp_path), "--output", str(output)]
    if name == "wrong-live-name":
        args.append("--allow-network")
    with pytest.raises(ValueError):
        runner.main(args)


def test_existing_claim_prevents_even_loading_data(monkeypatch, tmp_path):
    (tmp_path / runner.CLAIM_NAME).write_text("{}")
    monkeypatch.setattr(runner, "load_debug", lambda _: pytest.fail("must not read data"))
    with pytest.raises(FileExistsError):
        runner.main(
            [
                "--manifest",
                "unused",
                "--runs-root",
                str(tmp_path),
                "--output",
                str(tmp_path / runner.RUN_ID),
                "--allow-network",
            ]
        )


def test_exhausted_historical_reservation_blocks_new_run(monkeypatch, tmp_path):
    mock_plan(monkeypatch)
    monkeypatch.setattr(
        runner, "reconcile_history", lambda *args, **kwargs: {"prior_reserved_cny": 50.0}
    )
    monkeypatch.setattr(runner, "read_local_bailian_settings", lambda _: pytest.fail("no key read"))
    output = tmp_path / runner.RUN_ID
    with pytest.raises(ValueError, match="budget"):
        runner.main(
            [
                "--manifest",
                "unused",
                "--runs-root",
                str(tmp_path),
                "--output",
                str(output),
                "--allow-network",
            ]
        )
    assert not output.exists() and not (tmp_path / runner.CLAIM_NAME).exists()
