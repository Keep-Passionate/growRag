"""Synthetic orchestration tests: no provider calls and no real dataset labels."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import run_fresh_continuation as runner
from growrag.experiments.api_client import ChatResponse
from growrag.experiments.hotpot import HotpotExample
from growrag.experiments.protocol import Evidence, GoldRecord, RuntimeQuestion

Q = RuntimeQuestion("synthetic-new", "When was Northbridge founded?", "synthetic")
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

    def __init__(self, fail_at=None):
        self.requests, self.calls = [], []
        self.block_reason, self.fail_at = None, fail_at
        self.reason_count = 0

    @property
    def attempts(self):
        return len(self.requests)

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        self.calls.append(
            {
                "api_requests": 0,
                "input_tokens": 20,
                "output_tokens": 20,
                "estimated_actual_cny": 0,
            }
        )
        version = kwargs["prompt_version"]
        if version == runner.dualrag.GUARDED_PROMPT_VERSIONS["reason"]:
            self.reason_count += 1
            content = {
                "information_need": "Find founding year.",
                "need_retrieve": self.reason_count % 2 == 1,
            }
        elif version == runner.dualrag.GUARDED_PROMPT_VERSIONS["entities"]:
            content = {
                "entities": [{"entity": "Northbridge", "queries": ["Northbridge founding year"]}]
            }
        elif version == runner.dualrag.GUARDED_PROMPT_VERSIONS["summarize"]:
            content = {"summary": E.text, "evidence_ids": ["e0"]}
        elif version == runner.s2g.PROMPT_VERSIONS["extract"]:
            content = {"sentence_ids": ["e0"]}
        elif version == runner.s2g.PROMPT_VERSIONS["judge"]:
            content = {"sufficient": True, "gap_items": []}
        else:
            content = {"answer": "1901", "cited_evidence_ids": ["e0"]}
        return ChatResponse(
            "invalid" if self.attempts == self.fail_at else json.dumps(content),
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


def test_complete_triplet_keeps_gold_out_of_execution_and_prescores(tmp_path):
    client = Client()

    class Example:
        question, candidate_context, question_type = Q, (E,), "bridge"

        @property
        def gold(self):
            assert client.attempts == 9
            return GOLD

    report = runner.run_question(Example(), client, tmp_path / "question")
    assert report["complete_triplet"] and not client.block_reason
    assert report["question_type"] == "bridge"
    for arm in runner.ARMS:
        row = report["arms"][arm]
        assert row["feedback"]["answer_em"] == 1
        assert row["gold_support_recall_by_stage"] == {
            "retrieved": 1,
            "provided": 1,
            "retained": 1,
        }
        saved = json.loads((tmp_path / "question" / f"{arm}_execution.json").read_text())
        assert saved["feedback"] is None and "gold_support_recall_by_stage" not in saved
    assert all("gold" not in str(messages).lower() for messages, _ in client.requests)


@pytest.mark.parametrize("fail_at", [1, 2, 4, 7, 9])
def test_failure_stops_all_later_arms_without_gold_or_retry(tmp_path, fail_at):
    client = Client(fail_at)

    class Example:
        question, candidate_context = Q, (E,)

        @property
        def gold(self):
            pytest.fail("incomplete triplet must not access gold")

    report = runner.run_question(Example(), client, tmp_path / "question")
    assert not report["complete_triplet"] and client.block_reason
    assert client.attempts == fail_at
    assert all(report["arms"][a]["feedback"] is None for a in runner.ARMS)
    assert sum(len(report["arms"][a]["calls"]) for a in runner.ARMS) == fail_at
    summary = runner.summarize([report], planned=12, started=1)
    assert summary["complete_scored_triplets"] == 0
    assert all(v["em"] is None for v in summary["methods"].values())
    assert (
        sum(v["cost_including_failures"]["recorded_attempts"] for v in summary["methods"].values())
        == fail_at
    )


def test_arm_order_is_deterministic_per_question():
    assert runner.arm_order(Q.question_id) == runner.arm_order(Q.question_id)
    assert set(runner.arm_order(Q.question_id)) == set(runner.ARMS)
    assert len({runner.arm_order(str(i)) for i in range(100)}) == 6


def test_unknown_arm_is_rejected_without_calls():
    client = Client()
    with pytest.raises(ValueError):
        runner.execute_arm("not-registered", Q, (E,), client)
    assert client.attempts == 0


def test_execute_preserves_partial_summary_on_interruption(monkeypatch, tmp_path):
    def interrupted(*args):
        raise RuntimeError("private detail")

    monkeypatch.setattr(runner, "run_question", interrupted)
    with pytest.raises(RuntimeError):
        runner.execute(Client(), [object()] * 12, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["started"] == 1 and summary["reported"] == 0
    assert summary["not_started"] == 11
    assert "private detail" not in (tmp_path / "summary.json").read_text()


def mock_plan(monkeypatch, reserved=0.9119286):
    examples = tuple(HotpotExample(Q, (E,), GOLD) for _ in range(12))
    monkeypatch.setattr(runner, "load_fresh_dev", lambda _: (examples, {"role": "synthetic"}))
    ledgers = []

    def history(root, *, reviewed_extra_ledgers):
        ledgers.append(reviewed_extra_ledgers)
        return {"prior_reserved_cny": reserved, "prior_unknown_cost_calls": 1}

    monkeypatch.setattr(runner, "reconcile_history", history)
    monkeypatch.setattr(runner, "_git_state", lambda: {"commit": "synthetic"})
    return ledgers


def test_plan_is_zero_api_has_separate_guard_version_and_full_budget(monkeypatch, tmp_path):
    ledgers = mock_plan(monkeypatch)

    def forbidden(*args, **kwargs):
        pytest.fail("dry plan must not read credentials or use API")

    monkeypatch.setattr(runner, "read_local_bailian_settings", forbidden)
    monkeypatch.setattr(runner, "LiveChatClient", forbidden)
    output = tmp_path / "plan"
    assert (
        runner.main(["--manifest", "unused", "--runs-root", str(tmp_path), "--output", str(output)])
        == 0
    )
    plan = json.loads((output / "plan.json").read_text())
    assert len(ledgers[0]) == 5
    assert plan["max_api_calls"] == 180 and plan["subcap_cny"] == 5
    assert plan["dual_guard_version"] == "dualrag-deterministic-guards-v2"
    assert plan["dual_baseline_id"] == runner.dualrag.GUARDED_BASELINE_ID
    assert len(plan["prompts"]) == 7
    assert plan["model"] == "qwen3.7-flash-2026-07-15"
    assert not plan["check24_used"] and not plan["official_dev_test_used"]
    snapshot = json.loads((output / "source_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["sha256"] == plan["source_snapshot_sha256"]
    assert not (tmp_path / f"{runner.RUN_ID}.claim.json").exists()


@pytest.mark.parametrize("name", ["outside", "same", "exists", "wrong-live-name", "claim"])
def test_path_and_one_use_check_precede_any_data_read(monkeypatch, tmp_path, name):
    monkeypatch.setattr(runner, "load_fresh_dev", lambda _: pytest.fail("no data read allowed"))
    output = (
        tmp_path.parent / "outside"
        if name == "outside"
        else tmp_path
        if name == "same"
        else tmp_path / runner.RUN_ID
    )
    if name == "exists":
        output.mkdir()
    if name == "claim":
        (tmp_path / f"{runner.RUN_ID}.claim.json").write_text("{}")
    if name == "wrong-live-name":
        output = tmp_path / "wrong"
    args = ["--manifest", "unused", "--runs-root", str(tmp_path), "--output", str(output)]
    if name in {"wrong-live-name", "claim"}:
        args.append("--allow-network")
    with pytest.raises(ValueError):
        runner.main(args)


def test_exhausted_cap_blocks_before_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "LIVE_PROTOCOL_ENABLED", True)
    mock_plan(monkeypatch, reserved=50)
    monkeypatch.setattr(runner, "read_local_bailian_settings", lambda _: pytest.fail("no key read"))
    with pytest.raises(ValueError, match="budget"):
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
