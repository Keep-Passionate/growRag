"""Offline integration fixtures: no live API, user corpus, keys, or quality claim."""

import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.controller import ASSESS_PROMPT_VERSION, GAP_QUERY_PROMPT_VERSION, ROUTE_PROMPT_VERSION
from growrag.experiments import run_bounded_system as runner
from growrag.experiments.api_client import ChatResponse
from growrag.experiments.prior_budget import ROOT_LEDGERS, reconcile_history
from growrag.experiments.protocol import BackendCallError, Evidence, GoldRecord, RuntimeQuestion

Q = RuntimeQuestion("synthetic-target", "Which town contains the fictional school?", "synthetic")
E = Evidence("e1", "School", 0, "The fictional school is in Fiction Town.")
GOLD = GoldRecord(Q.question_id, ("SECRET_GOLD_NOT_SENT_TO_API",), (("School", 0),))


class FakeClient:
    """Scripted JSON by component, with synthetic cost entries for accounting tests."""

    transport_source = "mock"
    config = SimpleNamespace(base_url="https://synthetic.invalid/v1", model="MOCK")

    def __init__(self, *, sufficient=False, reader_content=None, invalid_first_assessment=False):
        self.sufficient = sufficient
        self.reader_content = reader_content
        self.invalid_first_assessment = invalid_first_assessment
        self.calls, self.requests = [], []
        self.block_reason = None

    def complete(self, messages, *, trace_id, prompt_version):
        payload = json.loads(messages[1]["content"])
        self.requests.append({"prompt_version": prompt_version, "payload": payload})
        self.calls.append(
            {
                "trace_id": trace_id,
                "prompt_version": prompt_version,
                "estimated_actual_cny": 0.01,
                "input_tokens": 10,
                "output_tokens": 3,
                "api_requests": 0,
            }
        )
        if prompt_version == runner.READER_VERSION:
            value = self.reader_content or {"answer": "Fiction Town", "cited_evidence_ids": ["e1"]}
        elif prompt_version == ASSESS_PROMPT_VERSION:
            if self.invalid_first_assessment:
                value = {"bad": "assessment"}
            else:
                sufficient = self.sufficient or payload["has_previous_round"]
                value = {
                    "requirements": [
                        {
                            "description": "town relation",
                            "status": "supported",
                            "evidence_ids": ["e1"],
                        }
                    ]
                    + (
                        []
                        if sufficient
                        else [
                            {
                                "description": "precise school location",
                                "status": "missing",
                                "evidence_ids": [],
                            }
                        ]
                    ),
                    "sufficient": sufficient,
                    "useful_gain": None,
                    "gap": "" if sufficient else "Need clearer school relation",
                    "next_intent": "" if sufficient else "Find the school town",
                    "reason": "Synthetic fixture observation",
                }
        elif prompt_version == ROUTE_PROMPT_VERSION:
            value = {
                "action": "BASE" if payload["stage"] == "PRE" else "FRESH",
                "memory_id": None,
                "reason": "Synthetic routing choice",
                "condition_checks": [],
            }
        elif prompt_version == GAP_QUERY_PROMPT_VERSION:
            value = {"query": "fictional school location municipality"}
        else:
            raise AssertionError(f"Unexpected mocked component: {prompt_version}")
        return ChatResponse(
            value if isinstance(value, str) else json.dumps(value),
            "MOCK",
            "MOCK",
            f"mock-response-{len(self.calls)}",
            None,
            10,
            3,
            0,
            Path("synthetic-not-written"),
            "mock",
        )


class GuardedExample:
    question = Q
    candidate_context = (E,)

    def __init__(self, client, gold=GOLD):
        self.client, self._gold = client, gold
        self.gold_access_calls = []

    @property
    def gold(self):
        self.gold_access_calls.append(len(self.client.calls))
        return self._gold


@pytest.mark.parametrize("answer,refs", [("Fiction Town", ["e1"]), ("", [])])
def test_short_reader_accepts_answer_or_explicit_abstention(answer, refs):
    client = FakeClient(reader_content={"answer": answer, "cited_evidence_ids": refs})
    result = runner.APIShortAnswerReader(client).answer(Q, (E,))
    assert result.value.text == answer
    assert result.value.cited_evidence_ids == tuple(refs)
    assert result.usage.api_requests == 0
    assert client.requests[0]["payload"] == {"original_question": Q.text, "evidence": [asdict(E)]}


@pytest.mark.parametrize(
    "value",
    [
        {"answer": "Town", "cited_evidence_ids": []},
        {"answer": "", "cited_evidence_ids": ["e1"]},
        {"answer": "Town", "cited_evidence_ids": ["foreign"]},
        {"answer": "Town", "cited_evidence_ids": [True]},
        {"answer": "Town", "cited_evidence_ids": "e1"},
        {"answer": 3, "cited_evidence_ids": ["e1"]},
        {"answer": "x" * 1001, "cited_evidence_ids": ["e1"]},
        {"answer": "Town", "cited_evidence_ids": ["e1"], "extra": "not allowed"},
        '{"answer":"first","answer":"second","cited_evidence_ids":["e1"]}',
    ],
)
def test_short_reader_rejects_malformed_response_without_retry(value):
    client = FakeClient(reader_content=value)
    with pytest.raises(BackendCallError, match="invalid bounded reader"):
        runner.APIShortAnswerReader(client).answer(Q, (E,))
    assert len(client.calls) == 1


@pytest.mark.parametrize("question,evidence", [(GOLD, (E,)), (Q, (GOLD,)), (Q, [E])])
def test_short_reader_rejects_non_runtime_inputs_before_mock_call(question, evidence):
    client = FakeClient()
    with pytest.raises(TypeError):
        runner.APIShortAnswerReader(client).answer(question, evidence)
    assert not client.requests


def test_all_arms_run_before_gold_and_runtime_payloads_remain_gold_free(tmp_path):
    client = FakeClient()
    example = GuardedExample(client)
    report = runner.run_question(example, (), client, tmp_path / "question")
    assert set(report["arms"]) == set(runner.ARMS)
    assert all(row["status"] == "completed" for row in report["arms"].values())
    assert example.gold_access_calls and set(example.gold_access_calls) == {len(client.calls)}
    payload_text = json.dumps(client.requests)
    assert "SECRET_GOLD_NOT_SENT_TO_API" not in payload_text
    assert "supporting_facts" not in payload_text
    assert report["execution_kind"] == "mock" and report["memory_updated"] is False
    assert all(row["api_requests"] == 0 for row in client.calls)
    assert (tmp_path / "question" / "report.md").is_file()
    assert (tmp_path / "question" / "report.json").is_file()


def test_reflective_arm_reuses_prefix_once_and_counts_only_new_calls_incrementally(tmp_path):
    client = FakeClient()
    report = runner.run_question(GuardedExample(client), (), client, tmp_path / "question")
    base = report["arms"]["BASE1"]
    reflective = report["arms"]["REFLECTIVE2"]
    assert reflective["result"]["state"]["rounds"][0] == base["result"]["state"]["rounds"][0]
    assert reflective["shared_prefix"]["actually_collected_once"] is True
    assert reflective["shared_prefix"]["cost"] == base["incremental_cost"]
    assert len(reflective["calls"]) == 3  # Rewrite, Reader, assessor; no repeated prefix calls.
    assert len(base["calls"]) == 2
    assert reflective["result"]["events"][:2] == base["result"]["events"]
    observed_ids = [row["trace_id"] for arm in report["arms"].values() for row in arm["calls"]]
    assert len(observed_ids) == len(set(observed_ids)) == len(client.calls)
    summary = runner.summarize([report], planned=1)
    assert summary["arms"]["REFLECTIVE2"]["shadow_path_estimated_cny"] == pytest.approx(0.05)
    assert sum(
        row["incremental_cost"]["estimated_actual_cny"] for row in report["arms"].values()
    ) == pytest.approx(sum(row["estimated_actual_cny"] for row in client.calls))


def test_sufficient_prefix_does_not_trigger_reflective_rewrite_or_second_reader(tmp_path):
    client = FakeClient(sufficient=True)
    report = runner.run_question(GuardedExample(client), (), client, tmp_path / "question")
    reflective = report["arms"]["REFLECTIVE2"]
    assert reflective["calls"] == []
    assert len(reflective["result"]["state"]["rounds"]) == 1
    assert reflective["result"]["stop_reason"] == "sufficient_signal"
    assert reflective["incremental_cost"]["estimated_actual_cny"] == 0
    assert reflective["shared_prefix"]["cost"]["estimated_actual_cny"] == pytest.approx(0.02)


def test_gap_append_arm_has_same_prefix_without_an_extra_rewrite_api_call(tmp_path):
    client = FakeClient()
    report = runner.run_question(GuardedExample(client), (), client, tmp_path / "question")
    gap_arm = report["arms"]["S2G_GAP2"]
    reflective = report["arms"]["REFLECTIVE2"]
    assert gap_arm["shared_prefix"] == reflective["shared_prefix"]
    assert gap_arm["result"]["state"]["rounds"][0] == reflective["result"]["state"]["rounds"][0]
    assert len(gap_arm["calls"]) == 2  # Reader + judge, not an extra rewriting model call.
    assert all(row["prompt_version"] != GAP_QUERY_PROMPT_VERSION for row in gap_arm["calls"])
    query = gap_arm["result"]["state"]["rounds"][1]["search_query"]
    assert query == Q.text + " precise school location"
    assert gap_arm["rewrite_records"][0]["status"] == "local_rule"
    rewrite_events = [e for e in gap_arm["result"]["events"] if e["operation"] == "rewrite"]
    assert rewrite_events[0]["usage"]["api_requests"] == 0
    assert rewrite_events[0]["transport_source"] == "local_compute"


def test_judge_release_and_raw_gold_scores_are_not_conflated(tmp_path):
    client = FakeClient()
    actual_gold = GoldRecord(Q.question_id, ("Fiction Town",), (("School", 0),))
    report = runner.run_question(
        GuardedExample(client, actual_gold), (), client, tmp_path / "question"
    )
    base = report["arms"]["BASE1"]
    assert base["feedback"]["answer_em"] == 1
    assert base["answer_release"]["allowed_by_judge"] is False
    assert base["released_feedback"]["answer_em"] == 0
    assert report["arms"]["REFLECTIVE2"]["released_feedback"]["answer_em"] == 1


def test_unlabelled_example_runs_without_fabricating_scores(tmp_path):
    client = FakeClient(sufficient=True)
    report = runner.run_question(GuardedExample(client, None), (), client, tmp_path / "question")
    assert all(row["status"] == "completed" for row in report["arms"].values())
    assert all(row["feedback"] is None for row in report["arms"].values())
    summary = runner.summarize([report], planned=1)
    assert summary["complete"] == 0
    assert all(row["raw_em"] is None for row in summary["arms"].values())


def test_first_component_error_stops_other_arms_and_retains_failed_attempt_cost(tmp_path):
    client = FakeClient(invalid_first_assessment=True)
    report = runner.run_question(GuardedExample(client), (), client, tmp_path / "question")
    assert len(client.calls) == 2
    assert client.block_reason == "bounded_component_failure"
    assert report["arms"]["BASE1"]["status"] == "failed"
    assert report["arms"]["BASE1"]["incremental_cost"]["estimated_actual_cny"] == pytest.approx(
        0.02
    )
    assert all(report["arms"][name]["status"] == "not_executed" for name in runner.ARMS[1:])
    assert all(report["arms"][name]["feedback"] is None for name in runner.ARMS)
    failure = report["arms"]["BASE1"]["failure"]
    assert failure["phase"] == "assess" and failure["stop_reason"] == "assessment_error"
    assert failure["completed_rounds"] == 1
    assert report["executed_route_order"] == ["BASE1"]
    assert set(report["planned_route_order"]) == set(runner.ARMS)


def test_summary_uses_same_complete_cohort_not_different_arm_denominators(tmp_path):
    client = FakeClient(sufficient=True)
    complete = runner.run_question(GuardedExample(client), (), client, tmp_path / "question")
    partial = deepcopy(complete)
    partial["arms"]["ADAPTIVE_MEMORY2"]["feedback"] = None
    partial["arms"]["BASE1"]["feedback"]["answer_em"] = 1
    result = runner.summarize([complete, partial], planned=8)
    assert result["planned"] == 8 and result["started"] == 2
    assert result["complete"] == 1 and result["not_started"] == 6
    assert all(row["n"] == 1 for row in result["arms"].values())
    assert result["arms"]["BASE1"]["raw_em"] == 0


def test_summary_preserves_unknown_cost_and_missing_support_labels(tmp_path):
    client = FakeClient(sufficient=True)
    report = runner.run_question(GuardedExample(client), (), client, tmp_path / "question")
    report["arms"]["REFLECTIVE2"]["shared_prefix"]["cost"]["estimated_actual_cny"] = None
    report["arms"]["BASE1"]["feedback"]["retrieved_gold_support_recall"] = None
    summary = runner.summarize([report], planned=1)
    assert summary["arms"]["REFLECTIVE2"]["shadow_path_estimated_cny"] is None
    assert summary["arms"]["BASE1"]["support_recall"] is None


def test_summary_counts_an_interrupted_question_even_without_a_finished_report():
    summary = runner.summarize([], planned=8, attempted=1)
    assert summary["started"] == 1 and summary["reported"] == 0
    assert summary["not_started"] == 7 and summary["complete"] == 0


def test_prompt_manifest_freezes_reader_and_all_controller_prompts():
    manifest = runner.prompt_manifest()
    assert runner.READER_VERSION in manifest
    assert len(manifest) == 4
    assert all(len(value["sha256"]) == 64 and value["text"] for value in manifest.values())
    assert "ONLY the supplied evidence" in manifest[runner.READER_VERSION]["text"]


def test_new_reviewed_budget_root_is_counted_once_with_previous_roots(tmp_path):
    relatives = (*ROOT_LEDGERS, *runner.REVIEWED_EXTRA)
    for i, relative in enumerate(relatives):
        path = tmp_path / relative
        audit = path.parent / "api_audit" / f"synthetic-{i}.json"
        audit.parent.mkdir(parents=True)
        row = {
            "trace_id": f"synthetic-{i}",
            "audit_path": str(audit),
            "status": "completed",
            "input_tokens": 1,
            "output_tokens": 1,
            "api_requests": 1,
            "estimated_actual_cny": 0.000001,
            "reserved_cny": 0.01,
        }
        audit.write_text(json.dumps({**row, "transport_source": "live_api"}), encoding="utf-8")
        path.write_text(
            json.dumps(
                {
                    "calls": [row],
                    "api_requests": 1,
                    "reserved_cny": 0.01,
                    "limits": {"input_per_million_cny": 0.2, "output_per_million_cny": 0.8},
                }
            ),
            encoding="utf-8",
        )
    history = reconcile_history(tmp_path, reviewed_extra_ledgers=runner.REVIEWED_EXTRA)
    assert history["prior_api_requests"] == 5
    assert history["prior_reserved_cny"] == pytest.approx(0.05)
    assert history["prior_known_estimated_cny"] == pytest.approx(0.000005)
