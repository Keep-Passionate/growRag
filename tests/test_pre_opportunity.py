"""Synthetic contracts only. No credentials, network, or held-out labels."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_representation_views import _bundle

from growrag.experiments.api_client import APIRequestError, ChatConfig, ChatResponse
from growrag.experiments.budget import BudgetedChatClient, PriceLimits
from growrag.experiments.hotpot import parse_hotpot_example
from growrag.experiments.llm_adapters import READER_PROMPT_VERSION
from growrag.experiments.pre_pilot import PRE_INTENT, write_json
from growrag.experiments.protocol import GoldRecord, RuntimeQuestion
from growrag.experiments.run_pre_opportunity import (
    VIEWS,
    DurableBudgetClient,
    FrozenCandidate,
    decode_view,
    run_question,
    selector_payload,
    shortlist,
    summarize_opportunities,
)


def candidate(label="alpha"):
    bundle = _bundle(label)
    view = bundle.canonical_action
    payload = json.loads(view.text)
    payload["intent"] = PRE_INTENT
    view = replace(view, intent=PRE_INTENT, text=json.dumps(payload))
    return FrozenCandidate(
        view, bundle.source_question.text, bundle.rewritten_query, bundle.source_fingerprint
    )


def example():
    return parse_hotpot_example(
        {
            "_id": "debug",
            "question": "When was another University founded?",
            "context": [["History", ["The University was established in 1900."]]],
            "answer": "GOLD_ONLY_MARKER",
            "supporting_facts": [["History", 0]],
        },
        dataset="synthetic",
    )


class Delegate:
    transport_source = "mock"

    def __init__(self, fail_at=None, unchanged=False):
        self.attempts, self.prompts, self.fail_at = 0, [], fail_at
        self.unchanged = unchanged
        self.config = ChatConfig(
            "https://example.invalid/v1",
            "fixture",
            "UNUSED",
            150,
            max_output_tokens=768,
            enable_thinking=False,
            temperature=0,
        )

    def complete(self, messages, *, trace_id, prompt_version):
        self.attempts += 1
        self.prompts.append((prompt_version, messages))
        assert "GOLD_ONLY_MARKER" not in json.dumps(messages)
        assert "SOURCE_ANSWER_SENTINEL" not in json.dumps(messages)
        if self.attempts == self.fail_at:
            raise APIRequestError("synthetic", api_requests=1)
        payload = json.loads(messages[-1]["content"])
        if "representation-selector" in prompt_version:
            value = {"candidate_id": "c1"}
        elif prompt_version == READER_PROMPT_VERSION:
            assert "optional_historical_procedure" not in payload
            assert payload["original_question"] == example().question.text
            value = {
                "answer": "1900",
                "cited_evidence_ids": [payload["evidence"][0]["evidence_id"]],
            }
        else:
            if "optional_historical_procedure" in payload:
                assert set(payload["optional_historical_procedure"]) == {"body"}
                assert "conditions" not in payload
                assert "example" not in payload
            value = {
                "query": example().question.text
                if self.unchanged
                else "University established year"
            }
        return ChatResponse(
            json.dumps(value),
            "fixture",
            "fixture",
            "r",
            "r",
            10,
            4,
            0.01,
            Path(f"synthetic-{self.attempts}"),
            "mock",
        )


def test_shortlist_is_fixed_across_views_and_excludes_same_question_new_id():
    pool = tuple(candidate(label) for label in ("gamma", "beta", "alpha", "delta"))
    query = example().question
    assert shortlist(query, pool) == shortlist(query, tuple(reversed(pool)))
    assert len(shortlist(query, pool)) == 3
    assert not shortlist(RuntimeQuestion("different-id", pool[0].source_question), (pool[0],))


def test_condition_projection_only_changes_selector_not_action():
    pool = (candidate(),)
    selected = shortlist(example().question, pool)
    left = selector_payload(example().question, selected, VIEWS[0])
    right = selector_payload(example().question, selected, VIEWS[1])
    a = json.loads(left["candidates"][0]["text"])
    b = json.loads(right["candidates"][0]["text"])
    assert b.pop("inferred_conditions_not_verified_guarantees")
    assert a == b
    assert "SOURCE_ANSWER_SENTINEL" not in json.dumps(right)
    with pytest.raises(ValueError):
        selector_payload(example().question, selected, "M4")


def test_all_candidates_execute_after_both_choices_and_gold_is_offline(tmp_path):
    delegate = Delegate()
    client = BudgetedChatClient(delegate, PriceLimits(budget_cny=3))
    report = run_question(
        example(),
        tuple(candidate(label) for label in ("alpha", "beta", "gamma")),
        client,
        tmp_path / "q",
    )
    assert len(report["outcomes"]) == 6
    assert delegate.attempts == 13
    assert all("representation-selector" in p for p, _ in delegate.prompts[:2])
    assert all(row["status"] == "completed" for row in report["outcomes"].values())
    assert report["synthetic_not_model_quality"] is True
    assert (tmp_path / "q" / "report.md").exists()
    assert summarize_opportunities([report])["complete_questions"] == 1


def test_changed_gold_leaves_same_runtime_queries_and_choices(tmp_path):
    pool = (candidate(),)
    first, second = Delegate(), Delegate()
    a = run_question(example(), pool, BudgetedChatClient(first, PriceLimits()), tmp_path / "a")
    altered = replace(example(), gold=GoldRecord("debug", ("1900",), (("History", 0),)))
    b = run_question(altered, pool, BudgetedChatClient(second, PriceLimits()), tmp_path / "b")
    assert first.prompts == second.prompts
    assert [x["query"] for x in a["outcomes"].values()] == [
        x["query"] for x in b["outcomes"].values()
    ]


def test_no_candidate_means_no_fake_selection_calls(tmp_path):
    delegate = Delegate()
    report = run_question(
        example(), (), BudgetedChatClient(delegate, PriceLimits()), tmp_path / "q"
    )
    assert delegate.attempts == 5
    assert len(report["outcomes"]) == 3
    assert all(v["status"] == "no_candidates" for v in report["selections"].values())


def test_branch_error_stops_all_later_paid_requests(tmp_path):
    delegate = Delegate(fail_at=4)
    client = BudgetedChatClient(delegate, PriceLimits())
    report = run_question(example(), (candidate(),), client, tmp_path / "q")
    assert delegate.attempts == 4
    assert client.block_reason == "transport_failure"
    assert any(row["status"] != "completed" for row in report["outcomes"].values())
    assert summarize_opportunities([report])["complete_questions"] == 0


def test_selector_error_stops_before_branches(tmp_path):
    delegate = Delegate(fail_at=1)
    report = run_question(
        example(), (candidate(),), BudgetedChatClient(delegate, PriceLimits()), tmp_path / "q"
    )
    assert delegate.attempts == 1
    assert report["selections"][VIEWS[0]]["status"] == "selector_error"
    assert report["selections"][VIEWS[1]]["status"] == "not_executed_after_error"
    assert (tmp_path / "q" / "report.json").exists()
    assert not (tmp_path / "q" / "BASE_execution.json").exists()
    summary = summarize_opportunities([report])
    assert summary["not_started_questions"] == 7
    assert summary["incomplete_started_questions"] == 1


def test_selected_memory_unchanged_query_is_not_executed_reuse(tmp_path):
    delegate = Delegate(unchanged=True)
    report = run_question(
        example(), (candidate(),), BudgetedChatClient(delegate, PriceLimits()), tmp_path / "q"
    )
    assert report["outcomes"]["c1"]["effective_action"] == "BASE"
    assert report["outcomes"]["c1"]["fallback_reason"] == "unchanged_query"
    assert report["outcomes"]["c1"]["call_cost"]["recorded_attempts"] == 2
    policy = summarize_opportunities([report])["policies"][VIEWS[0]]
    assert policy["reuse_selected"] == 1
    assert policy["reuse_executed"] == 0
    assert policy["selected_memory_fell_back_to_base"] == 1
    opportunity = summarize_opportunities([report])["opportunity"]["FRESH"]
    assert opportunity["n_with_executed_reuse"] == 0
    assert opportunity["best_executed_reuse_mean_delta"] is None


def test_exclusive_authorization_claim_not_overwritten(tmp_path):
    claim = tmp_path / "claim.json"
    write_json(claim, {"output": "first"})
    with pytest.raises(FileExistsError):
        write_json(claim, {"output": "second"})
    assert json.loads(claim.read_text())["output"] == "first"


def test_crash_started_question_is_not_counted_unstarted():
    summary = summarize_opportunities([], planned_questions=8, started_questions=1)
    assert summary["not_started_questions"] == 7
    assert summary["started_without_report"] == 1
    assert summary["total_not_complete_questions"] == 8


def test_write_ahead_and_error_reservations_survive(tmp_path):
    delegate = Delegate(fail_at=1)
    client = DurableBudgetClient(delegate, PriceLimits(), tmp_path / "journal")
    with pytest.raises(APIRequestError):
        client.complete([{"role": "user", "content": "{}"}], trace_id="t", prompt_version="fixture")
    before = json.loads((tmp_path / "journal" / "0000_intent.json").read_text())
    after = json.loads((tmp_path / "journal" / "0000_after.json").read_text())
    assert before["potential_reserved_cny"] == after["reserved_cny"] > 0
    assert after["estimated_actual_cny"] is None
    assert "messages" not in before
    with pytest.raises(FileExistsError):
        DurableBudgetClient(delegate, PriceLimits(), tmp_path / "journal")


def test_decode_projection_rejects_stage_tampering():
    from dataclasses import asdict

    data = asdict(candidate().view)
    data["stage"] = "post_retrieval"
    with pytest.raises(ValueError):
        decode_view(data)
