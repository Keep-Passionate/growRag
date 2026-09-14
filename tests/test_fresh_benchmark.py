"""Offline contracts only: synthetic responses are not evidence of RAG quality."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from growrag.experiments.api_client import ChatConfig, ChatResponse
from growrag.experiments.budget import BudgetedChatClient, PriceLimits
from growrag.experiments.data_protocol import role_for_question
from growrag.experiments.fresh_benchmark import (
    VARIANTS,
    call_totals,
    run_fresh_benchmark,
    summarize,
)
from growrag.experiments.hotpot import parse_hotpot_example
from growrag.experiments.llm_adapters import READER_PROMPT_VERSION
from growrag.experiments.run_fresh_benchmark import (
    PILOT_MODEL,
    carryover_budget,
    claim_continuation,
    select_records,
)


def rows_for_selection():
    rows = []
    for i in range(200):
        text = f"What happened to entity {i} in year 1950?"
        if role_for_question(text) == "selector_train":
            rows.append(
                {
                    "_id": str(i),
                    "question": text,
                    "type": "bridge" if len(rows) % 2 else "comparison",
                }
            )
    return rows


def test_selection_is_independent_of_input_order_and_answers():
    rows = rows_for_selection()
    manifest = {
        "exclusions": {"excluded_union_question_ids": []},
        "source_expansion_order": [],
        "selected": {"target": []},
    }
    chosen = select_records(rows, manifest, count=4)
    altered = [{**r, "answer": "UNUSED GOLD"} for r in reversed(rows)]
    assert [r["_id"] for r in chosen] == [
        r["_id"] for r in select_records(altered, manifest, count=4)
    ]
    manifest["selected"]["target"] = [chosen[0]["_id"]]
    assert chosen[0]["_id"] not in [r["_id"] for r in select_records(rows, manifest, count=4)]
    assert len({r["type"] for r in chosen}) == 2


def test_duplicate_questions_are_not_selected():
    rows = rows_for_selection()
    manifest = {
        "exclusions": {"excluded_union_question_ids": []},
        "source_expansion_order": [],
        "selected": {"target": []},
    }
    one = select_records(rows, manifest, count=2)[0]
    duplicated = rows + [{**one, "_id": "duplicate"}]
    assert one["question"] not in [
        r["question"] for r in select_records(duplicated, manifest, count=2)
    ]


def prior_budget_file(tmp_path):
    path = tmp_path / "source" / "final_budget.json"
    path.parent.mkdir()
    data = {
        "limits": {"budget_cny": 5.0, "input_per_million_cny": 0.2, "output_per_million_cny": 0.8},
        "block_reason": None,
        "api_requests": 1,
        "reserved_cny": 0.1,
        "estimated_actual_cny": 0.01,
        "input_tokens": 2,
        "output_tokens": 3,
        "calls": [
            {
                "status": "completed",
                "returned_model": PILOT_MODEL,
                "reserved_cny": 0.1,
                "estimated_actual_cny": 0.01,
                "input_tokens": 2,
                "output_tokens": 3,
            }
        ],
    }
    path.write_text(json.dumps(data))
    return path


def test_carryover_and_exclusive_claim(tmp_path):
    path = prior_budget_file(tmp_path)
    carry = carryover_budget(path)
    assert carry["prior_reserved_cny"] == 0.1
    assert carry["new_run_subcap_cny"] == 1.0
    claim = claim_continuation(path, tmp_path / "fresh", carry)
    assert claim.exists()
    with pytest.raises(FileExistsError):
        claim_continuation(path, tmp_path / "fresh-other", carry)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reserved_cny", 0.2),
        ("estimated_actual_cny", None),
        ("input_tokens", 9),
        ("block_reason", "unknown_usage"),
        ("api_requests", 2),
    ],
)
def test_bad_prior_ledger_blocks(tmp_path, field, value):
    path = prior_budget_file(tmp_path)
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, TypeError)):
        carryover_budget(path)


def test_unknown_cost_is_not_zero():
    assert call_totals([{"input_tokens": None}])["estimated_actual_cny"] is None
    assert call_totals([])["estimated_actual_cny"] == 0
    assert call_totals([{"api_requests": 0}])["api_requests"] == 0
    assert call_totals([{"api_requests": None}])["api_requests"] is None
    assert call_totals([{"api_requests": 0}])["recorded_attempts"] == 1


class MockDelegate:
    transport_source = "mock"

    def __init__(self, example):
        self.example, self.attempts, self.payloads = example, 0, []
        self.config = ChatConfig(
            "https://example.invalid/v1",
            "fixture",
            "UNUSED",
            max_calls=7,
            max_output_tokens=768,
            temperature=0,
        )

    def complete(self, messages, *, trace_id, prompt_version):
        self.attempts += 1
        payload = json.loads(messages[-1]["content"])
        self.payloads.append(payload)
        assert "GOLD_ONLY_MARKER" not in json.dumps(messages)
        if prompt_version == READER_PROMPT_VERSION:
            assert "SYNTHETIC_PASSAGE_MARKER" not in json.dumps(messages)
            value = {
                "answer": "1900",
                "cited_evidence_ids": [payload["evidence"][0]["evidence_id"]],
            }
        elif "query2doc" in prompt_version.lower():
            value = {"passage": "SYNTHETIC_PASSAGE_MARKER founding history University"}
        else:
            value = {"query": "University founding year"}
        return ChatResponse(
            json.dumps(value),
            "fixture",
            "fixture",
            "fixture-id",
            "fixture-request",
            20,
            10,
            0.001,
            Path("MOCK_NOT_REAL.json"),
            "mock",
        )


def test_all_arms_trace_no_gold_and_no_pseudo_evidence(tmp_path):
    example = parse_hotpot_example(
        {
            "_id": "fixture",
            "question": "When was the University established?",
            "answer": "1900",
            "supporting_facts": [["University", 0]],
            "context": [["University", ["The University was founded in 1900."]]],
        },
        dataset="synthetic-train",
    )
    example = replace(example, gold=replace(example.gold, answers=("1900", "GOLD_ONLY_MARKER")))
    delegate = MockDelegate(example)
    client = BudgetedChatClient(delegate, PriceLimits())
    report = run_fresh_benchmark((example,), client, tmp_path)
    assert delegate.attempts == 7
    assert report["completed_questions"] == 1
    assert set(report["methods"]) == set(VARIANTS)
    assert report["qpp_executed"] is False
    assert report["memory_updated"] is False
    assert (tmp_path / "questions/000/trace.md").exists()
    assert len(list((tmp_path / "questions/000").glob("*_execution.json"))) == 4
    assert sum(v["cost"]["api_requests"] for v in report["methods"].values()) == 7
    assert report["methods"]["BASE"]["stage_costs"]["query_generation"]["api_requests"] == 0


def test_invalid_first_response_stops_before_any_next_call(tmp_path):
    example = parse_hotpot_example(
        {
            "_id": "fixture",
            "question": "When was the University established?",
            "answer": "1900",
            "supporting_facts": [["University", 0]],
            "context": [["University", ["The University was founded in 1900."]]],
        },
        dataset="synthetic-train",
    )

    class InvalidDelegate(MockDelegate):
        def complete(self, *args, **kwargs):
            return replace(super().complete(*args, **kwargs), content='{"invalid": true}')

    delegate = InvalidDelegate(example)
    client = BudgetedChatClient(delegate, PriceLimits())
    with pytest.raises(RuntimeError, match="incomplete"):
        run_fresh_benchmark((example,), client, tmp_path)
    assert delegate.attempts == 1
    assert (tmp_path / "observations.json").exists()


def summary_arm(em, *, cost=0.01, seconds=1.0):
    """Synthetic accounting fixture; never a live experiment observation."""
    return {
        "feedback": (
            {"answer_em": em, "answer_f1": em, "retrieved_gold_support_recall": em}
            if em is not None
            else None
        ),
        "wall_seconds": seconds,
        "index_build_seconds": 0.001,
        "result": {"component_events": [{"operation": "rag.retrieve", "elapsed_seconds": 0.002}]},
        "model_calls": [
            {
                "prompt_version": READER_PROMPT_VERSION,
                "api_requests": 1,
                "input_tokens": 10,
                "output_tokens": 5,
                "estimated_actual_cny": cost,
            }
        ],
    }


def test_complete_case_methods_use_same_questions_without_hiding_partial_costs():
    complete = {"question_id": "complete", "arms": {name: summary_arm(1) for name in VARIANTS}}
    complete["arms"]["QUERY2DOC"] = summary_arm(0)
    partial = {"question_id": "partial", "arms": {"BASE": summary_arm(0, cost=0.05, seconds=9)}}
    failed = {"question_id": "failed", "arms": {name: summary_arm(0) for name in VARIANTS}}
    failed["arms"]["QUERY2DOC"] = summary_arm(None, cost=None)
    report = summarize([complete, partial, failed], planned_count=32)
    assert report["complete_case_count"] == report["completed_questions"] == 1
    assert report["complete_case_question_ids"] == ["complete"]
    cohort = report["complete_case_methods"]
    assert all(value["scored_questions"] == 1 for value in cohort.values())
    assert cohort["BASE"]["mean_em"] == 1
    assert cohort["BASE"]["mean_measured_seconds"] == 1
    assert cohort["BASE"]["cost"]["estimated_actual_cny"] == 0.01
    assert cohort["QUERY2DOC"]["harms_vs_base"] == 1
    # Original audit scope is deliberately preserved, not overwritten by the cohort.
    raw = report["methods"]
    assert raw["BASE"]["scored_questions"] == 3
    assert raw["BASE"]["mean_em"] == pytest.approx(1 / 3)
    assert raw["BASE"]["cost"]["estimated_actual_cny"] == pytest.approx(0.07)
    assert raw["QUERY2DOC"]["cost"]["estimated_actual_cny"] is None


@pytest.mark.parametrize(
    "rows", [[], [{"question_id": "only-base", "arms": {"BASE": summary_arm(1)}}]]
)
def test_no_complete_four_arm_cohort_has_unknown_quality_not_zero(rows):
    report = summarize(rows, planned_count=32)
    assert report["complete_case_count"] == 0
    assert report["complete_case_question_ids"] == []
    for value in report["complete_case_methods"].values():
        assert value["scored_questions"] == 0
        assert value["mean_em"] is None
        assert value["mean_f1"] is None
        assert value["mean_support_recall"] is None
        assert value["mean_measured_seconds"] is None
        assert value["cost"]["api_requests"] == 0
