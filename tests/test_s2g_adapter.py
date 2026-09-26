"""Offline diagnostic contracts only; no API calls or accuracy claims."""

import json
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments.api_client import ChatResponse
from growrag.experiments.protocol import (
    CallResult,
    Evidence,
    ExecutionKind,
    GoldRecord,
    RuntimeQuestion,
    Usage,
)
from growrag.experiments.s2g_adapter import (
    ADAPTATION_NOTES,
    PROMPT_VERSIONS,
    S2GAdapter,
    S2GLimits,
    build_gap_query,
)

Q = RuntimeQuestion("synthetic", "When was Northbridge founded?", "synthetic-only")
E = Evidence("s0", "Northbridge", 0, "Northbridge was founded in 1901.")
E2 = Evidence("s1", "Northbridge", 1, "Northbridge is a university.")
GAP = {
    "category": "attribute",
    "target": "Northbridge",
    "slot": "founding date",
    "description": "Find the date when Northbridge was founded.",
}
INSUFFICIENT = {"sufficient": False, "gap_items": [GAP]}
SUFFICIENT = {"sufficient": True, "gap_items": []}
ANSWER = {"answer": "1901", "cited_evidence_ids": ["s0"]}
EMPTY = {"answer": "", "cited_evidence_ids": []}


class Client:
    transport_source = "mock"
    config = SimpleNamespace(
        base_url="https://test.invalid/v1",
        model="TEST",
        max_output_tokens=512,
        json_schema_mode=False,
    )

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return ChatResponse(
            value if isinstance(value, str) else json.dumps(value),
            "TEST",
            "TEST",
            "response",
            "request",
            12,
            13,
            0,
            Path("offline-only.json"),
            "mock",
        )


class Retriever:
    execution_kind = ExecutionKind.MOCK

    def __init__(self, responses=((E,),)):
        self.responses, self.requests = list(responses), []

    def retrieve(self, query, *, top_k):
        self.requests.append((query, top_k))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return CallResult(value, usage=Usage(0, 0, 0), transport_source="local_compute")


def test_constructor_is_offline_and_hard_bound_is_five_model_calls():
    client, retriever = Client([]), Retriever()
    adapter = S2GAdapter(client, retriever)
    assert client.requests == retriever.requests == []
    assert adapter.budget_envelope()["max_model_calls"] == 5
    assert adapter.budget_envelope()["max_retrieval_calls"] == 2
    assert adapter.budget_envelope()["reserved_output_token_bound"] == 2560
    assert "not a verbatim" in " ".join(ADAPTATION_NOTES)
    assert "3B" in " ".join(ADAPTATION_NOTES)


def test_original_query_is_retrieved_first_then_original_sentences_reach_shared_reader():
    client = Client([{"sentence_ids": ["s0"]}, SUFFICIENT, ANSWER])
    retriever = Retriever()
    result = S2GAdapter(client, retriever).run(Q)
    assert result.status == "completed" and result.stop_reason == "judge_sufficient"
    assert result.completed_rounds == 1
    assert retriever.requests == [(Q.text, 6)]
    assert result.observed_evidence == result.evidence_context == (E,)
    assert result.answer.text == "1901"
    assert [event.operation for event in result.events] == [
        "retrieve",
        "extract",
        "judge",
        "answer",
    ]
    assert [req[1]["prompt_version"] for req in client.requests] == [
        PROMPT_VERSIONS[op] for op in ("extract", "judge", "answer")
    ]
    answer_input = json.loads(client.requests[-1][0][1]["content"])
    assert answer_input == {"original_question": Q.text, "evidence": [asdict(E)]}
    assert result.usage.api_requests == 0
    assert result.usage.input_tokens is None
    assert "gold" not in json.dumps(asdict(result)).lower()


def test_two_rounds_accumulate_deduplicate_and_use_only_first_gap():
    client = Client(
        [
            {"sentence_ids": ["s1"]},
            {"sufficient": False, "gap_items": [GAP, {**GAP, "target": "IgnoredSecondGap"}]},
            {"sentence_ids": ["s1", "s0"]},
            SUFFICIENT,
            ANSWER,
        ]
    )
    retriever = Retriever(((E2,), (E, E2)))
    result = S2GAdapter(client, retriever).run(Q)
    assert result.status == "completed" and result.completed_rounds == 2
    assert result.evidence_context == (E2, E)
    assert result.observed_evidence == (E2, E)
    assert retriever.requests[1][0] == f"{Q.text} Northbridge founding date"
    assert len(client.requests) == 5
    second_extract = [s for s in result.steps if s["operation"] == "extract"][1]
    assert second_extract["output"]["sentence_ids"] == ["s0", "s1"]
    assert second_extract["selected_evidence"] == [asdict(E), asdict(E2)]


def test_query_builder_description_fallback_and_no_free_rewrite():
    assert build_gap_query(Q.text, [{**GAP, "slot": ""}]) == f"{Q.text} {GAP['description']}"
    assert build_gap_query(Q.text, []) == Q.text
    with pytest.raises(ValueError):
        build_gap_query("x" * 2000, [GAP])


@pytest.mark.parametrize("field", list(S2GLimits.__dataclass_fields__))
@pytest.mark.parametrize("value", [0, -1, True, 1.5, 100000])
def test_limits_are_bounded_positive_integers(field, value):
    with pytest.raises(ValueError):
        S2GLimits(**{field: value})


def test_limits_immutable_and_extraction_cannot_exceed_retrieval():
    with pytest.raises(FrozenInstanceError):
        S2GLimits().max_rounds = 10
    with pytest.raises(ValueError):
        S2GLimits(top_k=1)


def test_runtime_rejects_gold_and_overlong_question_before_any_call():
    client, retriever = Client([]), Retriever()
    adapter = S2GAdapter(client, retriever)
    with pytest.raises(TypeError):
        adapter.run(GoldRecord("q", ("1901",)))
    with pytest.raises(ValueError):
        adapter.run(RuntimeQuestion("q", "x" * 2001))
    assert client.requests == retriever.requests == []


@pytest.mark.parametrize(
    "bad",
    [
        {"sentence_ids": ["unknown"]},
        {"sentence_ids": ["s0", "s0"]},
        {"sentence_ids": [0]},
        {"sentence_ids": "s0"},
        {"sentence_ids": [], "summary": "fact"},
        '{"sentence_ids": [], "sentence_ids": ["s0"]}',
        "not JSON",
        [],
    ],
)
def test_bad_extraction_stops_no_retry_or_answer(bad):
    client = Client([bad])
    result = S2GAdapter(client, Retriever()).run(Q)
    assert result.status == "failed" and result.stop_reason == "extract_error"
    assert result.answer is None and len(client.requests) == 1
    assert result.observed_evidence == (E,) and result.evidence_context == ()
    assert result.events[-1].audit_path == "offline-only.json"


@pytest.mark.parametrize(
    "bad",
    [
        {"sufficient": "true", "gap_items": []},
        {"sufficient": 1, "gap_items": []},
        {"sufficient": True, "gap_items": [GAP]},
        {"sufficient": False, "gap_items": [{**GAP, "category": "invented"}]},
        {"sufficient": False, "gap_items": [{**GAP, "target": "", "slot": "", "description": ""}]},
        {"sufficient": False, "gap_items": [{**GAP, "answer": "1901"}]},
        {"sufficient": False, "gap_items": [GAP] * 7},
    ],
)
def test_bad_judge_output_stops_before_further_retrieval(bad):
    client, retriever = Client([{"sentence_ids": ["s0"]}, bad]), Retriever()
    result = S2GAdapter(client, retriever).run(Q)
    assert result.status == "failed" and result.stop_reason == "judge_error"
    assert len(client.requests) == 2 and len(retriever.requests) == 1


def test_empty_context_cannot_be_treated_as_sufficient():
    client = Client([{"sentence_ids": []}, SUFFICIENT, EMPTY])
    result = S2GAdapter(client, Retriever(((),))).run(Q)
    assert result.status == "completed" and result.stop_reason == "no_new_evidence"
    assert result.answer.text == "" and result.completed_rounds == 1
    judge = next(step for step in result.steps if step["operation"] == "judge")
    assert judge["empty_context_sufficiency_blocked"] and not judge["effective_sufficient"]


def test_new_retrieved_but_unselected_sentence_does_not_fake_evidence_progress():
    client = Client(
        [{"sentence_ids": ["s0"]}, INSUFFICIENT, {"sentence_ids": ["s0"]}, INSUFFICIENT, ANSWER]
    )
    result = S2GAdapter(client, Retriever(((E,), (E, E2)))).run(Q)
    assert result.status == "completed" and result.stop_reason == "no_new_evidence"
    assert result.observed_evidence == (E, E2) and result.evidence_context == (E,)


def test_round_limit_still_calls_shared_reader_not_a_paper_success_claim():
    client = Client([{"sentence_ids": ["s0"]}, INSUFFICIENT, ANSWER])
    result = S2GAdapter(client, Retriever(), limits=S2GLimits(max_rounds=1)).run(Q)
    assert result.status == "completed" and result.stop_reason == "round_limit"
    assert len(client.requests) == 3


def test_no_gap_stops_with_an_answer_attempt():
    client = Client([{"sentence_ids": ["s0"]}, {"sufficient": False, "gap_items": []}, ANSWER])
    result = S2GAdapter(client, Retriever()).run(Q)
    assert result.stop_reason == "no_gap" and result.status == "completed"


def test_current_round_ids_only_even_when_previous_evidence_is_known():
    client = Client([{"sentence_ids": ["s0"]}, INSUFFICIENT, {"sentence_ids": ["s0"]}])
    result = S2GAdapter(client, Retriever(((E,), (E2,)))).run(Q)
    assert result.stop_reason == "extract_error"
    assert result.evidence_context == (E,)


def test_changed_evidence_under_same_id_is_rejected_before_second_extraction():
    client = Client([{"sentence_ids": ["s0"]}, INSUFFICIENT])
    result = S2GAdapter(client, Retriever(((E,), (replace(E, text="Different fact"),)))).run(Q)
    assert result.stop_reason == "retrieve_error" and len(client.requests) == 2


def test_prompt_and_response_caps_fail_without_hidden_retry():
    client = Client([])
    result = S2GAdapter(client, Retriever(), limits=S2GLimits(max_prompt_bytes=1)).run(Q)
    assert result.stop_reason == "extract_error" and client.requests == []
    assert result.events[-1].usage == Usage(0, 0, 0)
    client = Client(["x" * 20])
    result = S2GAdapter(client, Retriever(), limits=S2GLimits(max_response_bytes=10)).run(Q)
    assert result.stop_reason == "extract_error"
    assert result.steps[-1]["raw_output_truncated"]


def test_unknown_failure_usage_is_not_zero_and_no_retry():
    client = Client([RuntimeError("server response must not enter public audit")])
    result = S2GAdapter(client, Retriever()).run(Q)
    assert result.status == "failed" and result.usage.api_requests is None
    assert len(client.requests) == 1
    assert "server response" not in json.dumps(asdict(result))


def test_no_cross_question_memory():
    client = Client(
        [{"sentence_ids": ["s0"]}, SUFFICIENT, ANSWER, {"sentence_ids": []}, SUFFICIENT, EMPTY]
    )
    adapter = S2GAdapter(client, Retriever(((E,), ())))
    first, second = adapter.run(Q), adapter.run(RuntimeQuestion("other", "Different question?"))
    assert first.evidence_context == (E,) and second.evidence_context == ()
    assert json.loads(client.requests[4][0][1]["content"])["evidence"] == []


def test_live_unbudgeted_client_and_mixed_provenance_are_rejected():
    client = Client([])
    client.transport_source = "live_api"
    with pytest.raises(ValueError, match="mix"):
        S2GAdapter(client, Retriever())
    retriever = Retriever()
    retriever.execution_kind = ExecutionKind.REAL
    with pytest.raises(ValueError, match="budget"):
        S2GAdapter(client, retriever)


def test_unsupported_schema_mode_is_rejected():
    client = Client([])
    client.config = SimpleNamespace(**{**vars(client.config), "json_schema_mode": True})
    with pytest.raises(ValueError, match="JSON-object"):
        S2GAdapter(client, Retriever())


def test_repeated_query_guard_prevents_second_retrieval(monkeypatch):
    monkeypatch.setattr(
        "growrag.experiments.s2g_adapter.build_gap_query", lambda q, gaps, **kwargs: q
    )
    client, retriever = Client([{"sentence_ids": ["s0"]}, INSUFFICIENT, ANSWER]), Retriever()
    result = S2GAdapter(client, retriever).run(Q)
    assert result.status == "completed" and result.stop_reason == "repeated_query"
    assert len(retriever.requests) == 1 and len(client.requests) == 3


def test_answer_cannot_cite_unselected_retrieval_and_records_raw_output():
    invalid_answer = {"answer": "University", "cited_evidence_ids": ["s1"]}
    client = Client([{"sentence_ids": ["s0"]}, SUFFICIENT, invalid_answer])
    result = S2GAdapter(client, Retriever(((E, E2),))).run(Q)
    assert result.status == "failed" and result.stop_reason == "answer_error"
    assert result.observed_evidence == (E, E2) and result.evidence_context == (E,)
    assert json.loads(result.steps[-1]["raw_output"]) == invalid_answer
    assert result.answer is None and len(client.requests) == 3


def test_same_response_size_limit_applies_to_shared_reader_output():
    client = Client(
        [{"sentence_ids": ["s0"]}, SUFFICIENT, {"answer": "x" * 300, "cited_evidence_ids": ["s0"]}]
    )
    result = S2GAdapter(client, Retriever(), limits=S2GLimits(max_response_bytes=150)).run(Q)
    assert result.status == "failed" and result.stop_reason == "answer_error"
    assert result.steps[-1]["raw_output_truncated"]
    assert len(result.steps[-1]["raw_output"].encode("utf-8")) <= 150


def test_oversized_sentence_fails_before_any_model_call():
    client = Client([])
    result = S2GAdapter(client, Retriever(((replace(E, text="x" * 2001),),))).run(Q)
    assert result.status == "failed" and result.stop_reason == "retrieve_error"
    assert client.requests == []


def test_unreported_retrieval_cost_keeps_total_usage_unknown():
    class UnreportedRetriever(Retriever):
        def retrieve(self, query, *, top_k):
            return CallResult((E,), usage=Usage(), transport_source="mock")

    client = Client([{"sentence_ids": ["s0"]}, SUFFICIENT, ANSWER])
    result = S2GAdapter(client, UnreportedRetriever()).run(Q)
    assert result.status == "completed" and result.usage.api_requests is None
