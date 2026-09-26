"""Offline contracts for an adaptation; these tests assert no QA effectiveness."""

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments.api_client import APIRequestError, ChatResponse
from growrag.experiments.dualrag_adapter import (
    ADAPTATION_NOTES,
    ANSWER_PROMPT,
    BASELINE_ID,
    ENTITIES_PROMPT,
    GUARD_VERSION,
    GUARDED_BASELINE_ID,
    GUARDED_PROMPT_VERSIONS,
    PROMPT_VERSIONS,
    REASON_PROMPT,
    SUMMARY_PROMPT,
    DualRAGAdapter,
    DualRAGLimits,
)
from growrag.experiments.protocol import (
    CallResult,
    Evidence,
    ExecutionKind,
    GoldRecord,
    RuntimeQuestion,
    Usage,
)

Q = RuntimeQuestion("synthetic", "When was Northbridge founded?", "synthetic-only")
E = Evidence("s0", "Northbridge", 0, "Northbridge was founded in 1901.")
E2 = Evidence("s1", "Northbridge", 1, "Northbridge is a university.")
REASON = {"information_need": "Find the founding date of Northbridge.", "need_retrieve": True}
STOP = {"information_need": "The founding date is available.", "need_retrieve": False}
ENTITIES = {"entities": [{"entity": "Northbridge", "queries": ["Northbridge founding date"]}]}
SUMMARY = {"summary": "Northbridge was founded in 1901.", "evidence_ids": ["s0"]}
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
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        content = value if isinstance(value, str) else json.dumps(value)
        return ChatResponse(
            content,
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
        self.responses = list(responses)
        self.requests = []

    def retrieve(self, query, *, top_k):
        self.requests.append((query, top_k))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return CallResult(value, usage=Usage(0, 0, 0), transport_source="local_compute")


def successful_client():
    return Client([REASON, ENTITIES, SUMMARY, STOP, ANSWER])


def test_construction_is_offline_and_budget_envelope_is_bounded():
    client, retriever = successful_client(), Retriever()
    adapter = DualRAGAdapter(client, retriever)
    assert client.requests == retriever.requests == []
    bounds = adapter.budget_envelope()
    assert bounds["max_model_calls"] == 9
    assert bounds["max_retrieval_calls"] == 8
    assert bounds["reserved_output_token_bound"] == 9 * 512
    assert bounds["reserved_input_token_bound"] == 9 * (24000 + 1024)
    assert "No fine-tuning" in " ".join(ADAPTATION_NOTES)
    assert "not the paper" in " ".join(ADAPTATION_NOTES)


def test_guarded_v2_is_explicit_and_has_a_separate_audit_identity():
    legacy = DualRAGAdapter(successful_client(), Retriever())
    guarded = DualRAGAdapter(successful_client(), Retriever(), guarded_v2=True)

    assert legacy.baseline_id == BASELINE_ID
    assert legacy.guard_version is None
    assert legacy.prompt_manifest == {
        "baseline_id": BASELINE_ID,
        "guarded_v2": False,
        "guard_version": None,
        "prompt_versions": PROMPT_VERSIONS,
    }
    assert guarded.baseline_id == GUARDED_BASELINE_ID
    assert guarded.guard_version == GUARD_VERSION
    assert guarded.prompt_manifest["prompt_versions"] == GUARDED_PROMPT_VERSIONS
    assert guarded.prompt_manifest["guarded_v2"] is True
    with pytest.raises(TypeError, match="bool"):
        DualRAGAdapter(successful_client(), Retriever(), guarded_v2=1)


def test_legacy_prompt_bytes_remain_frozen_for_old_request_replay():
    expected = {
        "reason": "ac506cbee6b9794760577c2a706e589f87b14acc8d6e5c2397db6bffa4cb711c",
        "entities": "2cc7fcc35bba554b5b3d31d6d84008d426ac2e21de0227b3cd9ce3588d61da86",
        "summarize": "5029d0d6cbc66c0fdc1c61ae2ddb00ad91611834faaaa016c5160547b3a2e858",
        "answer": "4630c05086f395e6e9279c7835db2b1978b3c3d3f486556a50f3365585cb28d4",
    }
    prompts = {
        "reason": REASON_PROMPT,
        "entities": ENTITIES_PROMPT,
        "summarize": SUMMARY_PROMPT,
        "answer": ANSWER_PROMPT,
    }
    assert {name: hashlib.sha256(text.encode()).hexdigest() for name, text in prompts.items()} == (
        expected
    )


def test_complete_question_keeps_inputs_audit_and_source_backed_summaries():
    client, retriever = successful_client(), Retriever()
    result = DualRAGAdapter(client, retriever).run(Q)
    assert result.status == "completed"
    assert result.stop_reason == "reasoner_stop_signal"
    assert result.baseline_id == BASELINE_ID
    assert result.answer.text == "1901"
    assert result.answer.cited_evidence_ids == ("s0",)
    assert result.observed_evidence == (E,)
    assert result.knowledge[0].summary == SUMMARY["summary"]
    assert result.knowledge[0].evidence_ids == ("s0",)
    assert result.completed_rounds == 1
    assert [e.operation for e in result.events] == [
        "reason",
        "entities",
        "retrieve",
        "summarize",
        "reason",
        "answer",
    ]
    assert result.usage.api_requests == 0
    assert result.usage.input_tokens is None  # Synthetic token numbers are not billed.
    assert all(
        e.audit_path == "offline-only.json" for e in result.events if e.operation != "retrieve"
    )
    assert [request[1]["prompt_version"] for request in client.requests] == [
        PROMPT_VERSIONS[name] for name in ("reason", "entities", "summarize", "reason", "answer")
    ]
    answer_input = json.loads(client.requests[-1][0][1]["content"])
    assert answer_input["question"] == Q.text
    assert answer_input["knowledge"][0]["evidence_ids"] == ["s0"]
    assert "gold" not in json.dumps(result.steps).lower()
    json.dumps(asdict(result))


@pytest.mark.parametrize("field", list(DualRAGLimits.__dataclass_fields__))
@pytest.mark.parametrize("value", [0, -1, True, 1.5, 100000])
def test_limits_reject_invalid_or_unbounded_values(field, value):
    with pytest.raises(ValueError):
        DualRAGLimits(**{field: value})


def test_limits_are_immutable():
    with pytest.raises(FrozenInstanceError):
        DualRAGLimits().max_rounds = 100


def test_gold_and_overlong_queries_are_rejected_before_calls():
    client = successful_client()
    adapter = DualRAGAdapter(client, Retriever())
    with pytest.raises(TypeError):
        adapter.run(GoldRecord("id", ("1901",)))
    with pytest.raises(ValueError):
        adapter.run(RuntimeQuestion("long", "x" * 2001))
    assert client.requests == []


def test_one_adapter_does_not_carry_knowledge_between_questions():
    client = Client([REASON, ENTITIES, SUMMARY, STOP, ANSWER, STOP, EMPTY])
    adapter = DualRAGAdapter(client, Retriever())
    first = adapter.run(Q)
    second = adapter.run(RuntimeQuestion("other", "A different question?"))
    assert first.knowledge and not second.knowledge
    assert second.observed_evidence == ()
    assert json.loads(client.requests[5][0][1]["content"])["knowledge"] == []


def test_zero_retrieval_stop_can_only_abstain_without_evidence():
    client, retriever = Client([STOP, EMPTY]), Retriever()
    result = DualRAGAdapter(client, retriever).run(Q)
    assert result.status == "completed"
    assert result.answer.text == ""
    assert result.completed_rounds == 0
    assert retriever.requests == []
    bad = DualRAGAdapter(Client([STOP, ANSWER]), Retriever()).run(Q)
    assert bad.status == "failed" and bad.stop_reason == "answer_error"


def test_guarded_empty_knowledge_stop_is_overridden_without_forwarding_model_claims():
    unsupported_stop = {
        "information_need": "Northbridge was definitely founded in 1901.",
        "need_retrieve": False,
    }
    client = Client([unsupported_stop, ENTITIES, EMPTY])
    retriever = Retriever([()])

    result = DualRAGAdapter(client, retriever, guarded_v2=True).run(Q)

    assert result.status == "completed" and result.stop_reason == "no_new_evidence"
    assert result.baseline_id == GUARDED_BASELINE_ID
    assert retriever.requests == [("Northbridge founding date", 4)]
    entity_input = json.loads(client.requests[1][0][1]["content"])
    assert entity_input["information_need"] == (
        "Retrieve evidence needed to answer the original question."
    )
    assert "1901" not in entity_input["information_need"]
    guards = [step for step in result.steps if step["operation"] == "guard"]
    assert [step["guard"] for step in guards] == ["empty_knowledge_stop_overridden"]
    assert "gold" not in json.dumps(asdict(result)).casefold()


def test_guarded_empty_knowledge_override_remains_bounded_across_rounds():
    false_stop = {"information_need": "An unsupported answer fact.", "need_retrieve": False}
    second_entities = {
        "entities": [{"entity": "Northbridge", "queries": ["Northbridge university type"]}]
    }
    empty_summary = {"summary": "", "evidence_ids": []}
    client = Client(
        [
            false_stop,
            ENTITIES,
            empty_summary,
            false_stop,
            second_entities,
            empty_summary,
            EMPTY,
        ]
    )
    retriever = Retriever([(E,), (E2,)])

    result = DualRAGAdapter(
        client,
        retriever,
        guarded_v2=True,
        limits=DualRAGLimits(max_rounds=2),
    ).run(Q)

    assert result.status == "completed" and result.stop_reason == "round_limit"
    assert result.completed_rounds == 2
    assert len(retriever.requests) == 2
    guards = [
        step for step in result.steps if step.get("guard") == "empty_knowledge_stop_overridden"
    ]
    assert len(guards) == 2
    second_reason_input = json.loads(client.requests[3][0][1]["content"])
    assert second_reason_input["previous_observations"] == [
        "Retrieve evidence needed to answer the original question."
    ]


@pytest.mark.parametrize(
    "bad",
    [
        {"information_need": "missing", "need_retrieve": "true"},
        {"information_need": "missing", "need_retrieve": 1},
        {"information_need": "", "need_retrieve": True},
        {**REASON, "answer": "extra"},
        '{"information_need":"first","information_need":"second","need_retrieve":true}',
        "```json\n{}\n```",
        "not JSON",
        [],
    ],
)
def test_invalid_reasoner_output_stops_without_retry(bad):
    client, retriever = Client([bad]), Retriever()
    result = DualRAGAdapter(client, retriever).run(Q)
    assert result.status == "failed" and result.stop_reason == "reason_error"
    assert len(client.requests) == 1 and retriever.requests == []
    assert result.events[-1].audit_path == "offline-only.json"


@pytest.mark.parametrize(
    "bad",
    [
        {"entities": []},
        {"entities": [{"entity": "A", "queries": []}]},
        {"entities": [{"entity": "A", "queries": ["q", "Q"]}]},
        {"entities": [{"entity": "A", "queries": ["q", "r", "s"]}]},
        {"entities": [{"entity": "A", "queries": ["q"]}, {"entity": "a", "queries": ["r"]}]},
        {"entities": [{"entity": "A", "queries": ["q"], "answer": "bad"}]},
        {"entities": [{"entity": "A", "queries": ["x" * 2001]}]},
    ],
)
def test_invalid_entities_never_trigger_retrieval(bad):
    client, retriever = Client([REASON, bad]), Retriever()
    result = DualRAGAdapter(client, retriever).run(Q)
    assert result.status == "failed" and result.stop_reason == "entities_error"
    assert len(client.requests) == 2 and retriever.requests == []


@pytest.mark.parametrize(
    "bad",
    [
        {"summary": "unsupported", "evidence_ids": ["unseen"]},
        {"summary": "unsupported", "evidence_ids": []},
        {"summary": "", "evidence_ids": ["s0"]},
        {"summary": "dup", "evidence_ids": ["s0", "s0"]},
        {"summary": "x" * 1201, "evidence_ids": ["s0"]},
        {"summary": "text", "evidence_ids": ["s0"], "confidence": 1},
    ],
)
def test_summary_validation_is_fail_closed_not_a_fact_checker(bad):
    client = Client([REASON, ENTITIES, bad])
    result = DualRAGAdapter(client, Retriever()).run(Q)
    assert result.status == "failed" and result.stop_reason == "summarize_error"
    assert not result.knowledge and result.observed_evidence == (E,)
    assert len(client.requests) == 3


def test_summary_may_abstain_and_raw_evidence_is_not_fabricated_as_knowledge():
    client = Client([REASON, ENTITIES, {"summary": "", "evidence_ids": []}, EMPTY])
    result = DualRAGAdapter(client, Retriever(), limits=DualRAGLimits(max_rounds=1)).run(Q)
    assert result.status == "completed" and not result.knowledge
    assert result.observed_evidence == (E,)
    assert result.answer.text == ""


def test_query_lists_are_interleaved_before_entity_context_cap():
    more = [Evidence(f"e{i}", "Topic", i, f"sentence {i}") for i in range(4)]
    client = Client(
        [
            REASON,
            {"entities": [{"entity": "Topic", "queries": ["alpha", "beta"]}]},
            {"summary": "two facts", "evidence_ids": ["e0", "e2"]},
            {"answer": "two", "cited_evidence_ids": ["e0", "e2"]},
        ]
    )
    retriever = Retriever([tuple(more[:2]), tuple(more[2:])])
    limits = DualRAGLimits(max_rounds=1, max_evidence_per_entity=2)
    result = DualRAGAdapter(client, retriever, limits=limits).run(Q)
    summary_input = json.loads(client.requests[2][0][1]["content"])
    assert [e["evidence_id"] for e in summary_input["evidence"]] == ["e0", "e2"]
    assert result.observed_evidence == tuple(more)
    assert result.status == "completed"


def test_guarded_placeholder_is_removed_when_same_entity_has_a_legal_query():
    entities = {
        "entities": [
            {
                "entity": "award winner",
                "queries": ["[Winner Name] founding date", "Northbridge founding date"],
            }
        ]
    }
    client = Client([REASON, entities, SUMMARY, ANSWER])
    retriever = Retriever()

    result = DualRAGAdapter(
        client,
        retriever,
        guarded_v2=True,
        limits=DualRAGLimits(max_rounds=1),
    ).run(Q)

    assert result.status == "completed"
    assert retriever.requests == [("Northbridge founding date", 4)]
    summary_input = json.loads(client.requests[2][0][1]["content"])
    assert summary_input["queries"] == ["Northbridge founding date"]
    blocked = [step for step in result.steps if step.get("guard") == "placeholder_query_blocked"]
    assert [step["query"] for step in blocked] == ["[Winner Name] founding date"]
    assert not any(step.get("guard") == "all_placeholder_queries_fallback" for step in result.steps)


@pytest.mark.parametrize(
    "placeholder_query",
    [
        "[Winner Name] biography",
        "{entity} biography",
        "<person> biography",
        "$entity biography",
    ],
)
def test_guarded_all_placeholder_queries_fall_back_to_original_question(
    placeholder_query: str,
):
    entities = {"entities": [{"entity": "unknown slot", "queries": [placeholder_query]}]}
    client = Client([REASON, entities, SUMMARY, ANSWER])
    retriever = Retriever()

    result = DualRAGAdapter(
        client,
        retriever,
        guarded_v2=True,
        limits=DualRAGLimits(max_rounds=1),
    ).run(Q)

    assert result.status == "completed"
    assert retriever.requests == [(Q.text, 4)]
    assert any(step.get("guard") == "placeholder_query_blocked" for step in result.steps)
    assert any(step.get("guard") == "all_placeholder_queries_fallback" for step in result.steps)
    assert placeholder_query not in {query for query, _ in retriever.requests}


def test_guarded_real_bracketed_title_is_not_treated_as_a_placeholder():
    entities = {"entities": [{"entity": "[REC]", "queries": ["[REC] release date"]}]}
    client = Client([REASON, entities, SUMMARY, ANSWER])
    retriever = Retriever()

    result = DualRAGAdapter(
        client,
        retriever,
        guarded_v2=True,
        limits=DualRAGLimits(max_rounds=1),
    ).run(Q)

    assert result.status == "completed"
    assert retriever.requests == [("[REC] release date", 4)]
    assert not any(step["operation"] == "guard" for step in result.steps)


def test_guarded_original_question_fallback_obeys_repeated_query_bound():
    placeholder_entities = {
        "entities": [{"entity": "unknown", "queries": ["{entity} founding date"]}]
    }
    empty_summary = {"summary": "", "evidence_ids": []}
    client = Client(
        [
            REASON,
            placeholder_entities,
            empty_summary,
            REASON,
            placeholder_entities,
            EMPTY,
        ]
    )
    retriever = Retriever([(E,)])

    result = DualRAGAdapter(
        client,
        retriever,
        guarded_v2=True,
        limits=DualRAGLimits(max_rounds=5),
    ).run(Q)

    assert result.status == "completed" and result.stop_reason == "repeated_queries"
    assert result.completed_rounds == 2
    assert retriever.requests == [(Q.text, 4)]
    assert any(step["operation"] == "skip_repeated_query" for step in result.steps)


def test_repeated_query_stops_without_retrieval_or_summary_retry():
    client = Client([REASON, ENTITIES, SUMMARY, REASON, ENTITIES, ANSWER])
    retriever = Retriever()
    result = DualRAGAdapter(client, retriever).run(Q)
    assert result.status == "completed" and result.stop_reason == "repeated_queries"
    assert len(retriever.requests) == 1
    assert any(step["operation"] == "skip_repeated_query" for step in result.steps)


def test_new_query_with_no_new_evidence_stops_after_second_round():
    second = {"entities": [{"entity": "Northbridge", "queries": ["Northbridge date opened"]}]}
    client = Client([REASON, ENTITIES, SUMMARY, REASON, second, SUMMARY, ANSWER])
    retriever = Retriever([(E,), (E,)])
    result = DualRAGAdapter(client, retriever, limits=DualRAGLimits(max_rounds=5)).run(Q)
    assert result.status == "completed" and result.stop_reason == "no_new_evidence"
    assert result.completed_rounds == 2 and len(retriever.requests) == 2


def test_two_round_cap_is_not_overridden_by_always_retrieve_output():
    second = {"entities": [{"entity": "Northbridge", "queries": ["Northbridge university"]}]}
    client = Client(
        [
            REASON,
            ENTITIES,
            SUMMARY,
            REASON,
            second,
            {"summary": "It is a university.", "evidence_ids": ["s1"]},
            ANSWER,
        ]
    )
    result = DualRAGAdapter(client, Retriever([(E,), (E2,)])).run(Q)
    assert result.status == "completed" and result.stop_reason == "round_limit"
    assert result.completed_rounds == 2 and len(client.requests) == 7


@pytest.mark.parametrize(
    "bad", [[], (E, E), tuple(Evidence(str(i), "t", i, "x") for i in range(5))]
)
def test_invalid_retrieval_contract_is_logged_and_stops(bad):
    client = Client([REASON, ENTITIES])
    result = DualRAGAdapter(client, Retriever([bad])).run(Q)
    assert result.status == "failed" and result.stop_reason == "retrieve_error"
    assert result.events[-1].status == "error"
    assert len(client.requests) == 2


def test_changed_evidence_id_is_rejected_even_across_rounds():
    second = {"entities": [{"entity": "Northbridge", "queries": ["different query"]}]}
    client = Client([REASON, ENTITIES, SUMMARY, REASON, second])
    retriever = Retriever([(E,), (Evidence("s0", "Northbridge", 0, "Different content"),)])
    result = DualRAGAdapter(client, retriever).run(Q)
    assert result.status == "failed" and result.stop_reason == "retrieve_error"
    assert result.observed_evidence == (E,)


def test_empty_retrieval_does_not_trigger_summary_or_fake_citations():
    client = Client([REASON, ENTITIES, EMPTY])
    result = DualRAGAdapter(client, Retriever([()])).run(Q)
    assert result.status == "completed" and result.stop_reason == "no_new_evidence"
    assert not result.knowledge and not result.observed_evidence
    assert len(client.requests) == 3


def test_transport_failure_preserves_audit_and_never_falls_back():
    error = APIRequestError(
        "private provider error", transport_source="mock", audit_path=Path("failed-offline.json")
    )
    client = Client([error])
    result = DualRAGAdapter(client, Retriever()).run(Q)
    assert result.status == "failed" and len(client.requests) == 1
    assert result.events[-1].audit_path == "failed-offline.json"
    assert "private provider error" not in json.dumps(asdict(result))


def test_unexpected_client_failure_is_unknown_usage_not_free_or_retried():
    client = Client([RuntimeError("sensitive unexpected transport detail")])
    result = DualRAGAdapter(client, Retriever()).run(Q)
    assert result.status == "failed" and len(client.requests) == 1
    assert result.usage.api_requests is None
    assert result.events[0].usage == Usage()
    assert "sensitive unexpected" not in json.dumps(asdict(result))


def test_prompt_size_is_checked_before_calling_the_client():
    client = successful_client()
    result = DualRAGAdapter(client, Retriever(), limits=DualRAGLimits(max_prompt_bytes=1)).run(Q)
    assert result.status == "failed" and client.requests == []
    assert result.events[0].usage == Usage(0, 0, 0)


def test_json_schema_and_mixed_provenance_are_rejected_before_calls():
    client = successful_client()
    client.config = SimpleNamespace(**{**vars(client.config), "json_schema_mode": True})
    with pytest.raises(ValueError, match="JSON-schema"):
        DualRAGAdapter(client, Retriever())
    client = successful_client()
    retriever = Retriever()
    retriever.execution_kind = ExecutionKind.REAL
    with pytest.raises(ValueError, match="mix"):
        DualRAGAdapter(client, retriever)
    client.transport_source = "live_api"
    with pytest.raises(ValueError, match="budget"):
        DualRAGAdapter(client, retriever)
