"""Synthetic response transport only: these tests never read keys or use HTTP."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from growrag import controller, paired_execution
from growrag.controller import APIEvidenceAssessor, APIGapQueryGenerator, APIRoutingPolicy
from growrag.experiments.api_client import ChatConfig, ChatResponse
from growrag.experiments.llm_adapters import READER_PROMPT, READER_PROMPT_VERSION, APIReader
from growrag.experiments.protocol import Answer, BackendCallError, Evidence, RuntimeQuestion, Usage
from growrag.outer_loop import Feedback, LoopState, RagReply, RoundRecord
from growrag.paired_execution import PairedExecutionCache
from growrag.query_actions import RewriteDecision

Q = RuntimeQuestion("synthetic-q", "Which town contains Fiction School?", "synthetic")
E1 = Evidence("e1", "Fiction School", 0, "Fiction School is in Imaginary Town.")
E2 = Evidence("e2", "Imaginary Town", 0, "Imaginary Town contains Fiction School.")


class SyntheticClient:
    """Fake complete() for testing both mock and live-labelled provenance paths."""

    transport_source = "mock"

    def __init__(self):
        self.config = ChatConfig(
            "https://synthetic.invalid/v1",
            "fixed-model",
            "THIS_VARIABLE_IS_NEVER_READ",
            100,
            temperature=0,
            json_object_mode=True,
        )
        self.requests = []
        self.reader_value = {"answer": "Imaginary Town", "cited_evidence_ids": ["e1"]}
        self.assessment_value = {
            "requirements": [
                {"description": "School location", "status": "supported", "evidence_ids": ["e1"]}
            ],
            "sufficient": True,
            "useful_gain": None,
            "gap": "",
            "next_intent": "",
            "reason": "The supplied text states the school's location.",
        }
        self.reported_model = None
        self.input_tokens = 20

    def complete(self, messages, *, trace_id, prompt_version):
        self.requests.append({"messages": deepcopy(messages), "prompt_version": prompt_version})
        value = (
            self.assessment_value
            if prompt_version == controller.ASSESS_PROMPT_VERSION
            else self.reader_value
        )
        return ChatResponse(
            value if isinstance(value, str) else json.dumps(value),
            self.config.model,
            self.reported_model or self.config.model,
            f"response-{len(self.requests)}",
            f"request-{len(self.requests)}",
            self.input_tokens,
            5,
            0,
            Path(f"synthetic-audit-{len(self.requests)}.json"),
            self.transport_source,
        )


def reader(cache, client, *, version=READER_PROMPT_VERSION, text=READER_PROMPT):
    return cache.reader(APIReader(client), prompt_version=version, prompt_text=text)


def reply(evidence=(E1,), answer="Imaginary Town"):
    return RagReply(Answer(answer, ("e1",)), evidence)


def test_reader_identical_inputs_share_value_and_keep_actual_provenance():
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    client.transport_source = "live_api"  # Still entirely synthetic; no network client exists.
    original = reader(cache, client).answer(Q, (E1,))
    replayed = reader(cache, client).answer(Q, (E1,))
    assert len(client.requests) == 1
    assert original.value is replayed.value
    assert original.usage == Usage(20, 5, 1)
    assert replayed.usage == Usage(0, 0, 0)
    assert replayed.transport_source == "local_compute"
    assert replayed.provider == "local-paired-replay"
    assert replayed.audit_path == original.audit_path
    assert cache.records[-1]["source_usage"] == {
        "input_tokens": 20,
        "output_tokens": 5,
        "api_requests": 1,
    }
    assert cache.records[-1]["source_record_index"] == 0
    assert cache.records[-1]["cache_hit"] is True
    assert cache.records[-1]["source_transport_source"] == "live_api"
    assert "THIS_VARIABLE_IS_NEVER_READ" not in json.dumps(cache.records)


def test_reader_order_content_and_prompt_text_are_part_of_exact_identity():
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    wrapped = reader(cache, client)
    wrapped.answer(Q, (E1, E2))
    wrapped.answer(Q, (E2, E1))
    wrapped.answer(Q, (replace(E1, text=E1.text + " "), E2))
    reader(cache, client, text=READER_PROMPT + " ").answer(Q, (E1, E2))
    reader(cache, client, version="other-version").answer(Q, (E1, E2))
    assert len(client.requests) == 5
    assert not any(record["cache_hit"] for record in cache.records)


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other-model"},
        {"base_url": "https://other.invalid/v1"},
        {"temperature": 0.5},
        {"json_object_mode": False},
        {"enable_thinking": True},
        {"max_output_tokens": 1000},
        {"output_limit_parameter": "max_tokens"},
    ],
)
def test_configuration_change_never_replays(change):
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    wrapped = reader(cache, client)
    wrapped.answer(Q, (E1,))
    client.config = replace(client.config, **change)
    wrapped.answer(Q, (E1,))
    assert len(client.requests) == 2
    assert not cache.records[-1]["cache_hit"]


@pytest.mark.parametrize(
    "other",
    [replace(Q, question_id="other"), replace(Q, text=Q.text + " "), replace(Q, dataset="other")],
)
def test_no_cross_question_sharing(other):
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    with pytest.raises(ValueError, match="different questions"):
        reader(cache, client).answer(other, (E1,))
    with pytest.raises(ValueError, match="different questions"):
        cache.assessor(APIEvidenceAssessor(client))(LoopState(other), reply())
    assert not client.requests


def test_invalid_reader_responses_never_populate_cache():
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    client.reader_value = "not JSON"
    wrapped = reader(cache, client)
    for _ in range(2):
        with pytest.raises(BackendCallError):
            wrapped.answer(Q, (E1,))
    assert len(client.requests) == 2
    assert cache.records == []


@pytest.mark.parametrize("unknown", ["usage", "model"])
def test_unknown_paid_source_is_not_hidden_by_replay(unknown):
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    client.transport_source = "live_api"
    if unknown == "usage":
        client.input_tokens = None
    else:
        client.reported_model = "different-returned-model"
    wrapped = reader(cache, client)
    wrapped.answer(Q, (E1,))
    wrapped.answer(Q, (E1,))
    assert len(client.requests) == 2
    assert all(not event["cacheable"] and not event["cache_hit"] for event in cache.records)


def test_assessor_replay_preserves_latest_normalization_and_separate_records():
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    client.transport_source = "live_api"
    del client.assessment_value["reason"]
    first = cache.assessor(APIEvidenceAssessor(client))
    second = cache.assessor(APIEvidenceAssessor(client))
    original = first(LoopState(Q), reply())
    replayed = second(LoopState(Q), reply())
    assert len(client.requests) == 1
    assert second.latest == first.latest
    assert second.latest_question == Q
    assert replayed.value == original.value
    assert replayed.usage == Usage(0, 0, 0)
    assert first.records[0]["status"] == "ok"
    record = second.records[0]
    assert record["status"] == "cache_replay"
    assert record["normalization"]["reason_origin"] == "local_placeholder"
    assert record["metadata"]["transport_source"] == "local_compute"
    assert record["replay"]["source_metadata"]["usage"]["api_requests"] == 1
    record["normalization"]["reason_origin"] = "mutation-in-a-consumer"
    assert first.records[0]["normalization"]["reason_origin"] == "local_placeholder"
    # Existing strict controller/generator guards accept the shared assessor.
    APIRoutingPolicy(client, (), second)
    APIGapQueryGenerator(client, second)
    json.dumps(cache.records)
    json.dumps(second.records)


def test_assessor_identity_includes_answer_citations_context_and_previous_state():
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    judge = cache.assessor(APIEvidenceAssessor(client))
    state = LoopState(Q)
    judge(state, reply())
    judge(state, reply(answer="Fiction Town"))
    judge(state, reply((E1, E2)))
    judge(state, reply((E2, E1)))
    judge(state, RagReply(Answer("Imaginary Town", ("e1", "e2")), (E1, E2)))
    judge(replace(state, observed_evidence=(E1,)), reply())
    judge(replace(state, observed_evidence=(E2, E1)), reply())
    judge(replace(state, observed_evidence=(E1, E2)), reply())
    prior = RoundRecord(RewriteDecision(), Q.text, reply(), Feedback(False))
    judge(replace(state, rounds=(prior,)), reply())
    assert len(client.requests) == 9
    assert len({record["cache_key"] for record in cache.records}) == 9


def test_assessor_parser_or_prompt_change_invalidates_cache(monkeypatch):
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    judge = cache.assessor(APIEvidenceAssessor(client))
    judge(LoopState(Q), reply())
    monkeypatch.setattr(controller, "ASSESS_PARSER_VERSION", "different-parser")
    judge(LoopState(Q), reply())
    monkeypatch.setattr(controller, "ASSESS_PROMPT", controller.ASSESS_PROMPT + " ")
    judge(LoopState(Q), reply())
    assert len(client.requests) == 3


def test_assessor_failure_clears_latest_and_is_not_cached():
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    judge = cache.assessor(APIEvidenceAssessor(client))
    judge(LoopState(Q), reply())
    client.assessment_value = "malformed"
    changed = replace(LoopState(Q), observed_evidence=(E1,))
    for _ in range(2):
        with pytest.raises(BackendCallError):
            judge(changed, reply())
        assert judge.latest is None
    assert len(client.requests) == 3
    assert len(cache.records) == 1


def test_seed_is_question_bound_but_does_not_create_a_fake_cache_entry():
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    judge = cache.assessor(APIEvidenceAssessor(client))
    source = APIEvidenceAssessor(client)
    source(LoopState(Q), reply())
    judge.seed(Q, source.latest)
    assert judge.latest == source.latest
    assert judge.records == [] and cache.records == []
    with pytest.raises(ValueError, match="different questions"):
        judge.seed(replace(Q, question_id="other"), source.latest)
    judge(LoopState(Q), reply())
    assert len(client.requests) == 2  # Seed is not a paid-response identity.


def test_duplicates_rejected_before_reader_call_and_cache_is_not_cross_instance():
    client = SyntheticClient()
    first = PairedExecutionCache(Q)
    with pytest.raises(ValueError, match="duplicate"):
        reader(first, client).answer(Q, (E1, E1))
    reader(first, client).answer(Q, (E1,))
    reader(PairedExecutionCache(Q), client).answer(Q, (E1,))
    assert len(client.requests) == 2


def test_schema_mode_and_exact_schema_hash_are_cache_identity(monkeypatch):
    cache, client = PairedExecutionCache(Q), SyntheticClient()
    judge = cache.assessor(APIEvidenceAssessor(client))
    judge(LoopState(Q), reply())
    client.config = replace(client.config, json_object_mode=False, json_schema_mode=True)
    judge(LoopState(Q), reply())
    assert len(client.requests) == 2
    judge(LoopState(Q), reply())
    assert len(client.requests) == 2 and cache.records[-1]["cache_hit"]
    monkeypatch.setattr(paired_execution, "schema_fingerprint", lambda _: "changed-reviewed-schema")
    judge(LoopState(Q), reply())
    assert len(client.requests) == 3 and not cache.records[-1]["cache_hit"]
