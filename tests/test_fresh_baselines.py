"""Offline contract tests, not evidence that any FRESH method improves quality."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments.api_client import APIRequestError, ChatResponse
from growrag.experiments.fresh_baselines import (
    BASELINE_SPECS,
    MAX_PASSAGE_CHARS,
    QUERY2DOC_PROMPT,
    RRR_PROMPT,
    APIFreshBaselineGenerator,
    baseline_generator,
)
from growrag.experiments.protocol import (
    Action,
    Answer,
    BackendCallError,
    CallResult,
    Evidence,
    ExecutionKind,
    MemoryView,
    RuntimeQuestion,
    Usage,
)
from growrag.outer_loop import RetrieverReaderBackend, run_outer_loop
from growrag.query_actions import APISingleQueryGenerator, RewriteDecision
from growrag.query_operators import RewriteForm

Q = RuntimeQuestion("fixture-target", "When was Northbridge University founded?")
E = Evidence("fixture-evidence", "Northbridge", 0, "Northbridge was founded in 1901.")


class Client:
    transport_source = "mock"
    config = SimpleNamespace(base_url="https://test.invalid/v1", model="TEST")

    def __init__(self, content='{"query":"Northbridge University founding date"}'):
        self.content, self.requests = content, []

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        if isinstance(self.content, Exception):
            raise self.content
        return ChatResponse(
            self.content,
            "TEST",
            "TEST-returned",
            "fixture-response",
            "fixture-request",
            10,
            20,
            0,
            Path("test-only-not-written.json"),
            "mock",
        )


@pytest.mark.parametrize("variant", BASELINE_SPECS)
def test_construction_does_not_call_client(variant):
    client = Client()
    generator = baseline_generator(client, variant)
    assert isinstance(generator, APIFreshBaselineGenerator)
    assert generator.execution_kind is ExecutionKind.MOCK
    assert generator.comparison_protocol_id == BASELINE_SPECS[variant].prompt_version
    assert client.requests == []


def test_specs_are_frozen_and_explicit_about_adaptation_limits():
    with pytest.raises(TypeError):
        BASELINE_SPECS["NEW"] = BASELINE_SPECS["RRR_KEYWORDS"]
    with pytest.raises(FrozenInstanceError):
        BASELINE_SPECS["QUERY2DOC"].label = "HyDE reproduction"
    serialized = json.dumps(BASELINE_SPECS["QUERY2DOC"].to_dict())
    assert "not a HyDE reproduction" in serialized
    assert "distinct-term BM25" in serialized
    assert "no SFT/RL" in json.dumps(BASELINE_SPECS["RRR_KEYWORDS"].to_dict())


def test_unknown_variant_fails_before_any_call():
    client = Client()
    with pytest.raises(ValueError, match="unknown"):
        baseline_generator(client, "HyDE")
    assert client.requests == []


def test_simple_control_has_exactly_the_same_messages_and_prompt_version():
    old, new = Client(), Client()
    decision = BASELINE_SPECS["SIMPLE_PARAPHRASE"].decision()
    APISingleQueryGenerator(old, paired_prompt=True).generate(Q, decision)
    baseline_generator(new, "SIMPLE_PARAPHRASE").generate(Q, decision)
    assert old.requests[0][0] == new.requests[0][0]
    assert old.requests[0][1]["prompt_version"] == new.requests[0][1]["prompt_version"]


@pytest.mark.parametrize(
    "variant,content,prompt",
    [
        ("RRR_KEYWORDS", '{"query":"Northbridge founding date"}', RRR_PROMPT),
        ("QUERY2DOC", '{"passage":"An unverified hypothetical date is 1899."}', QUERY2DOC_PROMPT),
    ],
)
def test_new_variants_receive_only_question_text_and_keep_audit_metadata(variant, content, prompt):
    client = Client(content)
    spec = BASELINE_SPECS[variant]
    result = baseline_generator(client, variant).generate(Q, spec.decision())
    messages, options = client.requests[0]
    assert messages[0] == {"role": "system", "content": prompt}
    assert json.loads(messages[1]["content"]) == {"original_question": Q.text}
    assert options["prompt_version"] == spec.prompt_version
    assert len(client.requests) == 1
    assert result.model == "TEST-returned"
    assert result.request_id == "fixture-request"
    assert result.audit_path == "test-only-not-written.json"
    assert result.transport_source == "mock"
    assert result.usage.api_requests == 0  # Synthetic tokens must not look billed.
    assert result.usage.input_tokens is None


@pytest.mark.parametrize("variant", BASELINE_SPECS)
@pytest.mark.parametrize(
    "inputs,error",
    [
        ({"evidence": (E,)}, ValueError),
        ({"previous_queries": ("earlier query",)}, ValueError),
        ({"evidence": []}, TypeError),
        ({"previous_queries": []}, TypeError),
    ],
)
def test_no_post_retrieval_state_can_enter_pre_generator(variant, inputs, error):
    client = Client()
    with pytest.raises(error, match="PRE"):
        baseline_generator(client, variant).generate(
            Q, BASELINE_SPECS[variant].decision(), **inputs
        )
    assert client.requests == []


@pytest.mark.parametrize("variant", BASELINE_SPECS)
def test_reuse_or_base_decision_cannot_enter_no_history_generator(variant):
    spec, client = BASELINE_SPECS[variant], Client()
    reuse = RewriteDecision(
        Action.REUSE,
        spec.form,
        spec.intent,
        MemoryView("card", "source", "step", "procedure", "historical procedure"),
    )
    for decision in (reuse, RewriteDecision()):
        with pytest.raises(ValueError, match="history"):
            baseline_generator(client, variant).generate(Q, decision)
    assert client.requests == []


def test_decision_must_match_frozen_form_and_intent():
    client = Client()
    with pytest.raises(ValueError, match="frozen"):
        baseline_generator(client, "RRR_KEYWORDS").generate(
            Q, RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "modified instruction")
        )
    assert client.requests == []


@pytest.mark.parametrize("variant,key", [("RRR_KEYWORDS", "query"), ("QUERY2DOC", "passage")])
@pytest.mark.parametrize("bad_value", [None, [], {}, 7, "", " ", "???"])
def test_invalid_generated_value_is_counted_failure_without_retry(variant, key, bad_value):
    client = Client(json.dumps({key: bad_value}))
    with pytest.raises(BackendCallError, match="no retry") as failure:
        baseline_generator(client, variant).generate(Q, BASELINE_SPECS[variant].decision())
    assert len(client.requests) == 1
    assert failure.value.request_id == "fixture-request"
    assert failure.value.audit_path == "test-only-not-written.json"


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"query":"query"}\n```',
        '{"query":"query","answer":"forbidden extra"}',
        '{"query":"first","query":"second"}',
        '["query"]',
        "not json",
        '{"query":"' + "x" * 2001 + '"}',
    ],
)
def test_malformed_keyword_json_has_no_silent_repair_or_retry(content):
    client = Client(content)
    with pytest.raises(BackendCallError, match="no retry"):
        baseline_generator(client, "RRR_KEYWORDS").generate(
            Q, BASELINE_SPECS["RRR_KEYWORDS"].decision()
        )
    assert len(client.requests) == 1


def test_pseudo_passage_is_anchored_without_silent_truncation():
    client = Client('{"passage":"  This is hypothetical search text.  "}')
    result = baseline_generator(client, "QUERY2DOC").generate(
        Q, BASELINE_SPECS["QUERY2DOC"].decision()
    )
    assert result.value == Q.text + "\nThis is hypothetical search text."
    assert result.value.count(Q.text) == 1
    assert Q.text == "When was Northbridge University founded?"


def test_overlong_passage_and_combined_query_fail_without_truncation():
    cases = [
        (Q, "p" * (MAX_PASSAGE_CHARS + 1)),
        (RuntimeQuestion("long", "q" * 1500), "p" * 600),
    ]
    for question, passage in cases:
        client = Client(json.dumps({"passage": passage}))
        with pytest.raises(BackendCallError, match="no retry"):
            baseline_generator(client, "QUERY2DOC").generate(
                question, BASELINE_SPECS["QUERY2DOC"].decision()
            )
        assert len(client.requests) == 1


def test_transport_failure_is_not_replaced_by_mock_success():
    client = Client(APIRequestError("test unavailable", transport_source="mock"))
    with pytest.raises(BackendCallError, match="no mock fallback"):
        baseline_generator(client, "RRR_KEYWORDS").generate(
            Q, BASELINE_SPECS["RRR_KEYWORDS"].decision()
        )
    assert len(client.requests) == 1


def test_pseudo_passage_never_becomes_reader_evidence_or_answer_question():
    class Retriever:
        execution_kind = ExecutionKind.MOCK

        def retrieve(self, query, *, top_k):
            self.query = query
            return CallResult((E,), usage=Usage(api_requests=0), transport_source="mock")

    class Reader:
        execution_kind = ExecutionKind.MOCK

        def answer(self, question, evidence):
            self.question, self.evidence = question, evidence
            return CallResult(
                Answer("1901", (E.evidence_id,)),
                usage=Usage(api_requests=0),
                transport_source="mock",
            )

    client = Client('{"passage":"Imagined and unverified founding year 1899."}')
    retriever, reader = Retriever(), Reader()
    spec = BASELINE_SPECS["QUERY2DOC"]
    result = run_outer_loop(
        Q,
        RetrieverReaderBackend(retriever, reader),
        generator=baseline_generator(client, spec.variant_id),
        policy=lambda state: spec.decision(),
        max_rag_calls=1,
    )
    assert "1899" in retriever.query
    assert reader.question is Q
    assert reader.evidence == (E,)
    assert "1899" not in reader.evidence[0].text
    assert result.state.rounds[0].reply.answer.text == "1901"
    assert result.stop_reason == "unassessed"
    assert len(client.requests) == 1
