"""Prompt contract tests with a fake client: NEVER a live API experiment."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments.api_client import APIRequestError, ChatResponse
from growrag.experiments.llm_adapters import APIReader, APIRewriter
from growrag.experiments.protocol import (
    BackendCallError,
    DecisionState,
    Evidence,
    MemoryView,
    RuntimeQuestion,
)


class FakeClient:
    transport_source = "mock"
    config = SimpleNamespace(base_url="https://fake.invalid/v1", model="TEST-MODEL")

    def __init__(self, content):
        self.content = content
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        return ChatResponse(
            self.content,
            "TEST-MODEL",
            "TEST-MODEL",
            "TEST-ID",
            None,
            12,
            3,
            0.0,
            Path("unused-test-path"),
            "mock",
        )


def state():
    return DecisionState(
        RuntimeQuestion("target", "Original question?"),
        (Evidence("e1", "Title", 0, "Public evidence."),),
        "Missing relation",
        "TEST-CONTEXT",
    )


def test_fresh_and_reuse_share_prompt_and_state_but_only_reuse_gets_memory():
    client = FakeClient('{"query": "new search"}')
    rewriter = APIRewriter(client)
    rewriter.rewrite(state(), memory=None)
    memory = MemoryView("m", "source", "step", "concrete", "Historical procedure")
    result = rewriter.rewrite(state(), memory=memory)
    fresh, reuse = [request[0] for request in client.requests]
    assert fresh[0] == reuse[0]
    fresh_input, reuse_input = json.loads(fresh[1]["content"]), json.loads(reuse[1]["content"])
    assert "optional_historical_procedure" not in fresh_input
    del reuse_input["optional_historical_procedure"]
    assert fresh_input == reuse_input
    assert result.value == "new search"
    assert result.usage.input_tokens is None
    assert result.usage.api_requests == 0
    assert rewriter.execution_kind.value == "mock"


def test_invalid_schema_preserves_returned_usage_without_second_call():
    client = FakeClient('{"query": "new", "answer": "illegal extra field"}')
    with pytest.raises(BackendCallError) as error:
        APIRewriter(client).rewrite(state(), memory=None)
    assert error.value.usage.output_tokens is None
    assert len(client.requests) == 1


def test_reader_only_sees_original_question_and_own_evidence():
    client = FakeClient('{"answer": "answer", "cited_evidence_ids": ["e1"]}')
    result = APIReader(client).answer(state().question, state().evidence)
    payload = json.loads(client.requests[0][0][1]["content"])
    assert set(payload) == {"original_question", "evidence"}
    assert payload["original_question"] == "Original question?"
    assert result.value.cited_evidence_ids == ("e1",)


def test_invalid_citation_is_not_silently_removed():
    client = FakeClient('{"answer": "answer", "cited_evidence_ids": ["other-branch"]}')
    with pytest.raises(BackendCallError):
        APIReader(client).answer(state().question, state().evidence)


def test_api_failure_is_not_replaced_by_an_assistant_guess():
    class BrokenClient(FakeClient):
        def complete(self, messages, **kwargs):
            raise APIRequestError("offline test failure", transport_source="mock")

    with pytest.raises(BackendCallError, match="no mock fallback"):
        APIRewriter(BrokenClient("")).rewrite(state(), memory=None)


def test_missing_client_provenance_rejected():
    with pytest.raises(ValueError, match="provenance"):
        APIReader(SimpleNamespace())


def test_provider_failure_preserves_known_usage_and_audit_path():
    class DeclaredLiveErrorClient(FakeClient):
        transport_source = "live_api"

        def complete(self, messages, **kwargs):
            # Synthetic exception only, no request has actually been sent by this test.
            raise APIRequestError(
                "synthetic truncation",
                input_tokens=20,
                output_tokens=30,
                api_requests=1,
                audit_path=Path("TEST-audit.json"),
                request_id="TEST-ID",
            )

    with pytest.raises(BackendCallError) as error:
        APIRewriter(DeclaredLiveErrorClient("")).rewrite(state(), memory=None)
    assert error.value.usage.input_tokens == 20
    assert error.value.usage.output_tokens == 30
    assert error.value.usage.api_requests == 1
    assert error.value.request_id == "TEST-ID"
    assert error.value.audit_path == "TEST-audit.json"
    assert error.value.transport_source == "live_api"


@pytest.mark.parametrize("operation", ["rewrite", "answer"])
def test_mock_failure_cannot_report_simulated_usage_as_real_cost(operation):
    class MockUsageErrorClient(FakeClient):
        def complete(self, messages, **kwargs):
            self.requests.append((messages, kwargs))
            raise APIRequestError(
                "synthetic usage-bearing failure",
                input_tokens=20,
                output_tokens=30,
                api_requests=1,
                audit_path=Path("TEST-mock-audit.json"),
                request_id="TEST-MOCK-ID",
                transport_source="mock",
            )

    client = MockUsageErrorClient("")
    with pytest.raises(BackendCallError, match="no mock fallback") as error:
        if operation == "rewrite":
            APIRewriter(client).rewrite(state(), memory=None)
        else:
            APIReader(client).answer(state().question, state().evidence)
    assert error.value.usage.input_tokens is None
    assert error.value.usage.output_tokens is None
    assert error.value.usage.api_requests == 0
    assert error.value.transport_source == "mock"
    assert error.value.request_id == "TEST-MOCK-ID"
    assert error.value.audit_path == "TEST-mock-audit.json"
    assert len(client.requests) == 1


def test_answer_without_any_citation_is_rejected():
    client = FakeClient('{"answer": "a guess", "cited_evidence_ids": []}')
    with pytest.raises(BackendCallError):
        APIReader(client).answer(state().question, ())


def test_mock_exception_cannot_be_recorded_as_a_live_request():
    class WrongSourceClient(FakeClient):
        def complete(self, messages, **kwargs):
            raise APIRequestError("incorrectly declared", api_requests=1)

    with pytest.raises(BackendCallError, match="provenance mismatch") as error:
        APIRewriter(WrongSourceClient("")).rewrite(state(), memory=None)
    assert error.value.usage.api_requests == 0
    assert error.value.transport_source == "mock"
