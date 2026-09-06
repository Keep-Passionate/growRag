"""Only mock-client prompt/shape tests, no live model quality claim."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments.api_client import ChatResponse
from growrag.experiments.protocol import Action, BackendCallError, MemoryView, RuntimeQuestion
from growrag.query_actions import APISingleQueryGenerator, RewriteDecision, RewriteForm


class Client:
    transport_source = "mock"
    config = SimpleNamespace(base_url="https://test.invalid/v1", model="TEST")

    def __init__(self, content='{"query":"alternative wording"}'):
        self.content, self.requests = content, []

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        return ChatResponse(
            self.content,
            "TEST",
            "TEST",
            "test",
            None,
            1,
            1,
            0,
            Path("not-written-by-test"),
            "mock",
        )


@pytest.mark.parametrize("form", [RewriteForm.PARAPHRASE, RewriteForm.EXPAND])
def test_pre_rag_semantic_action_has_no_gap_requirement(form):
    client = Client()
    result = APISingleQueryGenerator(client).generate(
        RuntimeQuestion("q", "Original?"), RewriteDecision(Action.FRESH, form, "align terms")
    )
    payload = json.loads(client.requests[0][0][1]["content"])
    assert payload["current_evidence"] == []
    assert payload["form"] == form.value
    assert "gap" not in payload
    assert "optional_historical_procedure" not in payload
    assert result.usage.api_requests == 0
    assert result.transport_source == "mock"


def test_reuse_receives_procedure_without_changing_original_question():
    client = Client()
    decision = RewriteDecision(
        Action.REUSE,
        RewriteForm.EXPAND,
        "align terms",
        MemoryView("m@v1", "source", "step", "procedure", "add synonym"),
    )
    APISingleQueryGenerator(client).generate(RuntimeQuestion("q", "Original?"), decision)
    payload = json.loads(client.requests[0][0][1]["content"])
    assert payload["original_question"] == "Original?"
    assert payload["optional_historical_procedure"]["memory_id"] == "m@v1"


def test_unsupported_form_fails_before_call():
    client = Client()
    with pytest.raises(ValueError, match="single-query"):
        APISingleQueryGenerator(client).generate(
            RuntimeQuestion("q", "Original?"),
            RewriteDecision(Action.FRESH, RewriteForm.DECOMPOSE, "split"),
        )
    assert not client.requests


def test_invalid_generated_json_has_no_repair_retry():
    client = Client('{"query":"q", "extra":"not allowed"}')
    with pytest.raises(BackendCallError, match="no retry"):
        APISingleQueryGenerator(client).generate(
            RuntimeQuestion("q", "Original?"),
            RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "wording"),
        )
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "action,form",
    [
        (Action.BASE, RewriteForm.PARAPHRASE),
        (Action.FRESH, RewriteForm.KEEP),
    ],
)
def test_action_and_form_cannot_contradict_each_other(action, form):
    with pytest.raises(ValueError, match="BASE requires"):
        RewriteDecision(action, form, "intent")


def test_fresh_cannot_carry_a_hidden_memory():
    with pytest.raises(ValueError, match="cannot receive"):
        RewriteDecision(
            Action.FRESH,
            RewriteForm.EXPAND,
            "intent",
            MemoryView("m", "s", "step", "procedure", "text"),
        )
