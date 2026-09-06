from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments.api_client import APIRequestError, ChatResponse
from growrag.experiments.budget import BudgetedChatClient, PriceLimits


class Delegate:
    transport_source = "live_api"

    def __init__(self, *, fail=False, model="snapshot", usage=20):
        self.config = SimpleNamespace(model="snapshot", max_output_tokens=100)
        self.attempts, self.fail, self.model, self.usage = 0, fail, model, usage

    def complete(self, *args, **kwargs):
        self.attempts += 1
        if self.fail:
            raise APIRequestError("failed", api_requests=1)
        return ChatResponse(
            "{}",
            "snapshot",
            self.model,
            "id",
            "request",
            self.usage,
            5,
            0.1,
            Path("TEST_ONLY_NOT_REAL.json"),
            "live_api",
        )


def invoke(client):
    return client.complete(
        [{"role": "user", "content": "hello"}], trace_id="test", prompt_version="v"
    )


def test_usage_and_all_call_budget_report():
    client = BudgetedChatClient(Delegate(), PriceLimits())
    invoke(client)
    report = client.report()
    assert report["input_tokens"] == 20
    assert report["output_tokens"] == 5
    assert report["estimated_actual_cny"] == pytest.approx(0.000008)
    assert report["reserved_cny"] > report["estimated_actual_cny"]


@pytest.mark.parametrize(
    "delegate", [Delegate(fail=True), Delegate(model="different"), Delegate(usage=None)]
)
def test_bad_or_unmetered_response_blocks_all_further_calls(delegate):
    client = BudgetedChatClient(delegate, PriceLimits())
    with pytest.raises(APIRequestError):
        invoke(client)
    with pytest.raises(APIRequestError, match="blocked"):
        invoke(client)
    assert delegate.attempts == 1


def test_insufficient_budget_blocks_before_request():
    delegate = Delegate()
    client = BudgetedChatClient(delegate, PriceLimits(budget_cny=0.00000001))
    with pytest.raises(APIRequestError, match="budget"):
        invoke(client)
    assert delegate.attempts == 0


def test_prompt_size_blocks_before_request():
    delegate = Delegate()
    client = BudgetedChatClient(delegate, PriceLimits(max_prompt_bytes=1))
    with pytest.raises(APIRequestError, match="size"):
        invoke(client)
    assert delegate.attempts == 0
