"""Serial, fail-closed estimated-cost guard around one frozen API backend.

Preflight input reservation uses UTF-8 bytes plus a generous framing allowance,
not an exact tokenizer. This is an application guard under declared price/token
assumptions, not a provider billing guarantee. Unknown billing blocks more calls.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass

from .api_client import APIRequestError, ChatResponse, LiveChatClient


@dataclass(frozen=True)
class PriceLimits:
    budget_cny: float = 1.0
    input_per_million_cny: float = 0.20
    output_per_million_cny: float = 0.80
    max_prompt_bytes: int = 24000
    max_elapsed_seconds: float = 900.0

    def __post_init__(self):
        for value in (self.budget_cny, self.input_per_million_cny, self.output_per_million_cny):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("budget and declared unit prices must be finite and positive")
        if type(self.max_prompt_bytes) is not int or not 0 < self.max_prompt_bytes <= 30000:
            raise ValueError("prompt byte limit must be within the declared short-context tier")
        if not math.isfinite(self.max_elapsed_seconds) or self.max_elapsed_seconds <= 0:
            raise ValueError("elapsed-time limit must be finite and positive")


class BudgetedChatClient:
    def __init__(self, delegate: LiveChatClient, limits: PriceLimits) -> None:
        self.delegate, self.limits = delegate, limits
        self.config = delegate.config
        self.transport_source = delegate.transport_source
        self.calls: list[dict] = []
        self.reserved_cny = 0.0
        self.estimated_actual_cny = 0.0
        self.block_reason: str | None = None
        self.started_at = time.monotonic()

    @property
    def attempts(self):
        return self.delegate.attempts

    def _price(self, input_tokens, output_tokens):
        return (
            input_tokens * self.limits.input_per_million_cny
            + output_tokens * self.limits.output_per_million_cny
        ) / 1_000_000

    def complete(self, messages, *, trace_id: str, prompt_version: str) -> ChatResponse:
        if time.monotonic() - self.started_at > self.limits.max_elapsed_seconds:
            self.block_reason = "elapsed_time_limit"
        if self.block_reason:
            raise APIRequestError("budget client blocked after prior failure; no request sent")
        size = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
        if size > self.limits.max_prompt_bytes:
            self.block_reason = "prompt_size_limit"
            raise APIRequestError("prompt exceeds pilot size cap; no request sent")
        reserved_input = size + 1024
        reserve = self._price(reserved_input, self.config.max_output_tokens)
        if self.reserved_cny + reserve > self.limits.budget_cny:
            self.block_reason = "estimated_budget_limit"
            raise APIRequestError("pilot estimated budget exhausted; no request sent")
        # Never refund reservations: no optimism about missing or delayed billing.
        self.reserved_cny += reserve
        event = {"trace_id": trace_id, "prompt_version": prompt_version, "reserved_cny": reserve}
        try:
            response = self.delegate.complete(
                messages, trace_id=trace_id, prompt_version=prompt_version
            )
        except APIRequestError as error:
            self.block_reason = "transport_failure"
            event.update(
                status="failed",
                api_requests=error.api_requests,
                input_tokens=error.input_tokens,
                output_tokens=error.output_tokens,
                audit_path=str(error.audit_path) if error.audit_path else None,
            )
            if error.input_tokens is not None and error.output_tokens is not None:
                cost = self._price(error.input_tokens, error.output_tokens)
                event["estimated_actual_cny"] = cost
                self.estimated_actual_cny += cost
            self.calls.append(event)
            raise
        except Exception:
            self.block_reason = "unexpected_failure"
            event.update(status="failed", api_requests=None)
            self.calls.append(event)
            raise
        event.update(
            status="completed",
            api_requests=1,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            returned_model=response.returned_model,
            audit_path=str(response.audit_path),
        )
        self.calls.append(event)
        if response.input_tokens is None or response.output_tokens is None:
            self.block_reason = "unknown_usage"
        else:
            cost = self._price(response.input_tokens, response.output_tokens)
            event["estimated_actual_cny"] = cost
            self.estimated_actual_cny += cost
            if response.input_tokens > reserved_input or cost > reserve:
                self.block_reason = "reservation_assumption_exceeded"
        if response.returned_model != self.config.model:
            self.block_reason = "model_snapshot_mismatch"
        if self.block_reason:
            event["validation_status"] = self.block_reason
            raise APIRequestError(
                "response failed snapshot/usage budget contract; no retry",
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                api_requests=1,
                request_id=response.request_id,
                audit_path=response.audit_path,
            )
        return response

    def report(self) -> dict:
        missing = any(
            event.get("input_tokens") is None or event.get("output_tokens") is None
            for event in self.calls
        )
        return {
            "limits": asdict(self.limits),
            "api_requests": self.attempts,
            "reserved_cny": self.reserved_cny,
            "estimated_actual_cny": None if missing else self.estimated_actual_cny,
            "input_tokens": None if missing else sum(e["input_tokens"] for e in self.calls),
            "output_tokens": None if missing else sum(e["output_tokens"] for e in self.calls),
            "block_reason": self.block_reason,
            "calls": self.calls,
            "cost_scope": "all model calls including shared gap and source memory extraction",
            "billing_notice": "declared-price estimate, not provider-settled currency charge",
        }
