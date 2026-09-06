"""Small external RAG loop. No retriever/reader internals are changed.

The controller knows only RagBackend.run(original question, search query).
Feedback is a fallible observation, not gold. Answer-only backends degrade to
one unverified run. A complete RAG call may cost more than a retrieval-only repair.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from .experience.query_views import CardMemoryView
from .experiments.data_protocol import normalize_question
from .experiments.protocol import (
    Action,
    Answer,
    BackendCallError,
    CallEvent,
    CallResult,
    Evidence,
    ExecutionKind,
    Reader,
    Retriever,
    RuntimeQuestion,
    Usage,
)
from .query_actions import QueryGenerator, RewriteDecision, RewriteForm


@dataclass(frozen=True, slots=True)
class RagRequest:
    question: RuntimeQuestion
    search_query: str


@dataclass(frozen=True, slots=True)
class RagReply:
    answer: Answer
    # None means the backend did not expose evidence; () means retrieved nothing.
    evidence: tuple[Evidence, ...] | None = None
    component_events: tuple[CallEvent, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.answer, Answer):
            raise TypeError("answer must be an Answer")
        if self.evidence is not None:
            if not isinstance(self.evidence, tuple):
                raise TypeError("evidence must be an immutable tuple or unavailable")
            known = {item.evidence_id for item in self.evidence}
            if len(known) != len(self.evidence):
                raise ValueError("duplicate evidence ID")
            if not set(self.answer.cited_evidence_ids) <= known:
                raise ValueError("answer cites evidence outside its own RAG response")


class RagBackend(Protocol):
    execution_kind: ExecutionKind

    def run(self, request: RagRequest) -> CallResult[RagReply]: ...


@dataclass(frozen=True, slots=True)
class Feedback:
    sufficient: bool | None = None
    useful_gain: bool | None = None
    reason: str = "not assessed"
    origin: str = "unavailable"

    def __post_init__(self) -> None:
        if any(type(v) not in (bool, type(None)) for v in (self.sufficient, self.useful_gain)):
            raise TypeError("feedback flags must be bool or None (unknown)")


@dataclass(frozen=True, slots=True)
class RoundRecord:
    decision: RewriteDecision
    search_query: str
    reply: RagReply
    feedback: Feedback


@dataclass(frozen=True, slots=True)
class LoopState:
    question: RuntimeQuestion
    rounds: tuple[RoundRecord, ...] = ()
    observed_evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True, slots=True)
class LoopResult:
    state: LoopState
    stop_reason: str
    events: tuple[CallEvent, ...]
    # Details for audit only: do NOT sum them again with aggregate RAG events.
    component_events: tuple[CallEvent, ...] = ()
    # Read state.rounds[-1].reply for the raw answer, not a verified-answer claim.


Policy = Callable[[LoopState], RewriteDecision | None]
Assessor = Callable[[LoopState, RagReply], Feedback | CallResult[Feedback]]


def base_then_fresh(state: LoopState) -> RewriteDecision:
    """Transparent engineering fallback, not a learned optimal routing policy."""
    if not state.rounds:
        return RewriteDecision()
    return RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "try alternative wording")


def _event(operation: str, kind: ExecutionKind, start: float, result: object) -> CallEvent:
    return CallEvent(
        operation=operation,
        execution_kind=kind,
        status="error" if isinstance(result, BackendCallError) else "ok",
        elapsed_seconds=perf_counter() - start,
        usage=result.usage,
        provider=result.provider,
        model=result.model,
        request_id=result.request_id,
        audit_path=result.audit_path,
        transport_source=result.transport_source,
        error_type=type(result).__name__ if isinstance(result, BackendCallError) else None,
    )


def _checked(result: CallResult, kind: ExecutionKind) -> CallResult:
    allowed = {
        "local_compute",
        "composed_rag",
        "mock" if kind is ExecutionKind.MOCK else "live_api",
    }
    if result.transport_source not in allowed:
        raise BackendCallError(
            "missing or conflicting execution provenance",
            usage=result.usage,
            provider=result.provider,
            model=result.model,
            request_id=result.request_id,
            audit_path=result.audit_path,
            transport_source=result.transport_source,
        )
    return result


def _sum_usage(usages: tuple[Usage, ...]) -> Usage:
    def total(field: str) -> int | None:
        values = [getattr(item, field) for item in usages]
        return None if None in values else sum(values)

    return Usage(total("input_tokens"), total("output_tokens"), total("api_requests"))


class RagCallFailure(BackendCallError):
    """Preserve successful retrieval and failed reading as separate audit events."""

    def __init__(self, error: BackendCallError, events: tuple[CallEvent, ...]) -> None:
        super().__init__(
            "composed RAG failed; no retry",
            usage=_sum_usage(tuple(event.usage for event in events)),
            provider=error.provider,
            model=error.model,
            request_id=error.request_id,
            audit_path=error.audit_path,
            transport_source="composed_rag",
        )
        self.component_events = events


def run_outer_loop(
    question: RuntimeQuestion,
    backend: RagBackend,
    *,
    generator: QueryGenerator | None = None,
    policy: Policy = base_then_fresh,
    assessor: Assessor | None = None,
    max_rag_calls: int = 2,
    no_gain_patience: int = 1,
) -> LoopResult:
    """Policy may choose a rewrite BEFORE the first RAG call.

    No assessor -> single unverified run. Observable evidence is required for a
    feedback loop. Semantic intent preservation is a prompt constraint, not a
    theorem; structural tests cannot prove an LLM preserved every constraint.
    """
    if not isinstance(question, RuntimeQuestion):
        raise TypeError("runtime question only; never pass a labelled dataset record")
    if any(type(v) is not int or v < 1 for v in (max_rag_calls, no_gain_patience)):
        raise ValueError("loop budgets must be positive integers")
    if not isinstance(backend.execution_kind, ExecutionKind):
        raise ValueError("backend must declare execution provenance")
    if generator is not None and generator.execution_kind != backend.execution_kind:
        raise ValueError("mock and real components cannot be mixed")
    state, events, seen, stalls = LoopState(question), [], set(), 0
    components: list[CallEvent] = []

    def finish(reason: str) -> LoopResult:
        return LoopResult(state, reason, tuple(events), tuple(components))

    for _ in range(max_rag_calls):
        decision = policy(state)
        if decision is None:
            return finish("policy_stop")
        if not isinstance(decision, RewriteDecision):
            raise TypeError("policy must return RewriteDecision or None")
        if decision.form is RewriteForm.DECOMPOSE:
            return finish("decomposition_not_implemented")
        if decision.memory and decision.memory.is_source(question):
            return finish("same_question_memory_rejected")
        if isinstance(decision.memory, CardMemoryView):
            try:
                decision.memory.check_stage(
                    after_retrieval=bool(state.rounds), evidence=state.observed_evidence
                )
            except ValueError:
                return finish("card_stage_or_evidence_mismatch")
        query = question.text
        if decision.action is not Action.BASE:
            if generator is None:
                return finish("rewriter_unavailable")
            start = perf_counter()
            try:
                generated = generator.generate(
                    question,
                    decision,
                    evidence=state.observed_evidence,
                    previous_queries=tuple(r.search_query for r in state.rounds),
                )
                _checked(generated, backend.execution_kind)
            except BackendCallError as error:
                events.append(_event("rewrite", backend.execution_kind, start, error))
                return finish("rewrite_error")
            events.append(_event("rewrite", backend.execution_kind, start, generated))
            query = generated.value
            if not isinstance(query, str) or not query.strip() or len(query) > 2000:
                return finish("invalid_query")
        try:
            identity = normalize_question(query)
        except ValueError:
            return finish("invalid_query")
        if identity in seen:
            return finish("repeated_query")
        if not state.rounds and identity == normalize_question(question.text):
            # Generated an unchanged query: execute BASE, but retain generation cost.
            decision = RewriteDecision()
            query = question.text
        start = perf_counter()
        try:
            result = backend.run(RagRequest(question, query))
            _checked(result, backend.execution_kind)
        except BackendCallError as error:
            events.append(_event("rag", backend.execution_kind, start, error))
            components.extend(getattr(error, "component_events", ()))
            return finish("rag_error")
        events.append(_event("rag", backend.execution_kind, start, result))
        reply = result.value
        if not isinstance(reply, RagReply):
            raise TypeError("backend must return a RagReply")
        components.extend(reply.component_events)
        observed = {item.evidence_id: item for item in state.observed_evidence}
        before_ids = set(observed)
        for item in reply.evidence or ():
            if item.evidence_id in observed and observed[item.evidence_id] != item:
                raise ValueError("evidence ID changed content across rounds")
            observed[item.evidence_id] = item
        previous_state = state
        state = LoopState(
            question,
            (*state.rounds, RoundRecord(decision, query, reply, Feedback())),
            tuple(observed.values()),
        )
        feedback = Feedback()
        if assessor and reply.evidence is not None:
            start = perf_counter()
            try:
                assessed = assessor(previous_state, reply)
                if isinstance(assessed, Feedback):
                    # A bare callback is explicitly local and must perform no API calls.
                    assessed = CallResult(
                        assessed,
                        usage=Usage(0, 0, 0),
                        provider="local",
                        transport_source="local_compute",
                    )
                _checked(assessed, backend.execution_kind)
            except BackendCallError as error:
                events.append(_event("assess", backend.execution_kind, start, error))
                return finish("assessment_error")
            events.append(_event("assess", backend.execution_kind, start, assessed))
            feedback = assessed.value
        if not isinstance(feedback, Feedback):
            raise TypeError("assessor must return Feedback, not gold or raw labels")
        if feedback.sufficient and (not reply.answer.text.strip() or not reply.evidence):
            raise ValueError("sufficient feedback requires a nonempty answer and evidence")
        record = RoundRecord(decision, query, reply, feedback)
        state = LoopState(question, (*previous_state.rounds, record), tuple(observed.values()))
        seen.add(identity)
        if reply.evidence is None:
            return finish("evidence_unavailable")
        if assessor is None:
            return finish("unassessed")
        if feedback.sufficient:
            return finish("sufficient_signal")  # An assessor's claim, not verified truth.
        if len(state.rounds) > 1:
            no_new_ids = set(observed) == before_ids
            if feedback.useful_gain is True:
                stalls = 0
            else:
                stalls = stalls + 1 if feedback.useful_gain is False or no_new_ids else 0
            if stalls >= no_gain_patience:
                return finish("no_gain_signal" if feedback.useful_gain is False else "no_new_ids")
    return finish("rag_call_budget")


class RetrieverReaderBackend:
    """Thin adapter for existing components; the controller never accesses them.

    Each call's reader sees only THAT call's evidence, not the outer accumulated
    observations. Cumulative reading requires an explicit backend capability.
    """

    def __init__(self, retriever: Retriever, reader: Reader, *, top_k: int = 4) -> None:
        if retriever.execution_kind != reader.execution_kind:
            raise ValueError("mock and real backend components cannot be mixed")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.retriever, self.reader, self.top_k = retriever, reader, top_k
        self.execution_kind = retriever.execution_kind

    def run(self, request: RagRequest) -> CallResult[RagReply]:
        calls: list[CallEvent] = []
        start = perf_counter()
        try:
            retrieved = self.retriever.retrieve(request.search_query, top_k=self.top_k)
            _checked(retrieved, self.execution_kind)
        except BackendCallError as error:
            calls.append(_event("rag.retrieve", self.execution_kind, start, error))
            raise RagCallFailure(error, tuple(calls)) from None
        calls.append(_event("rag.retrieve", self.execution_kind, start, retrieved))
        if len(retrieved.value) > self.top_k:
            raise ValueError("retriever exceeded its top_k contract")
        start = perf_counter()
        try:
            answered = self.reader.answer(request.question, retrieved.value)
            _checked(answered, self.execution_kind)
        except BackendCallError as error:
            calls.append(_event("rag.answer", self.execution_kind, start, error))
            raise RagCallFailure(error, tuple(calls)) from None
        calls.append(_event("rag.answer", self.execution_kind, start, answered))

        return CallResult(
            RagReply(answered.value, retrieved.value, tuple(calls)),
            usage=_sum_usage((retrieved.usage, answered.usage)),
            provider=answered.provider,
            model=answered.model,
            request_id=answered.request_id,
            audit_path=answered.audit_path,
            transport_source="composed_rag",
        )
