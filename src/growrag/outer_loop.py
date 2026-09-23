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
    # None preserves legacy backends. With cumulative reading this is ONLY the
    # latest retrieval, while evidence is the context actually read this round.
    retrieved_evidence: tuple[Evidence, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.answer, Answer):
            raise TypeError("answer must be an Answer")
        if not isinstance(self.component_events, tuple) or not all(
            isinstance(event, CallEvent) for event in self.component_events
        ):
            raise TypeError("component events must be immutable CallEvent records")
        if self.evidence is not None:
            if not isinstance(self.evidence, tuple):
                raise TypeError("evidence must be an immutable tuple or unavailable")
            if not all(isinstance(item, Evidence) for item in self.evidence):
                raise TypeError("evidence must contain Evidence records")
            known = {item.evidence_id for item in self.evidence}
            if len(known) != len(self.evidence):
                raise ValueError("duplicate evidence ID")
            if not set(self.answer.cited_evidence_ids) <= known:
                raise ValueError("answer cites evidence outside its own RAG response")
        if self.retrieved_evidence is not None:
            _validate_evidence(self.retrieved_evidence, "retrieved evidence")
            visible = {item.evidence_id: item for item in self.evidence or ()}
            for item in self.retrieved_evidence:
                if item.evidence_id in visible and visible[item.evidence_id] != item:
                    raise ValueError("retrieved evidence conflicts with reader context")


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


Policy = Callable[[LoopState], RewriteDecision | None | CallResult[RewriteDecision | None]]
Assessor = Callable[[LoopState, RagReply], Feedback | CallResult[Feedback]]


def _validate_evidence(value: object, name: str) -> None:
    if not isinstance(value, tuple) or not all(isinstance(item, Evidence) for item in value):
        raise TypeError(f"{name} must be an immutable tuple of Evidence")
    if len({item.evidence_id for item in value}) != len(value):
        raise ValueError(f"{name} has duplicate evidence IDs")


def _checked_prefix(
    question: RuntimeQuestion,
    initial_state: LoopState | None,
    initial_events: tuple[CallEvent, ...],
    max_rag_calls: int,
    kind: ExecutionKind,
) -> tuple[LoopState, set[str], list[CallEvent]]:
    """共享首轮只能原样接续：不重跑、不重计轮数，也不借用其它题的轨迹。"""
    state = LoopState(question) if initial_state is None else initial_state
    if not isinstance(state, LoopState) or state.question != question:
        raise ValueError("initial state must belong to this exact runtime question")
    if not isinstance(state.rounds, tuple) or not all(
        isinstance(record, RoundRecord) for record in state.rounds
    ):
        raise TypeError("initial rounds must be immutable RoundRecord records")
    if len(state.rounds) > max_rag_calls:
        raise ValueError("initial rounds exceed the total RAG call budget")
    _validate_evidence(state.observed_evidence, "initial observed evidence")
    if not isinstance(initial_events, tuple) or not all(
        isinstance(event, CallEvent) for event in initial_events
    ):
        raise TypeError("initial events must be immutable CallEvent records")
    if any(event.execution_kind != kind for event in initial_events):
        raise ValueError("initial events and backend cannot mix execution provenance")
    observed = {item.evidence_id: item for item in state.observed_evidence}
    recorded: dict[str, Evidence] = {}
    seen: set[str] = set()
    components: list[CallEvent] = []
    for index, record in enumerate(state.rounds):
        if not isinstance(record.decision, RewriteDecision) or not isinstance(
            record.reply, RagReply
        ):
            raise TypeError("initial round must contain a decision and RagReply")
        if not isinstance(record.feedback, Feedback):
            raise TypeError("initial round must contain Feedback")
        identity = normalize_question(record.search_query)
        if identity in seen:
            raise ValueError("initial rounds contain repeated queries")
        seen.add(identity)
        if record.feedback.sufficient:
            if not record.reply.answer.text.strip() or not record.reply.evidence:
                raise ValueError("sufficient feedback requires a nonempty answer and evidence")
            if index != len(state.rounds) - 1:
                raise ValueError("initial rounds continued after a sufficient signal")
        for item in (*(record.reply.evidence or ()), *(record.reply.retrieved_evidence or ())):
            if item.evidence_id in recorded and recorded[item.evidence_id] != item:
                raise ValueError("evidence ID changed content across initial rounds")
            recorded[item.evidence_id] = item
        if any(event.execution_kind != kind for event in record.reply.component_events):
            raise ValueError("initial components and backend cannot mix execution provenance")
        components.extend(record.reply.component_events)
    if recorded != observed:
        raise ValueError("initial observed evidence must match its recorded rounds")
    return state, seen, components


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
    if not isinstance(result, CallResult) or not isinstance(result.usage, Usage):
        raise TypeError("component must return an audited CallResult with typed usage")
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


def _call_failure(error: Exception, returned: object = None) -> BackendCallError:
    """Sanitize unexpected component errors without inventing unobserved usage.

    A returned audited envelope may contain known cost even when its value is
    invalid. If the call raised before returning, only BackendCallError carries
    trustworthy call metadata. Never serialize an arbitrary exception message.
    """
    if isinstance(error, BackendCallError):
        return error
    metadata = {}
    if isinstance(returned, CallResult):
        metadata["usage"] = returned.usage if isinstance(returned.usage, Usage) else Usage()
        for field in ("provider", "model", "request_id", "audit_path", "transport_source"):
            value = getattr(returned, field)
            metadata[field] = value if isinstance(value, str) else None
    return BackendCallError(
        "component execution or response validation failed; no retry", **metadata
    )


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
    initial_state: LoopState | None = None,
    initial_events: tuple[CallEvent, ...] = (),
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
    state, seen, components = _checked_prefix(
        question, initial_state, initial_events, max_rag_calls, backend.execution_kind
    )
    events, stalls = list(initial_events), 0

    def finish(reason: str) -> LoopResult:
        return LoopResult(state, reason, tuple(events), tuple(components))

    if state.rounds and state.rounds[-1].feedback.sufficient:
        return finish("sufficient_signal")
    # 接续较长 prefix 时也不能清空已有的无增益计数，借此绕过停止条件。
    prior_ids: set[str] = set()
    for index, record in enumerate(state.rounds):
        if record.reply.evidence is None:
            return finish("evidence_unavailable")
        current_ids = {
            item.evidence_id
            for item in (*record.reply.evidence, *(record.reply.retrieved_evidence or ()))
        }
        no_new_ids = current_ids <= prior_ids
        prior_ids.update(current_ids)
        if index:
            if record.feedback.useful_gain is True:
                stalls = 0
            else:
                stalls = stalls + 1 if record.feedback.useful_gain is False or no_new_ids else 0
            if stalls >= no_gain_patience:
                return finish(
                    "no_gain_signal" if record.feedback.useful_gain is False else "no_new_ids"
                )
    # prefix 的轮数已经付费并消耗预算；max_rag_calls 是全程总数，不是新增轮数。
    for _ in range(max_rag_calls - len(state.rounds)):
        start = perf_counter()
        try:
            routed = policy(state)
        except BackendCallError as error:
            events.append(_event("route", backend.execution_kind, start, error))
            return finish("route_error")
        # 纯本地规则保留旧合同；付费路由必须返回可审计封装，不能把费用藏在裸返回值里。
        if isinstance(routed, CallResult):
            try:
                _checked(routed, backend.execution_kind)
                if routed.value is not None and not isinstance(routed.value, RewriteDecision):
                    raise TypeError("audited policy must return RewriteDecision or None")
            except Exception as error:
                events.append(
                    _event("route", backend.execution_kind, start, _call_failure(error, routed))
                )
                return finish("route_error")
            events.append(_event("route", backend.execution_kind, start, routed))
            decision = routed.value
        else:
            decision = routed
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
            generated = None
            try:
                generated = generator.generate(
                    question,
                    decision,
                    evidence=state.observed_evidence,
                    previous_queries=tuple(r.search_query for r in state.rounds),
                )
                _checked(generated, backend.execution_kind)
            except Exception as error:
                failure = _call_failure(error, generated)
                events.append(_event("rewrite", backend.execution_kind, start, failure))
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
        result = None
        try:
            result = backend.run(RagRequest(question, query))
            _checked(result, backend.execution_kind)
            if not isinstance(result.value, RagReply):
                raise TypeError("backend must return a RagReply")
        except Exception as error:
            failure = _call_failure(error, result)
            events.append(_event("rag", backend.execution_kind, start, failure))
            components.extend(getattr(failure, "component_events", ()))
            return finish("rag_error")
        events.append(_event("rag", backend.execution_kind, start, result))
        reply = result.value
        components.extend(reply.component_events)
        observed = {item.evidence_id: item for item in state.observed_evidence}
        before_ids = set(observed)
        # 观察库存用于审计；Reader 上下文可能因上限裁剪，两者不能混作支撑证明。
        for item in (*(reply.evidence or ()), *(reply.retrieved_evidence or ())):
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
            assessed = None
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
                if not isinstance(assessed.value, Feedback):
                    raise TypeError("assessor must return Feedback, not gold or raw labels")
            except Exception as error:
                failure = _call_failure(error, assessed)
                events.append(_event("assess", backend.execution_kind, start, failure))
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

    def _reader_context(
        self, request: RagRequest, evidence: tuple[Evidence, ...]
    ) -> tuple[Evidence, ...]:
        return evidence

    def _latest_evidence(self, evidence: tuple[Evidence, ...]) -> tuple[Evidence, ...] | None:
        return None  # Keep the old, non-cumulative response contract unchanged.

    def _accept_reply(self, reply: RagReply) -> None:
        pass

    def run(self, request: RagRequest) -> CallResult[RagReply]:
        calls: list[CallEvent] = []
        start = perf_counter()
        retrieved = None
        try:
            retrieved = self.retriever.retrieve(request.search_query, top_k=self.top_k)
            _checked(retrieved, self.execution_kind)
            if not isinstance(retrieved.value, tuple) or not all(
                isinstance(item, Evidence) for item in retrieved.value
            ):
                raise TypeError("retriever must return an immutable tuple of Evidence")
            if len(retrieved.value) > self.top_k:
                raise ValueError("retriever exceeded its top_k contract")
            if len({item.evidence_id for item in retrieved.value}) != len(retrieved.value):
                raise ValueError("retriever returned duplicate evidence IDs")
            reader_context = self._reader_context(request, retrieved.value)
        except Exception as error:
            failure = _call_failure(error, retrieved)
            calls.append(_event("rag.retrieve", self.execution_kind, start, failure))
            raise RagCallFailure(failure, tuple(calls)) from None
        calls.append(_event("rag.retrieve", self.execution_kind, start, retrieved))
        start = perf_counter()
        answered = None
        try:
            answered = self.reader.answer(request.question, reader_context)
            _checked(answered, self.execution_kind)
            # Validate the actual reader-visible evidence before accepting the
            # answer. A malformed response still consumed its reported cost.
            reply = RagReply(
                answered.value,
                reader_context,
                retrieved_evidence=self._latest_evidence(retrieved.value),
            )
        except Exception as error:
            failure = _call_failure(error, answered)
            calls.append(_event("rag.answer", self.execution_kind, start, failure))
            raise RagCallFailure(failure, tuple(calls)) from None
        calls.append(_event("rag.answer", self.execution_kind, start, answered))
        self._accept_reply(reply)

        return CallResult(
            RagReply(reply.answer, reply.evidence, tuple(calls), reply.retrieved_evidence),
            usage=_sum_usage((retrieved.usage, answered.usage)),
            provider=answered.provider,
            model=answered.model,
            request_id=answered.request_id,
            audit_path=answered.audit_path,
            transport_source="composed_rag",
        )


class CumulativeRetrieverReaderBackend(RetrieverReaderBackend):
    """Bounded current-question context, never a cross-question memory store.

    Construct a NEW backend for each branch, passing that branch's immutable
    shared prefix evidence. The retriever and reader may be reusable components;
    this mutable evidence container must not be shared between branches.
    """

    def __init__(
        self,
        retriever: Retriever,
        reader: Reader,
        *,
        question: RuntimeQuestion,
        initial_evidence: tuple[Evidence, ...] = (),
        top_k: int = 4,
        max_evidence: int = 8,
    ) -> None:
        super().__init__(retriever, reader, top_k=top_k)
        if not isinstance(question, RuntimeQuestion):
            raise TypeError("cumulative backend requires a bound RuntimeQuestion")
        if type(max_evidence) is not int or max_evidence < 1:
            raise ValueError("max_evidence must be a positive integer")
        _validate_evidence(initial_evidence, "initial reader evidence")
        if len(initial_evidence) > max_evidence:
            raise ValueError("initial reader evidence exceeds max_evidence")
        self.question, self.max_evidence = question, max_evidence
        self._evidence = initial_evidence
        # 被裁掉的证据仍在本题 ID 登记中，避免以后用相同 ID 偷换内容。
        self._known = {item.evidence_id: item for item in initial_evidence}

    def run(self, request: RagRequest) -> CallResult[RagReply]:
        if not isinstance(request, RagRequest) or request.question != self.question:
            raise BackendCallError(
                "cumulative backend cannot be reused across questions",
                usage=Usage(0, 0, 0),
                provider="local",
                transport_source="local_compute",
            )
        return super().run(request)

    def _reader_context(
        self, request: RagRequest, evidence: tuple[Evidence, ...]
    ) -> tuple[Evidence, ...]:
        for item in evidence:
            if item.evidence_id in self._known and self._known[item.evidence_id] != item:
                raise ValueError("evidence ID changed content across rounds")
        # 新证据优先、同 ID 确定性去重，再截断。上限只约束真正送给 Reader 的内容。
        merged = {item.evidence_id: item for item in evidence}
        for item in self._evidence:
            merged.setdefault(item.evidence_id, item)
        return tuple(merged.values())[: self.max_evidence]

    def _latest_evidence(self, evidence: tuple[Evidence, ...]) -> tuple[Evidence, ...]:
        return evidence

    def _accept_reply(self, reply: RagReply) -> None:
        self._evidence = reply.evidence or ()
        self._known.update((item.evidence_id, item) for item in reply.retrieved_evidence or ())
