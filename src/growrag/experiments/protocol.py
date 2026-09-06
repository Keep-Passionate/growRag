"""Gold-free, immutable inputs for one-step paired repair experiments.

GoldRecord is deliberately not an input to any runtime component. These types
enforce structural separation, not a semantic proof that arbitrary input text
cannot contain a leaked answer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar


class ExecutionKind(StrEnum):
    MOCK = "mock"
    REAL = "real"


class Action(StrEnum):
    BASE = "BASE"
    FRESH = "FRESH"
    REUSE = "REUSE"


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _tuple(value: object, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple, not a mutable collection")


@dataclass(frozen=True, slots=True)
class RuntimeQuestion:
    question_id: str
    text: str
    dataset: str = ""

    def __post_init__(self) -> None:
        _nonempty(self.question_id, "question_id")
        _nonempty(self.text, "text")


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    title: str
    sentence_id: int
    text: str

    def __post_init__(self) -> None:
        _nonempty(self.evidence_id, "evidence_id")
        _nonempty(self.title, "title")
        _nonempty(self.text, "text")
        if isinstance(self.sentence_id, bool) or not isinstance(self.sentence_id, int):
            raise TypeError("sentence_id must be an integer")
        if self.sentence_id < 0:
            raise ValueError("sentence_id must be non-negative")


@dataclass(frozen=True, slots=True)
class GoldRecord:
    """Held outside the runner, for post-execution evaluation only."""

    question_id: str
    answers: tuple[str, ...]
    supporting_facts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.question_id, "question_id")
        _tuple(self.answers, "answers")
        _tuple(self.supporting_facts, "supporting_facts")
        if not self.answers or not all(isinstance(answer, str) for answer in self.answers):
            raise ValueError("answers must contain at least one answer string")
        for fact in self.supporting_facts:
            if (
                not isinstance(fact, tuple)
                or len(fact) != 2
                or not isinstance(fact[0], str)
                or not fact[0]
                or isinstance(fact[1], bool)
                or not isinstance(fact[1], int)
                or fact[1] < 0
            ):
                raise ValueError("supporting_facts must contain (title, sentence_id) tuples")


@dataclass(frozen=True, slots=True)
class DecisionState:
    question: RuntimeQuestion
    evidence: tuple[Evidence, ...]
    gap: str
    context_fingerprint: str
    previous_queries: tuple[str, ...] = ()
    state_builder_version: str = "externally-supplied-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.question, RuntimeQuestion):
            raise TypeError("question must be RuntimeQuestion")
        _tuple(self.evidence, "evidence")
        if not all(isinstance(item, Evidence) for item in self.evidence):
            raise TypeError("evidence must contain Evidence objects")
        if len({item.evidence_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("shared evidence IDs must be unique")
        _tuple(self.previous_queries, "previous_queries")
        if not all(isinstance(query, str) and query.strip() for query in self.previous_queries):
            raise ValueError("previous_queries must contain non-empty strings")
        if not isinstance(self.gap, str):
            raise TypeError("gap must be a string")
        _nonempty(self.context_fingerprint, "context_fingerprint")
        _nonempty(self.state_builder_version, "state_builder_version")

    @property
    def fingerprint(self) -> str:
        serialized = json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MemoryView:
    memory_id: str
    source_query_id: str
    source_step_id: str
    view_type: str
    text: str

    def __post_init__(self) -> None:
        for field in ("memory_id", "source_query_id", "source_step_id", "view_type", "text"):
            _nonempty(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    cited_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("answer text must be a string")
        _tuple(self.cited_evidence_ids, "cited_evidence_ids")
        if not all(isinstance(ref, str) and ref for ref in self.cited_evidence_ids):
            raise ValueError("citations must be non-empty evidence IDs")


@dataclass(frozen=True, slots=True)
class Usage:
    """Actual backend-reported usage; None means unknown, never zero."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    api_requests: int | None = None

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "api_requests"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")


T = TypeVar("T")
UNKNOWN_USAGE = Usage()


@dataclass(frozen=True, slots=True)
class CallResult(Generic[T]):
    value: T
    usage: Usage = Usage()
    provider: str | None = None
    model: str | None = None
    request_id: str | None = None
    audit_path: str | None = None
    transport_source: str | None = None


class BackendCallError(RuntimeError):
    """Adapters can preserve known usage when a request fails.

    The message must be safe for a public experiment log: no API keys, auth
    headers, private endpoint query strings, or full server error bodies.
    """

    def __init__(
        self,
        message: str,
        *,
        usage: Usage = UNKNOWN_USAGE,
        provider: str | None = None,
        model: str | None = None,
        request_id: str | None = None,
        audit_path: str | None = None,
        transport_source: str | None = None,
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.provider = provider
        self.model = model
        self.request_id = request_id
        self.audit_path = audit_path
        self.transport_source = transport_source


class Retriever(Protocol):
    execution_kind: ExecutionKind

    def retrieve(self, query: str, *, top_k: int) -> CallResult[tuple[Evidence, ...]]: ...


class Rewriter(Protocol):
    execution_kind: ExecutionKind

    def rewrite(self, state: DecisionState, *, memory: MemoryView | None) -> CallResult[str]: ...


class Reader(Protocol):
    execution_kind: ExecutionKind

    def answer(
        self, question: RuntimeQuestion, evidence: tuple[Evidence, ...]
    ) -> CallResult[Answer]: ...


@dataclass(frozen=True, slots=True)
class BranchSpec:
    branch_id: str
    action: Action
    memory: MemoryView | None = None

    def __post_init__(self) -> None:
        _nonempty(self.branch_id, "branch_id")
        if not isinstance(self.action, Action):
            raise TypeError("action must be Action")
        if self.action is Action.REUSE and not isinstance(self.memory, MemoryView):
            raise ValueError("REUSE requires one MemoryView")
        if self.action is not Action.REUSE and self.memory is not None:
            raise ValueError("BASE and FRESH cannot receive memory")


@dataclass(frozen=True, slots=True)
class CallEvent:
    operation: str
    execution_kind: ExecutionKind
    status: str
    elapsed_seconds: float
    usage: Usage
    provider: str | None = None
    model: str | None = None
    request_id: str | None = None
    error_type: str | None = None
    audit_path: str | None = None
    transport_source: str | None = None


@dataclass(frozen=True, slots=True)
class BranchResult:
    branch_id: str
    action: Action
    state_fingerprint: str
    execution_kind: ExecutionKind
    memory: MemoryView | None
    status: str
    query: str | None
    new_evidence: tuple[Evidence, ...]
    cumulative_evidence: tuple[Evidence, ...]
    answer: Answer | None
    calls: tuple[CallEvent, ...]
    stop_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def retrieval_calls(self) -> int:
        return sum(call.operation == "retrieve" for call in self.calls)

    @property
    def model_component_calls(self) -> int:
        """Logical calls, including hand-written mock functions when in mock mode."""
        return sum(call.operation in {"rewrite", "answer"} for call in self.calls)

    @property
    def api_requests(self) -> int | None:
        values = [call.usage.api_requests for call in self.calls]
        if any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)
