"""Run independent, single-repair branches from an existing shared snapshot.

No initial retrieval or automatic gap judge is implemented here. Neither gold
answers nor gold support facts are accepted by this execution layer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from growrag.experiments.protocol import (
    Action,
    Answer,
    BackendCallError,
    BranchResult,
    BranchSpec,
    CallEvent,
    CallResult,
    DecisionState,
    Evidence,
    ExecutionKind,
    Reader,
    Retriever,
    Rewriter,
    RuntimeQuestion,
    Usage,
)

SCHEMA_VERSION = "growrag-single-repair-v1"
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class PairedRun:
    state: DecisionState
    execution_kind: ExecutionKind
    branches: tuple[BranchResult, ...]
    top_k: int

    def to_dict(self) -> dict[str, object]:
        branch_records = []
        for branch in self.branches:
            record = asdict(branch)
            record["retrieval_calls"] = branch.retrieval_calls
            record["model_component_calls"] = branch.model_component_calls
            record["api_requests"] = branch.api_requests
            branch_records.append(record)
        return {
            "schema_version": SCHEMA_VERSION,
            "execution_kind": self.execution_kind,
            "synthetic_demo": self.execution_kind is ExecutionKind.MOCK,
            "notice": (
                "Hand-written synthetic fixture, not HotpotQA, not an LLM/API run, "
                "and not evidence of research effectiveness."
                if self.execution_kind is ExecutionKind.MOCK
                else "Real backend execution; this label alone does not validate research quality."
            ),
            "cost_scope": "branch calls only; supplied shared-prefix costs are NOT included",
            "max_additional_retrievals_per_branch": 1,
            "retrieval_top_k": self.top_k,
            "state_fingerprint": self.state.fingerprint,
            "state": asdict(self.state),
            "branches": branch_records,
        }


def _normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


def _merge_evidence(
    initial: tuple[Evidence, ...], returned: tuple[Evidence, ...]
) -> tuple[Evidence, ...]:
    if not isinstance(returned, tuple) or not all(isinstance(item, Evidence) for item in returned):
        raise TypeError("retriever must return a tuple of Evidence")
    merged = {item.evidence_id: item for item in initial}
    for item in returned:
        if item.evidence_id in merged and merged[item.evidence_id] != item:
            raise ValueError("an evidence ID was reused with different content")
        merged[item.evidence_id] = item
    return tuple(merged.values())


class PairedRunner:
    """Components must be stateless with respect to branch-specific information.

    Inputs and outputs are immutable and each branch receives only its own
    evidence. This is not a sandbox for a malicious/stateful adapter. Adapter
    contract tests are still required for real backends.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        rewriter: Rewriter,
        reader: Reader,
        execution_kind: ExecutionKind,
        top_k: int = 5,
    ) -> None:
        if not isinstance(execution_kind, ExecutionKind):
            raise TypeError("execution_kind must be explicitly declared with ExecutionKind")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        for component in (retriever, rewriter, reader):
            if getattr(component, "execution_kind", None) != execution_kind:
                raise ValueError("mixed or missing backend kinds: no automatic mock fallback")
        self.retriever = retriever
        self.rewriter = rewriter
        self.reader = reader
        self.execution_kind = execution_kind
        self.top_k = top_k

    def run(self, state: DecisionState, specs: tuple[BranchSpec, ...]) -> PairedRun:
        if not isinstance(state, DecisionState):
            raise TypeError("state must be a gold-free DecisionState")
        if not isinstance(specs, tuple) or not specs:
            raise ValueError("specs must be a non-empty tuple")
        if not all(isinstance(spec, BranchSpec) for spec in specs):
            raise TypeError("specs must contain BranchSpec objects")
        if len({spec.branch_id for spec in specs}) != len(specs):
            raise ValueError("branch IDs must be unique")
        for spec in specs:
            if (
                spec.memory is not None
                and spec.memory.source_query_id == state.question.question_id
            ):
                raise ValueError("cross-query reuse cannot use the target query as its source")
        results = tuple(self._run_branch(state, spec) for spec in specs)
        return PairedRun(state, self.execution_kind, results, self.top_k)

    def _invoke(
        self, operation: str, callback: Callable[[], CallResult[T]], calls: list[CallEvent]
    ) -> T:
        started = time.perf_counter()
        try:
            response = callback()
            if not isinstance(response, CallResult) or not isinstance(response.usage, Usage):
                raise TypeError("backend must return CallResult with Usage")
        except Exception as error:
            usage = Usage(api_requests=0) if self.execution_kind is ExecutionKind.MOCK else Usage()
            provider = model = request_id = audit_path = transport_source = None
            if isinstance(error, BackendCallError):
                usage = error.usage
                provider, model, request_id = error.provider, error.model, error.request_id
                audit_path, transport_source = error.audit_path, error.transport_source
            calls.append(
                CallEvent(
                    operation,
                    self.execution_kind,
                    "error",
                    time.perf_counter() - started,
                    usage,
                    provider,
                    model,
                    request_id,
                    type(error).__name__,
                    audit_path,
                    transport_source,
                )
            )
            raise
        calls.append(
            CallEvent(
                operation,
                self.execution_kind,
                "completed",
                time.perf_counter() - started,
                response.usage,
                response.provider,
                response.model,
                response.request_id,
                audit_path=response.audit_path,
                transport_source=response.transport_source,
            )
        )
        return response.value

    def _run_branch(self, state: DecisionState, spec: BranchSpec) -> BranchResult:
        calls: list[CallEvent] = []
        query: str | None = None
        cumulative = state.evidence
        new_evidence: tuple[Evidence, ...] = ()
        answer: Answer | None = None
        status, stop_reason = "completed", None
        error_type = error_message = None
        try:
            if spec.action is not Action.BASE:
                query = self._invoke(
                    "rewrite", lambda: self.rewriter.rewrite(state, memory=spec.memory), calls
                )
                if not isinstance(query, str):
                    raise TypeError("rewriter must return one query string")
                query = query.strip()
                previous = {state.question.text, *state.previous_queries}
                if not query:
                    status, stop_reason = "stopped", "empty_query"
                elif _normalize_query(query) in {_normalize_query(item) for item in previous}:
                    status, stop_reason = "stopped", "duplicate_query"
                else:
                    returned = self._invoke(
                        "retrieve", lambda: self.retriever.retrieve(query, top_k=self.top_k), calls
                    )
                    if isinstance(returned, tuple) and len(returned) > self.top_k:
                        raise ValueError("retriever exceeded the declared top_k evidence budget")
                    cumulative = _merge_evidence(state.evidence, returned)
                    initial_ids = {item.evidence_id for item in state.evidence}
                    new_evidence = tuple(
                        item for item in cumulative if item.evidence_id not in initial_ids
                    )
            if status == "completed":
                answer = self._invoke(
                    "answer", lambda: self.reader.answer(state.question, cumulative), calls
                )
                if not isinstance(answer, Answer):
                    raise TypeError("reader must return Answer")
                known_ids = {item.evidence_id for item in cumulative}
                if not set(answer.cited_evidence_ids).issubset(known_ids):
                    raise ValueError("reader cited evidence unavailable to this branch")
        except Exception as error:
            status = "error"
            answer = None
            error_type = type(error).__name__
            # Only the explicit adapter error contract promises a sanitized message.
            # Unexpected exceptions can contain request bodies or credentials.
            error_message = (
                str(error)[:1000]
                if isinstance(error, BackendCallError)
                else "Unexpected backend or contract error; raw exception text was not logged."
            )
        return BranchResult(
            spec.branch_id,
            spec.action,
            state.fingerprint,
            self.execution_kind,
            spec.memory,
            status,
            query,
            new_evidence,
            cumulative,
            answer,
            tuple(calls),
            stop_reason,
            error_type,
            error_message,
        )


def write_run(run: PairedRun, output_dir: Path) -> Path:
    """Create a NEW result directory; never overwrite an earlier experiment."""
    output_dir.mkdir(parents=True, exist_ok=False)
    destination = output_dir / "run.json"
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(run.to_dict(), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return destination


def load_run(path: Path) -> dict[str, object]:
    """Replay saved observations without executing retrieval or model code."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported experiment result schema")
    state_data = data["state"]
    state = DecisionState(
        question=RuntimeQuestion(**state_data["question"]),
        evidence=tuple(Evidence(**item) for item in state_data["evidence"]),
        gap=state_data["gap"],
        context_fingerprint=state_data["context_fingerprint"],
        previous_queries=tuple(state_data["previous_queries"]),
        state_builder_version=state_data["state_builder_version"],
    )
    if state.fingerprint != data["state_fingerprint"]:
        raise ValueError("shared state fingerprint does not match saved content")
    if not isinstance(data.get("branches"), list) or not data["branches"]:
        raise ValueError("saved run must contain branch records")
    for branch in data["branches"]:
        if branch["state_fingerprint"] != state.fingerprint:
            raise ValueError("branch does not share the saved initial state")
    return data
