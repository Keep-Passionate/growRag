"""Hand-written test doubles, not LLMs and not a query-quality simulation."""

from __future__ import annotations

from dataclasses import dataclass

from growrag.experiments.paired_runner import PairedRun, PairedRunner
from growrag.experiments.protocol import (
    Action,
    Answer,
    BranchSpec,
    CallResult,
    DecisionState,
    Evidence,
    ExecutionKind,
    MemoryView,
    RuntimeQuestion,
    Usage,
)

MOCK_USAGE = Usage(api_requests=0)


class MockRewriter:
    execution_kind = ExecutionKind.MOCK

    def rewrite(self, state: DecisionState, *, memory: MemoryView | None) -> CallResult[str]:
        # Deliberately scripted branches test plumbing, not reasoning ability.
        query = "Aster author birthplace" if memory is None else "Mira birthplace"
        return CallResult(query, MOCK_USAGE, provider="hand-written-test-double")


@dataclass(frozen=True, slots=True)
class MockRetriever:
    execution_kind = ExecutionKind.MOCK

    def retrieve(self, query: str, *, top_k: int) -> CallResult[tuple[Evidence, ...]]:
        if query == "Mira birthplace":
            evidence = (Evidence("mira:0", "Mira", 0, "Mira was born in Lumen."),)
        else:
            evidence = (Evidence("aster:1", "Aster", 1, "Aster is a fictional short story."),)
        return CallResult(evidence[:top_k], MOCK_USAGE, provider="hand-written-test-double")


class MockReader:
    execution_kind = ExecutionKind.MOCK

    def answer(
        self, question: RuntimeQuestion, evidence: tuple[Evidence, ...]
    ) -> CallResult[Answer]:
        for item in evidence:
            if item.evidence_id == "mira:0":
                return CallResult(
                    Answer("Lumen", (item.evidence_id,)),
                    MOCK_USAGE,
                    provider="hand-written-test-double",
                )
        return CallResult(Answer("INSUFFICIENT_EVIDENCE"), MOCK_USAGE)


def demo_inputs() -> tuple[DecisionState, tuple[BranchSpec, ...]]:
    state = DecisionState(
        RuntimeQuestion("synthetic-target-1", "Where was the author of Aster born?", "synthetic"),
        (Evidence("aster:0", "Aster", 0, "Aster was written by Mira."),),
        gap="MANUALLY SCRIPTED TEST FIXTURE: the confirmed author's birthplace is missing.",
        context_fingerprint="synthetic-fixture-v1/no-real-index/no-llm",
        previous_queries=("Where was the author of Aster born?",),
        state_builder_version="manual-test-fixture-not-an-automatic-judge-v1",
    )
    memory = MemoryView(
        "synthetic-memory-1",
        "synthetic-source-1",
        "synthetic-source-1:step-1",
        "example-with-prerequisite",
        "TEST NOTE: once the author is supported by current evidence, query that author's "
        "missing attribute; do not transfer an old author's name.",
    )
    return state, (
        BranchSpec("base", Action.BASE),
        BranchSpec("fresh", Action.FRESH),
        BranchSpec("reuse", Action.REUSE, memory),
    )


def run_demo() -> PairedRun:
    state, specs = demo_inputs()
    runner = PairedRunner(
        retriever=MockRetriever(),
        rewriter=MockRewriter(),
        reader=MockReader(),
        execution_kind=ExecutionKind.MOCK,
        top_k=1,
    )
    return runner.run(state, specs)
