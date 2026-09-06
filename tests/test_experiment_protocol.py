from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from growrag.experiments.mock import demo_inputs
from growrag.experiments.protocol import (
    Action,
    BranchSpec,
    DecisionState,
    Evidence,
    GoldRecord,
    RuntimeQuestion,
    Usage,
)


def test_runtime_question_and_state_have_no_gold_fields() -> None:
    state, _ = demo_inputs()
    gold = GoldRecord(state.question.question_id, ("CANARY_GOLD_NEVER_FOR_RUNTIME",))
    assert gold.answers[0] not in str(asdict(state))
    assert set(asdict(state.question)) == {"question_id", "text", "dataset"}


def test_shared_state_is_immutable_and_requires_immutable_collections() -> None:
    state, _ = demo_inputs()
    with pytest.raises(FrozenInstanceError):
        state.gap = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        state.evidence[0].text = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="tuple"):
        replace(state, evidence=list(state.evidence))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple"):
        replace(state, previous_queries=[])  # type: ignore[arg-type]


def test_fingerprint_covers_question_evidence_gap_and_context() -> None:
    state, _ = demo_inputs()
    assert state.fingerprint == replace(state).fingerprint
    variants = (
        replace(state, gap="different missing information"),
        replace(state, context_fingerprint="different index or prompt"),
        replace(state, question=RuntimeQuestion("other", state.question.text)),
        replace(state, evidence=(replace(state.evidence[0], text="changed evidence"),)),
        replace(state, previous_queries=("changed earlier query",)),
        replace(state, state_builder_version="different judge prompt"),
    )
    assert all(variant.fingerprint != state.fingerprint for variant in variants)


def test_duplicate_shared_evidence_ids_are_rejected() -> None:
    state, _ = demo_inputs()
    with pytest.raises(ValueError, match="unique"):
        replace(state, evidence=state.evidence * 2)


@pytest.mark.parametrize("sentence_id", [-1, True, 1.5])
def test_evidence_rejects_invalid_sentence_ids(sentence_id: int) -> None:
    with pytest.raises((TypeError, ValueError)):
        Evidence("doc:0", "doc", sentence_id, "text")


def test_branch_memory_visibility_is_explicit() -> None:
    _, specs = demo_inputs()
    memory = specs[-1].memory
    with pytest.raises(ValueError, match="cannot receive memory"):
        BranchSpec("fresh", Action.FRESH, memory)
    with pytest.raises(ValueError, match="requires one MemoryView"):
        BranchSpec("reuse", Action.REUSE)
    with pytest.raises(TypeError, match="Action"):
        BranchSpec("base", "BASE")  # type: ignore[arg-type]


def test_unknown_usage_is_not_zero() -> None:
    assert asdict(Usage()) == {"input_tokens": None, "output_tokens": None, "api_requests": None}
    assert Usage(api_requests=0).input_tokens is None
    with pytest.raises(ValueError):
        Usage(input_tokens=-1)
    with pytest.raises(ValueError):
        Usage(output_tokens=True)


def test_gold_requires_immutable_supporting_fact_tuples() -> None:
    with pytest.raises(ValueError, match="supporting_facts"):
        GoldRecord("q", ("answer",), (["doc", 0],))  # type: ignore[arg-type]


def test_state_rejects_gold_as_runtime_question() -> None:
    with pytest.raises(TypeError, match="RuntimeQuestion"):
        DecisionState(GoldRecord("q", ("secret",)), (), "gap", "context")  # type: ignore[arg-type]
