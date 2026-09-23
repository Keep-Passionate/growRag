"""Synthetic offline contracts for shared-prefix repair, NOT measured QA quality."""

from dataclasses import replace

import pytest

from growrag.experiments.protocol import (
    Action,
    Answer,
    BackendCallError,
    CallResult,
    Evidence,
    ExecutionKind,
    RuntimeQuestion,
    Usage,
)
from growrag.outer_loop import (
    CumulativeRetrieverReaderBackend,
    Feedback,
    LoopState,
    RagReply,
    RagRequest,
    RoundRecord,
    run_outer_loop,
)
from growrag.query_actions import RewriteDecision, RewriteForm

Q = RuntimeQuestion("shared-question", "Which fictional town contains the invented school?")
A = Evidence("a", "School", 0, "The invented school is in Fiction Town.")
B = Evidence("b", "Town", 0, "Fiction Town is a fictional town.")
C = Evidence("c", "Third", 0, "A third synthetic source.")
FRESH = RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "reword")
ZERO_USAGE = Usage(0, 0, 0)


def response(value, usage=ZERO_USAGE):
    return CallResult(value, usage=usage, transport_source="mock")


class Search:
    execution_kind = ExecutionKind.MOCK

    def __init__(self, *outputs):
        self.outputs, self.calls = outputs, []

    def retrieve(self, query, *, top_k):
        self.calls.append((query, top_k))
        return response(self.outputs[len(self.calls) - 1], Usage(5, 0, 0))


class Read:
    execution_kind = ExecutionKind.MOCK

    def __init__(self):
        self.calls = []

    def answer(self, question, evidence):
        self.calls.append((question, evidence))
        return response(Answer("synthetic", tuple(e.evidence_id for e in evidence)), Usage(7, 2, 0))


class Generate:
    execution_kind = ExecutionKind.MOCK

    def __init__(self, query="invented school municipality"):
        self.query, self.calls = query, []

    def generate(self, question, decision, **kwargs):
        self.calls.append((question, decision, kwargs))
        return response(self.query, Usage(3, 2, 0))


def insufficient(state, reply):
    return Feedback(False, None, origin="synthetic_test")


def backend(*outputs, initial_evidence=(), max_evidence=8):
    search, read = Search(*outputs), Read()
    return CumulativeRetrieverReaderBackend(
        search, read, question=Q, initial_evidence=initial_evidence, max_evidence=max_evidence
    )


def prefix(*, sufficient=False):
    return run_outer_loop(
        Q,
        backend((A,)),
        assessor=lambda state, reply: Feedback(sufficient, origin="synthetic_test"),
        max_rag_calls=1,
    )


def test_second_reader_receives_actual_cumulative_context_and_original_question():
    rag = backend((A,), (B,))
    result = run_outer_loop(Q, rag, generator=Generate(), assessor=insufficient)
    assert rag.reader.calls == [(Q, (A,)), (Q, (B, A))]
    assert result.state.rounds[0].reply.retrieved_evidence == (A,)
    assert result.state.rounds[1].reply.retrieved_evidence == (B,)
    assert result.state.rounds[1].reply.evidence == (B, A)
    assert result.state.observed_evidence == (A, B)
    assert all(top_k == 4 for _, top_k in rag.retriever.calls)


def test_newest_first_unique_ids_and_reader_cap_do_not_erase_observation_audit():
    rag = backend((A, B), (B, C), max_evidence=2)
    result = run_outer_loop(Q, rag, generator=Generate(), assessor=insufficient)
    assert rag.reader.calls == [(Q, (A, B)), (Q, (B, C))]
    assert result.state.observed_evidence == (A, B, C)
    assert result.state.rounds[-1].reply.answer.cited_evidence_ids == ("b", "c")


def test_retrieved_but_cropped_evidence_remains_observed_not_automatically_supported():
    rag = backend((A, B, C), max_evidence=1)
    result = run_outer_loop(Q, rag)
    assert result.state.rounds[0].reply.evidence == (A,)
    assert result.state.rounds[0].reply.retrieved_evidence == (A, B, C)
    assert result.state.observed_evidence == (A, B, C)
    assert result.state.rounds[0].feedback.sufficient is None


def test_changed_id_rejected_before_reader_and_preserves_paid_retrieval_cost():
    rag = backend((A,), (replace(A, text="conflicting content"),))
    result = run_outer_loop(Q, rag, generator=Generate(), assessor=insufficient)
    assert result.stop_reason == "rag_error"
    assert len(rag.reader.calls) == 1
    assert result.events[-1].usage == Usage(5, 0, 0)
    assert result.component_events[-1].operation == "rag.retrieve"
    assert result.component_events[-1].status == "error"


def test_changed_evicted_id_is_still_rejected():
    rag = backend(
        (B,), (replace(A, text="changed after eviction"),), initial_evidence=(A,), max_evidence=1
    )
    rag.run(RagRequest(Q, "first"))
    with pytest.raises(BackendCallError):
        rag.run(RagRequest(Q, "second"))
    assert len(rag.reader.calls) == 1


@pytest.mark.parametrize(
    "other", [replace(Q, question_id="other"), replace(Q, text="another text")]
)
def test_backend_is_question_bound_and_rejects_before_any_calls(other):
    rag = backend((B,), initial_evidence=(A,))
    result = run_outer_loop(other, rag)
    assert result.stop_reason == "rag_error"
    assert result.events[-1].usage == Usage(0, 0, 0)
    assert not rag.reader.calls and not rag.retriever.calls


def test_two_branches_preserve_prefix_without_sharing_mutable_evidence():
    shared = prefix()
    left, right = backend((B,), initial_evidence=(A,)), backend((C,), initial_evidence=(A,))
    outputs = [
        run_outer_loop(
            Q,
            rag,
            generator=Generate(),
            assessor=insufficient,
            initial_state=shared.state,
            initial_events=shared.events,
            max_rag_calls=2,
        )
        for rag in (left, right)
    ]
    assert left.reader.calls == [(Q, (B, A))]
    assert right.reader.calls == [(Q, (C, A))]
    for output in outputs:
        assert output.state.rounds[0] is shared.state.rounds[0]
        assert output.events[: len(shared.events)] == shared.events
        assert len(output.state.rounds) == 2
        assert output.stop_reason == "rag_call_budget"
        assert len(output.component_events) == 4
    assert shared.state.observed_evidence == (A,)


def test_total_budget_already_used_does_not_invoke_policy_or_components():
    shared = prefix()
    rag, generator = backend((B,), initial_evidence=(A,)), Generate()
    result = run_outer_loop(
        Q,
        rag,
        generator=generator,
        policy=lambda state: pytest.fail("no extra routing"),
        initial_state=shared.state,
        initial_events=shared.events,
        max_rag_calls=1,
    )
    assert result.state is shared.state
    assert result.events == shared.events
    assert result.stop_reason == "rag_call_budget"
    assert not rag.retriever.calls and not generator.calls


def test_already_sufficient_shared_prefix_is_not_retrieved_again():
    shared = prefix(sufficient=True)
    rag = backend((B,), initial_evidence=(A,))
    result = run_outer_loop(
        Q,
        rag,
        initial_state=shared.state,
        initial_events=shared.events,
        policy=lambda state: pytest.fail("no routing after sufficient"),
    )
    assert result.stop_reason == "sufficient_signal"
    assert result.state is shared.state and result.events == shared.events
    assert not rag.retriever.calls


def test_repeated_prefix_query_stops_before_new_retrieval_but_keeps_rewrite_cost():
    shared = prefix()
    rag = backend((B,), initial_evidence=(A,))
    result = run_outer_loop(
        Q,
        rag,
        initial_state=shared.state,
        initial_events=shared.events,
        generator=Generate(Q.text.upper()),
        assessor=insufficient,
    )
    assert result.stop_reason == "repeated_query"
    assert not rag.retriever.calls
    assert result.events[-1].operation == "rewrite"
    assert result.events[-1].usage == Usage(3, 2, 0)


@pytest.mark.parametrize("fault", ["question", "budget", "repeat", "observations"])
def test_invalid_prefix_fails_before_external_call(fault):
    shared = prefix()
    state, budget = shared.state, 2
    if fault == "question":
        state = replace(state, question=replace(Q, question_id="different"))
    elif fault == "budget":
        state = replace(state, rounds=state.rounds * 2)
        budget = 1
    elif fault == "repeat":
        state = replace(state, rounds=state.rounds * 2)
    else:
        state = replace(state, observed_evidence=(B,))
    rag = backend((B,), initial_evidence=(A,))
    with pytest.raises(ValueError):
        run_outer_loop(Q, rag, initial_state=state, max_rag_calls=budget)
    assert not rag.retriever.calls


def test_typed_route_result_is_counted_and_bare_policy_remains_cost_free():
    result = run_outer_loop(
        Q,
        backend((A,)),
        generator=Generate(),
        policy=lambda state: response(FRESH, Usage(9, 3, 0)),
    )
    assert [event.operation for event in result.events] == ["route", "rewrite", "rag"]
    assert result.events[0].usage == Usage(9, 3, 0)


def test_route_api_failure_preserves_existing_prefix_and_has_no_retry():
    shared = prefix()
    attempts = []

    def broken(state):
        attempts.append(state)
        raise BackendCallError("synthetic route error", usage=Usage(9, 0, 0))

    result = run_outer_loop(
        Q,
        backend((B,), initial_evidence=(A,)),
        policy=broken,
        initial_state=shared.state,
        initial_events=shared.events,
    )
    assert result.stop_reason == "route_error"
    assert result.state is shared.state and len(attempts) == 1
    assert result.events[-1].usage == Usage(9, 0, 0)
    assert result.events[-1].status == "error"


def test_malformed_audited_route_preserves_reported_cost():
    result = run_outer_loop(
        Q, backend((A,)), policy=lambda state: response("invalid decision", Usage(9, 3, 0))
    )
    assert result.stop_reason == "route_error"
    assert result.events[0].usage == Usage(9, 3, 0)
    assert not result.state.rounds


def test_audited_route_can_choose_stop_and_retains_cost():
    result = run_outer_loop(Q, backend((A,)), policy=lambda state: response(None, Usage(9, 3, 0)))
    assert result.stop_reason == "policy_stop"
    assert result.events[0].operation == "route"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_reader_cap_is_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        backend((A,), max_evidence=value)


def test_initial_context_cannot_silently_exceed_cap_or_duplicate_ids():
    with pytest.raises(ValueError, match="exceeds"):
        backend((C,), initial_evidence=(A, B), max_evidence=1)
    with pytest.raises(ValueError, match="duplicate"):
        backend((C,), initial_evidence=(A, A))


def test_latest_and_reader_context_cannot_disagree_about_same_id():
    with pytest.raises(ValueError, match="conflicts"):
        RagReply(Answer(""), (A,), retrieved_evidence=(replace(A, text="different"),))


def test_foreign_evidence_cannot_be_injected_as_unrecorded_prefix_observations():
    state = LoopState(Q, (), (A,))
    with pytest.raises(ValueError, match="match its recorded"):
        run_outer_loop(Q, backend((B,)), initial_state=state)


def test_reader_cap_does_not_enable_citing_dropped_evidence():
    rag = backend((B,), initial_evidence=(A,), max_evidence=1)
    rag.reader.answer = lambda question, evidence: response(Answer("claim", ("a",)))
    result = run_outer_loop(Q, rag)
    assert result.stop_reason == "rag_error"
    assert result.component_events[-1].operation == "rag.answer"


def test_initial_true_sufficiency_without_evidence_is_invalid():
    state = LoopState(
        Q, (RoundRecord(RewriteDecision(), Q.text, RagReply(Answer(""), ()), Feedback(True)),), ()
    )
    with pytest.raises(ValueError, match="nonempty answer and evidence"):
        run_outer_loop(Q, backend((B,)), initial_state=state)


def test_resuming_prefix_preserves_prior_no_gain_stop():
    original = run_outer_loop(Q, backend((A,), (A,)), generator=Generate(), assessor=insufficient)
    assert original.stop_reason == "no_new_ids"
    rag = backend((B,), initial_evidence=(A,))
    continued = run_outer_loop(
        Q,
        rag,
        initial_state=original.state,
        initial_events=original.events,
        max_rag_calls=3,
        policy=lambda state: pytest.fail("prefix already stopped"),
    )
    assert continued.stop_reason == "no_new_ids"
    assert not rag.retriever.calls


def test_partial_no_gain_patience_is_not_reset_by_prefix_resume():
    original = run_outer_loop(
        Q,
        backend((A,), (A,)),
        generator=Generate(),
        assessor=insufficient,
        no_gain_patience=2,
    )
    assert original.stop_reason == "rag_call_budget"
    continued = run_outer_loop(
        Q,
        backend((A,), initial_evidence=(A,)),
        initial_state=original.state,
        initial_events=original.events,
        generator=Generate("third alternate query"),
        assessor=insufficient,
        max_rag_calls=3,
        no_gain_patience=2,
    )
    assert continued.stop_reason == "no_new_ids"
    assert len(continued.state.rounds) == 3
