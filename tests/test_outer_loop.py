"""Offline contract tests with explicit test doubles, NOT LLM quality results."""

import pytest

from growrag.experiments.protocol import (
    Action,
    Answer,
    BackendCallError,
    CallResult,
    Evidence,
    ExecutionKind,
    MemoryView,
    RuntimeQuestion,
    Usage,
)
from growrag.outer_loop import (
    Feedback,
    RagReply,
    RagRequest,
    RetrieverReaderBackend,
    run_outer_loop,
)
from growrag.query_actions import RewriteDecision, RewriteForm

Q = RuntimeQuestion("target", "When was Northbridge University founded?")
E = Evidence("e1", "Northbridge", 0, "A fictional test sentence.")


class StubRag:
    execution_kind = ExecutionKind.MOCK

    def __init__(self, replies=None):
        self.requests = []
        self.replies = replies or [RagReply(Answer("test", ("e1",)), (E,))]

    def run(self, request):
        self.requests.append(request)
        return CallResult(
            self.replies[min(len(self.requests) - 1, len(self.replies) - 1)],
            usage=Usage(0, 0, 0),
            transport_source="mock",
        )


class StubGenerator:
    execution_kind = ExecutionKind.MOCK

    def __init__(self, query="Northbridge University established date"):
        self.query, self.calls = query, []

    def generate(self, question, decision, **kwargs):
        self.calls.append((question, decision, kwargs))
        return CallResult(self.query, usage=Usage(0, 0, 0), transport_source="mock")


def insufficient(state, reply):
    return Feedback(False, None, "test fixture: incomplete", "synthetic_test")


def test_default_is_one_unassessed_base_call_without_a_judge():
    backend = StubRag()
    result = run_outer_loop(Q, backend)
    assert result.stop_reason == "unassessed"
    assert backend.requests == [RagRequest(Q, Q.text)]
    assert result.state.rounds[0].feedback.sufficient is None


@pytest.mark.parametrize("form", [RewriteForm.PARAPHRASE, RewriteForm.EXPAND])
def test_semantic_rewrite_can_happen_before_first_retrieval_without_a_gap(form):
    backend, generator = StubRag(), StubGenerator()
    decision = RewriteDecision(Action.FRESH, form, "vocabulary alignment")
    result = run_outer_loop(Q, backend, generator=generator, policy=lambda state: decision)
    assert generator.calls[0][2]["evidence"] == ()
    assert backend.requests[0].question == Q
    assert backend.requests[0].search_query != Q.text
    assert result.state.rounds[0].decision.form == form
    assert [event.operation for event in result.events] == ["rewrite", "rag"]


def test_result_returns_to_outer_layer_and_next_query_sees_current_observations():
    backend, generator = StubRag(), StubGenerator()
    result = run_outer_loop(Q, backend, generator=generator, assessor=insufficient)
    assert len(backend.requests) == 2
    assert generator.calls[0][2]["evidence"] == (E,)
    assert result.stop_reason == "no_new_ids"
    assert all(request.question == Q for request in backend.requests)


def test_sufficient_signal_is_reported_as_a_signal_not_a_gold_proof():
    result = run_outer_loop(
        Q, StubRag(), assessor=lambda state, reply: Feedback(True, origin="test_proxy")
    )
    assert result.stop_reason == "sufficient_signal"
    assert result.state.rounds[0].feedback.origin == "test_proxy"


def test_answer_only_backend_degrades_without_claiming_supported():
    called = []
    result = run_outer_loop(
        Q,
        StubRag([RagReply(Answer("opaque answer"))]),
        assessor=lambda state, reply: called.append(True),
    )
    assert result.stop_reason == "evidence_unavailable"
    assert result.state.rounds[0].feedback.sufficient is None
    assert not called


def test_normalized_query_repetition_stops_before_another_rag_call():
    backend = StubRag()
    result = run_outer_loop(
        Q,
        backend,
        generator=StubGenerator(Q.text.upper()),
        assessor=insufficient,
    )
    assert result.stop_reason == "repeated_query"
    assert len(backend.requests) == 1
    assert [e.operation for e in result.events] == ["rag", "assess", "rewrite"]


def test_unchanged_initial_rewrite_becomes_base_with_generation_cost_kept():
    decision = RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "wording")
    result = run_outer_loop(
        Q,
        StubRag(),
        generator=StubGenerator(Q.text.lower()),
        policy=lambda state: decision,
    )
    assert result.state.rounds[0].decision.action is Action.BASE
    assert result.state.rounds[0].search_query == Q.text
    assert [event.operation for event in result.events] == ["rewrite", "rag"]


def test_budget_one_does_not_generate_a_second_round():
    backend, generator = StubRag(), StubGenerator()
    result = run_outer_loop(
        Q,
        backend,
        generator=generator,
        assessor=insufficient,
        max_rag_calls=1,
    )
    assert result.stop_reason == "rag_call_budget"
    assert len(backend.requests) == 1
    assert generator.calls == []


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budgets_fail_before_any_call(budget):
    backend = StubRag()
    with pytest.raises(ValueError, match="positive integers"):
        run_outer_loop(Q, backend, max_rag_calls=budget)
    assert not backend.requests


def test_decomposition_is_reserved_not_silently_run_as_a_single_query():
    backend, generator = StubRag(), StubGenerator()
    result = run_outer_loop(
        Q,
        backend,
        generator=generator,
        policy=lambda state: RewriteDecision(Action.FRESH, RewriteForm.DECOMPOSE, "split"),
    )
    assert result.stop_reason == "decomposition_not_implemented"
    assert not generator.calls and not backend.requests


def test_reuse_keeps_selected_memory_out_of_rag_request():
    memory = MemoryView("m@v1", "source", "s", "procedure", "Try terminology alignment")
    decision = RewriteDecision(Action.REUSE, RewriteForm.PARAPHRASE, "align terms", memory)
    backend, generator = StubRag(), StubGenerator()
    run_outer_loop(Q, backend, generator=generator, policy=lambda state: decision)
    assert generator.calls[0][1].memory == memory
    assert set(RagRequest.__dataclass_fields__) == {"question", "search_query"}


def test_target_cannot_reuse_itself_and_no_model_call_occurs():
    memory = MemoryView("m", Q.question_id, "s", "procedure", "Do something")
    decision = RewriteDecision(Action.REUSE, RewriteForm.EXPAND, "expand", memory)
    backend, generator = StubRag(), StubGenerator()
    result = run_outer_loop(Q, backend, generator=generator, policy=lambda state: decision)
    assert result.stop_reason == "same_question_memory_rejected"
    assert not generator.calls and not backend.requests


def test_failure_has_one_attempt_and_keeps_known_usage():
    class Broken(StubRag):
        def run(self, request):
            self.requests.append(request)
            raise BackendCallError("test failure", usage=Usage(4, 0, 0))

    backend = Broken()
    result = run_outer_loop(Q, backend)
    assert result.stop_reason == "rag_error"
    assert len(backend.requests) == 1
    assert result.events[0].usage.input_tokens == 4


def test_provenance_mixing_is_rejected():
    generator = StubGenerator()
    generator.execution_kind = ExecutionKind.REAL
    with pytest.raises(ValueError, match="cannot be mixed"):
        run_outer_loop(Q, StubRag(), generator=generator)


def test_conflicting_evidence_ids_are_not_silently_overwritten():
    other = Evidence("e1", "Northbridge", 0, "Different content with a reused ID")
    backend = StubRag([RagReply(Answer(""), (E,)), RagReply(Answer(""), (other,))])
    with pytest.raises(ValueError, match="changed content"):
        run_outer_loop(Q, backend, generator=StubGenerator(), assessor=insufficient)


def test_reply_cannot_cite_previous_round_evidence_that_reader_did_not_receive():
    with pytest.raises(ValueError, match="outside"):
        RagReply(Answer("claim", ("old-id",)), (E,))


def test_adapter_separates_search_query_from_original_answering_objective():
    seen = []

    class Search:
        execution_kind = ExecutionKind.MOCK

        def retrieve(self, query, *, top_k):
            seen.append(("retrieve", query, top_k))
            return CallResult((E,), usage=Usage(0, 0, 0), transport_source="mock")

    class Read:
        execution_kind = ExecutionKind.MOCK

        def answer(self, question, evidence):
            seen.append(("read", question, evidence))
            return CallResult(
                Answer("test", ("e1",)),
                usage=Usage(0, 0, 0),
                transport_source="mock",
            )

    backend = RetrieverReaderBackend(Search(), Read())
    reply = backend.run(RagRequest(Q, "different search wording"))
    assert seen == [("retrieve", "different search wording", 4), ("read", Q, (E,))]
    assert reply.value.evidence == (E,)
    assert reply.usage.api_requests == 0


def test_reader_failure_keeps_retrieval_usage_and_both_component_audits():
    class Search:
        execution_kind = ExecutionKind.MOCK

        def retrieve(self, query, *, top_k):
            return CallResult(
                (E,),
                usage=Usage(10, 0, 0),
                audit_path="retrieve-test.json",
                transport_source="mock",
            )

    class BrokenReader:
        execution_kind = ExecutionKind.MOCK

        def answer(self, question, evidence):
            raise BackendCallError(
                "test error",
                usage=Usage(3, 0, 0),
                audit_path="reader-test.json",
                transport_source="mock",
            )

    result = run_outer_loop(Q, RetrieverReaderBackend(Search(), BrokenReader()))
    assert result.stop_reason == "rag_error"
    assert result.events[0].usage.input_tokens == 13
    assert [event.audit_path for event in result.component_events] == [
        "retrieve-test.json",
        "reader-test.json",
    ]


def test_assessor_failure_keeps_completed_round_and_unknown_sufficiency():
    def broken_assessor(state, reply):
        raise BackendCallError("test judge failure", usage=Usage(2, 0, 0))

    result = run_outer_loop(Q, StubRag(), assessor=broken_assessor)
    assert result.stop_reason == "assessment_error"
    assert len(result.state.rounds) == 1
    assert result.state.rounds[0].feedback.sufficient is None
    assert result.events[-1].operation == "assess"
    assert result.events[-1].usage.input_tokens == 2


def test_declared_real_backend_cannot_return_mock_as_an_accepted_result():
    backend = StubRag()
    backend.execution_kind = ExecutionKind.REAL
    result = run_outer_loop(Q, backend)
    assert result.stop_reason == "rag_error"
    assert result.events[0].status == "error"
    assert not result.state.rounds


def test_positive_gain_overrides_repeated_document_id_proxy():
    class DifferentQueries(StubGenerator):
        def generate(self, question, decision, **kwargs):
            self.calls.append(kwargs)
            return CallResult(
                f"new query {len(self.calls)}",
                usage=Usage(0, 0, 0),
                transport_source="mock",
            )

    result = run_outer_loop(
        Q,
        StubRag(),
        generator=DifferentQueries(),
        max_rag_calls=3,
        assessor=lambda state, reply: Feedback(False, True, origin="synthetic_test"),
    )
    assert result.stop_reason == "rag_call_budget"
    assert len(result.state.rounds) == 3
