"""Offline fault injection for narrow component boundaries, never live APIs."""

from dataclasses import asdict, replace

import pytest
from test_outer_loop import E, Q, StubGenerator, StubRag, insufficient

from growrag.experiments.protocol import (
    Action,
    Answer,
    CallResult,
    ExecutionKind,
    Usage,
)
from growrag.outer_loop import Feedback, RetrieverReaderBackend, run_outer_loop
from growrag.query_actions import RewriteDecision, RewriteForm

FRESH = RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "alternate wording")
SENSITIVE_ERROR = "synthetic-private-header-do-not-log"
DEFAULT_ANSWER = Answer("test", ("e1",))


class PaidRag(StubRag):
    def run(self, request):
        return replace(super().run(request), usage=Usage(11, 7, 0))


class PaidGenerator(StubGenerator):
    def generate(self, *args, **kwargs):
        return replace(super().generate(*args, **kwargs), usage=Usage(5, 2, 0))


def crash(*args, **kwargs):
    raise ValueError(SENSITIVE_ERROR)


@pytest.mark.parametrize("component", ["rewrite", "rag", "assess"])
def test_plain_component_exception_retains_prior_calls_and_unknown_failed_usage(component):
    backend, generator = PaidRag(), PaidGenerator()
    kwargs = {"generator": generator}
    if component == "rewrite":
        generator.generate = crash
        kwargs["assessor"] = insufficient
    elif component == "rag":
        backend.run = crash
        kwargs["policy"] = lambda state: FRESH
    else:
        kwargs["assessor"] = crash
    result = run_outer_loop(Q, backend, **kwargs)
    assert (
        result.stop_reason
        == {"rewrite": "rewrite_error", "rag": "rag_error", "assess": "assessment_error"}[component]
    )
    assert result.events[-1].operation == component
    assert result.events[-1].status == "error"
    assert result.events[-1].usage == Usage()
    assert result.events[0].usage == (Usage(5, 2, 0) if component == "rag" else Usage(11, 7, 0))
    assert len(result.state.rounds) == (0 if component == "rag" else 1)
    if result.state.rounds:
        assert result.state.rounds[-1].feedback.sufficient is not True
    assert SENSITIVE_ERROR not in repr(asdict(result))


@pytest.mark.parametrize("component", ["rewrite", "rag", "assess"])
@pytest.mark.parametrize("bad_return", [None, "not an audited envelope"])
def test_missing_envelope_is_a_failure_not_a_crash_or_zero_cost(component, bad_return):
    backend, generator = PaidRag(), PaidGenerator()
    kwargs = {"generator": generator}

    def callback(*args, **kwargs):
        return bad_return

    if component == "rewrite":
        generator.generate = callback
        kwargs["policy"] = lambda state: FRESH
    elif component == "rag":
        backend.run = callback
    else:
        kwargs["assessor"] = callback
    result = run_outer_loop(Q, backend, **kwargs)
    assert result.events[-1].status == "error"
    assert result.events[-1].usage == Usage()


@pytest.mark.parametrize("component", ["rag", "assess"])
def test_invalid_return_value_preserves_its_envelope_metadata(component):
    backend = PaidRag()
    response = CallResult(
        "wrong value type",
        usage=Usage(4, 3, 0),
        request_id="fixture-response",
        audit_path="fixture-audit.json",
        transport_source="mock",
    )

    def callback(*args):
        return response

    kwargs = {}
    if component == "rag":
        backend.run = callback
    else:
        kwargs["assessor"] = callback
    result = run_outer_loop(Q, backend, **kwargs)
    error = result.events[-1]
    assert error.status == "error"
    assert error.usage == Usage(4, 3, 0)
    assert error.request_id == "fixture-response"
    assert error.audit_path == "fixture-audit.json"


def test_invalid_usage_is_unknown_while_other_audit_metadata_is_retained():
    backend = PaidRag()
    backend.run = lambda request: CallResult(
        None, usage="invalid", request_id="fixture-response", transport_source="mock"
    )
    result = run_outer_loop(Q, backend)
    assert result.stop_reason == "rag_error"
    assert result.events[0].usage == Usage()
    assert result.events[0].request_id == "fixture-response"


class Search:
    execution_kind = ExecutionKind.MOCK

    def __init__(self, value=(E,)):
        self.value, self.calls = value, 0

    def retrieve(self, query, *, top_k):
        self.calls += 1
        return CallResult(
            self.value,
            usage=Usage(10, 0, 0),
            request_id="fixture-retrieve",
            audit_path="fixture-retrieve.json",
            transport_source="mock",
        )


class Read:
    execution_kind = ExecutionKind.MOCK

    def __init__(self, value=DEFAULT_ANSWER):
        self.value, self.calls = value, 0

    def answer(self, question, evidence):
        self.calls += 1
        return CallResult(
            self.value,
            usage=Usage(3, 2, 0),
            request_id="fixture-read",
            audit_path="fixture-read.json",
            transport_source="mock",
        )


@pytest.mark.parametrize("value", [None, [E], ("wrong",), (E, E)])
def test_invalid_retrieval_value_preserves_cost_and_does_not_call_reader(value):
    search, reader = Search(value), Read()
    result = run_outer_loop(Q, RetrieverReaderBackend(search, reader))
    assert result.stop_reason == "rag_error"
    assert search.calls == 1 and reader.calls == 0
    assert result.events[0].usage == Usage(10, 0, 0)
    assert len(result.component_events) == 1
    assert result.component_events[0].status == "error"
    assert result.component_events[0].audit_path == "fixture-retrieve.json"


def test_top_k_response_violation_preserves_known_cost_without_answering():
    other = replace(E, evidence_id="e2")
    search, reader = Search((E, other)), Read()
    result = run_outer_loop(Q, RetrieverReaderBackend(search, reader, top_k=1))
    assert result.stop_reason == "rag_error"
    assert result.events[0].usage == Usage(10, 0, 0)
    assert search.calls == 1 and reader.calls == 0


@pytest.mark.parametrize("value", [None, "not an Answer", Answer("claim", ("not-retrieved",))])
def test_invalid_reader_response_or_citation_preserves_both_known_costs(value):
    search, reader = Search(), Read(value)
    result = run_outer_loop(Q, RetrieverReaderBackend(search, reader))
    assert result.stop_reason == "rag_error"
    assert result.events[0].usage == Usage(13, 2, 0)
    assert [event.status for event in result.component_events] == ["ok", "error"]
    assert [event.usage for event in result.component_events] == [Usage(10, 0, 0), Usage(3, 2, 0)]
    assert result.events[0].audit_path == "fixture-read.json"
    assert not result.state.rounds


@pytest.mark.parametrize("component", ["retrieve", "answer"])
def test_subcomponent_plain_exception_preserves_prior_detail_and_unknown_total(component):
    search, reader = Search(), Read()
    if component == "retrieve":
        search.retrieve = crash
    else:
        reader.answer = crash
    result = run_outer_loop(Q, RetrieverReaderBackend(search, reader))
    assert result.stop_reason == "rag_error"
    assert result.events[0].usage == Usage()
    assert result.component_events[-1].status == "error"
    assert result.component_events[-1].usage == Usage()
    if component == "answer":
        assert result.component_events[0].usage == Usage(10, 0, 0)
        assert result.component_events[0].audit_path == "fixture-retrieve.json"
    else:
        assert reader.calls == 0
    assert SENSITIVE_ERROR not in repr(asdict(result))


@pytest.mark.parametrize("component", ["rewrite", "rag", "assess", "retrieve", "answer"])
@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_base_exceptions_propagate_at_every_component_boundary(component, exception_type):
    def interrupted(*args, **kwargs):
        raise exception_type()

    backend, generator = PaidRag(), PaidGenerator()
    kwargs = {"generator": generator}
    if component == "rewrite":
        generator.generate = interrupted
        kwargs["policy"] = lambda state: FRESH
    elif component == "rag":
        backend.run = interrupted
    elif component == "assess":
        kwargs["assessor"] = interrupted
    else:
        search, reader = Search(), Read()
        if component == "retrieve":
            search.retrieve = interrupted
        else:
            reader.answer = interrupted
        backend = RetrieverReaderBackend(search, reader)
    with pytest.raises(exception_type):
        run_outer_loop(Q, backend, **kwargs)


def test_policy_and_sufficiency_contract_errors_still_propagate():
    with pytest.raises(ValueError, match=SENSITIVE_ERROR):
        run_outer_loop(Q, PaidRag(), policy=crash)
    with pytest.raises(ValueError, match="nonempty answer and evidence"):
        run_outer_loop(
            Q,
            RetrieverReaderBackend(Search(()), Read(Answer(""))),
            assessor=lambda state, reply: Feedback(True),
        )


def test_invalid_query_retains_existing_stop_contract_and_generation_cost():
    backend, generator = PaidRag(), PaidGenerator("")
    result = run_outer_loop(Q, backend, generator=generator, policy=lambda state: FRESH)
    assert result.stop_reason == "invalid_query"
    assert result.events[0].usage == Usage(5, 2, 0)
    assert not backend.requests
