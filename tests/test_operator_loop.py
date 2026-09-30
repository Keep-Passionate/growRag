import inspect
import json
from dataclasses import asdict, replace

import pytest

from growrag.experiments.protocol import Evidence, GoldRecord, RuntimeQuestion
from growrag.macro_operators import (
    FieldEquals,
    GapField,
    GoalContract,
    GroundedBinding,
    OperatorSpec,
    QueryStep,
)
from growrag.operator_loop import ActionProposal, run_operator_episode


def proposal(text="original", query="new query"):
    return ActionProposal(
        GoalContract(text, "lookup"),
        OperatorSpec(
            "NEW", "1", ("lookup",), (GapField("terms"),), (QueryStep("search", "{terms}"),)
        ),
        {"terms": query},
    )


QUESTION = RuntimeQuestion("q1", "original")


def retrieve(query, k):
    return (Evidence(query, query, 0, "Current document " + query),)


def test_budget_is_actual_query_calls_and_state_carries_evidence():
    seen = []

    def decide(state):
        seen.append(state)
        return proposal(query=f"next {state.decision_number}")

    result = run_operator_episode(QUESTION, retrieve, decide)
    assert result.retrieval_calls == 3
    assert [s.remaining_retrievals for s in seen] == [2, 1]
    assert [len(s.evidence) for s in seen] == [1, 2]
    assert result.stop_reason == "retrieval_budget"


def test_base_only_does_not_call_decider():
    result = run_operator_episode(QUESTION, retrieve, lambda s: pytest.fail(), retrieval_budget=1)
    assert result.retrieval_calls == 1
    assert result.stop_reason == "retrieval_budget"


def test_changed_original_question_is_rejected_without_retrieval():
    result = run_operator_episode(QUESTION, retrieve, lambda s: proposal(text="changed"))
    assert result.stop_reason == "plan_rejected"
    assert result.rejected_error == "original_question_changed"
    assert result.retrieval_calls == 1


def test_controller_stop_is_claim_not_gold_success():
    result = run_operator_episode(
        QUESTION,
        retrieve,
        lambda s: ActionProposal(GoalContract("original", "lookup"), None, origin="stop"),
    )
    assert result.stop_reason == "controller_stop_claim"
    assert result.retrieval_calls == 1


def test_invalid_gap_recorded_and_not_silently_fixed():
    result = run_operator_episode(
        QUESTION, retrieve, lambda s: replace(proposal(), gap={"unknown": "x"})
    )
    assert result.stop_reason == "plan_rejected"
    assert result.retrieval_calls == 1


def test_no_new_evidence_stops():
    result = run_operator_episode(QUESTION, lambda q, k: retrieve("same", k), lambda s: proposal())
    assert result.stop_reason == "no_new_evidence"
    assert result.retrieval_calls == 2


def test_repeated_query_with_fixed_topk_stops_without_reissuing():
    result = run_operator_episode(QUESTION, retrieve, lambda s: proposal(query="original"))
    assert result.stop_reason == "repeated_query_same_fixed_topk"
    assert result.retrieval_calls == 1


def test_failures_raise_instead_of_becoming_fake_answers():
    def broken(q, k):
        raise ConnectionError("fixture")

    with pytest.raises(ConnectionError):
        run_operator_episode(QUESTION, broken, lambda s: proposal())


def test_conflicting_evidence_identity_raises():
    def wrong(query, k):
        return (Evidence("id", "Title", 0, query),)

    with pytest.raises(ValueError, match="changed content"):
        run_operator_episode(QUESTION, wrong, lambda s: proposal())


@pytest.mark.parametrize("budget", [0, 9, True, 1.5])
def test_invalid_limits(budget):
    with pytest.raises(ValueError):
        run_operator_episode(QUESTION, retrieve, lambda s: proposal(), retrieval_budget=budget)


def test_two_requests_spend_two_budget_units():
    action = replace(
        proposal(),
        spec=OperatorSpec(
            "DUAL", "1", ("lookup",), (), (QueryStep("a", "query a"), QueryStep("b", "query b"))
        ),
        gap={},
    )
    result = run_operator_episode(QUESTION, retrieve, lambda s: action)
    assert result.retrieval_calls == 3
    assert len(result.proposals) == 1
    rejected = run_operator_episode(QUESTION, retrieve, lambda s: action, retrieval_budget=2)
    assert rejected.stop_reason == "plan_rejected"
    assert rejected.retrieval_calls == 1


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"max_decisions": 0}, "decision_limit"),
        ({"max_decisions": 0, "retrieval_budget": 1}, "retrieval_budget"),
    ],
)
def test_zero_decisions_still_has_exactly_one_initial_search(options, reason):
    result = run_operator_episode(QUESTION, retrieve, lambda s: pytest.fail(), **options)
    assert result.retrieval_calls == 1
    assert result.proposals == ()
    assert result.stop_reason == reason


def test_decision_limit_is_independent_of_remaining_search_budget():
    result = run_operator_episode(
        QUESTION,
        retrieve,
        lambda s: proposal(query=f"next {s.decision_number}"),
        retrieval_budget=8,
        max_decisions=4,
    )
    assert result.retrieval_calls == 5
    assert len(result.proposals) == 4
    assert result.stop_reason == "decision_limit"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_decisions", -1),
        ("max_decisions", 5),
        ("max_decisions", True),
        ("top_k", 0),
        ("top_k", 7),
        ("top_k", True),
    ],
)
def test_invalid_decision_and_document_limits_fail_before_retrieval(option, value):
    with pytest.raises(ValueError, match="limits"):
        run_operator_episode(
            QUESTION, lambda q, k: pytest.fail(), lambda s: pytest.fail(), **{option: value}
        )


def test_no_gold_or_training_interface_and_state_is_new_for_each_question():
    parameters = inspect.signature(run_operator_episode).parameters
    assert not any("gold" in name or "update" in name or "memory" in name for name in parameters)
    observations = []

    def stop(state):
        observations.append(state)
        return ActionProposal(GoalContract(state.question.text, "lookup"), None, origin="stop")

    first = run_operator_episode(QUESTION, retrieve, stop)
    second = run_operator_episode(RuntimeQuestion("q2", "second"), retrieve, stop)
    assert [state.question.question_id for state in observations] == ["q1", "q2"]
    assert [state.remaining_retrievals for state in observations] == [2, 2]
    assert [state.decision_number for state in observations] == [1, 1]
    assert first.evidence[0].evidence_id == "original"
    assert second.evidence[0].evidence_id == "second"
    assert len(second.searches) == 1
    with pytest.raises(TypeError, match="runtime question"):
        run_operator_episode(GoldRecord("q1", ("answer",)), retrieve, stop)


@pytest.mark.parametrize("query", ["ORIGINAL", "  original  ", "\tORIGINAL\n"])
def test_normalized_repeat_does_not_spend_another_search(query):
    result = run_operator_episode(QUESTION, retrieve, lambda s: proposal(query=query))
    assert result.stop_reason == "repeated_query_same_fixed_topk"
    assert result.retrieval_calls == 1


def test_duplicate_queries_within_batch_are_rejected_atomically():
    action = replace(
        proposal(),
        spec=OperatorSpec(
            "DUAL", "1", ("lookup",), (), (QueryStep("a", "other"), QueryStep("b", "OTHER"))
        ),
        gap={},
    )
    result = run_operator_episode(QUESTION, retrieve, lambda s: action)
    assert result.stop_reason == "repeated_query_same_fixed_topk"
    assert result.retrieval_calls == 1


def test_whole_batch_runs_before_no_new_evidence_decision():
    action = replace(
        proposal(),
        spec=OperatorSpec(
            "DUAL", "1", ("lookup",), (), (QueryStep("a", "old"), QueryStep("b", "new"))
        ),
        gap={},
    )
    result = run_operator_episode(
        QUESTION,
        lambda q, k: retrieve("original" if q == "old" else q, k),
        lambda s: action,
    )
    assert result.retrieval_calls == 3
    assert [len(s.new_evidence_ids) for s in result.searches] == [1, 0, 1]
    assert result.stop_reason == "retrieval_budget"


def test_empty_plan_does_not_claim_supported_answer():
    action = replace(
        proposal(),
        spec=OperatorSpec(
            "COND",
            "1",
            ("lookup",),
            (GapField("missing", "bool"),),
            (QueryStep("maybe", "next query", (FieldEquals("missing", True),)),),
        ),
        gap={"missing": False},
    )
    result = run_operator_episode(QUESTION, retrieve, lambda s: action)
    assert result.stop_reason == "empty_plan_not_sufficiency_proof"
    assert result.retrieval_calls == 1


@pytest.mark.parametrize(
    "bindings",
    [
        (),
        (GroundedBinding("bridge", "original", ("foreign_evidence",)),),
        (GroundedBinding("bridge", "missing entity", ("original",)),),
    ],
)
def test_missing_or_ungrounded_bridge_cannot_trigger_downstream_search(bindings):
    action = replace(
        proposal(),
        spec=OperatorSpec(
            "BRIDGE",
            "1",
            ("lookup",),
            (),
            (QueryStep("next", "{bridge} birthplace", requires_bindings=("bridge",)),),
        ),
        gap={},
        bindings=bindings,
    )
    result = run_operator_episode(QUESTION, retrieve, lambda s: action)
    assert result.stop_reason == "plan_rejected"
    assert result.retrieval_calls == 1


def test_operator_version_cannot_change_body_inside_episode():
    def decide(state):
        action = proposal(query=f"new {state.decision_number}")
        if state.decision_number == 2:
            return replace(action, spec=replace(action.spec, steps=(QueryStep("search", "other"),)))
        return action

    result = run_operator_episode(QUESTION, retrieve, decide)
    assert result.stop_reason == "plan_rejected"
    assert result.retrieval_calls == 2
    assert len(result.proposals) == 2


def test_new_explicit_version_can_change_operator_body():
    def decide(state):
        action = proposal(query="first")
        if state.decision_number == 2:
            return replace(
                action,
                spec=replace(action.spec, version="2", steps=(QueryStep("search", "second"),)),
            )
        return action

    result = run_operator_episode(QUESTION, retrieve, decide)
    assert result.stop_reason == "retrieval_budget"
    assert [p.spec.version for p in result.proposals] == ["1", "2"]


@pytest.mark.parametrize(
    "rows",
    [
        [],
        ("not evidence",),
        (Evidence("same", "title", 0, "text"),) * 2,
        tuple(Evidence(str(i), "title", 0, "text") for i in range(7)),
    ],
)
def test_retrieval_protocol_violations_are_not_fake_completed_results(rows):
    with pytest.raises((TypeError, ValueError)):
        run_operator_episode(QUESTION, lambda q, k: rows, lambda s: pytest.fail())


def test_initial_empty_retrieval_can_be_repaired_without_gold():
    result = run_operator_episode(
        QUESTION, lambda q, k: () if q == "original" else retrieve(q, k), lambda s: proposal()
    )
    assert result.retrieval_calls == 2
    assert result.searches[0].evidence_ids == ()
    assert result.evidence[0].evidence_id == "new query"
    assert result.stop_reason == "repeated_query_same_fixed_topk"


def test_decider_error_is_not_converted_to_success():
    def fail(state):
        raise ConnectionError("fixture")

    events = []
    with pytest.raises(ConnectionError):
        run_operator_episode(QUESTION, retrieve, fail, on_event=events.append)
    assert not any(event["kind"] == "operator_episode_end" for event in events)


def test_attempt_event_precedes_failed_retrieval():
    events = []

    def fail(q, k):
        raise ConnectionError("fixture")

    with pytest.raises(ConnectionError):
        run_operator_episode(QUESTION, fail, lambda s: pytest.fail(), on_event=events.append)
    assert events == [
        {"kind": "operator_search_start", "query": "original", "step": 0, "retrieval_number": 1}
    ]


def test_external_gap_and_callback_cannot_mutate_executed_proposal():
    action = proposal()
    external_gap = {"terms": "external"}
    copied = replace(action, gap=external_gap)
    external_gap["terms"] = "changed"
    assert copied.gap["terms"] == "external"

    def observer(event):
        if event["kind"] == "operator_proposal":
            event["proposal"].gap["terms"] = "poisoned"
            action.gap["terms"] = "changed by captured reference"

    result = run_operator_episode(
        QUESTION, retrieve, lambda s: action, max_decisions=1, on_event=observer
    )
    assert result.searches[1].query == "new query"
    assert result.proposals[0].gap["terms"] == "new query"
    assert json.loads(json.dumps(asdict(result)))["question_id"] == "q1"


@pytest.mark.parametrize("origin", ["fresh", "reuse", "static"])
def test_origin_is_a_trace_label_not_an_update_command(origin):
    result = run_operator_episode(
        QUESTION, retrieve, lambda s: replace(proposal(), origin=origin), max_decisions=1
    )
    assert result.proposals[0].origin == origin
    assert result.retrieval_calls == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"origin": "invalid"},
        {"reason": "x" * 2001},
        {"bindings": ("not binding",)},
        {"origin": "stop"},
        {"spec": None},
        {"spec": None, "origin": "stop", "gap": {"terms": "x"}},
    ],
)
def test_invalid_proposals_are_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        replace(proposal(), **changes)


def test_wrong_decider_result_and_noncallable_inputs_rejected():
    with pytest.raises(TypeError, match="ActionProposal"):
        run_operator_episode(QUESTION, retrieve, lambda s: {})
    with pytest.raises(TypeError, match="callable"):
        run_operator_episode(QUESTION, retrieve, None)
    with pytest.raises(TypeError, match="callable"):
        run_operator_episode(QUESTION, retrieve, lambda s: proposal(), on_event=False)
