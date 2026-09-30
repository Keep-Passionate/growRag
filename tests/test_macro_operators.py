"""Pure local planner tests; no retrieval, model calls, gold or quality claims."""

from dataclasses import FrozenInstanceError, replace

import pytest

from growrag.experiments.protocol import Evidence
from growrag.macro_operators import (
    FieldEquals,
    GapField,
    GoalContract,
    GroundedBinding,
    OperatorRegistry,
    OperatorSpec,
    QueryStep,
    RuntimeState,
    seed_registry,
)

COMPARE = GoalContract(
    "Which university opened earlier, Alpha or Beta?", "compare", ("before 2000",)
)
COMPARE_GAP = {
    "left": "Alpha University",
    "right": "Beta University",
    "attribute": "opening year",
    "left_missing": False,
    "right_missing": True,
}
BRIDGE = GoalContract("Where was the director of Blue Lake born?", "bridge")
BRIDGE_GAP = {
    "anchor": "Blue Lake",
    "bridge_relation": "director",
    "target_relation": "birthplace",
    "bridge_missing": True,
}
EVIDENCE = Evidence("current-1", "Blue Lake", 0, "Blue Lake was directed by Jane Doe.")
BINDING = GroundedBinding("bridge", "Jane Doe", ("current-1",))


def test_compare_only_missing_side_keeps_original_goal_separate():
    plan = seed_registry().plan(
        "COMPARE_ALIGNED",
        "1",
        goal=COMPARE,
        gap=COMPARE_GAP,
        state=RuntimeState(),
    )
    assert plan.goal is COMPARE
    assert plan.retrieval_cost == 1
    assert plan.requests[0].query == "Beta University opening year before 2000"
    assert "Alpha" not in plan.requests[0].query


def test_no_missing_side_plans_no_search_with_zero_budget():
    plan = seed_registry().plan(
        "COMPARE_ALIGNED",
        "1",
        goal=COMPARE,
        gap={**COMPARE_GAP, "right_missing": False},
        state=RuntimeState(remaining_retrievals=0),
    )
    assert plan.requests == ()
    assert plan.retrieval_cost == 0


def test_two_missing_sides_require_two_retrievals_and_never_silently_truncate():
    registry = seed_registry()
    gap = {**COMPARE_GAP, "left_missing": True}
    with pytest.raises(ValueError, match="budget"):
        registry.plan("COMPARE_ALIGNED", "1", goal=COMPARE, gap=gap, state=RuntimeState())
    state = RuntimeState(remaining_retrievals=2)
    plan = registry.plan("COMPARE_ALIGNED", "1", goal=COMPARE, gap=gap, state=state)
    assert plan.retrieval_cost == 2
    assert state.remaining_retrievals == 2  # Planning does not spend or execute.


def test_bridge_first_batch_cannot_invent_or_execute_downstream_entity():
    plan = seed_registry().plan(
        "BRIDGE_HOP",
        "1",
        goal=BRIDGE,
        gap=BRIDGE_GAP,
        state=RuntimeState(),
    )
    assert [(item.step_id, item.query) for item in plan.requests] == [
        ("find_bridge", "Blue Lake director"),
    ]


def test_bridge_downstream_requires_current_evidence_binding():
    registry = seed_registry()
    gap = {**BRIDGE_GAP, "bridge_missing": False}
    with pytest.raises(ValueError, match="unbound"):
        registry.plan("BRIDGE_HOP", "1", goal=BRIDGE, gap=gap, state=RuntimeState())
    plan = registry.plan(
        "BRIDGE_HOP",
        "1",
        goal=BRIDGE,
        gap=gap,
        state=RuntimeState((EVIDENCE,), (BINDING,)),
    )
    assert plan.requests[0].query == "Jane Doe birthplace"
    assert plan.requests[0].evidence_ids == ("current-1",)


def test_binding_rejects_foreign_evidence_and_unsupported_lexical_value():
    with pytest.raises(ValueError, match="outside"):
        RuntimeState((), (BINDING,))
    with pytest.raises(ValueError, match="lexical"):
        RuntimeState((EVIDENCE,), (replace(BINDING, value="Unseen Person"),))
    with pytest.raises(ValueError, match="lexical"):
        RuntimeState((EVIDENCE,), (replace(BINDING, value="Jan"),))
    with pytest.raises(ValueError, match="outside"):
        RuntimeState((EVIDENCE,), (replace(BINDING, evidence_ids=("current-1", "foreign")),))


def test_lexical_support_normalizes_case_whitespace_but_is_not_semantic_entailment():
    denied = replace(EVIDENCE, text="Jane Doe did not direct Blue Lake.")
    state = RuntimeState((denied,), (replace(BINDING, value="JANE   DOE"),))
    assert state.bindings[0].value == "JANE   DOE"  # Deliberately not a truth validator.


def test_fourth_operator_new_name_and_gap_fields_execute_without_dispatch_changes():
    registry = seed_registry()
    spec = OperatorSpec(
        "ARCHIVE_YEAR_LOOKUP",
        "2026.1",
        ("archive",),
        (GapField("archive_name"), GapField("year", "integer"), GapField("needed", "bool")),
        (QueryStep("archive", "{archive_name} issue {year}", (FieldEquals("needed", True),)),),
    )
    registry.register(spec)
    plan = registry.plan(
        "ARCHIVE_YEAR_LOOKUP",
        "2026.1",
        goal=GoalContract("Find issue", "archive"),
        gap={"archive_name": "Local Review", "year": 1920, "needed": True},
        state=RuntimeState(),
    )
    assert plan.requests[0].query == "Local Review issue 1920"


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"unknown": "x"}, ValueError),
        ({"right_missing": "true"}, TypeError),
        ({"right_missing": 1}, TypeError),
        ({"left": []}, TypeError),
        ({"attribute": ""}, ValueError),
    ],
)
def test_gap_is_closed_and_strictly_typed(changes, error):
    with pytest.raises(error):
        seed_registry().plan(
            "COMPARE_ALIGNED",
            "1",
            goal=COMPARE,
            gap={**COMPARE_GAP, **changes},
            state=RuntimeState(),
        )


def test_missing_required_field_and_goal_mismatch_are_rejected():
    registry = seed_registry()
    with pytest.raises(ValueError, match="required"):
        registry.plan("COMPARE_ALIGNED", "1", goal=COMPARE, gap={}, state=RuntimeState())
    with pytest.raises(ValueError, match="intent"):
        registry.plan("COMPARE_ALIGNED", "1", goal=BRIDGE, gap=COMPARE_GAP, state=RuntimeState())


def test_versions_are_explicit_and_cannot_be_overwritten():
    registry = OperatorRegistry()
    first = OperatorSpec("CUSTOM", "1", ("lookup",), (), (QueryStep("one", "first query"),))
    registry.register(first)
    with pytest.raises(ValueError, match="overwritten"):
        registry.register(first)
    registry.register(replace(first, version="2", steps=(QueryStep("two", "second query"),)))
    inputs = {"goal": GoalContract("Find it", "lookup"), "gap": {}, "state": RuntimeState()}
    assert registry.plan("CUSTOM", "1", **inputs).requests[0].query == "first query"
    assert registry.plan("CUSTOM", "2", **inputs).requests[0].query == "second query"
    with pytest.raises(KeyError):
        registry.plan("CUSTOM", "latest", **inputs)
    with pytest.raises(FrozenInstanceError):
        first.version = "3"


@pytest.mark.parametrize(
    "template",
    [
        "{original_question.__class__}",
        "{original_question[0]}",
        "{original_question!r}",
        "{original_question:20}",
        "{original_question:}",
        "{__import__('os')}",
        "{}",
        "{broken",
    ],
)
def test_template_expressions_attribute_access_and_formatting_are_rejected(template):
    with pytest.raises(ValueError):
        QueryStep("bad", template)


def test_gap_text_is_not_recursively_interpreted_as_a_template():
    plan = seed_registry().plan(
        "CONCAT_AUGMENT",
        "1",
        goal=GoalContract("Find this", "lookup"),
        gap={"search_terms": "{original_question.__class__}"},
        state=RuntimeState(),
    )
    assert plan.requests[0].query == "Find this {original_question.__class__}"


def test_undeclared_names_conditions_and_gap_binding_collisions_are_rejected():
    with pytest.raises(ValueError, match="undeclared"):
        OperatorSpec("X", "1", ("x",), (), (QueryStep("a", "{unknown}"),))
    with pytest.raises(ValueError, match="undeclared"):
        OperatorSpec("X", "1", ("x",), (), (QueryStep("a", "query", (FieldEquals("x", True),)),))
    with pytest.raises(TypeError, match="declared type"):
        OperatorSpec(
            "X",
            "1",
            ("x",),
            (GapField("x", "bool"),),
            (QueryStep("a", "query", (FieldEquals("x", 1),)),),
        )
    with pytest.raises(ValueError, match="impersonate"):
        OperatorSpec(
            "X",
            "1",
            ("x",),
            (GapField("bridge"),),
            (QueryStep("a", "{bridge}", requires_bindings=("bridge",)),),
        )


def test_optional_fields_are_checked_only_when_selected_step_uses_them():
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec(
            "OPTIONAL",
            "1",
            ("x",),
            (GapField("value", required=False), GapField("run", "bool")),
            (QueryStep("optional", "{value}", (FieldEquals("run", True),)),),
        )
    )
    inputs = {"goal": GoalContract("Question", "x"), "state": RuntimeState()}
    assert registry.plan("OPTIONAL", "1", gap={"run": False}, **inputs).requests == ()
    with pytest.raises(ValueError, match="optional"):
        registry.plan("OPTIONAL", "1", gap={"run": True}, **inputs)


@pytest.mark.parametrize("budget", [True, -1, 1.5])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError):
        RuntimeState(remaining_retrievals=budget)


def test_mutable_schema_and_duplicate_ids_are_rejected():
    with pytest.raises(TypeError, match="immutable"):
        OperatorSpec("X", "1", ("x",), [], (QueryStep("a", "query"),))
    with pytest.raises(ValueError, match="duplicate"):
        RuntimeState((EVIDENCE, EVIDENCE))
    with pytest.raises(ValueError, match="duplicate"):
        RuntimeState((EVIDENCE,), (BINDING, BINDING))


def test_gap_fields_cannot_shadow_goal_or_current_evidence_bindings():
    with pytest.raises(ValueError, match="reserved"):
        GapField("original_question")
    with pytest.raises(ValueError, match="reserved"):
        GroundedBinding("constraints", "Jane Doe", ("current-1",))
    with pytest.raises(ValueError, match="unknown"):
        seed_registry().plan(
            "BRIDGE_HOP",
            "1",
            goal=BRIDGE,
            gap={**BRIDGE_GAP, "bridge": "Unverified Person"},
            state=RuntimeState(),
        )


def test_integer_field_does_not_accept_boolean():
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec(
            "INTEGER",
            "1",
            ("lookup",),
            (GapField("year", "integer"),),
            (QueryStep("year", "archive {year}"),),
        )
    )
    with pytest.raises(TypeError, match="integer"):
        registry.plan(
            "INTEGER",
            "1",
            goal=GoalContract("Find year", "lookup"),
            gap={"year": True},
            state=RuntimeState(),
        )


def test_rendered_query_length_is_bounded():
    with pytest.raises(ValueError, match="compiled query"):
        seed_registry().plan(
            "CONCAT_AUGMENT",
            "1",
            goal=GoalContract("q" * 1500, "lookup"),
            gap={"search_terms": "z" * 1500},
            state=RuntimeState(),
        )
