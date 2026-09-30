"""Basic-schema shape tests; array length and XOR are intentionally local checks."""

import pytest

from growrag.experiments.operator_schemas import planner_schema as previous_schema
from growrag.experiments.operator_schemas_v3 import (
    PLANNER_VERSIONS,
    READER_VERSION,
    REASON_CODES,
    action_list_schema_registry,
    planner_schema,
    validate_wire_shape,
)


def _check(node):
    assert set(node) <= {"type", "enum", "properties", "required", "additionalProperties", "items"}
    kinds = node["type"] if type(node["type"]) is list else [node["type"]]
    assert set(kinds) <= {"object", "array", "string", "integer", "boolean", "null"}
    if "object" in kinds:
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])
        for child in node["properties"].values():
            _check(child)
    elif "array" in kinds:
        _check(node["items"])


def test_registry_contains_only_new_planners_without_reversioning_reader():
    registry = action_list_schema_registry()
    assert set(registry) == {
        "growrag-operator-fresh-v4",
        "growrag-operator-static-v4",
        "growrag-operator-memory-v4",
    }
    assert set(registry) == set(PLANNER_VERSIONS.values())
    assert READER_VERSION == "growrag-operator-reader-v2" and READER_VERSION not in registry
    for name, schema in registry.values():
        assert "-" not in name
        _check(schema)


@pytest.mark.parametrize("mode", ["fresh", "static", "memory"])
def test_stop_cannot_carry_top_level_action_fields(mode):
    schema = planner_schema(mode)
    assert schema["required"] == ["reason", "intent", "constraints", "actions"]
    assert schema["properties"]["reason"] == {"type": "string", "enum": list(REASON_CODES)}
    stop = {"reason": "no_useful_query", "intent": "lookup", "constraints": [], "actions": []}
    validate_wire_shape(stop, schema)
    for key, value in {
        "decision": "stop",
        "operator": None,
        "selected_operator": None,
        "gap_entries": [],
        "gap": {},
        "bindings": [],
    }.items():
        with pytest.raises(ValueError, match="fields"):
            validate_wire_shape({**stop, key: value}, schema)


def test_mode_specific_nonnullable_action_targets():
    fresh = planner_schema("fresh")["properties"]["actions"]["items"]["properties"]
    static = planner_schema("static")["properties"]["actions"]["items"]["properties"]
    memory = planner_schema("memory")["properties"]["actions"]["items"]["properties"]
    assert fresh["selected_operator"] == {"type": "null"}
    assert fresh["operator"]["type"] == "object"
    assert static["operator"] == {"type": "null"}
    assert static["selected_operator"]["type"] == "object"
    assert memory["operator"]["type"] == ["object", "null"]
    assert memory["selected_operator"]["type"] == ["object", "null"]


@pytest.mark.parametrize("mode", ["fresh", "static", "memory"])
def test_all_dsl_and_binding_fields_equal_previous_schema(mode):
    before = previous_schema(mode)["properties"]
    after = planner_schema(mode)["properties"]["actions"]["items"]["properties"]
    assert set(after) == {"selected_operator", "operator", "gap_entries", "bindings"}
    for key in ("gap_entries", "bindings"):
        assert after[key] == before[key]
    for key in ("selected_operator", "operator"):
        expected = before[key]
        if (mode, key) in {("fresh", "operator"), ("static", "selected_operator")}:
            expected["type"] = "object"
        assert after[key] == expected


def test_shape_alone_does_not_pretend_to_enforce_array_cardinality_or_memory_xor():
    schema = planner_schema("memory")
    null_action = {"selected_operator": None, "operator": None, "gap_entries": [], "bindings": []}
    # These shapes pass the basic provider grammar but MUST fail the local decoder.
    for actions in ([null_action], [null_action, null_action]):
        validate_wire_shape(
            {
                "reason": "missing_evidence",
                "intent": "lookup",
                "constraints": [],
                "actions": actions,
            },
            schema,
        )


@pytest.mark.parametrize("mode", ["", None, 1, [], "unknown"])
def test_unknown_mode_is_rejected(mode):
    with pytest.raises(ValueError):
        planner_schema(mode)


def test_registry_and_old_schemas_cannot_be_mutated_by_new_schema_callers():
    old = {mode: previous_schema(mode) for mode in PLANNER_VERSIONS}
    expected = action_list_schema_registry()
    changed = action_list_schema_registry()
    changed[PLANNER_VERSIONS["fresh"]][1]["properties"]["actions"]["items"]["properties"].clear()
    assert action_list_schema_registry() == expected
    assert {mode: previous_schema(mode) for mode in PLANNER_VERSIONS} == old


@pytest.mark.parametrize("value", [None, 1, {}, "stop"])
def test_actions_must_be_an_explicit_array(value):
    output = {"reason": "no_useful_query", "intent": "lookup", "constraints": [], "actions": value}
    with pytest.raises(ValueError):
        validate_wire_shape(output, planner_schema("fresh"))
