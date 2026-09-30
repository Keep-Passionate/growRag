"""Offline schema contracts; these tests do not claim provider compatibility."""

from copy import deepcopy

import pytest

from growrag.experiments.operator_schemas import (
    PLANNER_VERSIONS,
    READER_VERSION,
    REASON_CODES,
    operator_schema_registry,
    planner_schema,
    reader_schema,
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


def test_only_new_exact_versions_are_registered():
    registry = operator_schema_registry()
    assert set(registry) == {
        "growrag-operator-fresh-v3",
        "growrag-operator-static-v3",
        "growrag-operator-memory-v3",
        "growrag-operator-reader-v2",
    }
    assert set(registry) == set(PLANNER_VERSIONS.values()) | {READER_VERSION}
    for name, schema in registry.values():
        assert "-" not in name
        _check(schema)


@pytest.mark.parametrize("mode", ["fresh", "static", "memory"])
def test_wire_has_no_open_gap_object_or_free_reason(mode):
    schema = planner_schema(mode)
    props = schema["properties"]
    assert set(props) == {
        "decision",
        "reason",
        "intent",
        "constraints",
        "selected_operator",
        "operator",
        "gap_entries",
        "bindings",
    }
    assert props["reason"] == {"type": "string", "enum": list(REASON_CODES)}
    entries = props["gap_entries"]["items"]
    assert entries["required"] == ["name", "value"]
    assert entries["properties"]["value"] == {"type": ["string", "integer", "boolean"]}
    assert "gap" not in props


def test_static_cannot_create_and_fresh_cannot_select_at_shape_level():
    assert planner_schema("static")["properties"]["operator"] == {"type": "null"}
    assert planner_schema("fresh")["properties"]["selected_operator"] == {"type": "null"}
    for name in ("operator", "selected_operator"):
        assert planner_schema("memory")["properties"][name]["type"] == ["object", "null"]
    assert (
        planner_schema("fresh")["properties"]["operator"]
        == (planner_schema("memory")["properties"]["operator"])
    )


@pytest.mark.parametrize("mode", ["", "v1", None, 1, []])
def test_unknown_mode_has_no_schema_fallback(mode):
    with pytest.raises(ValueError):
        planner_schema(mode)


def test_returned_registry_and_schemas_are_independent_copies():
    expected = operator_schema_registry()
    changed = operator_schema_registry()
    changed[PLANNER_VERSIONS["memory"]][1]["properties"]["reason"]["enum"].clear()
    assert operator_schema_registry() == expected
    copy = planner_schema("fresh")
    copy["properties"].clear()
    assert planner_schema("fresh")["properties"]


@pytest.mark.parametrize("value", [0, 3, -1, True, False, "0", "False", "", "实体"])
def test_dynamic_gap_wire_value_preserves_all_three_basic_types(value):
    schema = planner_schema("fresh")["properties"]["gap_entries"]["items"]["properties"]["value"]
    validate_wire_shape(value, schema)


@pytest.mark.parametrize("value", [None, 1.0, 2.5, [], {}, float("nan")])
def test_dynamic_gap_value_rejects_null_float_array_object(value):
    schema = planner_schema("fresh")["properties"]["gap_entries"]["items"]["properties"]["value"]
    with pytest.raises(ValueError):
        validate_wire_shape(value, schema)


@pytest.mark.parametrize("value", [True, False, 1.0, "1"])
def test_wire_integer_is_not_coerced(value):
    with pytest.raises(ValueError):
        validate_wire_shape(value, {"type": "integer"})


def test_nullable_object_still_closes_nested_fields():
    schema = planner_schema("memory")["properties"]["selected_operator"]
    validate_wire_shape(None, schema)
    validate_wire_shape({"operator_id": "x", "version": "1"}, schema)
    for value in ({}, {"operator_id": "x", "version": "1", "answer": "leak"}):
        with pytest.raises(ValueError):
            validate_wire_shape(value, schema)


def test_reader_shape_unchanged_but_new_exact_version():
    schema = reader_schema()
    good = {"answer": "Jane Doe", "supported": True, "evidence_ids": ["e1"]}
    validate_wire_shape(good, schema)
    for patch in ({"supported": 1}, {"gold": "secret"}, {"evidence_ids": "e1"}):
        with pytest.raises(ValueError):
            validate_wire_shape({**deepcopy(good), **patch}, schema)
