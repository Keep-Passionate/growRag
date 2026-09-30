"""New operator wire schemas; no registration changes to historical prompts.

Only basic provider JSON Schema constructs are used. Cross-field semantics,
dynamic gap names/types, provenance and budgets remain locally checked. These
shapes constrain output syntax, never factual correctness or transfer quality.
"""

from __future__ import annotations

from copy import deepcopy

PLANNER_VERSIONS = {mode: f"growrag-operator-{mode}-v3" for mode in ("fresh", "static", "memory")}
READER_VERSION = "growrag-operator-reader-v2"
REASON_CODES = (
    "evidence_sufficient",
    "no_useful_query",
    "missing_evidence",
    "query_alignment",
    "follow_evidence_binding",
)


def _object(properties, *, nullable=False):
    return {
        "type": ["object", "null"] if nullable else "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items):
    return {"type": "array", "items": items}


def _text():
    return {"type": "string"}


def _value():
    # bool and integer remain distinct local types; no casting is permitted.
    return {"type": ["string", "integer", "boolean"]}


def _operator():
    return _object(
        {
            "operator_id": _text(),
            "version": _text(),
            "supported_intents": _array(_text()),
            "gap_schema": _array(
                _object(
                    {
                        "name": _text(),
                        "kind": {"type": "string", "enum": ["text", "bool", "integer"]},
                        "required": {"type": "boolean"},
                    }
                )
            ),
            "steps": _array(
                _object(
                    {
                        "step_id": _text(),
                        "template": _text(),
                        "when": _array(_object({"field": _text(), "value": _value()})),
                        "requires_bindings": _array(_text()),
                    }
                )
            ),
        },
        nullable=True,
    )


def planner_schema(mode: str) -> dict:
    """Fresh copies. STATIC cannot CREATE; FRESH cannot SELECT on the wire."""
    if type(mode) is not str or mode not in PLANNER_VERSIONS:
        raise ValueError("unknown operator planner mode")
    return _object(
        {
            "decision": {"type": "string", "enum": ["act", "stop"]},
            "reason": {"type": "string", "enum": list(REASON_CODES)},
            "intent": _text(),
            "constraints": _array(_text()),
            "selected_operator": {"type": "null"}
            if mode == "fresh"
            else _object({"operator_id": _text(), "version": _text()}, nullable=True),
            "operator": {"type": "null"} if mode == "static" else _operator(),
            "gap_entries": _array(_object({"name": _text(), "value": _value()})),
            "bindings": _array(
                _object({"name": _text(), "value": _text(), "evidence_ids": _array(_text())})
            ),
        }
    )


def reader_schema() -> dict:
    return _object(
        {"answer": _text(), "supported": {"type": "boolean"}, "evidence_ids": _array(_text())}
    )


def operator_schema_registry() -> dict:
    """New exact versions only; caller may explicitly add these to its registry."""
    result = {
        version: (version.replace("-", "_"), planner_schema(mode))
        for mode, version in PLANNER_VERSIONS.items()
    }
    result[READER_VERSION] = ("growrag_operator_reader_v2", reader_schema())
    return deepcopy(result)


def validate_wire_shape(value, schema: dict, *, path="output") -> None:
    """Validate our deliberately small schema subset, not arbitrary JSON Schema.

    This repeats shape validation locally even if a provider advertises strict
    output. Booleans are not accepted as integers; floats are never coerced.
    """
    allowed_types = schema["type"]
    if type(allowed_types) is str:
        allowed_types = [allowed_types]
    kinds = {str: "string", int: "integer", bool: "boolean", list: "array", dict: "object"}
    kind = "null" if value is None else kinds.get(type(value))
    if kind not in allowed_types:
        raise ValueError(f"{path}: invalid wire type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: invalid enum value")
    if kind == "object":
        properties = schema["properties"]
        if set(value) != set(properties):
            raise ValueError(f"{path}: unexpected or missing fields")
        for key, child in properties.items():
            validate_wire_shape(value[key], child, path=f"{path}.{key}")
    elif kind == "array":
        for index, item in enumerate(value):
            validate_wire_shape(item, schema["items"], path=f"{path}[{index}]")
