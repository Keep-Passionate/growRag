"""Action-list wire schemas; old study schemas remain unchanged.

STOP has no action object in which stale parameters can be carried. Basic schema
constructs alone do not enforce at most one action or MEMORY SELECT/CREATE XOR;
both are hard local checks. The unchanged DSL still permits free text gap values:
this wire change does not establish provenance for every intermediate entity.
"""

from __future__ import annotations

from .operator_schemas import READER_VERSION, REASON_CODES, validate_wire_shape  # noqa: F401
from .operator_schemas import planner_schema as previous_planner_schema

PLANNER_VERSIONS = {mode: f"growrag-operator-{mode}-v4" for mode in ("fresh", "static", "memory")}


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def planner_schema(mode: str) -> dict:
    """Fresh shapes preserving the old DSL, with action-only parameters nested."""
    if type(mode) is not str or mode not in PLANNER_VERSIONS:
        raise ValueError("unknown action-list planner mode")
    previous = previous_planner_schema(mode)["properties"]
    action = {
        key: previous[key] for key in ("selected_operator", "operator", "gap_entries", "bindings")
    }
    if mode == "fresh":
        action["operator"]["type"] = "object"
    elif mode == "static":
        action["selected_operator"]["type"] = "object"
    return _object(
        {
            **{key: previous[key] for key in ("reason", "intent", "constraints")},
            "actions": {"type": "array", "items": _object(action)},
        }
    )


def action_list_schema_registry() -> dict:
    """Only three new planner versions; the tested reader-v2 is not re-versioned."""
    return {
        version: (version.replace("-", "_"), planner_schema(mode))
        for mode, version in PLANNER_VERSIONS.items()
    }
