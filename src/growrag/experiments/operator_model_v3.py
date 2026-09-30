"""Explicit zero-or-one action decoding with unchanged DSL and runtime validators.

An empty actions array means STOP by protocol definition; a singleton means ACT.
No field is guessed, repaired, cast, truncated, or silently dropped. Multi-action
arrays and ambiguous MEMORY actions are rejected, never reduced to one choice.
Plain text gap entries still lack typed provenance roles; this representation
does not claim that every intermediate entity is bound to current evidence.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

from . import operator_model as legacy
from .operator_model_v2 import _ContentView, answer_episode_v2
from .operator_schemas_v3 import (
    PLANNER_VERSIONS,
    READER_VERSION,  # noqa: F401 - Public unchanged Reader version for a separate runner.
    REASON_CODES,
    planner_schema,
    validate_wire_shape,
)

_CANONICAL_KEYS = frozenset(
    {
        "decision",
        "reason",
        "intent",
        "constraints",
        "selected_operator",
        "operator",
        "gap",
        "bindings",
    }
)


def normalize_wire_plan(text: str, *, mode: str) -> dict:
    """Decode the declared wire protocol; old semantic validation still follows."""
    value = legacy.strict_object(text)
    validate_wire_shape(value, planner_schema(mode))
    if len(value["actions"]) > 1:
        raise ValueError("at most one action is allowed; never truncate an action list")
    common = {key: deepcopy(value[key]) for key in ("reason", "intent", "constraints")}
    if not value["actions"]:
        return {
            **common,
            "decision": "stop",
            "selected_operator": None,
            "operator": None,
            "gap": {},
            "bindings": [],
        }
    action = value["actions"][0]
    if (action["selected_operator"] is None) == (action["operator"] is None):
        raise ValueError("an action must contain exactly one SELECT or CREATE")
    gap = {}
    for entry in action["gap_entries"]:
        name = entry["name"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("gap entry names must be plain identifiers")
        if name in gap:
            raise ValueError("duplicate gap entry name")
        gap[name] = entry["value"]  # Exact bool/int/string values; no coercion.
    return {
        **common,
        "decision": "act",
        "selected_operator": deepcopy(action["selected_operator"]),
        "operator": deepcopy(action["operator"]),
        "gap": gap,
        "bindings": deepcopy(action["bindings"]),
    }


def encode_wire_plan(canonical: dict, *, mode: str) -> dict:
    """Exact inverse for legal canonical plans; not a repair path for model output."""
    if type(canonical) is not dict or set(canonical) != _CANONICAL_KEYS:
        raise ValueError("unexpected canonical plan fields")
    if type(canonical["gap"]) is not dict:
        raise ValueError("canonical gap must be an object")
    common = {key: deepcopy(canonical[key]) for key in ("reason", "intent", "constraints")}
    if canonical["decision"] == "stop":
        if (
            canonical["selected_operator"] is not None
            or canonical["operator"] is not None
            or canonical["gap"] != {}
            or canonical["bindings"] != []
            or type(canonical["bindings"]) is not list
        ):
            raise ValueError("canonical STOP cannot carry action parameters")
        result = {**common, "actions": []}
    elif canonical["decision"] == "act":
        action = {
            key: deepcopy(canonical[key]) for key in ("selected_operator", "operator", "bindings")
        }
        action["gap_entries"] = [
            {"name": name, "value": value} for name, value in canonical["gap"].items()
        ]
        result = {**common, "actions": [action]}
    else:
        raise ValueError("unknown canonical decision")
    normalize_wire_plan(json.dumps(result, allow_nan=False), mode=mode)
    return result


def planner_prompt(mode: str) -> str:
    """Only wire representation changes; semantic and DSL instructions stay fixed."""
    planner_schema(mode)
    semantic, marker, _ = legacy.PLANNER_PROMPT.partition("Return exactly one JSON object")
    _, grammar_marker, grammar = legacy.PLANNER_PROMPT.partition("Operator specification grammar")
    if not marker or not grammar_marker:
        raise ValueError("historical planner prompt boundary changed")
    wire = """Return exactly one JSON object, no prose or reasoning trace.
The ONLY top-level fields are reason, intent, constraints, actions. No decision field.
reason is ONLY one enum token: """ + ", ".join(REASON_CODES)
    wire += """.
reason is an observation category, not an explanation, score or proof of sufficiency.
intent is a task label; constraints is an array of strings.
STOP is represented ONLY by actions=[]. Do not carry any action parameters outside actions.
ACT is represented by actions=[<one action>]. There must never be more than one action.
Each action has exactly selected_operator, operator, gap_entries, bindings.
SELECT: selected_operator={"operator_id":"listed ID","version":"listed version"}, operator=null.
CREATE: selected_operator=null, operator=<valid specification>.
Exactly one of selected_operator/operator is non-null in every action.
Fill only the chosen operator's declared gap fields and required bindings.
gap_entries is an array of {"name":"declared_field","value":"current_value"}.
Each name appears exactly once; preserve declared string/integer/boolean types.
Never stringify numbers/booleans or include gold, reference answers or scoring feedback.
bindings is an array of {"name":"...","value":"...","evidence_ids":["current ID"]}.
mode=static permits STOP or SELECT only: an action MUST select a listed ID/version and
operator MUST be null. Never create, copy, rename or edit a specification in static mode.
mode=fresh permits STOP or CREATE only: an action MUST contain a specification and
selected_operator MUST be null. No historical candidates are available.
mode=memory permits STOP, SELECT or CREATE as a fresh fallback; never both in one action.
One action may contain multiple conditional query steps. Each query still spends one
retrieval and the entire planned batch must fit remaining_retrievals.
"""
    result = semantic + wire + "\n" + grammar_marker + grammar
    examples = [(legacy._STOP_EXAMPLE, "no_useful_query")]
    if mode != "fresh":
        examples.append((legacy._SELECT_EXAMPLE, "missing_evidence"))
    if mode != "static":
        examples.append((legacy._CREATE_EXAMPLE, "missing_evidence"))
    result += "\nFORMAT EXAMPLES ONLY; use current evidence, not example factual values:\n"
    result += "\n".join(
        json.dumps(
            encode_wire_plan({**deepcopy(value), "reason": reason}, mode=mode),
            separators=(",", ":"),
        )
        for value, reason in examples
    )
    return result


class _ActionListClient:
    def __init__(self, client, *, mode, on_record):
        self.client, self.mode, self.on_record = client, mode, on_record

    def complete(self, messages, *, trace_id, prompt_version):
        if prompt_version != legacy.PLANNER_VERSION:
            raise ValueError("unexpected historical adapter prompt version")
        if (
            type(messages) is not list
            or len(messages) != 2
            or messages[0].get("role") != "system"
            or messages[1].get("role") != "user"
        ):
            raise ValueError("unexpected historical adapter message shape")
        payload = legacy.strict_object(messages[1]["content"])
        if payload.get("mode") != self.mode:
            raise ValueError("planner mode changed before transport")
        version = PLANNER_VERSIONS[self.mode]
        wire_messages = legacy._messages(planner_prompt(self.mode), payload)
        response = self.client.complete(wire_messages, trace_id=trace_id, prompt_version=version)
        self.on_record(
            deepcopy(
                {
                    "stage": "raw_wire",
                    "prompt_version": version,
                    "payload": payload,
                    "raw_content": response.content,
                }
            )
        )
        canonical = normalize_wire_plan(response.content, mode=self.mode)
        return _ContentView(
            response, json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
        )


class ModelOperatorPlannerV3:
    def __init__(self, client, *, mode, specs=(), trace_prefix, on_record=None):
        if on_record is not None and not callable(on_record):
            raise TypeError("on_record must be callable")
        record = on_record or (lambda event: None)

        def normalized(event):
            record(
                deepcopy(
                    {
                        "stage": "normalized_parser_input",
                        "semantic_validation_complete": False,
                        **event,
                    }
                )
            )

        self._planner = legacy.ModelOperatorPlanner(
            _ActionListClient(client, mode=mode, on_record=record),
            mode=mode,
            specs=specs,
            trace_prefix=trace_prefix,
            on_record=normalized,
        )

    def __call__(self, state):
        return self._planner(state)


ModelOperatorPlanner = ModelOperatorPlannerV3
answer_episode = answer_episode_v2
