"""Separate structured-wire adapter; the original v1 study stays byte-for-byte intact.

The old planner still checks canonical actions, bindings and selected candidates.
Only versioned wire prompts and an exact gap_entries-to-gap translation are new.
Raw outputs are recorded separately from normalized parser inputs. Invalid output
is never repaired, truncated, inferred into another action, retried or defaulted.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

from . import operator_model as legacy
from .operator_schemas import (
    PLANNER_VERSIONS,
    READER_VERSION,
    REASON_CODES,
    planner_schema,
    reader_schema,
    validate_wire_shape,
)


def normalize_wire_plan(text: str, *, mode: str) -> dict:
    """Lossless declared wire decoding only; semantic validation follows unchanged."""
    value = legacy.strict_object(text)
    validate_wire_shape(value, planner_schema(mode))
    gap = {}
    for entry in value["gap_entries"]:
        name = entry["name"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError("gap entry names must be plain identifiers")
        if name in gap:
            raise ValueError("duplicate gap entry name")
        gap[name] = entry["value"]  # Preserve exact bool/int/string type and content.
    result = deepcopy(value)
    del result["gap_entries"]
    result["gap"] = gap
    return result


def _example(value, reason):
    result = deepcopy(value)
    result["reason"] = reason
    result["gap_entries"] = [
        {"name": name, "value": item} for name, item in result.pop("gap").items()
    ]
    return json.dumps(result, separators=(",", ":"))


def planner_prompt(mode: str) -> str:
    """Identical semantic/DSL instructions; only exact structured wire forms differ."""
    planner_schema(mode)  # Reject unknown modes before building any request.
    semantic, marker, _ = legacy.PLANNER_PROMPT.partition("Return exactly one JSON object")
    _, grammar_marker, grammar = legacy.PLANNER_PROMPT.partition("Operator specification grammar")
    if not marker or not grammar_marker:
        raise ValueError("historical planner prompt boundary changed")
    wire = """Return exactly one JSON object, no prose or reasoning trace.
Required fields: decision, reason, intent, constraints, selected_operator, operator,
gap_entries, bindings. No other fields are allowed. reason is ONLY one enum token:
""" + ", ".join(REASON_CODES)
    wire += """.
reason is an observation category, not an explanation, score or proof of sufficiency.
intent is a task label; constraints is an array of strings.
gap_entries is an array of {"name":"declared_field","value":"current_value"}.
Each name appears exactly once. Values keep their declared string/integer/boolean type;
never stringify numbers/booleans or include gold, reference answers or scoring feedback.
bindings is an array of {"name":"...","value":"...","evidence_ids":["current ID"]}.
Choose exactly one mutually exclusive form:
STOP: decision="stop", selected_operator=null, operator=null, gap_entries=[], bindings=[].
SELECT: decision="act", selected_operator={"operator_id":"listed ID","version":"listed version"},
operator=null. Fill only that candidate's declared gap fields and required bindings.
CREATE: decision="act", selected_operator=null, operator=<valid specification>.
Never combine STOP with an operator or action arguments, or SELECT with CREATE.
mode=static permits ONLY STOP or SELECT; operator MUST be null, never a copied/edited spec.
mode=fresh permits ONLY STOP or CREATE; selected_operator MUST be null.
mode=memory permits STOP, SELECT, or CREATE as a fresh fallback.
"""
    result = semantic + wire + "\n" + grammar_marker + grammar
    examples = [(legacy._STOP_EXAMPLE, "no_useful_query")]
    if mode != "fresh":
        examples.append((legacy._SELECT_EXAMPLE, "missing_evidence"))
    if mode != "static":
        examples.append((legacy._CREATE_EXAMPLE, "missing_evidence"))
    result += "\nFORMAT EXAMPLES ONLY; use current evidence, not example factual values:\n"
    result += "\n".join(_example(value, reason) for value, reason in examples)
    return result


class _ContentView:
    """Preserve original response metadata while exposing declared decoded content."""

    def __init__(self, response, content):
        self._response = response
        self.content = content

    def __getattr__(self, name):
        return getattr(self._response, name)


class _WireClient:
    def __init__(self, client, *, mode, on_record):
        self.client, self.mode, self.on_record = client, mode, on_record

    def complete(self, messages, *, trace_id, prompt_version):
        reader = self.mode is None
        expected = legacy.READER_VERSION if reader else legacy.PLANNER_VERSION
        if prompt_version != expected:
            raise ValueError("unexpected historical adapter prompt version")
        if (
            type(messages) is not list
            or len(messages) != 2
            or messages[0].get("role") != "system"
            or messages[1].get("role") != "user"
        ):
            raise ValueError("unexpected historical adapter message shape")
        payload = legacy.strict_object(messages[1]["content"])
        if not reader and payload.get("mode") != self.mode:
            raise ValueError("planner mode changed before transport")
        version = READER_VERSION if reader else PLANNER_VERSIONS[self.mode]
        system = legacy.READER_PROMPT if reader else planner_prompt(self.mode)
        wire_messages = legacy._messages(system, payload)  # Recheck actual wire byte budget.
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
        if reader:
            value = legacy.strict_object(response.content)
            validate_wire_shape(value, reader_schema())
            return response
        normalized = normalize_wire_plan(response.content, mode=self.mode)
        return _ContentView(
            response, json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
        )


class ModelOperatorPlannerV2:
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
            _WireClient(client, mode=mode, on_record=record),
            mode=mode,
            specs=specs,
            trace_prefix=trace_prefix,
            on_record=normalized,
        )

    def __call__(self, state):
        return self._planner(state)


def answer_episode_v2(client, question, evidence, *, trace_id, on_record=None):
    if on_record is not None and not callable(on_record):
        raise TypeError("on_record must be callable")
    return legacy.answer_episode(
        _WireClient(client, mode=None, on_record=on_record or (lambda event: None)),
        question,
        evidence,
        trace_id=trace_id,
    )


# Deliberate names for a separate v2 runner; no monkeypatch of historical modules.
ModelOperatorPlanner = ModelOperatorPlannerV2
answer_episode = answer_episode_v2
