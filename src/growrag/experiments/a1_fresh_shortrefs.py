"""A1-only binding-reference transport composed with the unchanged A0-v2 planner.

The original planner computes full-ID evidence windows first. This client exposes
only local E labels and restores binding references by an exact lookup before the
old compiler sees them. It does not repair prior outputs, choose a FRESH variant,
enforce episode budgets, change query guidance, or perform retrieval itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy

from . import a0_runtime_v2 as a0
from . import operator_model as legacy
from .a0_runtime import A0LocalOutputError, _request_payload
from .operator_model_v2 import _ContentView

PLANNER_VERSIONS = {
    variant: f"growrag-a1-fresh-{variant}-binding-shortrefs-v1"
    for variant in ("original", "focused")
}
_SOURCE_VERSIONS = {
    "original": a0.FRESH_PLANNER_VERSION,
    "focused": a0.FOCUSED_PLANNER_VERSION,
}
_REFERENCE = re.compile(r"E[1-9][0-9]*\Z")
_PAYLOAD_KEYS = {
    "mode",
    "original_question",
    "evidence",
    "evidence_window_omitted_count",
    "previous_queries",
    "remaining_retrievals",
    "candidate_specs",
    "candidate_shortlist_omitted_count",
}
_BINDING_WIRE = """
A1 binding-reference wire protocol (same planning goal and operator grammar):
Displayed evidence_id values E1, E2, ... are local labels for THIS request only.
Each binding.evidence_ids must contain one or more exact displayed labels, without
duplicates within that binding. The same label may support different bindings.
Never return full corpus IDs, titles, e1, E01, [E1], spaces, or invented labels.
Do not guess, repair, or reuse a label from another request. Labels identify sources,
not entity values: each binding.value must still occur in its cited CURRENT evidence.
Return the unchanged planner JSON shape; do not add a mapping, hash, or other keys.
"""


def _variant(variant):
    if type(variant) is not str or variant not in PLANNER_VERSIONS:
        raise ValueError("A1 FRESH variant must be explicit original or focused")
    return variant


def _source_prompt(variant):
    return (
        a0.focused_planner_prompt() if _variant(variant) == "focused" else a0.fresh_planner_prompt()
    )


def planner_prompt(variant):
    """Only reference labels and their declared wire contract differ from A0-v2."""
    prompt = _source_prompt(variant)
    if variant == "focused":
        # This is fixed prompt authoring, not a repair of model-generated content.
        for old, new in (
            ("ID current-1 says", "ID E1 says"),
            ('"evidence_ids":["current-1"]', '"evidence_ids":["E1"]'),
        ):
            if prompt.count(old) != 1:
                raise ValueError("frozen focused example reference changed")
            prompt = prompt.replace(old, new, 1)
    return prompt + _BINDING_WIRE


def schema_for(version):
    """Independent version dispatch; never mutate the provider schema registry."""
    if version not in PLANNER_VERSIONS.values():
        raise ValueError("prompt not registered in A1 binding-reference transport")
    return a0.planner_schema()


def _fingerprint(value):
    wire = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


def _check_aliases(aliases):
    if type(aliases) is not dict or set(aliases) != {
        f"E{number}" for number in range(1, len(aliases) + 1)
    }:
        raise ValueError("binding aliases must be canonical and consecutive")
    ids = list(aliases.values())
    if any(type(identity) is not str or not identity.strip() for identity in ids):
        raise ValueError("binding aliases need nonempty original evidence IDs")
    if len(set(ids)) != len(ids):
        raise ValueError("binding aliases must be bijective")


def prepare_binding_request(messages, *, prompt_version, variant):
    """Alias already-windowed v2 rows; no text/window calculation happens here."""
    variant = _variant(variant)
    payload = _request_payload(messages)
    if (
        prompt_version != _SOURCE_VERSIONS[variant]
        or messages[0]["content"] != _source_prompt(variant)
        or set(payload) != _PAYLOAD_KEYS
        or payload["mode"] != "fresh"
        or payload["candidate_specs"] != []
        or payload["candidate_shortlist_omitted_count"] != 0
    ):
        raise ValueError("expected unchanged A0-v2 FRESH request without history")
    rows = payload["evidence"]
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise ValueError("binding evidence must be already-windowed rows")
    aliases = {f"E{number}": row.get("evidence_id") for number, row in enumerate(rows, 1)}
    _check_aliases(aliases)
    wire_payload = deepcopy(payload)
    for label, row in zip(aliases, wire_payload["evidence"], strict=True):
        row["evidence_id"] = label
    # Recheck the actual A1 prompt budget; shorter IDs never release more body text.
    return legacy._messages(planner_prompt(variant), wire_payload), aliases


def restore_binding_refs(raw_content, aliases):
    """Strict v2 wire validation, then only exact binding-reference substitution."""
    _check_aliases(aliases)
    _, decoded = a0.decode_fresh_wire(raw_content)
    restored = deepcopy(decoded["raw_wire"])
    resolutions = []
    for action_number, action in enumerate(restored["actions"]):
        for binding_number, binding in enumerate(action["bindings"]):
            refs = binding["evidence_ids"]
            if not refs or any(_REFERENCE.fullmatch(ref) is None for ref in refs):
                raise ValueError("binding citations require nonempty canonical short references")
            if len(set(refs)) != len(refs):
                raise ValueError("binding repeats a short reference")
            if not set(refs) <= aliases.keys():
                raise ValueError("binding cites an unknown short reference")
            binding["evidence_ids"] = [aliases[ref] for ref in refs]
            resolutions.append(
                {
                    "path": f"actions[{action_number}].bindings[{binding_number}].evidence_ids",
                    "operation": "exact_binding_reference_lookup",
                    "wire_value": list(refs),
                    "compiler_value": list(binding["evidence_ids"]),
                }
            )
    return restored, resolutions


class _BindingShortrefClient:
    """Planner-only adapter over the accounted raw transport, not an A0 contract client."""

    def __init__(self, transport, *, variant, on_record):
        self.variant = _variant(variant)
        if not transport.config.json_object_mode or transport.config.json_schema_mode:
            raise ValueError("A1 binding transport requires explicit JSON-object mode")
        self.transport, self.on_record = transport, on_record

    def complete(self, messages, *, trace_id, prompt_version):
        wire_messages, aliases = prepare_binding_request(
            messages, prompt_version=prompt_version, variant=self.variant
        )
        version = PLANNER_VERSIONS[self.variant]
        audit = {
            "trace_id": trace_id,
            "prompt_version": version,
            "source_prompt_version": prompt_version,
            "alias_to_evidence_id": aliases,
            "wire_request_sha256": _fingerprint(wire_messages),
            "alias_mapping_sha256": _fingerprint(aliases),
        }
        self.on_record(
            deepcopy({"stage": "binding_wire_request", **audit, "messages": wire_messages})
        )
        response = self.transport.complete(wire_messages, trace_id=trace_id, prompt_version=version)
        self.on_record(deepcopy({"stage": "raw_wire", **audit, "raw_content": response.content}))
        try:
            restored, resolutions = restore_binding_refs(response.content, aliases)
        except (ValueError, TypeError, KeyError) as error:
            self.on_record({"kind": "local_contract_failure", "scope": "arm", **deepcopy(audit)})
            raise A0LocalOutputError(str(error)) from error
        self.on_record(
            deepcopy(
                {
                    "stage": "binding_refs_restored",
                    **audit,
                    "restored_wire": restored,
                    "binding_reference_resolutions": resolutions,
                }
            )
        )
        self.on_record({"kind": "local_contract_pass", **deepcopy(audit)})
        return _ContentView(
            response, json.dumps(restored, ensure_ascii=False, separators=(",", ":"))
        )


class A1FreshPlanner:
    """Same v2 planning/semantic checks with an explicit variant and shortref client.

    Pass the accounted raw transport (the same delegate used by an A0ContractClient
    for the shared Reader). The caller owns one-decision/one-extra-search budgets
    and any failure policy. No default variant is selected here.
    """

    def __init__(self, transport, *, variant, trace_prefix, on_record=None):
        variant = _variant(variant)
        if on_record is not None and not callable(on_record):
            raise TypeError("on_record must be callable")
        record = on_record or (lambda event: None)
        version = PLANNER_VERSIONS[variant]

        def compiler_record(event):
            event = deepcopy(event)
            event["source_prompt_version"] = event["prompt_version"]
            event["prompt_version"] = version
            event["representation"] = "full_id_compiler_view_not_raw_model_output"
            if event["stage"] == "raw_wire":
                event["stage"] = "compiler_wire_input"
                event["compiler_content"] = event.pop("raw_content")
            elif event["stage"] == "wire_normalization":
                event["compiler_wire_input"] = event.pop("raw_wire")
            record(event)

        self._planner = a0.A0FreshPlanner(
            _BindingShortrefClient(transport, variant=variant, on_record=record),
            focused=variant == "focused",
            trace_prefix=trace_prefix,
            on_record=compiler_record,
        )

    def __call__(self, state):
        return self._planner(state)
