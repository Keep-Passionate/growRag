"""FILL-v2 compatibility wording, isolated from the frozen history-v1 method.

Only the selected-template argument request changes: explicit binding/evidence
allowlists supplement its prompt. Selection, DSL, models, rule rewriting, reader,
and budget client are unchanged. Invalid output is rejected, never cleaned up.
中文：不同题上的新旧运行不是提示词提升对照；本版本首先验证工程兼容。
"""

from __future__ import annotations

from growrag.operator_bank import operator_from_dict

from .history_context import _bounded_messages
from .history_runtime import FILL_PROMPT as V1_FILL_PROMPT
from .history_runtime import FILL_VERSION as V1_FILL_VERSION
from .history_runtime import HistoryContractClient, HistoryPlanner, schema_for
from .operator_model import strict_object
from .operator_schemas import validate_wire_shape

FILL_VERSION = "growrag-history-base-fill-v2"
FILL_PROMPT = (
    V1_FILL_PROMPT
    + """
Binding scope is explicit in allowed_binding_names. These are the ONLY permitted
binding names, derived from the selected template's requires_bindings fields.
If allowed_binding_names is [], return bindings: [] exactly. Do not introduce
city, date, person, or other bindings merely because the question mentions them.
Declared gap fields belong ONLY in gap_entries; do not duplicate them in bindings.
For required bindings, evidence_ids may contain ONLY allowed_evidence_ids, which
list the provided CURRENT visible evidence. original_question is a field, NOT an
EvidenceID, and MUST NEVER occur in evidence_ids. The question may supply gap
values but cannot be cited as documentary evidence for an intermediate binding.
Use only current inputs; historical examples provide a pattern, not current facts.
Keep the original output keys and types. Do not add allowlists to the response.
"""
)


def schema_for_v2(version):
    """Exact wire versions for v2 scoring; the FILL response shape is unchanged."""
    if version == FILL_VERSION:
        return schema_for(V1_FILL_VERSION)
    if version == V1_FILL_VERSION:
        raise ValueError("history-v2 wire output cannot use the old FILL version")
    return schema_for(version)


def _fill_request(messages):
    if (
        type(messages) is not list
        or len(messages) != 2
        or any(type(item) is not dict or set(item) != {"role", "content"} for item in messages)
        or messages[0] != {"role": "system", "content": V1_FILL_PROMPT}
        or messages[1]["role"] != "user"
        or type(messages[1]["content"]) is not str
    ):
        raise ValueError("unexpected original FILL request; no request sent")
    payload = strict_object(messages[1]["content"])
    if set(payload) != {
        "original_question",
        "evidence",
        "previous_queries",
        "remaining_retrievals",
        "selected_card",
    }:
        raise ValueError("unexpected original FILL payload fields")
    card = payload["selected_card"]
    if type(card) is not dict or card.get("action_kind") != "template":
        raise ValueError("FILL requires an unchanged selected template")
    spec = operator_from_dict(card.get("operator_spec"))
    names = sorted({name for step in spec.steps for name in step.requires_bindings})
    evidence = payload["evidence"]
    if type(evidence) is not list or any(
        type(row) is not dict
        or type(row.get("evidence_id")) is not str
        or not row["evidence_id"].strip()
        for row in evidence
    ):
        raise ValueError("FILL evidence must be visible evidence rows")
    evidence_ids = [row["evidence_id"] for row in evidence]
    if len(set(evidence_ids)) != len(evidence_ids) or "original_question" in evidence_ids:
        raise ValueError("invalid or reserved FILL evidence identity")
    payload["allowed_binding_names"] = names
    payload["allowed_evidence_ids"] = evidence_ids
    return _bounded_messages(FILL_PROMPT, payload), names, evidence_ids


def _check_scope(value, names, evidence_ids):
    """Reject the same undeclared/ungrounded arguments; never delete or infer them."""
    gap_names = {item["name"] for item in value["gap_entries"]}
    binding_names = {item["name"] for item in value["bindings"]}
    if binding_names - set(names):
        raise ValueError("binding is not declared by the selected template")
    if gap_names & binding_names:
        raise ValueError("gap fields cannot also be returned as bindings")
    cited = {key for item in value["bindings"] for key in item["evidence_ids"]}
    if "original_question" in cited or cited - set(evidence_ids):
        raise ValueError("binding must cite only actual visible evidence IDs")


class FillWireV2Client:
    """Per-instance adapter; original client handles every non-FILL request.

    FILL uses that SAME durable delegate directly so v2 identity is present in
    its intent, HTTP audit and budget ledger. No global schema/module patching.
    """

    def __init__(self, contract_client: HistoryContractClient):
        if not isinstance(contract_client, HistoryContractClient):
            raise TypeError("FILL-v2 requires the original HistoryContractClient instance")
        self.contract_client = contract_client
        self.delegate = contract_client.delegate

    @property
    def calls(self):
        return self.contract_client.calls

    def complete(self, messages, *, trace_id, prompt_version):
        if prompt_version != V1_FILL_VERSION:
            return self.contract_client.complete(
                messages, trace_id=trace_id, prompt_version=prompt_version
            )
        wire_messages, names, evidence_ids = _fill_request(messages)
        response = self.delegate.complete(
            wire_messages, trace_id=trace_id, prompt_version=FILL_VERSION
        )
        try:
            value = strict_object(response.content)
            validate_wire_shape(value, schema_for_v2(FILL_VERSION))
            _check_scope(value, names, evidence_ids)
        except (ValueError, TypeError, KeyError):
            self.delegate.block_reason = "local_output_contract_failure"
            self.contract_client.on_record(
                {
                    "kind": "local_contract_failure",
                    "trace_id": trace_id,
                    "prompt_version": FILL_VERSION,
                }
            )
            raise
        self.contract_client.on_record(
            {"kind": "local_contract_pass", "trace_id": trace_id, "prompt_version": FILL_VERSION}
        )
        return response  # The original content, usage and HTTP status stay intact.


class HistoryPlannerV2(HistoryPlanner):
    """The original planner with only its per-instance FILL wire adapter replaced."""

    def __init__(self, client, library, *, trace_prefix, origin="reuse", on_record=None):
        super().__init__(
            FillWireV2Client(client),
            library,
            trace_prefix=trace_prefix,
            origin=origin,
            on_record=on_record,
        )
