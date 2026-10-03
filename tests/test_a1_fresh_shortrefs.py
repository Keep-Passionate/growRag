"""Synthetic A1 reference-transport tests; no credentials, corpus, gold or network."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from growrag.experiments import a0_runtime_v2 as a0
from growrag.experiments import a1_fresh_shortrefs as a1
from growrag.experiments.a0_runtime import A0LocalOutputError
from growrag.experiments.api_client import APIRequestError
from growrag.experiments.output_schemas import registry_manifest, response_format_for
from growrag.experiments.protocol import Evidence, RuntimeQuestion
from growrag.macro_operators import OperatorRegistry, RuntimeState
from growrag.operator_loop import Observation, run_operator_episode

QUESTION = RuntimeQuestion("a1-synthetic", "Where was the founder of Cedar Lab born?")
FULL_ID = "corpus/" + "a" * 64
EVIDENCE = (Evidence(FULL_ID, "Cedar Lab", 0, "Avery Finch founded Cedar Lab."),)
STOP = {"reason": "no_useful_query", "intent": "lookup", "constraints": [], "actions": []}


class SyntheticTransport:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.config = SimpleNamespace(json_object_mode=True, json_schema_mode=False)
        self.requests, self.calls, self.responses = [], [], []
        self.block_reason = None

    def complete(self, messages, *, trace_id, prompt_version):
        self.requests.append(
            {"messages": deepcopy(messages), "trace_id": trace_id, "prompt_version": prompt_version}
        )
        if not self.outputs:
            pytest.fail("unexpected retry or fallback")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            self.block_reason = "transport_failed"
            raise output
        self.calls.append(
            {"trace_id": trace_id, "prompt_version": prompt_version, "status": "completed"}
        )
        response = SimpleNamespace(
            content=output if isinstance(output, str) else json.dumps(output),
            request_id="synthetic-unpaid",
            input_tokens=17,
            output_tokens=9,
        )
        self.responses.append(response)
        return response


def action(refs=None):
    value = deepcopy(a0._FOCUSED_EXAMPLE)
    value["actions"][0]["bindings"][0]["evidence_ids"] = ["E1"] if refs is None else refs
    return value


def state(evidence=EVIDENCE):
    return Observation(QUESTION, evidence, (), 1, 1)


def compile_proposal(proposal, evidence=EVIDENCE, budget=1):
    registry = OperatorRegistry()
    registry.register(proposal.spec)
    return registry.plan(
        proposal.spec.operator_id,
        proposal.spec.version,
        goal=proposal.goal,
        gap=proposal.gap,
        state=RuntimeState(evidence, proposal.bindings, budget),
    )


@pytest.mark.parametrize("variant", ["original", "focused"])
def test_same_v2_proposal_after_exact_binding_restore_and_truthful_audit(variant):
    raw = json.dumps(action(), indent=2)
    client, events = SyntheticTransport([raw]), []
    result = a1.A1FreshPlanner(client, variant=variant, trace_prefix="a1", on_record=events.append)(
        state()
    )
    old_output = action([FULL_ID])
    old_client = SyntheticTransport([old_output])
    expected = a0.A0FreshPlanner(
        a0.A0ContractClient(old_client), focused=variant == "focused", trace_prefix="a0"
    )(state())
    assert result == expected
    assert result.bindings[0].evidence_ids == (FULL_ID,)
    assert [r.query for r in compile_proposal(result).requests] == ["Avery Finch birthplace"]
    assert result.goal.original_question == QUESTION.text
    request = client.requests[0]
    assert request["prompt_version"] == a1.PLANNER_VERSIONS[variant]
    payload = json.loads(request["messages"][1]["content"])
    original_payload = json.loads(old_client.requests[0]["messages"][1]["content"])
    assert payload == {
        **original_payload,
        "evidence": [{**original_payload["evidence"][0], "evidence_id": "E1"}],
    }
    assert FULL_ID not in json.dumps(request["messages"])
    raw_events = [event for event in events if event.get("stage") == "raw_wire"]
    assert len(raw_events) == 1 and raw_events[0]["raw_content"] == raw
    assert raw_events[0]["alias_to_evidence_id"] == {"E1": FULL_ID}
    assert raw_events[0]["wire_request_sha256"] == a1._fingerprint(request["messages"])
    restored = next(event for event in events if event.get("stage") == "binding_refs_restored")
    assert restored["restored_wire"] == old_output
    assert restored["binding_reference_resolutions"][0]["wire_value"] == ["E1"]
    assert restored["binding_reference_resolutions"][0]["compiler_value"] == [FULL_ID]
    compiler = next(event for event in events if event.get("stage") == "compiler_wire_input")
    assert "raw_content" not in compiler and json.loads(compiler["compiler_content"]) == old_output
    normalized = next(event for event in events if event.get("stage") == "wire_normalization")
    assert "raw_wire" not in normalized
    assert normalized["compiler_wire_input"] == old_output
    assert all(event["prompt_version"] == a1.PLANNER_VERSIONS[variant] for event in events)
    assert client.calls[0]["status"] == "completed" and client.block_reason is None


@pytest.mark.parametrize("variant", ["original", "focused"])
def test_long_full_ids_and_unicode_keep_exact_windows_before_aliasing(variant):
    evidence = tuple(
        Evidence(f"corpus/{number}/" + "z" * 160, "标题" * 20, number, "内容" * 2500)
        for number in range(26)
    )
    old = SyntheticTransport([STOP])
    a0.A0FreshPlanner(old, trace_prefix="old", focused=variant == "focused")(state(evidence))
    client, events = SyntheticTransport([STOP]), []
    a1.A1FreshPlanner(client, variant=variant, trace_prefix="new", on_record=events.append)(
        state(evidence)
    )
    previous = json.loads(old.requests[0]["messages"][1]["content"])
    current = json.loads(client.requests[0]["messages"][1]["content"])
    assert len(current["evidence"]) == len(previous["evidence"]) == 24
    assert (
        current["evidence_window_omitted_count"] == previous["evidence_window_omitted_count"] == 2
    )
    for number, (before, after) in enumerate(
        zip(previous["evidence"], current["evidence"], strict=True), 1
    ):
        assert after == {**before, "evidence_id": f"E{number}"}
        assert after["window"]["text_truncated"]
    aliases = events[0]["alias_to_evidence_id"]
    assert list(aliases.values()) == [row["evidence_id"] for row in previous["evidence"]]
    assert all(e.evidence_id not in aliases.values() for e in evidence[24:])


@pytest.mark.parametrize(
    "refs",
    [
        [],
        ["E2"],
        ["E1", "E1"],
        ["E01"],
        ["e1"],
        ["E 1"],
        [" E1"],
        ["E1 "],
        ["[E1]"],
        ["E0"],
        ["E١"],
        [FULL_ID],
        ["Cedar Lab"],
        ["Q.text"],
        ["A1"],
        [1],
    ],
)
def test_binding_aliases_are_strict_without_repair_or_retry(refs):
    client, events = SyntheticTransport([action(refs)]), []
    with pytest.raises(A0LocalOutputError):
        a1.A1FreshPlanner(
            client, variant="original", trace_prefix="invalid", on_record=events.append
        )(state())
    assert len(client.requests) == len(client.calls) == 1
    assert events[-1]["kind"] == "local_contract_failure"
    assert any(event.get("stage") == "raw_wire" for event in events)
    assert not any(event.get("stage") == "binding_refs_restored" for event in events)
    assert client.block_reason is None


def test_multiple_bindings_can_share_refs_and_multi_ref_order_is_preserved():
    output = action(["E2", "E1"])
    output["actions"][0]["bindings"].append(
        {"name": "organization", "value": "Cedar Lab", "evidence_ids": ["E1"]}
    )
    output["actions"][0]["operator"]["steps"][0]["requires_bindings"].append("organization")
    evidence = EVIDENCE + (
        Evidence("second-id", "Avery Finch", 1, "Avery Finch founded Cedar Lab."),
    )
    client = SyntheticTransport([output])
    proposal = a1.A1FreshPlanner(client, variant="original", trace_prefix="shared")(state(evidence))
    assert proposal.bindings[0].evidence_ids == ("second-id", FULL_ID)
    assert proposal.bindings[1].evidence_ids == (FULL_ID,)
    assert compile_proposal(proposal, evidence).requests[0].evidence_ids == ("second-id", FULL_ID)


def test_only_binding_reference_fields_change_even_when_full_ids_look_like_labels():
    output = action(["E2", "E1"])
    output["actions"][0]["bindings"][0]["value"] = "E1"
    output["actions"][0]["gap_entries"][0]["value"] = "E2"
    output["actions"][0]["operator"]["steps"][0]["template"] += " E1"
    restored, resolutions = a1.restore_binding_refs(json.dumps(output), {"E1": "E2", "E2": "E1"})
    expected = deepcopy(output)
    expected["actions"][0]["bindings"][0]["evidence_ids"] = ["E1", "E2"]
    assert restored == expected
    assert resolutions[0]["wire_value"] == ["E2", "E1"]
    assert resolutions[0]["compiler_value"] == ["E1", "E2"]


@pytest.mark.parametrize("variant", ["original", "focused"])
def test_reference_protocol_does_not_choose_or_change_query_guidance(variant):
    source = a1._source_prompt(variant)
    if variant == "focused":
        source = source.replace("ID current-1 says", "ID E1 says").replace(
            '"evidence_ids":["current-1"]', '"evidence_ids":["E1"]'
        )
    assert a1.planner_prompt(variant) == source + a1._BINDING_WIRE
    assert a1.schema_for(a1.PLANNER_VERSIONS[variant]) == a0.planner_schema()
    assert "current-1" not in a1.planner_prompt(variant)
    registry = registry_manifest()
    with pytest.raises(ValueError):
        response_format_for(a1.PLANNER_VERSIONS[variant])
    assert registry_manifest() == registry


@pytest.mark.parametrize("variant", [None, "", "best", "fresh_original", True])
def test_variant_is_explicit_and_validated_before_transport(variant):
    client = SyntheticTransport([])
    with pytest.raises(ValueError, match="explicit original or focused"):
        a1.A1FreshPlanner(client, variant=variant, trace_prefix="invalid")
    assert client.requests == []


def test_no_default_variant_unknown_versions_and_wrong_mode():
    client = SyntheticTransport([])
    with pytest.raises(TypeError):
        a1.A1FreshPlanner(client, trace_prefix="missing")
    with pytest.raises(ValueError):
        a1.schema_for(a0.FRESH_PLANNER_VERSION)
    client.config.json_schema_mode = True
    with pytest.raises(ValueError, match="JSON-object"):
        a1.A1FreshPlanner(client, variant="original", trace_prefix="schema")
    assert client.requests == []


def test_full_id_semantic_checks_still_reject_hidden_or_absent_binding_values():
    for evidence in (
        (Evidence(FULL_ID, "Source", 0, "x" * 20000 + " Avery Finch"),),
        (Evidence(FULL_ID, "Source", 0, "An unrelated sentence."),),
    ):
        client = SyntheticTransport([action()])
        with pytest.raises(A0LocalOutputError, match="absent from the visible cited window"):
            a1.A1FreshPlanner(client, variant="original", trace_prefix="hidden")(state(evidence))
        assert len(client.calls) == 1 and client.block_reason is None


def test_v2_intent_integer_conditional_optional_and_boolean_capabilities_are_preserved():
    output = action()
    entry = output["actions"][0]
    entry["operator"]["gap_schema"].extend(
        [
            {"name": "year", "kind": "integer", "required": True},
            {"name": "optional", "kind": "text", "required": False},
            {"name": "enabled", "kind": "bool", "required": True},
        ]
    )
    entry["gap_entries"].extend(
        [{"name": "year", "value": "1977"}, {"name": "enabled", "value": True}]
    )
    entry["operator"]["steps"][0]["when"] = [
        {"field": "year", "value": "1977"},
        {"field": "enabled", "value": True},
    ]
    entry["operator"]["steps"].append(
        {
            "step_id": "optional",
            "template": "{optional}",
            "when": [{"field": "optional", "value": "present"}],
            "requires_bindings": [],
        }
    )
    client, events = SyntheticTransport([output]), []
    proposal = a1.A1FreshPlanner(
        client, variant="original", trace_prefix="v2", on_record=events.append
    )(state())
    assert proposal.gap["year"] == 1977 and type(proposal.gap["year"]) is int
    assert proposal.gap["enabled"] is True and "optional" not in proposal.gap
    assert proposal.spec.supported_intents == (proposal.goal.intent,)
    assert len(compile_proposal(proposal).requests) == 1
    audit = next(event for event in events if event.get("stage") == "wire_normalization")
    assert [row["operation"] for row in audit["normalization_differences"]] == [
        "derive_from_intent",
        "canonical_integer_string",
        "canonical_integer_string",
    ]


@pytest.mark.parametrize("output", [STOP, {**STOP, "actions": []}])
def test_stop_without_evidence_needs_no_fabricated_citations(output):
    client = SyntheticTransport([output])
    proposal = a1.A1FreshPlanner(client, variant="original", trace_prefix="stop")(state(()))
    assert proposal.spec is None and proposal.origin == "stop"
    assert json.loads(client.requests[0]["messages"][1]["content"])["evidence"] == []


def test_binding_free_actions_are_legal_without_current_evidence():
    output = action()
    output["actions"][0]["bindings"] = []
    step = output["actions"][0]["operator"]["steps"][0]
    step["requires_bindings"] = []
    step["template"] = "{original_question} {property}"
    client = SyntheticTransport([output])
    proposal = a1.A1FreshPlanner(client, variant="original", trace_prefix="no-binding")(state(()))
    assert not proposal.bindings and len(compile_proposal(proposal, ()).requests) == 1


def test_mapping_is_request_local_and_callbacks_cannot_mutate_wire_or_compiler_input():
    client = SyntheticTransport([action(), action()])

    def malicious(event):
        event.get("alias_to_evidence_id", {}).clear()
        event.get("messages", []).clear()
        event.get("restored_wire", {}).clear()
        event.get("compiler_wire_input", {}).clear()
        event.get("model_output", {}).clear()

    planner = a1.A1FreshPlanner(
        client, variant="original", trace_prefix="mapping", on_record=malicious
    )
    first = planner(state())
    other = (replace(EVIDENCE[0], evidence_id="different-current-id"),)
    second = planner(state(other))
    assert first.bindings[0].evidence_ids == (FULL_ID,)
    assert second.bindings[0].evidence_ids == ("different-current-id",)
    assert len(client.requests) == 2 and client.requests[0]["messages"]


def test_response_metadata_is_preserved_and_raw_response_never_overwritten():
    original = SyntheticTransport([STOP])
    a0.A0FreshPlanner(original, trace_prefix="capture")(state())
    request = original.requests[0]
    client = SyntheticTransport([action()])
    adapter = a1._BindingShortrefClient(client, variant="original", on_record=lambda event: None)
    response = adapter.complete(**request)
    assert response.request_id == "synthetic-unpaid"
    assert response.input_tokens == 17 and response.output_tokens == 9
    assert json.loads(response.content)["actions"][0]["bindings"][0]["evidence_ids"] == [FULL_ID]
    assert json.loads(client.responses[0].content)["actions"][0]["bindings"][0]["evidence_ids"] == [
        "E1"
    ]


@pytest.mark.parametrize(
    "stage",
    [
        "binding_wire_request",
        "raw_wire",
        "binding_refs_restored",
        "compiler_wire_input",
        "wire_normalization",
        "normalized_parser_input",
    ],
)
def test_audit_failures_are_not_classified_as_bad_model_output(stage):
    client = SyntheticTransport([action()])

    def fail(event):
        if event.get("stage") == stage:
            raise ValueError("synthetic audit error")

    with pytest.raises(ValueError) as caught:
        a1.A1FreshPlanner(client, variant="original", trace_prefix="audit", on_record=fail)(state())
    assert not isinstance(caught.value, A0LocalOutputError)
    assert len(client.requests) == (0 if stage == "binding_wire_request" else 1)
    assert client.block_reason is None


def test_transport_failure_preserves_block_and_has_no_fabricated_raw_output():
    client, events = SyntheticTransport([APIRequestError("synthetic network error")]), []
    with pytest.raises(APIRequestError):
        a1.A1FreshPlanner(
            client, variant="original", trace_prefix="transport", on_record=events.append
        )(state())
    assert client.block_reason == "transport_failed" and not client.calls
    assert [event["stage"] for event in events] == ["binding_wire_request"]


@pytest.mark.parametrize("steps", [1, 2])
def test_caller_budget_two_one_decision_never_truncates_or_adds_searches(steps):
    output = action()
    if steps == 2:
        second = deepcopy(output["actions"][0]["operator"]["steps"][0])
        second.update(step_id="second", template="{bridge_entity} biography")
        output["actions"][0]["operator"]["steps"].append(second)
    client, searches = SyntheticTransport([output]), []
    planner = a1.A1FreshPlanner(client, variant="original", trace_prefix="budget")

    def retrieve(query, top_k):
        searches.append((query, top_k))
        return (
            EVIDENCE if len(searches) == 1 else (Evidence("new-id", "Source", 0, "New evidence."),)
        )

    episode = run_operator_episode(
        QUESTION, retrieve, planner, retrieval_budget=2, max_decisions=1, top_k=6
    )
    assert len(searches) == (2 if steps == 1 else 1)
    assert len(client.calls) == len(episode.proposals) == 1
    assert episode.stop_reason == ("retrieval_budget" if steps == 1 else "plan_rejected")
    payload = json.loads(client.requests[0]["messages"][1]["content"])
    assert payload["remaining_retrievals"] == 1


def test_malformed_request_and_invalid_evidence_fail_before_transport():
    capture = SyntheticTransport([STOP])
    a0.A0FreshPlanner(capture, trace_prefix="capture")(state())
    request = capture.requests[0]
    client = SyntheticTransport([])
    adapter = a1._BindingShortrefClient(client, variant="original", on_record=lambda event: None)
    for changed in (
        {**request, "prompt_version": "unknown"},
        {**request, "messages": []},
        {**request, "messages": [{"role": "system", "content": "changed"}, request["messages"][1]]},
    ):
        with pytest.raises(ValueError):
            adapter.complete(**changed)
    planner = a1.A1FreshPlanner(client, variant="original", trace_prefix="invalid-evidence")
    with pytest.raises(ValueError, match="duplicate evidence IDs"):
        planner(state(EVIDENCE * 2))
    with pytest.raises(TypeError):
        planner(object())
    assert not client.requests


@pytest.mark.parametrize(
    "aliases", [{"E01": FULL_ID}, {"E2": FULL_ID}, {"E1": ""}, {"E1": FULL_ID, "E2": FULL_ID}]
)
def test_nonbijective_or_noncanonical_mapping_rejected(aliases):
    with pytest.raises(ValueError):
        a1.restore_binding_refs(json.dumps(STOP), aliases)


def test_restoration_does_not_depend_on_serialized_mapping_key_order():
    aliases = {f"E{number}": f"full-id-{number}" for number in range(1, 13)}
    reordered = json.loads(json.dumps(aliases, sort_keys=True))
    assert list(reordered) != list(aliases)
    restored, _ = a1.restore_binding_refs(json.dumps(action(["E12", "E2"])), reordered)
    assert restored["actions"][0]["bindings"][0]["evidence_ids"] == ["full-id-12", "full-id-2"]


@pytest.mark.parametrize("failure", ["duplicate_key", "extra_key", "multi_action", "old_intents"])
def test_inherited_wire_failures_remain_accounted_local_failures(failure):
    output = action()
    if failure == "duplicate_key":
        output = '{"reason":"missing_evidence","reason":"no_useful_query"}'
    elif failure == "extra_key":
        output["alias_mapping"] = {"E1": FULL_ID}
    elif failure == "multi_action":
        output["actions"] *= 2
    else:
        output["actions"][0]["operator"]["supported_intents"] = ["bridge"]
    client, events = SyntheticTransport([output]), []
    with pytest.raises(A0LocalOutputError):
        a1.A1FreshPlanner(
            client, variant="original", trace_prefix="shape", on_record=events.append
        )(state())
    assert len(client.requests) == len(client.calls) == 1
    assert events[-1]["kind"] == "local_contract_failure" and client.block_reason is None


def test_previous_requests_labels_are_not_accepted_outside_current_allowlist():
    second = Evidence("another-id", "Avery Finch", 0, "Avery Finch founded Cedar Lab.")
    client = SyntheticTransport([action(["E2"]), action(["E2"])])
    planner = a1.A1FreshPlanner(client, variant="original", trace_prefix="not-cached")
    first = planner(state(EVIDENCE + (second,)))
    assert first.bindings[0].evidence_ids == ("another-id",)
    with pytest.raises(A0LocalOutputError, match="unknown short reference"):
        planner(state())
    assert len(client.calls) == 2 and client.block_reason is None


def test_actual_a1_message_budget_is_checked_before_transport():
    # Use a full-ID v2 request below its budget but too close to fit the A1 suffix.
    capture = SyntheticTransport([STOP])
    a0.A0FreshPlanner(capture, trace_prefix="capture")(state())
    request = capture.requests[0]
    payload = json.loads(request["messages"][1]["content"])
    from growrag.experiments import operator_model as legacy

    while True:
        candidate = {**payload, "original_question": payload["original_question"] + "x" * 100}
        try:
            messages = legacy._messages(a0.fresh_planner_prompt(), candidate)
        except ValueError:
            break
        payload = candidate
        request["messages"] = messages
    client = SyntheticTransport([])
    adapter = a1._BindingShortrefClient(client, variant="original", on_record=lambda event: None)
    with pytest.raises(ValueError, match="prompt exceeds") as caught:
        adapter.complete(**request)
    assert not isinstance(caught.value, A0LocalOutputError)
    assert client.requests == []
