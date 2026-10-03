"""Synthetic prospective wire tests; no credentials, real corpus, labels or API."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from growrag.experiments import a0_runtime as v1
from growrag.experiments import a0_runtime_v2 as a0
from growrag.experiments import operator_model as legacy
from growrag.experiments.api_client import APIRequestError
from growrag.experiments.history_runtime_v2 import FILL_PROMPT, FILL_VERSION
from growrag.experiments.operator_schemas_v3 import planner_schema as v3_planner_schema
from growrag.experiments.output_schemas import registry_manifest, response_format_for
from growrag.experiments.protocol import Evidence, RuntimeQuestion
from growrag.history_library import FrozenHistoryLibrary, HistoryCard, HistoryRecord
from growrag.macro_operators import (
    GapField,
    OperatorRegistry,
    OperatorSpec,
    QueryStep,
    RuntimeState,
)
from growrag.operator_loop import Observation

QUESTION = RuntimeQuestion("synthetic-v2", "Where was the founder of Cedar Lab born?")
EVIDENCE = (Evidence("current-1", "Cedar Lab", 0, "Avery Finch founded Cedar Lab."),)
STOP = {"reason": "no_useful_query", "intent": "lookup", "constraints": [], "actions": []}
ANSWER = {"candidate_answer": "Stonebridge", "supported": True, "evidence_ids": ["E1"]}


class SyntheticDurable:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.config = SimpleNamespace(json_object_mode=True, json_schema_mode=False)
        self.requests, self.calls = [], []
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
            {
                "trace_id": trace_id,
                "prompt_version": prompt_version,
                "status": "completed",
                "api_requests": 1,
                "estimated_actual_cny": 0.0000106,
            }
        )
        return SimpleNamespace(
            content=output if isinstance(output, str) else json.dumps(output),
            request_id="synthetic-unpaid",
            input_tokens=17,
            output_tokens=9,
        )


def state():
    return Observation(QUESTION, EVIDENCE, (), 2, 1)


def contract(outputs, events=None):
    delegate = SyntheticDurable(outputs)
    return a0.A0ContractClient(
        delegate, on_record=None if events is None else events.append
    ), delegate


def integer_action(value="1977", condition=1977):
    return {
        "reason": "missing_evidence",
        "intent": "lookup",
        "constraints": ["Preserve the original goal"],
        "actions": [
            {
                "selected_operator": None,
                "operator": {
                    "operator_id": "YEAR_LOOKUP",
                    "version": "1",
                    "gap_schema": [
                        {"name": "year", "kind": "integer", "required": True},
                        {"name": "entity", "kind": "text", "required": True},
                        {"name": "optional", "kind": "text", "required": False},
                    ],
                    "steps": [
                        {
                            "step_id": "primary",
                            "template": "{entity} {year}",
                            "when": [],
                            "requires_bindings": [],
                        },
                        {
                            "step_id": "conditional",
                            "template": "{original_question} {year}",
                            "when": [{"field": "year", "value": condition}],
                            "requires_bindings": [],
                        },
                        {
                            "step_id": "optional",
                            "template": "{optional}",
                            "when": [{"field": "optional", "value": "present"}],
                            "requires_bindings": [],
                        },
                    ],
                },
                "gap_entries": [
                    {"name": "year", "value": value},
                    {"name": "entity", "value": "1977"},
                ],
                "bindings": [],
            }
        ],
    }


def compile_proposal(proposal):
    registry = OperatorRegistry()
    registry.register(proposal.spec)
    return registry.plan(
        proposal.spec.operator_id,
        proposal.spec.version,
        goal=proposal.goal,
        gap=proposal.gap,
        state=RuntimeState(EVIDENCE, proposal.bindings, 2),
    )


@pytest.mark.parametrize("value, expected", [("2", 2), ("1977", 1977), ("0", 0), ("-2", -2)])
@pytest.mark.parametrize("focused", [False, True])
def test_canonical_integer_wire_and_single_intent_reach_unchanged_compiler(
    value, expected, focused
):
    output = integer_action(value, condition=value)
    original = deepcopy(output)
    client, delegate = contract([output])
    events = []
    proposal = a0.A0FreshPlanner(
        client, trace_prefix="typed", on_record=events.append, focused=focused
    )(state())
    assert proposal.goal.intent == "lookup"
    assert proposal.spec.supported_intents == ("lookup",)
    assert proposal.gap == {"year": expected, "entity": "1977"}
    assert proposal.goal.original_question == QUESTION.text
    assert proposal.goal.constraints == ("Preserve the original goal",)
    assert [request.query for request in compile_proposal(proposal).requests] == [
        f"1977 {expected}",
        f"{QUESTION.text} {expected}",
    ]
    assert proposal.spec.gap_schema[-1].required is False
    assert proposal.spec.steps[1].when[0].value == expected
    assert proposal.spec.steps[2].when[0].value == "present"
    audit = next(event for event in events if event["stage"] == "wire_normalization")
    assert audit["raw_wire"] == original == output
    differences = audit["normalization_differences"]
    assert [row["operation"] for row in differences] == [
        "derive_from_intent",
        "canonical_integer_string",
        "canonical_integer_string",
    ]
    assert differences[1]["raw_value"] == differences[2]["raw_value"] == value
    assert differences[1]["normalized_value"] == differences[2]["normalized_value"] == expected
    normalized = next(event for event in events if event["stage"] == "normalized_parser_input")
    assert normalized["model_output"]["gap"]["year"] == expected
    expected_version = a0.FOCUSED_PLANNER_VERSION if focused else a0.FRESH_PLANNER_VERSION
    assert delegate.requests[0]["prompt_version"] == expected_version
    assert len(delegate.calls) == 1 and delegate.block_reason is None


@pytest.mark.parametrize(
    "bad",
    [
        "02",
        "00",
        "-02",
        "-0",
        "+2",
        "2.0",
        "2e0",
        " 2",
        "2 ",
        "two",
        "٢",
        True,
        False,
        2.0,
        None,
        "",
    ],
)
@pytest.mark.parametrize("slot", ["gap", "condition"])
def test_noncanonical_integer_encodings_fail_locally_without_retries(bad, slot):
    output = integer_action(bad) if slot == "gap" else integer_action(2, condition=bad)
    events = []
    client, delegate = contract([output], events)
    with pytest.raises(a0.A0LocalOutputError):
        a0.A0FreshPlanner(client, trace_prefix="bad-integer")(state())
    assert len(delegate.requests) == len(delegate.calls) == 1
    assert events[-1]["kind"] == "local_contract_failure"
    assert events[-1]["scope"] == "arm" and delegate.block_reason is None


def test_native_integer_boolean_and_text_capabilities_are_not_cast_or_narrowed():
    output = integer_action(1977)
    action = output["actions"][0]
    action["operator"]["gap_schema"].append({"name": "enabled", "kind": "bool", "required": True})
    action["gap_entries"].append({"name": "enabled", "value": True})
    action["operator"]["steps"][0]["when"] = [{"field": "enabled", "value": True}]
    canonical, audit = a0.decode_fresh_wire(json.dumps(output))
    assert canonical["gap"] == {"year": 1977, "entity": "1977", "enabled": True}
    assert len(audit["normalization_differences"]) == 1
    client, _ = contract([output])
    assert (
        len(compile_proposal(a0.A0FreshPlanner(client, trace_prefix="types")(state())).requests)
        == 2
    )


def test_old_supported_intents_field_is_rejected_not_silently_repaired():
    output = integer_action()
    output["actions"][0]["operator"]["supported_intents"] = ["bridge", "comparison"]
    client, delegate = contract([output])
    with pytest.raises(a0.A0LocalOutputError, match="unexpected or missing fields"):
        a0.A0FreshPlanner(client, trace_prefix="old-wire")(state())
    assert len(delegate.calls) == 1 and delegate.block_reason is None


@pytest.mark.parametrize("focused", [False, True])
def test_stop_carries_no_invented_parameters_and_same_current_evidence(focused):
    client, delegate = contract([STOP])
    proposal = a0.A0FreshPlanner(client, trace_prefix="stop", focused=focused)(state())
    assert proposal.origin == "stop" and proposal.spec is None and proposal.gap == {}
    payload = json.loads(delegate.requests[0]["messages"][1]["content"])
    assert payload["candidate_specs"] == [] and payload["candidate_shortlist_omitted_count"] == 0
    assert payload["evidence"] == legacy.visible_evidence(EVIDENCE)
    assert payload["original_question"] == QUESTION.text
    assert a0.decode_fresh_wire(json.dumps(STOP))[1]["normalization_differences"] == []


def test_schema_and_focused_guidance_are_independent_and_preserve_dsl_capabilities():
    old_schema = v3_planner_schema("fresh")
    expected = deepcopy(old_schema)
    operator = expected["properties"]["actions"]["items"]["properties"]["operator"]
    del operator["properties"]["supported_intents"]
    operator["required"].remove("supported_intents")
    assert a0.planner_schema() == expected
    assert v3_planner_schema("fresh") == old_schema
    assert a0.focused_planner_prompt().startswith(a0.fresh_planner_prompt())
    assert v1._FOCUSED_GUIDANCE in a0.focused_planner_prompt()
    assert a0._FOCUSED_EXAMPLE == a0._without_supported_intents(v1._FOCUSED_EXAMPLE)
    registry = registry_manifest()
    for version in (a0.FRESH_PLANNER_VERSION, a0.FOCUSED_PLANNER_VERSION, a0.SHORT_READER_VERSION):
        assert a0.schema_for(version)["additionalProperties"] is False
        with pytest.raises(ValueError):
            response_format_for(version)
    assert registry_manifest() == registry
    client, delegate = contract([])
    with pytest.raises(ValueError):
        client.complete([], trace_id="old-version", prompt_version=v1.SHORT_READER_VERSION)
    assert delegate.requests == []


@pytest.mark.parametrize("supported", [False, True])
def test_reader_preserves_raw_candidate_but_serves_only_supported_answer(supported):
    value = {**ANSWER, "supported": supported}
    raw = json.dumps(value, indent=2)
    events = []
    client, delegate = contract([raw])
    result = a0.answer_episode_shortrefs(
        client, QUESTION, EVIDENCE, trace_id="gated", on_record=events.append
    )
    assert result["candidate_answer"] == "Stonebridge"
    assert result["answer"] == ("Stonebridge" if supported else "")
    assert result["supported"] is supported
    assert result["raw_content"] == raw == events[1]["raw_content"]
    assert result["evidence_ids"] == ["current-1"] and result["evidence_refs"] == ["E1"]
    assert result["support_is_model_claim"] is True
    assert events[-1]["output"] == result and delegate.block_reason is None


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize(
    "refs",
    [
        ["E2"],
        ["E1", "E1"],
        ["E01"],
        ["e1"],
        ["E 1"],
        [" E1"],
        ["E1 "],
        ["[E1]"],
        ["current-1"],
        ["E١"],
        [1],
    ],
)
def test_reader_never_repairs_citations_even_when_unsupported(supported, refs):
    client, delegate = contract([{**ANSWER, "supported": supported, "evidence_ids": refs}])
    with pytest.raises(a0.A0LocalOutputError):
        a0.answer_episode_shortrefs(client, QUESTION, EVIDENCE, trace_id="bad-ref")
    assert len(delegate.calls) == 1 and delegate.block_reason is None


@pytest.mark.parametrize(
    "value",
    [
        {**ANSWER, "candidate_answer": ""},
        {**ANSWER, "candidate_answer": " "},
        {**ANSWER, "evidence_ids": []},
        {**ANSWER, "supported": "false"},
        {**ANSWER, "answer": "old-key"},
        {"answer": "old-key", "supported": False, "evidence_ids": []},
        {**ANSWER, "candidate_answer": None},
        '{"candidate_answer":"x","candidate_answer":"y","supported":false,"evidence_ids":[]}',
    ],
)
def test_reader_strict_keys_types_and_supported_evidence_contract(value):
    client, _ = contract([value])
    with pytest.raises(a0.A0LocalOutputError):
        a0.answer_episode_shortrefs(client, QUESTION, EVIDENCE, trace_id="strict")


def test_reader_empty_evidence_abstention_preserves_unserved_candidate():
    client, _ = contract([{**ANSWER, "supported": False, "evidence_ids": []}])
    result = a0.answer_episode_shortrefs(client, QUESTION, (), trace_id="empty")
    assert result["candidate_answer"] == "Stonebridge" and result["answer"] == ""
    assert result["evidence_ids"] == result["visible_evidence_ids"] == []


def test_reader_uses_identical_full_id_windows_and_callbacks_cannot_mutate_output():
    evidence = tuple(
        Evidence(f"long-document-{number}-" + "z" * 160, "标题" * 20, number, "内容" * 2500)
        for number in range(26)
    )
    old = SyntheticDurable([{"answer": "", "supported": False, "evidence_ids": []}])
    previous = v1.answer_episode_shortrefs(old, QUESTION, evidence, trace_id="previous")
    new = SyntheticDurable([ANSWER])

    def mutate(event):
        event.get("payload", {}).clear()
        event.get("alias_to_evidence_id", {}).clear()
        event.get("output", {}).clear()

    result = a0.answer_episode_shortrefs(new, QUESTION, evidence, trace_id="new", on_record=mutate)
    assert old.requests[0]["messages"][1] == new.requests[0]["messages"][1]
    assert result["evidence_windows"] == previous["evidence_windows"]
    assert result["evidence_window_omitted_count"] == 2
    assert result["evidence_ids"] == [evidence[0].evidence_id]


def library():
    spec = OperatorSpec(
        "BIRTHPLACE",
        "1",
        ("lookup",),
        (GapField("entity"),),
        (QueryStep("first", "{entity} birthplace"), QueryStep("second", "{entity} biography")),
    )
    return FrozenHistoryLibrary(
        "synthetic-v2",
        (),
        tuple(
            HistoryRecord(
                HistoryCard(
                    f"T{number:02d}",
                    "1",
                    "Birthplace lookup",
                    "Find current birthplace.",
                    "Search current entity.",
                    operator_spec=replace(spec, operator_id=f"S{number}"),
                ),
                "reference",
                "synthetic/manual",
                "0" * 64,
                status="published",
            )
            for number in range(9)
        ),
    )


@pytest.mark.parametrize("method", a0.METHODS)
def test_full_reports_and_bounded_executor_keep_scorer_compatibility(tmp_path, method):
    outputs = []
    if method.startswith("fresh_"):
        outputs.append(integer_action())
    elif method == "history_body8":
        outputs.extend(
            [
                {"selected_card_id": "T00", "reason": "condition_match"},
                {
                    "intent": "lookup",
                    "constraints": [],
                    "gap_entries": [{"name": "entity", "value": "Cedar Lab"}],
                    "bindings": [],
                },
            ]
        )
    outputs.append({**ANSWER, "supported": False})
    events, searches = [], []
    client, delegate = contract(outputs, events)

    def index(query, top_k):
        searches.append((query, top_k))
        return [SimpleNamespace(doc_id=f"doc/{len(searches)}", title="Source", text="Evidence.")]

    target = tmp_path / f"{method}.json"
    report = a0.execute_arm(
        QUESTION,
        method,
        index,
        client,
        library=library(),
        trace=method,
        log=events.append,
        target=target,
    )
    assert report["status"] == "completed" and report["method"] == report["arm"] == method
    assert (
        report["reader"]["answer"] == "" and report["reader"]["candidate_answer"] == "Stonebridge"
    )
    assert report["reader"]["prompt_version"] == a0.SHORT_READER_VERSION
    assert report["episode"] is not None and len(report["episode"]["proposals"]) <= 2
    assert len(searches) == (1 if method == "base" else 3)
    assert all(top_k == 6 for _, top_k in searches)
    assert report["calls"] == delegate.calls
    assert report["gold_loaded"] is False and report["memory_updated"] is False
    assert json.loads(target.read_text(encoding="utf-8")) == json.loads(json.dumps(report))
    if method == "history_body8":
        ranking = next(event for event in events if event["kind"] == "history_candidates")
        assert len(ranking["ranking"]) == 8
        fill = next(row for row in delegate.requests if row["prompt_version"] == FILL_VERSION)
        assert fill["messages"][0]["content"] == FILL_PROMPT
        assert json.loads(fill["messages"][1]["content"])["allowed_evidence_ids"] == ["doc/1"]
    before = len(delegate.calls)
    with pytest.raises(FileExistsError):
        a0.execute_arm(
            QUESTION,
            method,
            index,
            client,
            library=library(),
            trace=method,
            log=events.append,
            target=target,
        )
    assert len(delegate.calls) == before


@pytest.mark.parametrize("failure", ["reader", "executor", "transport", "retrieval"])
def test_failure_boundary_and_full_reports_are_preserved(tmp_path, failure):
    output = integer_action()
    if failure == "executor":
        output["actions"][0]["gap_entries"] = []
    outputs = {
        "reader": [{**ANSWER, "supported": False, "evidence_ids": ["E01"]}],
        "executor": [output],
        "transport": [APIRequestError("synthetic transport failure")],
        "retrieval": [output],
    }[failure]
    client, delegate = contract(outputs)
    searches = []

    def index(query, top_k):
        searches.append(query)
        if failure == "retrieval" and len(searches) > 1:
            raise ValueError("synthetic index integrity failure")
        return [SimpleNamespace(doc_id="doc/1", title="Source", text="Evidence.")]

    target = tmp_path / "failure.json"
    with pytest.raises((ValueError, APIRequestError)) as caught:
        a0.execute_arm(
            QUESTION,
            "fresh_original" if failure in {"executor", "retrieval"} else "base",
            index,
            client,
            library=None,
            trace="failure",
            log=lambda event: None,
            target=target,
        )
    assert isinstance(caught.value, a0.A0LocalOutputError) == (failure in {"reader", "executor"})
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["reader"] is None
    assert report["calls"] == delegate.calls
    assert delegate.block_reason == ("transport_failed" if failure == "transport" else None)
    assert len(delegate.requests) == 1


@pytest.mark.parametrize("stage", ["raw_wire", "wire_normalization", "normalized_parser_input"])
def test_audit_callback_errors_are_not_model_output_failures(stage):
    client, delegate = contract([integer_action()])

    def fail(event):
        if event.get("stage") == stage:
            raise ValueError("synthetic audit write failure")

    with pytest.raises(ValueError) as caught:
        a0.A0FreshPlanner(client, trace_prefix="audit-failure", on_record=fail)(state())
    assert not isinstance(caught.value, a0.A0LocalOutputError)
    assert len(delegate.calls) == 1 and delegate.block_reason is None


def test_grounded_focused_example_and_visibility_constraints_are_unchanged():
    client, _ = contract([a0._FOCUSED_EXAMPLE])
    proposal = a0.FocusedFreshPlanner(client, trace_prefix="grounded")(state())
    assert [row.query for row in compile_proposal(proposal).requests] == ["Avery Finch birthplace"]
    evidence = (Evidence("current-1", "Source", 0, "x" * 20000 + " Avery Finch"),)
    client, delegate = contract([a0._FOCUSED_EXAMPLE])
    with pytest.raises(a0.A0LocalOutputError, match="absent from the visible cited window"):
        a0.FocusedFreshPlanner(client, trace_prefix="hidden-tail")(
            replace(state(), evidence=evidence)
        )
    assert len(delegate.calls) == 1 and delegate.block_reason is None


def test_history_fill_does_not_gain_fresh_integer_string_normalization(tmp_path):
    spec = OperatorSpec(
        "YEAR",
        "1",
        ("lookup",),
        (GapField("year", "integer"),),
        (QueryStep("first", "{original_question} {year}"),),
    )
    card = HistoryCard(
        "T00",
        "1",
        "Year lookup",
        "Find current year.",
        "Search current year.",
        operator_spec=spec,
    )
    frozen = FrozenHistoryLibrary(
        "synthetic-year",
        (),
        (HistoryRecord(card, "reference", "synthetic/manual", "0" * 64, status="published"),),
    )
    events = []
    client, delegate = contract(
        [
            {"selected_card_id": "T00", "reason": "condition_match"},
            {
                "intent": "lookup",
                "constraints": [],
                "bindings": [],
                "gap_entries": [{"name": "year", "value": "1977"}],
            },
        ],
        events,
    )

    def index(query, top_k):
        return [SimpleNamespace(doc_id="doc/1", title="Source", text="Evidence.")]

    target = tmp_path / "history-strict.json"
    with pytest.raises(a0.A0LocalOutputError):
        a0.execute_arm(
            QUESTION,
            "history_body8",
            index,
            client,
            library=frozen,
            trace="history-strict",
            log=events.append,
            target=target,
        )
    assert len(delegate.calls) == 2 and delegate.block_reason is None
    fill = next(
        event
        for event in events
        if event.get("prompt_version") == FILL_VERSION and event.get("kind") == "a0_raw_wire"
    )
    assert json.loads(fill["raw_content"])["gap_entries"][0]["value"] == "1977"
    assert not any(event.get("stage") == "wire_normalization" for event in events)
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["reader"] is None
