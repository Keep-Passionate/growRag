"""Zero-API structured adapter regressions, including unchanged historical signature."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import operator_model as legacy
from growrag.experiments.operator_execution_signature import execution_signature
from growrag.experiments.operator_model_v2 import (
    ModelOperatorPlannerV2,
    answer_episode_v2,
    normalize_wire_plan,
    planner_prompt,
)
from growrag.experiments.operator_schemas import PLANNER_VERSIONS, READER_VERSION, REASON_CODES
from growrag.experiments.protocol import Evidence, GoldRecord, RuntimeQuestion
from growrag.macro_operators import (
    GapField,
    OperatorRegistry,
    OperatorSpec,
    QueryStep,
    RuntimeState,
)
from growrag.operator_bank import operator_to_dict
from growrag.operator_loop import Observation


class FakeClient:
    def __init__(self, output):
        self.output = json.dumps(output) if type(output) is dict else output
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": deepcopy(messages), **kwargs})
        return SimpleNamespace(content=self.output, request_id="fake-not-a-paid-request")


@pytest.fixture
def state():
    return Observation(
        RuntimeQuestion("q1", "Where did the director live?"),
        (Evidence("e1", "Film", 0, "Jane Doe directed the film."),),
        (),
        2,
        1,
    )


@pytest.fixture
def spec():
    return OperatorSpec(
        "LOOKUP",
        "1",
        ("lookup",),
        (GapField("term"),),
        (QueryStep("search", "{original_question} {term}"),),
    )


def stop():
    return {
        "decision": "stop",
        "reason": "no_useful_query",
        "intent": "lookup",
        "constraints": [],
        "selected_operator": None,
        "operator": None,
        "gap_entries": [],
        "bindings": [],
    }


def act(spec):
    return {
        **stop(),
        "decision": "act",
        "reason": "missing_evidence",
        "operator": operator_to_dict(spec),
        "gap_entries": [{"name": "term", "value": "location"}],
    }


def selected(spec):
    return {
        **act(spec),
        "operator": None,
        "selected_operator": {"operator_id": spec.operator_id, "version": spec.version},
    }


def planner(client, *, mode="fresh", **kwargs):
    return ModelOperatorPlannerV2(client, mode=mode, trace_prefix="synthetic/structured", **kwargs)


def test_fresh_uses_new_wire_and_same_canonical_operator(spec, state):
    client = FakeClient(act(spec))
    proposal = planner(client)(state)
    assert proposal.spec == legacy.canonical_operator(spec)
    assert proposal.origin == "fresh" and proposal.gap == {"term": "location"}
    assert client.calls[0]["prompt_version"] == PLANNER_VERSIONS["fresh"]
    assert client.calls[0]["messages"][0]["content"] == planner_prompt("fresh")
    assert client.calls[0]["trace_id"] == "synthetic/structured/plan/1"
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["candidate_specs"] == []
    assert len(client.calls) == 1


def test_static_selects_unchanged_seed(state):
    seed = legacy.seed_specs()[0]
    output = selected(seed)
    output["gap_entries"] = [{"name": "search_terms", "value": "location"}]
    client = FakeClient(output)
    proposal = planner(client, mode="static")(state)
    assert proposal.spec == seed and proposal.origin == "static"
    assert client.calls[0]["prompt_version"] == PLANNER_VERSIONS["static"]


def test_memory_reuses_or_falls_back_without_changing_creation_grammar(spec, state):
    reused = planner(FakeClient(selected(spec)), mode="memory", specs=(spec,))(state)
    fresh = planner(FakeClient(act(spec)), mode="memory", specs=(spec,))(state)
    assert reused.origin == "reuse" and reused.spec == spec
    assert fresh.origin == "fresh" and fresh.spec == legacy.canonical_operator(spec)


@pytest.mark.parametrize("mode", ["fresh", "static", "memory"])
def test_stop_requires_no_action_arguments(mode, state, spec):
    patches = (
        {"operator": operator_to_dict(spec)},
        {"selected_operator": {"operator_id": spec.operator_id, "version": "1"}},
        {"gap_entries": [{"name": "term", "value": "x"}]},
        {"bindings": [{"name": "bridge", "value": "Jane Doe", "evidence_ids": ["e1"]}]},
    )
    for patch in patches:
        client = FakeClient({**stop(), **patch})
        with pytest.raises((ValueError, TypeError)):
            planner(client, mode=mode)(state)
        assert len(client.calls) == 1  # No salvage or corrective API call.


def test_static_never_copies_creates_or_renames_a_spec(spec, state):
    for output in (act(spec), act(legacy.seed_specs()[0])):
        client = FakeClient(output)
        with pytest.raises(ValueError, match="operator"):
            planner(client, mode="static")(state)
        assert len(client.calls) == 1
    unknown = selected(replace(legacy.seed_specs()[0], operator_id="RENAMED"))
    with pytest.raises(ValueError, match="not offered"):
        planner(FakeClient(unknown), mode="static")(state)


def test_fresh_cannot_select_even_if_response_invents_candidate(spec, state):
    with pytest.raises(ValueError, match="selected_operator"):
        planner(FakeClient(selected(spec)))(state)
    client = FakeClient(stop())
    with pytest.raises(ValueError, match="FRESH"):
        planner(client, specs=(spec,))
    assert client.calls == []


def test_select_and_create_cannot_coexist(spec, state):
    output = {**selected(spec), "operator": operator_to_dict(spec)}
    with pytest.raises(ValueError, match="invalid selected"):
        planner(FakeClient(output), mode="memory", specs=(spec,))(state)


@pytest.mark.parametrize("reason", ["", "Need the location", "A" * 1000, "A" * 2001, 1, None])
def test_reason_is_exact_diagnostic_enum_not_trimmed_or_repaired(reason, state):
    client = FakeClient({**stop(), "reason": reason})
    with pytest.raises(ValueError):
        planner(client)(state)
    assert len(client.calls) == 1


@pytest.mark.parametrize("reason", REASON_CODES)
def test_reason_enum_does_not_supervise_sufficiency(reason, state):
    result = planner(FakeClient({**stop(), "reason": reason}))(state)
    assert result.reason == reason and result.spec is None


@pytest.mark.parametrize("value", ["abc", "0", "false", 0, -2, True, False])
def test_gap_decoder_preserves_value_and_exact_type(value):
    output = {**stop(), "gap_entries": [{"name": "dynamicField_9", "value": value}]}
    original = deepcopy(output)
    decoded = normalize_wire_plan(json.dumps(output), mode="fresh")
    assert decoded["gap"]["dynamicField_9"] == value
    assert type(decoded["gap"]["dynamicField_9"]) is type(value)
    assert "gap_entries" not in decoded
    assert output == original


@pytest.mark.parametrize(
    "entries",
    [
        [{"name": "term", "value": "x"}, {"name": "term", "value": "x"}],
        [{"name": "term", "value": "x"}, {"name": "term", "value": "y"}],
        [{"name": "term", "value": 1.0}],
        [{"name": "term", "value": None}],
        [{"name": "", "value": "x"}],
        [{"name": "bad.name", "value": "x"}],
        [{"name": "term"}],
        [{"name": "term", "value": "x", "extra": True}],
    ],
)
def test_gap_entries_reject_duplicates_invalid_names_and_wrong_shapes(entries):
    with pytest.raises(ValueError):
        normalize_wire_plan(json.dumps({**stop(), "gap_entries": entries}), mode="memory")


def test_dynamic_typed_dsl_and_local_type_check_remain_intact(state):
    typed = OperatorSpec(
        "COUNT_QUERY",
        "1",
        ("lookup",),
        (GapField("count", "integer"),),
        (QueryStep("search", "{original_question} {count}"),),
    )
    for value, valid in ((3, True), (True, False), ("3", False)):
        output = {**act(typed), "gap_entries": [{"name": "count", "value": value}]}
        proposal = planner(FakeClient(output))(state)
        registry = OperatorRegistry()
        registry.register(proposal.spec)
        if valid:
            plan = registry.plan(
                proposal.spec.operator_id,
                "1",
                goal=proposal.goal,
                gap=proposal.gap,
                state=RuntimeState(state.evidence, (), 2),
            )
            assert plan.requests[0].query.endswith(" 3")
        else:
            with pytest.raises(TypeError):
                registry.plan(
                    proposal.spec.operator_id,
                    "1",
                    goal=proposal.goal,
                    gap=proposal.gap,
                    state=RuntimeState(state.evidence, (), 2),
                )


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        '{"x":1,"x":2}',
        '{"x":NaN}',
        '{"x":Infinity}',
        "```json\n{}\n```",
        "{} trailing",
    ],
)
def test_no_silent_json_salvage_and_raw_failure_is_logged(content, state):
    events, client = [], FakeClient(content)
    with pytest.raises(ValueError):
        planner(client, on_record=events.append)(state)
    assert len(client.calls) == 1
    assert len(events) == 1
    assert events[0]["stage"] == "raw_wire" and events[0]["raw_content"] == content


def test_raw_and_normalized_logs_are_separate_and_callback_cannot_mutate_input(spec, state):
    output, events = act(spec), []

    def record(event):
        events.append(deepcopy(event))
        if event["stage"] == "normalized_parser_input":
            event["model_output"]["gap"]["term"] = "tampered"
        else:
            event["payload"].clear()

    result = planner(FakeClient(output), on_record=record)(state)
    assert result.gap == {"term": "location"}
    assert [event["stage"] for event in events] == ["raw_wire", "normalized_parser_input"]
    assert json.loads(events[0]["raw_content"]) == output
    assert events[1]["model_output"]["gap"] == {"term": "location"}
    assert events[1]["semantic_validation_complete"] is False


def test_gold_objects_and_foreign_source_fields_do_not_enter_runtime(state):
    client = FakeClient(stop())
    gold = GoldRecord("q1", ("SECRET_REFERENCE",))
    with pytest.raises(TypeError, match="gold-free"):
        planner(client)(replace(state, question=gold))
    with pytest.raises(TypeError, match="gold-free"):
        answer_episode_v2(client, gold, (), trace_id="synthetic/reader")
    assert client.calls == []
    planner(client)(state)
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert set(payload) == {
        "mode",
        "original_question",
        "evidence",
        "evidence_window_omitted_count",
        "previous_queries",
        "remaining_retrievals",
        "candidate_specs",
        "candidate_shortlist_omitted_count",
    }
    assert "SECRET_REFERENCE" not in client.calls[0]["messages"][1]["content"]


def test_gold_named_operator_field_is_still_rejected(spec, state):
    output = act(spec)
    output["operator"]["gap_schema"][0]["name"] = "gold_answer"
    output["operator"]["steps"][0]["template"] = "{original_question} {gold_answer}"
    output["gap_entries"][0]["name"] = "gold_answer"
    with pytest.raises(ValueError, match="gold"):
        planner(FakeClient(output))(state)


@pytest.mark.parametrize(
    "value,ids",
    [
        ("Never Appeared", ["e1"]),
        ("Jane Doe", ["unseen"]),
        ("Jane Doe", []),
    ],
)
def test_evidence_binding_cannot_bypass_old_validator(spec, state, value, ids):
    output = act(spec)
    output["bindings"] = [{"name": "bridge", "value": value, "evidence_ids": ids}]
    with pytest.raises((ValueError, TypeError)):
        planner(FakeClient(output))(state)


def test_reader_new_version_preserves_original_reader_semantics(state):
    output = {"answer": "Jane Doe", "supported": True, "evidence_ids": ["e1"]}
    events, client = [], FakeClient(output)
    result = answer_episode_v2(
        client, state.question, state.evidence, trace_id="synthetic/reader", on_record=events.append
    )
    assert result["answer"] == "Jane Doe" and result["support_is_model_claim"] is True
    assert client.calls[0]["prompt_version"] == READER_VERSION
    assert client.calls[0]["messages"][0]["content"] == legacy.READER_PROMPT
    assert events[0]["stage"] == "raw_wire"


@pytest.mark.parametrize(
    "patch",
    [
        {"supported": False},
        {"evidence_ids": ["unseen"]},
        {"evidence_ids": ["e1", "e1"]},
        {"answer": ""},
        {"supported": 1},
        {"extra": "not allowed"},
    ],
)
def test_reader_cross_field_or_citation_failure_remains_hard(patch, state):
    output = {"answer": "Jane Doe", "supported": True, "evidence_ids": ["e1"], **patch}
    client = FakeClient(output)
    with pytest.raises(ValueError):
        answer_episode_v2(client, state.question, state.evidence, trace_id="synthetic/reader")
    assert len(client.calls) == 1


@pytest.mark.parametrize("mode", ["fresh", "static", "memory"])
def test_prompt_examples_use_wire_format_and_mode_constraints(mode):
    prompt = planner_prompt(mode)
    assert "not an explanation, score or proof of sufficiency" in prompt
    examples = [json.loads(line) for line in prompt.splitlines() if line.startswith('{"decision"')]
    assert len(examples) == (3 if mode == "memory" else 2)
    for value in examples:
        normalize_wire_plan(json.dumps(value), mode=mode)
        assert "gap" not in value
    assert legacy.PLANNER_PROMPT.split("Return exactly one JSON object")[0] in prompt


def test_oversized_question_is_rejected_before_real_client(state):
    client = FakeClient(stop())
    with pytest.raises(ValueError, match="prompt exceeds"):
        planner(client)(replace(state, question=RuntimeQuestion("q", "x" * 40000)))
    assert client.calls == []


def test_historical_method_files_and_signature_remain_unchanged():
    root = Path(__file__).resolve().parents[1]
    assert execution_signature(root)["sha256"] == (
        "e9122473d934e50a6800d909e6e90fe0db97dfd15f49b51d9a549655f7089836"
    )
    assert legacy.PLANNER_VERSION == "growrag-operator-planner-v2"
    assert legacy.READER_VERSION == "growrag-operator-reader-v1"
