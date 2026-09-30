"""Action-list decoding is versioned representation, never output salvage."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import operator_model as legacy
from growrag.experiments import operator_model_v2 as previous
from growrag.experiments.operator_execution_signature import execution_signature
from growrag.experiments.operator_model_v3 import (
    ModelOperatorPlannerV3,
    answer_episode,
    encode_wire_plan,
    normalize_wire_plan,
    planner_prompt,
)
from growrag.experiments.operator_schemas_v3 import PLANNER_VERSIONS, READER_VERSION, REASON_CODES
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
        return SimpleNamespace(content=self.output, request_id="synthetic-unpaid")


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


def canonical_stop():
    return {
        "decision": "stop",
        "reason": "no_useful_query",
        "intent": "lookup",
        "constraints": [],
        "selected_operator": None,
        "operator": None,
        "gap": {},
        "bindings": [],
    }


def canonical_action(spec, *, select=False):
    return {
        **canonical_stop(),
        "decision": "act",
        "reason": "missing_evidence",
        "operator": None if select else operator_to_dict(spec),
        "selected_operator": {"operator_id": spec.operator_id, "version": spec.version}
        if select
        else None,
        "gap": {"term": "location"},
    }


def wire_stop():
    return {"reason": "no_useful_query", "intent": "lookup", "constraints": [], "actions": []}


def planner(client, *, mode="fresh", **kwargs):
    return ModelOperatorPlannerV3(client, mode=mode, trace_prefix="synthetic/action-list", **kwargs)


@pytest.mark.parametrize("mode", ["fresh", "static", "memory"])
def test_empty_action_list_decodes_exact_stop_and_is_semantically_identical(mode, state):
    client = FakeClient(wire_stop())
    actual = planner(client, mode=mode)(state)
    expected = legacy.ModelOperatorPlanner(
        FakeClient(canonical_stop()), mode=mode, trace_prefix="legacy"
    )(state)
    assert actual == expected
    assert actual.spec is None and actual.gap == {} and actual.bindings == ()
    assert client.calls[0]["prompt_version"] == PLANNER_VERSIONS[mode]
    assert client.calls[0]["messages"][0]["content"] == planner_prompt(mode)


@pytest.mark.parametrize(
    "mode,select", [("fresh", False), ("static", True), ("memory", False), ("memory", True)]
)
def test_same_canonical_action_is_semantically_identical_to_old_planner(mode, select, spec, state):
    chosen = legacy.seed_specs()[0] if mode == "static" else spec
    canonical = canonical_action(chosen, select=select)
    if mode == "static":
        canonical["gap"] = {"search_terms": "location"}
    specs = (chosen,) if mode == "memory" else ()
    wire = encode_wire_plan(canonical, mode=mode)
    assert normalize_wire_plan(json.dumps(wire), mode=mode) == canonical
    actual = planner(FakeClient(wire), mode=mode, specs=specs)(state)
    expected = legacy.ModelOperatorPlanner(
        FakeClient(canonical), mode=mode, specs=specs, trace_prefix="legacy"
    )(state)
    assert actual == expected


@pytest.mark.parametrize("mode", ["fresh", "static", "memory"])
def test_stop_rejects_all_old_top_level_action_parameters(mode, state):
    for key, value in {
        "decision": "stop",
        "operator": None,
        "selected_operator": None,
        "gap_entries": [],
        "gap": {},
        "bindings": [{"name": "entity", "value": "Jane Doe", "evidence_ids": ["e1"]}],
    }.items():
        client, events = FakeClient({**wire_stop(), key: value}), []
        with pytest.raises(ValueError, match="fields"):
            planner(client, mode=mode, on_record=events.append)(state)
        assert len(client.calls) == 1
        assert [event["stage"] for event in events] == ["raw_wire"]


@pytest.mark.parametrize("count", [2, 3, 8])
def test_multiple_actions_are_never_truncated_or_executed(count, spec, state):
    output = encode_wire_plan(canonical_action(spec), mode="fresh")
    output["actions"] *= count
    client, events = FakeClient(output), []
    with pytest.raises(ValueError, match="at most one"):
        planner(client, on_record=events.append)(state)
    assert len(client.calls) == 1 and len(events) == 1
    assert len(json.loads(events[0]["raw_content"])["actions"]) == count


@pytest.mark.parametrize("both", [False, True])
def test_memory_action_requires_exact_select_create_xor(both, spec, state):
    output = encode_wire_plan(canonical_action(spec), mode="memory")
    action = output["actions"][0]
    action["operator"] = operator_to_dict(spec) if both else None
    action["selected_operator"] = (
        {"operator_id": spec.operator_id, "version": "1"} if both else None
    )
    client = FakeClient(output)
    with pytest.raises(ValueError, match="exactly one"):
        planner(client, mode="memory", specs=(spec,))(state)
    assert len(client.calls) == 1


def test_mode_constraints_are_nonnullable_even_inside_one_action(spec, state):
    fresh = encode_wire_plan(canonical_action(spec), mode="fresh")
    fresh["actions"][0]["operator"] = None
    with pytest.raises(ValueError, match="type"):
        planner(FakeClient(fresh))(state)
    static = {
        **wire_stop(),
        "actions": [
            {"selected_operator": None, "operator": None, "gap_entries": [], "bindings": []}
        ],
    }
    with pytest.raises(ValueError, match="type"):
        planner(FakeClient(static), mode="static")(state)


def test_static_create_and_fresh_select_are_not_reinterpreted(spec, state):
    create = encode_wire_plan(canonical_action(spec), mode="fresh")
    select = encode_wire_plan(canonical_action(spec, select=True), mode="memory")
    for mode, output in (("static", create), ("fresh", select)):
        with pytest.raises(ValueError):
            planner(FakeClient(output), mode=mode)(state)


@pytest.mark.parametrize("reason", ["", "A" * 2001, "why stop", None, 1])
def test_reason_is_not_repaired_or_truncated(reason, state):
    client = FakeClient({**wire_stop(), "reason": reason})
    with pytest.raises(ValueError):
        planner(client)(state)
    assert len(client.calls) == 1


@pytest.mark.parametrize("reason", REASON_CODES)
def test_reason_does_not_control_or_supervise_sufficiency(reason, state):
    actual = planner(FakeClient({**wire_stop(), "reason": reason}))(state)
    assert actual.reason == reason and actual.spec is None


@pytest.mark.parametrize("value", ["x", "false", 0, -4, True, False])
def test_explicit_encoding_round_trip_preserves_gap_value_types(value, spec):
    canonical = canonical_action(spec)
    canonical["gap"] = {"dynamicField_8": value}
    original = deepcopy(canonical)
    wire = encode_wire_plan(canonical, mode="fresh")
    decoded = normalize_wire_plan(json.dumps(wire), mode="fresh")
    assert decoded == canonical == original
    assert type(decoded["gap"]["dynamicField_8"]) is type(value)


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
def test_single_action_still_rejects_duplicate_or_invalid_gap_entries(entries, spec):
    output = encode_wire_plan(canonical_action(spec), mode="fresh")
    output["actions"][0]["gap_entries"] = entries
    with pytest.raises(ValueError):
        normalize_wire_plan(json.dumps(output), mode="fresh")


def test_one_macro_action_can_still_issue_two_queries_and_respects_budget(state):
    multi = OperatorSpec(
        "TWO_QUERIES",
        "1",
        ("lookup",),
        (GapField("term"),),
        (QueryStep("first", "{term} location"), QueryStep("second", "{term} residence")),
    )
    canonical = canonical_action(multi)
    wire = encode_wire_plan(canonical, mode="fresh")
    assert len(wire["actions"]) == 1
    proposal = planner(FakeClient(wire))(state)
    registry = OperatorRegistry()
    registry.register(proposal.spec)
    plan = registry.plan(
        proposal.spec.operator_id,
        "1",
        goal=proposal.goal,
        gap=proposal.gap,
        state=RuntimeState(state.evidence, (), 2),
    )
    assert [r.query for r in plan.requests] == ["location location", "location residence"]
    with pytest.raises(ValueError, match="budget"):
        registry.plan(
            proposal.spec.operator_id,
            "1",
            goal=proposal.goal,
            gap=proposal.gap,
            state=RuntimeState(state.evidence, (), 1),
        )


def test_plain_gap_provenance_limit_is_preserved_not_misrepresented_as_grounding(spec, state):
    canonical = canonical_action(spec)
    canonical["gap"] = {"term": "Unseen Hamlet"}
    proposal = planner(FakeClient(encode_wire_plan(canonical, mode="fresh")))(state)
    registry = OperatorRegistry()
    registry.register(proposal.spec)
    plan = registry.plan(
        proposal.spec.operator_id,
        "1",
        goal=proposal.goal,
        gap=proposal.gap,
        state=RuntimeState(state.evidence, (), 2),
    )
    assert plan.requests[0].query.endswith("Unseen Hamlet") and proposal.bindings == ()
    # This test documents a limitation, not permission to call the value evidence-grounded.


@pytest.mark.parametrize("value,ids", [("Never Appeared", ["e1"]), ("Jane Doe", ["unseen"])])
def test_explicit_bindings_keep_the_original_grounding_checks(value, ids, spec, state):
    canonical = canonical_action(spec)
    canonical["bindings"] = [{"name": "bridge", "value": value, "evidence_ids": ids}]
    with pytest.raises((TypeError, ValueError)):
        planner(FakeClient(encode_wire_plan(canonical, mode="fresh")))(state)


@pytest.mark.parametrize(
    "content", ["[]", '{"x":1,"x":2}', '{"x":NaN}', "```json\n{}\n```", "{} trailing"]
)
def test_raw_malformed_outputs_remain_failed_without_salvage_or_retry(content, state):
    client, events = FakeClient(content), []
    with pytest.raises(ValueError):
        planner(client, on_record=events.append)(state)
    assert len(client.calls) == 1 and len(events) == 1
    assert events[0]["raw_content"] == content


def test_raw_and_canonical_logs_separate_and_callback_cannot_mutate(spec, state):
    output, events = encode_wire_plan(canonical_action(spec), mode="fresh"), []

    def callback(event):
        events.append(deepcopy(event))
        if event["stage"] == "normalized_parser_input":
            event["model_output"]["gap"].clear()
        else:
            event["payload"].clear()

    actual = planner(FakeClient(output), on_record=callback)(state)
    assert actual.gap == {"term": "location"}
    assert [e["stage"] for e in events] == ["raw_wire", "normalized_parser_input"]
    assert json.loads(events[0]["raw_content"]) == output
    assert events[1]["model_output"] == canonical_action(spec)
    assert events[1]["semantic_validation_complete"] is False


def test_gold_question_rejected_before_api_and_runtime_projection_unchanged(state):
    client = FakeClient(wire_stop())
    gold = GoldRecord("q1", ("SECRET_REFERENCE",))
    with pytest.raises(TypeError, match="gold-free"):
        planner(client)(replace(state, question=gold))
    with pytest.raises(TypeError, match="gold-free"):
        answer_episode(client, gold, (), trace_id="synthetic/reader")
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


def test_history_is_still_forbidden_for_fresh_and_unknown_selection_rejected(spec, state):
    client = FakeClient(wire_stop())
    with pytest.raises(ValueError, match="FRESH"):
        planner(client, specs=(spec,))
    assert client.calls == []
    unknown = canonical_action(replace(spec, operator_id="UNLISTED"), select=True)
    with pytest.raises(ValueError, match="not offered"):
        planner(FakeClient(encode_wire_plan(unknown, mode="memory")), mode="memory", specs=(spec,))(
            state
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"gap": {"term": "x"}},
        {"bindings": [{"name": "entity"}]},
        {"operator": {}},
        {"selected_operator": {"operator_id": "x", "version": "1"}},
        {"bindings": ()},
    ],
)
def test_encoder_never_drops_illegal_canonical_stop_parameters(patch):
    with pytest.raises(ValueError):
        encode_wire_plan({**canonical_stop(), **patch}, mode="memory")


def test_reader_is_exact_alias_with_unchanged_prompt_and_version(state):
    assert answer_episode is previous.answer_episode_v2
    assert READER_VERSION == previous.READER_VERSION == "growrag-operator-reader-v2"
    client = FakeClient({"answer": "Jane Doe", "supported": True, "evidence_ids": ["e1"]})
    result = answer_episode(client, state.question, state.evidence, trace_id="synthetic/reader")
    assert result["answer"] == "Jane Doe"
    assert client.calls[0]["prompt_version"] == READER_VERSION
    assert client.calls[0]["messages"][0]["content"] == legacy.READER_PROMPT


@pytest.mark.parametrize("mode", ["fresh", "static", "memory"])
def test_examples_use_only_empty_or_single_action_and_round_trip(mode):
    prompt = planner_prompt(mode)
    assert "not an explanation, score or proof of sufficiency" in prompt
    assert "There must never be more than one action" in prompt
    assert legacy.PLANNER_PROMPT.split("Return exactly one JSON object")[0] in prompt
    examples = [json.loads(line) for line in prompt.splitlines() if line.startswith('{"reason"')]
    assert len(examples) == (3 if mode == "memory" else 2)
    for output in examples:
        assert (
            encode_wire_plan(normalize_wire_plan(json.dumps(output), mode=mode), mode=mode)
            == output
        )


def test_input_budget_still_rejects_before_transport(state):
    client = FakeClient(wire_stop())
    with pytest.raises(ValueError, match="prompt exceeds"):
        planner(client)(replace(state, question=RuntimeQuestion("q", "x" * 40000)))
    assert client.calls == []


def test_old_method_and_v2_files_remain_unchanged():
    root = Path(__file__).resolve().parents[1]
    assert execution_signature(root)["sha256"] == (
        "e9122473d934e50a6800d909e6e90fe0db97dfd15f49b51d9a549655f7089836"
    )
    frozen = {
        "src/growrag/experiments/operator_model_v2.py": (
            "03b71d0be5c1905161124e00f27bd85f547fa229c386ec16f0d62715c4e1a981"
        ),
        "src/growrag/experiments/operator_schemas.py": (
            "8ae38ca9b462edeae32c6e1fb6d108294f3910abd6faaccb2b483171840283a5"
        ),
    }
    for name, expected in frozen.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
