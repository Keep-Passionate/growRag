"""Frozen prompt adapter checks using a zero-network fake client only."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from growrag.experiments.operator_model import (
    MAX_CANDIDATES,
    MAX_PROMPT_BYTES,
    MAX_VISIBLE_BYTES,
    MAX_VISIBLE_CHARS,
    PLANNER_VERSION,
    READER_VERSION,
    ModelOperatorPlanner,
    answer_episode,
    canonical_operator,
    planner_prompt,
    seed_specs,
    shortlist_specs,
    strict_object,
    visible_evidence,
)
from growrag.experiments.protocol import Evidence, GoldRecord, RuntimeQuestion
from growrag.macro_operators import GapField, OperatorSpec, QueryStep
from growrag.operator_bank import operator_to_dict
from growrag.operator_loop import Observation


class FakeClient:
    def __init__(self, output):
        self.output = json.dumps(output) if type(output) is dict else output
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return SimpleNamespace(content=self.output)


@pytest.fixture
def spec():
    return OperatorSpec(
        "LOOKUP",
        "1",
        ("lookup",),
        (GapField("term"),),
        (QueryStep("search", "{original_question} {term}"),),
    )


@pytest.fixture
def observation():
    return Observation(
        RuntimeQuestion("q1", "Where did the director live?"),
        (Evidence("e1", "Film", 0, "Jane Doe directed the film."),),
        (),
        2,
        1,
    )


def act(spec):
    return {
        "decision": "act",
        "reason": "Need the location",
        "intent": "lookup",
        "constraints": [],
        "selected_operator": None,
        "operator": operator_to_dict(spec),
        "gap": {"term": "location"},
        "bindings": [],
    }


def stop():
    return {
        "decision": "stop",
        "reason": "No useful next search",
        "intent": "lookup",
        "constraints": [],
        "selected_operator": None,
        "operator": None,
        "gap": {},
        "bindings": [],
    }


def selected(spec):
    output = act(spec)
    output.update(
        selected_operator={"operator_id": spec.operator_id, "version": spec.version}, operator=None
    )
    return output


def planner(client, **kwargs):
    return ModelOperatorPlanner(client, trace_prefix="test/run", **kwargs)


def payload(client):
    return json.loads(client.calls[-1]["messages"][1]["content"])


def test_fresh_constructs_same_grammar_without_reading_history(spec, observation):
    client = FakeClient(act(spec))
    proposal = planner(client, mode="fresh")(observation)
    assert proposal.origin == "fresh"
    assert proposal.spec == canonical_operator(spec)
    assert proposal.goal.original_question == observation.question.text
    assert payload(client)["candidate_specs"] == []
    assert client.calls[0]["messages"][0]["content"] == planner_prompt("fresh")
    assert client.calls[0]["trace_id"] == "test/run/plan/1"
    assert client.calls[0]["prompt_version"] == PLANNER_VERSION


def test_fresh_refuses_history_before_any_client_call(spec):
    client = FakeClient(stop())
    with pytest.raises(ValueError, match="FRESH"):
        planner(client, mode="fresh", specs=(spec,))
    assert client.calls == []


def test_static_default_is_only_seed_specs(observation):
    seed = seed_specs()[0]
    output = selected(seed)
    output["gap"] = {"search_terms": "location"}
    client = FakeClient(output)
    result = planner(client, mode="static")(observation)
    assert result.spec == seed
    assert result.origin == "static"
    assert len(payload(client)["candidate_specs"]) == 3


def test_static_rejects_altered_seed_or_extra_template(spec):
    with pytest.raises(ValueError, match="seed"):
        planner(FakeClient(stop()), mode="static", specs=(spec,))
    with pytest.raises(ValueError, match="seed"):
        planner(FakeClient(stop()), mode="static", specs=(replace(seed_specs()[0], version="2"),))


def test_static_cannot_create_a_new_spec(spec, observation):
    with pytest.raises(ValueError, match="cannot create"):
        planner(FakeClient(act(spec)), mode="static")(observation)


def test_memory_can_reuse_or_fall_back_to_fresh(spec, observation):
    reuse = planner(FakeClient(selected(spec)), mode="memory", specs=(spec,))(observation)
    fresh = planner(FakeClient(act(spec)), mode="memory", specs=(spec,))(observation)
    assert reuse.origin == "reuse" and reuse.spec == spec
    assert fresh.origin == "fresh" and fresh.spec == canonical_operator(spec)


def test_memory_cannot_select_unoffered_spec(spec, observation):
    with pytest.raises(ValueError, match="not offered"):
        planner(FakeClient(selected(spec)), mode="memory")(observation)


def test_repeated_candidate_id_is_rejected(spec):
    with pytest.raises(ValueError, match="duplicate"):
        planner(FakeClient(stop()), mode="memory", specs=(spec, spec))


def test_same_evidence_window_and_prompt_grammar_in_all_arms(spec, observation):
    evidence = tuple(Evidence(f"e{i}", "A document", i, "facts " * 7000) for i in range(6))
    state = replace(observation, evidence=evidence)
    visible = []
    for mode in ("fresh", "static", "memory"):
        client = FakeClient(stop())
        planner(client, mode=mode, specs=(spec,) if mode == "memory" else ())(state)
        visible.append(payload(client)["evidence"])
        assert client.calls[0]["messages"][0]["content"] == planner_prompt(mode)
        assert (
            len(
                json.dumps(
                    client.calls[0]["messages"], ensure_ascii=False, separators=(",", ":")
                ).encode()
            )
            <= MAX_PROMPT_BYTES
        )
    assert visible[0] == visible[1] == visible[2]
    assert len(visible[0]) == 6


def test_oversized_first_document_is_visible_and_audited():
    rows = (Evidence("e1", "First", 0, "x" * 60000), Evidence("e2", "Second", 0, "small"))
    visible = visible_evidence(rows)
    assert len(visible) == 2
    assert visible[0]["text"]
    assert visible[0]["window"]["text_truncated"] is True
    assert visible[0]["window"]["original_text_chars"] == 60000
    assert visible[0]["window"]["text_end"] == len(visible[0]["text"])
    assert rows[0].text == "x" * 60000


def test_unicode_and_escaping_respect_window_byte_and_char_limits():
    rows = tuple(Evidence(f"e{i}", "Unicode", i, '汉字\\"' * 20000) for i in range(6))
    visible = visible_evidence(rows)
    assert len(visible) == 6
    assert sum(len(row["title"]) + len(row["text"]) for row in visible) <= MAX_VISIBLE_CHARS
    assert (
        sum(
            len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())
            for row in visible
        )
        <= MAX_VISIBLE_BYTES
    )


def test_too_large_original_question_stops_before_api_call(observation):
    client = FakeClient(stop())
    state = replace(observation, question=RuntimeQuestion("q", "x" * 40000))
    with pytest.raises(ValueError, match="30000-byte"):
        planner(client, mode="fresh")(state)
    assert client.calls == []


def test_shortlist_is_bounded_stable_and_ignores_oversized_specs(spec):
    many = tuple(replace(spec, operator_id=f"LOOKUP_{i}") for i in range(5))
    large = replace(
        spec, operator_id="BIG", steps=tuple(QueryStep(f"s{i}", "word " * 300) for i in range(4))
    )
    actual = shortlist_specs("Where location?", (*reversed(many), large))
    assert len(actual) == MAX_CANDIDATES
    assert actual == shortlist_specs("Where location?", (*many, large))
    assert large not in actual
    with pytest.raises(ValueError, match="limit"):
        shortlist_specs("question", many, limit=4)


def test_runtime_payload_has_no_source_feedback_record(spec, observation):
    client = FakeClient(stop())
    planner(client, mode="memory", specs=(spec,))(observation)
    assert set(payload(client)) == {
        "mode",
        "original_question",
        "evidence",
        "evidence_window_omitted_count",
        "previous_queries",
        "remaining_retrievals",
        "candidate_specs",
        "candidate_shortlist_omitted_count",
    }
    assert "source_trace_ref" not in client.calls[0]["messages"][1]["content"]
    assert "gold_answer" not in client.calls[0]["messages"][1]["content"]


def test_labelled_question_is_not_accepted(observation):
    client = FakeClient(stop())
    gold = GoldRecord("q", ("SECRET_SOURCE_ANSWER",))
    with pytest.raises(TypeError, match="gold-free"):
        planner(client, mode="fresh")(replace(observation, question=gold))
    with pytest.raises(TypeError, match="gold-free"):
        answer_episode(client, gold, (), trace_id="test")
    assert client.calls == []


@pytest.mark.parametrize(
    "patch",
    [
        {"extra": True},
        {"gap": []},
        {"constraints": "not-array"},
        {"bindings": {}},
        {"decision": "maybe"},
        {"intent": 1},
        {"reason": 3},
    ],
)
def test_bad_plan_schema_is_rejected(spec, observation, patch):
    output = act(spec)
    output.update(patch)
    with pytest.raises((TypeError, ValueError)):
        planner(FakeClient(output), mode="fresh")(observation)


@pytest.mark.parametrize(
    "patch",
    [
        {"gap": {"term": "x"}},
        {"bindings": [{"name": "x"}]},
        {"operator": {}},
        {"selected_operator": {"operator_id": "X", "version": "1"}},
    ],
)
def test_stop_cannot_hide_an_action(observation, patch):
    output = stop()
    output.update(patch)
    with pytest.raises((TypeError, ValueError)):
        planner(FakeClient(output), mode="fresh")(observation)


@pytest.mark.parametrize(
    "content", ["[]", '{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "```json\n{}\n```"]
)
def test_model_output_is_strict_json(content):
    with pytest.raises(ValueError):
        strict_object(content)


def test_binding_must_be_in_visible_cited_text(spec, observation):
    output = act(spec)
    output["bindings"] = [{"name": "bridge", "value": "Jane Doe", "evidence_ids": ["e1"]}]
    result = planner(FakeClient(output), mode="fresh")(observation)
    assert result.bindings[0].value == "Jane Doe"
    output["bindings"][0]["value"] = "Never Appeared"
    with pytest.raises(ValueError, match="visible cited"):
        planner(FakeClient(output), mode="fresh")(observation)


def test_binding_cannot_use_unseen_tail_of_visible_document(spec, observation):
    state = replace(
        observation, evidence=(Evidence("e1", "Title", 0, "x" * 30000 + " Hidden Person"),)
    )
    output = act(spec)
    output["bindings"] = [{"name": "bridge", "value": "Hidden Person", "evidence_ids": ["e1"]}]
    with pytest.raises(ValueError, match="visible cited"):
        planner(FakeClient(output), mode="fresh")(state)


@pytest.mark.parametrize("ids", [["not-current"], [1], ["e1", "e1"], []])
def test_binding_citations_must_be_current_unique_nonempty(spec, observation, ids):
    output = act(spec)
    output["bindings"] = [{"name": "bridge", "value": "Jane Doe", "evidence_ids": ids}]
    with pytest.raises((TypeError, ValueError)):
        planner(FakeClient(output), mode="fresh")(observation)


def test_audit_callback_cannot_mutate_validated_model_output(spec, observation):
    def mutate(record):
        record["model_output"]["gap"]["term"] = "corrupted"
        record["payload"]["original_question"] = "corrupted"

    result = planner(FakeClient(act(spec)), mode="fresh", on_record=mutate)(observation)
    assert result.gap["term"] == "location"
    assert result.goal.original_question == observation.question.text


def test_reader_returns_claim_not_truth_and_records_window(observation):
    client = FakeClient({"answer": "Jane Doe", "supported": True, "evidence_ids": ["e1"]})
    result = answer_episode(client, observation.question, observation.evidence, trace_id="read/q1")
    assert result["support_is_model_claim"] is True
    assert result["visible_evidence_ids"] == ["e1"]
    assert result["evidence_windows"][0]["text_start"] == 0
    assert client.calls[0]["prompt_version"] == READER_VERSION


def test_reader_can_abstain_without_evidence(observation):
    client = FakeClient({"answer": "", "supported": False, "evidence_ids": []})
    result = answer_episode(client, observation.question, (), trace_id="read/q1")
    assert result["answer"] == ""
    assert result["visible_evidence_ids"] == []


@pytest.mark.parametrize(
    "patch",
    [
        {"answer": "", "supported": True},
        {"answer": "guess", "supported": False},
        {"evidence_ids": ["not-current"]},
        {"evidence_ids": ["e1", "e1"]},
        {"evidence_ids": [1]},
        {"evidence_ids": []},
        {"supported": "true"},
        {"answer": 5},
        {"gold": "leaked"},
    ],
)
def test_reader_rejects_invalid_support_claims(observation, patch):
    output = {"answer": "Jane Doe", "supported": True, "evidence_ids": ["e1"]}
    output.update(patch)
    with pytest.raises(ValueError):
        answer_episode(FakeClient(output), observation.question, observation.evidence, trace_id="r")


def test_canonical_identity_ignores_proposed_name_but_not_content(spec):
    assert canonical_operator(spec) == canonical_operator(
        replace(spec, operator_id="OTHER", version="7")
    )
    changed = replace(spec, steps=(QueryStep("search", "{term}"),))
    assert canonical_operator(spec).operator_id != canonical_operator(changed).operator_id


def test_v2_prompt_versions_and_mode_specific_examples_are_explicit(observation):
    assert PLANNER_VERSION == "growrag-operator-planner-v2"
    assert READER_VERSION == "growrag-operator-reader-v1"
    assert MAX_PROMPT_BYTES == 30000
    assert planner_prompt("fresh") == planner_prompt("memory")
    static = planner_prompt("static")
    assert "ONE\nshort sentence" in static
    assert "EXACTLY ONE mutually exclusive" in static
    assert "STATIC FINAL CHECK" in static
    assert '"operator_id":"ADD_TERM"' not in static
    examples = [
        strict_object(line) for line in static.splitlines() if line.startswith('{"decision"')
    ]
    assert len(examples) == 2
    assert all(item["operator"] is None for item in examples)
    for item in examples:
        planner(FakeClient(item), mode="static")(observation)


def test_v2_dynamic_response_examples_follow_existing_parser(observation):
    examples = [
        strict_object(line)
        for line in planner_prompt("fresh").splitlines()
        if line.startswith('{"decision"')
    ]
    assert len(examples) == 3
    assert examples[0]["decision"] == "stop"
    assert examples[1]["selected_operator"] is not None
    assert examples[2]["operator"] is not None
    for item in (examples[0], examples[2]):
        planner(FakeClient(item), mode="fresh")(observation)
    planner(FakeClient(examples[1]), mode="memory", specs=seed_specs())(observation)


def test_invalid_static_stop_plus_new_operator_is_not_guessed_into_act(spec, observation):
    output = stop()
    output["operator"] = operator_to_dict(spec)
    client = FakeClient(output)
    with pytest.raises(ValueError, match="stop cannot carry"):
        planner(client, mode="static")(observation)
    assert len(client.calls) == 1  # No corrective API retry and no inferred action.


def test_short_reason_instruction_does_not_add_brittle_sentence_length_gate(observation):
    output = stop()
    output["reason"] = "A" * 1000
    result = planner(FakeClient(output), mode="static")(observation)
    assert result.reason == output["reason"]  # Existing 2000-char safety bound remains.
