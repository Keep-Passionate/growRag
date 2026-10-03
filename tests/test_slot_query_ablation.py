"""Synthetic slot-query ablation contracts; no model, real question, or gold is loaded.

这里只测试如何构造控制查询，不把夹具当实验数据。保存槽值不是只保存实体：
属性、关系等文本同样需要保留，且不得混入模板从未引用的字段。
"""

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from growrag.experiments import slot_query_ablation as ablation
from growrag.experiments.shared_s2g_corpus import CorpusDocument

QUESTION = "When was the fictional academy attended by Avery founded?"
OPERATOR_ID = "SYNTHETIC_FOUNDED"


def proposal(template="{entity} {property}", gap=None):
    """One sealed-style reuse proposal; names and facts are deliberately fictional."""
    return {
        "goal": {"original_question": QUESTION, "intent": "lookup", "constraints": []},
        "origin": "reuse",
        "spec": {
            "operator_id": OPERATOR_ID,
            "version": "1",
            "supported_intents": ["lookup"],
            "gap_schema": [
                {"name": "entity", "kind": "text", "required": True},
                {"name": "property", "kind": "text", "required": True},
            ],
            "steps": [
                {"step_id": "find", "template": template, "when": [], "requires_bindings": []}
            ],
        },
        "gap": {"entity": "Avery Academy", "property": "founded year"} if gap is None else gap,
        "bindings": [],
        "reason": "The academy founding year is missing.",
    }


def saved_query(item):
    """Fixture construction only; production must independently verify saved_query."""
    values = {"original_question": QUESTION, **item["gap"]}
    return item["spec"]["steps"][0]["template"].format_map(values)


def test_fixed_slot_values_create_three_exact_views():
    item = proposal()
    result = ablation.query_variants(QUESTION, item, saved_query(item))
    expected = {
        "queries": {
            "history_actual": "Avery Academy founded year",
            "slots_only": "Avery Academy founded year",
            "query_plus_slots": QUESTION + " Avery Academy founded year",
        },
        "slot_names": ["entity", "property"],
        "slot_values": ["Avery Academy", "founded year"],
        "uses_original_question": False,
        "template": "{entity} {property}",
        "operator_id": OPERATOR_ID,
    }
    assert {key: result[key] for key in expected} == expected
    assert result["structure_group"] == "slot_passthrough"


def test_slots_follow_first_template_reference_not_gap_or_schema_order():
    item = proposal("{property} {entity} {property}")
    result = ablation.query_variants(QUESTION, item, saved_query(item))
    assert result["slot_names"] == ["property", "entity"]
    assert result["slot_values"] == ["founded year", "Avery Academy"]
    assert result["queries"]["slots_only"] == "founded year Avery Academy"


def test_repeated_slots_do_not_repeat_control_values():
    item = proposal("{entity} founded {property} {entity} {property}")
    result = ablation.query_variants(QUESTION, item, saved_query(item))
    assert result["queries"]["history_actual"] == (
        "Avery Academy founded founded year Avery Academy founded year"
    )
    assert result["queries"]["slots_only"] == "Avery Academy founded year"
    assert result["slot_names"] == ["entity", "property"]


def test_different_slots_with_equal_values_are_not_deduplicated_by_content():
    item = proposal(gap={"entity": "same text", "property": "same text"})
    result = ablation.query_variants(QUESTION, item, saved_query(item))
    assert result["slot_names"] == ["entity", "property"]
    assert result["slot_values"] == ["same text", "same text"]
    assert result["queries"]["slots_only"] == "same text same text"


def test_original_question_is_special_source_not_a_control_slot():
    item = proposal("{original_question} {entity} nicknames aliases")
    result = ablation.query_variants(QUESTION, item, saved_query(item))
    assert result["uses_original_question"] is True
    assert result["slot_names"] == ["entity"]
    assert result["slot_values"] == ["Avery Academy"]
    assert result["queries"]["history_actual"] == QUESTION + " Avery Academy nicknames aliases"
    assert result["queries"]["slots_only"] == "Avery Academy"
    assert result["queries"]["query_plus_slots"] == QUESTION + " Avery Academy"


def test_unused_gap_values_do_not_leak_into_the_control_query():
    item = proposal("{entity} aliases", {"entity": "Avery Academy", "unused": "UNUSED_FACT"})
    item["spec"]["gap_schema"] = [
        {"name": "entity", "kind": "text", "required": True},
        {"name": "unused", "kind": "text", "required": False},
    ]
    result = ablation.query_variants(QUESTION, item, saved_query(item))
    assert result["slot_names"] == ["entity"]
    assert result["slot_values"] == ["Avery Academy"]
    assert all("UNUSED_FACT" not in text for text in result["queries"].values())


def test_attribute_slot_is_preserved_and_not_mislabeled_as_entity_only():
    item = proposal("{property} lookup", {"property": "birth date"})
    item["spec"]["gap_schema"] = [{"name": "property", "kind": "text", "required": True}]
    result = ablation.query_variants(QUESTION, item, saved_query(item))
    assert result["slot_names"] == ["property"]
    assert result["slot_values"] == ["birth date"]
    assert result["queries"]["slots_only"] == "birth date"
    assert "entity" not in result


def test_gap_slot_literal_braces_are_not_reinterpreted_as_format_syntax():
    item = proposal("{entity} aliases", {"entity": "Avery {Academy}"})
    result = ablation.query_variants(QUESTION, item, saved_query(item))
    assert result["queries"]["slots_only"] == "Avery {Academy}"


def test_escaped_template_braces_are_literal_not_slots():
    item = proposal("{{reference}} {entity}")
    result = ablation.query_variants(QUESTION, item, saved_query(item))
    assert result["template"] == "{{reference}} {entity}"
    assert result["queries"]["history_actual"] == "{reference} Avery Academy"
    assert result["slot_names"] == ["entity"]
    assert result["queries"]["slots_only"] == "Avery Academy"


def test_query_construction_does_not_mutate_sealed_proposal():
    item = proposal("{entity} founded {property} {entity}")
    before = deepcopy(item)
    result = ablation.query_variants(QUESTION, item, saved_query(item))
    assert item == before
    result["slot_names"].append("new")
    result["slot_values"].append("new")
    assert item == before


@pytest.mark.parametrize(
    ("template", "group", "identical_control"),
    [
        ("{entity} aliases", "slots_with_fixed_words", None),
        ("{original_question} {entity}", "question_plus_slots", "query_plus_slots"),
        ("{entity}", "slot_passthrough", "slots_only"),
    ],
)
def test_three_template_groups_expose_their_query_negative_control(
    template, group, identical_control
):
    item = proposal(template)
    result = ablation.query_variants(QUESTION, item, saved_query(item))
    assert result["structure_group"] == group
    if identical_control is None:
        assert result["queries"]["history_actual"] != result["queries"]["slots_only"]
        assert result["queries"]["history_actual"] != result["queries"]["query_plus_slots"]
    else:
        assert result["queries"]["history_actual"] == result["queries"][identical_control]


@pytest.mark.parametrize("actual", ["Avery Academy founded year ", "avery Academy founded year"])
def test_saved_query_reconstruction_is_exact_not_normalized(actual):
    with pytest.raises(ValueError):
        ablation.query_variants(QUESTION, proposal(), actual)


@pytest.mark.parametrize(
    "template",
    [
        "{entity.name}",
        "{entity[0]}",
        "{entity!s}",
        "{entity!r}",
        "{entity:>20}",
        "{entity:{property}}",
        "{}",
        "{0}",
        "{entity-name}",
        "{entity",
        "entity}",
    ],
)
def test_nonidentifier_conversion_formatting_and_broken_templates_are_rejected(template):
    item = proposal(template)
    with pytest.raises(ValueError):
        ablation.query_variants(QUESTION, item, "never used")


@pytest.mark.parametrize("template", ["literal query only", "{original_question}"])
def test_no_control_slot_cannot_be_silently_treated_as_no_action(template):
    item = proposal(template)
    with pytest.raises(ValueError):
        ablation.query_variants(QUESTION, item, saved_query(item))


def test_missing_slot_cannot_be_filled_with_empty_string():
    item = proposal(gap={"entity": "Avery Academy"})
    with pytest.raises(ValueError):
        ablation.query_variants(QUESTION, item, "Avery Academy ")


@pytest.mark.parametrize("invalid", [None, True, 23, [], {}, "", "   "])
def test_every_gap_value_requires_nonempty_text(invalid):
    item = proposal(gap={"entity": "Avery Academy", "property": invalid})
    with pytest.raises(ValueError):
        ablation.query_variants(QUESTION, item, "Avery Academy founded year")


def test_original_question_cannot_be_shadowed_in_gap():
    item = proposal(gap={"entity": "Avery Academy", "original_question": "OTHER_QUESTION"})
    with pytest.raises(ValueError):
        ablation.query_variants(QUESTION, item, "Avery Academy founded year")


@pytest.mark.parametrize("invalid", [None, "fresh", "static", "stop"])
def test_proposal_must_be_actual_reuse_origin(invalid):
    item = proposal()
    item["origin"] = invalid
    with pytest.raises(ValueError):
        ablation.query_variants(QUESTION, item, saved_query(item))


def test_proposal_must_preserve_exact_original_question():
    item = proposal()
    item["goal"]["original_question"] = QUESTION + " "
    with pytest.raises(ValueError):
        ablation.query_variants(QUESTION, item, saved_query(item))


@pytest.mark.parametrize("count", [0, 2])
def test_multistep_or_empty_operator_is_outside_this_ablation_contract(count):
    item = proposal()
    item["spec"]["steps"] = item["spec"]["steps"] * count
    with pytest.raises(ValueError):
        ablation.query_variants(QUESTION, item, "Avery Academy founded year")


@pytest.mark.parametrize("field", ["when", "requires_bindings"])
def test_conditional_or_bound_operator_is_not_silently_flattened(field):
    item = proposal()
    item["spec"]["steps"][0][field] = ["nonempty"]
    with pytest.raises(ValueError):
        ablation.query_variants(QUESTION, item, saved_query(item))


@pytest.mark.parametrize("field", ["when", "requires_bindings"])
def test_missing_step_constraint_array_is_not_treated_as_empty(field):
    item = proposal()
    del item["spec"]["steps"][0][field]
    with pytest.raises(ValueError):
        ablation.query_variants(QUESTION, item, saved_query(item))


class FakeIndex:
    """In-memory index contract fixture, deliberately separate from all real corpora."""

    def __init__(self, documents, returned):
        self.documents = {d.doc_id: d for d in documents}
        self.returned = returned
        self.calls = []

    def __call__(self, query, top_k):
        self.calls.append((query, top_k))
        return tuple(d.as_author_document() for d in self.returned[query])

    def document(self, identity):
        return self.documents[identity]


def replay_fixture(*, alternate_order=False, zero_match=False):
    first = CorpusDocument("d0", "Fictional initial source", ("Initial source sentence.",))
    second = CorpusDocument("d1", "Fictional repair source", ("Second source sentence.",))
    third = CorpusDocument("d2", "Fictional repair source two", ("Third source sentence.",))
    actual = () if zero_match else (second, third)
    control = tuple(reversed(actual)) if alternate_order else actual
    plan = {
        "question_id": "synthetic-question",
        "question": QUESTION,
        "queries": {
            ablation.H_ACTUAL: "ALPHA beta",
            ablation.SLOTS: "beta alpha alpha",
            ablation.Q_SLOTS: QUESTION + " alpha beta",
        },
    }
    returned = {
        QUESTION: (first,),
        plan["queries"][ablation.H_ACTUAL]: actual,
        plan["queries"][ablation.SLOTS]: control,
        plan["queries"][ablation.Q_SLOTS]: actual,
    }
    previous = {
        "initial": {"documents": [asdict(first)]},
        "methods": {ablation.HISTORY: {"repair_documents": [asdict(d) for d in actual]}},
    }
    return FakeIndex((first, second, third), returned), plan, previous


def test_equal_normalized_term_sets_have_identical_ranked_documents():
    index, plan, previous = replay_fixture()
    result = ablation.replay_plan(index, plan, previous)
    actual, control = result["variants"][ablation.H_ACTUAL], result["variants"][ablation.SLOTS]
    assert actual["normalized_unique_terms"] == control["normalized_unique_terms"]
    assert actual["repair_documents"] == control["repair_documents"]
    assert actual["documents"] == control["documents"]
    assert result["local_retrieval_calls"] == 4
    assert index.calls == [(QUESTION, 6)] + [
        (plan["queries"][name], 6) for name in ablation.VARIANTS
    ]


def test_equal_normalized_terms_with_different_rank_order_are_rejected():
    index, plan, previous = replay_fixture(alternate_order=True)
    with pytest.raises(ValueError, match="same normalized BM25 term set"):
        ablation.replay_plan(index, plan, previous)


def test_legal_zero_match_is_retained_not_dropped_or_given_a_fake_document():
    index, plan, previous = replay_fixture(zero_match=True)
    result = ablation.replay_plan(index, plan, previous)
    assert result["local_retrieval_calls"] == 4
    for item in result["variants"].values():
        assert item["repair_documents"] == []
        assert item["new_document_count"] == 0
        assert item["documents"] == result["initial"]


@pytest.mark.parametrize("field", ["title", "text"])
def test_retrieval_callback_must_match_exact_source_version(field):
    original = CorpusDocument("d1", "Fictional source", ("Original source sentence.",))
    changed = CorpusDocument(
        original.doc_id,
        "Changed title" if field == "title" else original.title,
        ("Changed source sentence.",) if field == "text" else original.sentences,
    )
    index = FakeIndex((original,), {"synthetic query": (changed,)})
    with pytest.raises(ValueError, match="callback/source"):
        ablation.retrieve(index, "synthetic query")


def test_run_does_not_overwrite_existing_output(tmp_path, monkeypatch):
    target = tmp_path / ablation.OUTPUT
    target.mkdir(parents=True)
    marker = target / "original.txt"
    marker.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        ablation, "load_source", lambda *args: pytest.fail("must stop before inputs")
    )
    with pytest.raises(FileExistsError):
        ablation.run(tmp_path)
    assert marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("output", ["../outside", "runs/nested/output", "knowledge/output"])
def test_run_output_is_a_direct_runs_child(tmp_path, output, monkeypatch):
    monkeypatch.setattr(
        ablation, "load_source", lambda *args: pytest.fail("must stop before inputs")
    )
    with pytest.raises(ValueError, match="direct runs child"):
        ablation.run(tmp_path, output)


def test_queries_are_sealed_before_retrieval_and_retrieval_failure_keeps_gold_closed(
    tmp_path, monkeypatch
):
    """The only mocked object is a closeable index; no synthetic QA score is produced."""
    (tmp_path / "runs").mkdir()
    index = type("CloseableIndex", (), {"closed": False, "close": lambda self: None})()

    def close():
        index.closed = True

    index.close = close
    plan = {"question_id": "synthetic", "question": QUESTION, "queries": {}}
    manifest = {
        "corpus_ref": {"index_path": "index", "path": "corpus", "sha256": "a" * 64, "rows": 1}
    }
    monkeypatch.setattr(
        ablation,
        "load_source",
        lambda *args: ([plan], {"synthetic": {}}, manifest, {}),
    )
    monkeypatch.setattr(ablation, "SharedBM25Index", lambda *args: index)

    def fail_retrieval(*args):
        target = tmp_path / ablation.OUTPUT
        stored_plans = json.loads((target / "PLANS.json").read_text(encoding="utf-8"))
        seal = json.loads((target / "plans_frozen.json").read_text(encoding="utf-8"))
        assert stored_plans == [plan]
        assert seal["files"]["PLANS.json"] == ablation._sha(target / "PLANS.json")
        raise ValueError("synthetic ranking mismatch")

    monkeypatch.setattr(ablation, "replay_plan", fail_retrieval)
    monkeypatch.setattr(
        ablation,
        "load_gold",
        lambda *args: pytest.fail("gold must remain closed after retrieval failure"),
    )
    with pytest.raises(ValueError, match="synthetic ranking mismatch"):
        ablation.run(tmp_path)
    target = tmp_path / ablation.OUTPUT
    failure = json.loads((target / "FAILED.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "retrieval"
    assert failure["error"] == "synthetic ranking mismatch"
    assert index.closed is True
    assert not (target / "SUMMARY.json").exists()
    assert not (target / "TERMINAL.json").exists()
