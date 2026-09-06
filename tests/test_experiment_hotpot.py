"""Synthetic format fixtures, not sampled HotpotQA benchmark questions."""

import json
from dataclasses import asdict

import pytest

from growrag.experiments.hotpot import (
    evaluate_layered_feedback,
    load_hotpot,
    parse_hotpot_example,
)
from growrag.experiments.protocol import Answer, Evidence, GoldRecord


def row():
    return {
        "_id": "synthetic-question",
        "question": "Where was the author born?",
        "context": [["Synthetic Book", ["The author is Person A.", ""]]],
        "answer": "GOLD-CANARY-LAND",
        "supporting_facts": [["Synthetic Book", 0], ["Missing Person Page", 1]],
        "type": "bridge",
        "level": "hard",
    }


def test_gold_and_analysis_metadata_do_not_enter_runtime_question():
    item = parse_hotpot_example(row(), dataset="synthetic-format-test")
    runtime_json = json.dumps(asdict(item.question))
    assert "GOLD-CANARY" not in runtime_json
    assert "supporting_facts" not in runtime_json
    assert "bridge" not in runtime_json
    assert "hard" not in runtime_json
    assert item.gold.answers == ("GOLD-CANARY-LAND",)
    assert len(item.candidate_context) == 1
    assert item.gold.supporting_facts[-1] == ("Missing Person Page", 1)


def test_hidden_test_has_no_fabricated_gold():
    raw = row()
    del raw["answer"], raw["supporting_facts"]
    assert parse_hotpot_example(raw, dataset="hidden-test").gold is None


def test_sentence_indices_are_preserved_after_blank_sentence():
    raw = row()
    raw["context"][0][1].append("A later sentence.")
    item = parse_hotpot_example(raw, dataset="synthetic")
    assert [sentence.sentence_id for sentence in item.candidate_context] == [0, 2]


def test_duplicate_question_id_rejected(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps([row(), row()]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_hotpot(path, dataset="synthetic")


def test_new_evidence_and_wrong_answer_are_separate_observations():
    item = parse_hotpot_example(row(), dataset="synthetic")
    extra = Evidence("new", "Missing Person Page", 1, "Person A was born in Country B.")
    feedback = evaluate_layered_feedback(
        item.question,
        item.candidate_context,
        (*item.candidate_context, extra),
        Answer("wrong", ("new",)),
        item.gold,
    )
    assert feedback.answer_em == 0
    assert feedback.retrieved_gold_support_recall == 1
    assert feedback.new_gold_support == (("Missing Person Page", 1),)
    assert feedback.answer_supported is None


def test_correct_answer_does_not_imply_new_support_or_entailment():
    item = parse_hotpot_example(row(), dataset="synthetic")
    feedback = evaluate_layered_feedback(
        item.question,
        item.candidate_context,
        item.candidate_context,
        Answer("GOLD-CANARY-LAND", ("nonexistent",)),
        item.gold,
    )
    assert feedback.answer_em == 1
    assert feedback.new_gold_support == ()
    assert not feedback.citation_ids_resolve
    assert feedback.cited_gold_support_precision == 0
    assert feedback.answer_supported is None


def test_yes_no_answer_f1_special_case_and_wrong_gold_id():
    item = parse_hotpot_example(row(), dataset="synthetic")
    gold = GoldRecord(item.question.question_id, ("yes",), ())
    feedback = evaluate_layered_feedback(item.question, (), (), Answer("yes indeed"), gold)
    assert feedback.answer_f1 == 0
    assert feedback.retrieved_gold_support_recall is None
    with pytest.raises(ValueError, match="different question"):
        evaluate_layered_feedback(
            item.question, (), (), Answer("yes"), GoldRecord("other", ("yes",))
        )


@pytest.mark.parametrize("fact", [["Title", True], ["Title", -1], ["Title"], ["", 0]])
def test_malformed_gold_fact_rejected(fact):
    raw = row()
    raw["supporting_facts"] = [fact]
    with pytest.raises(ValueError):
        parse_hotpot_example(raw, dataset="synthetic")
