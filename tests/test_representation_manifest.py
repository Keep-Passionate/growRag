import json
from dataclasses import replace

import pytest

from growrag.experiments.data_protocol import normalize_question, role_for_question
from growrag.experiments.hotpot import parse_hotpot_example
from growrag.experiments.protocol import GoldRecord
from growrag.experiments.representation_manifest import (
    InsufficientRepresentationPoolError,
    build_representation_manifest,
)

SHA = "a" * 64


def sample(index, kind="bridge"):
    return parse_hotpot_example(
        {
            "_id": f"synthetic-{kind}-{index}",
            "question": f"Where did synthetic {kind} author number {index} live?",
            "answer": "SECRET ANSWER DO NOT EXPORT",
            "type": kind,
            "level": "hard",
            "context": [["PRIVATE SUPPORT TITLE", ["PRIVATE SUPPORT TEXT"]]],
            "supporting_facts": [["PRIVATE SUPPORT TITLE", 0]],
        },
        dataset="hotpotqa-distractor-train",
    )


@pytest.fixture(scope="module")
def examples():
    # Entire supplied synthetic population only; never claim downloaded data.
    return tuple(sample(index, kind) for index in range(1000) for kind in ("bridge", "comparison"))


def build(examples, **kwargs):
    options = {
        "source_sha256": SHA,
        "official_split": "train",
        "complete_train_declared": True,
        "source_count": 4,
        "source_cap": 8,
        "target_count": 4,
        "debug_count": 2,
    }
    options.update(kwargs)
    return build_representation_manifest(examples, **options)


def test_default_counts_and_existing_roles(examples):
    result = build_representation_manifest(
        examples, source_sha256=SHA, official_split="train", complete_train_declared=True
    )
    by_id = {example.question.question_id: example for example in examples}
    assert result["selected_counts"] == {
        "source": {"total": 64, "by_type": {"bridge": 32, "comparison": 32}},
        "target": {"total": 32, "by_type": {"bridge": 16, "comparison": 16}},
        "debug": {"total": 8, "by_type": {"bridge": 4, "comparison": 4}},
        "check": {"total": 24, "by_type": {"bridge": 12, "comparison": 12}},
    }
    assert len(result["source_expansion_order"]) == 256
    for name, qids in result["selected"].items():
        expected = "memory_seed" if name == "source" else "calibration_dev"
        assert all(role_for_question(by_id[qid].question.text) == expected for qid in qids)
    assert not set(result["selected"]["source"]) & set(result["selected"]["target"])
    assert not set(result["selected"]["debug"]) & set(result["selected"]["check"])
    assert result["selected"]["debug"] + result["selected"]["check"] == result["selected"]["target"]
    assert result["selector_train_used"] is False


def test_order_independent_and_source_expansion_preserves_prefix(examples):
    first = build(examples)
    assert first == build(tuple(reversed(examples)))
    expanded = build(examples, source_count=8)
    assert expanded["selected"]["source"][:4] == first["selected"]["source"]
    assert expanded["source_expansion_order"] == first["source_expansion_order"]
    assert expanded["selected"]["target"] == first["selected"]["target"]


def test_manifest_json_has_no_questions_answers_or_support(examples):
    result = build(examples)
    serialized = json.dumps(result)
    for text in ("SECRET ANSWER", "PRIVATE SUPPORT", examples[0].question.text, "supporting_facts"):
        assert text not in serialized
    assert json.loads(serialized) == result
    assert result["source_bytes_verified_by_builder"] is False
    assert result["checks"]["semantic_near_duplicates_checked"] is False
    assert result["checks"]["document_disjointness_checked"] is False
    assert not result["official_validation_used"] and not result["official_test_used"]


def test_exclusions_match_ids_or_normalized_text_and_report_unknowns(examples):
    initial = build(examples)
    by_id = {example.question.question_id: example for example in examples}
    source_id = initial["selected"]["source"][0]
    target_id = initial["selected"]["target"][0]
    target_text = "  " + by_id[target_id].question.text.upper().replace("?", "!!!") + "  "
    result = build(
        examples,
        excluded_question_ids=(source_id, "previously-exposed-not-in-file"),
        excluded_question_texts=(target_text, "This old question is not in the file"),
    )
    selected_ids = set(result["source_expansion_order"] + result["selected"]["target"])
    assert source_id not in selected_ids and target_id not in selected_ids
    assert result["exclusions"]["matched_by_id"] == [source_id]
    assert result["exclusions"]["matched_by_normalized_text"] == [target_id]
    assert result["exclusions"]["unmatched_requested_question_ids"] == [
        "previously-exposed-not-in-file"
    ]
    assert result["exclusions"]["unmatched_normalized_text_count"] == 1
    assert result["exclusions"]["excluded_union_count"] == 2
    assert target_text not in json.dumps(result)


def test_duplicate_ids_always_rejected_even_if_excluded(examples):
    with pytest.raises(ValueError, match="duplicate question IDs"):
        build(
            examples + (examples[0],),
            duplicate_policy="exclude_all",
            excluded_question_ids=(examples[0].question.question_id,),
        )


def test_normalized_duplicate_rejected_by_default_or_all_members_removed(examples):
    original = examples[0]
    alternate_question = replace(
        original.question, question_id="another-id", text=original.question.text.upper()
    )
    duplicate = replace(
        original, question=alternate_question, gold=replace(original.gold, question_id="another-id")
    )
    population = examples + (duplicate,)
    with pytest.raises(ValueError, match="duplicate normalized questions"):
        build(population)
    result = build(population, duplicate_policy="exclude_all")
    excluded = result["exclusions"]
    assert excluded["duplicate_normalized_member_count"] == 2
    assert excluded["excluded_union_count"] == 2
    assert excluded["duplicate_normalized_groups"][0]["question_ids"] == sorted(
        ["another-id", original.question.question_id]
    )
    assert original.question.text not in json.dumps(excluded)
    assert normalize_question(original.question.text) not in json.dumps(excluded)
    assert result == build(tuple(reversed(population)), duplicate_policy="exclude_all")


def test_insufficient_type_or_role_reports_counts_not_fallback(examples):
    only_bridges = tuple(example for example in examples if example.question_type == "bridge")
    with pytest.raises(InsufficientRepresentationPoolError) as error:
        build(only_bridges)
    assert error.value.counts["memory_seed"]["comparison"] == 0
    assert error.value.required["memory_seed"] == {"bridge": 4, "comparison": 4}
    without_calibration = tuple(
        example
        for example in examples
        if role_for_question(example.question.text) != "calibration_dev"
    )
    with pytest.raises(InsufficientRepresentationPoolError) as error:
        build(without_calibration)
    assert error.value.counts["calibration_dev"] == {"bridge": 0, "comparison": 0}


@pytest.mark.parametrize(
    "override",
    [
        {"official_split": "dev"},
        {"official_split": "test"},
        {"complete_train_declared": False},
        {"complete_train_declared": 1},
        {"source_sha256": "not-a-hash"},
        {"source_sha256": None},
        {"seed": True},
        {"seed": 43},
        {"source_count": 3},
        {"source_count": True},
        {"source_count": 10},
        {"source_cap": 3},
        {"source_cap": 0},
        {"target_count": 3},
        {"debug_count": 0},
        {"debug_count": 4},
        {"duplicate_policy": "keep_best"},
        {"excluded_question_ids": ["x"]},
        {"excluded_question_ids": ("x", "x")},
        {"excluded_question_texts": ("Same?", "SAME!")},
    ],
)
def test_invalid_options_rejected(examples, override):
    with pytest.raises(ValueError):
        build(examples, **override)


@pytest.mark.parametrize("population", [(), [], ("not-a-hotpot-example",)])
def test_non_tuple_or_incomplete_population_rejected(population):
    with pytest.raises(ValueError):
        build(population)


def test_gold_presence_identity_and_nonempty_fields_required(examples):
    first = examples[0]
    invalid = (
        replace(first, gold=None),
        replace(first, gold=replace(first.gold, question_id="wrong-id")),
        replace(
            first, gold=GoldRecord(first.question.question_id, (" ",), first.gold.supporting_facts)
        ),
        replace(first, gold=GoldRecord(first.question.question_id, ("answer",), ())),
        replace(first, candidate_context=[]),
        replace(first, question_type=None),
    )
    for replacement in invalid:
        with pytest.raises(ValueError):
            build((replacement, *examples[1:]))


def test_metadata_only_projection_has_identical_selection(examples):
    metadata_only = tuple(replace(example, candidate_context=()) for example in examples)
    assert build(metadata_only) == build(examples)


@pytest.mark.parametrize(
    "label", ["musique-train", "hotpotqa-dev", "hotpotqa-test", "hotpotqa-train-preview"]
)
def test_non_hotpot_or_non_complete_train_labels_rejected(examples, label):
    replacement = replace(examples[0], question=replace(examples[0].question, dataset=label))
    with pytest.raises(ValueError):
        build((replacement, *examples[1:]))


def test_mixed_dataset_labels_rejected(examples):
    replacement = replace(
        examples[0], question=replace(examples[0].question, dataset="hotpotqa-other-train")
    )
    with pytest.raises(ValueError, match="consistent source dataset"):
        build((replacement, *examples[1:]))
