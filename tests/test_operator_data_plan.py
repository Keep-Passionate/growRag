"""Synthetic metadata tests; this module never reads real questions or gold."""

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from growrag.experiments.operator_data_plan import (
    SCHEMA,
    QuestionMetadata,
    build_operator_data_plan,
    question_metadata,
)


def _row(number, split="train", *, text=None):
    return question_metadata(
        f"{number:024x}", text or f"synthetic question {number}", official_split=split
    )


def _small(**kwargs):
    options = {"source_sizes": (2, 4), "calibration_count": 2, "evaluation_count": 3}
    options.update(kwargs)
    return build_operator_data_plan(
        [_row(i) for i in range(20)], [_row(i, "dev") for i in range(100, 120)], **options
    )


def test_default_plan_has_500_source_100_calibration_and_500_fixed_dev():
    train = [_row(i) for i in range(1000)]
    dev = [_row(i, "dev") for i in range(2000, 3000)]
    result = build_operator_data_plan(train, dev)
    assert result["schema_version"] == SCHEMA
    assert result["counts"] == {"source": 500, "calibration": 100, "evaluation": 500}
    assert result["official_splits"] == {
        "source": "train",
        "calibration": "train",
        "evaluation": "dev",
    }
    assert result["evaluation_protocol"] == "official-dev-subset"
    for size in (50, 100, 250, 500):
        assert result["nested_source_ids"][str(size)] == result["roles"]["source"][:size]
    flat = [qid for ids in result["roles"].values() for qid in ids]
    assert len(set(flat)) == 1100
    assert result["all_scales_share_calibration_and_evaluation"] is True
    assert result["evaluation_memory_updates_allowed"] is False
    assert result["calibration_memory_updates_allowed"] is False
    assert result["official_test_used"] is False


def test_sampling_reproducible_input_order_independent_and_json_serializable():
    train, dev = [_row(i) for i in range(30)], [_row(i, "dev") for i in range(50, 70)]
    options = dict(source_sizes=(2, 4), calibration_count=2, evaluation_count=3)
    first = build_operator_data_plan(train, dev, **options)
    reversed_plan = build_operator_data_plan(reversed(train), reversed(dev), **options)
    assert first == reversed_plan
    assert first == json.loads(json.dumps(first))
    changed = build_operator_data_plan(train, dev, seed="different", **options)
    assert changed["roles"] != first["roles"]


def test_existing_ids_and_equivalent_text_are_excluded_across_all_roles():
    train = [_row(i) for i in range(20)]
    dev = [_row(i, "dev") for i in range(100, 120)]
    dev.append(_row(200, "dev", text="SYNTHETIC QUESTION 3???"))
    blocked_hash = train[4].normalized_question_sha256
    result = build_operator_data_plan(
        train,
        dev,
        forbidden_ids=[train[3].question_id],
        forbidden_question_hashes=[blocked_hash],
        source_sizes=(2, 4),
        calibration_count=2,
        evaluation_count=3,
    )
    selected = {qid for ids in result["roles"].values() for qid in ids}
    assert not selected & {train[3].question_id, train[4].question_id, f"{200:024x}"}
    assert train[3].normalized_question_sha256 in result["forbidden_question_hashes"]


def test_normalized_duplicate_groups_excluded_not_split_or_randomly_kept():
    train = [_row(i) for i in range(20)] + [_row(40, text="same item"), _row(41, text="Same-item!")]
    dev = [_row(i, "dev") for i in range(100, 120)] + [_row(42, "dev", text="same item")]
    result = build_operator_data_plan(
        train, dev, source_sizes=(2, 4), calibration_count=2, evaluation_count=3
    )
    assert {f"{i:024x}" for i in (40, 41, 42)} <= set(result["excluded_question_ids"])
    assert len(result["duplicate_normalized_question_sha256s"]) == 1


def test_eligibility_controls_both_sources_and_default_calibration():
    eligible = {f"{i:024x}" for i in range(8)}
    result = _small(eligible_source_ids=eligible)
    assert set(result["roles"]["source"]) <= eligible
    assert set(result["roles"]["calibration"]) <= eligible


def test_calibration_only_pool_does_not_steal_scarce_source_slots():
    result = _small(
        eligible_source_ids={f"{i:024x}" for i in range(4)},
        eligible_calibration_ids={f"{i:024x}" for i in range(8)},
    )
    assert set(result["roles"]["source"]) == {f"{i:024x}" for i in range(4)}
    assert set(result["roles"]["calibration"]) <= {f"{i:024x}" for i in range(4, 8)}


def test_missing_dev_never_silently_falls_back_to_train():
    with pytest.raises(ValueError, match="dev metadata is missing"):
        build_operator_data_plan(
            [_row(i) for i in range(30)],
            source_sizes=(2, 4),
            calibration_count=2,
            evaluation_count=3,
        )


def test_train_heldout_requires_explicit_opt_in_and_cannot_claim_official_test():
    with pytest.raises(ValueError, match="explicit"):
        _small(evaluation_split="train_heldout")
    result = build_operator_data_plan(
        [_row(i) for i in range(30)],
        source_sizes=(2, 4),
        calibration_count=2,
        evaluation_count=3,
        evaluation_split="train_heldout",
        allow_train_heldout=True,
    )
    assert result["official_splits"]["evaluation"] == "train"
    assert "not-official-test" in result["evaluation_protocol"]
    assert result["evaluation_is_official_test"] is False
    assert len({qid for ids in result["roles"].values() for qid in ids}) == 9


def test_never_backfills_from_excluded_or_ineligible_questions():
    with pytest.raises(ValueError, match="source questions"):
        _small(eligible_source_ids={f"{i:024x}" for i in range(5)})
    with pytest.raises(ValueError, match="calibration questions"):
        _small(eligible_calibration_ids=())
    with pytest.raises(ValueError, match="evaluation questions"):
        _small(evaluation_count=21)
    with pytest.raises(ValueError, match="unknown eligible source"):
        _small(eligible_source_ids={f"{999:024x}"})


def test_hashes_identify_frozen_order_and_each_nested_source_prefix():
    result = _small()
    for role, ids in result["roles"].items():
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        assert result["role_ids_sha256"][role] == digest
    assert result["nested_source_ids_sha256"]["4"] == result["role_ids_sha256"]["source"]


def test_metadata_has_no_answer_score_type_difficulty_context_or_question_text_fields():
    row = _row(1)
    assert set(vars(row)) == {"question_id", "normalized_question_sha256", "official_split"}
    with pytest.raises(FrozenInstanceError):
        row.official_split = "dev"
    with pytest.raises(TypeError):
        QuestionMetadata(**vars(row), answer="forbidden")
    result = _small()
    assert result["selection_uses_answers_or_scores"] is False
    assert result["selection_uses_type_or_difficulty"] is False
    assert result["model_weight_training_planned"] is False


def test_train_and_dev_metadata_cannot_be_mislabeled_or_duplicate_ids():
    with pytest.raises(ValueError, match="only train"):
        build_operator_data_plan([_row(1, "dev")], [_row(100, "dev")])
    with pytest.raises(ValueError, match="duplicate question ID"):
        build_operator_data_plan([_row(1), _row(1)], [_row(100, "dev")])
    with pytest.raises(ValueError, match="overlaps"):
        build_operator_data_plan([_row(1)], [_row(1, "dev")])
    with pytest.raises(ValueError, match="train metadata is required"):
        build_operator_data_plan([], [_row(1, "dev")])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_sizes": ()},
        {"source_sizes": (4, 2)},
        {"source_sizes": (2, 2)},
        {"source_sizes": (True, 4)},
        {"source_sizes": [2, 4]},
        {"source_sizes": (0, 4)},
        {"calibration_count": True},
        {"evaluation_count": 0},
        {"seed": ""},
        {"seed": 12},
        {"evaluation_split": "test"},
        {"allow_train_heldout": 1},
        {"forbidden_ids": "a" * 24},
        {"forbidden_ids": ["bad"]},
        {"forbidden_question_hashes": ["bad"]},
        {"forbidden_question_hashes": "a" * 64},
    ],
)
def test_invalid_protocol_parameters_fail_closed(kwargs):
    with pytest.raises(ValueError):
        _small(**kwargs)


@pytest.mark.parametrize(
    "qid,fingerprint,split",
    [
        ("bad", "a" * 64, "train"),
        ("a" * 24, "bad", "train"),
        ("a" * 24, "a" * 64, "test"),
        ("a" * 24, "a" * 64, "validation"),
    ],
)
def test_invalid_metadata_is_rejected(qid, fingerprint, split):
    with pytest.raises(ValueError):
        QuestionMetadata(qid, fingerprint, split)


def test_normalization_matches_existing_registry():
    left = question_metadata("a" * 24, "ＡLPHA—Beta???", official_split="train")
    right = question_metadata("b" * 24, "alpha beta", official_split="dev")
    assert left.normalized_question_sha256 == right.normalized_question_sha256


def test_training_exclusion_list_always_contains_calibration_evaluation_and_old_reservations():
    old = f"{999:024x}"
    result = _small(forbidden_ids=[old])
    excluded = set(result["training_exclusion_question_ids"])
    assert excluded >= {old} | set(result["roles"]["calibration"]) | set(
        result["roles"]["evaluation"]
    )
    assert not set(result["roles"]["source"]) & excluded
