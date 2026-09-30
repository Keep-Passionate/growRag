"""Metadata-only, leakage-checked sampling for the dynamic-operator scale study.

中文：500题是最大建库来源数，不是500张已验证卡片；50/100/250/500共用一个
来源顺序、校准集和评测集。这里不读文件、不读答案、不训练模型，也不调用API。
导出器负责提供既有开发/封存题的ID与已有规范化hash，不能悄悄漏掉旧登记。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from .data_protocol import normalize_question

SCHEMA = "growrag-dynamic-operator-data-plan-v1"
SEED = "growrag-dynamic-operators-20260930-v1"
SOURCE_SIZES = (50, 100, 250, 500)
_ID = re.compile(r"[0-9a-f]{24}")
_SHA = re.compile(r"[0-9a-f]{64}")


def _validate(value: str, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"invalid {label}")


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class QuestionMetadata:
    """Only ID, normalized-text fingerprint and official split may enter sampling."""

    question_id: str
    normalized_question_sha256: str
    official_split: str

    def __post_init__(self) -> None:
        _validate(self.question_id, _ID, "Hotpot question ID")
        _validate(self.normalized_question_sha256, _SHA, "normalized question hash")
        if self.official_split not in {"train", "dev"}:
            raise ValueError("official_split must be train or dev, never test")


def question_metadata(question_id: str, question: str, *, official_split: str) -> QuestionMetadata:
    """For an already-approved ID/text projection; no labels/context are accepted.

    Do not call this on sealed old question text merely to reconstruct exclusions.
    Use an existing hash registry or the explicit conservative exclusion audit.
    """
    fingerprint = hashlib.sha256(normalize_question(question).encode()).hexdigest()
    return QuestionMetadata(question_id, fingerprint, official_split)


def _records(records: Iterable[QuestionMetadata], split: str) -> dict[str, QuestionMetadata]:
    result = {}
    for row in records:
        if not isinstance(row, QuestionMetadata) or row.official_split != split:
            raise ValueError(f"{split} metadata must contain only {split} QuestionMetadata")
        if row.question_id in result:
            raise ValueError("duplicate question ID in metadata")
        result[row.question_id] = row
    return result


def _id_set(values: Iterable[str], label: str) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a collection of IDs")
    result = set(values)
    for value in result:
        _validate(value, _ID, label)
    return result


def _eligible(values: Iterable[str] | None, available: set[str], label: str) -> set[str]:
    requested = available if values is None else _id_set(values, label)
    if not requested <= available:
        raise ValueError(f"unknown {label}; do not silently drop missing input IDs")
    return requested


def build_operator_data_plan(
    train_metadata: Iterable[QuestionMetadata],
    dev_metadata: Iterable[QuestionMetadata] = (),
    *,
    forbidden_ids: Iterable[str] = (),
    forbidden_question_hashes: Iterable[str] = (),
    eligible_source_ids: Iterable[str] | None = None,
    eligible_calibration_ids: Iterable[str] | None = None,
    source_sizes: tuple[int, ...] = SOURCE_SIZES,
    calibration_count: int = 100,
    evaluation_count: int = 500,
    seed: str = SEED,
    evaluation_split: str = "dev",
    allow_train_heldout: bool = False,
) -> dict:
    """Create an immutable-by-convention plan; the caller freezes its serialized SHA.

    Official train provides sources and calibration. Official dev provides the
    evaluation subset unless a caller explicitly declares a train-heldout study.
    Calibration is for threshold/prompt decisions, NEVER for building memory.
    The default calibration eligibility follows source eligibility so both can
    be restricted to question IDs whose contexts are in a frozen shared corpus.

    Duplicate normalized questions are removed as entire groups, not assigned
    by score or selected separately across roles. This checks exact normalized
    equality, NOT paraphrase, entity, document, or semantic disjointness.
    """
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("seed must be a nonempty string")
    if (
        not isinstance(source_sizes, tuple)
        or not source_sizes
        or any(type(size) is not int or size <= 0 for size in source_sizes)
        or tuple(sorted(set(source_sizes))) != source_sizes
    ):
        raise ValueError("source_sizes must be a strictly increasing positive integer tuple")
    if any(type(size) is not int or size <= 0 for size in (calibration_count, evaluation_count)):
        raise ValueError("calibration/evaluation counts must be positive integers")
    if type(allow_train_heldout) is not bool:
        raise ValueError("allow_train_heldout must be a boolean")
    if evaluation_split not in {"dev", "train_heldout"}:
        raise ValueError("evaluation_split must be dev or explicit train_heldout")
    if evaluation_split == "train_heldout" and not allow_train_heldout:
        raise ValueError("train-heldout requires explicit allow_train_heldout=True")

    train, dev = _records(train_metadata, "train"), _records(dev_metadata, "dev")
    if not train:
        raise ValueError("official train metadata is required")
    if evaluation_split == "dev" and not dev:
        raise ValueError("official dev metadata is missing; no automatic train fallback")
    if set(train) & set(dev):
        raise ValueError("question ID overlaps official train/dev inputs")
    metadata = train | dev
    source_eligible = _eligible(eligible_source_ids, set(train), "eligible source ID")
    calibration_eligible = (
        set(source_eligible)
        if eligible_calibration_ids is None
        else _eligible(eligible_calibration_ids, set(train), "eligible calibration ID")
    )

    blocked_ids = _id_set(forbidden_ids, "forbidden question ID")
    if isinstance(forbidden_question_hashes, (str, bytes)):
        raise ValueError("forbidden_question_hashes must be a collection")
    blocked_hashes = set(forbidden_question_hashes)
    for fingerprint in blocked_hashes:
        _validate(fingerprint, _SHA, "forbidden normalized question hash")
    # If a blocked ID is in the metadata, all text-equivalent IDs are blocked too.
    blocked_hashes.update(
        metadata[qid].normalized_question_sha256 for qid in blocked_ids & metadata.keys()
    )
    frequency = Counter(row.normalized_question_sha256 for row in metadata.values())
    duplicate_hashes = {fingerprint for fingerprint, count in frequency.items() if count > 1}
    excluded_hashes = blocked_hashes | duplicate_hashes
    excluded = blocked_ids | {
        qid for qid, row in metadata.items() if row.normalized_question_sha256 in excluded_hashes
    }

    def ordered(ids: Iterable[str], namespace: str) -> list[str]:
        return sorted(
            ids,
            key=lambda qid: (hashlib.sha256(f"{seed}:{namespace}:{qid}".encode()).hexdigest(), qid),
        )

    evaluation_pool = set(dev if evaluation_split == "dev" else train) - excluded
    evaluation = ordered(evaluation_pool, "evaluation")[:evaluation_count]
    if len(evaluation) != evaluation_count:
        raise ValueError("not enough independent evaluation questions; never replace exclusions")
    source_pool = source_eligible - excluded - set(evaluation)
    calibration_pool = calibration_eligible - excluded - set(evaluation)
    # Prefer calibration-only candidates before the overlap. This avoids stealing
    # scarce source candidates when the two eligibility pools differ.
    calibration = (
        ordered(calibration_pool - source_pool, "calibration")
        + ordered(calibration_pool & source_pool, "calibration")
    )[:calibration_count]
    if len(calibration) != calibration_count:
        raise ValueError("not enough independent calibration questions")
    source = ordered(source_pool - set(calibration), "source")[: source_sizes[-1]]
    if len(source) != source_sizes[-1]:
        raise ValueError("not enough independent source questions for largest nested size")

    roles = {"source": source, "calibration": calibration, "evaluation": evaluation}
    normalized = {
        role: [metadata[qid].normalized_question_sha256 for qid in ids]
        for role, ids in roles.items()
    }
    selected_ids = [qid for ids in roles.values() for qid in ids]
    selected_hashes = [fingerprint for hashes in normalized.values() for fingerprint in hashes]
    if len(set(selected_ids)) != len(selected_ids) or len(set(selected_hashes)) != len(
        selected_hashes
    ):
        raise ValueError("internal role overlap; plan rejected")
    return {
        "schema_version": SCHEMA,
        "seed": seed,
        "dataset": "HotpotQA",
        "configuration": "distractor",
        "roles": roles,
        "counts": {role: len(ids) for role, ids in roles.items()},
        "source_sizes": list(source_sizes),
        "nested_source_ids": {str(size): source[:size] for size in source_sizes},
        "official_splits": {
            "source": "train",
            "calibration": "train",
            "evaluation": "dev" if evaluation_split == "dev" else "train",
        },
        "evaluation_protocol": "official-dev-subset"
        if evaluation_split == "dev"
        else "explicit-train-heldout-subset-not-official-test",
        "official_test_used": False,
        "evaluation_is_official_test": False,
        "evaluation_memory_updates_allowed": False,
        "calibration_memory_updates_allowed": False,
        "model_weight_training_planned": False,
        "all_scales_share_calibration_and_evaluation": True,
        "selection_uses_answers_or_scores": False,
        "selection_uses_type_or_difficulty": False,
        "eligibility_counts": {
            "source": len(source_eligible),
            "calibration": len(calibration_eligible),
        },
        "eligibility_ids_sha256": {
            "source": _digest(sorted(source_eligible)),
            "calibration": _digest(sorted(calibration_eligible)),
        },
        "selected_normalized_question_sha256s": normalized,
        "role_ids_sha256": {role: _digest(ids) for role, ids in roles.items()},
        "nested_source_ids_sha256": {str(size): _digest(source[:size]) for size in source_sizes},
        "input_metadata_sha256": _digest(
            [
                (qid, row.normalized_question_sha256, row.official_split)
                for qid, row in sorted(metadata.items())
            ]
        ),
        "forbidden_question_ids": sorted(blocked_ids),
        "forbidden_question_hashes": sorted(blocked_hashes),
        "excluded_question_ids": sorted(excluded),
        "duplicate_normalized_question_sha256s": sorted(duplicate_hashes),
        "training_exclusion_question_ids": sorted(excluded | set(calibration) | set(evaluation)),
        "normalization_notice": "NFKC, casefold, alphanumeric tokens; exact normalized "
        "duplicates only. No claim of semantic, entity or document disjointness.",
        "source_notice": "Source counts are experience-collection questions, not card counts "
        "or model-gradient training. Reuse shared source trajectories across nested sizes; "
        "freeze each memory snapshot before evaluation. Caller must supply all prior "
        "exposure and reservation IDs/hashes, including sealed roles.",
    }
