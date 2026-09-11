"""Offline, train-only stratified IDs for the representation-selection pilot.

This module never downloads data or calls a model. The caller declares that the
complete train file was loaded and supplies its SHA-256; this function cannot
verify those provenance claims without the source bytes. Gold is validated but
is never copied to the returned, JSON-serializable manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

from .data_protocol import DATASET_ID, ROLES, SPLIT_VERSION, normalize_question, role_for_question
from .hotpot import HotpotExample
from .protocol import Evidence, GoldRecord, RuntimeQuestion

SCHEMA_VERSION = "growrag-representation-manifest-v1"
QUESTION_TYPES = ("bridge", "comparison")


class InsufficientRepresentationPoolError(ValueError):
    """Requested role/type quotas cannot be met without changing the protocol."""

    def __init__(self, counts: dict, required: dict) -> None:
        self.counts = counts
        self.required = required
        super().__init__(
            "insufficient role/type pools: "
            + json.dumps({"available": counts, "required": required}, sort_keys=True)
        )


@dataclass(frozen=True, slots=True)
class _Row:
    question_id: str
    normalized_question: str
    question_type: str
    role: str


def _digest(values: list[str]) -> str:
    encoded = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _question_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ids(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must be a tuple of nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate entries")
    return values


def _validate_example(example: HotpotExample, *, seed: int) -> _Row:
    if not isinstance(example, HotpotExample) or not isinstance(example.question, RuntimeQuestion):
        raise ValueError("examples must contain complete HotpotExample objects")
    question = example.question
    if not isinstance(question.dataset, str) or "hotpot" not in question.dataset.casefold():
        raise ValueError("each question must declare a HotpotQA dataset")
    if re.search(
        r"(?:^|[-_/\s])(dev|validation|test|preview|subset)(?:$|[-_/\s])",
        question.dataset.casefold(),
    ):
        raise ValueError("dataset label denotes dev/test/preview/subset, not complete train")
    if example.question_type not in QUESTION_TYPES:
        raise ValueError(f"{question.question_id}: question_type must be bridge or comparison")
    if not isinstance(example.candidate_context, tuple) or not all(
        isinstance(item, Evidence) for item in example.candidate_context
    ):
        raise ValueError(f"{question.question_id}: candidate_context must be an Evidence tuple")
    gold = example.gold
    if not isinstance(gold, GoldRecord) or gold.question_id != question.question_id:
        raise ValueError(f"{question.question_id}: matching gold is required")
    if not gold.answers or any(not answer.strip() for answer in gold.answers):
        raise ValueError(f"{question.question_id}: nonempty gold answers are required")
    if not gold.supporting_facts:
        raise ValueError(f"{question.question_id}: gold supporting facts are required")
    # Context may be omitted in an offline metadata-only projection of a complete
    # file. This builder never needs its text or allocates sentence indices.
    # The parser permits missing support in fullwiki context. Keep that contract:
    # matching here means gold/question identity, not a claim of semantic truth.
    normalized = normalize_question(question.text)
    return _Row(
        question.question_id,
        normalized,
        example.question_type,
        role_for_question(question.text, seed=seed),
    )


def _type_counts(rows: list[_Row]) -> dict[str, int]:
    counts = Counter(row.question_type for row in rows)
    return {kind: counts[kind] for kind in QUESTION_TYPES}


def _balanced_order(rows: list[_Row], *, seed: int) -> list[_Row]:
    strata = {
        kind: sorted(
            (row for row in rows if row.question_type == kind),
            key=lambda row: (
                hashlib.sha256(f"selection:{seed}:{row.question_id}".encode()).hexdigest(),
                row.question_id,
            ),
        )
        for kind in QUESTION_TYPES
    }
    # Interleave to ensure every even prefix remains balanced. Concatenating the
    # two strata would replace earlier selected sources when source_count grows.
    pairs = zip(*(strata[kind] for kind in QUESTION_TYPES), strict=False)
    return [row for pair in pairs for row in pair]


def build_representation_manifest(
    examples: tuple[HotpotExample, ...],
    *,
    source_sha256: str,
    official_split: str,
    complete_train_declared: bool,
    seed: int = 42,
    source_count: int = 64,
    source_cap: int = 256,
    target_count: int = 32,
    debug_count: int = 8,
    excluded_question_ids: tuple[str, ...] = (),
    excluded_question_texts: tuple[str, ...] = (),
    duplicate_policy: Literal["reject", "exclude_all"] = "reject",
) -> dict:
    """Freeze balanced source and target IDs using the existing train roles.

    ``source_cap`` reserves a complete balanced expansion order, not permission
    to run or pay for it. Rebuilding with a larger ``source_count`` preserves the
    earlier prefix when the source file, exclusions, seed and cap are unchanged.
    Targets use calibration_dev only; selector_train is retained but unused.
    ``exclude_all`` drops every member of each normalized-question duplicate
    group, never choosing a winner by its answer or outcome. Duplicate IDs are
    always invalid. Exact/normalized checks do not detect semantic near copies.
    """
    if not isinstance(examples, tuple) or not examples:
        raise ValueError("examples must be a nonempty tuple loaded from complete official train")
    if official_split != "train" or complete_train_declared is not True:
        raise ValueError("only explicitly declared complete official train is accepted")
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_sha256):
        raise ValueError("source_sha256 must be a 64-digit SHA-256 hexadecimal declaration")
    if type(seed) is not int or seed != 42:
        raise ValueError("this frozen protocol requires seed=42")
    for name, count in (
        ("source_count", source_count),
        ("source_cap", source_cap),
        ("target_count", target_count),
        ("debug_count", debug_count),
    ):
        if type(count) is not int or count <= 0 or count % 2:
            raise ValueError(f"{name} must be a positive even integer")
    if source_count > source_cap:
        raise ValueError("source_count cannot exceed the predeclared source_cap")
    if debug_count >= target_count:
        raise ValueError("debug_count must leave a positive even check batch")
    if duplicate_policy not in ("reject", "exclude_all"):
        raise ValueError("duplicate_policy must be reject or exclude_all")
    excluded_ids = set(_ids(excluded_question_ids, "excluded_question_ids"))
    raw_excluded_texts = _ids(excluded_question_texts, "excluded_question_texts")
    excluded_texts = {normalize_question(text) for text in raw_excluded_texts}
    if len(excluded_texts) != len(raw_excluded_texts):
        raise ValueError("excluded_question_texts contains normalized duplicates")

    rows = [_validate_example(example, seed=seed) for example in examples]
    identifiers = [row.question_id for row in rows]
    duplicated_ids = sorted(qid for qid, count in Counter(identifiers).items() if count > 1)
    if duplicated_ids:
        raise ValueError("duplicate question IDs: " + json.dumps(duplicated_ids))
    if len({example.question.dataset for example in examples}) != 1:
        raise ValueError("examples must use one consistent source dataset label")
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row.normalized_question].append(row.question_id)
    duplicate_groups = sorted(
        (
            {"normalized_question_sha256": _question_digest(text), "question_ids": sorted(ids)}
            for text, ids in groups.items()
            if len(ids) > 1
        ),
        key=lambda group: group["normalized_question_sha256"],
    )
    if duplicate_groups and duplicate_policy == "reject":
        raise ValueError("duplicate normalized questions: " + json.dumps(duplicate_groups))
    duplicate_member_ids = {qid for group in duplicate_groups for qid in group["question_ids"]}
    matches_by_id = {row.question_id for row in rows if row.question_id in excluded_ids}
    matches_by_text = {row.question_id for row in rows if row.normalized_question in excluded_texts}
    removed_ids = duplicate_member_ids | matches_by_id | matches_by_text
    eligible = [row for row in rows if row.question_id not in removed_ids]
    role_rows = {role: [row for row in eligible if row.role == role] for role in ROLES}
    available = {role: _type_counts(items) for role, items in role_rows.items()}
    required = {
        "memory_seed": {kind: source_cap // 2 for kind in QUESTION_TYPES},
        "calibration_dev": {kind: target_count // 2 for kind in QUESTION_TYPES},
    }
    if any(
        available[role][kind] < count
        for role, need in required.items()
        for kind, count in need.items()
    ):
        raise InsufficientRepresentationPoolError(available, required)
    source_order = _balanced_order(role_rows["memory_seed"], seed=seed)[:source_cap]
    targets = _balanced_order(role_rows["calibration_dev"], seed=seed)[:target_count]
    selected = {
        "source": source_order[:source_count],
        "target": targets,
        "debug": targets[:debug_count],
        "check": targets[debug_count:],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET_ID,
        "input_dataset_label": examples[0].question.dataset,
        "official_split": "train",
        "source_sha256": source_sha256.lower(),
        "complete_train_declared": True,
        "source_bytes_verified_by_builder": False,
        "provenance_notice": (
            "Caller declares complete train and source hash; builder sees no file bytes."
        ),
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "role_weights": dict(zip(ROLES, (0.8, 0.1, 0.1), strict=True)),
        "input_record_count": len(rows),
        "eligible_record_count": len(eligible),
        "input_role_counts": {role: sum(row.role == role for row in rows) for role in ROLES},
        "eligible_role_type_counts": available,
        "source_cap": source_cap,
        "source_expansion_order": [row.question_id for row in source_order],
        "source_cap_rule": (
            "Freeze source_cap/2 per type; interleave bridge/comparison; "
            "run only a prefix. No spending authorization."
        ),
        "selection_rule": (
            "Per type SHA256(selection:42:question_id), then question_id tie-break; "
            "bridge/comparison interleave."
        ),
        "selected": {name: [row.question_id for row in items] for name, items in selected.items()},
        "selected_counts": {
            name: {"total": len(items), "by_type": _type_counts(items)}
            for name, items in selected.items()
        },
        "selected_roles": {
            "source": "memory_seed",
            "target": "calibration_dev",
            "debug": "calibration_dev",
            "check": "calibration_dev",
        },
        "selector_train_used": False,
        "exclusions": {
            "requested_question_ids": sorted(excluded_ids),
            "requested_question_ids_sha256": _digest(list(excluded_ids)),
            "requested_normalized_question_sha256s": sorted(
                _question_digest(text) for text in excluded_texts
            ),
            "requested_normalized_questions_digest": _digest(list(excluded_texts)),
            "matched_by_id": sorted(matches_by_id),
            "matched_by_normalized_text": sorted(matches_by_text),
            "unmatched_requested_question_ids": sorted(excluded_ids - set(identifiers)),
            "unmatched_normalized_text_count": len(excluded_texts - set(groups)),
            "duplicate_policy": duplicate_policy,
            "duplicate_normalized_groups": duplicate_groups,
            "duplicate_normalized_member_count": len(duplicate_member_ids),
            "excluded_union_question_ids": sorted(removed_ids),
            "excluded_union_count": len(removed_ids),
            "excluded_union_ids_sha256": _digest(list(removed_ids)),
            "reason_counts_may_overlap": True,
        },
        "checks": {
            "matching_gold_identity_present": True,
            "candidate_context_not_required_or_exported": True,
            "gold_content_semantically_verified": False,
            "duplicate_question_ids_rejected": True,
            "normalized_duplicates_handled": True,
            "semantic_near_duplicates_checked": False,
            "document_disjointness_checked": False,
            "full_file_completeness_independently_verified": False,
        },
        "official_validation_used": False,
        "official_test_used": False,
        "sampling_notice": (
            "Balanced internal train diagnostic, "
            "not benchmark-natural proportions or official test."
        ),
        "runtime_notice": (
            "Offline IDs only; question types used for stratification "
            "must not become runtime gold features."
        ),
    }
