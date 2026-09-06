"""Read-only, outcome-independent selection for the bounded PRE comparison.

Preserves the original 200-row Hotpot train manifest and its roles. The role
lists already have stable SHA-256 ordering; select their first unique normalized
questions without looking at answers, scores, source success or model outputs.
No download, credentials, model call, training, or file write occurs here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data_protocol import DATASET_ID, SPLIT_VERSION, build_manifest, normalize_question
from .hotpot import HotpotExample, parse_hotpot_example

SCHEMA_VERSION = "growrag-pre-query-manifest-v1"
MAX_SOURCE_COUNT = 32
MAX_TARGET_COUNT = 16


def _count(value: int, maximum: int, name: str) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")


def _question_hash(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def load_pre_examples(
    manifest_path: Path, source_count: int = 32, target_count: int = 16
) -> tuple[tuple[HotpotExample, ...], tuple[HotpotExample, ...], dict]:
    """Return sources, targets and a NEW manifest; never overwrite the old one.

    The 5 CNY batch ceiling is recorded as authorization metadata, not enforced
    here: the executing runner must independently enforce its paid-call budget.
    Calibration questions have already been exposed during development. They
    must not be described as held-out official validation or test questions.
    """
    _count(source_count, MAX_SOURCE_COUNT, "source_count")
    _count(target_count, MAX_TARGET_COUNT, "target_count")
    manifest_path = Path(manifest_path).resolve()
    original_bytes = manifest_path.read_bytes()
    original = json.loads(original_bytes)
    if not isinstance(original, dict) or (
        original.get("schema_version") != "growrag-hotpot-manifest-v1"
        or original.get("dataset") != DATASET_ID
        or original.get("configuration") != "distractor"
        or original.get("official_split") != "train"
        or original.get("split_version") != SPLIT_VERSION
    ):
        raise ValueError("expected the original frozen Hotpot distractor train manifest")
    filename = original.get("data_file")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).is_absolute()
        or Path(filename).name != filename
    ):
        raise ValueError("data_file must be a filename in the manifest directory")
    data_path = (manifest_path.parent / filename).resolve()
    if data_path.parent != manifest_path.parent:
        raise ValueError("resolved data file must stay inside the manifest directory")
    data_bytes = data_path.read_bytes()
    data_sha256 = hashlib.sha256(data_bytes).hexdigest()
    if data_sha256 != original.get("data_sha256"):
        raise ValueError("data SHA-256 differs from the original manifest")
    records = json.loads(data_bytes)
    if not isinstance(records, list) or len(records) != 200:
        raise ValueError("this bounded PRE batch requires the existing 200-row train preview")
    # Revalidate the OLD default 8+8 selection as well as all role assignments.
    # Parser validation may read gold fields, but selection never uses outcomes.
    expected = build_manifest(records, seed=original["seed"])
    for key in (
        "roles",
        "selected",
        "role_counts",
        "record_count",
        "role_weights",
        "official_validation_used",
        "official_test_used",
    ):
        if original.get(key) != expected[key]:
            raise ValueError(f"original {key} differs from the frozen split algorithm")
    by_id = {record["_id"]: record for record in records}
    selected, selected_queries, skipped_duplicates = {}, {}, {}
    for role, needed in (("memory_seed", source_count), ("calibration_dev", target_count)):
        identifiers, query_rows, skipped, seen = [], [], [], set()
        for qid in original["roles"][role]:
            digest = _question_hash(by_id[qid]["question"])
            if digest in seen:
                skipped.append(qid)
                continue
            seen.add(digest)
            identifiers.append(qid)
            query_rows.append({"question_id": qid, "normalized_question_sha256": digest})
            if len(identifiers) == needed:
                break
        if len(identifiers) != needed:
            raise ValueError(f"{role} lacks {needed} unique questions; no replacement across roles")
        selected[role], selected_queries[role], skipped_duplicates[role] = (
            identifiers,
            query_rows,
            skipped,
        )
    source_ids, target_ids = selected["memory_seed"], selected["calibration_dev"]
    source_hashes = {item["normalized_question_sha256"] for item in selected_queries["memory_seed"]}
    target_hashes = {
        item["normalized_question_sha256"] for item in selected_queries["calibration_dev"]
    }
    if set(source_ids) & set(target_ids) or source_hashes & target_hashes:
        raise ValueError("source and target questions must be disjoint in ID and normalized text")
    new_manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_version": "growrag-pre-role-prefix-unique-v1",
        "dataset": DATASET_ID,
        "configuration": "distractor",
        "official_split": "train",
        "split_version": SPLIT_VERSION,
        "seed": original["seed"],
        "data_file": filename,
        "data_sha256": data_sha256,
        "source_manifest_file": manifest_path.name,
        "source_manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "source_manifest_schema": original["schema_version"],
        "record_count": len(records),
        "roles": original["roles"],
        "role_counts": original["role_counts"],
        "role_weights": original["role_weights"],
        "selected": selected,
        "selected_counts": {role: len(ids) for role, ids in selected.items()},
        "selected_queries": selected_queries,
        "skipped_duplicate_question_ids": skipped_duplicates,
        "selection_order": "first unique normalized questions in validated original role order; "
        "that order is ascending SHA256(selection:{seed}:{question_id})",
        "selection_uses_gold_outcomes": False,
        "max_source_count": MAX_SOURCE_COUNT,
        "max_target_count": MAX_TARGET_COUNT,
        "authorized_batch_budget_cny": 5.0,
        "budget_notice": "executor must enforce budget separately; manifest loading makes no calls",
        "official_validation_used": False,
        "official_test_used": False,
        "selector_train_used": False,
        "sampling_notice": "first 200 mirror train rows; convenience development sample, "
        "not a random benchmark sample",
        "calibration_previously_exposed": True,
        "exposure_notice": "calibration_dev has previously been used during development, "
        "including pilot or QPP diagnostics; this is not a held-out evaluation set",
        "source_reuse_notice": "memory_seed may overlap earlier development source runs; "
        "this new batch must retain its own provenance and results",
    }
    sources = tuple(
        parse_hotpot_example(by_id[qid], dataset="hotpotqa-distractor-train-preview-200")
        for qid in source_ids
    )
    targets = tuple(
        parse_hotpot_example(by_id[qid], dataset="hotpotqa-distractor-train-preview-200")
        for qid in target_ids
    )
    return sources, targets, new_manifest
