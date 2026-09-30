"""Freeze fresh source500 while byte-preserving calibration100/evaluation500/corpus.

Default is a read-only preflight. Only --prepare-new creates a new bundle. No
labels, calibration/evaluation question text, API, claim release or old predictions
are consumed. A failed partial output is retained and must never be overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path

from . import prepare_operator_data as previous
from .fresh_dev_manifest import _read_json, _sha
from .operator_data_plan import SCHEMA, SOURCE_SIZES, _digest, question_metadata

DEFAULT_PARENT = "data/hotpotqa/operator_scale_sep30_v1/manifest.json"
DEFAULT_OUTPUT = "data/hotpotqa/operator_scale_action_v3"
SEED = "growrag-action-list-v3-fresh-source-20260930-v1"
MIGRATION_SCHEMA = "growrag-operator-source-migration-v1"
COUNTS = {"source": 500, "calibration": 100, "evaluation": 500}
PRESERVED = (
    "calibration_runtime_questions.jsonl",
    "evaluation_runtime_questions.jsonl",
    "corpus.jsonl",
)
_ID = re.compile(r"[0-9a-f]{24}")
_SHA = re.compile(r"[0-9a-f]{64}")


def _history_paths(root: Path, output: Path) -> set[Path]:
    paths = {previous._safe(root, name) for name in previous.RESERVATIONS}
    paths.update((root / "data/hotpotqa").rglob("manifest.json"))
    for pattern in (
        "launch_plan.json",
        "*.claim.json",
        "batch_manifest.json",
        "source_pool_manifest.json",
    ):
        paths.update((root / "runs").rglob(pattern))
    return {previous._safe(root, path) for path in paths if not path.is_relative_to(output)}


def _parent(root: Path, path: Path) -> tuple[dict, list[dict]]:
    path = previous._safe(root, path)
    parent = _read_json(path)
    sidecar = previous._safe(root, path.with_suffix(".sha256"))
    if (
        path.name != "manifest.json"
        or sidecar.read_text().strip() != f"{_sha(path)}  manifest.json"
    ):
        raise ValueError("parent manifest SHA sidecar mismatch")
    if (
        parent.get("schema_version") != SCHEMA
        or parent.get("counts") != COUNTS
        or parent.get("source_sizes") != list(SOURCE_SIZES)
        or parent.get("official_splits")
        != {"source": "train", "calibration": "train", "evaluation": "dev"}
        or parent.get("gold_projected") is not False
        or parent.get("official_test_used") is not False
        or parent.get("evaluation_memory_updates_allowed") is not False
        or parent.get("calibration_memory_updates_allowed") is not False
    ):
        raise ValueError("parent must be the frozen label-free train500/train100/dev500 study")
    roles, hashes = parent.get("roles"), parent.get("selected_normalized_question_sha256s")
    if (
        not isinstance(roles, dict)
        or set(roles) != set(COUNTS)
        or not isinstance(hashes, dict)
        or set(hashes) != set(COUNTS)
    ):
        raise ValueError("parent role/hash registries are incomplete")
    for role, count in COUNTS.items():
        for values, pattern in ((roles[role], _ID), (hashes[role], _SHA)):
            if (
                type(values) is not list
                or len(values) != count
                or any(type(value) is not str or not pattern.fullmatch(value) for value in values)
            ):
                raise ValueError("invalid parent role IDs or normalized hashes")
        if parent.get("role_ids_sha256", {}).get(role) != _digest(roles[role]):
            raise ValueError("parent role order fingerprint mismatch")
    if len({qid for ids in roles.values() for qid in ids}) != sum(COUNTS.values()) or len(
        {value for values in hashes.values() for value in values}
    ) != sum(COUNTS.values()):
        raise ValueError("parent roles or normalized questions overlap")
    if parent.get("nested_source_ids") != {
        str(size): roles["source"][:size] for size in SOURCE_SIZES
    }:
        raise ValueError("parent source prefixes changed")
    inputs = [
        {"path": p.relative_to(root).as_posix(), **previous._artifact(p)} for p in (path, sidecar)
    ]
    for name in (*PRESERVED, "source_runtime_questions.jsonl"):
        info = parent.get("artifacts", {}).get(name, {})
        artifact = previous._safe(root, path.parent / name)
        expected_rows = COUNTS.get(name.removesuffix("_runtime_questions.jsonl"))
        if (
            info.get("contains_gold") is not False
            or info.get("runtime_safe") is not True
            or type(info.get("rows")) is not int
            or info["rows"] <= 0
            or expected_rows is not None
            and info["rows"] != expected_rows
            or info.get("sha256") != _sha(artifact)
            or info.get("bytes") != artifact.stat().st_size
        ):
            raise ValueError("parent runtime artifact metadata/hash mismatch")
        inputs.append(
            {"path": artifact.relative_to(root).as_posix(), **previous._artifact(artifact)}
        )
    if type(parent.get("input_files")) is not list:
        raise ValueError("parent provenance inputs missing")
    inputs.extend(deepcopy(parent["input_files"]))
    return parent, inputs


def _project_candidates(shards, eligible: set[str], forbidden: set[str]):
    """Filter IDs in Arrow BEFORE materializing question strings; never project labels."""
    import pyarrow.dataset as ds

    dataset = ds.dataset([str(path) for path in shards], format="parquet")
    all_ids = dataset.to_table(columns=["id"])["id"].to_pylist()
    if len(set(all_ids)) != len(all_ids) or any(
        type(qid) is not str or not _ID.fullmatch(qid) for qid in all_ids
    ):
        raise ValueError("invalid or duplicate train mirror IDs")
    wanted = eligible - forbidden
    if not wanted <= set(all_ids):
        raise ValueError("shared background has missing train IDs; never silently resample")
    rows = dataset.to_table(
        columns=["id", "question"], filter=ds.field("id").isin(sorted(wanted))
    ).to_pylist()
    metadata, texts = [], {}
    for row in rows:
        if row["id"] not in wanted:
            raise ValueError("sealed or non-background question escaped projection filter")
        item = question_metadata(row["id"], row["question"], official_split="train")
        metadata.append(item)
        texts[item.question_id] = row["question"]
    if set(texts) != wanted:
        raise ValueError("candidate projection coverage mismatch")
    return metadata, texts, len(all_ids)


def _inputs_unchanged(root, inputs, history, output):
    if (root / "runs/.operator-study.lock").exists():
        raise ValueError("operator study is running; cannot freeze changing reservations")
    if _history_paths(root, output) != history:
        raise ValueError("reservation inventory changed during migration")
    for item in inputs:
        path = previous._safe(root, item["path"])
        if _sha(path) != item["sha256"]:
            raise ValueError("input changed during migration; partial bundle invalid")


def _copy_preserved(source: Path, target: Path, expected: dict):
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    if _sha(target) != expected["sha256"] or target.stat().st_size != expected["bytes"]:
        raise ValueError("preserved artifact byte-copy verification failed")


def migrate_operator_sources(
    root: Path,
    parent_manifest: Path = Path(DEFAULT_PARENT),
    output_dir: Path = Path(DEFAULT_OUTPUT),
    *,
    prepare_new: bool = False,
    seed: str = SEED,
) -> dict:
    """Preflight or write a new source-only migration; never decode preserved runtime files."""
    root = Path(root).resolve(strict=True)
    output = (root / output_dir).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError("output must stay inside project, never its root")
    if output.exists():
        raise FileExistsError("output already exists; never overwrite or resume a partial bundle")
    if type(seed) is not str or not seed.strip():
        raise ValueError("seed must be a nonempty string")
    parent_path = previous._safe(root, parent_manifest)
    parent, entries = _parent(root, parent_path)
    forbidden, blocked_hashes, old_inputs, shared, _ = previous._inputs(root)
    entries.extend(old_inputs)
    history = _history_paths(root, output)
    for path in sorted(history | {parent_path}):
        value = _read_json(path)
        forbidden.update(previous._reservation_ids(value))
        blocked_hashes.update(previous._hash_registry(value))
        extra = value.get("forbidden_question_hashes", [])
        if type(extra) is not list or any(
            type(h) is not str or not _SHA.fullmatch(h) for h in extra
        ):
            raise ValueError("invalid historical forbidden question hashes")
        blocked_hashes.update(extra)
        entries.append({"path": path.relative_to(root).as_posix(), **previous._artifact(path)})
    background = shared.get("background_question_ids")
    if (
        type(background) is not list
        or len(set(background)) != len(background)
        or any(type(q) is not str or not _ID.fullmatch(q) for q in background)
    ):
        raise ValueError("invalid shared background ID registry")
    shards, provenance, train_inputs = previous._mirror(root, previous.TRAIN_DIR, "train")
    entries.extend(train_inputs)
    inputs = {}
    for entry in entries:
        path = previous._safe(root, entry["path"])
        name = path.relative_to(root).as_posix()
        if name in inputs and inputs[name]["sha256"] != entry["sha256"]:
            raise ValueError("conflicting frozen input provenance")
        inputs[name] = {**entry, "path": name}
    inputs = [inputs[name] for name in sorted(inputs)]
    _inputs_unchanged(root, inputs, history, output)
    metadata, texts, train_count = _project_candidates(shards, set(background), forbidden)
    if train_count != provenance.get("row_count"):
        raise ValueError("train mirror row count mismatch")
    frequency = Counter(row.normalized_question_sha256 for row in metadata)
    duplicate_hashes = {h for h, count in frequency.items() if count > 1}
    excluded = forbidden | {
        row.question_id
        for row in metadata
        if row.normalized_question_sha256 in blocked_hashes | duplicate_hashes
    }
    available = {row.question_id: row for row in metadata if row.question_id not in excluded}
    source = sorted(
        available,
        key=lambda qid: (hashlib.sha256(f"{seed}:source:{qid}".encode()).hexdigest(), qid),
    )[:500]
    if len(source) != 500:
        raise ValueError(
            "not enough fresh shared-background source questions; never relax exclusions"
        )
    result = deepcopy(parent)
    result["roles"]["source"] = source
    result["selected_normalized_question_sha256s"]["source"] = [
        available[q].normalized_question_sha256 for q in source
    ]
    selected_hashes = [
        h for values in result["selected_normalized_question_sha256s"].values() for h in values
    ]
    if set(source) & forbidden or len(set(selected_hashes)) != sum(COUNTS.values()):
        raise ValueError("new source overlaps reserved IDs or normalized questions")
    result.update(
        seed=seed,
        migration_schema=MIGRATION_SCHEMA,
        parent_manifest={
            "path": parent_path.relative_to(root).as_posix(),
            "sha256": _sha(parent_path),
        },
        preserved_role_ids_sha256={
            role: parent["role_ids_sha256"][role] for role in ("calibration", "evaluation")
        },
        preserved_artifacts={
            name: {
                "path": (parent_path.parent / name).relative_to(root).as_posix(),
                **parent["artifacts"][name],
            }
            for name in PRESERVED
        },
        role_ids_sha256={role: _digest(ids) for role, ids in result["roles"].items()},
        nested_source_ids={str(size): source[:size] for size in SOURCE_SIZES},
        nested_source_ids_sha256={str(size): _digest(source[:size]) for size in SOURCE_SIZES},
        forbidden_question_ids=sorted(forbidden),
        forbidden_question_hashes=sorted(blocked_hashes),
        excluded_question_ids=sorted(excluded),
        duplicate_normalized_question_sha256s=sorted(duplicate_hashes),
        training_exclusion_question_ids=sorted(
            excluded | set(parent["roles"]["calibration"]) | set(parent["roles"]["evaluation"])
        ),
        prior_reservation_count=len(forbidden),
        prior_normalized_hash_count=len(blocked_hashes),
        input_metadata_sha256=_digest(
            [
                (row.question_id, row.normalized_question_sha256, row.official_split)
                for row in sorted(metadata, key=lambda row: row.question_id)
            ]
        ),
        input_metadata_scope="fresh_shared_background_train_candidates_only",
        input_files=inputs,
        reservation_inputs=[
            {"path": path.relative_to(root).as_posix(), "sha256": _sha(path)}
            for path in sorted(history)
        ],
        selection_uses_answers_or_scores=False,
        selection_uses_type_or_difficulty=False,
        gold_projected=False,
        old_reserved_text_projected=False,
        calibration_evaluation_runtime_decoded=False,
        corpus_rebuilt=False,
        old_claims_released=False,
        old_predictions_reused=False,
        api_calls=0,
        migration_policy=(
            "Replace source only; preserve calibration/evaluation IDs, order and runtime bytes, "
            "and exact corpus bytes. Never release claims or reuse historical source outcomes."
        ),
        normalized_exclusion_coverage=(
            "Fresh shared-background train candidates and all registered normalized hashes only. "
            "Sealed text is never opened to reconstruct missing historical hashes; "
            "semantic disjointness is not certified."
        ),
        implementation_sha256=_sha(Path(__file__)),
    )
    result["eligibility_counts"]["source"] = len(metadata)
    result["eligibility_ids_sha256"]["source"] = _digest(sorted(set(background) - forbidden))
    result["raw_source_rows"]["train"] = train_count
    result["artifacts"] = {name: deepcopy(parent["artifacts"][name]) for name in PRESERVED}
    _inputs_unchanged(root, inputs, history, output)
    if not prepare_new:
        return result
    output.mkdir(parents=True, exist_ok=False)
    for name in PRESERVED:
        _copy_preserved(parent_path.parent / name, output / name, parent["artifacts"][name])
    target = output / "source_runtime_questions.jsonl"
    with target.open("xb") as handle:
        for qid in source:
            handle.write(
                previous._canonical(
                    {
                        "question_id": qid,
                        "text": texts[qid],
                        "dataset": "hotpotqa-dynamic-operator-source-action-list-v3",
                    }
                )
                + b"\n"
            )
    result["artifacts"][target.name] = {
        **previous._artifact(target, rows=500),
        "contains_gold": False,
        "runtime_safe": True,
    }
    _inputs_unchanged(root, inputs, history, output)
    manifest = output / "manifest.json"
    with manifest.open("xb") as handle:
        handle.write(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
        )
    _inputs_unchanged(root, inputs, history, output)
    with (output / "manifest.sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{_sha(manifest)}  manifest.json\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--parent-manifest", type=Path, default=Path(DEFAULT_PARENT))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--prepare-new", action="store_true")
    args = parser.parse_args(argv)
    result = migrate_operator_sources(
        args.root,
        args.parent_manifest,
        args.output_dir,
        prepare_new=args.prepare_new,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "migration_schema",
                    "counts",
                    "source_sizes",
                    "parent_manifest",
                    "preserved_role_ids_sha256",
                    "prior_reservation_count",
                    "api_calls",
                    "calibration_evaluation_runtime_decoded",
                    "corpus_rebuilt",
                )
            }
            | {"prepared": args.prepare_new},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
