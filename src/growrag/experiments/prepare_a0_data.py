"""Freeze 100 new train development questions for A0 query-construction controls.

Default is read-only. Only --prepare-new writes a new, exclusive bundle. Selection
uses IDs and normalized question identities, never gold, scores, type or difficulty.
The existing history library, corpus and already-built index are reused unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .data_protocol import normalize_question
from .fresh_dev_manifest import _ids, _read_json, _sha, _unique
from .prepare_history_opportunity import (
    BACKGROUND_MANIFEST,
    CORPUS,
    INDEX,
    LIBRARY,
    LIBRARY_FINGERPRINT,
    PINS,
    PROVENANCE,
    SOURCE_MANIFEST,
    _anchors,
    _canonical,
    _check,
    _digest,
    _fixture,
    _path,
    _project,
    _registry,
)
from .prepare_operator_data import _hash_registry
from .protocol import RuntimeQuestion

SCHEMA = "growrag-a0-data-v1"
SEED = "growrag-a0-query-construction-20261002-v1"
COUNT = 100
OUTPUT = "data/hotpotqa/a0_query_construction_20261002_v1"
DATASET = "hotpotqa-a0-query-construction-development-v1"
ROLE = "development_query_construction_not_training"
_ID = re.compile(r"[0-9a-f]{24}")
_SHA = re.compile(r"[0-9a-f]{64}")
_RUNTIME = re.compile(r"(?:[a-z_]+_)?runtime_questions\.jsonl")


def _question_hash(text):
    return hashlib.sha256(normalize_question(text).encode("utf-8")).hexdigest()


def _fixed():
    """Exact protocol fields, shared by export and strict runtime validation."""
    return {
        "schema_version": SCHEMA,
        "seed": SEED,
        "count": COUNT,
        "official_split": "train",
        "role": ROLE,
        "gold_projected": False,
        "api_calls": 0,
        "model_training": False,
        "memory_updates_allowed": False,
        "official_test_used": False,
        "projected_columns": ["id", "question"],
        "selection_uses_scores_or_difficulty": False,
        "all_selected_ids_protected_from_future_resampling": True,
        "cohort_notice": "Available-context subset of frozen train background, not representative "
        "of all train and not an official HotpotQA benchmark test.",
        "library": {"path": LIBRARY, "sha256": PINS[LIBRARY], "fingerprint": LIBRARY_FINGERPRINT},
        "source_manifest": {"path": SOURCE_MANIFEST, "sha256": PINS[SOURCE_MANIFEST]},
        "background_manifest": {"path": BACKGROUND_MANIFEST, "sha256": PINS[BACKGROUND_MANIFEST]},
        "source_provenance": {"path": PROVENANCE, "sha256": PINS[PROVENANCE]},
        "runtime_input_allowlist": ["runtime_questions.jsonl", "corpus_ref"],
        "normalized_exclusion_policy": "Exclude registered hashes, all available blocked train "
        "id/question hashes, prior runtime question hashes, and every within-pool duplicate. "
        "Unavailable old text is explicitly counted; no labels or sealed gold are read.",
    }


def _runtime_rows(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line, object_pairs_hook=_unique)
            _check(
                isinstance(row, dict)
                and set(row) == {"question_id", "text", "dataset"}
                and isinstance(row["question_id"], str)
                and _ID.fullmatch(row["question_id"])
                and isinstance(row["text"], str)
                and isinstance(row["dataset"], str),
                "runtime fields/identity changed or unexpected non-question fields",
            )
            _question_hash(row["text"])
            rows.append(row)
    return rows


def _registry_snapshot(root):
    """Extend the old metadata-only reservation registry with label-free runtime files.

    Only question-file siblings of registered real metadata are examined. No results,
    answer records, context files or arbitrary JSONL are opened. Pending claims and
    previous 50-question opportunity manifests remain protected by the shared helper.
    """
    registry = _registry(root)
    # The parent helper covers the established one-level data layout. Also protect
    # nested/additive manifests, including real run manifests, without modifying
    # that older experiment's discovery rules or any of its global constants.
    known = {entry["path"] for entry in registry["inputs"]}
    extra_paths = set((root / "data/hotpotqa").rglob("manifest.json"))
    extra_paths.update(
        path for path in (root / "runs").rglob("manifest.json") if not _fixture(path, root)
    )
    extra_ids, extra_hashes, extra_inputs = set(), set(), []
    for candidate in sorted(extra_paths):
        path = _path(root, candidate)
        relative = path.relative_to(root).as_posix()
        if relative in known:
            continue
        value = _read_json(path)
        value = {
            key: child
            for key, child in value.items()
            if key not in {"background_question_ids", "corpus_source_question_ids"}
        }
        extra_ids.update(_ids(value))
        extra_hashes.update(_hash_registry(value))
        extra_inputs.append({"path": relative, "sha256": _sha(path)})
    registry["inputs"] = registry["inputs"] + extra_inputs
    registry["actual_metadata_files"] += len(extra_inputs)
    paths = set()
    for entry in registry["inputs"]:
        parent = _path(root, entry["path"]).parent
        paths.update(
            p for p in parent.glob("*runtime_questions.jsonl") if _RUNTIME.fullmatch(p.name)
        )
    ids = set(registry["question_ids"]) | extra_ids
    hashes = set(registry["normalized_question_sha256s"]) | extra_hashes
    runtime_ids, runtime_inputs = set(), []
    for candidate in sorted(paths):
        path = _path(root, candidate)
        digest = _sha(path)
        rows = _runtime_rows(path)
        _check(_sha(path) == digest, "prior runtime question bytes changed during inspection")
        runtime_ids.update(row["question_id"] for row in rows)
        hashes.update(_question_hash(row["text"]) for row in rows)
        runtime_inputs.append({"path": path.relative_to(root).as_posix(), "sha256": digest})
    inputs = sorted(registry["inputs"] + runtime_inputs, key=lambda row: row["path"])
    return {
        **registry,
        "inputs": inputs,
        "sha256": _digest(inputs),
        "question_ids": sorted(ids | runtime_ids),
        "normalized_question_sha256s": sorted(hashes),
        "runtime_text_question_ids": sorted(runtime_ids),
        "runtime_question_files_checked": len(runtime_inputs),
    }


def _projected_metadata(shards, available, snapshot):
    blocked = set(snapshot["question_ids"])
    requested = available | blocked
    rows = _project(shards, requested)
    _check(
        all(isinstance(row, dict) and set(row) == {"id", "question"} for row in rows),
        "unexpected projected columns",
    )
    ids = [row["id"] for row in rows]
    _check(
        all(isinstance(qid, str) and _ID.fullmatch(qid) for qid in ids)
        and len(ids) == len(set(ids))
        and set(ids) <= requested
        and available <= set(ids),
        "availability IDs missing/duplicate or projection escaped requested IDs",
    )
    metadata = sorted((row["id"], _question_hash(row["question"])) for row in rows)
    train_ids = set(ids) & blocked
    hashes = set(snapshot["normalized_question_sha256s"])
    hashes.update(digest for qid, digest in metadata if qid in blocked)
    registry = {
        **snapshot,
        "normalized_question_sha256s": sorted(hashes),
        "train_text_question_ids": sorted(train_ids),
        "unavailable_text_question_ids": sorted(
            blocked - train_ids - set(snapshot["runtime_text_question_ids"])
        ),
        "blocked_train_projected_columns": ["id", "question"],
    }
    eligible_metadata = [(qid, digest) for qid, digest in metadata if qid in available - blocked]
    return rows, eligible_metadata, registry


def prepare(root, prepare_new=False):
    root = Path(root).resolve(strict=True)
    output = (root / OUTPUT).resolve()
    _check(output.is_relative_to(root), "output escapes project")
    if output.exists():
        raise FileExistsError("bundle exists; do not resample, reuse or overwrite")
    source, available, shards, shard_inputs = _anchors(root)
    index_sha = _sha(_path(root, INDEX))
    snapshot = _registry_snapshot(root)
    rows, metadata, registry = _projected_metadata(shards, available, snapshot)
    blocked = set(registry["question_ids"])
    blocked_hashes = set(registry["normalized_question_sha256s"])
    frequency = Counter(digest for _, digest in metadata)
    good = [
        (qid, digest)
        for qid, digest in metadata
        if frequency[digest] == 1 and digest not in blocked_hashes
    ]
    selected = sorted(
        good,
        key=lambda row: (
            hashlib.sha256(f"{SEED}:development:{row[0]}".encode()).hexdigest(),
            row[0],
        ),
    )[:COUNT]
    _check(len(selected) == COUNT, "not enough independent available questions; no replacements")
    ids = [qid for qid, _ in selected]
    texts = {row["id"]: row["question"] for row in rows}
    runtime = b"".join(
        _canonical({"question_id": qid, "text": texts[qid], "dataset": DATASET}) + b"\n"
        for qid in ids
    )
    manifest = {
        **_fixed(),
        "question_ids": ids,
        "normalized_question_sha256s": [digest for _, digest in selected],
        "question_ids_sha256": _digest(ids),
        "corpus_ref": {
            "path": CORPUS,
            "sha256": PINS[CORPUS],
            "rows": 76691,
            "index_path": INDEX,
            "index_sha256": index_sha,
            "index_version": "growrag-shared-paragraph-sqlite-bm25-v1",
        },
        "parquet_inputs": shard_inputs,
        "exclusion_registry": registry,
        "runtime_artifact": {
            "path": "runtime_questions.jsonl",
            "sha256": hashlib.sha256(runtime).hexdigest(),
            "rows": COUNT,
            "bytes": len(runtime),
            "contains_gold": False,
        },
        "preflight": {
            "background_ids": len(available),
            "excluded_background_ids": len(available & blocked),
            "after_id_exclusion": len(metadata),
            "eligible_count": len(good),
            "known_hash_matches": sum(digest in blocked_hashes for _, digest in metadata),
            "duplicate_hash_groups": sum(number > 1 for number in frequency.values()),
            "projected_metadata_sha256": _digest(metadata),
            "blocked_train_texts_projected": len(registry["train_text_question_ids"]),
            "prior_runtime_texts_checked": len(registry["runtime_text_question_ids"]),
            "unavailable_blocked_texts": len(registry["unavailable_text_question_ids"]),
            "old_role_overlap": {
                role: len(set(qids) & set(ids)) for role, qids in source["roles"].items()
            },
        },
        "exporter_sha256": _sha(Path(__file__)),
    }
    if not prepare_new:
        return manifest
    _check(_registry_snapshot(root) == snapshot, "reservation registry changed during preparation")
    _check(
        all(_sha(_path(root, path)) == digest for path, digest in PINS.items())
        and all(_sha(_path(root, row["path"])) == row["sha256"] for row in shard_inputs)
        and _sha(_path(root, INDEX)) == index_sha,
        "source/index bytes changed during preparation",
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "runtime_questions.jsonl").open("xb") as handle:
        handle.write(runtime)
    with (output / "manifest.json").open("xb") as handle:
        handle.write(_canonical(manifest) + b"\n")
    with (output / "manifest.sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{_sha(output / 'manifest.json')}  manifest.json\n")
    return manifest


def _validate_manifest(project, manifest):
    fixed = _fixed()
    variable = {
        "question_ids",
        "normalized_question_sha256s",
        "question_ids_sha256",
        "corpus_ref",
        "parquet_inputs",
        "exclusion_registry",
        "runtime_artifact",
        "preflight",
        "exporter_sha256",
    }
    _check(
        isinstance(manifest, dict)
        and set(manifest) == set(fixed) | variable
        and _canonical({key: manifest[key] for key in fixed}) == _canonical(fixed),
        "unregistered runtime bundle fields/protocol",
    )
    for relative, expected in PINS.items():
        _check(_sha(_path(project, relative)) == expected, "runtime anchor changed")
    corpus_ref = manifest["corpus_ref"]
    _check(
        isinstance(corpus_ref, dict)
        and set(corpus_ref)
        == {"path", "sha256", "rows", "index_path", "index_sha256", "index_version"}
        and _canonical({key: value for key, value in corpus_ref.items() if key != "index_sha256"})
        == _canonical(
            {
                "path": CORPUS,
                "sha256": PINS[CORPUS],
                "rows": 76691,
                "index_path": INDEX,
                "index_version": "growrag-shared-paragraph-sqlite-bm25-v1",
            }
        )
        and _sha(_path(project, INDEX)) == corpus_ref["index_sha256"],
        "library/index reference changed",
    )
    provenance = _read_json(_path(project, PROVENANCE))
    expected_inputs = [
        {"path": (Path(PROVENANCE).parent / shard["file"]).as_posix(), "sha256": shard["sha256"]}
        for shard in provenance["shards"]
    ]
    _check(manifest["parquet_inputs"] == expected_inputs, "parquet provenance changed")
    ids, hashes = manifest["question_ids"], manifest["normalized_question_sha256s"]
    _check(
        isinstance(ids, list)
        and len(ids) == COUNT
        and all(isinstance(qid, str) and _ID.fullmatch(qid) for qid in ids)
        and len(set(ids)) == COUNT
        and _digest(ids) == manifest["question_ids_sha256"]
        and isinstance(hashes, list)
        and len(hashes) == COUNT
        and all(isinstance(digest, str) and _SHA.fullmatch(digest) for digest in hashes)
        and len(set(hashes)) == COUNT,
        "runtime ID/hash coverage mismatch",
    )
    registry = manifest["exclusion_registry"]
    source = _read_json(_path(project, SOURCE_MANIFEST))
    background = _read_json(_path(project, BACKGROUND_MANIFEST))
    protected = {qid for group in source["roles"].values() for qid in group}
    _check(
        set(ids) <= set(background["background_question_ids"])
        and not set(ids) & (protected | set(registry["question_ids"]))
        and not set(hashes) & set(registry["normalized_question_sha256s"])
        and _digest(registry["inputs"]) == registry["sha256"]
        and manifest["preflight"]["old_role_overlap"] == {role: 0 for role in source["roles"]},
        "runtime IDs/hashes escape availability or overlap frozen protections",
    )


def load_bundle(project, *, expected_sha):
    """Return only frozen, gold-free runtime inputs; an external SHA is mandatory."""
    _check(
        isinstance(expected_sha, str) and _SHA.fullmatch(expected_sha),
        "externally pinned manifest SHA is required",
    )
    project = Path(project).resolve(strict=True)
    path = _path(project, Path(OUTPUT) / "manifest.json")
    digest = _sha(path)
    _check(digest == expected_sha, "externally pinned manifest SHA mismatch")
    _check(
        _path(project, Path(OUTPUT) / "manifest.sha256").read_text(encoding="ascii")
        == f"{digest}  manifest.json\n",
        "manifest sidecar mismatch",
    )
    manifest = _read_json(path)
    _validate_manifest(project, manifest)
    info = manifest["runtime_artifact"]
    _check(
        isinstance(info, dict)
        and set(info) == {"path", "sha256", "rows", "bytes", "contains_gold"}
        and info["path"] == "runtime_questions.jsonl"
        and type(info["rows"]) is int
        and info["rows"] == COUNT
        and type(info["bytes"]) is int
        and info["bytes"] > 0
        and isinstance(info["sha256"], str)
        and _SHA.fullmatch(info["sha256"])
        and info["contains_gold"] is False,
        "invalid runtime artifact declaration",
    )
    runtime_path = _path(project, Path(OUTPUT) / info["path"])
    _check(
        _sha(runtime_path) == info["sha256"] and runtime_path.stat().st_size == info["bytes"],
        "runtime artifact bytes changed",
    )
    rows = _runtime_rows(runtime_path)
    _check(
        [row["question_id"] for row in rows] == manifest["question_ids"]
        and all(row["dataset"] == DATASET for row in rows),
        "runtime fields/order/dataset changed",
    )
    _check(
        [_question_hash(row["text"]) for row in rows] == manifest["normalized_question_sha256s"],
        "runtime normalized identity changed",
    )
    _check(_sha(path) == digest and _sha(runtime_path) == info["sha256"], "runtime bytes changed")
    return manifest, tuple(RuntimeQuestion(**row) for row in rows), _path(project, CORPUS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-new", action="store_true")
    args = parser.parse_args(argv)
    manifest = prepare(Path.cwd(), prepare_new=args.prepare_new)
    print(
        json.dumps(
            {
                "prepared": args.prepare_new,
                "output": OUTPUT,
                "count": COUNT,
                "question_ids_sha256": manifest["question_ids_sha256"],
                "preflight": manifest["preflight"],
                "api_calls": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
