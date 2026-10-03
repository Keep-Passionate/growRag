"""Freeze a new 100-question train development cohort for the A0 wire-repair v2.

Read-only unless --prepare-new is supplied. The old A0 cohort is entirely sealed,
including unattempted/unpaired questions. Selection uses only IDs and normalized
question text, with a new seed, never scores, labels, difficulty or model output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import prepare_a0_data as a0
from .fresh_dev_manifest import _read_json, _sha
from .prepare_a0_data import _projected_metadata, _question_hash, _registry_snapshot, _runtime_rows
from .prepare_history_opportunity import (
    CORPUS,
    INDEX,
    PINS,
    _anchors,
    _canonical,
    _check,
    _digest,
    _path,
)
from .protocol import RuntimeQuestion

SCHEMA = "growrag-a0-v2-data-v1"
SEED = "growrag-a0-wire-v2-20261003"
COUNT = 100
OUTPUT = "data/hotpotqa/a0_wire_v2_20261003_v1"
DATASET = "hotpotqa-a0-wire-development-v2"
ROLE = "development_wire_query_construction_not_training"
OLD_A0_MANIFEST = f"{a0.OUTPUT}/manifest.json"
OLD_A0_MANIFEST_SHA = "7d029f01d4425853f54f81ae390de601ecb0f38cac86d78b3265aa834c9674b8"


def _fixed():
    """Keep the established manifest contract, changing only cohort identity."""
    return {**a0._fixed(), "schema_version": SCHEMA, "seed": SEED, "count": COUNT, "role": ROLE}


def _validate_old_cohort(root, registry):
    """Protect all 100 v1 IDs and hashes, not merely attempted/completed records."""
    path = _path(root, OLD_A0_MANIFEST)
    _check(_sha(path) == OLD_A0_MANIFEST_SHA, "sealed old A0 manifest changed")
    old = _read_json(path)
    _check(
        len(old["question_ids"]) == len(set(old["question_ids"])) == 100
        and set(old["question_ids"]) <= set(registry["question_ids"])
        and set(old["normalized_question_sha256s"]) <= set(registry["normalized_question_sha256s"])
        and {"path": OLD_A0_MANIFEST, "sha256": OLD_A0_MANIFEST_SHA} in registry["inputs"],
        "complete sealed old A0 cohort is missing from exclusions",
    )


def prepare(root, prepare_new=False):
    root = Path(root).resolve(strict=True)
    output = (root / OUTPUT).resolve()
    _check(output.is_relative_to(root), "output escapes project")
    if output.exists():
        raise FileExistsError("bundle exists; do not resample, reuse or overwrite")
    source, available, shards, shard_inputs = _anchors(root)
    index_sha = _sha(_path(root, INDEX))
    snapshot = _registry_snapshot(root)
    _validate_old_cohort(root, snapshot)
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


def _validate_manifest(root, manifest):
    fixed = _fixed()
    _check(
        isinstance(manifest, dict)
        and set(fixed) <= set(manifest)
        and _canonical({key: manifest[key] for key in fixed}) == _canonical(fixed),
        "unregistered v2 runtime bundle fields/protocol",
    )
    # The shared v1 structural/anchor validator receives a new contract view, not
    # mutated globals or mutated input. Only the already-checked cohort identity
    # is translated; all variable fields retain the exact bytes-derived values.
    a0._validate_manifest(root, {**manifest, **a0._fixed()})
    _validate_old_cohort(root, manifest["exclusion_registry"])


def load_bundle(root, *, expected_sha):
    """Load only frozen runtime questions and corpus; require an external SHA."""
    _check(
        isinstance(expected_sha, str) and a0._SHA.fullmatch(expected_sha),
        "externally pinned manifest SHA is required",
    )
    root = Path(root).resolve(strict=True)
    path = _path(root, Path(OUTPUT) / "manifest.json")
    digest = _sha(path)
    _check(digest == expected_sha, "externally pinned manifest SHA mismatch")
    _check(
        _path(root, Path(OUTPUT) / "manifest.sha256").read_text(encoding="ascii")
        == f"{digest}  manifest.json\n",
        "manifest sidecar mismatch",
    )
    manifest = _read_json(path)
    _validate_manifest(root, manifest)
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
        and a0._SHA.fullmatch(info["sha256"])
        and info["contains_gold"] is False,
        "invalid runtime artifact declaration",
    )
    runtime_path = _path(root, Path(OUTPUT) / info["path"])
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
    return manifest, tuple(RuntimeQuestion(**row) for row in rows), _path(root, CORPUS)


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
