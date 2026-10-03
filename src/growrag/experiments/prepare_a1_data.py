"""Prepare one new 40-question A1 train cohort using only IDs/question text.

Default is read-only, but still projects question text: the caller must first pass
the separately sealed synthetic gate before even this preflight. No API, labels,
source answers, historical examples, model scores or difficulty are read here.
The writer is exclusive; an existing/partial bundle cannot be replaced or resampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from . import prepare_a0_data as a0
from .fresh_dev_manifest import _ids, _read_json, _sha
from .prepare_a0_data import _projected_metadata, _question_hash, _runtime_rows
from .prepare_a0_v2_data import OLD_A0_MANIFEST, OLD_A0_MANIFEST_SHA
from .prepare_history_opportunity import (
    BACKGROUND_MANIFEST,
    CORPUS,
    INDEX,
    LIBRARY,
    LIBRARY_FINGERPRINT,
    PINS,
    PROVENANCE,
    SOURCE_MANIFEST,
    _canonical,
    _check,
    _digest,
    _fixture,
    _path,
)
from .prepare_operator_data import _hash_registry
from .protocol import RuntimeQuestion

SCHEMA = "growrag-a1-data-v1"
SEED = "growrag-a1-applicability-shadow-20261003-v1"
COUNT = 40
OUTPUT = "data/hotpotqa/a1_applicability_shadow_20261003_v1"
DATASET = "hotpotqa-a1-applicability-development-v1"
ROLE = "development_applicability_shadow_not_training"
A0_COHORTS = {
    OLD_A0_MANIFEST: OLD_A0_MANIFEST_SHA,
    "data/hotpotqa/a0_wire_v2_20261003_v1/manifest.json": (
        "7fb96c19099dd1215a9752fb9ab6dd109131eed00cc2a3d058913f3535c96148"
    ),
}


def _fixed():
    return {
        **a0._fixed(),
        "schema_version": SCHEMA,
        "seed": SEED,
        "count": COUNT,
        "role": ROLE,
        "library": {"path": LIBRARY, "sha256": PINS[LIBRARY], "fingerprint": LIBRARY_FINGERPRINT},
        "source_manifest": {"path": SOURCE_MANIFEST, "sha256": PINS[SOURCE_MANIFEST]},
        "background_manifest": {"path": BACKGROUND_MANIFEST, "sha256": PINS[BACKGROUND_MANIFEST]},
        "source_provenance": {"path": PROVENANCE, "sha256": PINS[PROVENANCE]},
        "protected_a0_cohorts": [
            {"path": path, "sha256": digest, "count": 100} for path, digest in A0_COHORTS.items()
        ],
    }


def _anchors(root):
    """Verify old byte identities without deserializing source examples in the library."""
    for relative, expected in PINS.items():
        _check(_sha(_path(root, relative)) == expected, "pinned input bytes changed")
    _path(root, INDEX)
    source = _read_json(_path(root, SOURCE_MANIFEST))
    background = _read_json(_path(root, BACKGROUND_MANIFEST))
    provenance = _read_json(_path(root, PROVENANCE))
    available = background["background_question_ids"]
    _check(
        len(available) == len(set(available)) == 7692
        and all(type(qid) is str and a0._ID.fullmatch(qid) for qid in available)
        and background["official_split"] == provenance["official_split"] == "train",
        "availability must be the frozen official-train background",
    )
    source_inputs = {r["path"].replace("\\", "/"): r["sha256"] for r in source["input_files"]}
    _check(
        source_inputs.get(BACKGROUND_MANIFEST) == PINS[BACKGROUND_MANIFEST]
        and source_inputs.get(PROVENANCE) == PINS[PROVENANCE]
        and source["artifacts"]["corpus.jsonl"]["sha256"] == PINS[CORPUS]
        and source["artifacts"]["corpus.jsonl"]["rows"] == 76691,
        "source manifest does not bind background/provenance/corpus",
    )
    shards, inputs = [], []
    for shard in provenance["shards"]:
        path = _path(root, Path(PROVENANCE).parent / shard["file"])
        relative = path.relative_to(root).as_posix()
        _check(
            path.parent == (root / PROVENANCE).parent
            and _sha(path) == shard["sha256"] == source_inputs.get(relative),
            "official train shard differs from pinned provenance/source",
        )
        shards.append(str(path))
        inputs.append({"path": relative, "sha256": shard["sha256"]})
    _check(shards and len(set(shards)) == len(shards), "missing/duplicate train shards")
    return source, set(available), shards, inputs


def _validate_a0_exclusions(root, registry):
    """Protect all 200 frozen IDs and normalized texts, including never-attempted IDs."""
    all_ids = set()
    for relative, expected in A0_COHORTS.items():
        path = _path(root, relative)
        _check(_sha(path) == expected, "sealed A0 cohort changed")
        old = _read_json(path)
        ids, hashes = old["question_ids"], old["normalized_question_sha256s"]
        _check(
            type(ids) is list
            and len(ids) == len(set(ids)) == 100
            and all(type(qid) is str and a0._ID.fullmatch(qid) for qid in ids)
            and type(hashes) is list
            and len(hashes) == len(set(hashes)) == 100
            and all(type(value) is str and a0._SHA.fullmatch(value) for value in hashes)
            and not all_ids.intersection(ids)
            and set(ids) <= set(registry["question_ids"])
            and set(hashes) <= set(registry["normalized_question_sha256s"])
            and {"path": relative, "sha256": expected} in registry["inputs"],
            "complete sealed A0 cohort is missing from exclusions",
        )
        all_ids.update(ids)


def _registry_snapshot(root):
    """Reject redirected metadata/runtime inputs before the old scanner opens any.

    The old scanner validates question shape after resolving its lexical runtime
    filename. A link to a sealed labels/results file must be rejected before that
    read, not merely by its later schema validation. Do not alter the old helper.
    """
    metadata = set((root / "data/hotpotqa").rglob("manifest.json"))
    metadata.update((root / "runs").rglob("*.claim.json"))
    for name in (
        "manifest.json",
        "launch_plan.json",
        "plan.json",
        "batch_manifest.json",
        "source_pool_manifest.json",
    ):
        metadata.update(path for path in (root / "runs").rglob(name) if not _fixture(path, root))
    for candidate in sorted(metadata):
        resolved = candidate.resolve(strict=True)
        _check(
            resolved.is_relative_to(root)
            and resolved.is_file()
            and resolved == candidate.absolute(),
            "redirected exclusion metadata is not permitted",
        )
        for runtime in candidate.parent.glob("*runtime_questions.jsonl"):
            if not a0._RUNTIME.fullmatch(runtime.name):
                continue
            resolved = runtime.resolve(strict=True)
            _check(
                resolved.is_relative_to(root)
                and resolved.is_file()
                and a0._RUNTIME.fullmatch(resolved.name)
                and resolved == runtime.absolute(),
                "redirected runtime question input is not permitted",
            )
    return a0._registry_snapshot(root)


def prepare(root, prepare_new=False):
    """Called only after the external synthetic gate; no resume/replacement behavior."""
    _check(type(prepare_new) is bool, "prepare_new must be a boolean")
    root = Path(root).resolve(strict=True)
    output = (root / OUTPUT).resolve()
    _check(output.is_relative_to(root), "output escapes project")
    if output.exists():
        raise FileExistsError("bundle exists; do not resample, reuse or overwrite")
    source, available, shards, shard_inputs = _anchors(root)
    index_sha = _sha(_path(root, INDEX))
    snapshot = _registry_snapshot(root)
    _validate_a0_exclusions(root, snapshot)
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
    _validate_a0_exclusions(root, registry)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (
        ("runtime_questions.jsonl", runtime),
        ("manifest.json", _canonical(manifest) + b"\n"),
    ):
        with (output / name).open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    with (output / "manifest.sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{_sha(output / 'manifest.json')}  manifest.json\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def _validate_registry(project, registry):
    """Audit sealed metadata/runtime provenance without reopening train Parquet."""
    _check(type(registry) is dict, "invalid exclusion registry")
    for key, pattern in (
        ("question_ids", a0._ID),
        ("normalized_question_sha256s", a0._SHA),
    ):
        values = registry[key]
        _check(
            type(values) is list
            and all(type(value) is str and pattern.fullmatch(value) for value in values)
            and values == sorted(set(values)),
            "invalid exclusion identities",
        )
    inputs = registry["inputs"]
    _check(
        type(inputs) is list
        and all(
            type(row) is dict
            and set(row) == {"path", "sha256"}
            and type(row["path"]) is str
            and type(row["sha256"]) is str
            and a0._SHA.fullmatch(row["sha256"])
            for row in inputs
        )
        and len({row["path"] for row in inputs}) == len(inputs)
        and _digest(inputs) == registry["sha256"],
        "exclusion provenance changed",
    )
    required_ids, required_hashes = set(), set()
    for entry in inputs:
        path = _path(project, entry["path"])
        runtime = a0._RUNTIME.fullmatch(path.name)
        _check(
            runtime
            or path.name == "manifest.json"
            or path.name
            in {"launch_plan.json", "plan.json", "batch_manifest.json", "source_pool_manifest.json"}
            or path.name.endswith(".claim.json"),
            "unregistered exclusion metadata filename",
        )
        _check(_sha(path) == entry["sha256"], "sealed exclusion input changed")
        if runtime:
            rows = _runtime_rows(path)
            required_ids.update(row["question_id"] for row in rows)
            required_hashes.update(_question_hash(row["text"]) for row in rows)
        else:
            value = _read_json(path)
            value = {
                key: child
                for key, child in value.items()
                if key not in {"background_question_ids", "corpus_source_question_ids"}
            }
            required_ids.update(_ids(value))
            required_hashes.update(_hash_registry(value))
    _check(
        required_ids <= set(registry["question_ids"])
        and required_hashes <= set(registry["normalized_question_sha256s"]),
        "registered exclusions were removed",
    )
    _validate_a0_exclusions(project, registry)


def _validate_manifest(project, manifest):
    """Independent 40-row contract; never disguise A1 as the old 100-row validator."""
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
        type(manifest) is dict
        and set(manifest) == set(fixed) | variable
        and _canonical({key: manifest[key] for key in fixed}) == _canonical(fixed),
        "unregistered A1 runtime bundle fields/protocol",
    )
    for relative, expected in PINS.items():
        _check(_sha(_path(project, relative)) == expected, "runtime anchor changed")
    corpus = manifest["corpus_ref"]
    _check(
        type(corpus) is dict
        and set(corpus) == {"path", "sha256", "rows", "index_path", "index_sha256", "index_version"}
        and _canonical({key: value for key, value in corpus.items() if key != "index_sha256"})
        == _canonical(
            {
                "path": CORPUS,
                "sha256": PINS[CORPUS],
                "rows": 76691,
                "index_path": INDEX,
                "index_version": "growrag-shared-paragraph-sqlite-bm25-v1",
            }
        )
        and _sha(_path(project, INDEX)) == corpus["index_sha256"],
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
        type(ids) is list
        and len(ids) == COUNT
        and all(type(qid) is str and a0._ID.fullmatch(qid) for qid in ids)
        and len(set(ids)) == COUNT
        and _digest(ids) == manifest["question_ids_sha256"]
        and type(hashes) is list
        and len(hashes) == COUNT
        and all(type(value) is str and a0._SHA.fullmatch(value) for value in hashes)
        and len(set(hashes)) == COUNT,
        "runtime ID/hash coverage mismatch",
    )
    registry = manifest["exclusion_registry"]
    _validate_registry(project, registry)
    source = _read_json(_path(project, SOURCE_MANIFEST))
    background = _read_json(_path(project, BACKGROUND_MANIFEST))
    protected = {qid for group in source["roles"].values() for qid in group}
    _check(
        background["official_split"] == provenance["official_split"] == "train"
        and set(ids) <= set(background["background_question_ids"])
        and not set(ids) & (protected | set(registry["question_ids"]))
        and not set(hashes) & set(registry["normalized_question_sha256s"])
        and manifest["preflight"]["old_role_overlap"] == {role: 0 for role in source["roles"]},
        "runtime IDs/hashes escape availability or overlap frozen protections",
    )


def load_bundle(project, *, expected_sha):
    """Return only externally pinned runtime questions/corpus, without train projection."""
    _check(
        type(expected_sha) is str and a0._SHA.fullmatch(expected_sha),
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
        type(info) is dict
        and set(info) == {"path", "sha256", "rows", "bytes", "contains_gold"}
        and info["path"] == "runtime_questions.jsonl"
        and type(info["rows"]) is int
        and info["rows"] == COUNT
        and type(info["bytes"]) is int
        and info["bytes"] > 0
        and type(info["sha256"]) is str
        and a0._SHA.fullmatch(info["sha256"])
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
        len(rows) == COUNT
        and [row["question_id"] for row in rows] == manifest["question_ids"]
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
