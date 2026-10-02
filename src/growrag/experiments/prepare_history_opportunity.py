"""只导出50道新的train开发题，保持旧历史库、76,691段语料及索引不变。

默认只读预检。候选来自预先冻结的7,692背景题，不按难度、检索表现或gold挑题。
只投影id/question；答案、support、context和type/level均不读取。新题不是训练题，
也不是官方测试集。输出目录一旦存在便拒绝重新抽样或覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from growrag.history_library import FrozenHistoryLibrary

from .data_protocol import normalize_question
from .fresh_dev_manifest import _ids, _read_json, _sha
from .prepare_operator_data import _hash_registry
from .protocol import RuntimeQuestion

SCHEMA = "growrag-history-opportunity-data-v1"
SEED = "growrag-candidate-opportunity-20261002-v1"
COUNT = 50
OUTPUT = "data/hotpotqa/history_opportunity_20261002_v1"
SOURCE_MANIFEST = "data/hotpotqa/operator_scale_action_v3/manifest.json"
BACKGROUND_MANIFEST = "data/hotpotqa/shared500_sep27_v1/manifest.json"
LIBRARY = "runs/history_foundation_20261001_v1/combined.json"
PROVENANCE = "data/hotpotqa/official_train_v1_1/mirror_provenance.json"
CORPUS = "data/hotpotqa/operator_scale_action_v3/corpus.jsonl"
INDEX = "data/hotpotqa/operator_scale_action_v3/index.sqlite3"
PINS = {
    SOURCE_MANIFEST: "afc54ffb487b095b9cabf751cc3139fe0b84df93b7577770cb44462766488618",
    BACKGROUND_MANIFEST: "2fb117c03b64be3ab12398a297c82ce3315600c14e33f5dcd56c5c6c5c59e064",
    LIBRARY: "73eab3f46d9d6edaf20c497cc39e6082086f1ab4c43afe45b606abf35397dd88",
    PROVENANCE: "1aeafe38d67da758a6cbd75a60ce3d167c22dbc710bf318c8cef0178b1d35730",
    CORPUS: "618c9eee41cd53c85f8f6cfb87299c959f9d6d8b29fed937089492e0c9bfc7a0",
}
LIBRARY_FINGERPRINT = "8b6aa7f14baa54a7048d539ec791c7884e1029c0aa20afbce0693a55ce7934c7"
_ID = re.compile(r"[0-9a-f]{24}")
_META = {"launch_plan.json", "plan.json", "batch_manifest.json", "source_pool_manifest.json"}


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _path(root, relative):
    path = (root / relative).resolve(strict=True)
    _check(path.is_relative_to(root) and path.is_file(), "input escapes project or is not a file")
    return path


def _fixture(path, root):
    return any(
        part.startswith(("test_", "pytest")) or "_tests" in part
        for part in path.relative_to(root / "runs").parts[:-1]
    )


def _valid_claim_ids(value):
    """合成claim只补充合法ID作保守排除，不把synthetic协议当实际运行。"""
    found = set()

    def values(node):
        if isinstance(node, str) and _ID.fullmatch(node):
            found.add(node)
        elif isinstance(node, dict):
            for child in node.values():
                values(child)
        elif isinstance(node, list):
            for child in node:
                values(child)

    def visit(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if (
                    key == "question_id"
                    or key.endswith("question_ids")
                    or key in {"selected", "roles", "source_expansion_order"}
                ):
                    values(child)
                elif isinstance(child, (list, dict)):
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return found


def _registry(root):
    manifests = set((root / "data/hotpotqa").glob("*/manifest.json"))
    claims = set((root / "runs").rglob("*.claim.json"))
    paths = manifests | {p for p in claims if not _fixture(p, root)}
    for name in _META:
        paths.update(p for p in (root / "runs").rglob(name) if not _fixture(p, root))
    blocked, hashes, inputs = set(), set(), []
    for candidate in sorted(paths):
        path = _path(root, candidate)
        value = _read_json(path)
        # 旧背景语料不是监督题；旧source/cal/eval、已声明与pending题均排除。
        value = {
            k: v
            for k, v in value.items()
            if k not in {"background_question_ids", "corpus_source_question_ids"}
        }
        blocked.update(_ids(value))
        hashes.update(_hash_registry(value))
        inputs.append({"path": path.relative_to(root).as_posix(), "sha256": _sha(path)})
    extra = set()
    for candidate in sorted(claims - paths):
        extra.update(_valid_claim_ids(_read_json(_path(root, candidate))))
    return {
        "inputs": inputs,
        "sha256": _digest(inputs),
        "question_ids": sorted(blocked | extra),
        "normalized_question_sha256s": sorted(hashes),
        "conservative_extra_claim_ids": sorted(extra - blocked),
        "claim_files_checked": len(claims),
        "actual_metadata_files": len(paths),
    }


def _anchors(root):
    for relative, expected in PINS.items():
        _check(_sha(_path(root, relative)) == expected, "pinned input bytes changed")
    _path(root, INDEX)  # Already-built index only; never create/copy/rebuild it here.
    source = _read_json(_path(root, SOURCE_MANIFEST))
    background = _read_json(_path(root, BACKGROUND_MANIFEST))
    provenance = _read_json(_path(root, PROVENANCE))
    library = FrozenHistoryLibrary.from_json(_path(root, LIBRARY).read_text(encoding="utf-8"))
    _check(
        library.fingerprint == LIBRARY_FINGERPRINT
        and set(library.allowed_source_ids) == set(source["roles"]["source"]),
        "history library/source identity differs",
    )
    available = background["background_question_ids"]
    _check(
        len(available) == len(set(available)) == 7692
        and all(isinstance(qid, str) and _ID.fullmatch(qid) for qid in available)
        and background["official_split"] == provenance["official_split"] == "train",
        "availability cohort must be the original 7692 official-train background IDs",
    )
    source_inputs = {r["path"].replace("\\", "/"): r["sha256"] for r in source["input_files"]}
    _check(
        source_inputs.get(BACKGROUND_MANIFEST) == PINS[BACKGROUND_MANIFEST]
        and source_inputs.get(PROVENANCE) == PINS[PROVENANCE]
        and source["artifacts"]["corpus.jsonl"]["sha256"] == PINS[CORPUS]
        and source["artifacts"]["corpus.jsonl"]["rows"] == 76691,
        "old source manifest does not bind this background/provenance/corpus",
    )
    inputs, shards = [], []
    for shard in provenance["shards"]:
        path = _path(root, Path(PROVENANCE).parent / shard["file"])
        relative = path.relative_to(root).as_posix()
        _check(
            path.parent == (root / PROVENANCE).parent
            and _sha(path) == shard["sha256"] == source_inputs.get(relative),
            "official train shard differs from pinned provenance/source manifest",
        )
        shards.append(str(path))
        inputs.append({"path": relative, "sha256": shard["sha256"]})
    _check(shards and len(set(shards)) == len(shards), "missing/duplicate train shards")
    return source, set(available), shards, inputs


def _project(shards, eligible_ids):
    import pyarrow.dataset as ds

    dataset = ds.dataset(shards, format="parquet")
    # No context/type/level/answer/support projection, even for balancing.
    return dataset.to_table(
        columns=["id", "question"], filter=ds.field("id").isin(sorted(eligible_ids))
    ).to_pylist()


def prepare(root, prepare_new=False):
    root = Path(root).resolve(strict=True)
    output = (root / OUTPUT).resolve()
    _check(output.is_relative_to(root), "output escapes project")
    if output.exists():
        raise FileExistsError("bundle exists; do not resample, reuse or overwrite")
    source, available, shards, shard_inputs = _anchors(root)
    registry = _registry(root)
    forbidden = set(registry["question_ids"])
    eligible = available - forbidden
    rows = _project(shards, eligible)
    _check(
        len(rows) == len(eligible)
        and {r["id"] for r in rows} == eligible
        and all(set(r) == {"id", "question"} for r in rows),
        "availability IDs missing/duplicate or unexpected projected columns",
    )
    metadata = [
        (r["id"], hashlib.sha256(normalize_question(r["question"]).encode()).hexdigest())
        for r in rows
    ]
    frequency = Counter(digest for _, digest in metadata)
    blocked_hashes = set(registry["normalized_question_sha256s"])
    good = [
        (qid, digest)
        for qid, digest in metadata
        if frequency[digest] == 1 and digest not in blocked_hashes
    ]
    ordered = sorted(
        good,
        key=lambda r: (hashlib.sha256(f"{SEED}:development:{r[0]}".encode()).hexdigest(), r[0]),
    )
    selected = ordered[:COUNT]
    _check(len(selected) == COUNT, "not enough independent available questions; no replacements")
    ids = [qid for qid, _ in selected]
    texts = {r["id"]: r["question"] for r in rows}
    runtime = b"".join(
        _canonical(
            {
                "question_id": qid,
                "text": texts[qid],
                "dataset": "hotpotqa-history-opportunity-development-v1",
            }
        )
        + b"\n"
        for qid in ids
    )
    manifest = {
        "schema_version": SCHEMA,
        "seed": SEED,
        "count": COUNT,
        "official_split": "train",
        "role": "development_candidate_opportunity_not_training",
        "question_ids": ids,
        "normalized_question_sha256s": [digest for _, digest in selected],
        "question_ids_sha256": _digest(ids),
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
        "corpus_ref": {
            "path": CORPUS,
            "sha256": PINS[CORPUS],
            "rows": 76691,
            "index_path": INDEX,
            "index_version": "growrag-shared-paragraph-sqlite-bm25-v1",
        },
        "source_manifest": {"path": SOURCE_MANIFEST, "sha256": PINS[SOURCE_MANIFEST]},
        "background_manifest": {"path": BACKGROUND_MANIFEST, "sha256": PINS[BACKGROUND_MANIFEST]},
        "source_provenance": {"path": PROVENANCE, "sha256": PINS[PROVENANCE]},
        "parquet_inputs": shard_inputs,
        "exclusion_registry": registry,
        "runtime_input_allowlist": ["runtime_questions.jsonl", "corpus_ref"],
        "runtime_artifact": {
            "path": "runtime_questions.jsonl",
            "sha256": hashlib.sha256(runtime).hexdigest(),
            "rows": COUNT,
            "bytes": len(runtime),
            "contains_gold": False,
        },
        "preflight": {
            "background_ids": len(available),
            "excluded_background_ids": len(available & forbidden),
            "after_id_exclusion": len(rows),
            "eligible_count": len(good),
            "known_hash_matches": sum(d in blocked_hashes for _, d in metadata),
            "duplicate_hash_groups": sum(n > 1 for n in frequency.values()),
            "projected_metadata_sha256": _digest(metadata),
            "old_role_overlap": {
                role: len(set(qids) & set(ids)) for role, qids in source["roles"].items()
            },
        },
        "exporter_sha256": _sha(Path(__file__)),
    }
    if not prepare_new:
        return manifest
    _check(_registry(root) == registry, "reservation registry changed during preparation")
    _check(
        all(_sha(_path(root, p)) == sha for p, sha in PINS.items())
        and all(_sha(_path(root, r["path"])) == r["sha256"] for r in shard_inputs),
        "source bytes changed during preparation",
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "runtime_questions.jsonl").open("xb") as handle:
        handle.write(runtime)
    with (output / "manifest.json").open("xb") as handle:
        handle.write(_canonical(manifest) + b"\n")
    with (output / "manifest.sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{_sha(output / 'manifest.json')}  manifest.json\n")
    return manifest


def load_bundle(project, expected_sha=None):
    """运行侧只返回无标签题和旧语料路径；真实runner应提供外部冻结manifest SHA。"""
    project = Path(project).resolve(strict=True)
    path = _path(project, Path(OUTPUT) / "manifest.json")
    digest = _sha(path)
    _check(
        expected_sha is None or digest == expected_sha, "externally pinned manifest SHA mismatch"
    )
    _check(
        _path(project, Path(OUTPUT) / "manifest.sha256").read_text().strip()
        == f"{digest}  manifest.json",
        "manifest sidecar mismatch",
    )
    manifest = _read_json(path)
    required = {
        "schema_version": SCHEMA,
        "seed": SEED,
        "count": COUNT,
        "official_split": "train",
        "role": "development_candidate_opportunity_not_training",
        "gold_projected": False,
        "memory_updates_allowed": False,
        "model_training": False,
        "runtime_input_allowlist": ["runtime_questions.jsonl", "corpus_ref"],
        "projected_columns": ["id", "question"],
    }
    _check(all(manifest.get(k) == v for k, v in required.items()), "unregistered runtime bundle")
    for key, relative in (
        ("library", LIBRARY),
        ("corpus_ref", CORPUS),
        ("source_manifest", SOURCE_MANIFEST),
        ("background_manifest", BACKGROUND_MANIFEST),
    ):
        entry = manifest[key]
        _check(
            entry["path"] == relative
            and entry["sha256"] == PINS[relative]
            and _sha(_path(project, relative)) == PINS[relative],
            "runtime anchor changed",
        )
    _check(
        manifest["library"]["fingerprint"] == LIBRARY_FINGERPRINT
        and manifest["corpus_ref"]["index_path"] == INDEX
        and manifest["corpus_ref"]["rows"] == 76691
        and manifest["corpus_ref"]["index_version"] == "growrag-shared-paragraph-sqlite-bm25-v1",
        "library/index reference changed",
    )
    _path(project, INDEX)
    ids = manifest["question_ids"]
    _check(
        len(ids) == len(set(ids)) == COUNT and _digest(ids) == manifest["question_ids_sha256"],
        "runtime ID coverage mismatch",
    )
    background = _read_json(_path(project, BACKGROUND_MANIFEST))
    source = _read_json(_path(project, SOURCE_MANIFEST))
    registry = manifest["exclusion_registry"]
    protected = {qid for group in source["roles"].values() for qid in group}
    _check(
        set(ids) <= set(background["background_question_ids"])
        and not set(ids) & (protected | set(registry["question_ids"]))
        and _digest(registry["inputs"]) == registry["sha256"],
        "runtime IDs escape availability or overlap frozen protections",
    )
    info = manifest["runtime_artifact"]
    _check(
        info["path"] == "runtime_questions.jsonl"
        and info["rows"] == COUNT
        and info["contains_gold"] is False,
        "invalid runtime artifact declaration",
    )
    runtime_path = _path(project, Path(OUTPUT) / info["path"])
    _check(
        _sha(runtime_path) == info["sha256"] and runtime_path.stat().st_size == info["bytes"],
        "runtime artifact bytes changed",
    )
    rows = [json.loads(line) for line in runtime_path.read_text(encoding="utf-8").splitlines()]
    _check(
        [r["question_id"] for r in rows] == ids
        and all(set(r) == {"question_id", "text", "dataset"} for r in rows),
        "runtime fields/order changed",
    )
    actual_hashes = [
        hashlib.sha256(normalize_question(r["text"]).encode()).hexdigest() for r in rows
    ]
    _check(
        actual_hashes == manifest["normalized_question_sha256s"]
        and len(set(actual_hashes)) == COUNT
        and not set(actual_hashes) & set(registry["normalized_question_sha256s"]),
        "runtime normalized identity changed",
    )
    return manifest, tuple(RuntimeQuestion(**r) for r in rows), _path(project, CORPUS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-new", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(Path.cwd(), prepare_new=args.prepare_new)
    print(
        json.dumps(
            {
                "prepared": args.prepare_new,
                "output": OUTPUT,
                "question_ids_sha256": result["question_ids_sha256"],
                "preflight": result["preflight"],
                "api_calls": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
