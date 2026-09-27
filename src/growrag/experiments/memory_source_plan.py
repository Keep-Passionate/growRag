"""Freeze independent memory-source IDs without generating or promoting any memory.

中文：从已在共享文档库中的背景题划分来源/调参/探针。文档共用可以，问题监督
不能交叉。默认只计划；显式prepare-new仅导出来源题原问，不读取任何答案。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path

from .data_protocol import normalize_question
from .fresh_dev_manifest import _QID, SOURCE_SHA, _ids, _project, _record_spans, _sha
from .pre_pilot import write_json
from .run_reformer_pilot import FROZEN_MANIFEST_SHA

SEED = "growrag-expel-source-20260927-v1"
COUNTS = {"source": 64, "calibration": 32, "probe": 64}
SCHEMA = "growrag-independent-memory-source-plan-v1"
OLD_MANIFESTS = {
    "data/hotpotqa/shared_sep27_v1/manifest.json": (  # stable reservation hashes
        "2dc35c7baa470b9714b063c70221aa70a42654a7d3afe8894e5a2dfe67889ab2"
    ),
    "data/hotpotqa/representation_sep11_v1/manifest.json": (
        "1182602b67b9dec2eb3ef638e247548b30da507ed4975f1ad9023307d8201612"
    ),
}


def select_roles(background_ids, forbidden_ids):
    """Only IDs determine the split; answers, difficulty and method scores cannot."""
    if any(not isinstance(q, str) or not _QID.fullmatch(q) for q in background_ids):
        raise ValueError("invalid background question ID")
    if len(background_ids) != len(set(background_ids)):
        raise ValueError("duplicate background IDs")
    available = set(background_ids) - set(forbidden_ids)
    ordered = sorted(
        available, key=lambda q: (hashlib.sha256(f"{SEED}:{q}".encode()).hexdigest(), q)
    )
    if len(ordered) < sum(COUNTS.values()):
        raise ValueError("not enough unexposed background questions")
    cursor, roles = 0, {}
    for role, count in COUNTS.items():
        roles[role] = ordered[cursor : cursor + count]
        cursor += count
    return roles


def plan(root):
    root = Path(root).resolve(strict=True)
    frozen = root / "data/hotpotqa/shared500_sep27_v1/manifest.json"
    if _sha(frozen) != FROZEN_MANIFEST_SHA:
        raise ValueError("shared500 manifest differs from reviewed source")
    manifest = json.loads(frozen.read_bytes())
    if manifest["official_split"] != "train" or manifest["official_dev_test_used"]:
        raise ValueError("official train only")
    # This snapshot already includes old source/target/check reservations and32 IDs.
    forbidden = set(manifest["excluded_question_ids"]) | set(manifest["question_ids"])
    inputs = [{"path": frozen.relative_to(root).as_posix(), "sha256": FROZEN_MANIFEST_SHA}]
    for name, digest in OLD_MANIFESTS.items():
        if _sha(root / name) != digest:
            raise ValueError("earlier reservation/duplicate audit changed")
        inputs.append({"path": name, "sha256": digest})
    exposure_files = sorted(
        [*(root / "runs").glob("*/launch_plan.json"), *(root / "runs").glob("*.claim.json")]
    )
    for path in exposure_files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("exposure metadata escapes project")
        value = json.loads(path.read_bytes())
        forbidden.update(_ids(value))
        inputs.append({"path": path.relative_to(root).as_posix(), "sha256": _sha(path)})
    roles = select_roles(manifest["background_question_ids"], forbidden)
    return {
        "schema_version": SCHEMA,
        "seed": SEED,
        "official_split": "train",
        "official_dev_test_used": False,
        "roles": roles,
        "counts": COUNTS,
        "forbidden_question_ids": sorted(forbidden),
        "source_manifest_sha256": FROZEN_MANIFEST_SHA,
        "corpus_sha256": manifest["artifacts"]["corpus.jsonl"]["sha256"],
        "source_data_sha256": SOURCE_SHA,
        "exposure_inputs": inputs,
        "selection_uses_answers_or_scores": False,
        "source_gold_exported": False,
        "calibration_probe_text_exported": False,
        "memory_cards_created": 0,
        "source_training_started": False,
        "document_disjoint": False,
        "notice": "Existing shared corpus, independent question IDs. Background contexts were "
        "previously retrieval-only; this NEW plan designates only source IDs for future "
        "experience collection. Calibration/probe never build memory. No claim of semantic "
        "near-duplicate or entity disjointness; official test remains untouched.",
    }


def source_questions(raw_path, roles, forbidden_hashes):
    """Project only ID/text for chosen sources; no labels or held-out question text."""
    wanted, found, seen_hashes = set(roles["source"]), {}, set()
    with (
        Path(raw_path).open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        for start, end in _record_spans(mapped):
            raw = mapped[start:end]
            qid = _project(raw, {"_id"}).get("_id")
            if qid not in wanted:
                continue
            row = _project(raw, {"_id", "question"})
            text = row.get("question")
            if qid in found or not isinstance(text, str) or not text.strip():
                raise ValueError("duplicate or invalid frozen source question; no replacement")
            digest = hashlib.sha256(normalize_question(text).encode()).hexdigest()
            if digest in seen_hashes or digest in forbidden_hashes:
                raise ValueError("normalized source overlap; stop without replacement")
            seen_hashes.add(digest)
            found[qid] = {
                "question_id": qid,
                "text": text,
                "dataset": "hotpotqa-train-independent-memory-source-v1",
            }
    if set(found) != wanted:
        raise ValueError("frozen source ID missing")
    return [found[qid] for qid in roles["source"]]


def prepare(root, output, *, prepare_new=False):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    if prepare_new and output.exists():
        raise FileExistsError("source plan already exists; never overwrite or silently resplit")
    result = plan(root)
    if not prepare_new:
        return result
    source = root / "data/hotpotqa/official_train_v1_1/hotpot_train_v1.1_hf_mirror.json"
    if _sha(source) != SOURCE_SHA:
        raise ValueError("official train mirror hash mismatch")
    # Same-source global duplicate audit is already part of shared500 exclusions.
    # Additionally check normalized source text against all earlier exposed hashes.
    hashes = set()
    for name in (
        "data/hotpotqa/shared500_sep27_v1/manifest.json",
        "data/hotpotqa/shared_sep27_v1/manifest.json",
    ):
        item = json.loads((root / name).read_bytes())
        hashes.update(item["selected_normalized_question_sha256s"])
    old = json.loads((root / "data/hotpotqa/representation_sep11_v1/manifest.json").read_bytes())
    hashes.update(old["exclusions"]["requested_normalized_question_sha256s"])
    for group in old["exclusions"]["duplicate_normalized_groups"]:
        hashes.add(group["normalized_question_sha256"])
    rows = source_questions(source, result["roles"], hashes)
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "source_runtime_questions.jsonl"
    with runtime.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result["artifacts"] = {
        runtime.name: {"sha256": _sha(runtime), "rows": len(rows), "contains_gold": False}
    }
    result["implementation_sha256"] = _sha(Path(__file__))
    write_json(output / "manifest.json", result)
    with (output / "manifest.sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{_sha(output / 'manifest.json')}  manifest.json\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-new", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(args.project_root, args.output, prepare_new=args.prepare_new)
    print(
        json.dumps(
            {k: result[k] for k in ("schema_version", "counts", "memory_cards_created")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
