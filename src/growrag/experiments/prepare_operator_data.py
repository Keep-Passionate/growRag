"""Freeze source500/calibration100/dev500 without reading gold or calling an API.

中文：默认只读预检；--prepare-new 才生成新目录。旧目录不覆盖，旧实验不修改。
只投影题目ID/文本，旧封存ID先过滤；评测上下文作为无标签检索语料加入共享索引。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fresh_dev_manifest import _ids, _read_json, _sha
from .operator_data_plan import SOURCE_SIZES, build_operator_data_plan, question_metadata
from .shared_hotpot_dev import document_id

TRAIN_DIR = "data/hotpotqa/official_train_v1_1"
DEV_DIR = "data/hotpotqa/official_dev_v1"
SHARED = "data/hotpotqa/shared500_sep27_v1/manifest.json"
RESERVATIONS = (
    SHARED,
    "data/hotpotqa/shared_sep27_v1/manifest.json",
    "data/hotpotqa/representation_sep11_v1/manifest.json",
    "data/hotpotqa/memory_sources_sep27_v1/manifest.json",
    "data/hotpotqa/train_preview_200_v1/manifest.json",
)
DEFAULT_OUTPUT = "data/hotpotqa/operator_scale_sep30_v1"


def _safe(root: Path, name: str | Path) -> Path:
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("input file must stay within project")
    return path


def _artifact(path: Path, *, rows: int | None = None) -> dict:
    result = {"sha256": _sha(path), "bytes": path.stat().st_size}
    if rows is not None:
        result["rows"] = rows
    return result


def _hash_registry(value: object) -> set[str]:
    """Read only named normalized-question hash fields, never prose or labels."""
    found = set()

    def collect(node):
        if isinstance(node, str):
            if len(node) != 64 or any(c not in "0123456789abcdef" for c in node):
                raise ValueError("invalid registered normalized question hash")
            found.add(node)
        elif isinstance(node, list):
            for item in node:
                collect(item)
        elif isinstance(node, dict):
            for item in node.values():
                collect(item)
        else:
            raise ValueError("invalid normalized hash registry")

    def visit(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if "normalized_question" in key and ("sha256" in key or "hashes" in key):
                    collect(child)
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return found


def _reservation_ids(value: dict) -> set[str]:
    # Background contexts are NOT supervised exposure. Do not exclude all8192.
    return _ids(
        {
            key: child
            for key, child in value.items()
            if key not in {"background_question_ids", "corpus_source_question_ids"}
        }
    )


def _inputs(root: Path):
    paths = {_safe(root, name) for name in RESERVATIONS}
    paths.update((root / "runs").rglob("launch_plan.json"))
    paths.update((root / "runs").rglob("*.claim.json"))
    forbidden, fingerprints, inputs = set(), set(), []
    for path in sorted(paths):
        path = _safe(root, path)
        value = _read_json(path)
        forbidden.update(_reservation_ids(value))
        fingerprints.update(_hash_registry(value))
        inputs.append({"path": path.relative_to(root).as_posix(), **_artifact(path)})
    shared = _read_json(_safe(root, SHARED))
    if shared.get("official_split") != "train":
        raise ValueError("shared source corpus must come from official train")
    corpus = _safe(root, Path(SHARED).parent / "corpus.jsonl")
    expected = shared["artifacts"]["corpus.jsonl"]
    if _sha(corpus) != expected["sha256"]:
        raise ValueError("old shared corpus SHA mismatch")
    inputs.append({"path": corpus.relative_to(root).as_posix(), **_artifact(corpus)})
    return forbidden, fingerprints, inputs, shared, corpus


def _mirror(root: Path, directory: str, split: str):
    provenance_path = _safe(root, f"{directory}/mirror_provenance.json")
    provenance = _read_json(provenance_path)
    if provenance.get("official_split") != split:
        raise ValueError("official mirror split mismatch")
    data_path = _safe(root, Path(directory) / provenance["output_file"])
    if _sha(data_path) != provenance["output_sha256"]:
        raise ValueError("raw mirror SHA mismatch")
    entries = [
        {"path": p.relative_to(root).as_posix(), **_artifact(p)}
        for p in (provenance_path, data_path)
    ]
    shards = []
    for shard in provenance["shards"]:
        path = _safe(root, Path(directory) / shard["file"])
        if _sha(path) != shard["sha256"]:
            raise ValueError("mirror parquet SHA mismatch")
        entries.append({"path": path.relative_to(root).as_posix(), **_artifact(path)})
        shards.append(path)
    if not shards:
        raise ValueError("mirror has no parquet shards")
    return shards, provenance, entries


def _read_candidates(shards: list[Path], split: str, forbidden: set[str]):
    """No answer/support/type columns; old IDs filtered before text materialization."""
    try:
        import pyarrow.dataset as ds
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required for label-free parquet column projection"
        ) from error
    dataset = ds.dataset([str(path) for path in shards], format="parquet")
    all_ids = dataset.to_table(columns=["id"])["id"].to_pylist()
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate raw mirror question IDs")
    rows = dataset.to_table(
        columns=["id", "question"], filter=~ds.field("id").isin(sorted(forbidden))
    ).to_pylist()
    metadata, texts = [], {}
    for row in rows:
        item = question_metadata(row["id"], row["question"], official_split=split)
        metadata.append(item)
        texts[item.question_id] = row["question"]
    return metadata, texts, len(all_ids)


def _dev_contexts(shards: list[Path], selected: list[str]) -> dict[str, list]:
    import pyarrow.dataset as ds

    dataset = ds.dataset([str(path) for path in shards], format="parquet")
    rows = dataset.to_table(
        columns=["id", "context"], filter=ds.field("id").isin(selected)
    ).to_pylist()
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError("duplicate frozen dev context ID")
        context = row["context"]
        result[row["id"]] = [
            list(pair) for pair in zip(context["title"], context["sentences"], strict=True)
        ]
    if set(result) != set(selected):
        raise ValueError("frozen dev context missing; never resample")
    return result


def _canonical(row: object) -> bytes:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _write_corpus(old: Path, target: Path, contexts: dict[str, list], order: list[str]) -> dict:
    """Keep old bytes/order; append new content-hashed docs, never merge by title."""
    seen, titles = set(), set()
    old_rows, additions = 0, 0
    with old.open("rb") as source, target.open("xb") as destination:
        for line in source:
            row = json.loads(line)
            if set(row) != {"doc_id", "title", "sentences"}:
                raise ValueError("unexpected field in frozen runtime corpus")
            identity = document_id(row["title"], row["sentences"])
            if identity != row["doc_id"] or identity in seen or not line.endswith(b"\n"):
                raise ValueError("old corpus duplicate/hash/newline mismatch")
            seen.add(identity)
            titles.add(row["title"])
            old_rows += 1
            destination.write(line)
        for qid in order:
            for title, sentences in contexts[qid]:
                if (
                    not isinstance(title, str)
                    or not title.strip()
                    or not isinstance(sentences, list)
                    or not all(isinstance(s, str) for s in sentences)
                ):
                    raise ValueError("invalid unlabeled dev context")
                identity = document_id(title, sentences)
                if identity in seen:
                    continue
                seen.add(identity)
                titles.add(title)
                destination.write(
                    _canonical({"doc_id": identity, "title": title, "sentences": sentences}) + b"\n"
                )
                additions += 1
    return {
        "old_documents": old_rows,
        "added_documents": additions,
        "total_documents": len(seen),
        "unique_titles": len(titles),
    }


def prepare_operator_data(
    root: Path,
    output_dir: Path,
    *,
    prepare_new: bool = False,
    source_sizes: tuple[int, ...] = SOURCE_SIZES,
    calibration_count: int = 100,
    evaluation_count: int = 500,
) -> dict:
    """Dry run by default; a partial output cannot be reused after an interruption."""
    root = Path(root).resolve(strict=True)
    output_dir = (root / output_dir).resolve()
    if not output_dir.is_relative_to(root) or output_dir == root:
        raise ValueError("output directory must be inside project, never its root")
    if prepare_new and output_dir.exists():
        raise FileExistsError("output already exists; never overwrite or silently resplit")
    forbidden, fingerprints, inputs, shared, old_corpus = _inputs(root)
    train_shards, train_provenance, train_inputs = _mirror(root, TRAIN_DIR, "train")
    dev_shards, dev_provenance, dev_inputs = _mirror(root, DEV_DIR, "dev")
    train_metadata, train_texts, train_count = _read_candidates(train_shards, "train", forbidden)
    dev_metadata, dev_texts, dev_count = _read_candidates(dev_shards, "dev", forbidden)
    if train_count != train_provenance["row_count"] or dev_count != dev_provenance["row_count"]:
        raise ValueError("projected mirror row count mismatch")
    eligible = set(shared["background_question_ids"]) - forbidden
    result = build_operator_data_plan(
        train_metadata,
        dev_metadata,
        forbidden_ids=forbidden,
        forbidden_question_hashes=fingerprints,
        eligible_source_ids=eligible,
        source_sizes=source_sizes,
        calibration_count=calibration_count,
        evaluation_count=evaluation_count,
    )
    result.update(
        {
            "preparation_schema": "growrag-dynamic-operator-runtime-bundle-v1",
            "input_files": inputs + train_inputs + dev_inputs,
            "prior_reservation_count": len(forbidden),
            "prior_normalized_hash_count": len(fingerprints),
            "old_reserved_text_projected": False,
            "gold_projected": False,
            "api_calls": 0,
            "raw_source_rows": {"train": train_count, "dev": dev_count},
            "runtime_input_allowlist": [
                "source_runtime_questions.jsonl",
                "calibration_runtime_questions.jsonl",
                "evaluation_runtime_questions.jsonl",
                "corpus.jsonl",
            ],
            "runtime_must_not_read_manifest_or_raw_source": True,
            "corpus_policy": "Old shared corpus bytes plus all unlabeled contexts of fixed dev "
            "questions; identical corpus/index settings for every arm and source scale. "
            "Closed-corpus diagnostic, not official distractor/fullwiki leaderboard protocol.",
            "old_corpus_sha256": shared["artifacts"]["corpus.jsonl"]["sha256"],
            "full_wikipedia": False,
            "normalized_exclusion_coverage": "All new eligible train/dev questions and all "
            "previously registered normalized hashes are checked. Old reserved IDs are skipped "
            "before question text projection. Old reservations without per-question hash mapping "
            "cannot be certified cross-split text-disjoint; no sealed text was read to fill gaps.",
            "compatibility_policy": "Only a predeclared prefix of calibration may tune prompts; "
            "no calibration memory writes. Seal all evaluation predictions before label scoring; "
            "no evaluation-driven prompt, router, memory or threshold changes.",
        }
    )
    if not prepare_new:
        return result

    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for role, ids in result["roles"].items():
        path = output_dir / f"{role}_runtime_questions.jsonl"
        texts = dev_texts if role == "evaluation" else train_texts
        with path.open("xb") as handle:
            for qid in ids:
                handle.write(
                    _canonical(
                        {
                            "question_id": qid,
                            "text": texts[qid],
                            "dataset": f"hotpotqa-dynamic-operator-{role}-v1",
                        }
                    )
                    + b"\n"
                )
        artifacts[path.name] = {
            **_artifact(path, rows=len(ids)),
            "contains_gold": False,
            "runtime_safe": True,
        }
    contexts = _dev_contexts(dev_shards, result["roles"]["evaluation"])
    corpus = output_dir / "corpus.jsonl"
    statistics = _write_corpus(old_corpus, corpus, contexts, result["roles"]["evaluation"])
    if statistics["old_documents"] != shared["artifacts"]["corpus.jsonl"]["rows"]:
        raise ValueError("old corpus row count mismatch; partial bundle invalid")
    artifacts[corpus.name] = {
        **_artifact(corpus, rows=statistics["total_documents"]),
        "contains_gold": False,
        "runtime_safe": True,
    }
    for item in result["input_files"]:
        if _sha(_safe(root, item["path"])) != item["sha256"]:
            raise ValueError("input changed during preparation; partial bundle invalid")
    result["corpus_statistics"], result["artifacts"] = statistics, artifacts
    result["implementation_sha256"] = _sha(Path(__file__))
    manifest = output_dir / "manifest.json"
    with manifest.open("xb") as handle:
        handle.write(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
        )
    with (output_dir / "manifest.sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{_sha(manifest)}  manifest.json\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--prepare-new", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_operator_data(args.root, args.output_dir, prepare_new=args.prepare_new)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "preparation_schema",
                    "counts",
                    "source_sizes",
                    "evaluation_protocol",
                    "role_ids_sha256",
                    "prior_reservation_count",
                    "prior_normalized_hash_count",
                    "api_calls",
                )
            }
            | {"prepared": args.prepare_new, "corpus_statistics": result.get("corpus_statistics")},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
