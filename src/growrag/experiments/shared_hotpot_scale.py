"""Freeze 500 train-development queries and 8192 context sources, without API calls.

默认只读计划。显式 prepare-new 才创建新目录；保留原32题，绝不覆盖或补抽。
流式写出共享文档，只在内存保留文档哈希；背景不是训练数据或经验库。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from . import shared_hotpot_dev as base

SCHEMA = "growrag-shared-hotpot-scale-development-v1"
SEED = 2026092702
TARGET_COUNT, CORPUS_SOURCE_COUNT = 500, 8192
FROZEN32 = "data/hotpotqa/shared_sep27_v1/manifest.json"
FROZEN32_SHA = "2dc35c7baa470b9714b063c70221aa70a42654a7d3afe8894e5a2dfe67889ab2"
RUNTIME_FILES = ("runtime_questions.jsonl", "corpus.jsonl")


def _ordered(ids, namespace):
    return sorted(
        ids, key=lambda q: (hashlib.sha256(f"{namespace}:{SEED}:{q}".encode()).hexdigest(), q)
    )


def _quotas(counts: dict[str, int]) -> dict[str, int]:
    """Largest remainder allocation uses only the unused pool's official types."""
    total = sum(counts.values())
    if total < CORPUS_SOURCE_COUNT:
        raise ValueError("insufficient unused corpus sources; never resample from exclusions")
    quotas = {kind: TARGET_COUNT * count // total for kind, count in counts.items()}
    order = sorted(counts, key=lambda k: (-(TARGET_COUNT * counts[k] % total), k))
    for kind in order[: TARGET_COUNT - sum(quotas.values())]:
        quotas[kind] += 1
    return quotas


def _plan(root: Path):
    old_plan, metadata, normalized, source = base._plan(root)
    old_path = base._safe_file(root, FROZEN32)
    if base.legacy._sha(old_path) != FROZEN32_SHA:
        raise ValueError("frozen32 manifest SHA changed")
    frozen = base.legacy._read_json(old_path)
    if (
        frozen.get("schema_version") != base.SCHEMA
        or frozen.get("official_split") != "train"
        or frozen.get("role") != "development"
        or frozen.get("question_ids") != old_plan["question_ids"]
        or frozen.get("selected_ids_sha256") != old_plan["selected_ids_sha256"]
    ):
        raise ValueError("frozen32 reservation/source protocol mismatch")
    excluded = set(old_plan["excluded_question_ids"]) | set(frozen["question_ids"])
    normalized |= set(frozen["selected_normalized_question_sha256s"])
    available = {q: kind for q, (kind, _, _) in metadata.items() if q not in excluded}
    counts = dict(Counter(available.values()))
    quotas = _quotas(counts)
    targets = []
    for kind, count in quotas.items():
        targets.extend(
            _ordered((q for q, t in available.items() if t == kind), "shared-scale-dev")[:count]
        )
    targets = _ordered(targets, "shared-scale-order")
    background = _ordered(set(available) - set(targets), "shared-scale-background")[
        : CORPUS_SOURCE_COUNT - TARGET_COUNT
    ]
    plan = {
        "schema_version": SCHEMA,
        "seed": SEED,
        "role": "development",
        "official_split": "train",
        "dataset": old_plan["dataset"],
        "configuration": old_plan["configuration"],
        "question_ids": targets,
        "question_types": {q: metadata[q][0] for q in targets},
        "question_type_counts": quotas,
        "eligible_type_counts": counts,
        "selected_ids_sha256": base.legacy._digest(targets),
        "background_question_ids": background,
        "background_ids_sha256": base.legacy._digest(background),
        "corpus_source_question_ids": targets + background,
        "corpus_source_count": CORPUS_SOURCE_COUNT,
        "source_record_count": len(metadata),
        "excluded_question_ids": sorted(excluded),
        "excluded_id_count": len(excluded),
        "excluded_ids_sha256": base.legacy._digest(excluded),
        "exclusion_sources": old_plan["exclusion_sources"]
        + [
            {
                "path": FROZEN32,
                "sha256": FROZEN32_SHA,
                "id_count": len(frozen["question_ids"]),
                "ids_sha256": frozen["selected_ids_sha256"],
            }
        ],
        "input_files": old_plan["input_files"],
        "history_cutoff": old_plan["history_cutoff"],
        "selection": (
            "Largest remainder type quotas; SHA256(shared-scale-dev:SEED:ID) per type; "
            "independent shared-scale-order hash mixes runtime order."
        ),
        "background_selection": (
            "Independent SHA256(shared-scale-background:SEED:ID); unused non-target IDs; "
            "prior background reuse allowed."
        ),
        "selection_uses_gold_or_difficulty": False,
        "invalid_label_policy": (
            "Keep selected ID, mark invalid offline; never replace or select by outcome."
        ),
        "old_check_question_or_gold_projected": False,
        "prior32_question_or_gold_projected": False,
        "background_is_memory_training": False,
        "target_contexts_included": True,
        "target_ids_forbidden_for_training_or_memory": targets,
        "training_exclusion_enforced_globally": False,
        "runtime_input_allowlist": list(RUNTIME_FILES),
        "runtime_must_not_read_manifest_gold_or_source_edges": True,
        "official_dev_test_used": False,
        "full_wikipedia": False,
        "document_disjointness_checked": False,
        "source_notice": (
            "Official train HF mirror; closed shared diagnostic corpus, "
            "not official test/fullwiki. "
            "Background contexts are not experience-training permission."
        ),
        "normalization_notice": old_plan["normalization_notice"],
        "implementation_sha256": base.legacy._sha(Path(__file__)),
        "base_implementation_sha256": base.legacy._sha(Path(base.__file__)),
        "projection_implementation_sha256": base.legacy._sha(Path(base.legacy.__file__)),
    }
    return plan, metadata, normalized, source


def plan_shared_scale(root: Path) -> dict:
    """Metadata-only planning; no question, context or gold decoding."""
    return _plan(Path(root).resolve(strict=True))[0]


def _gold(row: dict) -> dict:
    """Invalid annotations remain in the frozen batch, never silently replaced."""
    result = {"question_id": row["_id"], "annotation_status": "valid"}
    try:
        example = base.parse_hotpot_example(row, dataset="hotpotqa-train-shared500-development")
        if example.gold is None or not example.gold.supporting_facts:
            raise ValueError("missing answer or supporting facts")
        result.update(
            answers=list(example.gold.answers),
            supporting_facts=list(example.gold.supporting_facts),
            exact_support=base._exact_support(row["context"], example.gold.supporting_facts),
        )
    except ValueError as error:
        result.update(
            annotation_status="invalid",
            annotation_issue=str(error),
            answers=[],
            supporting_facts=[],
            exact_support=[],
        )
    return result


def prepare_shared_scale(root: Path, output_dir: Path, *, prepare_new: bool = False) -> dict:
    """Create only a new directory; interruption leaves an unpublishable partial bundle."""
    root, output_dir = Path(root).resolve(strict=True), Path(output_dir).resolve()
    if prepare_new and output_dir.exists():
        raise FileExistsError("output exists, including partial runs; choose a new directory")
    plan, metadata, excluded_hashes, source = _plan(root)
    if not prepare_new:
        return plan
    output_dir.mkdir(parents=True, exist_ok=False)
    names = (*RUNTIME_FILES, "gold.jsonl", "corpus_source_edges.jsonl")
    digests, row_counts = {n: hashlib.sha256() for n in names}, Counter()
    seen, title_counts, normalized, invalid = set(), Counter(), set(), []
    targets = set(plan["question_ids"])
    with ExitStack() as stack:
        files = {n: stack.enter_context((output_dir / n).open("xb")) for n in names}

        def emit(name, row):
            blob = base._canonical(row) + b"\n"
            files[name].write(blob)
            digests[name].update(blob)
            row_counts[name] += 1

        handle = stack.enter_context(source.open("rb"))
        mapped = stack.enter_context(mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ))
        for qid in plan["corpus_source_question_ids"]:
            wanted = {"context"} | (
                {"_id", "question", "answer", "supporting_facts"} if qid in targets else set()
            )
            row = base.legacy._project(mapped[metadata[qid][1] : metadata[qid][2]], wanted)
            if qid in targets:
                text = row.get("question")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("invalid frozen question text; never replace question")
                digest = hashlib.sha256(base.normalize_question(text).encode()).hexdigest()
                if digest in normalized or digest in excluded_hashes:
                    raise ValueError("normalized overlap; stop without resampling")
                normalized.add(digest)
                emit(
                    "runtime_questions.jsonl",
                    {
                        "question_id": qid,
                        "text": text,
                        "dataset": "hotpotqa-train-shared500-development",
                    },
                )
                label = _gold(row)
                emit("gold.jsonl", label)
                if label["annotation_status"] == "invalid":
                    invalid.append(qid)
            if not isinstance(row.get("context"), list):
                raise ValueError("invalid context")
            owned = set()
            for paragraph in row["context"]:
                if not isinstance(paragraph, list) or len(paragraph) != 2:
                    raise ValueError("invalid paragraph")
                title, sentences = paragraph
                if (
                    not isinstance(title, str)
                    or not title.strip()
                    or not isinstance(sentences, list)
                    or not all(isinstance(s, str) for s in sentences)
                ):
                    raise ValueError("invalid title/sentence array")
                doc_id = base.document_id(title, sentences)
                if doc_id not in seen:
                    seen.add(doc_id)
                    title_counts[title] += 1
                    emit("corpus.jsonl", {"doc_id": doc_id, "title": title, "sentences": sentences})
                if doc_id not in owned:
                    emit("corpus_source_edges.jsonl", {"doc_id": doc_id, "source_question_id": qid})
                    owned.add(doc_id)
    for item in plan["input_files"] + plan["exclusion_sources"]:
        if base.legacy._sha(base._safe_file(root, item["path"])) != item["sha256"]:
            raise ValueError("source changed during preparation; partial bundle not valid")
    plan["selected_normalized_question_sha256s"] = sorted(normalized)
    plan["annotation_summary"] = {
        "valid": TARGET_COUNT - len(invalid),
        "invalid": len(invalid),
        "invalid_question_ids": invalid,
        "replacement_count": 0,
    }
    plan["corpus_statistics"] = {
        "documents": len(seen),
        "unique_titles": len(title_counts),
        "titles_with_multiple_versions": sum(n > 1 for n in title_counts.values()),
    }
    plan["artifacts"] = {
        n: {
            "sha256": digests[n].hexdigest(),
            "bytes": (output_dir / n).stat().st_size,
            "rows": row_counts[n],
            "contains_gold": n == "gold.jsonl",
            "runtime_safe": n in RUNTIME_FILES,
        }
        for n in names
    }
    manifest = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
    digest = hashlib.sha256(manifest).hexdigest()
    with (output_dir / "manifest.json").open("xb") as handle:
        handle.write(manifest)
    with (output_dir / "manifest.sha256").open("xb") as handle:
        handle.write(f"{digest}  manifest.json\n".encode())
    return {
        "manifest_path": str(output_dir / "manifest.json"),
        "manifest_sha256": digest,
        "question_count": TARGET_COUNT,
        "corpus_sources": CORPUS_SOURCE_COUNT,
        "corpus_documents": len(seen),
        "annotation_summary": plan["annotation_summary"],
        "total_artifact_bytes": sum(a["bytes"] for a in plan["artifacts"].values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("data/hotpotqa/shared500_sep27_v1"))
    parser.add_argument("--prepare-new", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare_shared_scale(args.root, args.output_dir, prepare_new=args.prepare_new),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
