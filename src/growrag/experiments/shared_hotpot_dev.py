"""Offline NEW32 + 992-background shared Hotpot train diagnostic corpus.

中文：默认只给计划；--prepare-new 才落盘。题目、gold、文档分别导出；
背景只是检索语料，不是经验库。只解码新32题标签，旧check永不投影。
The corpus deliberately includes all target contexts; this is NOT fullwiki/test.
"""

# ruff: noqa: E501 -- The immutable path/SHA registry is intentionally one row per source.
from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from collections import Counter
from pathlib import Path

from . import fresh_dev_manifest as legacy
from .data_protocol import normalize_question
from .hotpot import parse_hotpot_example

SCHEMA = "growrag-shared-hotpot-development-v1"
SEED = 20260927
SOURCE_SHA = legacy.SOURCE_SHA
DATA_ROOT = "data/hotpotqa/official_train_v1_1"
REPRESENTATION = "data/hotpotqa/representation_sep11_v1/manifest.json"
PREVIEW = "data/hotpotqa/train_preview_200_v1/manifest.json"
HISTORY_CUTOFF = "2026-09-27"
EXPOSURE_SHA256 = dict(
    line.split()
    for line in """
data/hotpotqa/representation_sep11_v1/manifest.json 1182602b67b9dec2eb3ef638e247548b30da507ed4975f1ad9023307d8201612
data/hotpotqa/train_preview_200_v1/manifest.json 3fe541b4fa0baffb19785ac8d5d482081e0cd0ea9b3bd292ea490191aac33203
runs/2026-09-06_hotpot_pilot_v1/launch_plan.json 36918bcc1e11f4d0abb4485769655c1e9c3c0fde4c893009f3b3691cf1f238aa
runs/2026-09-06_pre_pilot_32_16_v1/batch_manifest.json e1d35791c9925a6671eb897f0b4300489d6864a77a77a7d186f8e44225f9e030
runs/2026-09-06_pre_pilot_32_16_v1/launch_plan.json 6b758113d5591486f8beec8c0e947b52cb67efb8a3858d04ff6c5ccd6c49104d
runs/2026-09-07_pattern_probe_v1/report.json 2e2010eb5b0d21bb07fe65307fcb949096d93f0c0d12299ecef2d6d4dc535fc2
runs/2026-09-07_pre_source_retrieval_probe_v1/probe.json 95015cc5b24092da2be015e7d51c868a09ca7bea30a20b5f6053566581c23423
runs/2026-09-11_source_pool_64_v1/launch_plan.json 95533b9dc5dcb2c1ef3bddff952a5e5dd2edb126f36607de6469905bef3640a1
runs/2026-09-11_source_pool_64_v1/source_pool_manifest.json 87a9d704b035e65aaa0b4e6261a4ae5d57fe1c1839af42a3cd09880fa2ed768b
runs/2026-09-14_fresh_baseline_32_v1/launch_plan.json fd24f5895701879009f185dd4522c188ba40833e5d5a63c24ad9ab79cfedd686
runs/2026-09-16_prompt_plan_v1/plan.json 627d8557b152c1cd6801ab17d4ef4218d877cdaaad0859a684ba35a5497adfa8
runs/2026-09-20_pre_opportunity_8_v1/launch_plan.json 5633dc1c8a150c5dca348bca1ec32c916e1a739d5731f00a2ce649553a0651ec
runs/2026-09-20_pre_opportunity_plan_final/plan.json 5633dc1c8a150c5dca348bca1ec32c916e1a739d5731f00a2ce649553a0651ec
runs/2026-09-20_pre_opportunity_plan_v1/plan.json e543e42a2231d00a3fa8570c837b22a7daf22cf61054fa3e0c14d166d6b4af5a
runs/2026-09-20_pre_opportunity_plan_v2/plan.json e543e42a2231d00a3fa8570c837b22a7daf22cf61054fa3e0c14d166d6b4af5a
runs/2026-09-23_bounded_system_8_v1/launch_plan.json d48d0b41348ccbf53339aab6e97bc79120ab16b3a2fde1a3c26a4c614a265ccc
runs/2026-09-23_bounded_system_dry_v1/plan.json a1ea97d6e97eb53a1e43eb664a68df4f62bbac7814c5efb7b5df4dc859d8f620
runs/2026-09-23_bounded_system_unstarted7_v2/launch_plan.json f88d5736f44de86876688e6ab82f7fb4d6c2741502482917ef5f9b22082909e2
runs/2026-09-23_bounded_system_unstarted_dry_v2/plan.json 5bb9751b57d9a5bc578eee6c7134cc8d2305dec4983f05b6760dcc87903fd00e
runs/2026-09-23_judge_paired_stability_preflight_v1/plan.json a817933c3f743960a44367e25a16dc2dc3d8bdd019cf8397984462e1c6f06e8b
runs/2026-09-23_judge_paired_stability_v1/launch_plan.json 0f948fbf39d372ca1a681a6f23be05a6f5a5216b28985ede514d0e44bafa3a84
runs/2026-09-24_dualrag_debug8_v1/launch_plan.json 41a10b85a54c4a9fb04e3443f8cc2ffff88b766b775f6d712deade3219700cb6
runs/2026-09-24_dualrag_plan_v1/plan.json 41a10b85a54c4a9fb04e3443f8cc2ffff88b766b775f6d712deade3219700cb6
runs/2026-09-27_s2g_author_api_capacity8_plan_v4/plan.json 08ea2167865db392faae496bec9237d18b70da29735996d0dfb7d48c28cc1ebd
runs/2026-09-27_s2g_author_api_capacity8_v4/launch_plan.json 5f7fea53aa4602d2064a68165eca3709dc5a1376430376ac9272723187bc0114
runs/2026-09-27_s2g_author_api_debug8_v1/launch_plan.json 7e6b9ef59487304fa08db9f5b967c45cfc715f0ba7db1eeac6a8e73dbf5eb19f
runs/2026-09-27_s2g_author_api_json8_v2/launch_plan.json d0e767a34e5c61a5e3c94e77764e8fabc589661cb0136ef82abd7da22a93ebfd
runs/2026-09-27_s2g_author_api_plan_v1/plan.json 57eca1fdbb37fc963d6d6558283914d2dc78b995b37ed250f467b44b60f377fe
runs/2026-09-27_s2g_author_api_schema8_v3/launch_plan.json 59630290fda95771b3b571047186c2ec237210099a95aba4822049d82940321d
""".strip().splitlines()
)


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def document_id(title: str, sentences: list[str]) -> str:
    """Full ordered sentence array, including empties: never merge by title."""
    return hashlib.sha256(_canonical([title, sentences])).hexdigest()


def _exact_support(context: list, facts: tuple) -> list[dict]:
    """Gold-only version binding; runtime corpus never receives support labels."""
    bound = []
    for title, index in facts:
        candidates = [sentences for candidate, sentences in context if candidate == title]
        if len(candidates) != 1 or index >= len(candidates[0]) or not candidates[0][index].strip():
            raise ValueError("missing/ambiguous exact gold support in target context")
        sentences = candidates[0]
        bound.append(
            {
                "doc_id": document_id(title, sentences),
                "title": title,
                "sentence_index": index,
                "text_sha256": hashlib.sha256(sentences[index].encode()).hexdigest(),
            }
        )
    return bound


def _safe_file(root: Path, name: str) -> Path:
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("source file escapes its root")
    return path


def _inputs(root: Path):
    observed = {
        p.relative_to(root).as_posix()
        for p in (root / "runs").glob("2026-??-??_*/*")
        if p.is_file() and p.name in legacy.META_NAMES and p.parent.name[:10] <= HISTORY_CUTOFF
    }
    if observed - EXPOSURE_SHA256.keys():
        raise ValueError("unreviewed historical plan; update exclusion registry explicitly")
    excluded, origins, objects = set(), [], {}
    for name, digest in EXPOSURE_SHA256.items():
        path = _safe_file(root, name)
        if legacy._sha(path) != digest:
            raise ValueError(f"exclusion source SHA changed: {name}")
        value = legacy._read_json(path)
        if name.endswith("2026-09-07_pattern_probe_v1/report.json"):
            value = {"manifest": value["manifest"]}  # Ignore action-level `selected`.
        ids = legacy._ids(value)
        if not ids:
            raise ValueError("exclusion source has no recognized IDs")
        objects[name] = value
        excluded.update(ids)
        origins.append(
            {
                "path": name,
                "sha256": digest,
                "id_count": len(ids),
                "ids_sha256": legacy._digest(ids),
            }
        )
    old, preview = objects[REPRESENTATION], objects[PREVIEW]
    audit = old["exclusions"]
    groups = audit["duplicate_normalized_groups"]
    duplicate_ids = {qid for group in groups for qid in group["question_ids"]}
    if (
        old.get("source_sha256") != SOURCE_SHA
        or old.get("official_split") != "train"
        or old.get("complete_train_declared") is not True
        or audit.get("duplicate_policy") != "exclude_all"
        or old.get("checks", {}).get("normalized_duplicates_handled") is not True
        or len(set(old["source_expansion_order"])) != 256
        or len(set(old["selected"]["target"])) != 32
        or len(set(old["selected"]["check"])) != 24
        or not set(old["selected"]["check"]) <= set(old["selected"]["target"])
        or set(old["source_expansion_order"]) & set(old["selected"]["target"])
        or len(legacy._ids(preview)) != 200
        or preview.get("official_split") != "train"
        or preview.get("record_count") != 200
        or len(audit["requested_normalized_question_sha256s"]) != 200
        or len(duplicate_ids) != audit.get("duplicate_normalized_member_count")
        or not duplicate_ids <= set(audit["excluded_union_question_ids"])
    ):
        raise ValueError("complete same-source exclusion/duplicate audit required")
    mirror_path = _safe_file(root, f"{DATA_ROOT}/mirror_provenance.json")
    mirror = legacy._read_json(mirror_path)
    source = _safe_file(mirror_path.parent, mirror["output_file"])
    hashes = [
        {"path": str(p.relative_to(root)), "sha256": legacy._sha(p)}
        for p in [
            mirror_path,
            source,
            *[_safe_file(mirror_path.parent, s["file"]) for s in mirror["shards"]],
        ]
    ]
    if (
        mirror.get("official_split") != "train"
        or mirror.get("original_cmu_json_bytes") is not False
        or hashes[1]["sha256"] != SOURCE_SHA
        or mirror["output_sha256"] != SOURCE_SHA
        or len(mirror["shards"]) != 2
        or sum(s["row_count"] for s in mirror["shards"]) != mirror["row_count"]
        or any(
            h["sha256"] != s["sha256"] for h, s in zip(hashes[2:], mirror["shards"], strict=True)
        )
    ):
        raise ValueError("train mirror/source shard SHA mismatch")
    normalized = set(audit["requested_normalized_question_sha256s"])
    normalized.update(g["normalized_question_sha256"] for g in groups)
    return source, mirror, excluded, normalized, origins, hashes


def _ordered(ids, namespace):
    return sorted(
        ids, key=lambda qid: (hashlib.sha256(f"{namespace}:{SEED}:{qid}".encode()).hexdigest(), qid)
    )


def _plan(root: Path):
    source, mirror, excluded, normalized, origins, hashes = _inputs(root)
    metadata = {}
    with (
        source.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        for start, end in legacy._record_spans(mapped):
            row = legacy._project(mapped[start:end], {"_id", "type"})
            qid, kind = row.get("_id"), row.get("type")
            if (
                not isinstance(qid, str)
                or not legacy._QID.fullmatch(qid)
                or qid in metadata
                or kind not in legacy.KINDS
            ):
                raise ValueError("invalid/duplicate source metadata")
            metadata[qid] = (kind, start, end)
    if len(metadata) != mirror["row_count"] or not excluded <= metadata.keys():
        raise ValueError("source completeness or exposure coverage mismatch")
    pools = {
        kind: _ordered(
            (q for q, (t, _, _) in metadata.items() if t == kind and q not in excluded),
            "shared-dev",
        )
        for kind in legacy.KINDS
    }
    if any(len(pool) < 16 for pool in pools.values()):
        raise ValueError("insufficient unused questions for 16+16")
    targets = [
        q for pair in zip(pools["bridge"][:16], pools["comparison"][:16], strict=True) for q in pair
    ]
    background = _ordered(set(metadata) - excluded - set(targets), "shared-background")[:992]
    if len(background) != 992:
        raise ValueError("insufficient unused background records")
    plan = {
        "schema_version": SCHEMA,
        "seed": SEED,
        "official_split": "train",
        "role": "development",
        "configuration": "distractor",
        "dataset": "hotpotqa/hotpot_qa",
        "question_ids": targets,
        "question_types": {q: metadata[q][0] for q in targets},
        "question_type_counts": {"bridge": 16, "comparison": 16},
        "selected_ids_sha256": legacy._digest(targets),
        "background_question_ids": background,
        "background_ids_sha256": legacy._digest(background),
        "corpus_source_question_ids": targets + background,
        "corpus_source_count": 1024,
        "source_record_count": len(metadata),
        "eligible_type_counts": {k: len(v) for k, v in pools.items()},
        "target_selection": "SHA256(shared-dev:20260927:ID),16 per type,interleaved",
        "background_selection": "SHA256(shared-background:20260927:ID),992 non-target unused IDs",
        "excluded_question_ids": sorted(excluded),
        "excluded_id_count": len(excluded),
        "excluded_ids_sha256": legacy._digest(excluded),
        "history_cutoff": HISTORY_CUTOFF,
        "exclusion_sources": origins,
        "input_files": hashes,
        "old_check_question_or_gold_projected": False,
        "normalized_duplicate_audit_reused": True,
        "selection_uses_gold": False,
        "corpus_selection_uses_gold": False,
        "target_contexts_included": True,
        "target_ids_forbidden_for_training_or_memory": targets,
        "background_is_memory_training": False,
        "training_exclusion_enforced_globally": False,
        "training_exclusion_notice": "Downstream runners must enforce this reservation; this exporter is not a global training firewall.",
        "official_dev_test_used": False,
        "full_wikipedia": False,
        "document_disjointness_checked": False,
        "source_notice": "Official train HF mirror, not CMU original bytes; closed shared 32-target + 992-background diagnostic corpus, not test/fullwiki.",
        "normalization_notice": "Reuse same-source full-corpus duplicate audit; no semantic near-duplicate guarantee.",
        "implementation_sha256": legacy._sha(Path(__file__)),
        "projection_implementation_sha256": legacy._sha(Path(legacy.__file__)),
    }
    return plan, metadata, normalized, source


def plan_shared_dev(root: Path) -> dict:
    """Read-only dry plan: metadata only, no question/answer/context decoding."""
    return _plan(Path(root).resolve(strict=True))[0]


def prepare_shared_dev(root: Path, output_dir: Path, *, prepare_new: bool = False) -> dict:
    """Explicit opt-in creates a NEW directory; partial failures are never overwritten."""
    root, output_dir = Path(root).resolve(strict=True), Path(output_dir).resolve()
    if prepare_new and output_dir.exists():
        raise FileExistsError("output already exists; choose a new directory")
    plan, metadata, excluded_hashes, source = _plan(root)
    if not prepare_new:
        return plan
    runtime, gold, docs, owners, normalized = [], [], {}, {}, set()
    targets = set(plan["question_ids"])
    with (
        source.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        for qid in plan["corpus_source_question_ids"]:
            wanted = {"context"} | (
                {"_id", "question", "answer", "supporting_facts", "type"}
                if qid in targets
                else set()
            )
            row = legacy._project(mapped[metadata[qid][1] : metadata[qid][2]], wanted)
            if qid in targets:
                example = parse_hotpot_example(
                    row, dataset="hotpotqa-train-shared-development-sep27"
                )
                if example.gold is None:
                    raise ValueError("new target lacks offline gold")
                digest = hashlib.sha256(
                    normalize_question(example.question.text).encode()
                ).hexdigest()
                if digest in normalized or digest in excluded_hashes:
                    raise ValueError("normalized question overlap: stop, never resample")
                normalized.add(digest)
                runtime.append(
                    {
                        "question_id": qid,
                        "text": example.question.text,
                        "dataset": example.question.dataset,
                    }
                )
                gold.append(
                    {
                        "question_id": qid,
                        "answers": list(example.gold.answers),
                        "supporting_facts": list(example.gold.supporting_facts),
                        "exact_support": _exact_support(
                            row["context"], example.gold.supporting_facts
                        ),
                    }
                )
            if not isinstance(row["context"], list):
                raise ValueError("invalid context")
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
                doc_id = document_id(title, sentences)
                docs[doc_id] = {"doc_id": doc_id, "title": title, "sentences": sentences}
                owners.setdefault(doc_id, set()).add(qid)
    for item in plan["input_files"] + plan["exclusion_sources"]:
        if legacy._sha(_safe_file(root, item["path"])) != item["sha256"]:
            raise ValueError("source changed during preparation")
    payloads = {
        "runtime_questions.jsonl": runtime,
        "gold.jsonl": gold,
        "corpus.jsonl": [docs[k] for k in sorted(docs)],
        "corpus_sources.jsonl": [
            {"doc_id": k, "source_question_ids": sorted(owners[k])} for k in sorted(docs)
        ],
    }
    blobs = {
        name: b"".join(_canonical(row) + b"\n" for row in rows) for name, rows in payloads.items()
    }
    plan["artifacts"] = {
        name: {
            "sha256": hashlib.sha256(blob).hexdigest(),
            "bytes": len(blob),
            "rows": len(payloads[name]),
            "contains_gold": name == "gold.jsonl",
            "runtime_safe": name in {"runtime_questions.jsonl", "corpus.jsonl"},
        }
        for name, blob in blobs.items()
    }
    plan["runtime_input_allowlist"] = ["runtime_questions.jsonl", "corpus.jsonl"]
    plan["audit_only_notice"] = (
        "Do not expose manifest, gold or corpus_sources to methods: source ownership would reveal each question's supplied candidate pool."
    )
    counts = Counter(doc["title"] for doc in docs.values())
    plan["corpus_statistics"] = {
        "documents": len(docs),
        "unique_titles": len(counts),
        "titles_with_multiple_versions": sum(n > 1 for n in counts.values()),
    }
    plan["selected_normalized_question_sha256s"] = sorted(normalized)
    manifest = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"
    blobs["manifest.json"] = manifest
    digest = hashlib.sha256(manifest).hexdigest()
    blobs["manifest.sha256"] = f"{digest}  manifest.json\n".encode()
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, blob in blobs.items():
        with (output_dir / name).open("xb") as handle:
            handle.write(blob)
    return {
        "manifest_path": str(output_dir / "manifest.json"),
        "manifest_sha256": digest,
        "question_count": 32,
        "corpus_sources": 1024,
        "corpus_documents": len(docs),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("data/hotpotqa/shared_sep27_v1"))
    parser.add_argument("--prepare-new", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_shared_dev(args.root, args.output_dir, prepare_new=args.prepare_new)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
