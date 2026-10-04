"""Read-only replay of exported OLD S2G development traces, without model or gold.

中文：只检查当时的查询、检索排序、选句指针和下一轮判决前证据能否还原。
这不重新调用 Judge/Extractor，也不评价答案，更不能证明某个改写更好。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import sqlite3
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = {
    "src/growrag/experiments/shared_s2g_corpus.py": (
        "SharedBM25Index",
        "CorpusDocument",
        "_configure",
        "_canonical",
        "_sha",
        "_document",
        "_json",
        "_unique",
        "INDEX_VERSION",
    ),
    "src/growrag/experiments/lexical_retriever.py": (
        "_tokens",
        "_finite_number",
        "_TOKEN_PATTERN",
    ),
    "src/growrag/experiments/s2g_author_api.py": ("AuthorDocument",),
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def _definition(tree, name):
    for node in tree.body:
        if getattr(node, "name", None) == name:
            return node
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node
    raise ValueError(f"missing reviewed definition: {name}")


def verify_source(snapshot, launch, root=ROOT):
    """Require frozen text integrity and matching retrieval definitions/constants.

    Formatting can change file bytes; AST equality includes defaults and globals.
    It does not claim identical Python/SQLite runtime versions.
    """
    actual = hashlib.sha256(
        json.dumps(
            snapshot["files"],
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    if actual != snapshot["sha256"] or actual != launch["source_sha256"]:
        raise ValueError("frozen source aggregate mismatch")
    audited = {}
    for relative, names in DEPENDENCIES.items():
        item = snapshot["files"][relative]
        if hashlib.sha256(item["text"].encode()).hexdigest() != item["sha256"]:
            raise ValueError(f"source text hash mismatch: {relative}")
        old = ast.parse(item["text"])
        current = ast.parse((root / relative).read_bytes())
        for name in names:
            if ast.dump(_definition(old, name)) != ast.dump(_definition(current, name)):
                raise ValueError(f"retrieval semantics differ: {relative}:{name}")
        # Imports provide the semantic environment of those definitions.
        old_imports = [ast.dump(n) for n in old.body if isinstance(n, ast.Import | ast.ImportFrom)]
        new_imports = [
            ast.dump(n) for n in current.body if isinstance(n, ast.Import | ast.ImportFrom)
        ]
        if old_imports != new_imports:
            raise ValueError(f"retrieval imports differ: {relative}")
        audited[relative] = {
            "frozen_sha256": item["sha256"],
            "current_sha256": digest(root / relative),
            "ast_equal_definitions": list(names),
            "imports_equal": True,
        }
    return audited


def replay_trace(trace, index, scope):
    """Compare observed sequence, including evidence BEFORE the next Judge.

    The top-50 is recomputed here. The original event stores the post-exclusion
    top-six, so only that list is compared against the historical observation.
    """
    question = trace["question"]
    previous, context, pending_query, last_documents = [], "", None, None
    current_gaps = []
    rows, checks = [], []
    for event in trace["events"]:
        kind = event["kind"]
        position = event.get("event_index")
        if "prior_retrieved_doc_ids" in event:
            if event["prior_retrieved_doc_ids"] != previous:
                raise ValueError(f"prior document IDs differ at event {position}")
        if kind == "judge":
            if event["evidence_contexts"] != [context]:
                raise ValueError(f"pre-Judge evidence differs at event {position}")
            current_gaps = event["gap_items"][0]
            if not previous and not context and event["verdicts"] == [True]:
                current_gaps = []  # Author's empty-evidence first-retrieval guard.
            checks.append({"kind": "pre_judge_context", "event_index": position, "match": True})
        elif kind == "query":
            if event["gap_items"] != current_gaps:
                raise ValueError(f"query gap differs from preceding Judge at event {position}")
            if event["gap_profile"] != "paper_k1":
                raise ValueError("only frozen shared500 paper_k1 profile is reviewed")
            actual = scope["build_query_from_missing"](
                question,
                current_gaps,
                max_facts=1,
                dataset_name="hotpotqa",
            )
            if actual != event["query"]:
                raise ValueError(f"exact query mismatch at event {position}")
            pending_query = actual
            checks.append({"kind": "query", "event_index": position, "match": True})
        elif kind == "retrieval":
            if pending_query is None or event["queries"] != [pending_query]:
                raise ValueError(f"retrieval query sequence mismatch at event {position}")
            raw = index.search_with_scores(pending_query, 50)
            filtered = [doc for doc, _ in raw if doc.doc_id not in set(previous)][:6]
            if event["documents"] != [[asdict(doc) for doc in filtered]]:
                raise ValueError(f"ordered retrieval IDs/title/text mismatch at event {position}")
            rows.append(
                {
                    "round": event["round"],
                    "query": pending_query,
                    "prior_doc_ids": list(previous),
                    "recomputed_raw_top50_ids": [doc.doc_id for doc, _ in raw],
                    "observed_postfilter_ids": [doc.doc_id for doc in filtered],
                    "ordered_ids_title_text_match": True,
                }
            )
            last_documents = filtered
            previous.extend(doc.doc_id for doc in filtered if doc.doc_id not in previous)
            pending_query = None
        elif kind == "extraction":
            if last_documents is None:
                raise ValueError("extraction has no preceding retrieval")
            pointers = event["pointers"]
            if not last_documents:
                # Author retains one "No results found." placeholder document.
                if pointers not in ([[]], [[[]]]) or event["sources"]:
                    raise ValueError("empty retrieval has nonempty evidence")
                checks.append(
                    {
                        "kind": "extraction_pointers",
                        "event_index": position,
                        "match": True,
                    }
                )
                continue
            if len(pointers) != 1 or len(pointers[0]) != len(last_documents):
                raise ValueError("extraction pointer dimensions differ")
            sources = []
            for doc, ids in zip(last_documents, pointers[0], strict=True):
                sentences = scope["split_wiki_sentences"](doc.text)
                for sid in ids:
                    if type(sid) is not int or not 1 <= sid <= len(sentences):
                        raise ValueError("invalid sentence pointer")
                    sources.append(
                        {
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "sentence_id": sid,
                            "text": sentences[sid - 1],
                        }
                    )
            if sources != event["sources"]:
                raise ValueError("selected sentence pointer/text differs")
            merged = scope["merge_evidence_only"](
                [doc.title for doc in last_documents],
                [doc.text for doc in last_documents],
                pointers[0],
            )
            context = scope["append_evidence_context"](context, merged)
            checks.append({"kind": "extraction_pointers", "event_index": position, "match": True})
    return {
        "question_id": trace["question_id"],
        "run_id": trace["run_id"],
        "manifest_index": trace["manifest_index"],
        "retrievals": rows,
        "checks": checks,
    }


def description_probe(scope):
    """Constructed engineering example, explicitly NOT a HotpotQA score."""
    question = "Which company purchased ExampleWorks?"
    gap = {"category": "relation", "target": "ExampleWorks", "slot": "purchaser"}
    first, second = (
        {**gap, "description": "Find the purchaser."},
        {
            **gap,
            "description": "Find its fictional mascot instead.",
        },
    )
    queries = [
        scope["build_query_from_missing"](question, [g], max_facts=1) for g in (first, second)
    ]
    prompts = []

    def capture(**kwargs):
        prompts.extend(kwargs["task_contents"])
        return ['{"evidence_global_ids": []}']

    scope["call_reasoner_batch"] = capture
    for g in (first, second):
        scope["concat_and_pick_sentences_batch"](
            questions=[question],
            titles_batch=[["ExampleWorks"]],
            texts_batch=[["ExampleWorks was purchased by SampleCo."]],
            missing_facts_batch=[[g]],
            dataset_name="hotpotqa",
        )
    return {
        "material": "constructed_engineering_example_not_QA_result",
        "query_unchanged": queries[0] == queries[1],
        "selector_user_prompt_unchanged": prompts[0] == prompts[1],
        "scope": "target and slot both nonempty; description fallback is a different case",
        "api_calls": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("preserve previous replay output")
    package = read_json(args.package / "package_manifest.json")
    traces_path = args.package / "selected_traces.jsonl"
    expected = package["artifacts"]["selected_traces.jsonl"]["sha256"]
    if digest(traces_path) != expected:
        raise ValueError("exported trace hash mismatch")
    traces = [json.loads(line) for line in traces_path.read_bytes().splitlines()]
    if not traces or len({t["question_id"] for t in traces}) > 10:
        raise ValueError("bounded first-ten handoff required")
    manifest_info = package["resources"]["manifest"]
    if digest(manifest_info["path"]) != manifest_info["sha256"]:
        raise ValueError("dataset manifest hash mismatch")
    manifest = read_json(manifest_info["path"])
    if (manifest["official_split"], manifest["role"]) != ("train", "development"):
        raise ValueError("only old train/development material is permitted")
    selected = manifest["question_ids"][: package["sample_count"]]
    if package["selected_question_ids"] != selected:
        raise ValueError("sample is not the fixed manifest prefix")
    if list(dict.fromkeys(t["question_id"] for t in traces)) != selected:
        raise ValueError("trace membership/order differs from fixed sample")
    for trace in traces:
        if selected[trace["manifest_index"]] != trace["question_id"]:
            raise ValueError("trace position differs from manifest")
    batches = {batch["run_id"]: batch for batch in package["batches"]}
    launches, verified = {}, {}
    for run_id in dict.fromkeys(t["run_id"] for t in traces):
        batch = batches[run_id]
        for name, sha_key in (
            ("launch_plan", "launch_plan_sha256"),
            ("source_snapshot", "source_snapshot_file_sha256"),
        ):
            if digest(batch["paths"][name]) != batch[sha_key]:
                raise ValueError(f"handoff source file hash mismatch: {name}")
        launch = read_json(batch["paths"]["launch_plan"])
        launches[run_id] = launch
        verified[run_id] = verify_source(read_json(batch["paths"]["source_snapshot"]), launch)
        if (launch["max_retrieval_rounds"], launch["top_docs"], launch["gap_profile"]) != (
            4,
            6,
            "paper_k1",
        ):
            raise ValueError("unreviewed retrieval configuration")
    # Import only after confirming exact reviewed definitions and their constants.
    from growrag.experiments.s2g_author_api import load_author_scope
    from growrag.experiments.shared_s2g_corpus import SharedBM25Index

    scope = load_author_scope(args.upstream)
    first_launch = next(iter(launches.values()))
    meta = first_launch["runtime_metadata"]
    corpus = Path(first_launch["manifest_path"]).parent / "corpus.jsonl"
    index_path = Path(first_launch["index_path"])
    if (
        not corpus.is_file()
        or not index_path.is_file()
        or not index_path.with_suffix(
            index_path.suffix + ".sha256",
        ).is_file()
    ):
        raise FileNotFoundError("existing corpus/index/sidecar required; rebuild forbidden")
    if digest(corpus) != meta["corpus_sha256"]:
        raise ValueError("corpus hash mismatch")
    index_hash = digest(index_path)
    if index_hash != package["resources"]["index"]["sha256"]:
        raise ValueError("index differs from exported package")
    for launch in launches.values():
        if launch["runtime_metadata"] != meta or launch["index_path"] != str(index_path):
            raise ValueError("selected traces span different retrieval environments")
    index = SharedBM25Index(index_path, corpus, meta["corpus_sha256"], meta["document_count"])
    try:
        if index.context_fingerprint != meta["context_fingerprint"]:
            raise ValueError("retrieval context fingerprint mismatch")
        results = [replay_trace(trace, index, scope) for trace in traces]
    finally:
        index.close()
    if digest(index_path) != index_hash:
        raise ValueError("index changed during read-only replay")
    report = {
        "schema": "growrag-s2g-handoff-replay-v1",
        "status": "replay_verified",
        "api_calls": 0,
        "gold_read": False,
        "training": False,
        "package_manifest_sha256": digest(args.package / "package_manifest.json"),
        "selected_traces_sha256": digest(traces_path),
        "index_sha256": index_hash,
        "script_sha256": digest(Path(__file__)),
        "source_checks": verified,
        "python_version": platform.python_version(),
        "sqlite_version": sqlite3.sqlite_version,
        "description_probe": description_probe(scope),
        "results": results,
        "limits": [
            "Judge and Extractor model decisions are replayed from observed outputs, not rerun.",
            "Raw top50 is reconstructed; only postfilter top6 was in original retrieval events.",
            "Matching traces establish reproducibility, not semantic correctness or improvement.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "questions": len({t["question_id"] for t in traces}),
                "attempts": len(traces),
                "retrievals": sum(len(r["retrievals"]) for r in results),
                "api_calls": 0,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
