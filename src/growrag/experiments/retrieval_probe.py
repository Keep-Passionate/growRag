"""Posthoc local BM25 query ablations; no reader, model client, or API call.

Identical retrieved evidence does not establish identical generated answers.
Manual candidates are diagnostic, not preregistered method improvements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path

from .hotpot import HotpotExample
from .lexical_retriever import BM25SentenceRetriever
from .pre_manifest import load_pre_examples


def probe_queries(example: HotpotExample, queries: Mapping[str, str], top_k: int = 4) -> dict:
    """Rank all named queries against the same corpus; inspect gold only afterward."""
    if not isinstance(example, HotpotExample):
        raise TypeError("example must be a HotpotExample")
    if not isinstance(queries, Mapping) or not queries:
        raise ValueError("queries must be a nonempty uniquely named mapping")
    names = list(queries)
    if any(not isinstance(name, str) or not name.strip() or name != name.strip() for name in names):
        raise ValueError("query names must be nonempty, without surrounding whitespace")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("query names must be unique, including case variants")
    if any(not isinstance(query, str) or not query.strip() for query in queries.values()):
        raise ValueError("each query must be nonempty text")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if example.gold is not None and example.gold.question_id != example.question.question_id:
        raise ValueError("gold belongs to a different question")
    index = BM25SentenceRetriever(example.candidate_context)
    retrieved = {name: index.retrieve(query, top_k=top_k).value for name, query in queries.items()}
    gold = set(example.gold.supporting_facts) if example.gold else set()
    rows = {}
    for name, evidence in retrieved.items():
        refs = [(item.title, item.sentence_id) for item in evidence]
        rows[name] = {
            "query": queries[name],
            "evidence_ids": [item.evidence_id for item in evidence],
            "sentence_refs": refs,
            "gold_support_recall": len(set(refs) & gold) / len(gold) if gold else None,
        }
    return {
        "schema_version": "growrag-retrieval-probe-v1",
        "diagnostic_stage": "posthoc",
        "question_id": example.question.question_id,
        "original_question": example.question.text,
        "top_k": top_k,
        "retriever": "growrag-title-sentence-bm25-v1",
        "corpus_fingerprint": index.corpus_fingerprint,
        "context_fingerprint": index.context_fingerprint,
        "transport_source": "local_compute",
        "reader_calls": 0,
        "api_requests": 0,
        "answer_effect": "unknown",
        "gold_support_available": bool(gold),
        "queries": rows,
        "pairs": [
            {
                "left": left,
                "right": right,
                "same_order": rows[left]["evidence_ids"] == rows[right]["evidence_ids"],
                "same_set": set(rows[left]["evidence_ids"]) == set(rows[right]["evidence_ids"]),
            }
            for left, right in combinations(names, 2)
        ],
        "notice": "Actual local retrieval only; no reader or API. Posthoc evidence agreement "
        "is not an answer-effect estimate, causal finding, or held-out benchmark result.",
    }


def _stored_queries(record: dict, example: HotpotExample) -> tuple[dict, dict]:
    """Require completed independent source arms; do not interpret answer text."""
    queries, expected = {}, {}
    try:
        if record["schema_version"] != "pre_source_pair.v1" or (
            record["question"]["question_id"] != example.question.question_id
            or record["question"]["text"] != example.question.text
        ):
            raise ValueError("source record question/schema differs from the fixed source")
        for action, field in (("BASE", "base"), ("FRESH", "fresh")):
            arm = record[field]
            if arm["state"]["question"] != record["question"]:
                raise ValueError("archived arm belongs to a different source question")
            rounds, events = arm["state"]["rounds"], arm["events"]
            if len(rounds) != 1 or not events or any(e["status"] != "ok" for e in events):
                raise ValueError("archived source arm must be completed and error-free")
            row = rounds[0]
            if row["decision"]["memory"] is not None or row["decision"]["action"] not in {
                action,
                "BASE",
            }:
                raise ValueError("archived source arm is not an independent BASE/FRESH arm")
            queries[action] = row["search_query"]
            expected[action] = [item["evidence_id"] for item in row["reply"]["evidence"]]
        if queries["BASE"] != example.question.text:
            raise ValueError("archived BASE must use the original query")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("invalid archived source record structure") from error
    return queries, expected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--candidate", nargs=2, action="append", default=[], metavar=("NAME", "QUERY")
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("output already exists; no overwrite")
    sources, _, manifest = load_pre_examples(args.manifest)
    raw = args.source_record.read_bytes()
    record = json.loads(raw)
    question_id = (
        record.get("question", {}).get("question_id") if isinstance(record, dict) else None
    )
    example = next((item for item in sources if item.question.question_id == question_id), None)
    if example is None:
        raise ValueError("source must belong to the fixed 32 selected memory_seed questions")
    queries, expected = _stored_queries(record, example)
    origins = {name: "original_batch_frozen_arm" for name in queries}
    for name, query in args.candidate:
        if name.casefold() in {key.casefold() for key in queries}:
            raise ValueError("candidate names must be unique and cannot replace BASE/FRESH")
        queries[name], origins[name] = query, "manual_posthoc"
    report = probe_queries(example, queries, args.top_k)
    for action, evidence_ids in expected.items():
        if report["queries"][action]["evidence_ids"] != evidence_ids:
            raise ValueError(f"recomputed {action} evidence order differs from the source archive")
    report.update(
        query_origins=origins,
        original_arm_replay_verified=True,
        selected_source_role="memory_seed",
        source_record_sha256=hashlib.sha256(raw).hexdigest(),
        data_sha256=manifest["data_sha256"],
        original_manifest_sha256=manifest["source_manifest_sha256"],
        selection_notice="Fixed first 32 unique memory_seed questions; the inspected source "
        "and manual ablations are posthoc. Original batch arms retain their frozen identities.",
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(f"Local-only retrieval probe saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
