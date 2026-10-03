"""Replay saved first repair queries on a sealed index, without any model calls.

中文：这不是重新盲测系统，也不生成答案。先核验旧预测，再实际重放共同初检与
首条已执行query；全部检索封存后，仅投影旧许可99题的标注，测证据覆盖。
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter

from .a0_v2_feedback import load_gold
from .operator_model import visible_evidence
from .protocol import Evidence
from .research_diagnostic import (
    FEEDBACK,
    FRESH,
    HISTORY,
    _inside,
    _read,
    _sha,
    _write,
    validate_rows,
)
from .score_operator_sources import _parse_gold, _support_coverage
from .shared_s2g_corpus import CorpusDocument, SharedBM25Index

OUTPUT = "runs/first_repair_replay_20261003_v1"
PROTOCOL = "knowledge/experiments/2026-10-03_共同初态首条修复_执行前协议.md"
MANIFEST = "data/hotpotqa/a0_wire_v2_20261003_v1/manifest.json"
METHODS = ("base", FRESH, HISTORY)


def first_pair(fresh_report, history_report, question_text):
    """Require a common initial state; take one actual query, not one whole macro.

    不能拿第二轮的query充当首轮修复，也不能拿被选中但未执行的卡充当检索。
    """
    searches = []
    for report in (fresh_report, history_report):
        if report["question"]["text"] != question_text:
            raise ValueError("report question differs")
        rows = report["episode"]["searches"]
        if (
            not rows
            or rows[0]["query"] != question_text
            or type(rows[0]["step"]) is not int
            or rows[0]["step"] != 0
        ):
            raise ValueError("initial search differs from original question")
        if len(rows) > 1 and (type(rows[1]["step"]) is not int or rows[1]["step"] != 1):
            raise ValueError("first repair is not from initial decision")
        for search in rows:
            ids = search["evidence_ids"]
            if len(ids) != len(set(ids)) or not 0 < len(ids) <= 6:
                raise ValueError("invalid saved retrieval IDs")
        if rows[0]["new_evidence_ids"] != rows[0]["evidence_ids"]:
            raise ValueError("initial new IDs differ from all initial IDs")
        searches.append(rows)
    a, b = (rows[0] for rows in searches)
    if a["evidence_ids"] != b["evidence_ids"]:
        raise ValueError("initial ranked IDs differ between arms")
    # Check original text as well as IDs; final pools still contain initial rows.
    pools = [
        {e["evidence_id"]: e for e in report["episode"]["evidence"]}
        for report in (fresh_report, history_report)
    ]
    if any(
        eid not in pools[0] or eid not in pools[1] or pools[0][eid] != pools[1][eid]
        for eid in a["evidence_ids"]
    ):
        raise ValueError("initial evidence content differs between arms")
    return {
        "initial": a,
        FRESH: searches[0][1] if len(searches[0]) > 1 else None,
        HISTORY: searches[1][1] if len(searches[1]) > 1 else None,
    }


def merge_documents(initial, extra):
    """Append unique exact-version paragraphs in original insertion order."""
    merged = {}
    for doc in (*initial, *extra):
        if doc.doc_id in merged and merged[doc.doc_id] != doc:
            raise ValueError("same document ID changed content")
        merged.setdefault(doc.doc_id, doc)
    return tuple(merged.values())


def coverage(documents, gold_docs, targets):
    """Raw paragraph coverage and complete visible-sentence coverage, no QA score."""
    if not targets:
        raise ValueError("empty support targets")
    evidence = tuple(Evidence(d.doc_id, d.title, 0, d.text) for d in documents)
    visible = visible_evidence(evidence)
    report = {
        "episode": {"evidence": [asdict(e) for e in evidence]},
        "reader": {
            "visible_evidence_ids": [r["evidence_id"] for r in visible],
            "evidence_windows": [{"evidence_id": r["evidence_id"], **r["window"]} for r in visible],
        },
    }
    raw, shown, issue = _support_coverage(report, gold_docs, targets)
    if issue is not None or shown is None:
        raise ValueError("invalid deterministic support window: " + str(issue))
    return {
        "raw_support_recall": raw,
        "visible_support_recall": shown,
        "support_total": len(targets),
        "raw_supported": round(raw * len(targets)),
        "visible_supported": round(shown * len(targets)),
    }


def load_inputs(project):
    """Verify historical seals before opening any old arm reports or annotations."""
    folder = project / FEEDBACK
    seal = _read(_inside(project, folder / "feedback_frozen.json"))
    required = {"audit.json", "missing_questions.json", "per_question.json", "SUMMARY.json"}
    if set(seal["files"]) != required:
        raise ValueError("feedback seal coverage differs")
    inputs = {}

    def register(path, expected=None):
        path = _inside(project, path)
        actual = _sha(path)
        if expected is not None and actual != expected:
            raise ValueError("input SHA differs: " + path.name)
        inputs[path.relative_to(project).as_posix()] = actual
        return path

    register(folder / "feedback_frozen.json")
    for name, digest in seal["files"].items():
        register(folder / name, digest)
    rows, summary, audit = (
        _read(folder / name) for name in ("per_question.json", "SUMMARY.json", "audit.json")
    )
    if (
        summary.get("prediction_modified") is not False
        or summary.get("memory_updated") is not False
    ):
        raise ValueError("feedback isolation not sealed")
    validate_rows(rows, summary)
    for relative, key in (
        ("runs/a0_wire_v2_freeze_v1.json", "external_freeze_sha256"),
        ("runs/a0_wire_v2_v1/TERMINAL.json", "external_terminal_sha256"),
    ):
        path = register(project / relative, summary[key])
        if audit["inputs"].get(str(path)) != summary[key]:
            raise ValueError("original audit anchor differs")
    freeze = _read(project / "runs/a0_wire_v2_freeze_v1.json")
    manifest = _read(register(project / MANIFEST, freeze["bundle_sha256"]))
    ids = [r["question_id"] for r in rows]
    if ids != [qid for qid in manifest["question_ids"] if qid in set(ids)]:
        raise ValueError("paired order differs from original manifest")
    register(project / PROTOCOL)
    # Snapshot every helper that determines retrieval/windows/scoring, not just this file.
    for relative in (
        "src/growrag/experiments/first_repair_replay.py",
        "src/growrag/experiments/research_diagnostic.py",
        "src/growrag/experiments/a0_v2_feedback.py",
        "src/growrag/experiments/score_operator_sources.py",
        "src/growrag/experiments/operator_model.py",
        "src/growrag/experiments/shared_s2g_corpus.py",
        "src/growrag/experiments/lexical_retriever.py",
        "src/growrag/experiments/protocol.py",
    ):
        register(project / relative)
    ref = manifest["corpus_ref"]
    register(project / ref["path"], ref["sha256"])
    register(project / ref["index_path"], ref["index_sha256"])
    register(project / (ref["index_path"] + ".sha256"))
    reports = {}
    for row in rows:
        qid = row["question_id"]
        reports[qid] = {}
        for method in (FRESH, HISTORY):
            matches = [Path(p) for p in audit["inputs"] if Path(p).name == f"{qid}_{method}.json"]
            if len(matches) != 1:
                raise ValueError("audited arm identity ambiguous")
            report = _read(register(matches[0], audit["inputs"][str(matches[0])]))
            if (
                report["question_id"] != qid
                or report["method"] != method
                or report["status"] != "completed"
                or report["gold_loaded"] is not False
                or report["memory_updated"] is not False
                or report["reader"]["answer"] != row["answers"][method]
                or [s["query"] for s in report["episode"]["searches"]]
                != row["usage"][method]["queries"]
            ):
                raise ValueError("audited report identity/isolation differs")
            reports[qid][method] = report
        first_pair(reports[qid][FRESH], reports[qid][HISTORY], row["question"])
    return rows, reports, manifest, inputs


def replay_search(index, search, report_pools):
    """Actually execute SQLite retrieval and require exact old ordering/content."""
    started = perf_counter()
    retrieved = index(search["query"], 6)
    # The callback returns AuthorDocument(text), while audit storage needs sentences.
    docs = tuple(index.document(d.doc_id) for d in retrieved)
    if any(d.text != old.text for d, old in zip(docs, retrieved, strict=True)):
        raise ValueError("retrieval callback/source paragraph differs")
    elapsed = perf_counter() - started
    if [d.doc_id for d in docs] != search["evidence_ids"]:
        raise ValueError("retrieval replay ordered IDs differ")
    for pool in report_pools:
        for doc in docs:
            old = pool.get(doc.doc_id)
            if old is None or old["title"] != doc.title or old["text"] != doc.text:
                raise ValueError("retrieval replay document content differs")
    return docs, elapsed


def replay_case(index, row, reports):
    pair = first_pair(reports[FRESH], reports[HISTORY], row["question"])
    a, f, h = (pair[m] for m in ("initial", FRESH, HISTORY))
    pools = {
        m: {e["evidence_id"]: e for e in reports[m]["episode"]["evidence"]}
        for m in (FRESH, HISTORY)
    }
    initial, elapsed = replay_search(index, a, list(pools.values()))
    result = {
        "question_id": row["question_id"],
        "question": row["question"],
        "initial": {"query": a["query"], "documents": [asdict(d) for d in initial]},
        "methods": {},
        "local_retrieval_calls": 1,
        "local_retrieval_seconds": elapsed,
    }
    for method, search in ((FRESH, f), (HISTORY, h)):
        docs, elapsed = replay_search(index, search, [pools[method]]) if search else ((), 0.0)
        merged = merge_documents(initial, docs)
        expected_new = [d.doc_id for d in docs if d.doc_id not in {d.doc_id for d in initial}]
        if search and expected_new != search["new_evidence_ids"]:
            raise ValueError("saved first-repair new IDs differ")
        result["methods"][method] = {
            "action_executed": search is not None,
            "query": search["query"] if search else None,
            "documents": [asdict(d) for d in merged],
            "repair_documents": [asdict(d) for d in docs],
            "repair_new_document_count": len(expected_new),
            "repair_returned_document_count": len(docs),
            "omitted_later_query_count": max(0, len(reports[method]["episode"]["searches"]) - 2),
            "source_stop_reason": reports[method]["episode"]["stop_reason"],
            "source_proposals": reports[method]["episode"]["proposals"][:1],
            "replay_seconds": elapsed,
        }
        result["local_retrieval_calls"] += int(search is not None)
        result["local_retrieval_seconds"] += elapsed
    return result


def _documents(rows):
    return tuple(CorpusDocument(r["doc_id"], r["title"], tuple(r["sentences"])) for r in rows)


def score_case(replay, gold, old_feedback):
    _, gold_docs, targets = _parse_gold(gold)
    initial = _documents(replay["initial"]["documents"])
    base = coverage(initial, gold_docs, targets)
    for metric in ("raw_support_recall", "visible_support_recall"):
        if base[metric] != old_feedback["raw_scores"]["base"][metric]:
            raise ValueError("replayed initial coverage differs from sealed BASE")
    methods = {"base": {"coverage": base, "action_executed": False}}
    for method in (FRESH, HISTORY):
        item = replay["methods"][method]
        cov = coverage(_documents(item["documents"]), gold_docs, targets)
        if cov["raw_supported"] < base["raw_supported"]:
            raise ValueError("append-only raw coverage decreased")
        if item["omitted_later_query_count"] == 0 and any(
            cov[metric] != old_feedback["raw_scores"][method][metric]
            for metric in ("raw_support_recall", "visible_support_recall")
        ):
            raise ValueError("unchanged whole-pool coverage differs from sealed feedback")
        methods[method] = {
            k: v for k, v in item.items() if k not in {"documents", "repair_documents"}
        }
        methods[method].update(
            coverage=cov,
            repair_only_coverage=(
                coverage(_documents(item["repair_documents"]), gold_docs, targets)
                if item["action_executed"]
                else None
            ),
            # Historical whole-loop evidence results, NOT new one-step answer scores.
            legacy_full_coverage={
                k: old_feedback["raw_scores"][method][k]
                for k in ("raw_support_recall", "visible_support_recall")
            },
        )
    return {
        "question_id": replay["question_id"],
        "question": replay["question"],
        "methods": methods,
        "support_targets": [
            {"doc_id": eid, "title": gold_docs[eid][0], "sentence_index": index}
            for eid, index in sorted(targets)
        ],
    }


def summarize(rows, replay_rows):
    result = {
        "protocol": "growrag-fixed-state-first-query-replay-v1",
        "n": len(rows),
        "api_calls": 0,
        "new_gold_opened": False,
        "new_answers_generated": False,
        "memory_updated": False,
        "local_retrieval_calls": sum(r["local_retrieval_calls"] for r in replay_rows),
        "local_retrieval_seconds": sum(r["local_retrieval_seconds"] for r in replay_rows),
        "methods": {},
        "history_minus_fresh": {},
        "action_cells": {
            "neither": sum(
                not r["methods"][FRESH]["action_executed"]
                and not r["methods"][HISTORY]["action_executed"]
                for r in rows
            ),
            "fresh_only": sum(
                r["methods"][FRESH]["action_executed"]
                and not r["methods"][HISTORY]["action_executed"]
                for r in rows
            ),
            "history_only": sum(
                not r["methods"][FRESH]["action_executed"]
                and r["methods"][HISTORY]["action_executed"]
                for r in rows
            ),
            "both": sum(
                r["methods"][FRESH]["action_executed"] and r["methods"][HISTORY]["action_executed"]
                for r in rows
            ),
        },
        "notice": "Retrospective development diagnosis of saved queries, not a blind strategy "
        "comparison or new answer EM/F1. Raw union recall cannot measure answer harm.",
    }
    for method in METHODS:
        items = [r["methods"][method] for r in rows]
        stats = {
            "action_executed_n": sum(i["action_executed"] for i in items),
            "raw_support_recall": mean(i["coverage"]["raw_support_recall"] for i in items),
            "visible_support_recall": mean(i["coverage"]["visible_support_recall"] for i in items),
            "raw_full_support_n": sum(i["coverage"]["raw_support_recall"] == 1 for i in items),
            "visible_full_support_n": sum(
                i["coverage"]["visible_support_recall"] == 1 for i in items
            ),
        }
        if method != "base":
            stats.update(
                more_raw_support_than_base_n=sum(
                    i["coverage"]["raw_support_recall"]
                    > r["methods"]["base"]["coverage"]["raw_support_recall"]
                    for r, i in zip(rows, items, strict=True)
                ),
                less_visible_support_than_base_n=sum(
                    i["coverage"]["visible_support_recall"]
                    < r["methods"]["base"]["coverage"]["visible_support_recall"]
                    for r, i in zip(rows, items, strict=True)
                ),
                executed_but_no_extra_raw_support_n=sum(
                    i["action_executed"]
                    and i["coverage"]["raw_support_recall"]
                    == r["methods"]["base"]["coverage"]["raw_support_recall"]
                    for r, i in zip(rows, items, strict=True)
                ),
                executed_at_full_raw_baseline_n=sum(
                    i["action_executed"]
                    and r["methods"]["base"]["coverage"]["raw_support_recall"] == 1
                    for r, i in zip(rows, items, strict=True)
                ),
                repair_only_raw_support_recall_executed=(
                    mean(
                        i["repair_only_coverage"]["raw_support_recall"]
                        for i in items
                        if i["action_executed"]
                    )
                    if any(i["action_executed"] for i in items)
                    else None
                ),
                later_queries_omitted=sum(i["omitted_later_query_count"] for i in items),
                full_loop_more_raw_support_n=sum(
                    i["legacy_full_coverage"]["raw_support_recall"]
                    > i["coverage"]["raw_support_recall"]
                    for i in items
                ),
                new_documents=sum(i["repair_new_document_count"] for i in items),
                returned_documents=sum(i["repair_returned_document_count"] for i in items),
            )
        result["methods"][method] = stats
    for metric in ("raw_support_recall", "visible_support_recall"):
        deltas = [
            r["methods"][HISTORY]["coverage"][metric] - r["methods"][FRESH]["coverage"][metric]
            for r in rows
        ]
        result["history_minus_fresh"][metric] = {
            "positive_n": sum(d > 0 for d in deltas),
            "negative_n": sum(d < 0 for d in deltas),
            "tied_n": sum(d == 0 for d in deltas),
            "mean_gain": mean(max(d, 0) for d in deltas),
            "mean_loss": mean(max(-d, 0) for d in deltas),
            "net": mean(deltas),
            "positive_ids": [r["question_id"] for r, d in zip(rows, deltas, strict=True) if d > 0],
            "negative_ids": [r["question_id"] for r, d in zip(rows, deltas, strict=True) if d < 0],
        }
    both = [r for r in rows if all(r["methods"][m]["action_executed"] for m in (FRESH, HISTORY))]
    result["both_executed_description_only"] = {
        "n": len(both),
        "mean_raw_difference": mean(
            r["methods"][HISTORY]["coverage"]["raw_support_recall"]
            - r["methods"][FRESH]["coverage"]["raw_support_recall"]
            for r in both
        )
        if both
        else None,
        "notice": "Selection-biased small subset, not a separate unbiased test.",
    }
    return result


def run(project, output=OUTPUT):
    project = Path(project).resolve(strict=True)
    target = (project / output).resolve()
    if target.parent != project / "runs":
        raise ValueError("replay output must be a direct runs child")
    if target.exists():
        raise FileExistsError("replay exists; never overwrite")
    rows, reports, manifest, inputs = load_inputs(project)
    # load_inputs requires the pre-existing index and SHA. Constructor cannot create a new one.
    ref = manifest["corpus_ref"]
    index = SharedBM25Index(
        project / ref["index_path"], project / ref["path"], ref["sha256"], ref["rows"]
    )
    target.mkdir(exist_ok=False)
    stage, replay_rows = "retrieval", []
    try:
        with (target / "REPLAY.jsonl").open("x", encoding="utf-8") as handle:
            for n, row in enumerate(rows, 1):
                replay = replay_case(index, row, reports[row["question_id"]])
                replay_rows.append(replay)
                handle.write(json.dumps(replay, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                print(f"replayed {n}/{len(rows)} {row['question_id']}", flush=True)
        stage = "replay_seal"
        _write(
            target / "replay_frozen.json",
            {"files": {"REPLAY.jsonl": _sha(target / "REPLAY.jsonl")}},
        )
        if any(_sha(project / p) != digest for p, digest in inputs.items()):
            raise ValueError("input changed during retrieval replay")
        stage = "already_opened_gold_projection"
        gold, gold_inputs = load_gold(project, manifest, [r["question_id"] for r in rows])
        audit_inputs = _read(project / FEEDBACK / "audit.json")["inputs"]
        for path, digest in gold_inputs.items():
            if audit_inputs.get(path) != digest:
                raise ValueError("gold input not in successful old audit")
            inputs[Path(path).relative_to(project).as_posix()] = digest
        stage = "evidence_scoring"
        scored = [
            score_case(replay, gold[row["question_id"]], row)
            for replay, row in zip(replay_rows, rows, strict=True)
        ]
        summary = summarize(scored, replay_rows)
        summary["environment"] = {"python": sys.version, "platform": platform.platform()}
        summary["gold_ids"] = [r["question_id"] for r in rows]
        _write(target / "per_question.json", scored)
        _write(target / "SUMMARY.json", summary)
        _write(target / "INPUTS.json", inputs)
        fields = [
            "question_id",
            "question",
            "method",
            "action_executed",
            "query",
            "raw_support_recall",
            "visible_support_recall",
            "repair_new_document_count",
            "omitted_later_query_count",
        ]
        with (target / "per_question.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in scored:
                for method, item in row["methods"].items():
                    writer.writerow(
                        {
                            "question_id": row["question_id"],
                            "question": row["question"],
                            "method": method,
                            "action_executed": item["action_executed"],
                            "query": item.get("query"),
                            **{
                                k: item["coverage"][k]
                                for k in ("raw_support_recall", "visible_support_recall")
                            },
                            **{k: item.get(k, 0) for k in fields[-2:]},
                        }
                    )
        if any(_sha(project / p) != digest for p, digest in inputs.items()):
            raise ValueError("input changed during scoring")
        _write(target / "TERMINAL.json", {"status": "completed", "n": len(rows), "api_calls": 0})
        _write(target / "frozen.json", {"files": {p.name: _sha(p) for p in target.iterdir()}})
        return summary
    except BaseException as error:
        _write(
            target / "FAILED.json",
            {
                "status": "failed",
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        index.close()


def verify(project, output=OUTPUT):
    """Read-only check of the new artifact and source hashes, without re-reading gold."""
    project = Path(project).resolve(strict=True)
    target = _inside(project, project / output)
    if target.parent != project / "runs":
        raise ValueError("replay output must be a direct runs child")
    seal = _read(target / "frozen.json")
    required = {
        "REPLAY.jsonl",
        "replay_frozen.json",
        "per_question.json",
        "SUMMARY.json",
        "INPUTS.json",
        "per_question.csv",
        "TERMINAL.json",
    }
    if set(seal["files"]) != required:
        raise ValueError("replay seal coverage differs")
    for name, digest in seal["files"].items():
        if Path(name).name != name or _sha(_inside(project, target / name)) != digest:
            raise ValueError("replay artifact drift")
    for relative, digest in _read(target / "INPUTS.json").items():
        if _sha(_inside(project, project / relative)) != digest:
            raise ValueError("replay input drift")
    summary = _read(target / "SUMMARY.json")
    records = _read(target / "per_question.json")
    replay_rows = [
        json.loads(line)
        for line in (target / "REPLAY.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected = summarize(records, replay_rows)
    if any(summary.get(k) != v for k, v in expected.items()):
        raise ValueError("replay summary differs from per-question data")
    if summary["gold_ids"] != [r["question_id"] for r in records] or (
        summary["gold_ids"] != [r["question_id"] for r in replay_rows]
    ):
        raise ValueError("replay paired IDs differ")
    return {
        "status": "verified",
        "n": len(records),
        "api_calls": 0,
        "local_retrieval_calls": summary["local_retrieval_calls"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            (verify if args.verify else run)(args.project_root, args.output),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
