"""Exploratory post-execution query-vs-pagination diagnostic; no model/API calls.

Same observed past-doc state, same top50/exclude-seen/top6 author retrieval.
Gold is loaded only after retrieval replays and never chooses a query or state.
This measures raw annotated support ARRIVAL, not retained evidence or answer gain.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import multiprocessing
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from growrag.experiments.s2g_author_api import PINNED_HASHES, _CallbackSearcher, load_author_scope
from growrag.experiments.shared_s2g_corpus import load_gold_after_execution, load_runtime

MANIFEST_SHA = "2fb117c03b64be3ab12398a297c82ce3315600c14e33f5dcd56c5c6c5c59e064"
ARMS = ("BASE1_AUTHOR_READER", "S2G_AUTHOR_API4")
_WORKER_RUNTIME = None
_WORKER_SCOPE = None


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class AuthorReplay:
    """Execute the pinned upstream function, not a copied S2G-like approximation."""

    def __init__(self, scope, index):
        self.scope, self.index = scope, index
        self.cache = {}
        self.search_count, self.search_seconds = 0, 0.0
        self.logical_requests, self.cache_hits = 0, 0
        scope["corpus"] = {}
        scope["searcher"] = _CallbackSearcher(self._search, scope["corpus"])

    def _search(self, query, k):
        self.logical_requests += 1
        key = (query, k)
        if key not in self.cache:
            started = time.perf_counter()
            self.cache[key] = tuple(self.index(query, k))
            self.search_seconds += time.perf_counter() - started
            self.search_count += 1
        else:
            self.cache_hits += 1
        return self.cache[key]

    def __call__(self, query, past_doc_ids):
        titles, texts, ids = self.scope["bm25_search_batch"](
            [query], [list(past_doc_ids)], k=6, remove_repeat_docs=True
        )
        return [
            {"doc_id": doc_id, "title": title, "text": text}
            for doc_id, title, text in zip(ids[0], titles[0], texts[0], strict=True)
            if doc_id
        ]


def replay_item(item, runtime, scope):
    """One task owns a fresh per-question cache; never shares SQLite across processes."""
    if item["context_fingerprint"] != runtime.metadata["context_fingerprint"]:
        raise ValueError("retrieval context/index changed")
    question = next(q.text for q in runtime.questions if q.question_id == item["question_id"])
    replay = AuthorReplay(scope, runtime.index)
    value = replay_question(question, item["events"], replay, runtime.index.document)
    if len(value["states"]) != item["reported_rounds"]:
        raise ValueError("logged retrieval rounds incomplete")
    return {
        "replay": {
            "question_id": item["question_id"],
            "run_id": item["run_id"],
            "offset": item["offset"],
            **value,
        },
        "index_calls": replay.search_count,
        "index_seconds": replay.search_seconds,
        "logical_requests": replay.logical_requests,
        "cache_hits": replay.cache_hits,
    }


def _initialize_worker(manifest, index, upstream):
    global _WORKER_RUNTIME, _WORKER_SCOPE
    # Each spawned process opens and verifies its own read-only connection.
    _WORKER_RUNTIME = load_runtime(
        manifest, index_path=index, expected_manifest_sha256=MANIFEST_SHA
    )
    atexit.register(_WORKER_RUNTIME.close)
    _WORKER_SCOPE = load_author_scope(Path(upstream))


def _worker_replay(item):
    if _WORKER_RUNTIME is None or _WORKER_SCOPE is None:
        raise RuntimeError("offline worker not initialized")
    return replay_item(item, _WORKER_RUNTIME, _WORKER_SCOPE)


def replay_question(question, events, retrieve, document):
    """No gold parameter exists: both choices use the exact same observed state."""
    past, states = [], []
    for event in events:
        if event.get("kind") != "retrieval":
            continue
        if (
            event.get("round") != len(states) + 1
            or len(event.get("queries", [])) != 1
            or len(event.get("documents", [])) != 1
        ):
            raise ValueError("retrieval event is not the reviewed single-question ordered format")
        observed = event["documents"][0]
        query = event["queries"][0]
        if not isinstance(query, str):
            raise ValueError("query must be logged text")
        version_valid = True
        for row in observed:
            try:
                raw = document(row["doc_id"])
                version_valid &= row["title"] == raw.title and row["text"] == raw.text
            except (KeyError, TypeError):
                version_valid = False
        actual_replay = retrieve(query, tuple(past))
        original_replay = retrieve(question, tuple(past))
        observed_ids = [r["doc_id"] for r in observed]
        actual_ids = [r["doc_id"] for r in actual_replay]
        match = version_valid and actual_ids == observed_ids and actual_replay == observed
        states.append(
            {
                "round": event["round"],
                "past_doc_ids": list(past),
                "executed_query": query,
                "original_question": question,
                "query_text_changed": query.strip() != question.strip(),
                "observed_doc_ids": observed_ids,
                "executed_replay_doc_ids": actual_ids,
                "original_replay_doc_ids": [r["doc_id"] for r in original_replay],
                "observed_exact_versions_valid": version_valid,
                "exact_replay_match": match,
                "executed_replay_documents": [
                    {"doc_id": r["doc_id"], "title": r["title"]} for r in actual_replay
                ],
                "original_replay_documents": [
                    {"doc_id": r["doc_id"], "title": r["title"]} for r in original_replay
                ],
            }
        )
        # Only actual, historically observed documents update the next state.
        for doc_id in observed_ids:
            if doc_id and doc_id != "No results found." and doc_id not in past:
                past.append(doc_id)
    return {
        "states": states,
        "all_replays_match": bool(states) and all(s["exact_replay_match"] for s in states),
    }


def exact_targets(gold, document):
    targets = set()
    for support in gold.exact_support:
        doc = document(support.doc_id)
        if (
            doc.title != support.title
            or support.sentence_index >= len(doc.sentences)
            or digest(doc.sentences[support.sentence_index].encode()) != support.text_sha256
        ):
            raise ValueError("gold sentence/version fingerprint mismatch")
        targets.add((support.doc_id, support.sentence_index))
    return targets


def score_question(replay, gold, document):
    if not replay["all_replays_match"]:
        return [
            {**state, "comparison_status": "uncomparable_replay_mismatch"}
            for state in replay["states"]
        ]
    if gold.annotation_status != "valid":
        return [
            {**state, "comparison_status": "unscorable_annotation"} for state in replay["states"]
        ]
    targets = exact_targets(gold, document)
    target_docs = {doc_id for doc_id, _ in targets}
    rows = []
    for state in replay["states"]:
        past = set(state["past_doc_ids"])
        previous = {ref for ref in targets if ref[0] in past}
        previous_docs = target_docs & past
        choices = {}
        for name, field in (
            ("executed", "executed_replay_doc_ids"),
            ("original", "original_replay_doc_ids"),
        ):
            arrived = set(state[field]) - past
            new = {ref for ref in targets if ref[0] in arrived}
            new_docs = target_docs & arrived
            choices[name] = {
                "new_gold_support_sentences": sorted(new),
                "new_gold_support_documents": sorted(new_docs),
                "new_sentence_count": len(new),
                "new_document_count": len(new_docs),
                "new_sentence_coverage": len(new) / len(targets),
                "new_document_coverage": len(new_docs) / len(target_docs),
                "missing_gold_sentences_after": len(targets - previous - new),
                "missing_gold_documents_after": len(target_docs - previous_docs - new_docs),
            }
        delta = (
            choices["executed"]["new_sentence_count"] - choices["original"]["new_sentence_count"]
        )
        document_delta = (
            choices["executed"]["new_document_count"] - choices["original"]["new_document_count"]
        )
        rows.append(
            {
                **state,
                "comparison_status": "comparable",
                "gold_sentence_count": len(targets),
                "gold_document_count": len(target_docs),
                "missing_gold_sentences_before": len(targets - previous),
                "missing_gold_documents_before": len(target_docs - previous_docs),
                "choices": choices,
                "new_sentence_count_delta": delta,
                "new_document_count_delta": document_delta,
                "new_sentence_coverage_delta": delta / len(targets),
                "new_document_coverage_delta": document_delta / len(target_docs),
                "sentence_outcome": "positive"
                if delta > 0
                else "negative"
                if delta < 0
                else "zero",
                "document_outcome": "positive"
                if document_delta > 0
                else "negative"
                if document_delta < 0
                else "zero",
            }
        )
    return rows


def closed_predictions(runs_root, limit):
    rows, excluded, hashes, seen = [], [], {}, {}
    for directory in sorted(runs_root.glob("*_s2g_shared500_v1_*")):
        # Authorization *.claim.json files share this prefix but are not batches.
        if not directory.is_dir():
            continue
        paths = [
            directory / name
            for name in ("launch_plan.json", "reports.json", "final_budget.json", "events.jsonl")
        ]
        if not all(p.is_file() for p in paths):
            excluded.append({"run_id": directory.name, "reason": "not_closed"})
            continue
        events = [json.loads(line) for line in paths[-1].read_bytes().splitlines() if line.strip()]
        if not events or events[-1].get("kind") != "exit":
            excluded.append({"run_id": directory.name, "reason": "not_terminal"})
            continue
        launch = json.loads(paths[0].read_bytes())
        if (
            launch["manifest_sha256"] != MANIFEST_SHA
            or launch["series"] != "500_v1"
            or launch["top_docs"] != 6
            or launch["max_retrieval_rounds"] != 4
        ):
            raise ValueError("wrong frozen experiment scope")
        # Reports contain offline gold/feedback. Check existence, but never read
        # them here; runtime execution files are frozen before gold scoring.
        for path in (paths[0], paths[2], paths[3]):
            hashes[str(path)] = digest(path.read_bytes())
        for completion in (e for e in events if e.get("kind") == "question_complete"):
            qid = completion["question_id"]
            if qid in seen:
                raise ValueError("duplicate question completion event")
            if qid not in launch["question_ids"] or completion.get("status") not in {
                "completed",
                "failed",
            }:
                raise ValueError("question completion does not match its frozen launch")
            offset = launch["start"] + launch["question_ids"].index(qid)
            question_directory = directory / "questions" / f"{offset:04d}"
            if not (question_directory / "report.json").is_file():
                raise ValueError("terminal question event lacks a finalized report file")
            seen[qid] = offset
            if completion["status"] != "completed":
                excluded.append(
                    {
                        "run_id": directory.name,
                        "question_id": qid,
                        "reason": "incomplete_prediction_pair",
                    }
                )
                continue
            outcomes = {}
            for arm in ARMS:
                path = question_directory / f"{arm}_execution.json"
                raw = path.read_bytes()
                hashes[str(path)] = digest(raw)
                outcome = json.loads(raw)
                if (
                    outcome.get("status") != "completed"
                    or outcome.get("feedback") is not None
                    or "offline_gold_answers" in outcome
                    or outcome.get("result", {}).get("question_id") != qid
                ):
                    raise ValueError("execution is not a completed gold-free matching prediction")
                outcomes[arm] = outcome
            result = outcomes["S2G_AUTHOR_API4"]["result"]
            provenance = result["provenance"]
            if (
                provenance["source_sha256"] != PINNED_HASHES
                or provenance["remove_repeat_docs"] is not True
                or provenance["dedup_key"] != "document_id"
                or provenance["top_documents"] != 6
            ):
                raise ValueError("unreviewed author retrieval provenance")
            rows.append(
                {
                    "question_id": qid,
                    "offset": offset,
                    "run_id": directory.name,
                    "events": [e for e in result["events"] if e.get("kind") == "retrieval"],
                    "reported_rounds": result["retrieval_rounds"],
                    "context_fingerprint": launch["runtime_metadata"]["context_fingerprint"],
                }
            )
    rows.sort(key=lambda row: row["offset"])
    return rows if limit is None else rows[:limit], excluded, hashes, seen


def validate_report_coverage(reported_offsets, questions, *, require_all):
    """Check identities and positions, not just a superficially plausible count."""
    ordered_ids = [q.question_id for q in questions]
    for qid, offset in reported_offsets.items():
        if (
            type(offset) is not int
            or not 0 <= offset < len(ordered_ids)
            or ordered_ids[offset] != qid
        ):
            raise ValueError("reported ID/offset differs from frozen runtime question order")
    if require_all and (
        len(ordered_ids) != 500
        or set(reported_offsets) != set(ordered_ids)
        or set(reported_offsets.values()) != set(range(500))
    ):
        raise ValueError("full diagnostic requires exact frozen 500-ID/offset coverage")


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--runs-root", type=Path, default=root / "runs")
    parser.add_argument(
        "--manifest", type=Path, default=root / "data/hotpotqa/shared500_sep27_v1/manifest.json"
    )
    parser.add_argument(
        "--index", type=Path, default=root / "data/indexes/shared500_offline_bm25_v1.sqlite"
    )
    parser.add_argument(
        "--upstream",
        type=Path,
        default=root / "external/s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 2),
        default=1,
        help="Offline CPU processes only; independent read-only SQLite connections",
    )
    parser.add_argument(
        "--all-closed",
        action="store_true",
        help="Explicit full-series exploratory diagnostic after main run finishes",
    )
    args = parser.parse_args()
    if not args.index.is_file() or (not args.all_closed and not 1 <= args.limit <= 6):
        raise ValueError("existing index required; smoke limit must be 1..6")
    selected, excluded, hashes, reported_offsets = closed_predictions(
        args.runs_root, None if args.all_closed else args.limit
    )
    reported_count = len(reported_offsets)
    if args.all_closed and (
        reported_count != 500
        or any(item["reason"] in {"not_closed", "not_terminal"} for item in excluded)
    ):
        raise ValueError("full diagnostic requires all 500 questions to have closed reports")
    if not selected:
        raise ValueError("no completed prediction pairs")
    scope = load_author_scope(args.upstream)
    runtime = load_runtime(
        args.manifest, index_path=args.index, expected_manifest_sha256=MANIFEST_SHA
    )
    try:
        validate_report_coverage(reported_offsets, runtime.questions, require_all=args.all_closed)
        args.output.mkdir(parents=True, exist_ok=False)
        replay_started = time.perf_counter()
        if args.workers == 1:
            executed = [replay_item(item, runtime, scope) for item in selected]
        else:
            with ProcessPoolExecutor(
                max_workers=2,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(
                    str(args.manifest.resolve()),
                    str(args.index.resolve()),
                    str(args.upstream.resolve()),
                ),
            ) as executor:
                executed = list(executor.map(_worker_replay, selected, chunksize=1))
        replay_wall_seconds = time.perf_counter() - replay_started
        executed.sort(key=lambda item: item["replay"]["offset"])
        replays = [item["replay"] for item in executed]
        diagnostic_searches = sum(item["index_calls"] for item in executed)
        diagnostic_seconds = sum(item["index_seconds"] for item in executed)
        logical_requests = sum(item["logical_requests"] for item in executed)
        cache_hits = sum(item["cache_hits"] for item in executed)
        # Freeze both retrieval alternatives before private gold is even loaded.
        frozen_path = args.output / "retrieval_replays_before_gold.json"
        frozen_path.write_text(json.dumps(replays, ensure_ascii=False, indent=2), encoding="utf-8")
        gold = load_gold_after_execution(
            args.manifest,
            completed_question_ids=[r["question_id"] for r in replays],
            expected_manifest_sha256=MANIFEST_SHA,
        )
        rows = []
        for replay in replays:
            for state in score_question(
                replay, gold[replay["question_id"]], runtime.index.document
            ):
                rows.append(
                    {
                        "question_id": replay["question_id"],
                        "run_id": replay["run_id"],
                        "offset": replay["offset"],
                        **state,
                    }
                )
        comparable = [r for r in rows if r["comparison_status"] == "comparable"]
        rows.sort(key=lambda row: (row["offset"], row["round"]))
        summary = {
            "schema_version": "growrag-offline-query-state-replay-v2",
            "created_utc": datetime.now(UTC).isoformat(),
            "exploratory_not_preregistered": True,
            "selection": "first completed pairs by frozen offset; no answer/gold-based selection",
            "selected_questions": len(replays),
            "closed_question_completion_events_seen": reported_count,
            "gold_boundary": "Before frozen retrieval replays, only launch/events/final budget "
            "and gold-free execution files are read. Reports with labels are not read.",
            "retrieval_states": len(rows),
            "comparable_states": len(comparable),
            "status_counts": dict(Counter(r["comparison_status"] for r in rows)),
            "sentence_outcomes": dict(Counter(r["sentence_outcome"] for r in comparable)),
            "document_outcomes": dict(Counter(r["document_outcome"] for r in comparable)),
            "round1_sentence_outcomes": dict(
                Counter(r["sentence_outcome"] for r in comparable if r["round"] == 1)
            ),
            "round1_document_outcomes": dict(
                Counter(r["document_outcome"] for r in comparable if r["round"] == 1)
            ),
            "later_round_sentence_outcomes": dict(
                Counter(r["sentence_outcome"] for r in comparable if r["round"] > 1)
            ),
            "later_round_document_outcomes": dict(
                Counter(r["document_outcome"] for r in comparable if r["round"] > 1)
            ),
            "identical_query_states_included": sum(not r["query_text_changed"] for r in comparable),
            "identical_query_sentence_outcomes": dict(
                Counter(r["sentence_outcome"] for r in comparable if not r["query_text_changed"])
            ),
            "changed_query_sentence_outcomes": dict(
                Counter(r["sentence_outcome"] for r in comparable if r["query_text_changed"])
            ),
            "offline_unique_index_searches": diagnostic_searches,
            "offline_index_search_seconds": diagnostic_seconds,
            "offline_workers": args.workers,
            "offline_logical_retrieval_requests": logical_requests,
            "offline_per_question_cache_hits": cache_hits,
            "offline_replay_stage_wall_seconds": replay_wall_seconds,
            "timing_notice": "Index seconds sums elapsed search time across workers; "
            "replay wall time includes worker startup/verification. Cache is diagnostic-only.",
            "api_requests": 0,
            "reader_calls": 0,
            "runtime_metadata": runtime.metadata,
            "author_source_sha256": PINNED_HASHES,
            "excluded": excluded,
            "source_file_hashes": hashes,
            "frozen_replays_sha256": digest(frozen_path.read_bytes()),
            "script_sha256": digest(Path(__file__).read_bytes()),
            "limits": [
                "Same observed states, not complete counterfactual trajectories.",
                "Raw exact-version annotation arrival only; "
                "no retained-evidence or semantic support claim.",
                "No Reader calls: cannot infer final answer benefit or safety.",
                "Offline additional CPU retrieval is not a free online QPP estimate.",
                "Development exploratory diagnostic; "
                "correlated rounds are not independent questions.",
            ],
        }
        for name, expected in hashes.items():
            if digest(Path(name).read_bytes()) != expected:
                raise ValueError("closed source changed during diagnostic")
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output / "states.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        readme = (
            "# 查询变化与继续翻页：离线探索诊断\n\n"
            "比较发生在同一个已观察状态：作者执行的 gap query，和不改写的原问题，"
            "都经过同一个 SQLite BM25、top50、排除实际已见 doc ID、top6。"
            "反事实分支不改变下一轮的历史状态。\n\n"
            f"本次 {len(replays)} 题、{len(rows)} 个状态，{len(comparable)} 个可比。"
            f"额外索引搜索 {diagnostic_searches} 次，索引搜索用时 {diagnostic_seconds:.3f} 秒。"
            "这不是免费在线 QPP，也不包含任何新增 API/Reader 调用。\n\n"
            f"离线进程数 {args.workers}；逻辑检索 {logical_requests} 次；"
            f"实际索引调用 {diagnostic_searches} 次；每题缓存命中 {cache_hits} 次。"
            "索引耗时是各进程耗时求和，不等于并行墙钟时间；缓存不是在线方法优势。\n\n"
            "先逐轮重放执行 query：有序 doc ID、标题和全文必须与历史事件完全相同，"
            "否则整题标为不可比。然后才加载 gold，"
            "用版本精确的 doc ID、原句位置和文本哈希计算覆盖。\n\n"
            "sentence_outcome 为 gap query 新到达支持句数减去原问题新到达支持句数的符号；"
            "document_outcome 同理按支持文档计数。它们是原文到达，不是 Extractor 留下的证据，"
            "更不是回答正确率或语义充分性。missing_gold_* 表示距完整人工标注覆盖还差多少。\n\n"
            "summary.json 分别列出 round1（空历史、PRE）与 round>=2（已观察本题状态、POST）。"
            "相同 query 也保留在分母中，正常应为零差。轮次有关联，不当作独立问题做显著性推断。\n\n"
            "states.jsonl 保存 query、相同的 past_doc_ids、两路候选 doc IDs、"
            "支持覆盖增量及不可比状态。"
            "retrieval_replays_before_gold.json 是读取 gold 前冻结的重放结果。\n\n"
            "全量诊断应等 500 题封口后使用 --all-closed，并提供新的 --output 目录。"
            "不要依据这个探索诊断修改仍在执行的主实验。\n"
        )
        (args.output / "README.md").write_text(readme, encoding="utf-8")
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "selected_questions",
                        "retrieval_states",
                        "comparable_states",
                        "sentence_outcomes",
                        "offline_unique_index_searches",
                        "offline_index_search_seconds",
                    )
                },
                indent=2,
            )
        )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
