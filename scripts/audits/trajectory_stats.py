"""只读分析封口的 shared500 S2G 轨迹；不调用 API，不载入题库或运行模型。

默认只向 stdout 打印汇总。--output-dir 必须是不存在的新目录，且只允许
写入项目的 ignored runs 目录。逐题输出只保留 ID/hash/计数，不抄题干答案。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

BASE, S2G = "BASE1_AUTHOR_READER", "S2G_AUTHOR_API4"
PREFIX = "2026-09-27_s2g_shared500_v1_"
EXPECTED_MANIFEST = "2fb117c03b64be3ab12398a297c82ce3315600c14e33f5dcd56c5c6c5c59e064"
COST_FIELDS = ("api_requests", "input_tokens", "output_tokens", "estimated_actual_cny")
PLAN_FIELDS = (
    "protocol",
    "manifest_sha256",
    "model",
    "generation_profile",
    "max_retrieval_rounds",
    "top_docs",
    "gap_profile",
)
# 三个完整源码快照仅在已核查的未启动续跑/离线分析代码上不同。
# 不是无限允许未来 runner 修改；任何新源码快照都必须重新审计后显式登记。
REVIEWED_SOURCE_SNAPSHOTS = frozenset(
    {
        "de5910a7b5105a990d6b6693b6cb7d85386c4c0f1e9283e75bd68e4b44496d5a",
        "71dda6fd1177897f634be3dfe6182a49902032c0eed12d4773fd90936dfceca0",
        "0dd575c630fe3a9cfcee53664a619c2ac1a5eb66315ebcd456b0b1113439b2ed",
    }
)
REVIEWED_NONMETHOD_DIFFERENCES = frozenset(
    {
        "src/growrag/experiments/run_shared_s2g.py",
        "src/growrag/experiments/shared_continuation.py",
        "src/growrag/experiments/analyze_shared_s2g.py",
    }
)


def canonical_sha(value: object) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def method_signature(plan: dict, snapshot: dict) -> dict:
    """Fail closed on unreviewed source or missing method-defining provenance."""
    source_sha = plan["source_sha256"]
    if source_sha not in REVIEWED_SOURCE_SNAPSHOTS:
        raise ValueError("unreviewed source snapshot")
    files = snapshot["files"]
    if snapshot["sha256"] != source_sha or canonical_sha(files) != source_sha:
        raise ValueError("source snapshot aggregate hash mismatch")
    for item in files.values():
        if digest(item["text"]) != item["sha256"]:
            raise ValueError("source snapshot file text/hash mismatch")
    for name in ("s2g_author_api", "shared_s2g_corpus", "api_client", "run_s2g_author_pilot"):
        if f"src/growrag/experiments/{name}.py" not in files:
            raise ValueError("source snapshot lacks core implementation")
    author, runtime = plan["author"], plan["runtime_metadata"]
    author_required = {
        "upstream_commit",
        "source_sha256",
        "author_prompt_sha256",
        "sentence_splitter",
        "remove_repeat_docs",
        "dedup_key",
        "trained_author_judge_used",
        "gap_profile",
        "retrieval_backend",
        "backend_generation_settings",
        "gold_provided_to_author_loop",
    }
    runtime_required = {
        "manifest_sha256",
        "corpus_sha256",
        "document_count",
        "question_count",
        "retrieval_config",
        "context_fingerprint",
    }
    if not author_required <= author.keys() or not runtime_required <= runtime.keys():
        raise ValueError("missing method provenance")
    if (
        not plan["no_training_no_memory_updates"]
        or plan["official_dev_test_used"]
        or author["gold_provided_to_author_loop"]
        or author["trained_author_judge_used"]
        or author["remove_repeat_docs"] is not True
    ):
        raise ValueError("unexpected training/gold/dedup policy")
    signature = {
        key: plan[key]
        for key in (
            *PLAN_FIELDS,
            "backend_output_caps",
            "answer_length_policy",
            "arms",
            "no_training_no_memory_updates",
            "official_dev_test_used",
            "price_input_cny_per_million",
            "price_output_cny_per_million",
        )
    }
    signature["author"] = {k: v for k, v in author.items() if k != "actual_main_batch_filename"}
    signature["runtime"] = {k: v for k, v in runtime.items() if k != "index_path"}
    signature["method_source_sha256"] = canonical_sha(
        {
            path: item["sha256"]
            for path, item in files.items()
            if path not in REVIEWED_NONMETHOD_DIFFERENCES
        }
    )
    return signature


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def read_stable(path: Path) -> tuple[object, str]:
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed while reading: {path}")
    return json.loads(data), digest(data)


def exit_record(path: Path) -> dict | None:
    """最后一条 exit 在 final_budget 写完后产生；失败封口同样可审计。"""
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 16384))
        lines = handle.read().splitlines()
    if not lines:
        return None
    try:
        record = json.loads(lines[-1])
    except (ValueError, UnicodeDecodeError):
        return None
    return record if record.get("kind") == "exit" else None


def totals(calls: list[dict]) -> dict:
    result = {
        "recorded_attempts": len(calls),
        "statuses": dict(Counter(c.get("status", "unknown") for c in calls)),
    }
    for field in COST_FIELDS:
        values = [c.get(field) for c in calls]
        missing = sum(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
        result[field] = None if missing else sum(values)
        if missing:
            result[f"{field}_unknown_calls"] = missing
    return result


def normalized(query: str) -> str:
    # 仅空白合并和 lower；不删标点、不推断语义等价。
    return " ".join(query.split()).lower()


def elapsed_stats(values: list[float | None]) -> dict:
    known = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0)
    return {
        "records": len(values),
        "known_records": len(known),
        "unknown_records": len(values) - len(known),
        "known_sum_seconds": sum(known),
        "mean_seconds": mean(known) if known else None,
        "median_seconds": median(known) if known else None,
        "p90_seconds_nearest_rank": known[math.ceil(0.9 * len(known)) - 1] if known else None,
    }


def source_key(source: dict) -> tuple[str, int, str]:
    return source["doc_id"], source["sentence_id"], digest(source["text"])


def inspect_trace(report: dict, batch_name: str) -> dict:
    arm = report["arms"][S2G]
    result = arm["result"]
    events = result["events"]
    searches = [e for e in events if e["kind"] == "retrieval"]
    extracts = [e for e in events if e["kind"] == "extraction"]
    if len(searches) != result["retrieval_rounds"] or len(extracts) != len(searches):
        raise ValueError("retrieval/extraction round count mismatch")
    if len({e["round"] for e in extracts}) != len(extracts):
        raise ValueError("duplicate extraction round")
    by_round = {e["round"]: e for e in extracts}
    prior_queries, prior_docs, prior_sources = [], set(), set()
    rounds, source_entries, retrieved_entries = [], [], []
    for i, search in enumerate(searches):
        if search["round"] != i + 1 or len(search["queries"]) != 1 or len(search["documents"]) != 1:
            raise ValueError("unexpected single-question retrieval shape or round order")
        query = search["queries"][0]
        if not isinstance(query, str):
            raise ValueError("query must be text")
        docs = search["documents"][0]
        doc_ids = [doc["doc_id"] for doc in docs if doc["doc_id"]]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("duplicate document inside one post-filter retrieval")
        extraction = by_round[search["round"]]
        if extraction["event_index"] <= search["event_index"]:
            raise ValueError("extraction did not follow its retrieval")
        if i + 1 < len(searches) and extraction["event_index"] >= searches[i + 1]["event_index"]:
            raise ValueError("extraction crossed the next retrieval boundary")
        real_sources = [s for s in extraction["sources"] if s["doc_id"]]
        if any(s["doc_id"] not in doc_ids for s in real_sources):
            raise ValueError("retained source was not passed to this extractor")
        keys = {source_key(s) for s in real_sources}
        new_sources = keys - prior_sources
        exact_previous = [j + 1 for j, old in enumerate(prior_queries) if old == query]
        norm_previous = [
            j + 1 for j, old in enumerate(prior_queries) if normalized(old) == normalized(query)
        ]
        continuing = i + 1 < len(searches)
        rounds.append(
            {
                "round": i + 1,
                "query_sha256": digest(query),
                "normalized_query_sha256": digest(normalized(query)),
                "exact_repeat_of_rounds": exact_previous,
                "normalized_repeat_of_rounds": norm_previous,
                "post_filter_document_count": len(doc_ids),
                "new_document_ids_count": len(set(doc_ids) - prior_docs),
                "previously_passed_document_ids_count": len(set(doc_ids) & prior_docs),
                "selected_source_entries": len(extraction["sources"]),
                "placeholder_source_entries": len(extraction["sources"]) - len(real_sources),
                "unique_real_retained_pointers": len(keys),
                "new_unique_real_retained_pointers": len(new_sources),
                "later_retrieval_observed": continuing,
                "zero_new_pointers_but_continued": not new_sources and continuing,
            }
        )
        prior_queries.append(query)
        prior_docs.update(doc_ids)
        prior_sources.update(keys)
        source_entries.extend(source_key(s) for s in extraction["sources"])
        retrieved_entries.extend(doc_ids)
    if source_entries != [source_key(s) for s in result["sources"]]:
        raise ValueError("result sources disagree with extraction events")
    if retrieved_entries != [d["doc_id"] for d in result["retrieved_documents"]]:
        raise ValueError("result documents disagree with post-filter events")
    if len(arm["calls"]) != result["api_calls"]:
        raise ValueError("result API-call count mismatch")
    response_ids = [e["trace_id"] for e in events if e["kind"] == "api_response"]
    if response_ids != [c["trace_id"] for c in arm["calls"]]:
        raise ValueError("response-to-accounting trace IDs mismatch")
    feedback = {name: report["arms"][name].get("feedback") or {} for name in (BASE, S2G)}
    invalid = any(f.get("unscorable_annotation") for f in feedback.values())
    metric_values = [feedback[a].get(m) for a in (BASE, S2G) for m in ("answer_em", "answer_f1")]
    scorable = (
        not invalid
        and report.get("scoring_status") == "completed"
        and all(
            isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in metric_values
        )
    )
    return {
        "question_id": report["question_id"],
        "offset": report["offset"],
        "batch": batch_name,
        "trace_report": f"{batch_name}/questions/{report['offset']:04d}/report.json",
        "retrieval_rounds": len(searches),
        "stop_reason": result["stop_reason"],
        "exact_repeat_rounds": sum(bool(r["exact_repeat_of_rounds"]) for r in rounds),
        "normalized_repeat_rounds": sum(bool(r["normalized_repeat_of_rounds"]) for r in rounds),
        "normalized_repeat_rounds_with_new_docs": sum(
            bool(r["normalized_repeat_of_rounds"]) and r["new_document_ids_count"] > 0
            for r in rounds
        ),
        "zero_new_retention_continue_rounds": sum(
            r["zero_new_pointers_but_continued"] for r in rounds
        ),
        "annotation_invalid": invalid,
        "scorable_pair": scorable,
        "scoring_status": report.get("scoring_status"),
        "metrics": {
            a: {m: feedback[a].get(m) for m in ("answer_em", "answer_f1")} for a in (BASE, S2G)
        },
        "costs": {a: totals(report["arms"][a]["calls"]) for a in (BASE, S2G)},
        "elapsed_seconds": {a: report["arms"][a].get("elapsed_seconds") for a in (BASE, S2G)},
        "rounds": rounds,
    }


def grouped(rows: list[dict], costs: dict[str, list[dict]]) -> dict:
    """整题 S2G 分支账单，不能解释成该行为额外造成的费用。"""
    return {
        "question_count": len(rows),
        "scorable_pairs": sum(row["scorable_pair"] for row in rows),
        "s2g_whole_question_cost": totals([c for row in rows for c in costs[row["question_id"]]]),
        "mean_retrieval_rounds": mean(r["retrieval_rounds"] for r in rows) if rows else None,
    }


def metric_upper_bounds(rows: list[dict]) -> dict:
    rows = [r for r in rows if r["scorable_pair"]]
    output = {
        "n": len(rows),
        "notice": (
            "逐题看答案后的指标独立 oracle 上界；不是已实现路由器。"
            "EM/F1 可选择不同方法，未计路由成本。"
        ),
    }
    if not rows:
        return output
    for metric in ("answer_em", "answer_f1"):
        values = [[r["metrics"][arm][metric] for r in rows] for arm in (BASE, S2G)]
        baseline, s2g = map(mean, values)
        upper = mean(max(pair) for pair in zip(*values, strict=True))
        output[metric] = {
            "base": baseline,
            "s2g": s2g,
            "best_constant": max(baseline, s2g),
            "posthoc_oracle": upper,
            "oracle_minus_best_constant": upper - max(baseline, s2g),
        }
    output["paired_em"] = {
        "repairs_base0_s2g1": sum(
            r["metrics"][BASE]["answer_em"] == 0 and r["metrics"][S2G]["answer_em"] == 1
            for r in rows
        ),
        "harms_base1_s2g0": sum(
            r["metrics"][BASE]["answer_em"] == 1 and r["metrics"][S2G]["answer_em"] == 0
            for r in rows
        ),
    }
    return output


def analyze(runs_root: Path, max_batches: int | None = None) -> tuple[dict, list[dict]]:
    rows, evidence, skipped, issues, ledger_calls = [], [], [], [], []
    costs, complete_ids, all_call_ids = {}, set(), set()
    reference_plan = reference_signature = None
    reports_seen = 0
    arm_failures = Counter()
    attempted_elapsed = {a: [] for a in (BASE, S2G)}
    failed_elapsed = {a: [] for a in (BASE, S2G)}
    for directory in sorted(runs_root.glob(PREFIX + "*")):
        if not directory.is_dir():
            continue
        required = (
            "reports.json",
            "summary.json",
            "final_budget.json",
            "launch_plan.json",
            "source_snapshot.json",
        )
        sealed_exit = exit_record(directory / "events.jsonl")
        if not all((directory / name).is_file() for name in required) or sealed_exit is None:
            skipped.append(directory.name)
            continue
        if max_batches is not None and len(evidence) >= max_batches:
            break
        loaded, fingerprints = {}, {}
        for name in required:
            loaded[name], fingerprints[name] = read_stable(directory / name)
        plan = loaded["launch_plan.json"]
        protocol = {key: plan[key] for key in PLAN_FIELDS}
        signature = method_signature(plan, loaded["source_snapshot.json"])
        if protocol["manifest_sha256"] != EXPECTED_MANIFEST:
            raise ValueError("manifest mismatch: " + directory.name)
        if reference_plan is not None and protocol != reference_plan:
            raise ValueError("cannot pool mixed protocols: " + directory.name)
        if reference_signature is not None and signature != reference_signature:
            raise ValueError("cannot pool mixed method provenance: " + directory.name)
        reference_plan = protocol
        reference_signature = signature
        reports = loaded["reports.json"]
        paired = 0
        for report in reports:
            reports_seen += 1
            for a in (BASE, S2G):
                arm = report["arms"].get(a, {})
                if arm.get("status") in {"completed", "failed"}:
                    attempted_elapsed[a].append(arm.get("elapsed_seconds"))
                if arm.get("status") == "failed":
                    failed_elapsed[a].append(arm.get("elapsed_seconds"))
            completed = report.get("complete_pair") and all(
                report["arms"].get(a, {}).get("status") == "completed" for a in (BASE, S2G)
            )
            if not completed:
                issues.append(
                    {
                        "batch": directory.name,
                        "question_id": report["question_id"],
                        "offset": report["offset"],
                        "reason": "incomplete_pair",
                        "arm_status": {
                            a: report["arms"].get(a, {}).get("status") for a in (BASE, S2G)
                        },
                    }
                )
                for a in (BASE, S2G):
                    status = report["arms"].get(a, {}).get("status", "missing")
                    if status != "completed":
                        arm_failures[f"{a}:{status}"] += 1
                continue
            paired += 1
            qid = report["question_id"]
            if qid in complete_ids:
                raise ValueError("duplicate completed question ID: " + qid)
            complete_ids.add(qid)
            rows.append(inspect_trace(report, directory.name))
            costs[qid] = report["arms"][S2G]["calls"]
        if paired != loaded["summary.json"]["complete_pairs"]:
            raise ValueError("sealed summary complete-pair count disagrees")
        for call in loaded["final_budget.json"]["calls"]:
            if call["trace_id"] in all_call_ids:
                raise ValueError("duplicate ledger trace ID")
            all_call_ids.add(call["trace_id"])
            ledger_calls.append(call)
        if exit_record(directory / "events.jsonl") != sealed_exit:
            raise ValueError("batch changed after seal check")
        evidence.append(
            {
                "batch": directory.name,
                "sha256": fingerprints,
                "source_snapshot_sha256": plan["source_sha256"],
                "exit": sealed_exit,
                "reports": len(reports),
                "complete_pairs": paired,
                "started_without_report": loaded["summary.json"].get("started_without_report", 0),
            }
        )
    categories = {
        "all_complete_pairs": rows,
        "has_exact_repeat": [r for r in rows if r["exact_repeat_rounds"]],
        "has_normalized_repeat_including_exact": [r for r in rows if r["normalized_repeat_rounds"]],
        "has_normalized_repeat_and_new_docs": [
            r for r in rows if r["normalized_repeat_rounds_with_new_docs"]
        ],
        "has_zero_new_retention_but_continued": [
            r for r in rows if r["zero_new_retention_continue_rounds"]
        ],
        "no_normalized_repeat": [r for r in rows if not r["normalized_repeat_rounds"]],
    }
    rounds = [step for row in rows for step in row["rounds"]]
    result = {
        "schema": "growrag-shared500-trace-audit-v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "protocol": reference_plan,
        "method_signature": reference_signature,
        "method_signature_sha256": canonical_sha(reference_signature),
        "sealed_batches": evidence,
        "unsealed_skipped": skipped,
        "reports_seen": reports_seen,
        "complete_pairs_analyzed": len(rows),
        "incomplete_pairs": issues,
        "arm_noncompletion_counts": dict(arm_failures),
        "started_without_report_total": sum(e["started_without_report"] for e in evidence),
        "bad_annotation_complete_pairs": sum(r["annotation_invalid"] for r in rows),
        "scoring_failed_complete_pairs": sum(r["scoring_status"] == "failed" for r in rows),
        "retrieval_round_distribution": dict(
            sorted(Counter(r["retrieval_rounds"] for r in rows).items())
        ),
        "stop_reason_distribution": dict(Counter(r["stop_reason"] for r in rows)),
        "round_phenomena": {
            "total_retrieval_rounds": len(rounds),
            "exact_repeat_rounds": sum(bool(r["exact_repeat_of_rounds"]) for r in rounds),
            "normalized_repeat_rounds_including_exact": sum(
                bool(r["normalized_repeat_of_rounds"]) for r in rounds
            ),
            "normalized_only_repeat_rounds": sum(
                bool(r["normalized_repeat_of_rounds"]) and not r["exact_repeat_of_rounds"]
                for r in rounds
            ),
            "zero_new_retention_but_continued_rounds": sum(
                r["zero_new_pointers_but_continued"] for r in rounds
            ),
            "new_document_ids_per_round_distribution": dict(
                sorted(Counter(r["new_document_ids_count"] for r in rounds).items())
            ),
            "previously_passed_document_ids_total": sum(
                r["previously_passed_document_ids_count"] for r in rounds
            ),
            "placeholder_source_entries_total": sum(
                r["placeholder_source_entries"] for r in rounds
            ),
        },
        "categories_overlap": {name: grouped(cohort, costs) for name, cohort in categories.items()},
        "all_sealed_ledger_calls_including_incomplete_or_unreported": totals(ledger_calls),
        "latency": {
            "complete_pair_arms": {
                a: elapsed_stats([r["elapsed_seconds"][a] for r in rows]) for a in (BASE, S2G)
            },
            "all_reported_attempted_arms_including_failed": {
                a: elapsed_stats(attempted_elapsed[a]) for a in (BASE, S2G)
            },
            "failed_reported_arms_only": {a: elapsed_stats(failed_elapsed[a]) for a in (BASE, S2G)},
            "unreported_started_questions_timing_unknown": sum(
                e["started_without_report"] for e in evidence
            ),
            "notice": (
                "本机串行实验各 arm 的已有 elapsed_seconds：含检索/API往返/部分日志开销，"
                "不含建索引、停机排错或全部封口开销；不是稳定部署延迟 benchmark。"
                "配对均值不含失败；失败已报耗时另列，未产报告的耗时未知，不当作零。"
                "p90 用 sorted[ceil(0.9*n)-1]；未知值不参加均值但保留计数。"
            ),
        },
        "posthoc_metric_upper_bounds": metric_upper_bounds(rows),
        "interpretation": [
            "仅封口且两路 completed 的题进入轨迹/方法表现分析；坏标注仍可看轨迹但不进 EM/F1。",
            "retrieval 已经过 past_doc_ids 过滤，是交给选句器的真实文档；不含最初50候选。",
            "同query可因过去文档过滤继续取新文档，重复文本不是无效检索或失败原因的证明。",
            "规范化仅合并空白并转小写；不检查语义等价。exact 是 normalized 的子集。",
            "新保留证据=此前未见(doc_id,作者1-based句号,原句SHA)指针；不是新事实/新信息量的语义判断。",
            "零新增但继续要求其后确实观察到下一次 retrieval；不是仅有 judge/answer 调用。",
            "分类可重叠；各类成本为这些题完整 S2G 分支费用，不是该行为的因果或边际成本。",
            "estimated_actual_cny 是 token 的本地价格估算，不是供应商发票；未知账单保留 null。",
            "oracle 是事后有标签上界且未计路由费用；并不说明检索前能选到或已实现这样的路由。",
            "这是自制共享语料上的 Qwen 迁移开发诊断，不是原论文官方测试复现。",
        ],
    }
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=Path, default=Path(__file__).resolve().parents[2] / "runs"
    )
    parser.add_argument("--max-sealed-batches", type=int)
    parser.add_argument(
        "--output-dir", type=Path, help="optional NEW directory inside the project runs folder"
    )
    args = parser.parse_args()
    if args.max_sealed_batches is not None and args.max_sealed_batches < 1:
        parser.error("--max-sealed-batches must be positive")
    if args.output_dir is not None:
        destination = args.output_dir.resolve()
        if destination.exists() or not destination.is_relative_to(
            Path(__file__).resolve().parents[2] / "runs"
        ):
            parser.error("output must be a nonexistent descendant of the project runs folder")
    result, rows = analyze(args.runs_root, args.max_sealed_batches)
    if args.output_dir is not None:
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        with (destination / "questions.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"sealed_batches", "incomplete_pairs", "interpretation"}
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
