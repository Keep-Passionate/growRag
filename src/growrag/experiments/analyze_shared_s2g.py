"""Offline aggregation of one frozen shared-S2G series; never loads gold or calls APIs.

中文：效果只在同一批完整可评分配对题上比较；所有尝试的成本从最终账本汇总，
不能只算成功题。失败、未安排、无执行记录、标注不可评分、评分异常分别报告。
诊断索引只描述已经记录的现象，不将相关性当成根因或记忆收益。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from .shared_continuation import verify_unstarted_continuation

ARMS = ("BASE1_AUTHOR_READER", "S2G_AUTHOR_API4")
METRICS = ("answer_em", "answer_f1", "raw_support_recall", "retained_support_recall")
SIGNATURE_FIELDS = (
    "protocol",
    "manifest_sha256",
    "generation_profile",
    "model",
    "backend_output_caps",
    "max_retrieval_rounds",
    "top_docs",
    "gap_profile",
    "arms",
    "author",
    "answer_length_policy",
    "price_input_cny_per_million",
    "price_output_cny_per_million",
)


def _read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _token(value):
    return type(value) is int and value >= 0


def _metric(value, name):
    if value is not None and (not _number(value) or not 0 <= value <= 1):
        raise ValueError(f"invalid metric {name}")
    if name == "answer_em" and value is not None and value not in (0, 1):
        raise ValueError("answer EM must be zero or one")
    return value


def _compact_arm(row):
    status = row.get("status")
    if status not in {"completed", "failed", "not_executed"}:
        raise ValueError("unknown execution status")
    result = row.get("result") or {}
    feedback = row.get("feedback") or {}
    rounds = result.get("retrieval_rounds")
    if rounds is not None and (not _token(rounds) or rounds > 4):
        raise ValueError("invalid retrieval rounds")
    events = row.get("events", result.get("events", []))
    observed_rounds = {e.get("round") for e in events if e.get("kind") == "retrieval"}
    observed_rounds.discard(None)
    return {
        "status": status,
        "error_type": row.get("error_type"),
        "answer": result.get("answer"),
        "stop_reason": result.get("stop_reason"),
        "retrieval_rounds": rounds,
        "partial_retrieval_rounds_observed": len(observed_rounds),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "feedback": {name: _metric(feedback.get(name), name) for name in METRICS},
        "unscorable_annotation": feedback.get("unscorable_annotation", False),
        "annotation_status": feedback.get("annotation_status"),
        "annotation_issue": feedback.get("annotation_issue"),
        "unaligned_sources": feedback.get("unaligned_sources"),
        "partial_raw_sentences": feedback.get("partial_raw_sentences"),
        "calls": row.get("calls", []),
    }


def _row_status(row):
    arms = row["arms"]
    if any(arms[a]["status"] == "failed" for a in ARMS):
        return "execution_failed"
    if not all(arms[a]["status"] == "completed" for a in ARMS):
        return "incomplete_pair"
    if any(arms[a]["unscorable_annotation"] for a in ARMS):
        if not all(arms[a]["unscorable_annotation"] for a in ARMS):
            raise ValueError("paired annotation status disagrees")
        return "unscorable_annotation"
    if row.get("scoring_status") == "failed":
        return "scoring_failed"
    if all(arms[a]["feedback"][m] is not None for a in ARMS for m in ("answer_em", "answer_f1")):
        return "scored_pair"
    return "complete_unscored"


def _cost(calls):
    """Token/currency sums are known lower bounds whenever corresponding unknowns exist."""
    result = {
        "recorded_attempts": len(calls),
        "known_api_requests": 0,
        "unknown_api_request_attempts": 0,
        "known_input_tokens": 0,
        "known_output_tokens": 0,
        "unknown_input_token_attempts": 0,
        "unknown_output_token_attempts": 0,
        "known_estimated_cny": 0.0,
        "unknown_cost_attempts": 0,
        "reserved_cny": 0.0,
        "unknown_reservation_attempts": 0,
        "failed_attempts": 0,
        "completed_attempts": 0,
        "other_status_attempts": 0,
    }
    for call in calls:
        requests = call.get("api_requests")
        if _token(requests):
            result["known_api_requests"] += requests
        else:
            result["unknown_api_request_attempts"] += 1
        for field in ("input_tokens", "output_tokens"):
            value = call.get(field)
            if _token(value):
                result[f"known_{field}"] += value
            elif requests != 0:
                result[f"unknown_{field[:-1]}_attempts"] += 1
        estimate = call.get("estimated_actual_cny")
        if _number(estimate) and estimate >= 0:
            result["known_estimated_cny"] += estimate
        elif requests != 0:
            result["unknown_cost_attempts"] += 1
        reserve = call.get("reserved_cny")
        if _number(reserve) and reserve >= 0:
            result["reserved_cny"] += reserve
        else:
            result["unknown_reservation_attempts"] += 1
        status = call.get("status")
        result[
            {"failed": "failed_attempts", "completed": "completed_attempts"}.get(
                status, "other_status_attempts"
            )
        ] += 1
    return result


def _effect(rows):
    """All arm means share the same paired answer cohort; metric-specific n stays visible."""
    result = {"paired_scored_n": len(rows), "arms": {}}
    for arm in ARMS:
        result["arms"][arm] = {}
        for metric in METRICS:
            values = [
                r["arms"][arm]["feedback"][metric]
                for r in rows
                if r["arms"][arm]["feedback"][metric] is not None
            ]
            result["arms"][arm][metric] = {
                "mean": mean(values) if values else None,
                "n": len(values),
            }
    pairs = [(r["arms"][ARMS[0]]["feedback"], r["arms"][ARMS[1]]["feedback"]) for r in rows]
    base_wrong = sum(a["answer_em"] == 0 for a, _ in pairs)
    base_right = sum(a["answer_em"] == 1 for a, _ in pairs)
    repairs = sum(a["answer_em"] == 0 and b["answer_em"] == 1 for a, b in pairs)
    harms = sum(a["answer_em"] == 1 and b["answer_em"] == 0 for a, b in pairs)
    result["paired_em"] = {
        "repairs": repairs,
        "harms": harms,
        "net_repaired_questions": repairs - harms,
        "both_correct": sum(a["answer_em"] == b["answer_em"] == 1 for a, b in pairs),
        "both_wrong": sum(a["answer_em"] == b["answer_em"] == 0 for a, b in pairs),
        "base_wrong_n": base_wrong,
        "base_correct_n": base_right,
        "repair_rate_among_base_wrong": repairs / base_wrong if base_wrong else None,
        "harm_rate_among_base_correct": harms / base_right if base_right else None,
    }
    deltas = [b["answer_f1"] - a["answer_f1"] for a, b in pairs]
    result["paired_f1"] = {
        "mean_delta": mean(deltas) if deltas else None,
        "improved": sum(d > 1e-12 for d in deltas),
        "worsened": sum(d < -1e-12 for d in deltas),
        "unchanged": sum(abs(d) <= 1e-12 for d in deltas),
    }
    return result


def _diagnoses(row):
    if row["analysis_status"] != "scored_pair":
        return [row["analysis_status"]]
    base, s2g = (row["arms"][a] for a in ARMS)
    a, b = base["feedback"], s2g["feedback"]
    labels = []
    if a["answer_em"] == 0 and b["answer_em"] == 1:
        labels.append("paired_repair")
    if a["answer_em"] == 1 and b["answer_em"] == 0:
        labels.append("paired_harm")
    if a["answer_em"] == b["answer_em"] == 0:
        labels.append("both_wrong")
    if b["answer_em"] == 0 and s2g["stop_reason"] == "sufficient":
        labels.append("s2g_stopped_sufficient_but_em_wrong")
    if b["answer_em"] == 0 and s2g["stop_reason"] == "max_turns":
        labels.append("s2g_budget_stop_and_em_wrong")
    raw, retained = b["raw_support_recall"], b["retained_support_recall"]
    if raw is not None and retained is not None and raw > retained:
        labels.append("s2g_retained_annotated_coverage_below_raw")
    if raw is not None and raw < 1:
        labels.append("s2g_did_not_retrieve_all_annotated_support")
    if b["answer_em"] == 1 and raw is not None and raw < 1:
        labels.append("s2g_em_correct_without_all_annotated_support")
    if s2g.get("unaligned_sources"):
        labels.append("s2g_unaligned_source_pointers")
    return labels


def _ordered_launch_paths(runs_root, series):
    """Parents first, independent of lexical filenames; reject cycles and missing parents."""
    records, ignored = {}, []
    for path in sorted(runs_root.glob("*_s2g_shared*/launch_plan.json")):
        launch, _ = _read(path)
        if launch.get("series") != series:
            ignored.append(path.parent.name)
            continue
        records[path.parent.name] = (path, launch.get("continuation_of"))
    ordered, visiting, visited = [], set(), set()

    def visit(name):
        if name in visiting:
            raise ValueError("continuation cycle")
        if name in visited:
            return
        if name not in records:
            raise ValueError("continuation parent is missing or belongs to another series")
        visiting.add(name)
        path, parent = records[name]
        if parent is not None:
            if not isinstance(parent, str):
                raise ValueError("continuation_of must be exact prior run name")
            visit(parent)
        visiting.remove(name)
        visited.add(name)
        ordered.append(path)

    for name in records:
        visit(name)
    return ordered, ignored


def analyze_series(
    manifest_path,
    runs_root,
    *,
    series,
    protocol,
    generation_profile,
    expected_manifest_sha256=None,
    max_examples=10,
):
    """One exact series only; conflicting cohorts or repeated claimed IDs are errors."""
    if not re.fullmatch(r"(?:32|500)_v[1-9][0-9]*", series):
        raise ValueError("explicit 32/500 series required")
    if type(max_examples) is not int or max_examples < 0:
        raise ValueError("max_examples must be nonnegative")
    manifest_path, runs_root = (
        Path(manifest_path).resolve(strict=True),
        Path(runs_root).resolve(strict=True),
    )
    manifest, digest = _read(manifest_path)
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError("manifest SHA mismatch")
    if manifest_path.with_suffix(".sha256").read_text().strip() != f"{digest}  manifest.json":
        raise ValueError("manifest sidecar mismatch")
    ids = manifest.get("question_ids")
    types = manifest.get("question_types", {})
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) != len(set(ids))
        or set(types) != set(ids)
        or manifest.get("role") != "development"
        or manifest.get("official_split") != "train"
    ):
        raise ValueError("invalid frozen development manifest")
    expected_size = int(series.split("_")[0])
    if len(ids) != expected_size:
        raise ValueError("series size does not match frozen manifest")
    positions = {qid: i for i, qid in enumerate(ids)}
    claimed, observed, calls_by_trace, owners, batches = {}, {}, {}, {}, []
    launch_paths, ignored = _ordered_launch_paths(runs_root, series)
    continuations = []
    reference_signature = None
    costs_incomplete = []
    source_versions = set()
    for launch_path in launch_paths:
        launch, launch_sha = _read(launch_path)
        if launch.get("series") != series:
            ignored.append(launch_path.parent.name)
            continue
        if (
            launch.get("protocol") != protocol
            or launch.get("manifest_sha256") != digest
            or launch.get("generation_profile") != generation_profile
        ):
            raise ValueError("matching series mixes protocol/manifest/generation profile")
        signature = {key: launch.get(key) for key in SIGNATURE_FIELDS}
        signature["retrieval_config"] = (launch.get("runtime_metadata") or {}).get(
            "retrieval_config"
        )
        signature["corpus_sha256"] = (launch.get("runtime_metadata") or {}).get("corpus_sha256")
        if launch.get("arms") != list(ARMS) or any(
            launch.get(key) is None for key in SIGNATURE_FIELDS
        ):
            raise ValueError("missing run signature or unexpected arms")
        if reference_signature is not None and signature != reference_signature:
            raise ValueError("execution configuration changed within series")
        reference_signature = signature
        source_versions.add(launch.get("source_sha256"))
        batch_ids = launch.get("question_ids", [])
        start, count = launch.get("start"), launch.get("count")
        if (
            type(start) is not int
            or type(count) is not int
            or start < 0
            or count < 1
            or batch_ids != ids[start : start + count]
            or count != len(batch_ids)
        ):
            raise ValueError("repeated question IDs or batch not matching frozen slice")
        directory, run_id = launch_path.parent, launch_path.parent.name
        if launch.get("run_id") != run_id:
            raise ValueError("run directory does not match launch identity")
        overlap = set(batch_ids) & set(claimed)
        continuation_of = launch.get("continuation_of")
        if continuation_of is not None:
            if (
                overlap != set(batch_ids)
                or {claimed[qid] for qid in overlap} != {continuation_of}
                or overlap & set(observed)
            ):
                raise ValueError("continuation overlaps executed or differently claimed questions")
            proof = verify_unstarted_continuation(
                runs_root,
                continuation_of,
                manifest_sha256=digest,
                question_ids=batch_ids,
                generation_profile=generation_profile,
                protocol=protocol,
                model=launch["model"],
            )
            if (
                launch.get("continuation_proof") is not None
                and launch["continuation_proof"] != proof
            ):
                raise ValueError("frozen continuation proof changed")
            continuations.append({"run_id": run_id, **proof})
        elif overlap:
            raise ValueError("repeated question IDs without proven explicit continuation")
        claimed.update({qid: run_id for qid in batch_ids})
        report_path, budget_path = directory / "reports.json", directory / "final_budget.json"
        reports, reports_sha = _read(report_path) if report_path.exists() else ([], None)
        if not isinstance(reports, list):
            raise ValueError("reports must be a list")
        arm_calls = {}
        for n, report in enumerate(reports):
            qid = report.get("question_id")
            if qid not in batch_ids or qid in observed or set(report.get("arms", {})) != set(ARMS):
                raise ValueError("report question/arms mismatch or duplicate report")
            if report.get("offset") != positions[qid] or report.get("question_type") != types[qid]:
                raise ValueError("report offset/type conflicts with frozen manifest")
            compact = {
                "question_id": qid,
                "question_type": types[qid],
                "run_id": run_id,
                "offset": positions[qid],
                "report_path": str(report_path),
                "report_index": n,
                "scoring_status": report.get("scoring_status"),
                "arms": {arm: _compact_arm(report["arms"][arm]) for arm in ARMS},
            }
            complete = all(compact["arms"][a]["status"] == "completed" for a in ARMS)
            if report.get("complete_pair") is not complete:
                raise ValueError("complete_pair flag disagrees with arm statuses")
            compact["analysis_status"] = _row_status(compact)
            for arm in ARMS:
                for call in compact["arms"][arm].pop("calls"):
                    trace = call.get("trace_id")
                    if not isinstance(trace, str) or not trace or trace in arm_calls:
                        raise ValueError("duplicate/missing arm call trace")
                    arm_calls[trace] = call
                    owners[trace] = (qid, arm)
            observed[qid] = compact
        if budget_path.exists():
            budget, budget_sha = _read(budget_path)
            calls = budget.get("calls")
            if not isinstance(calls, list):
                raise ValueError("final budget lacks calls")
            authoritative = {call.get("trace_id"): call for call in calls}
            if len(authoritative) != len(calls) or not set(arm_calls) <= authoritative.keys():
                raise ValueError("final ledger has duplicates or omits arm attempts")
            for trace, call in arm_calls.items():
                if call != authoritative[trace]:
                    raise ValueError("arm call differs from final ledger")
        else:
            budget_sha = None
            calls = list(arm_calls.values())
            costs_incomplete.append(run_id)
        for call in calls:
            trace = call.get("trace_id")
            if not isinstance(trace, str) or not trace or trace in calls_by_trace:
                raise ValueError("duplicate/missing canonical call trace across runs")
            calls_by_trace[trace] = call
            if trace not in owners:
                matches = [
                    (qid, arm) for qid in batch_ids for arm in ARMS if f"/{qid}/{arm}/" in trace
                ]
                if len(matches) == 1:
                    owners[trace] = matches[0]
        batches.append(
            {
                "run_id": run_id,
                "start": start,
                "count": count,
                "report_count": len(reports),
                "launch_sha256": launch_sha,
                "reports_sha256": reports_sha,
                "final_budget_sha256": budget_sha,
                "final_budget_present": budget_sha is not None,
            }
        )
    started_by_calls = {qid for qid, _ in owners.values()}
    for qid in ids:
        if qid not in observed:
            status = (
                "unclaimed"
                if qid not in claimed
                else (
                    "started_without_report"
                    if qid in started_by_calls
                    else "claimed_no_execution_record"
                )
            )
            observed[qid] = {
                "question_id": qid,
                "question_type": types[qid],
                "offset": positions[qid],
                "run_id": claimed.get(qid),
                "analysis_status": status,
                "arms": {},
            }
    rows = [observed[qid] for qid in ids]
    scored = [row for row in rows if row["analysis_status"] == "scored_pair"]
    state_counts = dict(Counter(row["analysis_status"] for row in rows))
    arm_execution = {}
    for arm in ARMS:
        available = [row["arms"][arm] for row in rows if arm in row["arms"]]
        rounds = [a["retrieval_rounds"] for a in available if a["retrieval_rounds"] is not None]
        arm_execution[arm] = {
            "status_counts": dict(Counter(a["status"] for a in available)),
            "with_arm_record_n": len(available),
            "without_arm_record_n": len(rows) - len(available),
            "retrieval_rounds_recorded_n": len(rounds),
            "retrieval_rounds_unknown_n": len(available) - len(rounds),
            "retrieval_rounds_distribution": dict(sorted(Counter(rounds).items())),
            "retrieval_rounds_total": sum(rounds),
            "partial_failed_rounds_observed_lower_bound": sum(
                a["partial_retrieval_rounds_observed"]
                for a in available
                if a["retrieval_rounds"] is None
            ),
            "all_attempt_cost": _cost(
                [
                    call
                    for trace, call in calls_by_trace.items()
                    if owners.get(trace, (None, None))[1] == arm
                ]
            ),
        }
    by_type = {}
    for kind in sorted(set(types.values())):
        subset = [row for row in rows if row["question_type"] == kind]
        by_type[kind] = {
            "planned_n": len(subset),
            "status_counts": dict(Counter(r["analysis_status"] for r in subset)),
            **_effect([r for r in subset if r["analysis_status"] == "scored_pair"]),
        }
    diagnostics = {}
    for row in rows:
        for label in _diagnoses(row):
            group = diagnostics.setdefault(label, {"count": 0, "examples": []})
            group["count"] += 1
            if len(group["examples"]) < max_examples:
                group["examples"].append(
                    {
                        k: row.get(k)
                        for k in (
                            "question_id",
                            "question_type",
                            "offset",
                            "run_id",
                            "report_path",
                            "report_index",
                        )
                    }
                )
    return {
        "schema_version": "growrag-shared-s2g-series-analysis-v1",
        "series": series,
        "protocol": protocol,
        "generation_profile": generation_profile,
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest,
        "planned_questions": len(ids),
        "claimed_questions": len(claimed),
        "paired_scored_questions": len(scored),
        "question_status_counts": state_counts,
        "effect": _effect(scored),
        "by_question_type": by_type,
        "execution": arm_execution,
        "all_attempt_cost": _cost(list(calls_by_trace.values())),
        "cost_completeness": {
            "all_matching_batches_have_final_budget": not costs_incomplete,
            "batches_missing_final_budget": costs_incomplete,
            "unattributed_attempts": len(set(calls_by_trace) - owners.keys()),
        },
        "batches": sorted(batches, key=lambda b: b["start"]),
        "verified_unstarted_continuations": continuations,
        "ignored_other_series": ignored,
        "execution_signature": reference_signature,
        "source_snapshot_sha256s": sorted(x for x in source_versions if x is not None),
        "multiple_source_snapshots": len(source_versions) > 1,
        "diagnostic_indices": diagnostics,
        "question_index": rows,
        "notices": [
            "Closed shared train-development corpus; not official test/fullwiki "
            "or trained-LoRA reproduction.",
            "BASE1 vs S2G4 changes retrieval budget and extraction; observed gains are not "
            "memory gains or single-factor causal effects.",
            "EM/F1 use complete scored pairs only. Every planned ID remains "
            "in the status denominator.",
            "No execution record does not prove no execution; missing ledgers imply "
            "incomplete cost lower bounds.",
            "Final-ledger attempts include failures/interruption; historical cumulative "
            "budgets are NOT added again.",
            "Estimates are declared-price costs, not provider-settled bills; "
            "unknown costs are not zero.",
            "Diagnostic labels describe observations, not verified causal explanations "
            "or semantic entailment.",
        ],
    }


def _display(value):
    return (
        "未知/不适用"
        if value is None
        else f"{value:.4f}"
        if isinstance(value, float)
        else str(value)
    )


def render_markdown(report):
    lines = [
        "# S2G 共享语料批次聚合",
        "",
        f"系列：`{report['series']}`；协议：`{report['protocol']}`。",
        "",
        "本报告只读取既有运行记录，没有重新调用模型或读取 gold 文件。",
        "",
        f"计划 {report['planned_questions']} 题；已安排 {report['claimed_questions']} 题；"
        f"两臂完整可评分 {report['paired_scored_questions']} 题。",
        "",
        "## 状态与分母",
        "",
        "| 状态 | 数量 |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {status} | {count} |" for status, count in report["question_status_counts"].items()
    )
    lines += [
        "",
        "## 同一可评分配对集合的效果",
        "",
        "| 方法 | EM（n） | F1（n） | 原文支持召回（n） | 保留支持召回（n） |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        metrics = report["effect"]["arms"][arm]
        values = [f"{_display(metrics[m]['mean'])} ({metrics[m]['n']})" for m in METRICS]
        lines.append(f"| {arm} | " + " | ".join(values) + " |")
    paired = report["effect"]["paired_em"]
    lines += [
        "",
        f"修复 {paired['repairs']} 题；损害 {paired['harms']} 题；"
        f"净差 {paired['net_repaired_questions']} 题。",
        "这不是历史经验净收益，也不是单因素消融。",
        "",
        "## 全部尝试的成本",
        "",
    ]
    cost = report["all_attempt_cost"]
    lines += [
        f"记录 {cost['recorded_attempts']} 次尝试（失败 {cost['failed_attempts']} 次）；"
        f"已知输入/输出 tokens：{cost['known_input_tokens']}/{cost['known_output_tokens']}。",
        f"已知估价 ¥{cost['known_estimated_cny']:.6f}；"
        f"未知费用尝试 {cost['unknown_cost_attempts']} 次。",
        f"缺最终账本批次：{len(report['cost_completeness']['batches_missing_final_budget'])}。",
        "",
        "## 检索轮数",
        "",
    ]
    for arm in ARMS:
        execution = report["execution"][arm]
        lines.append(
            f"- {arm}：{execution['retrieval_rounds_distribution']}；"
            f"有执行记录但轮数未知 {execution['retrieval_rounds_unknown_n']} 个。"
        )
    lines += ["", "## 逐题诊断入口", "", "只表示可观察现象；逐条打开报告后才能提出与验证原因。", ""]
    for label, group in report["diagnostic_indices"].items():
        lines += [f"### {label}：{group['count']} 题", ""]
        for example in group["examples"]:
            link = example.get("report_path")
            qid = example["question_id"]
            lines.append(
                f"- [{qid}](<{Path(link).as_posix()}>)：条目 {example['report_index']}"
                if link
                else f"- {qid}：尚无逐题报告。"
            )
        lines.append("")
    lines += ["## 解读边界", "", *(f"- {notice}" for notice in report["notices"]), ""]
    return "\n".join(lines)


def write_analysis(report, output_dir):
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "analysis.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "analysis.md").open("x", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "runs-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("series", "protocol", "generation-profile"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args(argv)
    report = analyze_series(
        args.manifest,
        args.runs_root,
        series=args.series,
        protocol=args.protocol,
        generation_profile=args.generation_profile,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    output = write_analysis(report, args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                "paired_scored_n": report["paired_scored_questions"],
                "planned": report["planned_questions"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
