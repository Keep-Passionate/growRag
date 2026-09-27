"""Offline ReFormeR migration diagnostics; never calls a model or changes predictions.

中文：同一批题、同一完整配对分母；旧 BASE/S2G 是缓存结果，不重复计入本次花费。
Oracle 只表示事后可选机会，不是已经训练出来的路由器。失败和未知费用不能丢掉。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
BASE, S2G = "BASE1_AUTHOR_READER", "S2G_AUTHOR_API4"
REFORMER = "REFORMER"
EXPECTED_MANIFEST = "2fb117c03b64be3ab12398a297c82ce3315600c14e33f5dcd56c5c6c5c59e064"
EXPECTED_BASELINE = "5b99b0f7f021f8de3b47e92a9c624d441283771566dd15925f37c63886cb5b72"


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def usage(calls):
    """Known subtotals are not full totals if even one request has unknown usage."""
    result = {"recorded_call_count": len(calls)}
    for field in ("api_requests", "reserved_cny"):
        if any(not number(c.get(field)) for c in calls):
            raise ValueError(f"invalid {field}")
        result[field] = sum(c[field] for c in calls)
    for source, label in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("estimated_actual_cny", "estimated_cost_cny"),
    ):
        values = [c.get(source) for c in calls]
        if any(v is not None and not number(v) for v in values):
            raise ValueError(f"invalid {source}")
        known = [v for v in values if v is not None]
        result[f"known_{label}_subtotal"] = sum(known)
        result[f"unknown_{label}_calls"] = len(values) - len(known)
        result[f"total_{label}"] = sum(known) if len(values) == len(known) else None
    return result


def merge_usage(items):
    """Combine already-audited cached per-question costs without charging them again."""
    result = {key: sum(x[key] for x in items) for key in ("api_requests", "reserved_cny")}
    for label in ("input_tokens", "output_tokens", "estimated_cost_cny"):
        known, unknown = f"known_{label}_subtotal", f"unknown_{label}_calls"
        result[known] = sum(x[known] for x in items)
        result[unknown] = sum(x[unknown] for x in items)
        result[f"total_{label}"] = result[known] if not result[unknown] else None
    return result


def call_role(call):
    prompt = call["prompt_version"].lower()
    if "select" in prompt:
        return "pattern_selection"
    if "application" in prompt or "apply" in prompt or "rewrite" in prompt:
        return "pattern_application"
    if "answer" in prompt:
        return "answer"
    raise ValueError("unknown model-call role")


def scoreable(outcome):
    feedback = outcome.get("feedback")
    return (
        outcome.get("status") == "completed"
        and isinstance(feedback, dict)
        and not outcome.get("unscorable_annotation", feedback.get("unscorable_annotation", False))
        and all(number(feedback.get(k)) and feedback[k] <= 1 for k in ("answer_em", "answer_f1"))
    )


def compare(rows, against):
    pairs = [r for r in rows if r["same_complete_case"]]
    left = [r["arms"][against]["feedback"] for r in pairs]
    right = [r["arms"][REFORMER]["feedback"] for r in pairs]
    return {
        "n": len(pairs),
        "em_repairs": sum(
            a["answer_em"] == 0 and b["answer_em"] == 1 for a, b in zip(left, right, strict=True)
        ),
        "em_harms": sum(
            a["answer_em"] == 1 and b["answer_em"] == 0 for a, b in zip(left, right, strict=True)
        ),
        "f1_improved": sum(
            b["answer_f1"] > a["answer_f1"] for a, b in zip(left, right, strict=True)
        ),
        "f1_worsened": sum(
            b["answer_f1"] < a["answer_f1"] for a, b in zip(left, right, strict=True)
        ),
        "mean_f1_delta": mean(
            b["answer_f1"] - a["answer_f1"] for a, b in zip(left, right, strict=True)
        )
        if pairs
        else None,
    }


def paired_resources(paired, arm):
    """Same score cohort, not all-attempt totals or cross-run deployment benchmarks."""
    observed = [row["new_usage"] if arm == REFORMER else row["cached_usage"][arm] for row in paired]
    available = [item for item in observed if item is not None]
    durations = [row["arms"][arm].get("elapsed_seconds") for row in paired]
    if any(value is not None and not number(value) for value in durations):
        raise ValueError("invalid observed arm elapsed time")
    known = [value for value in durations if value is not None]
    return {
        "n": len(paired),
        "usage_observed_n": len(available),
        "usage_missing_n": len(paired) - len(available),
        "usage": merge_usage(available) if len(available) == len(paired) and paired else None,
        "elapsed_seconds": {
            "known_n": len(known),
            "missing_n": len(paired) - len(known),
            "known_subtotal": sum(known),
            "total": sum(known) if len(known) == len(paired) and paired else None,
            "mean": mean(known) if len(known) == len(paired) and paired else None,
        },
    }


def assemble(expected_ids, baseline_questions, baseline_cost_rows, batches):
    """Pure analysis projection: tests need no files, API keys or author repository."""
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate expected question")
    baseline = {x["question_id"]: x for x in baseline_questions}
    cached_cost = {x["question_id"]: x for x in baseline_cost_rows}
    if len(baseline) != len(baseline_questions) or len(cached_cost) != len(baseline_cost_rows):
        raise ValueError("duplicate cached question")
    if not set(expected_ids) <= baseline.keys() or not set(expected_ids) <= cached_cost.keys():
        raise ValueError("cached cohort incomplete")
    reports, all_calls, trace_ids = {}, [], set()
    for batch in batches:
        ledger = batch["budget"]
        canonical = {c["trace_id"]: c for c in ledger["calls"]}
        if len(canonical) != len(ledger["calls"]):
            raise ValueError("duplicate ledger trace")
        reported = {}
        for report in batch["reports"]:
            qid, offset = report["question_id"], report["offset"]
            if (
                type(offset) is not int
                or not 0 <= offset < len(expected_ids)
                or expected_ids[offset] != qid
                or qid in reports
            ):
                raise ValueError("question ID/offset mismatch or repeated execution")
            arm_name, outcome = "REFORMER_PUBLIC_QWEN", report["outcome"]
            for call in outcome.get("calls", []):
                trace = call["trace_id"]
                if trace in reported or trace in trace_ids:
                    raise ValueError("duplicate call ownership")
                parts = trace.split("/")
                if parts[:3] != [batch["run_id"], qid, arm_name]:
                    raise ValueError("call outside its question/arm")
                reported[trace] = call
                call_role(call)
            reports[qid] = (report, outcome, batch["run_id"])
        if reported != canonical:
            raise ValueError("reports do not own every ledger call exactly once")
        totals = usage(list(canonical.values()))
        for key in ("api_requests", "reserved_cny"):
            if not math.isclose(totals[key], ledger[key], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("root request ledger sum mismatch")
        trace_ids.update(canonical)
        all_calls.extend(canonical.values())

    rows = []
    for offset, qid in enumerate(expected_ids):
        old = baseline[qid]
        record, outcome, run_id = reports.get(qid, ({}, {"status": "not_executed"}, None))
        arms = {name: old["arms"][name] for name in (BASE, S2G)}
        arms[REFORMER] = outcome
        result = outcome.get("result", {})
        calls = outcome.get("calls", [])
        selected = result.get("selected_pattern") or {}
        original = record.get("question", result.get("question", ""))
        rewritten = result.get("rewritten_query")
        current = {
            "question_id": qid,
            "offset": offset,
            "question_type": record.get("question_type", old.get("question_type")),
            "question": record.get("question", result.get("question")),
            "run_id": run_id,
            "source_report": record.get("_report_path"),
            "cached_report": old.get("report_path"),
            "same_complete_case": all(scoreable(a) for a in arms.values()),
            "arms": arms,
            "new_usage": usage(calls) if outcome["status"] != "not_executed" else None,
            "new_roles": {
                role: usage([c for c in calls if call_role(c) == role])
                for role in ("pattern_selection", "pattern_application", "answer")
            },
            "cached_usage": {
                name: cached_cost[qid]["arms"][name].get("usage") for name in (BASE, S2G)
            },
            "rewrite_diagnostics": {
                "selected_pattern": selected.get("pattern_name"),
                "canonical_library_match": result.get("canonical_library_match"),
                "author_fallback_stages": result.get("author_fallback_stages", []),
                "rewritten_query": rewritten,
                "retrieval_query": result.get("retrieval_query"),
                "rewrite_equals_original": rewritten.strip() == original.strip()
                if isinstance(rewritten, str)
                else None,
                "original_characters": len(original) if original else None,
                "rewrite_characters": len(rewritten) if isinstance(rewritten, str) else None,
                "retrieval_query_characters": len(result["retrieval_query"])
                if isinstance(result.get("retrieval_query"), str)
                else None,
                "initial3_support_recall": outcome.get("initial_feedback", {}).get(
                    "raw_support_recall"
                ),
                "final6_support_recall": (outcome.get("feedback") or {}).get("raw_support_recall"),
                "base6_support_recall": (arms[BASE].get("feedback") or {}).get(
                    "raw_support_recall"
                ),
            },
        }
        rows.append(current)
    paired = [r for r in rows if r["same_complete_case"]]
    method_scores = {
        name: {
            key: mean(r["arms"][name]["feedback"][key] for r in paired) if paired else None
            for key in ("answer_em", "answer_f1")
        }
        for name in (BASE, S2G, REFORMER)
    }
    completed = [r for r in rows if r["arms"][REFORMER]["status"] == "completed"]
    diagnostics = [r["rewrite_diagnostics"] for r in completed]
    library_known = [d for d in diagnostics if type(d["canonical_library_match"]) is bool]
    evidence = [
        r["rewrite_diagnostics"]
        for r in paired
        if all(
            number(r["rewrite_diagnostics"][key])
            for key in ("initial3_support_recall", "final6_support_recall", "base6_support_recall")
        )
    ]
    oracle = {"n": len(paired)}
    for metric in ("em", "f1"):
        old_best = [
            max(r["arms"][a]["feedback"][f"answer_{metric}"] for a in (BASE, S2G)) for r in paired
        ]
        new_best = [
            max(old, row["arms"][REFORMER]["feedback"][f"answer_{metric}"])
            for old, row in zip(old_best, paired, strict=True)
        ]
        oracle[f"base_s2g_{metric}"] = mean(old_best) if paired else None
        oracle[f"three_arm_{metric}"] = mean(new_best) if paired else None
        oracle[f"extra_over_base_s2g_{metric}"] = (
            mean(new - old for new, old in zip(new_best, old_best, strict=True)) if paired else None
        )
    oracle["reformer_repairs_both_others_wrong"] = sum(
        r["arms"][REFORMER]["feedback"]["answer_em"] == 1
        and all(r["arms"][a]["feedback"]["answer_em"] == 0 for a in (BASE, S2G))
        for r in paired
    )
    summary = {
        "planned_questions": len(expected_ids),
        "new_status_counts": dict(Counter(r["arms"][REFORMER]["status"] for r in rows)),
        "same_complete_case_n": len(paired),
        "paired_scores": method_scores,
        "paired_vs_base": compare(rows, BASE),
        "paired_vs_s2g": compare(rows, S2G),
        "new_api_usage_all_attempts": usage(all_calls),
        "new_role_usage_all_attempts": {
            role: usage([c for c in all_calls if call_role(c) == role])
            for role in ("pattern_selection", "pattern_application", "answer")
        },
        "cached_usage_all_selected_questions": {
            name: merge_usage(
                [r["cached_usage"][name] for r in rows if r["cached_usage"][name] is not None]
            )
            for name in (BASE, S2G)
        },
        "same_complete_case_online_resources": {
            "n": len(paired),
            "arms": {arm: paired_resources(paired, arm) for arm in (BASE, S2G, REFORMER)},
            "notice": "Online inference estimates exclude historical pattern induction. "
            "Old arms are cached, not new charges. Elapsed times are observed serialized "
            "arm durations across runs, not a controlled deployment latency benchmark. "
            "All-attempt costs, including failures excluded here, remain reported separately.",
        },
        "oracle_opportunity_not_a_router": oracle,
        "completed_new_questions": len(completed),
        "rewrite_diagnostics": {
            "selected_pattern_counts": dict(
                Counter(d["selected_pattern"] or "unknown" for d in diagnostics)
            ),
            "library_match_known_n": len(library_known),
            "library_object_mismatch_n": sum(
                not d["canonical_library_match"] for d in library_known
            ),
            "library_object_mismatch_rate": (
                mean(not d["canonical_library_match"] for d in library_known)
                if library_known
                else None
            ),
            "library_object_mismatch_notice": "Strict full-object inequality can mean omitted "
            "examples or fields, not a changed rule or semantic drift. The source library "
            "is never updated by this baseline; inspect field-level differences separately.",
            "repeat_rewrites": sum(d["rewrite_equals_original"] is True for d in diagnostics),
            "fallback_questions": sum(bool(d["author_fallback_stages"]) for d in diagnostics),
            "mean_characters": {
                key: mean(d[key] for d in diagnostics if d[key] is not None)
                if any(d[key] is not None for d in diagnostics)
                else None
                for key in (
                    "original_characters",
                    "rewrite_characters",
                    "retrieval_query_characters",
                )
            },
        },
        "exact_support_coverage": {
            "n": len(evidence),
            **{
                key: mean(d[key] for d in evidence) if evidence else None
                for key in (
                    "initial3_support_recall",
                    "final6_support_recall",
                    "base6_support_recall",
                )
            },
            "final6_improved_vs_base6": sum(
                d["final6_support_recall"] > d["base6_support_recall"] for d in evidence
            ),
            "final6_worsened_vs_base6": sum(
                d["final6_support_recall"] < d["base6_support_recall"] for d in evidence
            ),
            "notice": "Initial selector sees 3 docs, final answer sees 6: their change confounds "
            "context size. Compare BASE6 vs final6 for equal document counts; "
            "neither is an entailment judgment.",
        },
    }
    return rows, summary


def analyze(runs_root, manifest_path, output, *, count=100):
    """Read sealed runs only; all source hashes recorded, output cannot overwrite."""
    runs_root, output = Path(runs_root).resolve(), Path(output).resolve()
    if output == runs_root or not output.is_relative_to(runs_root):
        raise ValueError("output must be a new directory inside runs root")
    if output.exists():
        raise FileExistsError(output)
    hashes = {}

    def read(path, expected_hash=None):
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_hash is not None and digest != expected_hash:
            raise ValueError("frozen source hash mismatch")
        hashes[str(Path(path).resolve())] = digest
        return json.loads(raw)

    manifest = read(manifest_path, EXPECTED_MANIFEST)
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("only the first frozen 100-question cohort is allowed")
    baseline = read(
        runs_root / "2026-09-27_shared500_analysis_final_v1/analysis.json", EXPECTED_BASELINE
    )
    cost_path = runs_root / "2026-09-27_shared500_cost_export_v1/per_question_costs.jsonl"
    raw_cost = cost_path.read_bytes()
    hashes[str(cost_path)] = hashlib.sha256(raw_cost).hexdigest()
    costs = [json.loads(line) for line in raw_cost.splitlines() if line.strip()]
    batches = []
    for directory in sorted(runs_root.glob("2026-09-27_reformer_hotpot_v1_*")):
        if not directory.is_dir():
            continue
        if not all(
            (directory / f).is_file()
            for f in ("reports.json", "final_budget.json", "launch_plan.json", "events.jsonl")
        ):
            raise ValueError(f"unsealed ReFormeR run: {directory.name}")
        launch = read(directory / "launch_plan.json")
        if (
            launch.get("manifest_sha256") != EXPECTED_MANIFEST
            or launch.get("protocol") != "growrag-reformer-public-qwen-v1"
        ):
            raise ValueError("unexpected ReFormeR execution protocol")
        event_raw = (directory / "events.jsonl").read_bytes()
        hashes[str(directory / "events.jsonl")] = hashlib.sha256(event_raw).hexdigest()
        events = [json.loads(line) for line in event_raw.splitlines() if line.strip()]
        if not events or events[-1].get("kind") != "exit":
            raise ValueError("ReFormeR execution has not terminated")
        reports = read(directory / "reports.json")
        for row in reports:
            row["_report_path"] = str(directory / "reports.json")
        batches.append(
            {
                "run_id": directory.name,
                "reports": reports,
                "budget": read(directory / "final_budget.json"),
            }
        )
    rows, summary = assemble(
        manifest["question_ids"][:count], baseline["question_index"], costs, batches
    )
    summary.update(
        schema_version="growrag-reformer-pilot-analysis-v2",
        created_utc=datetime.now(UTC).isoformat(),
        source_sha256=hashes,
    )
    summary["limits"] = [
        "Development migration on a fixed shared corpus, "
        "not an untouched test or original-paper reproduction.",
        "BASE/S2G predictions are cached; only ReFormeR calls are new charges. "
        "All failures remain visible.",
        "Same-complete-case comparisons exclude failures; "
        "missing answers are not silently counted as wrong.",
        "Oracle uses gold after execution and is an opportunity upper bound, "
        "never a learned online router.",
        "EM/F1 and exact evidence coverage do not establish semantic faithfulness or causality.",
    ]
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "per_question.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    write_markdown(output, rows, summary)
    return summary


def write_markdown(output, rows, summary):
    """Readable complete prediction review plus first-by-offset successes/failures."""
    lines = [
        "# ReFormeR 迁移小测：完整配对与费用",
        "",
        f"计划{summary['planned_questions']}题，三路均完成且可评分{summary['same_complete_case_n']}题。",
        "",
        "这里对比的是同一批开发题上的系统；BASE/S2G使用原先已冻结的缓存，ReFormeR为本次新调用。不是未见测试集成绩，也不是等预算的记忆消融。",
        "",
        "| 方法 | EM | F1 |",
        "| --- | ---: | ---: |",
    ]
    for arm, values in summary["paired_scores"].items():
        em, f1 = values["answer_em"], values["answer_f1"]
        lines.append(
            f"| {arm} | {em:.4f} | {f1:.4f} |"
            if em is not None
            else f"| {arm} | 未形成配对 | 未形成配对 |"
        )
    lines.extend(
        [
            "",
            f"新运行状态：{summary['new_status_counts']}。所有失败调用计入费用；未知费用不按零处理。",
            "",
            "| 新增API角色 | 请求数 | 已知估价小计¥ | 未知费用调用 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for role, values in summary["new_role_usage_all_attempts"].items():
        lines.append(
            f"| {role} | {values['api_requests']} | "
            f"{values['known_estimated_cost_cny_subtotal']:.8f} | "
            f"{values['unknown_estimated_cost_cny_calls']} |"
        )
    lines.extend(
        [
            "",
            "旧BASE/S2G成本单列在analysis.json的cached_usage_all_selected_questions，不算本次重复花费。",
            "",
            "## 同一完整配对的在线资源",
            "",
            "下表与EM/F1使用同一批完整题；不含离线模式建库。被排除的失败调用仍计入上面的全尝试费用。",
            "",
            "| 方法 | 完整配对请求数 | 完整配对估价¥ | 平均每题观测耗时秒 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm, resources in summary["same_complete_case_online_resources"]["arms"].items():
        u = resources["usage"] or {}
        lines.append(
            f"| {arm} | {u.get('api_requests')} | {u.get('total_estimated_cost_cny')} | "
            f"{resources['elapsed_seconds']['mean']} |"
        )
    lines.extend(
        [
            "",
            "耗时是跨批次实测轨迹用时，包含检索/API等开销，不是控制了缓存与网络条件的部署延迟基准。",
            "",
            "## 可以检查的差异",
            "",
        ]
    )
    for other in (BASE, S2G):
        values = summary["paired_vs_base" if other == BASE else "paired_vs_s2g"]
        lines.append(
            f"ReFormeR相对{other}：修复{values['em_repairs']}题，损害{values['em_harms']}题（同一完整配对分母）。"
        )
        for label, old_em, new_em in (("修复", 0, 1), ("损害", 1, 0)):
            cases = [
                r
                for r in rows
                if r["same_complete_case"]
                and r["arms"][other]["feedback"]["answer_em"] == old_em
                and r["arms"][REFORMER]["feedback"]["answer_em"] == new_em
            ]
            refs = [f"[{r['offset']:04d}](QUESTIONS.md#q{r['offset']:04d})" for r in cases[:5]]
            lines.append(f"{label}案例（仅按固定offset取前5）：" + ("、".join(refs) or "无"))
        lines.append("")
    oracle = summary["oracle_opportunity_not_a_router"]
    mismatch = summary["rewrite_diagnostics"]
    lines.extend(
        [
            "Oracle事后用gold挑最好者，只显示存在多少机会，不代表已经可以在线选中；不能作为方法分数。",
            "",
            f"原BASE/S2G Oracle：EM={oracle['base_s2g_em']}，F1={oracle['base_s2g_f1']}；"
            f"加入ReFormeR后额外机会：EM={oracle['extra_over_base_s2g_em']}，"
            f"F1={oracle['extra_over_base_s2g_f1']}。",
            "",
            f"选择结果与原模式库完整对象不一致：{mismatch['library_object_mismatch_n']} / "
            f"{mismatch['library_match_known_n']}。这可能仅是漏掉示例或字段，"
            "不能据此声称规则被改写、记忆被污染或语义发生漂移。原模式库没有被更新。",
            "",
            "初始选择器只有3篇，最终回答器有6篇，因此初始→最终证据覆盖变化不能全部归功于改写。另看BASE6→最终6对照。",
            "",
            "全部题目的原问题、模式、改写、答案、分数与原始日志链接见[QUESTIONS.md](QUESTIONS.md)。逐题token和角色成本见per_question.jsonl。",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    questions = [
        "# 全部问题逐题审查",
        "",
        "未执行、失败和完整题均保留；旧结果不传给当前选择器。以下gold反馈只用于运行结束后的离线审查。",
        "",
    ]
    for row in rows:
        diag = row["rewrite_diagnostics"]
        questions.extend(
            [
                f'<a id="q{row["offset"]:04d}"></a>',
                f"## {row['offset']:04d} · {row['question_id']}",
                "",
                row["question"] or "原问题请见缓存报告",
                "",
                f"模式：{diag['selected_pattern']}；与作者库完整对象原样匹配："
                f"{diag['canonical_library_match']}（字段省略也会不匹配，不等于规则漂移）。",
                "",
                f"改写：{diag['rewritten_query']}",
                "",
                f"实际检索query：{diag['retrieval_query']}",
                "",
                "| 方法 | 状态 | 答案 | EM | F1 |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        for arm, outcome in row["arms"].items():
            feedback = outcome.get("feedback") or {}
            answer = outcome.get("answer", outcome.get("result", {}).get("answer"))
            safe_answer = str(answer).replace("|", "\\|").replace("\n", " ")
            questions.append(
                f"| {arm} | {outcome['status']} | {safe_answer} | "
                f"{feedback.get('answer_em')} | {feedback.get('answer_f1')} |"
            )
        questions.extend(
            [
                "",
                f"证据覆盖：初始3篇={diag['initial3_support_recall']}，最终6篇={diag['final6_support_recall']}，BASE6篇={diag['base6_support_recall']}。",
                "",
            ]
        )
        for label, key in (
            ("当前完整预测/费用", "source_report"),
            ("旧BASE/S2G完整轨迹", "cached_report"),
        ):
            if row[key]:
                path = Path(row[key]).resolve().as_posix()
                questions.append(f"[{label}](<{path}>)")
        questions.append("")
    (output / "QUESTIONS.md").write_text("\n".join(questions) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/hotpotqa/shared500_sep27_v1/manifest.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    result = analyze(args.runs_root, args.manifest, args.output, count=args.count)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "new_status_counts",
                    "same_complete_case_n",
                    "paired_scores",
                    "new_api_usage_all_attempts",
                )
            },
            indent=2,
        )
    )
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
