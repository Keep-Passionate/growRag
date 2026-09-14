"""Frozen no-history baseline comparison, not a deployed winner-selection policy.

中文导读：各方法独立运行；先记录真实 query/证据/回答，再读取 gold 评分。
所有方法都付费执行是离线对照成本，不能冒充线上只执行一次的成本。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter

from growrag.outer_loop import RetrieverReaderBackend, run_outer_loop
from growrag.query_actions import RewriteDecision

from .fresh_baselines import BASELINE_SPECS, baseline_generator
from .hotpot import HotpotExample, evaluate_layered_feedback
from .llm_adapters import READER_PROMPT_VERSION, APIReader, _execution_kind
from .pre_pilot import _IndexAdapter, write_json
from .protocol import Action, ExecutionKind

VARIANTS = ("BASE", "SIMPLE_PARAPHRASE", "RRR_KEYWORDS", "QUERY2DOC")


def call_totals(calls: list[dict]) -> dict:
    """Missing usage stays unknown, rather than turning into zero cost."""
    result = {"recorded_attempts": len(calls)}
    for field in ("api_requests", "input_tokens", "output_tokens", "estimated_actual_cny"):
        result[field] = (
            sum(row[field] for row in calls)
            if all(row.get(field) is not None for row in calls)
            else None
        )
    return result


def _method_summaries(rows: list[dict]) -> dict:
    """Apply identical metric/cost definitions to an explicitly supplied cohort."""
    methods = {}
    for name in VARIANTS:
        observed = [row["arms"][name] for row in rows if name in row["arms"]]
        scored = [arm for arm in observed if arm["feedback"] is not None]
        paired = [
            row
            for row in rows
            if name in row["arms"]
            and "BASE" in row["arms"]
            and row["arms"][name]["feedback"] is not None
            and row["arms"]["BASE"]["feedback"] is not None
        ]
        methods[name] = {
            "attempted_questions": len(observed),
            "scored_questions": len(scored),
            "mean_em": mean(a["feedback"]["answer_em"] for a in scored) if scored else None,
            "mean_f1": mean(a["feedback"]["answer_f1"] for a in scored) if scored else None,
            "mean_support_recall": (
                mean(a["feedback"]["retrieved_gold_support_recall"] for a in scored)
                if scored
                and all(a["feedback"]["retrieved_gold_support_recall"] is not None for a in scored)
                else None
            ),
            "paired_with_base_count": len(paired),
            "rescues_vs_base": sum(
                row["arms"]["BASE"]["feedback"]["answer_em"] == 0
                and row["arms"][name]["feedback"]["answer_em"] == 1
                for row in paired
            ),
            "harms_vs_base": sum(
                row["arms"]["BASE"]["feedback"]["answer_em"] == 1
                and row["arms"][name]["feedback"]["answer_em"] == 0
                for row in paired
            ),
            "mean_measured_seconds": (
                mean(a["wall_seconds"] for a in observed) if observed else None
            ),
            "cost": call_totals([c for a in observed for c in a["model_calls"]]),
            "stage_costs": {
                "reader": call_totals(
                    [
                        c
                        for a in observed
                        for c in a["model_calls"]
                        if c["prompt_version"] == READER_PROMPT_VERSION
                    ]
                ),
                "query_generation": call_totals(
                    [
                        c
                        for a in observed
                        for c in a["model_calls"]
                        if c["prompt_version"] != READER_PROMPT_VERSION
                    ]
                ),
            },
            "mean_index_build_seconds": (
                mean(a["index_build_seconds"] for a in observed) if observed else None
            ),
            "mean_retrieval_seconds": (
                mean(
                    sum(
                        e["elapsed_seconds"]
                        for e in a["result"]["component_events"]
                        if e["operation"] == "rag.retrieve"
                    )
                    for a in observed
                )
                if observed
                else None
            ),
        }
    return methods


def summarize(rows: list[dict], planned_count: int) -> dict:
    """Retain all attempts, and separately compare only the shared scored cohort.

    中文：原始 methods 保留失败和不完整题的费用；横向比较效果时使用
    complete_case_methods，即四条路线都获得有效评分的同一批题。不能把
    12 题的 BASE 均值与 11 题的其他方法均值直接比较。该交集不是盲测集，
    中断导致被排除的问题与未知账单仍必须报告，不能通过筛选掩盖失败。
    """
    complete_rows = [
        row
        for row in rows
        if all(
            name in row["arms"] and row["arms"][name]["feedback"] is not None for name in VARIANTS
        )
    ]
    return {
        "schema_version": "growrag-fresh-benchmark-v1",
        "planned_questions": planned_count,
        "completed_questions": len(complete_rows),
        "methods": _method_summaries(rows),
        "complete_case_count": len(complete_rows),
        "complete_case_question_ids": [row["question_id"] for row in complete_rows],
        "complete_case_methods": _method_summaries(complete_rows),
        "complete_case_notice": "Four-arm non-null scored intersection only; all methods use "
        "the same questions for effects, paired rescues/harms, costs and latency. "
        "Excluded partial/failed attempts remain in methods and the total budget ledger. "
        "Complete-case cost is not total experiment spend or proof that missingness is random.",
        "no_history": True,
        "qpp_executed": False,
        "memory_updated": False,
        "parameter_training": False,
        "notice": "Balanced train diagnostic, not official test or a trained selector. "
        "Aggregate all-arm audit cost is not a one-winner deployment cost. "
        "Latency is measured serial API time with network noise, not a hardware benchmark.",
    }


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def question_markdown(row: dict) -> str:
    lines = [
        "# FRESH 基线逐题记录",
        "",
        f"原问题：{row['question']}",
        "",
        "本题不调用历史记忆，不运行 QPP；四条路线独立执行，gold 只在执行后评分。",
        "伪文档只参与构造搜索 query，不作为回答证据。",
        "",
        "| 方法 | 实际 query | 答案 | EM | F1 | 支持召回 | API 次数 | 估算元 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, arm in row["arms"].items():
        feedback, cost = arm["feedback"] or {}, arm["cost"]
        lines.append(
            "| "
            + " | ".join(
                _cell(v)
                for v in (
                    name,
                    arm["query"],
                    arm["answer"],
                    feedback.get("answer_em", "未知"),
                    feedback.get("answer_f1", "未知"),
                    feedback.get("retrieved_gold_support_recall", "未知"),
                    cost["api_requests"],
                    cost["estimated_actual_cny"],
                )
            )
            + " |"
        )
    for name, arm in row["arms"].items():
        lines += [
            "",
            f"## {name}",
            "",
            f"停止原因：{arm['stop_reason']}",
            f"实测分支时间：{arm['wall_seconds']:.3f} 秒；已包含索引时间，不再重复相加。",
            "",
        ]
        for evidence in arm["evidence"]:
            lines.append(f"- {evidence['title']}，句 {evidence['sentence_id']}：{evidence['text']}")
        lines += ["", "原始调用、prompt 版本与 tokens 见同目录分支 JSON 的 model_calls。"]
    return "\n".join(lines) + "\n"


def run_fresh_benchmark(
    examples: tuple[HotpotExample, ...], client, output: Path, *, allow_real: bool = False
) -> dict:
    """No retries/resume/memory mutation. Abort further calls after any invalid arm."""
    kind = _execution_kind(client)
    if kind is ExecutionKind.REAL and allow_real is not True:
        raise ValueError("real benchmark requires explicit allow_real")
    if not examples or len({e.question.question_id for e in examples}) != len(examples):
        raise ValueError("benchmark requires nonempty, unique questions")
    if any(e.gold is None for e in examples):
        raise ValueError("offline diagnostic requires post-run labels")
    output = Path(output)
    # A launch plan may already exist, but never append to an old question run.
    (output / "questions").mkdir(parents=True, exist_ok=False)
    rows = []
    for index, example in enumerate(examples):
        directory = output / "questions" / f"{index:03d}"
        directory.mkdir()
        row = {
            "question_id": example.question.question_id,
            "question": example.question.text,
            "arms": {},
        }
        order = sorted(
            VARIANTS,
            key=lambda name: hashlib.sha256(
                f"fresh-benchmark-v1:42:{example.question.question_id}:{name}".encode()
            ).hexdigest(),
        )
        write_json(
            directory / "plan.json",
            {
                "question_id": example.question.question_id,
                "order": order,
                "variants": {name: asdict(spec) for name, spec in BASELINE_SPECS.items()},
                "gold_visible_to_runtime": False,
                "top_k": 4,
                "max_rag_calls_per_arm": 1,
            },
        )
        for name in order:
            start, call_start = perf_counter(), len(client.calls)
            index_start = perf_counter()
            backend = RetrieverReaderBackend(
                _IndexAdapter(example.candidate_context, kind), APIReader(client), top_k=4
            )
            index_seconds = perf_counter() - index_start
            if name == "BASE":
                decision, generator = RewriteDecision(), None
            else:
                spec = BASELINE_SPECS[name]
                decision = RewriteDecision(Action.FRESH, spec.form, spec.intent)
                generator = baseline_generator(client, name)
            result = run_outer_loop(
                example.question,
                backend,
                generator=generator,
                policy=lambda state, chosen=decision: chosen,
                max_rag_calls=1,
            )
            calls = list(client.calls[call_start:])
            arm = {
                "result": asdict(result),
                "model_calls": calls,
                "wall_seconds": perf_counter() - start,
                "index_build_seconds": index_seconds,
                "cost": call_totals(calls),
                "query": None,
                "answer": None,
                "evidence": [],
                "feedback": None,
                "stop_reason": result.stop_reason,
            }
            # Persist the unscored execution BEFORE touching the offline gold.
            write_json(directory / f"{name}_execution.json", arm)
            if result.state.rounds:
                final = result.state.rounds[-1]
                arm.update(
                    query=final.search_query,
                    answer=final.reply.answer.text,
                    evidence=[asdict(e) for e in final.reply.evidence or ()],
                )
                if final.reply.evidence is not None:
                    arm["feedback"] = asdict(
                        evaluate_layered_feedback(
                            example.question,
                            (),
                            final.reply.evidence,
                            final.reply.answer,
                            example.gold,
                        )
                    )
            write_json(directory / f"{name}_scored.json", arm)
            row["arms"][name] = arm
            if arm["feedback"] is None or client.block_reason:
                rows.append(row)
                write_json(output / "observations.json", rows)
                write_json(output / "summary.json", summarize(rows, len(examples)))
                raise RuntimeError("baseline arm incomplete; retained audit, no retry")
        rows.append(row)
        write_json(directory / "row.json", row)
        (directory / "trace.md").write_text(question_markdown(row), encoding="utf-8")
        write_json(directory / "budget_checkpoint.json", client.report())
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "planned": len(examples),
                    "api_requests": client.attempts,
                    "estimated_cny": client.report()["estimated_actual_cny"],
                }
            ),
            flush=True,
        )
    report = summarize(rows, len(examples))
    write_json(output / "observations.json", rows)
    write_json(output / "summary.json", report)
    return report
