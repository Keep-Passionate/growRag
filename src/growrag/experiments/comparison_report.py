"""Post-run gold evaluation and exclusive artifacts, never a runtime selector."""

from __future__ import annotations

import json
from dataclasses import asdict
from html import escape
from pathlib import Path

from growrag.experiments.data_protocol import normalize_question
from growrag.experiments.hotpot import evaluate_layered_feedback
from growrag.experiments.protocol import Action, GoldRecord, Usage
from growrag.experiments.query_comparison import QueryComparison


def _sum_usage(values: tuple[Usage, ...]) -> dict:
    result = {}
    for name in ("input_tokens", "output_tokens", "api_requests"):
        observed = [getattr(value, name) for value in values]
        result[name] = None if None in observed else sum(observed)
    return result


def comparison_report(run: QueryComparison, *, gold: GoldRecord | None = None) -> dict:
    if not isinstance(run, QueryComparison):
        raise TypeError("a completed QueryComparison record is required")
    if gold is not None and (
        not isinstance(gold, GoldRecord) or gold.question_id != run.spec.question.question_id
    ):
        raise ValueError("gold must belong to the compared question")
    rows = {}
    all_usage = []
    for arm in run.arms:
        result = arm.result
        failed = any(event.status == "error" for event in result.events)
        last = result.state.rounds[-1] if result.state.rounds else None
        completed = last is not None and not failed
        usage = tuple(event.usage for event in result.events)
        all_usage.extend(usage)
        row = {
            "planned_action": arm.planned_action.value,
            "executed_action": last.decision.action.value if last else None,
            "completed": completed,
            "stop_reason": result.stop_reason,
            "query": last.search_query if last else None,
            "query_changed": (
                normalize_question(last.search_query) != normalize_question(run.spec.question.text)
                if last
                else None
            ),
            "answer": last.reply.answer.text if last else None,
            "abstained": not last.reply.answer.text.strip() if completed else None,
            "evidence": (
                [asdict(item) for item in last.reply.evidence]
                if last and last.reply.evidence is not None
                else None
            ),
            "usage": _sum_usage(usage),
            "rag_calls": sum(event.operation == "rag" for event in result.events),
            "rewrite_calls": sum(event.operation == "rewrite" for event in result.events),
            "elapsed_seconds": sum(event.elapsed_seconds for event in result.events),
            "feedback": None,
        }
        if gold is not None and completed:
            feedback = asdict(
                evaluate_layered_feedback(
                    run.spec.question, (), last.reply.evidence or (), last.reply.answer, gold
                )
            )
            if last.reply.evidence is None:
                for name in (
                    "retrieved_gold_support_recall",
                    "new_gold_support",
                    "cited_gold_support_precision",
                    "cited_gold_support_recall",
                    "citation_ids_resolve",
                ):
                    feedback[name] = None
            row["feedback"] = feedback
        rows[arm.planned_action.value] = row
    contrasts = {}
    for action, reference in (
        (Action.REUSE, Action.BASE),
        (Action.REUSE, Action.FRESH),
        (Action.FRESH, Action.BASE),
    ):
        treatment, baseline = rows[action.value], rows[reference.value]
        t, b = treatment["feedback"], baseline["feedback"]
        distinct_queries = (
            normalize_question(treatment["query"]) != normalize_question(baseline["query"])
            if treatment["completed"] and baseline["completed"]
            else None
        )
        contrasts[f"{action.value}_minus_{reference.value}"] = {
            "available": t is not None and b is not None,
            "different_executed_queries": distinct_queries,
            "interpretation": (
                "Same query: answer variation is not evidence of a query-change benefit."
                if distinct_queries is False
                else "Descriptive comparison, not causal credit."
            ),
            "answer_em_delta": t["answer_em"] - b["answer_em"] if t and b else None,
            "answer_f1_delta": t["answer_f1"] - b["answer_f1"] if t and b else None,
            "support_recall_delta": (
                t["retrieved_gold_support_recall"] - b["retrieved_gold_support_recall"]
                if t
                and b
                and t["retrieved_gold_support_recall"] is not None
                and b["retrieved_gold_support_recall"] is not None
                else None
            ),
            "correct_to_wrong_answer": (
                b["answer_em"] == 1 and t["answer_em"] == 0 and not treatment["abstained"]
                if t and b
                else None
            ),
            "correct_to_abstention": (
                b["answer_em"] == 1 and treatment["abstained"] if t and b else None
            ),
        }
    return {
        "schema_version": "growrag-query-comparison-report-v1",
        "spec_fingerprint": run.spec.fingerprint,
        "execution_kind": run.execution_kind.value,
        "synthetic_demo": run.execution_kind.value == "mock",
        "notice": "Paired descriptive observations, not proof of causality or model effectiveness.",
        "gold_available": gold is not None,
        "completed_arm_count": sum(row["completed"] for row in rows.values()),
        "uncompleted_actions": [key for key, row in rows.items() if not row["completed"]],
        "arms": rows,
        "contrasts": contrasts,
        "total_experiment_usage": _sum_usage(tuple(all_usage)),
        "cost_note": "Total sums all three executed arms, not a deployed single route. "
        "Component detail events are NOT added again to aggregate RAG events. "
        "Timing excludes factory construction, index building and memory construction.",
    }


def render_comparison_markdown(run: QueryComparison, report: dict) -> str:
    """Human-readable inspection; model-provided text is escaped, not trusted markup."""

    def cell(value: object) -> str:
        if value is None:
            return "未知／不可评价"
        if type(value) is bool:
            return "是" if value else "否"
        if isinstance(value, float):
            return f"{value:.3f}"
        rendered = escape(str(value))
        for special in "\\`*_{}[]()!":
            rendered = rendered.replace(special, "\\" + special)
        return rendered.replace("|", "&#124;").replace("\n", "<br>")

    notice = (
        "手写测试替身演示：不是实际模型表现，不可用作论文分数。"
        if report["synthetic_demo"]
        else "实际适配器运行记录；单题观察不能证明研究效果。"
    )
    lines = [
        "# 单题三分支比较",
        "",
        notice,
        "",
        f"原问题：{cell(run.spec.question.text)}",
        "",
        "| 分支 | 实际动作 | 改变查询 | 完成 | 答案 | 答案 EM | 支持召回 | API 次数 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for action in Action:
        row = report["arms"][action.value]
        feedback = row["feedback"] or {}
        values = (
            action.value,
            row["executed_action"],
            row["query_changed"],
            row["completed"],
            row["answer"],
            feedback.get("answer_em"),
            feedback.get("retrieved_gold_support_recall"),
            row["usage"]["api_requests"],
        )
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines.extend(["", "## 实际查询与停止原因", ""])
    for action in Action:
        row = report["arms"][action.value]
        lines.extend(
            [
                f"- {action.value} 查询：{cell(row['query'])}",
                f"  停止原因：{cell(row['stop_reason'])}",
            ]
        )
    lines.extend(
        [
            "",
            "## 配对差值",
            "",
            "| 比较 | 可评价 | 答案 F1 差值 | 支持召回差值 | 实际查询不同 |",
            "|---|---|---|---|---|",
        ]
    )
    for name, contrast in report["contrasts"].items():
        values = (
            name,
            contrast["available"],
            contrast["answer_f1_delta"],
            contrast["support_recall_delta"],
            contrast["different_executed_queries"],
        )
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    lines.extend(
        [
            "",
            "相同查询的答案差异不记为查询改写的证据；本报告不会自动晋升经验。",
            "gold 只在运行后评分使用。缺少证据不等于支持召回为零；引用存在不等于内容蕴含。",
            "费用合计包含三个实际执行分支，不能当作部署时单一路由费用；耗时不含建索引／建库。",
            "",
        ]
    )
    return "\n".join(lines)


def save_comparison(
    run: QueryComparison, directory: Path, *, gold: GoldRecord | None = None
) -> Path:
    """Prepare serialization before claiming an exclusive new output directory."""
    report = comparison_report(run, gold=gold)
    outputs = {
        "run.json": json.dumps(run.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        "report.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        "report.md": render_comparison_markdown(run, report),
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in outputs.items():
        with (directory / name).open("x", encoding="utf-8") as handle:
            handle.write(content + "\n")
    return directory
