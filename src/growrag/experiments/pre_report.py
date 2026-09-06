"""Descriptive batch reporting; missing arms are not successful BASE clones."""

from __future__ import annotations

from html import escape


def summarize_batch(result: dict) -> dict:
    targets = result["targets"]
    arms = {}
    for action in ("BASE", "FRESH", "REUSE", "LEXICAL_POLICY"):
        rows = []
        for target in targets:
            key = target["selection"]["proposed_route"] if action == "LEXICAL_POLICY" else action
            rows.append(target["report"]["arms"][key])
        completed = [row for row in rows if row["completed"] and row["feedback"] is not None]
        arms[action] = {
            "total_targets": len(rows),
            "completed": len(completed),
            "correct": sum(row["feedback"]["answer_em"] == 1 for row in completed),
            "abstentions": sum(row["abstained"] for row in completed),
            "mean_answer_f1_completed": (
                sum(row["feedback"]["answer_f1"] for row in completed) / len(completed)
                if completed
                else None
            ),
        }
    pairs = [row["report"]["contrasts"]["REUSE_minus_FRESH"] for row in targets]
    available = [pair for pair in pairs if pair["available"]]
    return {
        "source_count": result["source_count"],
        "target_count": len(targets),
        "candidate_count": result["candidate_count"],
        "no_card_targets": sum(row["selection"]["selected_memory_id"] is None for row in targets),
        "arms": arms,
        "reuse_vs_fresh": {
            "available_pairs": len(available),
            "answer_f1_wins": sum(pair["answer_f1_delta"] > 0 for pair in available),
            "answer_f1_ties": sum(pair["answer_f1_delta"] == 0 for pair in available),
            "answer_f1_losses": sum(pair["answer_f1_delta"] < 0 for pair in available),
            "correct_to_wrong_answer": sum(pair["correct_to_wrong_answer"] for pair in available),
            "correct_to_abstention": sum(pair["correct_to_abstention"] for pair in available),
            "identical_executed_queries": sum(
                pair["different_executed_queries"] is False for pair in available
            ),
        },
        "budget": result["budget"],
        "notice": "LEXICAL_POLICY is the recorded pre-outcome branch choice, "
        "not another API run or trusted policy. Completed-only means accompany counts. "
        "No-card REUSE is unavailable, not BASE performance.",
    }


def render_batch(result: dict) -> str:
    stats = summarize_batch(result)

    def cell(value):
        if value is None:
            return "未知／未执行"
        text = escape(str(value))
        for char in "\\`*_{}[]()!":
            text = text.replace(char, "\\" + char)
        return text.replace("|", "&#124;").replace("\n", "<br>")

    def score(row):
        if not row["completed"]:
            return "未完成"
        return (
            "弃答"
            if row["abstained"]
            else ("正确" if row["feedback"]["answer_em"] else "不匹配 gold")
        )

    lines = [
        "# PRE 来源建库与目标比较",
        "",
        "手写替身测试，不是模型实验。"
        if result["execution_kind"] == "mock"
        else "真实 API 开发诊断；仅 Hotpot train 便利样本的候选池检索，不是正式基准成绩。",
        "",
        f"来源题 {stats['source_count']}；候选卡 {stats['candidate_count']}；"
        f"目标题 {stats['target_count']}；无卡题 {stats['no_card_targets']}。",
        "",
        "候选卡只通过源题观察筛选，尚未获得跨题可信性认证。自然语言适用条件尚无独立语义检查器。",
        "",
        "| 路径 | 正确 | 完成 | 全部目标题 | 弃答 |",
        "|---|---:|---:|---:|---:|",
    ]
    for action, row in stats["arms"].items():
        lines.append(
            f"| {action} | {row['correct']} | {row['completed']} | "
            f"{row['total_targets']} | {row['abstentions']} |"
        )
    pair = stats["reuse_vs_fresh"]
    lines.extend(
        [
            "",
            f"REUSE 对 FRESH：可比较 {pair['available_pairs']} 题；按答案 F1 胜／平／负 = "
            f"{pair['answer_f1_wins']}／{pair['answer_f1_ties']}／{pair['answer_f1_losses']}。",
            f"其中正确变错误 {pair['correct_to_wrong_answer']}，"
            f"正确变弃答 {pair['correct_to_abstention']}；"
            f"相同查询 {pair['identical_executed_queries']}。",
            "",
            "## 逐题结果",
            "",
            "| 题号 | 原问题 | 选择 | BASE | FRESH | REUSE |",
            "|---|---|---|---|---|---|",
        ]
    )
    for number, target in enumerate(result["targets"]):
        rows = target["report"]["arms"]
        question = target.get("question", target["question_id"])
        values = [
            number,
            question,
            target["selection"]["proposed_route"],
            score(rows["BASE"]),
            score(rows["FRESH"]),
            score(rows["REUSE"]),
        ]
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")
    budget = result.get("budget")
    if budget:
        lines.extend(
            [
                "",
                "## 实际调用与估算费用",
                "",
                f"请求数：{cell(budget['api_requests'])}；"
                f"输入 tokens：{cell(budget['input_tokens'])}；"
                f"输出 tokens：{cell(budget['output_tokens'])}。",
                f"来源与目标合计估算费用：{cell(budget['estimated_actual_cny'])} 元；"
                f"预算阻断原因：{cell(budget['block_reason'])}。",
                "按冻结的官方标价估算，不是已核对厂商账单；包含建库、提卡与所有实际对照分支。",
            ]
        )
    lines.extend(
        [
            "",
            "LEXICAL_POLICY 是事前词面匹配选择的路径，不是额外运行一次，也不是已验证的可信控制器。",
            "无卡时默认选择 BASE，但 REUSE 本身仍标未执行；不把 BASE 分数复制给 REUSE。",
            "语义支持未验证；引用存在、答案匹配、命中 gold 支持句是不同观察。",
            "本批内部校准题已有开发曝光，不能称未见测试；不报告显著性或一般优越性。",
            "",
        ]
    )
    return "\n".join(lines)
