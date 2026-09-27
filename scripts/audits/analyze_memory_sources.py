"""Offline source-trajectory summary; never reads raw labels or creates memory cards."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import mean

from growrag.experiments import source_collection_core as core

ROOT = Path(__file__).resolve().parents[2]
BASE, S2G = "BASE1_AUTHOR_READER", "S2G_AUTHOR_API4"
ARMS = (BASE, S2G)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load_module("_source_analysis_shared", ROOT / "scripts/audits/analyze_reformer_pilot.py")


def prediction_only(row):
    row = deepcopy(row)
    row.pop("offline_gold_answers", None)
    row.pop("scoring_status", None)
    for arm in ARMS:
        row["arms"][arm].pop("feedback", None)
    return row


def candidate_policy_exploration(rows):
    """Post-seal source-policy counts, not admission decisions or causal evidence."""
    groups = {name: [] for name in ("A", "B", "C", "D")}
    details = {
        name: []
        for name in (
            "s2g_answer_unscorable",
            "A_raw_delta_unknown",
            "B_base_answer_unscorable",
            "A_support_field_unknown",
            "D_raw_incomplete",
            "D_retained_incomplete",
            "D_both_incomplete",
        )
    }
    flags = {}

    def recall(outcome, key):
        feedback = outcome.get("feedback") or {}
        value = feedback.get(key)
        if outcome.get("status") != "completed" or feedback.get("unscorable_annotation"):
            return None
        return value if shared.number(value) and value <= 1 else None

    for row in rows:
        qid = row["question_id"]
        b, s = (row["arms"][arm] for arm in ARMS)
        flags[qid] = {name: False for name in groups}
        if not shared.scoreable(s):
            details["s2g_answer_unscorable"].append(qid)
            continue
        if s["feedback"]["answer_em"] != 1:
            continue
        groups["A"].append(qid)
        raw, retained = (
            recall(s, key) for key in ("raw_support_recall", "retained_support_recall")
        )
        base_raw = recall(b, "raw_support_recall")
        if raw is None or base_raw is None:
            details["A_raw_delta_unknown"].append(qid)
        elif raw > base_raw:
            groups["B"].append(qid)
            if not shared.scoreable(b):
                details["B_base_answer_unscorable"].append(qid)
            elif b["feedback"]["answer_em"] == 0:
                groups["C"].append(qid)
        raw_incomplete, retained_incomplete = (
            raw is not None and raw < 1,
            retained is not None and retained < 1,
        )
        if raw is None or retained is None:
            details["A_support_field_unknown"].append(qid)
        if raw_incomplete:
            details["D_raw_incomplete"].append(qid)
        if retained_incomplete:
            details["D_retained_incomplete"].append(qid)
        if raw_incomplete and retained_incomplete:
            details["D_both_incomplete"].append(qid)
        if raw_incomplete or retained_incomplete:
            groups["D"].append(qid)
        flags[qid] = {name: qid in ids for name, ids in groups.items()}
    definitions = {
        "A": "S2G answer EM=1 (source success only)",
        "B": "A and S2G raw annotated-support recall > BASE raw annotated-support recall",
        "C": "B and BASE answer EM=0",
        "D": "A and either known S2G raw or retained annotated-support recall <1",
    }
    return {
        "exploratory_only": True,
        "source_cohort_n": len(rows),
        "groups": {
            name: {"definition": definitions[name], "n": len(ids), "question_ids": ids}
            for name, ids in groups.items()
        },
        "details": {name: {"n": len(ids), "question_ids": ids} for name, ids in details.items()},
        "notice": "C is a subset of B, B of A; D overlaps these groups. Unknown support is "
        "not zero and does not itself qualify for D. D means incomplete annotated-support "
        "coverage, NOT that the answer is unsupported. These source-policy alternatives "
        "are exploratory; none promotes an episode to trusted or proves action causality.",
    }, flags


def assemble(scored, predictions, ledger_calls, paths):
    """Use only already-scored source records and the unchanged execution ledger."""
    if len(scored) != len(predictions) or [r["question_id"] for r in scored] != [
        r["question_id"] for r in predictions
    ]:
        raise ValueError("scored source cohort changed")
    if len({r["question_id"] for r in scored}) != len(scored):
        raise ValueError("duplicate source question")
    all_calls = [c for r in scored for a in ARMS for c in r["arms"][a]["calls"]]
    trace_ids = [c["trace_id"] for c in all_calls]
    if (
        len(set(trace_ids)) != len(trace_ids)
        or {c["trace_id"]: c for c in all_calls} != {c["trace_id"]: c for c in ledger_calls}
        or len(ledger_calls) != len(all_calls)
    ):
        raise ValueError("scored records do not own the entire source ledger exactly once")
    rows = []
    for row, before in zip(scored, predictions, strict=True):
        if prediction_only(row) != prediction_only(before):
            raise ValueError("prediction changed after gold was loaded")
        arms = row["arms"]
        complete = all(shared.scoreable(arms[a]) for a in ARMS)
        out = {
            "question_id": row["question_id"],
            "offset": row["offset"],
            "question": row["question"],
            "source_prediction": paths[row["question_id"]],
            "same_complete_case": complete,
            "offline_gold_answers": row.get("offline_gold_answers"),
            "arms": {},
        }
        for arm in ARMS:
            outcome = arms[arm]
            execution = outcome.get("result", {})
            queries = []
            for event in execution.get("events", []):
                if event.get("kind") == "retrieval":
                    queries.extend(event.get("queries", []))
                    if "query" in event:
                        queries.append(event["query"])
            if arm == BASE and not queries:
                queries = [row["question"]]  # BASE uses original query outside Reader events.
            out["arms"][arm] = {
                "status": outcome["status"],
                "answer": execution.get("answer"),
                "feedback": outcome.get("feedback"),
                "retrieval_rounds": execution.get("retrieval_rounds"),
                "stop_reason": execution.get("stop_reason"),
                "elapsed_seconds": outcome.get("elapsed_seconds"),
                "usage": shared.usage(outcome["calls"]),
                "queries": queries,
            }
        if complete:
            b, s = (arms[a]["feedback"] for a in ARMS)
            out["deltas"] = {
                key: s[key] - b[key]
                if shared.number(b.get(key)) and shared.number(s.get(key))
                else None
                for key in (
                    "answer_em",
                    "answer_f1",
                    "raw_support_recall",
                    "retained_support_recall",
                    "raw_support_hits",
                    "retained_support_hits",
                )
            }
        else:
            out["deltas"] = None
        rows.append(out)
    pairs = [r for r in rows if r["same_complete_case"]]
    complete_execution_n = sum(
        all(r["arms"][a]["status"] == "completed" for a in ARMS) for r in rows
    )
    summary = {
        "schema_version": "growrag-memory-source-collection-summary-v1",
        "source_question_count": len(rows),
        "same_complete_case_n": len(pairs),
        "excluded_unscorable_n": len(rows) - len(pairs),
        "complete_execution_pairs_n": complete_execution_n,
        "technical_failure_pairs_n": len(rows) - complete_execution_n,
        "unscorable_annotation_pairs_n": complete_execution_n - len(pairs),
        "memory_cards_created": 0,
        "actual_usage_all_source_attempts": shared.usage(ledger_calls),
        "arms": {},
        "repairs": sum(
            r["arms"][BASE]["feedback"]["answer_em"] == 0
            and r["arms"][S2G]["feedback"]["answer_em"] == 1
            for r in pairs
        ),
        "harms": sum(
            r["arms"][BASE]["feedback"]["answer_em"] == 1
            and r["arms"][S2G]["feedback"]["answer_em"] == 0
            for r in pairs
        ),
        "f1_improved": sum(r["deltas"]["answer_f1"] > 0 for r in pairs),
        "f1_worsened": sum(r["deltas"]["answer_f1"] < 0 for r in pairs),
        "support_deltas": {},
        "notice": "These are supervised source trajectories for later experience induction, "
        "not held-out test scores and not memory effectiveness. BASE1 and S2G4 differ in "
        "retrieval budget and evidence extraction. Correct answers or extra gold support "
        "are observations, not causal proof that an individual repair deserves persistent memory.",
    }
    for arm in ARMS:
        selected = [r["arms"][arm] for r in pairs]
        summary["arms"][arm] = {
            "n": len(selected),
            **{
                metric: mean(r["feedback"][metric] for r in selected) if selected else None
                for metric in ("answer_em", "answer_f1")
            },
            "all_source_usage": shared.merge_usage([r["arms"][arm]["usage"] for r in rows]),
            "same_complete_case_usage": shared.merge_usage([r["usage"] for r in selected]),
            "retrieval_round_counts": dict(
                Counter(r["arms"][arm]["retrieval_rounds"] for r in rows)
            ),
            "stop_reason_counts": dict(Counter(r["arms"][arm]["stop_reason"] for r in rows)),
        }
    for key in (
        "raw_support_recall",
        "retained_support_recall",
        "raw_support_hits",
        "retained_support_hits",
    ):
        values = [r["deltas"][key] for r in pairs if r["deltas"][key] is not None]
        summary["support_deltas"][key] = {
            "n": len(values),
            "mean": mean(values) if values else None,
            "increased": sum(v > 0 for v in values),
            "decreased": sum(v < 0 for v in values),
        }
    summary["candidate_policy_exploration"], candidate_flags = candidate_policy_exploration(rows)
    for row in rows:
        row["candidate_policy_flags"] = candidate_flags[row["question_id"]]
    return rows, summary


def write_markdown(output, rows, summary):
    lines = [
        "# 独立来源题：BASE与S2G轨迹汇总",
        "",
        "这批题用于形成候选经验，不是测试集；当前仍没有生成或上线记忆卡。",
        "",
        f"来源{summary['source_question_count']}题，同一完整可评分分母{summary['same_complete_case_n']}题。",
        f"技术失败{summary['technical_failure_pairs_n']}题；完整执行但标注不可评分"
        f"{summary['unscorable_annotation_pairs_n']}题。失败不记EM0，实际费用仍计入。",
        "",
        "| 方法 | EM | F1 | 本批实际API请求 | 本批估价¥ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        value = summary["arms"][arm]
        u = value["all_source_usage"]
        lines.append(
            f"| {arm} | {value['answer_em']} | {value['answer_f1']} | {u['api_requests']} | "
            f"{u['total_estimated_cost_cny']} |"
        )
    lines.extend(
        [
            "",
            f"S2G相对BASE：修复{summary['repairs']}题，退步{summary['harms']}题；F1改善{summary['f1_improved']}题、下降{summary['f1_worsened']}题。",
            "",
            "| S2G−BASE支持覆盖 | 可比较题数 | 平均增量 | 增加题数 | 减少题数 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, value in summary["support_deltas"].items():
        lines.append(
            f"| {key} | {value['n']} | {value['mean']} | "
            f"{value['increased']} | {value['decreased']} |"
        )
    lines.extend(
        [
            "",
            "raw表示检索到的原文覆盖；retained表示最终保留给回答器的内容覆盖。S2G有更多轮次，覆盖提高不等于某条经验的因果收益，也不是语义支持判定。",
            "",
            "缺失费用不当成零；完整逐题记录、原始轨迹入口见[QUESTIONS.md](QUESTIONS.md)。per_question.jsonl保留每题轮次、token、成本和支撑增量。",
            "",
            "## 下一步只做候选诊断",
            "",
            "区分BASE本来会做、S2G真正新增修复、S2G退步，以及是否真的检索到新证据；不要把Reader偶然纠错自动写成可迁移规则。后续提炼只看来源题，校准和探针不建库。",
        ]
    )
    lines.extend(
        [
            "",
            "## 来源筛选口径探索（尚未选定录入政策）",
            "",
            "A=最终S2G正确；B=A且raw标注支持召回增加；C=B且BASE原本错误；D=A但raw或retained标注支持覆盖不完整。C⊆B⊆A，D可能与它们重叠。",
            "",
            "| 口径 | 题数 | 来源ID |",
            "| --- | ---: | --- |",
        ]
    )
    exploration = summary["candidate_policy_exploration"]
    for name, value in exploration["groups"].items():
        lines.append(f"| {name} | {value['n']} | {', '.join(value['question_ids']) or '无'} |")
    lines.extend(
        [
            "",
            "D不能称为‘答案不受支持’：这里只知道标注证据没有完整覆盖，可能存在其他正确证据。缺失指标不当成0。A/B/C均不能自动证明某条动作有效或可迁移，也不自动成为trusted记忆。",
            "",
            "unknown及D的raw/retained细分数量与ID见analysis.json的candidate_policy_exploration.details。",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = [
        "# 全部来源题：逐题轨迹入口",
        "",
        "Gold仅在整批预测冻结后评分；本页面是离线审计，不进入模型提示。",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['offset']:04d} · {row['question_id']}",
                "",
                row["question"],
                "",
                f"离线参考答案：{row['offline_gold_answers']}",
                "",
                "| 方法 | 答案 | EM | F1 | 检索轮数 | 输入token | 输出token | 估价¥ |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for arm in ARMS:
            value = row["arms"][arm]
            feedback, u = value["feedback"] or {}, value["usage"]
            answer = str(value["answer"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {arm} | {answer} | {feedback.get('answer_em')} | {feedback.get('answer_f1')} | "
                f"{value['retrieval_rounds']} | {u['total_input_tokens']} | "
                f"{u['total_output_tokens']} | {u['total_estimated_cost_cny']} |"
            )
        for arm in ARMS:
            value = row["arms"][arm]
            lines.extend(
                ["", f"{arm}停止原因：{value['stop_reason']}；检索query：{value['queries']}", ""]
            )
        lines.extend(
            [
                f"支持及分数差值：{row['deltas']}",
                f"探索性来源资格（非trusted判定）：{row['candidate_policy_flags']}",
                "",
                f"[完整预测和逐轮证据](<{Path(row['source_prediction']).resolve().as_posix()}>)",
                "",
            ]
        )
    (output / "QUESTIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "data/hotpotqa/memory_sources_sep27_v1/manifest.json",
    )
    parser.add_argument("--scored-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.resolve().is_relative_to(args.runs_root.resolve()):
        raise ValueError("output must remain inside runs")
    questions, plan = core.load_source_questions(args.source_manifest)
    predictions, prediction_hashes = core.verify_collection(args.runs_root, plan, questions)
    proof_path, scores_path = (
        args.scored_dir / "collection_frozen.json",
        args.scored_dir / "scored_source_reports.json",
    )
    proof = json.loads(proof_path.read_bytes())
    if (
        proof.get("protocol") != core.PROTOCOL
        or proof.get("source_plan_sha256") != core.PLAN_SHA
        or proof.get("question_ids") != plan["roles"]["source"]
        or proof.get("prediction_artifact_sha256") != prediction_hashes
    ):
        raise ValueError("global source seal mismatch")
    scored = json.loads(scores_path.read_bytes())
    paths = {json.loads(Path(p).read_bytes())["question_id"]: p for p in prediction_hashes}
    calls, budget_hashes = [], {}
    for path in sorted(args.runs_root.glob(f"{core.PREFIX}*/final_budget.json")):
        budget_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        calls.extend(json.loads(path.read_bytes())["calls"])
    rows, summary = assemble(scored, predictions, calls, paths)
    summary["source_sha256"] = {
        **prediction_hashes,
        **budget_hashes,
        str(proof_path.resolve()): hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        str(scores_path.resolve()): hashlib.sha256(scores_path.read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    (args.output / "per_question.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    write_markdown(args.output, rows, summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "source_question_count",
                    "same_complete_case_n",
                    "repairs",
                    "harms",
                    "f1_improved",
                    "f1_worsened",
                    "actual_usage_all_source_attempts",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
