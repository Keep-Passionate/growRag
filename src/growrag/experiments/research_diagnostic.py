"""Export sealed development feedback for research diagnosis, never re-score it.

只使用已经开放并封存的逐题反馈。没有 API、模型训练或经验写入入口；
答案列明确区分真实原始回答和分析控制回答。机械标签是观察，不是原因。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

FRESH = "fresh_original"
HISTORY = "history_body8"
METHODS = ("base", FRESH, "fresh_focused", HISTORY)
FEEDBACK = "runs/a0_wire_v2_feedback_v1"
OUTPUT = "runs/research_diagnostic_20261003_v1"
SALT = "growrag-research-audit-v1|"
QUOTAS = {"both_correct": 18, "both_wrong": 19}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(project, path):
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(project):
        raise ValueError("input leaves project")
    return resolved


def _write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def has_history_search(row):
    """选中不等于执行；第二次控制器选择也不一定发生检索。"""
    return any(c.get("search_executed") is True for c in row["usage"][HISTORY]["executed_cards"])


def cell(row, score_key="controlled_reader_scores"):
    f, h = (row[score_key][m]["answer_em"] for m in (FRESH, HISTORY))
    if f not in (0, 1) or h not in (0, 1):
        raise ValueError("EM is not binary")
    return {
        (1, 1): "both_correct",
        (1, 0): "fresh_only",
        (0, 1): "history_only",
        (0, 0): "both_wrong",
    }[f, h]


def validate_rows(rows, summary):
    ids = [r["question_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("empty or duplicate paired IDs")
    if set(ids) != set(summary["gold_ids"]) or len(ids) != summary["joint_complete_questions"]:
        raise ValueError("sealed paired ID coverage differs")
    for row in rows:
        for key in ("raw_scores", "controlled_reader_scores"):
            if set(row[key]) != set(METHODS):
                raise ValueError("method coverage differs")
            for method in METHODS:
                score = row[key][method]
                if score["status"] != "completed":
                    raise ValueError("failed arm in paired feedback")
                for metric in ("answer_em", "answer_f1"):
                    value = score[metric]
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise ValueError("invalid score")
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError("invalid score")
            cell(row, key)
        for method in METHODS:
            canonical = row["canonical_reader_method"][method]
            if canonical not in METHODS or (
                row["usage"][canonical]["reader_input_sha256"]
                != row["usage"][method]["reader_input_sha256"]
            ):
                raise ValueError("canonical Reader input differs")
            for metric in ("answer_em", "answer_f1"):
                if (
                    row["controlled_reader_scores"][method][metric]
                    != row["raw_scores"][canonical][metric]
                ):
                    raise ValueError("controlled answer metric differs from canonical raw score")
    for key, label in (("raw_scores", "raw"), ("controlled_reader_scores", "controlled_reader")):
        for method in METHODS:
            # Check published means, without recalculating any answer labels.
            for metric in ("answer_em", "answer_f1"):
                expected = summary["methods"][method][label][metric]["mean"]
                actual = mean(r[key][method][metric] for r in rows)
                if not math.isclose(actual, expected, abs_tol=1e-12):
                    raise ValueError("sealed summary/row score drift")


def paired_metrics(rows, key):
    deltas = [r[key][HISTORY]["answer_f1"] - r[key][FRESH]["answer_f1"] for r in rows]
    counts = Counter(cell(r, key) for r in rows)
    return {
        "n": len(rows),
        "em_cells": {
            c: counts[c] for c in ("both_correct", "fresh_only", "history_only", "both_wrong")
        },
        "f1_positive_n": sum(d > 0 for d in deltas),
        "f1_negative_n": sum(d < 0 for d in deltas),
        "f1_tied_n": sum(d == 0 for d in deltas),
        "f1_positive_gain": mean(max(d, 0) for d in deltas),
        "f1_negative_loss": mean(max(-d, 0) for d in deltas),
        "f1_net_gain": mean(deltas),
        "fresh_f1": mean(r[key][FRESH]["answer_f1"] for r in rows),
        "fixed_outcome_oracle_f1": mean(
            max(r[key][FRESH]["answer_f1"], r[key][HISTORY]["answer_f1"]) for r in rows
        ),
        "fixed_outcome_oracle_em": mean(
            max(r[key][FRESH]["answer_em"], r[key][HISTORY]["answer_em"]) for r in rows
        ),
    }


def select_diagnostic(rows):
    """Fixed diagnostic quotas, all wins/losses and all actually executed history."""
    groups = {
        name: [r for r in rows if cell(r) == name]
        for name in (*QUOTAS, "fresh_only", "history_only")
    }
    chosen = groups["fresh_only"] + groups["history_only"]
    for name, quota in QUOTAS.items():
        mandatory = [r for r in groups[name] if has_history_search(r)]
        if len(mandatory) > quota or len(groups[name]) < quota:
            raise ValueError("fixed diagnostic quota not satisfiable")
        remaining = [r for r in groups[name] if not has_history_search(r)]
        remaining.sort(key=lambda r: hashlib.sha256((SALT + r["question_id"]).encode()).hexdigest())
        chosen += mandatory + remaining[: quota - len(mandatory)]
    selected = {r["question_id"] for r in chosen}
    return [r for r in rows if r["question_id"] in selected]


def observations(row):
    tags = []
    if not has_history_search(row):
        tags.append("history_no_extra_search")
    if row["usage"][FRESH]["retrieval_calls"] > 1:
        tags.append("fresh_extra_search")
    if row["usage"][FRESH]["reader_input_sha256"] == row["usage"][HISTORY]["reader_input_sha256"]:
        tags.append("identical_reader_input")
    if row["model_support_claims"][HISTORY] is False:
        tags.append("history_reader_claims_unsupported")
    return tags


def flat_row(row):
    result = {
        "question_id": row["question_id"],
        "question": row["question"],
        "reference_answer": row["reference_answer"],
        "cell": cell(row),
        "raw_cell": cell(row, "raw_scores"),
        "observations_not_causes": "|".join(observations(row)),
    }
    for method in METHODS:
        result[f"{method}_raw_answer"] = row["answers"][method]
        result[f"{method}_controlled_answer"] = row["answers"][
            row["canonical_reader_method"][method]
        ]
        result[f"{method}_canonical_reader_method"] = row["canonical_reader_method"][method]
        for prefix, key in (("raw", "raw_scores"), ("controlled", "controlled_reader_scores")):
            for metric in (
                "answer_em",
                "answer_f1",
                "raw_support_recall",
                "visible_support_recall",
            ):
                result[f"{method}_{prefix}_{metric}"] = row[key][method][metric]
        for key in (
            "api_requests",
            "input_tokens",
            "output_tokens",
            "retrieval_calls",
            "estimated_actual_cny",
            "reserved_cny",
            "unknown_cost_requests",
        ):
            result[f"{method}_{key}"] = row["usage"][method][key]
        for key in ("queries", "selected_cards", "executed_cards"):
            result[f"{method}_{key}"] = json.dumps(row["usage"][method][key], ensure_ascii=False)
    result["controlled_delta_f1_history_minus_fresh"] = (
        row["controlled_reader_scores"][HISTORY]["answer_f1"]
        - row["controlled_reader_scores"][FRESH]["answer_f1"]
    )
    return result


def _casebook(cases, project):
    lines = [
        f"# {len(cases)}题研究诊断阅读册",
        "",
        "这是分层诊断样本，不是新的测试成绩。先读三个详细例子，再按本册填写原因假设。",
        "",
        "`controlled` 是原封存的同输入首次回答分析口径；`raw` 是实际生成。"
        "机器标签只记现象。所有问题禁止回写持久库。",
        "",
    ]
    for n, case in enumerate(cases, 1):
        r = case["feedback"]
        lines += [
            f"## {n:02d}. {r['question_id']} / {cell(r)}",
            "",
            r["question"],
            "",
            f"参考答案：{r['reference_answer']}；"
            f"观察：{', '.join(observations(r)) or '无上述机械标签'}。",
            "",
        ]
        for method in (FRESH, HISTORY):
            report = case["reports"][method]
            score = r["controlled_reader_scores"][method]
            answer = r["answers"][r["canonical_reader_method"][method]]
            source = (project / case["report_paths"][method]).as_posix()
            lines += [
                f"### {method}",
                "",
                f"实际答案：{r['answers'][method] or '弃答'}；controlled答案：{answer or '弃答'}；"
                f"EM={score['answer_em']:.0f}，F1={score['answer_f1']:.4f}。",
                "",
                f"检索={r['usage'][method]['retrieval_calls']}；实际估费={r['usage'][method]['estimated_actual_cny']}元；停止={report['episode']['stop_reason']}。",
                "",
                "实际检索字符串：",
                "",
            ]
            lines += [f"- `{q}`" for q in r["usage"][method]["queries"]]
            lines += [
                "",
                "选择记录：`"
                + json.dumps(r["usage"][method]["selected_cards"], ensure_ascii=False)
                + "`",
                "",
                "实际动作 / gap：",
                "",
                "```json",
                json.dumps(report["episode"]["proposals"], ensure_ascii=False, indent=2),
                "```",
                "",
                f"[完整真实记录]({source})",
                "",
            ]
        lines += [
            "待填写：最早的可核验差异 → 原因假设（非结论） → 只改一个因素 → "
            "预期中间指标 → 什么结果会否定假设。",
            "",
        ]
    return "\n".join(lines)


def run(project, output=OUTPUT):
    project = Path(project).resolve(strict=True)
    target = project / output
    if target.resolve().parent != project / "runs":
        raise ValueError("diagnostic output must be a direct runs child")
    if target.exists():
        raise FileExistsError("diagnostic exists; never overwrite")
    feedback = project / FEEDBACK
    seal = _read(feedback / "feedback_frozen.json")
    required = {"audit.json", "missing_questions.json", "per_question.json", "SUMMARY.json"}
    if set(seal["files"]) != required:
        raise ValueError("feedback seal coverage differs")
    inputs = {}
    for name, digest in seal["files"].items():
        path = _inside(project, feedback / name)
        if _sha(path) != digest:
            raise ValueError("feedback seal drift")
        inputs[path.relative_to(project).as_posix()] = digest
    inputs[(feedback / "feedback_frozen.json").relative_to(project).as_posix()] = _sha(
        feedback / "feedback_frozen.json"
    )
    rows, summary, audit = (
        _read(feedback / name) for name in ("per_question.json", "SUMMARY.json", "audit.json")
    )
    if (
        summary.get("prediction_modified") is not False
        or summary.get("memory_updated") is not False
    ):
        raise ValueError("feedback isolation not sealed")
    validate_rows(rows, summary)
    for relative, digest_key in (
        ("runs/a0_wire_v2_freeze_v1.json", "external_freeze_sha256"),
        ("runs/a0_wire_v2_v1/TERMINAL.json", "external_terminal_sha256"),
    ):
        anchor = _inside(project, project / relative)
        if (
            _sha(anchor) != summary[digest_key]
            or audit["inputs"].get(str(anchor)) != summary[digest_key]
        ):
            raise ValueError("original freeze/terminal anchor drift")
        inputs[relative] = summary[digest_key]
    chosen = select_diagnostic(rows)
    reports = {}
    # Only already-opened IDs are read. Each report must belong to the old audit.
    for row in chosen:
        reports[row["question_id"]] = {"feedback": row, "reports": {}, "report_paths": {}}
        for method in (FRESH, HISTORY):
            name = f"{row['question_id']}_{method}.json"
            matches = [Path(p) for p in audit["inputs"] if Path(p).name == name]
            if len(matches) != 1:
                raise ValueError("audited arm identity ambiguous")
            path = _inside(project, matches[0])
            if _sha(path) != audit["inputs"][str(matches[0])]:
                raise ValueError("audited report drift")
            report = _read(path)
            if (
                report["question_id"] != row["question_id"]
                or report["method"] != method
                or report["status"] != "completed"
            ):
                raise ValueError("audited report identity differs")
            if report["gold_loaded"] is not False or report["memory_updated"] is not False:
                raise ValueError("report runtime isolation differs")
            if report["reader"]["answer"] != row["answers"][method]:
                raise ValueError("raw answer/report mismatch")
            if [s["query"] for s in report["episode"]["searches"]] != row["usage"][method][
                "queries"
            ]:
                raise ValueError("actual query/report mismatch")
            relative = path.relative_to(project).as_posix()
            inputs[relative] = _sha(path)
            reports[row["question_id"]]["reports"][method] = report
            reports[row["question_id"]]["report_paths"][method] = relative
    executed = [r for r in rows if has_history_search(r)]
    lost = [r for r in rows if cell(r) == "fresh_only"]
    findings = {
        "source": FEEDBACK,
        "source_policy": summary["primary_metric_policy"],
        "split": summary["split"],
        "api_calls": 0,
        "new_gold_opened": False,
        "memory_updated": False,
        "predictions_modified": False,
        "controlled": paired_metrics(rows, "controlled_reader_scores"),
        "raw": paired_metrics(rows, "raw_scores"),
        "history_extra_search_questions": len(executed),
        "fresh_only_history_executed": sum(has_history_search(r) for r in lost),
        "fresh_only_history_not_executed": sum(not has_history_search(r) for r in lost),
        "fresh_only_no_action_reasons": dict(
            Counter(
                s["reason"]
                for r in lost
                if not has_history_search(r)
                for s in r["usage"][HISTORY]["selected_cards"]
            )
        ),
        "diagnostic_n": len(chosen),
        "diagnostic_cells": dict(Counter(cell(r) for r in chosen)),
        "diagnostic_ids": [r["question_id"] for r in chosen],
        "oracle_notice": "Fixed observed outcomes only; not a deployed router or future ceiling.",
        "causal_notice": "Observations are not causal labels; "
        "old development data cannot prove transfer.",
        "seal_notice": "Verifies sealed feedback and selected arm reports; "
        "does not redo the original HTTP/replay audit or scoring.",
    }
    if any(_sha(project / p) != digest for p, digest in inputs.items()):
        raise ValueError("input changed during export")
    target.mkdir(exist_ok=False)
    _write(target / "SUMMARY.json", findings)
    _write(target / "INPUTS.json", inputs)
    with (target / "paired_99.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        flat = [flat_row(r) for r in rows]
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    with (target / "diagnostic_50.jsonl").open("x", encoding="utf-8") as handle:
        for case in reports.values():
            handle.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n")
    with (target / "CASEBOOK.md").open("x", encoding="utf-8") as handle:
        handle.write(_casebook(list(reports.values()), project))
    _write(
        target / "frozen.json",
        {"files": {p.name: _sha(p) for p in target.iterdir()}, "api_calls": 0},
    )
    return findings


def verify_export(project, output=OUTPUT):
    """Check an existing export without changing files, scores or labels."""
    project = Path(project).resolve(strict=True)
    target = _inside(project, project / output)
    if target.parent != project / "runs":
        raise ValueError("diagnostic output must be a direct runs child")
    seal = _read(target / "frozen.json")
    for name, digest in seal["files"].items():
        if Path(name).name != name or _sha(_inside(project, target / name)) != digest:
            raise ValueError("diagnostic artifact drift")
    for relative, digest in _read(target / "INPUTS.json").items():
        if _sha(_inside(project, project / relative)) != digest:
            raise ValueError("diagnostic input drift")
    rows = _read(project / FEEDBACK / "per_question.json")
    validate_rows(rows, _read(project / FEEDBACK / "SUMMARY.json"))
    findings = _read(target / "SUMMARY.json")
    if findings["diagnostic_ids"] != [r["question_id"] for r in select_diagnostic(rows)]:
        raise ValueError("diagnostic selection differs")
    for label, key in (("controlled", "controlled_reader_scores"), ("raw", "raw_scores")):
        if findings[label] != paired_metrics(rows, key):
            raise ValueError("diagnostic paired aggregates differ")
    return {
        "status": "verified",
        "paired_n": len(rows),
        "diagnostic_n": findings["diagnostic_n"],
        "api_calls": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument(
        "--verify", action="store_true", help="Read-only verification of existing export"
    )
    args = parser.parse_args(argv)
    result = (verify_export if args.verify else run)(args.project_root, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
