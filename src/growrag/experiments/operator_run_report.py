"""Readable source/calibration execution report; no gold, scoring, API or retries.

仅访问运行目录里的白名单日志。默认只输出报告；--write-new 才新建 REPORT.md，
不覆盖旧文件。模型 reason/内部反思不进入报告，预测答案也不展示或判分。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

PROTOCOL = "growrag-operator-study-v1"


def _path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if path.parent != root:
        raise ValueError("run artifact resolves outside the selected directory")
    return path


def _read(root: Path, name: str, *, optional=False):
    path = _path(root, name)
    if optional and not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _safe(value: object) -> str:
    """Do not permit query text to inject Markdown structure into the report."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "\\|")
        .replace("`", "&#96;")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def _number(value, *, money=False) -> str:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return "未知"
    return f"{value:.8f} 元" if money else f"{value:g}"


def _call_total(calls, key, *, money=False) -> str:
    if type(calls) is not list or not calls:
        return "未知（无账单记录）"
    known = [item.get(key) for item in calls if type(item) is dict]
    values = [v for v in known if type(v) in (int, float) and math.isfinite(v) and v >= 0]
    missing = len(calls) - len(values)
    if not values:
        return f"未知（{missing} 项）"
    text = _number(sum(values), money=money)
    return text + (f"（部分已知；另 {missing} 项未知）" if missing else "")


def _events(root: Path):
    """Project necessary execution fields only; never render model_output/reason."""
    rows = defaultdict(lambda: {"searches": [], "operators": [], "errors": [], "started": False})
    path = _path(root, "events.jsonl")
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError("event row must be an object")
            key = value.get("question_id"), value.get("arm")
            kind = value.get("kind")
            if not all(isinstance(item, str) and item for item in key):
                continue
            if kind not in {
                "operator_search_start",
                "operator_search",
                "operator_proposal",
                "operator_rejection",
                "failure",
                "arm_complete",
            }:
                continue
            item = rows[key]
            item["started"] = True
            if kind == "operator_search" and type(value.get("search")) is dict:
                search = value["search"]
                item["searches"].append({"query": search.get("query"), "step": search.get("step")})
            elif kind == "operator_proposal" and type(value.get("proposal")) is dict:
                proposal = value["proposal"]
                spec = proposal.get("spec")
                if type(spec) is dict:
                    item["operators"].append(
                        {
                            "operator_id": spec.get("operator_id"),
                            "version": spec.get("version"),
                            "origin": proposal.get("origin"),
                            "step": value.get("decision"),
                        }
                    )
            elif kind in {"failure", "operator_rejection"}:
                item["errors"].append(value.get("error_type", "unspecified"))
    return rows


def build_report(directory: Path) -> str:
    """Validate the frozen prediction payload before producing unscored Markdown."""
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("run directory required")
    launch = _read(root, "launch_plan.json")
    if type(launch) is not dict:
        raise ValueError("launch plan must be an object")
    if launch.get("phase") not in {"source", "calibration"}:
        raise ValueError("evaluation reporting is locked; source/calibration only")
    if launch.get("protocol") != PROTOCOL:
        raise ValueError("unknown operator study protocol")
    ids, arms = launch.get("question_ids"), launch.get("arms")
    if (
        type(ids) is not list
        or not ids
        or any(type(q) is not str for q in ids)
        or len(set(ids)) != len(ids)
        or type(arms) is not list
        or not arms
        or any(type(a) is not str for a in arms)
        or len(set(arms)) != len(arms)
    ):
        raise ValueError("invalid planned question/arm identities")
    predictions = _read(root, "predictions.json")
    frozen = _read(root, "predictions_frozen.json")
    if (
        type(predictions) is not list
        or any(type(row) is not dict for row in predictions)
        or type(frozen) is not dict
        or frozen.get("sha256") != _fingerprint(predictions)
        or frozen.get("phase") != launch["phase"]
        or frozen.get("gold_loaded") is not False
    ):
        raise ValueError("predictions frozen fingerprint/phase/gold-free contract mismatch")
    actual_ids = [row.get("question_id") for row in predictions]
    if frozen.get("question_ids") != actual_ids or actual_ids != ids[: len(actual_ids)]:
        raise ValueError("predictions must match the frozen planned question prefix")
    by_id = {}
    for row in predictions:
        if (
            type(row.get("arms")) is not dict
            or set(row["arms"]) - set(arms)
            or any(type(record) is not dict for record in row["arms"].values())
        ):
            raise ValueError("unplanned or malformed arm reports")
        if any(record.get("feedback") is not None for record in row["arms"].values()):
            raise ValueError("reporter accepts prediction-only records, not scored feedback")
        by_id[row["question_id"]] = row["arms"]
    events = _events(root)
    budget = _read(root, "final_budget.json", optional=True)
    if budget is not None and type(budget) is not dict:
        raise ValueError("malformed final budget")
    budget = budget or {}
    counts = defaultdict(int)
    statuses = {}
    for qid in ids:
        for arm in arms:
            report = by_id.get(qid, {}).get(arm)
            event = events.get((qid, arm), {})
            status = (
                "完成"
                if report and report.get("status") == "completed"
                else "失败"
                if report and report.get("status") == "failed"
                else "中断/无完整报告"
                if report or event.get("started")
                else "未启动"
            )
            statuses[qid, arm] = status
            counts[status] += 1
    touched = sum(any(statuses[qid, arm] != "未启动" for arm in arms) for qid in ids)
    finished = sum(all(statuses[qid, arm] == "完成" for arm in arms) for qid in ids)
    lines = [
        "# 动态动作算子运行报告（未评分）",
        "",
        "本报告只核查执行与费用；未读取 gold，未判断任何答案正确与否。模型预测和长推理不展示。",
        "",
        f"- 运行：{_safe(root.name)}；阶段：{_safe(launch['phase'])}。",
        f"- 协议：{_safe(launch['protocol'])}；模型：{_safe(launch.get('model', '未知'))}。",
        f"- 计划 {len(ids)} 题 × {len(arms)} 臂 = {len(ids) * len(arms)} 条路径；"
        f"实际涉及 {touched}/{len(ids)} 题，全部臂完成 {finished} 题。",
        f"- 路径状态：完成 {counts['完成']}，失败 {counts['失败']}，"
        f"中断/无完整报告 {counts['中断/无完整报告']}，未启动 {counts['未启动']}。",
        f"- 批次 API 请求：{_number(budget.get('api_requests'))}；"
        f"输入 tokens：{_number(budget.get('input_tokens'))}；"
        f"输出 tokens：{_number(budget.get('output_tokens'))}。",
        f"- 批次估算费用：{_number(budget.get('estimated_actual_cny'), money=True)}；"
        f"保守预留：{_number(budget.get('reserved_cny'), money=True)}。",
        "- 费用是声明单价估算，不是服务商结算账单；缺失用量不补零。",
        "- 预测载荷指纹已核验；失败路径补充来自 events.jsonl，事件文件未单独封存。",
        "",
        "## 每题状态",
        "",
        "| 题号 | " + " | ".join(_safe(a) for a in arms) + " |",
        "| --- | " + " | ".join("---" for _ in arms) + " |",
    ]
    for qid in ids:
        lines.append("| " + _safe(qid) + " | " + " | ".join(statuses[qid, a] for a in arms) + " |")
    lines += ["", "## 实际查询与动作", ""]
    for qid in ids:
        if all(statuses[qid, arm] == "未启动" for arm in arms):
            continue
        lines += [f"### {_safe(qid)}", ""]
        for arm in arms:
            report = by_id.get(qid, {}).get(arm, {})
            event = events.get((qid, arm), {})
            # Failed runs explicitly persist episode=null when planning aborted.
            # The event journal can still contain the already completed searches.
            episode = report.get("episode")
            if episode is None:
                episode = {}
            elif type(episode) is not dict:
                raise ValueError("episode must be an object or null after failure")
            searches = episode.get("searches", event.get("searches", []))
            executed_steps = {row.get("step") for row in searches}
            lines += [f"#### {_safe(arm)}：{statuses[qid, arm]}", ""]
            if searches:
                for number, search in enumerate(searches, 1):
                    label = "原 query / 初检" if search.get("step") == 0 else f"实际查询 {number}"
                    lines.append(f"- {label}：{_safe(search.get('query', '未知'))}")
            else:
                lines.append("- 暂无已完成检索记录；search_start 不等同于检索已完成。")
            operators = event.get("operators", [])
            if "proposals" in episode:
                operators = [
                    {
                        "operator_id": p["spec"].get("operator_id"),
                        "version": p["spec"].get("version"),
                        "origin": p.get("origin"),
                        "step": i,
                    }
                    for i, p in enumerate(episode["proposals"], 1)
                    if type(p.get("spec")) is dict
                ]
            for op in operators:
                if op["step"] in executed_steps:
                    lines.append(
                        f"- 已执行算子：{_safe(op['operator_id'])}@{_safe(op['version'])}；"
                        f"来源：{_safe(op['origin'])}。"
                    )
            stop = (
                episode.get("stop_reason")
                or report.get("error_type")
                or ", ".join(event.get("errors", []))
                or "未记录"
            )
            lines.append(f"- 停止/异常标识：{_safe(stop)}（不是正确性判定）。")
            calls = report.get("calls")
            lines += [
                f"- API请求：{_call_total(calls, 'api_requests')}；"
                f"输入/输出 tokens：{_call_total(calls, 'input_tokens')} / "
                f"{_call_total(calls, 'output_tokens')}。",
                f"- 估算费用：{_call_total(calls, 'estimated_actual_cny', money=True)}；"
                f"预留：{_call_total(calls, 'reserved_cny', money=True)}。",
                "",
            ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--write-new", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(args.run_directory)
    if args.write_new:
        root = args.run_directory.resolve(strict=True)
        output = _path(root, "REPORT.md")
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(report)
        print(str(output))
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
