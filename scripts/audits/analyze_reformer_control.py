"""Read sealed ID/examples experiments; no model calls or prediction changes.

中文：先比较共用同一次选择的“有示例/无示例”，再用另一个完整配对分母
比较缓存基线。共享选择实际只付一次费；单路部署成本是明确标注的假设成本。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "_reformer_shared_analysis", Path(__file__).with_name("analyze_reformer_pilot.py")
)
_SHARED = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SHARED)
usage, merge_usage, scoreable = _SHARED.usage, _SHARED.merge_usage, _SHARED.scoreable
number, call_role = _SHARED.number, _SHARED.call_role
BASE, S2G, OLD = _SHARED.BASE, _SHARED.S2G, _SHARED.REFORMER
WITH, WITHOUT = "WITH_EXAMPLES", "WITHOUT_EXAMPLES"
ARMS = (WITH, WITHOUT)
ALL_ARMS = (BASE, S2G, OLD, WITH, WITHOUT)
PROTOCOL = "growrag-reformer-id-examples-v1"
PREFIX = "2026-09-27_reformer_control_v1_"
EXPECTED_MANIFEST = _SHARED.EXPECTED_MANIFEST
EXPECTED_CACHE = "b541f6b0bf853dbbd32527135ffab5f9a7bbdf8e6c8d07f6dee1b06da9d2543f"
EXPECTED_CACHE_SUMMARY = "00751439c53298641441c3daad644f3d40127093d3de76393e0d0e831f81aaf1"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def expected_order(offset):
    return list(ARMS if offset % 2 == 0 else reversed(ARMS))


def selection_digest(selection):
    payload = {k: v for k, v in selection.items() if k != "selection_sha256"}
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def paired_difference(rows, left, right):
    """Right minus left; caller determines and names the common denominator."""
    pairs = [(r["arms"][left]["feedback"], r["arms"][right]["feedback"]) for r in rows]
    return {
        "n": len(rows),
        "left": left,
        "right": right,
        "em_repairs": sum(a["answer_em"] == 0 and b["answer_em"] == 1 for a, b in pairs),
        "em_harms": sum(a["answer_em"] == 1 and b["answer_em"] == 0 for a, b in pairs),
        "f1_improved": sum(b["answer_f1"] > a["answer_f1"] for a, b in pairs),
        "f1_worsened": sum(b["answer_f1"] < a["answer_f1"] for a, b in pairs),
        **{
            f"mean_{metric}_delta": mean(b[metric] - a[metric] for a, b in pairs) if pairs else None
            for metric in ("answer_em", "answer_f1")
        },
    }


def scores(rows, arms):
    return {
        arm: {
            metric: mean(r["arms"][arm]["feedback"][metric] for r in rows) if rows else None
            for metric in ("answer_em", "answer_f1")
        }
        for arm in arms
    }


def verify_pair(report):
    """The examples field may differ; question, selected rule and context may not."""
    if report.get("arm_order") != expected_order(report["offset"]):
        raise ValueError("counterbalanced arm order mismatch")
    selection = report["selection"]
    if selection["status"] != "completed":
        if any(report["arms"][a]["status"] == "completed" for a in ARMS):
            raise ValueError("arm completed without shared selection")
        return
    shared = selection["result"]
    if shared.get("question") != report["question"] or shared.get(
        "selection_sha256"
    ) != selection_digest(shared):
        raise ValueError("shared selection identity missing")
    canonical = shared["selected_pattern"]
    if not isinstance(canonical.get("examples"), list) or not canonical["examples"]:
        raise ValueError("canonical examples absent; not the planned examples contrast")
    for arm in ARMS:
        outcome = report["arms"][arm]
        if outcome["status"] != "completed":
            continue
        result = outcome["result"]
        expected = deepcopy(canonical)
        if arm == WITHOUT:
            expected["examples"] = []
        if (
            result.get("selected_pattern") != expected
            or result.get("selected_pattern_id", result.get("pattern_id")) != shared["pattern_id"]
            or result.get("initial_selector_documents") != shared["initial_selector_documents"]
            or result.get("shared_selection_sha256") != shared["selection_sha256"]
            or result.get("shared_selection_question_id") != shared["question_id"]
            or result.get("question") != report["question"]
            or result.get("include_examples") is not (arm == WITH)
            or result.get("example_count") != len(expected["examples"])
        ):
            raise ValueError("paired rule, source, question or examples mismatch")


def _validate_calls(report, batch, owned, all_trace_ids):
    qid = report["question_id"]
    for name, outcome in [("SELECTION", report["selection"]), *report["arms"].items()]:
        calls = outcome.get("calls", [])
        expected_roles = (
            ["pattern_selection"] if name == "SELECTION" else ["pattern_application", "answer"]
        )
        if outcome["status"] == "completed" and [call_role(c) for c in calls] != expected_roles:
            raise ValueError("completed stage call roles/count mismatch")
        if outcome["status"] in ("not_started", "not_executed") and calls:
            raise ValueError("unstarted stage owns a paid call")
        for call in calls:
            trace = call["trace_id"]
            if trace in owned or trace in all_trace_ids:
                raise ValueError("duplicate call ownership")
            if trace.split("/")[:3] != [batch["run_id"], qid, name]:
                raise ValueError("call outside its question/stage")
            owned[trace] = call


def _oracle(rows, additions):
    result = {"n": len(rows), "added_arms": list(additions)}
    for metric in ("answer_em", "answer_f1"):
        old = [max(r["arms"][a]["feedback"][metric] for a in (BASE, S2G)) for r in rows]
        new = [
            max([v, *[r["arms"][a]["feedback"][metric] for a in additions]])
            for r, v in zip(rows, old, strict=True)
        ]
        result[f"base_s2g_{metric}"] = mean(old) if rows else None
        result[f"expanded_{metric}"] = mean(new) if rows else None
        result[f"extra_{metric}"] = (
            mean(n - o for n, o in zip(new, old, strict=True)) if rows else None
        )
        result[f"extra_{metric}_question_ids"] = [
            r["question_id"] for r, n, o in zip(rows, new, old, strict=True) if n > o
        ]
    return result


def assemble(expected_ids, cached_rows, batches):
    """Pure paired/accounting projection, also usable with tiny synthetic fixtures."""
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate expected question")
    cached = {r["question_id"]: r for r in cached_rows}
    if len(cached) != len(cached_rows) or not set(expected_ids) <= cached.keys():
        raise ValueError("duplicate or missing cached cohort")
    records, all_calls, trace_ids = {}, [], set()
    for batch in batches:
        ledger, owned = batch["budget"], {}
        canonical = {c["trace_id"]: c for c in ledger["calls"]}
        if len(canonical) != len(ledger["calls"]):
            raise ValueError("duplicate ledger trace")
        for report in batch["reports"]:
            qid, offset = report["question_id"], report["offset"]
            if (
                type(offset) is not int
                or not 0 <= offset < len(expected_ids)
                or expected_ids[offset] != qid
                or qid in records
            ):
                raise ValueError("question ID/offset mismatch or repeated execution")
            if set(report["arms"]) != set(ARMS):
                raise ValueError("unexpected paired arms")
            if report["question"] != cached[qid].get("question"):
                raise ValueError("cached and current questions differ")
            verify_pair(report)
            _validate_calls(report, batch, owned, trace_ids)
            records[qid] = (report, batch["run_id"])
        if owned != canonical:
            raise ValueError("reports do not own every ledger call exactly once")
        totals = usage(list(canonical.values()))
        for key in ("api_requests", "reserved_cny"):
            if not math.isclose(totals[key], ledger[key], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError("root request ledger sum mismatch")
        trace_ids.update(canonical)
        all_calls.extend(canonical.values())
    rows = []
    for offset, qid in enumerate(expected_ids):
        old = cached[qid]
        report, run_id = records.get(qid, ({}, None))
        selection = report.get("selection", {"status": "not_executed", "calls": []})
        new = report.get("arms", {a: {"status": "not_executed", "calls": []} for a in ARMS})
        arms = {**old["arms"], **new}
        shared_calls = selection.get("calls", [])
        actual_calls = [*shared_calls, *[c for a in ARMS for c in new[a].get("calls", [])]]
        pair = all(scoreable(new[a]) for a in ARMS)
        row = {
            "question_id": qid,
            "offset": offset,
            "question": old["question"],
            "run_id": run_id,
            "source_report": report.get("_report_path"),
            "cached_report": old.get("source_report"),
            "selection": selection,
            "arm_order": report.get("arm_order", expected_order(offset)),
            "arms": arms,
            "new_pair_complete": pair,
            "all_five_complete": all(scoreable(arms[a]) for a in ALL_ARMS),
            "actual_new_usage": usage(actual_calls) if report else None,
            "selection_usage": usage(shared_calls) if report else None,
            "arm_exclusive_usage": {
                a: usage(new[a].get("calls", [])) if report else None for a in ARMS
            },
            "hypothetical_one_path_usage": {
                a: usage([*shared_calls, *new[a].get("calls", [])]) if report else None
                for a in ARMS
            },
            "hypothetical_one_path_elapsed_seconds": {
                a: selection["elapsed_seconds"] + new[a]["elapsed_seconds"]
                if number(selection.get("elapsed_seconds"))
                and number(new[a].get("elapsed_seconds"))
                else None
                for a in ARMS
            },
            "cached_usage": {
                BASE: old["cached_usage"][BASE],
                S2G: old["cached_usage"][S2G],
                OLD: old["new_usage"],
            },
        }
        row["diagnostics"] = {
            "pattern_id": selection.get("result", {}).get("pattern_id"),
            "same_rewritten_query": new[WITH]["result"].get("rewritten_query")
            == new[WITHOUT]["result"].get("rewritten_query")
            if pair
            else None,
            "same_final_document_ids": [
                d["doc_id"] for d in new[WITH]["result"].get("retrieved_documents", [])
            ]
            == [d["doc_id"] for d in new[WITHOUT]["result"].get("retrieved_documents", [])]
            if pair
            else None,
            "rewrite_equals_original": {
                a: new[a].get("result", {}).get("rewritten_query") == old["question"]
                if new[a]["status"] == "completed"
                else None
                for a in ARMS
            },
            "fallback_stages": {
                a: new[a].get("result", {}).get("author_fallback_stages", []) for a in ARMS
            },
            "truncated_response_calls": {
                a: sum(c.get("_audit_finish_reason") == "length" for c in new[a].get("calls", []))
                for a in ARMS
            },
        }
        rows.append(row)
    pairs = [r for r in rows if r["new_pair_complete"]]
    all_five = [r for r in rows if r["all_five_complete"]]
    summary = {
        "planned_questions": len(expected_ids),
        "executed_questions": len(records),
        "status_counts": {a: dict(Counter(r["arms"][a]["status"] for r in rows)) for a in ARMS},
        "selection_status_counts": dict(Counter(r["selection"]["status"] for r in rows)),
        "new_pair_complete_n": len(pairs),
        "paired_scores": scores(pairs, ARMS),
        "with_vs_without": paired_difference(pairs, WITHOUT, WITH),
        "all_five_complete_n": len(all_five),
        "all_five_scores": scores(all_five, ALL_ARMS),
        "all_five_differences": {
            a: {b: paired_difference(all_five, b, a) for b in (BASE, S2G, OLD)} for a in ARMS
        },
        "oracle_opportunity_not_a_router": {a: _oracle(all_five, (a,)) for a in (*ARMS, OLD)},
        "oracle_combined_new_arms_not_a_router": _oracle(all_five, ARMS),
        "actual_new_usage_all_attempts": usage(all_calls),
        "actual_new_role_usage": {
            role: usage([c for c in all_calls if call_role(c) == role])
            for role in ("pattern_selection", "pattern_application", "answer")
        },
        "hypothetical_one_path_usage_on_complete_pairs": {
            a: merge_usage([r["hypothetical_one_path_usage"][a] for r in pairs]) for a in ARMS
        },
        "hypothetical_one_path_mean_elapsed_seconds_on_complete_pairs": {
            a: mean(r["hypothetical_one_path_elapsed_seconds"][a] for r in pairs)
            if pairs
            and all(r["hypothetical_one_path_elapsed_seconds"][a] is not None for r in pairs)
            else None
            for a in ARMS
        },
        "cost_notice": "Actual selector requests charged ONCE. Hypothetical one-path costs "
        "separately add that same selector to each exclusive arm; never sum them as billed cost. "
        "Old BASE/S2G/ReFormeR are cached, not new charges. Missing usage is unknown, not zero. "
        "Observed serial durations are not a controlled deployment latency benchmark.",
        "diagnostics": {
            "same_rewritten_query_n": sum(
                r["diagnostics"]["same_rewritten_query"] is True for r in pairs
            ),
            "same_final_document_ids_n": sum(
                r["diagnostics"]["same_final_document_ids"] is True for r in pairs
            ),
            "pattern_counts": dict(Counter(r["diagnostics"]["pattern_id"] for r in pairs)),
            "fallback_question_counts": {
                a: sum(bool(r["diagnostics"]["fallback_stages"][a]) for r in rows) for a in ARMS
            },
            "truncated_response_call_counts": {
                a: sum(r["diagnostics"]["truncated_response_calls"][a] for r in rows) for a in ARMS
            },
            "order_counts": dict(Counter(r["arm_order"][0] for r in pairs)),
        },
        "limits": [
            "Exploratory frozen development cohort, "
            "not untouched test or original-paper reproduction.",
            "Examples change prompt content and length together; "
            "this does not isolate semantic content from token length.",
            "Original cached ReFormeR used another selection prompt: "
            "comparisons to it do not isolate examples.",
            "One model sample per arm; identical prompts may still vary. "
            "No significance or causal mechanism claim.",
            "Oracle uses gold after execution; it is opportunity, never an online learned router.",
        ],
    }
    return rows, summary


def _prediction_projection(report):
    value = deepcopy(report)
    for key in ("offline_gold_answers", "scoring_error_type", "question_type"):
        value.pop(key, None)
    value.pop("scoring_status", None)
    for outcome in [value["selection"], *value["arms"].values()]:
        outcome.pop("feedback", None)
        outcome.pop("initial_feedback", None)
    return value


def read_batch(directory, read, hashes, expected_ids):
    """Check stored prediction seal, code text, request ownership and launch identity."""
    launch = read(directory / "launch_plan.json")
    if (
        launch.get("protocol") != PROTOCOL
        or launch.get("manifest_sha256") != EXPECTED_MANIFEST
        or launch.get("run_id") != directory.name
    ):
        raise ValueError("unexpected control protocol/manifest/run identity")
    start, count = launch.get("start"), launch.get("count")
    if (
        type(start) is not int
        or type(count) is not int
        or count < 1
        or start < 0
        or start + count > len(expected_ids)
        or launch.get("question_ids") != expected_ids[start : start + count]
        or launch.get("max_calls") != 5 * count
    ):
        raise ValueError("launch cohort differs from frozen first100")
    snapshot = read(directory / "source_snapshot.json")
    if (
        not snapshot.get("files")
        or snapshot.get("sha256") != fingerprint(snapshot["files"])
        or snapshot["sha256"] != launch.get("source_sha256")
    ):
        raise ValueError("source snapshot aggregate hash mismatch")
    for item in snapshot["files"].values():
        if hashlib.sha256(item["text"].encode()).hexdigest() != item["sha256"]:
            raise ValueError("source snapshot text hash mismatch")
    event_path = directory / "events.jsonl"
    raw = event_path.read_bytes()
    hashes[str(event_path)] = hashlib.sha256(raw).hexdigest()
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not events or events[-1].get("kind") != "exit":
        raise ValueError("control execution has not terminated")
    reports, predictions = read(directory / "reports.json"), []
    for report in reports:
        if report["question_id"] not in launch["question_ids"]:
            raise ValueError("report outside launch cohort")
        question_dir = directory / "questions" / f"{report['offset']:04d}"
        prediction = read(question_dir / "prediction_report.json")
        if report != read(question_dir / "report.json") or _prediction_projection(
            prediction
        ) != _prediction_projection(report):
            raise ValueError("prediction changed after scoring")
        if "offline_gold_answers" in prediction or any(
            o.get("feedback") is not None
            for o in [prediction["selection"], *prediction["arms"].values()]
        ):
            raise ValueError("prediction contains offline gold feedback")
        predictions.append(prediction)
    seal = read(directory / "predictions_frozen.json")
    completed = [
        r["question_id"]
        for r in predictions
        if all(r["arms"][a]["status"] == "completed" for a in ARMS)
    ]
    completed_arms = [
        {"question_id": r["question_id"], "arm": a}
        for r in predictions
        for a in ARMS
        if r["arms"][a]["status"] == "completed"
    ]
    if (
        seal.get("run_id") != directory.name
        or seal.get("reports_sha256_before_scoring") != fingerprint(predictions)
        or seal.get("completed_question_ids") != completed
        or seal.get("completed_arm_ids") != completed_arms
    ):
        raise ValueError("prediction freeze seal mismatch")
    budget = read(directory / "final_budget.json")
    # Do not add audit fields to the canonical calls: reports must match the ledger exactly.
    audit_finishes = {}
    for call in budget["calls"]:
        path = Path(call["audit_path"]).resolve()
        if not path.is_relative_to((directory / "api_audit").resolve()):
            raise ValueError("API audit escaped its run directory")
        audit = read(path)
        if any(
            audit.get(key) != call.get(key)
            for key in (
                "trace_id",
                "prompt_version",
                "api_requests",
                "input_tokens",
                "output_tokens",
            )
        ) or audit.get("request", {}).get("model") != launch.get("model"):
            raise ValueError("API audit does not match ledger/model")
        if audit.get("retry_count") != 0:
            raise ValueError("unexpected retry in no-retry protocol")
        audit_finishes[call["trace_id"]] = audit.get("finish_reason")
    for report in reports:
        report["_report_path"] = str(
            directory / "questions" / f"{report['offset']:04d}" / "report.json"
        )
    return {
        "run_id": directory.name,
        "reports": reports,
        "budget": budget,
        "launch": launch,
        "audit_finish_reasons": audit_finishes,
    }


def analyze(runs_root, manifest_path, output, *, count=100):
    runs_root, output = Path(runs_root).resolve(), Path(output).resolve()
    if output == runs_root or not output.is_relative_to(runs_root):
        raise ValueError("output must be a new directory inside runs root")
    if output.exists():
        raise FileExistsError(output)
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("only the frozen first100 cohort is allowed")
    hashes = {}

    def raw_read(path, expected=None):
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("frozen source hash mismatch")
        hashes[str(Path(path).resolve())] = digest
        return raw

    def read(path, expected=None):
        return json.loads(raw_read(path, expected))

    manifest = read(manifest_path, EXPECTED_MANIFEST)
    expected_ids = manifest["question_ids"][:count]
    cache_dir = runs_root / "2026-09-27_reformer_analysis_v2"
    read(cache_dir / "analysis.json", EXPECTED_CACHE_SUMMARY)
    cached = [
        json.loads(line)
        for line in raw_read(cache_dir / "per_question.jsonl", EXPECTED_CACHE).splitlines()
        if line.strip()
    ]
    batches = [
        read_batch(d, read, hashes, expected_ids)
        for d in sorted(runs_root.glob(f"{PREFIX}*"))
        if d.is_dir()
    ]
    identities = {
        fingerprint(
            {
                k: b["launch"].get(k)
                for k in (
                    "source_sha256",
                    "model",
                    "generation_profile",
                    "temperature",
                    "top_p",
                    "enable_thinking",
                    "index_path",
                    "author",
                )
            }
        )
        for b in batches
    }
    if len(identities) > 1:
        raise ValueError("control batches changed source/model/retrieval configuration")
    rows, summary = assemble(expected_ids, cached, batches)
    finishes = {
        trace: value for batch in batches for trace, value in batch["audit_finish_reasons"].items()
    }
    for row in rows:
        for arm in ARMS:
            row["diagnostics"]["truncated_response_calls"][arm] = sum(
                finishes.get(c["trace_id"]) == "length" for c in row["arms"][arm].get("calls", [])
            )
    summary["diagnostics"]["truncated_response_call_counts"] = {
        a: sum(r["diagnostics"]["truncated_response_calls"][a] for r in rows) for a in ARMS
    }
    summary.update(
        schema_version="growrag-reformer-control-analysis-v1",
        created_utc=datetime.now(UTC).isoformat(),
        source_sha256=hashes,
    )
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
    lines = [
        "# 模式ID与示例配对实验",
        "",
        "同题共用一次选择；唯一计划中的正文差异是是否提供原模式示例。",
        "",
        f"计划{summary['planned_questions']}题，新配对完整{summary['new_pair_complete_n']}题；五路缓存对比完整{summary['all_five_complete_n']}题。两个分母不得混用。",
        "",
        "| 当前同题配对 | EM | F1 |",
        "| --- | ---: | ---: |",
    ]
    for arm, value in summary["paired_scores"].items():
        lines.append(f"| {arm} | {value['answer_em']} | {value['answer_f1']} |")
    delta = summary["with_vs_without"]
    lines.extend(
        [
            "",
            f"有示例相对无示例：修复{delta['em_repairs']}题，损害{delta['em_harms']}题；F1改善{delta['f1_improved']}题，下降{delta['f1_worsened']}题。",
            "",
            "| 同一五路完整配对 | EM | F1 |",
            "| --- | ---: | ---: |",
        ]
    )
    for arm, value in summary["all_five_scores"].items():
        lines.append(f"| {arm} | {value['answer_em']} | {value['answer_f1']} |")
    cost = summary["actual_new_usage_all_attempts"]
    lines.extend(
        [
            "",
            f"本次实际请求{cost['api_requests']}次；已知估价¥{cost['known_estimated_cost_cny_subtotal']:.8f}，未知费用{cost['unknown_estimated_cost_cny_calls']}次。共享选择只计一次。",
            "",
            "单路部署假设：选择一次，再只执行一个分支；下面各行不可相加当作实付成本。",
            "",
            "| 单路部署假设（新完整配对） | 请求数 | 估价¥ | 平均观测耗时秒 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm in ARMS:
        value = summary["hypothetical_one_path_usage_on_complete_pairs"][arm]
        lines.append(
            f"| {arm} | {value['api_requests']} | {value['total_estimated_cost_cny']} | "
            f"{summary['hypothetical_one_path_mean_elapsed_seconds_on_complete_pairs'][arm]} |"
        )
    lines.extend(
        [
            "",
            "旧BASE/S2G/ReFormeR来自冻结缓存，不算新增花费。失败仍计入全尝试成本；缺失不按零处理。耗时不是控制网络条件的部署速度基准。",
            "",
            "## Oracle：只是事后机会，不是路由器",
            "",
        ]
    )
    for arm in ARMS:
        value = summary["oracle_opportunity_not_a_router"][arm]
        lines.append(
            f"{arm}相对BASE+S2G：额外EM={value['extra_answer_em']}，额外F1={value['extra_answer_f1']}（n={value['n']}）。"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "示例内容和提示长度一起改变，不等于已经隔离语义价值；原ReFormeR缓存的选择提示也不同，不是纯示例消融。没有训练动态记忆、没有训练路由器，不作显著性或新颖性声明。",
            "",
            "所有题的模式、两路改写、答案、评分与轨迹链接见[QUESTIONS.md](QUESTIONS.md)。",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    questions = ["# 全部题逐题对照", "", "以下评分只在预测冻结后取得；不进入选择器或改写器。", ""]
    for row in rows:
        selected = row["selection"].get("result", {})
        questions.extend(
            [
                f'<a id="q{row["offset"]:04d}"></a>',
                f"## {row['offset']:04d} · {row['question_id']}",
                "",
                row["question"],
                "",
                f"共用模式：{selected.get('pattern_id')} / "
                f"{selected.get('selected_pattern', {}).get('pattern_name')}；"
                f"执行顺序：{row['arm_order']}。",
                "",
                f"规则：{selected.get('selected_pattern', {}).get('transformation_rule')}",
                "",
            ]
        )
        for arm in ARMS:
            result = row["arms"][arm].get("result", {})
            questions.extend(
                [
                    f"{arm}，示例{result.get('example_count')}个：",
                    "",
                    str(result.get("rewritten_query")),
                    "",
                    f"实际检索：{result.get('retrieval_query')}",
                    "",
                ]
            )
        questions.extend(["| 方法 | 状态 | 答案 | EM | F1 |", "| --- | --- | --- | ---: | ---: |"])
        for arm in ALL_ARMS:
            outcome = row["arms"][arm]
            feedback = outcome.get("feedback") or {}
            answer = (
                str(outcome.get("answer", outcome.get("result", {}).get("answer")))
                .replace("|", "\\|")
                .replace("\n", " ")
            )
            questions.append(
                f"| {arm} | {outcome['status']} | {answer} | "
                f"{feedback.get('answer_em')} | {feedback.get('answer_f1')} |"
            )
        questions.append("")
        for key, label in (("source_report", "本轮完整记录"), ("cached_report", "原ReFormeR记录")):
            if row[key]:
                questions.append(f"[{label}](<{Path(row[key]).resolve().as_posix()}>)")
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
    summary = analyze(args.runs_root, args.manifest, args.output, count=args.count)
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "status_counts",
                    "new_pair_complete_n",
                    "paired_scores",
                    "with_vs_without",
                    "actual_new_usage_all_attempts",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
