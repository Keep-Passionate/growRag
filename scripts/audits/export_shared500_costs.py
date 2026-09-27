"""Export audited shared500 costs, without API calls or changing source records.

One JSONL row is one question; arms and roles remain separate. Unknown usage is
not zero, failed calls still cost money, and reservations are not provider bills.
This exporter is intentionally specific to the frozen 500_v1 development run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("BASE1_AUTHOR_READER", "S2G_AUTHOR_API4")
ROLES = ("judge", "extract", "answer")
EXPECTED_MANIFEST = "2fb117c03b64be3ab12398a297c82ce3315600c14e33f5dcd56c5c6c5c59e064"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def usage(calls):
    """Known subtotals are useful even when the full total cannot be known."""
    result = {
        "recorded_call_count": len(calls),
        "api_requests": sum(c["api_requests"] for c in calls),
        "status_counts": dict(Counter(c["status"] for c in calls)),
        "requested_models": sorted({c["requested_model"] for c in calls}),
        "returned_models": sorted({c["returned_model"] for c in calls if c["returned_model"]}),
        "unknown_returned_model_calls": sum(c["returned_model"] is None for c in calls),
        "reserved_cny": sum(c["reserved_cny"] for c in calls),
    }
    for field, label in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("recomputed_estimate_cny", "estimated_cost_cny"),
        ("elapsed_seconds", "api_elapsed_seconds"),
    ):
        known = [c[field] for c in calls if c[field] is not None]
        result[f"known_{label}_subtotal"] = sum(known)
        result[f"unknown_{label}_calls"] = len(calls) - len(known)
        result[f"total_{label}"] = sum(known) if len(known) == len(calls) else None
    return result


def retrieval_counts(events):
    """Track the active arm because BASE retrieval events lack a question ID."""
    counts, active = Counter(), None
    for event in events:
        if event["kind"] == "arm_start":
            active = (event["question_id"], event["arm"])
            if active in counts:
                raise ValueError("duplicate arm_start")
            counts[active] = 0
        elif event["kind"] == "retrieval":
            if active is None or event["arm"] != active[1]:
                raise ValueError("retrieval outside active arm")
            counts[active] += 1
            if event["round"] != counts[active]:
                raise ValueError("retrieval round sequence mismatch")
    return counts


def assemble(expected_ids, batches, calls):
    """Pure projection, testable with small synthetic cohorts and no local data."""
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate manifest question")
    by_owner, traces = defaultdict(list), set()
    for call in calls:
        if call["trace_id"] in traces:
            raise ValueError("duplicate call trace")
        traces.add(call["trace_id"])
        by_owner[(call["run_id"], call["question_id"], call["arm"])].append(call)
    questions, consumed = {}, set()
    for batch in batches:
        counts = retrieval_counts(batch["events"])
        for report in batch["reports"]:
            qid, offset = report["question_id"], report["offset"]
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or not 0 <= offset < len(expected_ids)
                or expected_ids[offset] != qid
                or offset in questions
            ):
                raise ValueError("question ID/offset mismatch or duplicate")
            if set(report["arms"]) - set(ARMS):
                raise ValueError("unexpected arm")
            row = {
                "question_id": qid,
                "offset": offset,
                "run_id": batch["run_id"],
                "complete_pair": report["complete_pair"],
                "arms": {},
            }
            for arm in ARMS:
                outcome = report["arms"].get(arm)
                arm_calls = by_owner.get((batch["run_id"], qid, arm), [])
                if outcome is None or outcome["status"] == "not_executed":
                    if arm_calls or (qid, arm) in counts or (outcome and outcome.get("calls")):
                        raise ValueError("unreported executed arm")
                    row["arms"][arm] = {"status": "not_executed", "usage": None}
                    continue
                if outcome["status"] not in {"completed", "failed"}:
                    raise ValueError("unexpected arm status")
                if (qid, arm) not in counts:
                    raise ValueError("missing arm_start")
                reported_traces = [c["trace_id"] for c in outcome["calls"]]
                if set(reported_traces) != {c["trace_id"] for c in arm_calls} or len(
                    reported_traces
                ) != len(arm_calls):
                    raise ValueError("report call ownership mismatch")
                consumed.update(reported_traces)
                rounds = counts[(qid, arm)]
                if (
                    outcome["status"] == "completed"
                    and outcome["result"]["retrieval_rounds"] != rounds
                ):
                    raise ValueError("reported retrieval rounds mismatch")
                roles = {}
                for role in ROLES:
                    selected = [c for c in arm_calls if c["role"] == role]
                    roles[role] = {
                        "status": "called" if selected else "not_called",
                        "usage": usage(selected) if selected else None,
                        "trace_ids": [c["trace_id"] for c in selected],
                    }
                if any(c["role"] not in ROLES for c in arm_calls):
                    raise ValueError("unrecognized role")
                row["arms"][arm] = {
                    "status": outcome["status"],
                    "error_type": outcome.get("error_type"),
                    "elapsed_seconds": outcome["elapsed_seconds"],
                    "retrieval_rounds_observed": rounds,
                    "retrieval_rounds_complete": outcome["status"] == "completed",
                    "usage": usage(arm_calls),
                    "roles": roles,
                }
            if row["complete_pair"] != all(
                arm["status"] == "completed" for arm in row["arms"].values()
            ):
                raise ValueError("complete_pair does not match arm outcomes")
            questions[offset] = row
    if set(questions) != set(range(len(expected_ids))):
        raise ValueError("incomplete cohort")
    if consumed != traces:
        raise ValueError("orphan call outside reported cohort")
    ordered = [questions[i] for i in range(len(expected_ids))]
    by_arm = {}
    paired_qids = {q["question_id"] for q in ordered if q["complete_pair"]}
    for arm in ARMS:
        outcomes = [q["arms"][arm] for q in ordered]
        by_arm[arm] = {
            "status_counts": dict(Counter(o["status"] for o in outcomes)),
            "usage_all_attempts": usage([c for c in calls if c["arm"] == arm]),
            "usage_complete_pairs": usage(
                [c for c in calls if c["arm"] == arm and c["question_id"] in paired_qids]
            ),
            "elapsed_seconds_all_attempts": sum(
                o["elapsed_seconds"] for o in outcomes if o["status"] != "not_executed"
            ),
            "retrieval_rounds_observed_all_attempts": sum(
                o["retrieval_rounds_observed"] for o in outcomes if o["status"] != "not_executed"
            ),
            "roles": {
                role: usage([c for c in calls if c["arm"] == arm and c["role"] == role])
                for role in ROLES
            },
        }
    return ordered, {
        "question_count": len(ordered),
        "complete_pairs": len(paired_qids),
        "usage_all_attempts": usage(calls),
        "arms": by_arm,
    }


def export(runs_root, manifest_path, output):
    runs_root, output = runs_root.resolve(), output.resolve()
    if output == runs_root or not output.is_relative_to(runs_root):
        raise ValueError("output must be a new directory inside runs root")
    if output.exists():
        raise FileExistsError(output)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if sha(raw) != EXPECTED_MANIFEST or len(manifest["question_ids"]) != 500:
        raise ValueError("not the frozen shared500 manifest")
    # Reuse the existing transport/journal/ledger audit instead of a second auditor.
    spec = importlib.util.spec_from_file_location(
        "shared500_transport_audit", Path(__file__).with_name("audit_shared500.py")
    )
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    audit, calls = auditor.audit_series(runs_root)
    if audit["issues"] or audit["excluded_unclosed_batches"]:
        raise ValueError("source transport audit did not pass or has unclosed batches")
    prices = {
        (b["price_input_cny_per_million"], b["price_output_cny_per_million"])
        for b in audit["closed_batches"]
    }
    if prices != {(0.2, 0.8)}:
        raise ValueError("not the frozen shared500 price schedule")

    def verified_read(path):
        data = path.read_bytes()
        if audit["source_file_hashes"].get(str(path)) != sha(data):
            raise ValueError(f"source changed or not audited: {path}")
        return data

    batches = []
    for batch in audit["closed_batches"]:
        directory = runs_root / batch["run_id"]
        batches.append(
            {
                "run_id": batch["run_id"],
                "reports": json.loads(verified_read(directory / "reports.json")),
                "events": [
                    json.loads(line)
                    for line in verified_read(directory / "events.jsonl").splitlines()
                    if line.strip()
                ],
            }
        )
    for call in calls:
        raw_audit = json.loads(verified_read(Path(call["audit_path"])))
        call["elapsed_seconds"] = raw_audit.get("elapsed_seconds")
        call["role"] = next(
            (
                role
                for role in ROLES
                if call["prompt_version"] == f"s2g-author-5d842a6-{role}-api-v1"
            ),
            None,
        )
    questions, summary = assemble(manifest["question_ids"], batches, calls)
    if len(calls) != 3350 or len(batches) != 28:
        raise ValueError("not the completed frozen 28-batch/3350-call series")
    summary.update(
        schema_version="shared500-per-question-cost-v1",
        created_utc=datetime.now(UTC).isoformat(),
        manifest_sha256=EXPECTED_MANIFEST,
        source_audit_status=audit["audit_status"],
        source_file_count=audit["source_file_count"],
        source_hash_map_sha256=sha(
            json.dumps(audit["source_file_hashes"], sort_keys=True).encode()
        ),
        sealed_batch_count=len(batches),
        price_input_cny_per_million=0.2,
        price_output_cny_per_million=0.8,
        price_checked_date="2026-09-27",
        price_source="https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
        limits=[
            "Token-based estimates, not provider-settled invoices; unknown totals remain null.",
            "All attempted calls, including failures, counted once; history/previous not added.",
            "A not-executed arm has null usage, not a successful zero-cost answer.",
            "Failed arms expose observed retrieval events, not a claim of a finished trajectory.",
            "Elapsed sums are serialized arm/API durations, not deployment wall time.",
            "No model calls, training, gold scoring or counterfactual early-stop answers.",
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "per_question_costs.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False, allow_nan=False) + "\n" for q in questions),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    total = summary["usage_all_attempts"]
    lines = [
        "# Shared500 逐题成本\n",
        "500题全部保留，492题完成双路运行；失败也计费，不重新调用API。\n",
        "| 方法 | 请求数 | 已知输入token | 已知输出token | 已知估价¥ | 未知费用调用 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, details in summary["arms"].items():
        u = details["usage_all_attempts"]
        lines.append(
            f"| {arm} | {u['api_requests']} | {u['known_input_tokens_subtotal']} | "
            f"{u['known_output_tokens_subtotal']} | {u['known_estimated_cost_cny_subtotal']:.8f} | "
            f"{u['unknown_estimated_cost_cny_calls']} |"
        )
    lines.extend(
        [
            "",
            f"合计{total['api_requests']}次真实请求，已知估价小计"
            f"¥{total['known_estimated_cost_cny_subtotal']:.8f}。未知费用不按零处理；"
            "reserved为预算预留，不是实际账单。",
            "",
            "每题详见per_question_costs.jsonl：按BASE/S2G，再按Judge/Extractor/Answer列出"
            "token、估价、未知次数、耗时、轮次和trace_id；summary.json包含全量和完整配对的独立小计。",
            "",
            "失败题的retrieval_rounds_observed仅是已写入的检索事件数；未执行arm的usage为null。"
            "角色not_called表示确实没有调用，不等于故障调用的用量已知为零。",
            "",
            "不要从四轮记录截断后直接声称两轮答案效果：预算改变后的最终回答必须另行冻结协议评价。",
        ]
    )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/hotpotqa/shared500_sep27_v1/manifest.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = export(args.runs_root, args.manifest, args.output)
    print(json.dumps(summary["usage_all_attempts"], ensure_ascii=False, indent=2))
    print("Output:", args.output.resolve())


if __name__ == "__main__":
    main()
