"""Independent, offline transport audit. Writes only a new audit output directory.

No API/key/gold/model-output content is loaded into this report. JSON audit files
are read locally; only request metadata and token/cost counts are exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_MANIFEST = "2fb117c03b64be3ab12398a297c82ce3315600c14e33f5dcd56c5c6c5c59e064"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def same_number(left, right):
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
    )


def proven_component_stop(after, budget, reports, events):
    """Independently recognize only the explicitly evidenced post-API stop."""
    if (
        after.get("block_reason", "absent") is not None
        or budget.get("block_reason") != "author_component_failure"
        or after != {**budget, "block_reason": None}
        or not budget.get("calls")
        or len(events) < 4
    ):
        return False
    call = budget["calls"][-1]
    qid, arm = call["trace_id"].split("/")[1:3]
    tail = events[-4:]
    response, failed, complete, terminal = tail
    report = next((r for r in reports if r.get("question_id") == qid), {})
    outcome = report.get("arms", {}).get(arm, {})
    return (
        call.get("status") == "completed"
        and call.get("api_requests") == 1
        and [e.get("kind") for e in tail]
        == ["api_response", "arm_failed", "question_complete", "exit"]
        and all(e.get("arm") == arm for e in tail)
        and response.get("trace_id") == call["trace_id"]
        and failed.get("error_type") == "ValueError"
        and complete.get("question_id") == qid
        and complete.get("status") == terminal.get("status") == "failed"
        and outcome.get("status") == "failed"
        and outcome.get("error_type") == "ValueError"
    )


def audit_series(runs_root, *, series="500_v1", manifest_sha256=EXPECTED_MANIFEST):
    issues, warnings, skipped, batches, rows, hashes = [], [], [], [], [], {}
    traces, request_ids, response_ids, reported_qids = {}, {}, {}, {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = sha(raw)
        return json.loads(raw)

    def check(ok, code, where):
        if not ok:
            issues.append({"code": code, "where": where})

    def unique(mapping, value, where, code):
        if not value:
            return
        check(value not in mapping, code, where)
        mapping[value] = where

    for directory in sorted(Path(runs_root).glob(f"*_s2g_shared{series}_*")):
        if not directory.is_dir():
            continue
        required = [
            directory / n
            for n in ("launch_plan.json", "reports.json", "final_budget.json", "events.jsonl")
        ]
        if not all(path.is_file() for path in required):
            skipped.append({"run_id": directory.name, "reason": "not_finalized"})
            continue
        events_raw = required[-1].read_bytes()
        events = [json.loads(line) for line in events_raw.splitlines() if line.strip()]
        if not events or events[-1].get("kind") != "exit":
            skipped.append({"run_id": directory.name, "reason": "not_terminal"})
            continue
        hashes[str(required[-1])] = sha(events_raw)
        launch, reports, budget = [read(path) for path in required[:3]]
        scope = launch.get("series") == series and launch.get("manifest_sha256") == manifest_sha256
        check(scope, "wrong_series_or_manifest", directory.name)
        if not scope:
            continue
        check(launch.get("run_id") == directory.name, "run_identity_mismatch", directory.name)
        check(
            events[-1].get("status") in {"completed", "failed"},
            "invalid_exit_status",
            directory.name,
        )
        check(
            not any(e.get("kind") == "exit" for e in events[:-1]),
            "event_after_exit",
            directory.name,
        )
        calls = budget["calls"]
        canonical = {call["trace_id"]: call for call in calls}
        check(len(canonical) == len(calls), "duplicate_ledger_trace", directory.name)
        all_report_calls = {}
        for report in reports:
            qid = report["question_id"]
            unique(reported_qids, qid, directory.name, "question_reported_twice")
            check(qid in launch["question_ids"], "report_outside_plan", directory.name)
            for arm in report.get("arms", {}).values():
                for call in arm.get("calls", []):
                    trace = call["trace_id"]
                    check(trace not in all_report_calls, "duplicate_report_call", trace)
                    all_report_calls[trace] = call
        check(all_report_calls == canonical, "reports_ledger_mismatch", directory.name)
        journal_dir, audit_dir = directory / "request_journal", directory / "api_audit"
        check(
            journal_dir.is_dir() and audit_dir.is_dir(),
            "missing_raw_evidence_directory",
            directory.name,
        )
        raw_audits = {}
        for path in sorted(audit_dir.glob("*.json")):
            audit = read(path)
            trace = audit.get("trace_id")
            check(trace in canonical, "unaccounted_raw_api_audit", str(path))
            check(trace not in raw_audits, "duplicate_raw_api_audit", str(path))
            raw_audits[trace] = (path, audit)
        expected_journal = {
            f"{index:04d}_{kind}.json"
            for index in range(len(calls))
            for kind in ("intent", "after")
        }
        check(
            {p.name for p in journal_dir.iterdir()} == expected_journal,
            "dangling_or_missing_journal_record",
            directory.name,
        )
        price_in, price_out = (
            launch["price_input_cny_per_million"],
            launch["price_output_cny_per_million"],
        )
        batch_rows = []
        prior_reserve = 0.0
        post_transport_component_failure = False
        for index, call in enumerate(calls):
            trace = call["trace_id"]
            unique(traces, trace, directory.name, "trace_executed_twice")
            pieces = trace.split("/")
            check(
                len(pieces) >= 4
                and pieces[0] == directory.name
                and pieces[1] in launch["question_ids"],
                "trace_outside_plan",
                trace,
            )
            intent_path = journal_dir / f"{index:04d}_intent.json"
            after_path = journal_dir / f"{index:04d}_after.json"
            intent = None
            if intent_path.is_file() and after_path.is_file():
                intent, after = read(intent_path), read(after_path)
                check(intent.get("trace_id") == trace, "intent_trace_mismatch", trace)
                check(
                    intent.get("prompt_version") == call.get("prompt_version"),
                    "intent_prompt_mismatch",
                    trace,
                )
                check(after.get("calls") == calls[: index + 1], "after_prefix_mismatch", trace)
                check(
                    same_number(intent.get("prior_reserved_cny"), prior_reserve),
                    "intent_reserve_mismatch",
                    trace,
                )
                check(
                    same_number(intent.get("potential_reserved_cny"), call.get("reserved_cny")),
                    "intent_call_reserve_mismatch",
                    trace,
                )
                if index == len(calls) - 1:
                    if after != budget:
                        post_transport_component_failure = proven_component_stop(
                            after, budget, reports, events
                        )
                    check(
                        after == budget or post_transport_component_failure,
                        "last_after_final_mismatch",
                        trace,
                    )
            prior_reserve += call.get("reserved_cny", 0)
            if trace not in raw_audits:
                check(call.get("api_requests") == 0, "missing_transport_audit", trace)
                continue
            path, audit = raw_audits[trace]
            check(
                Path(call.get("audit_path", "")).resolve() == path.resolve(),
                "audit_pointer_mismatch",
                trace,
            )
            check(path.stem == sha(trace.encode()), "audit_filename_mismatch", trace)
            request, response = audit.get("request", {}), audit.get("response", {})
            if intent is not None:
                expected_fingerprint = sha(
                    json.dumps(
                        request.get("messages"), ensure_ascii=False, sort_keys=True, allow_nan=False
                    ).encode()
                )
                check(
                    intent.get("request_fingerprint") == expected_fingerprint,
                    "intent_request_fingerprint_mismatch",
                    trace,
                )
            check(
                sha(json.dumps(request, ensure_ascii=False, allow_nan=False).encode())
                == audit.get("request_sha256"),
                "request_body_hash_mismatch",
                trace,
            )
            check(audit.get("transport_source") == "live_api", "nonlive_transport", trace)
            check(audit.get("retry_count") == 0, "transport_retry", trace)
            check(
                audit.get("prompt_version") == call.get("prompt_version"),
                "audit_prompt_mismatch",
                trace,
            )
            for key in ("status", "api_requests", "input_tokens", "output_tokens"):
                check(call.get(key) == audit.get(key), f"audit_ledger_{key}_mismatch", trace)
            check(request.get("model") == launch.get("model"), "requested_model_mismatch", trace)
            returned_model = response.get("model")
            if response:
                check(returned_model == launch.get("model"), "returned_model_mismatch", trace)
                if call.get("status") == "completed" or "returned_model" in call:
                    check(
                        call.get("returned_model") == returned_model,
                        "ledger_returned_model_mismatch",
                        trace,
                    )
                usage = response.get("usage", {})
                check(
                    usage.get("prompt_tokens") == audit.get("input_tokens"),
                    "response_input_tokens_mismatch",
                    trace,
                )
                check(
                    usage.get("completion_tokens") == audit.get("output_tokens"),
                    "response_output_tokens_mismatch",
                    trace,
                )
            choices = response.get("choices", [])
            finish = (
                choices[0].get("finish_reason") if len(choices) == 1 else audit.get("finish_reason")
            )
            if call.get("status") == "completed":
                check(
                    audit.get("http_status") == 200 and audit.get("network_attempted") is True,
                    "completed_without_http_200",
                    trace,
                )
                check(
                    bool(audit.get("request_id")) and bool(response.get("id")),
                    "completed_missing_provider_id",
                    trace,
                )
                check(finish == audit.get("finish_reason"), "finish_reason_mismatch", trace)
                allowed_length = (
                    call.get("prompt_version") == "s2g-author-5d842a6-answer-api-v1"
                    and audit.get("accepted_truncated_rationale") is True
                )
                check(
                    finish == "stop" or (finish == "length" and allowed_length),
                    "unsupported_finish_reason",
                    trace,
                )
            unique(request_ids, audit.get("request_id"), trace, "duplicate_provider_request_id")
            unique(response_ids, response.get("id"), trace, "duplicate_provider_response_id")
            input_tokens, output_tokens = audit.get("input_tokens"), audit.get("output_tokens")
            estimate = None
            if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                estimate = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
                check(
                    same_number(call.get("estimated_actual_cny"), estimate),
                    "token_price_mismatch",
                    trace,
                )
            else:
                check(
                    call.get("estimated_actual_cny") is None,
                    "cost_claim_without_token_counts",
                    trace,
                )
            row = {
                "run_id": directory.name,
                "trace_id": trace,
                "question_id": pieces[1],
                "arm": pieces[2],
                "prompt_version": call.get("prompt_version"),
                "status": call.get("status"),
                "api_requests": call.get("api_requests"),
                "requested_model": request.get("model"),
                "returned_model": returned_model,
                "http_status": audit.get("http_status"),
                "finish_reason": finish,
                "accepted_truncated_rationale": audit.get("accepted_truncated_rationale", False),
                "request_id": audit.get("request_id"),
                "response_id": response.get("id"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "recomputed_estimate_cny": estimate,
                "reserved_cny": call.get("reserved_cny"),
                "audit_path": str(path),
                "audit_file_sha256": hashes[str(path)],
                "request_sha256": audit.get("request_sha256"),
                "response_sha256": audit.get("response_sha256"),
            }
            batch_rows.append(row)
        check(
            budget.get("api_requests") == sum(c.get("api_requests", 0) for c in calls),
            "budget_request_total_mismatch",
            directory.name,
        )
        check(
            same_number(budget.get("reserved_cny"), prior_reserve),
            "budget_reserve_total_mismatch",
            directory.name,
        )
        known_batch = all(
            c.get("input_tokens") is not None and c.get("output_tokens") is not None for c in calls
        )
        for name, field in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("estimated_actual_cny", "estimated_actual_cny"),
        ):
            expected_total = sum(c.get(field, 0) for c in calls) if known_batch else None
            check(
                budget.get(name) is None
                if expected_total is None
                else same_number(budget.get(name), expected_total),
                f"budget_{name}_total_mismatch",
                directory.name,
            )
        rows.extend(batch_rows)
        batches.append(
            {
                "run_id": directory.name,
                "terminal_status": events[-1]["status"],
                "planned_questions": len(launch["question_ids"]),
                "reported_questions": len(reports),
                "ledger_calls": len(calls),
                "audited_calls": len(batch_rows),
                "continuation_of": launch.get("continuation_of"),
                "protocol": launch.get("protocol"),
                "generation_profile": launch.get("generation_profile"),
                "model": launch.get("model"),
                "price_input_cny_per_million": price_in,
                "price_output_cny_per_million": price_out,
                "post_transport_component_failure": post_transport_component_failure,
                "component_failure_trace": calls[-1]["trace_id"]
                if post_transport_component_failure
                else None,
            }
        )
    # Re-read every included file to reject a changing snapshot, never lock running output.
    for name, digest in hashes.items():
        check(sha(Path(name).read_bytes()) == digest, "source_changed_during_audit", name)
    known = [r for r in rows if r["recomputed_estimate_cny"] is not None]
    unknown = [r for r in rows if r["recomputed_estimate_cny"] is None]
    profiles = {(b["protocol"], b["generation_profile"], b["model"]) for b in batches}
    check(len(profiles) <= 1, "mixed_generation_profiles", series)
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "series": series,
        "manifest_sha256": manifest_sha256,
        "audit_status": "passed" if not issues else "issues_found",
        "closed_batches": batches,
        "excluded_unclosed_batches": skipped,
        "issues": issues,
        "warnings": warnings,
        "reported_questions": len(reported_qids),
        "audited_call_count": len(rows),
        "api_requests": sum(r["api_requests"] or 0 for r in rows),
        "status_distribution": dict(Counter(r["status"] for r in rows)),
        "returned_model_distribution": dict(Counter(str(r["returned_model"]) for r in rows)),
        "http_status_distribution": dict(Counter(str(r["http_status"]) for r in rows)),
        "finish_reason_distribution": dict(Counter(str(r["finish_reason"]) for r in rows)),
        "accepted_truncated_rationale_count": sum(r["accepted_truncated_rationale"] for r in rows),
        "post_transport_component_failure_batches": [
            b["run_id"] for b in batches if b["post_transport_component_failure"]
        ],
        "unique_request_ids": len(request_ids),
        "unique_response_ids": len(response_ids),
        "known_input_tokens_subtotal": sum(r["input_tokens"] or 0 for r in rows),
        "known_output_tokens_subtotal": sum(r["output_tokens"] or 0 for r in rows),
        "known_estimate_cny_subtotal": sum(r["recomputed_estimate_cny"] for r in known),
        "unknown_cost_call_count": len(unknown),
        "unknown_cost_traces": [r["trace_id"] for r in unknown],
        "unknown_cost_reserved_cny": sum(r["reserved_cny"] or 0 for r in unknown),
        "total_estimate_cny": None if unknown else sum(r["recomputed_estimate_cny"] for r in known),
        "scope": "Only each included final_budget.calls once, including failed calls; "
        "historical_budget/previous ignored.",
        "limits": [
            "Provider-settled billing is not available; estimates are not invoices.",
            "A failed transport may still be billed; unknown is never treated as zero.",
            "Response raw bytes were not retained, so response_sha256 is recorded "
            "but not independently recomputed.",
            "Local logs provide execution evidence, not independent attestation by the provider.",
        ],
        "source_file_count": len(hashes),
        "audit_script_sha256": sha(Path(__file__).read_bytes()),
        "source_file_hashes": hashes,
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root", type=Path, default=Path(__file__).resolve().parents[2] / "runs"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = audit_series(args.runs_root)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "calls.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    text = (
        f"# S2G 500 开发系列独立 API 审计\n\n状态：{summary['audit_status']}。"
        f"仅包含 {len(summary['closed_batches'])} 个已封口批次；"
        f"排除 {len(summary['excluded_unclosed_batches'])} 个未封口批次。\n\n"
        f"实际请求 {summary['api_requests']}；"
        f"已知输入 token {summary['known_input_tokens_subtotal']}；"
        f"已知输出 token {summary['known_output_tokens_subtotal']}。\n\n"
        f"已知估价小计 ¥{summary['known_estimate_cny_subtotal']:.8f}；"
        f"未知费用请求 {summary['unknown_cost_call_count']}。"
        "未知不是零，不能给出完整实际总费用。\n\n"
        "每条请求的模型、HTTP、停止原因、供应商 ID、token、估价见 calls.jsonl。"
        "history/previous 只作上下文，不重复累计。未读取 gold、未评估答案、未调用 API。\n\n"
        "注意：响应原始字节没有保留，因此 response_sha256 仅留存，无法重新计算验证；"
        "这份审计证明本地记录的一致性，不等于供应商账单或第三方真实性认证。\n"
    )
    (args.output / "README.md").write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "audit_status",
                    "reported_questions",
                    "audited_call_count",
                    "api_requests",
                    "known_estimate_cny_subtotal",
                    "unknown_cost_call_count",
                )
            },
            indent=2,
        )
    )
    print("issues:", len(summary["issues"]), "output:", args.output)
    return 0 if not summary["issues"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
