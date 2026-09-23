"""Read-only reconciliation of all five historical live runs, without double billing.

中文：历史汇总不是只承接最近一条链。逐个核对互不重叠的请求；失败费用
保持未知，但保留其完整预留。新实验不得覆盖旧账本或重放旧请求。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT_LEDGERS = (
    "2026-09-06_hotpot_pilot_v1/budget_report.json",
    "2026-09-06_pre_pilot_32_16_v1/final_budget.json",
    "2026-09-11_source_pool_64_v1/final_budget.json",
    "2026-09-14_fresh_baseline_32_v1/final_budget.json",
)
CONNECTIVITY_RUN = "2026-09-06_qwen_flash_connectivity_v1"


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("unknown or invalid historical amount")
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid historical amount")
    return float(value)


def reconcile_history(runs: Path, *, reviewed_extra_ledgers: tuple[str, ...] = ()) -> dict:
    """Consume root call ledgers, not aggregate copies or cumulative checkpoints.

    Scan live audit folders to detect omitted runs/requests. Only the one known
    connectivity call may lack a root ledger. This is intentionally not a general
    automatic resume mechanism; a subsequent paid run needs explicit review.
    """
    runs = Path(runs).resolve()
    roots, calls, seen_traces, seen_audits = [], [], set(), set()
    accounted_paths = set()
    if not isinstance(reviewed_extra_ledgers, tuple):
        raise TypeError("additional roots must be explicitly reviewed as a tuple")
    relative_roots = (*ROOT_LEDGERS, *reviewed_extra_ledgers)
    if len(set(relative_roots)) != len(relative_roots):
        raise ValueError("duplicate historical ledger root")
    for relative in relative_roots:
        path = (runs / relative).resolve()
        if not path.is_relative_to(runs):
            raise ValueError("historical ledger must remain inside runs")
        raw = path.read_bytes()
        ledger = json.loads(raw)
        rows = ledger.get("calls", [])
        if not rows or ledger.get("api_requests") != len(rows):
            raise ValueError("historical request totals mismatch")
        limits = ledger.get("limits", {})
        if (limits.get("input_per_million_cny"), limits.get("output_per_million_cny")) != (
            0.2,
            0.8,
        ):
            raise ValueError("historical price profile changed")
        if not math.isclose(
            sum(_number(row.get("reserved_cny")) for row in rows),
            _number(ledger.get("reserved_cny")),
            abs_tol=1e-10,
        ):
            raise ValueError("historical reservation totals mismatch")
        for row in rows:
            trace = row.get("trace_id")
            audit_name = Path(row.get("audit_path", "")).name
            if not trace or not audit_name or trace in seen_traces or audit_name in seen_audits:
                raise ValueError("overlapping or missing historical request identity")
            seen_traces.add(trace)
            seen_audits.add(audit_name)
            audit = path.parent / "api_audit" / audit_name
            data = json.loads(audit.read_bytes())
            accounted_paths.add(audit.resolve())
            if data.get("trace_id") != trace or data.get("transport_source") != "live_api":
                raise ValueError("historical ledger does not match live audit")
            if data.get("status") != row.get("status"):
                raise ValueError("historical request status mismatch")
            known = row.get("estimated_actual_cny")
            for field in ("input_tokens", "output_tokens", "api_requests"):
                if data.get(field) != row.get(field):
                    raise ValueError("historical usage differs from audit")
            if known is not None:
                known = _number(known)
                computed = (
                    _number(row.get("input_tokens")) * 0.2 + _number(row.get("output_tokens")) * 0.8
                ) / 1_000_000
                if not math.isclose(known, computed, abs_tol=1e-10):
                    raise ValueError("historical declared-price cost mismatch")
                if known > row["reserved_cny"]:
                    raise ValueError("historical reservation assumption exceeded")
            calls.append(
                {
                    "trace_id": trace,
                    "audit": str(audit),
                    "reserved_cny": row["reserved_cny"],
                    "known_estimated_cny": known,
                }
            )
        roots.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})

    # Do not trust a hand-written summary in place of the actual single request.
    for audit in sorted(runs.glob("*/api_audit/*.json")):
        data = json.loads(audit.read_bytes())
        if data.get("transport_source") != "live_api":
            continue
        if audit.resolve() in accounted_paths:
            continue
        if audit.name in seen_audits:
            raise ValueError("duplicate audit identity in another live run")
        if audit.parent.parent.name != CONNECTIVITY_RUN or data.get("trace_id") != (
            "connectivity-v1"
        ):
            raise ValueError("unaccounted live request; review the new root ledger")
        if data.get("status") != "completed" or data.get("api_requests") != 1:
            raise ValueError("connectivity accounting is incomplete")
        trace = data["trace_id"]
        if trace in seen_traces:
            raise ValueError("overlapping connectivity request")
        request = data["request"]
        input_tokens, output_tokens = data.get("input_tokens"), data.get("output_tokens")
        cost = (_number(input_tokens) * 0.2 + _number(output_tokens) * 0.8) / 1_000_000
        prompt_bytes = len(json.dumps(request["messages"], ensure_ascii=False).encode())
        reserve = ((prompt_bytes + 1024) * 0.2 + request["max_tokens"] * 0.8) / 1_000_000
        if cost > reserve:
            raise ValueError("connectivity reservation assumption exceeded")
        seen_traces.add(trace)
        seen_audits.add(audit.name)
        calls.append(
            {
                "trace_id": trace,
                "audit": str(audit),
                "reserved_cny": reserve,
                "known_estimated_cny": cost,
                "reservation_origin": "reconstructed conservative byte bound, not invoice",
            }
        )
        roots.append({"path": str(audit), "sha256": hashlib.sha256(audit.read_bytes()).hexdigest()})
    unknown = sum(row["known_estimated_cny"] is None for row in calls)
    return {
        "schema_version": "growrag-project-budget-reconciliation-v1",
        "roots": roots,
        "calls": calls,
        "prior_api_requests": len(calls),
        "prior_reserved_cny": sum(row["reserved_cny"] for row in calls),
        "prior_known_estimated_cny": sum(row["known_estimated_cny"] or 0 for row in calls),
        "prior_unknown_cost_requests": unknown,
        "prior_total_actual_cny": None
        if unknown
        else sum(row["known_estimated_cny"] for row in calls),
        "authorized_total_cny": 50.0,
        "authorization_date": "2026-09-20",
        "notice": "All known project live requests; estimates are not provider invoices. "
        "Unknown legacy request remains reserved and will not be replayed.",
    }
