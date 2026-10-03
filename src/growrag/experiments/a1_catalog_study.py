"""Prospective catalog-v3 development interface regression, never a blind retest.

This is an additive once-only runner. Old quote-v2 responses/results are immutable
and are never repaired or rerun here. The 32 input/label/order rows are reused and
previously exposed: nominal agreement is not new semantic generalization evidence.
The new wire is replayed from actual HTTP before its derived v2 report reaches the
unchanged A1 gate. No natural questions, retrieval, answers, or training live here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from . import a1_observer_refs as observer
from .a0_v2_feedback import _same_amount, audit_http, totals
from .a1_catalog_budget import PREFIX, PROTOCOL, reviewed_history
from .a1_conditions import (
    ConditionContractError,
    canonical_bytes,
    decide,
    prepare_payload,
)
from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_dev_manifest import _sha
from .pre_pilot import write_json
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_operator_study import serial_lock
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient

FIXTURE = "experiments/fixtures/a1_conditions_v3.json"
FIXTURE_SHA = "c42a1aaa2428fac192fad9964eba0094cca8d8e7767e2bd12da03b7870d14454"
NOTE = "docs/experiments/2026-10-03_A1_catalog_v3预登记.md"
FREEZE = "runs/a1_catalog_freeze_v1.json"
RUN_ID = PREFIX + "catalog_v1"
OUTPUT = "runs/" + RUN_ID
FEEDBACK = "runs/a1_catalog_feedback_v1"
KEY_VARIABLE = "GROWRAG_A1_CATALOG_API_KEY"
PROMPT_VERSION = observer.PROMPT_VERSION
CONFIG = {
    "protocol": PROTOCOL,
    "model": PILOT_MODEL,
    "max_output_tokens": 2048,
    "temperature": 0,
    "top_p": 1,
    "enable_thinking": False,
    "json_object_mode": True,
    "json_schema_mode": False,
    "output_limit_parameter": "max_tokens",
    "prompt_version": PROMPT_VERSION,
    "project_cap_cny": 200.0,
    "series_cap_cny": 1.0,
    "planned_rows": 32,
    "unique_inputs": 30,
    "retry": False,
    "gold_loaded": False,
    "memory_updated": False,
    "cost_notice": "Declared input0.2/output0.8 CNY/million; not provider invoice",
    "suite_role": "development_interface_regression_not_blind_test",
    "catalog_version": observer.CATALOG_VERSION,
    "conversion_version": observer.CONVERSION_VERSION,
}


def input_sha(payload):
    """A1 input uses compact canonical JSON, unlike legacy journal fingerprint."""
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def load_fixture(project):
    path = project / FIXTURE
    if _sha(path) != FIXTURE_SHA:
        raise ValueError("registered synthetic fixture changed")
    fixture = json.loads(path.read_bytes())
    rows = fixture["rows"]
    if len(rows) != 32 or len({r["row_id"] for r in rows}) != 32:
        raise ValueError("fixture row identity changed")
    for row in rows:
        if input_sha(row["input"]) != fixture["integrity"]["input_sha256_by_row_id"][row["row_id"]]:
            raise ValueError("fixture input hash differs")
        prepare_payload(row["input"])
    return rows


def protocol_identity(project):
    """Expected annotations never appear in these model-visible messages."""
    rows = load_fixture(project)
    return {
        "configuration": CONFIG,
        "fixture_sha256": FIXTURE_SHA,
        "source_sha256": source_snapshot(project)["sha256"],
        "protocol_note_sha256": _sha(project / NOTE),
        "prompt_sha256": observer.PROMPT_SHA256,
        "catalog_sha256": {
            r["row_id"]: input_sha(observer.catalog(prepare_payload(r["input"]))) for r in rows
        },
        "row_ids": [r["row_id"] for r in rows],
        "messages_sha256": {
            r["row_id"]: input_sha(observer.messages(prepare_payload(r["input"]))) for r in rows
        },
    }


def freeze(project):
    if (project / FREEZE).exists() or (project / OUTPUT).exists():
        raise FileExistsError("A1 catalog phase already frozen/started")
    git = _git_state()
    if not git["commit"] or git["worktree_dirty"]:
        raise ValueError("commit reviewed implementation before freezing")
    value = {**protocol_identity(project), "git": git, "api_calls": 0}
    write_json(project / FREEZE, value)
    return {"freeze_path": FREEZE, "freeze_sha256": _sha(project / FREEZE)}


def load_freeze(project, sha):
    if type(sha) is not str or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("external freeze SHA required")
    if _sha(project / FREEZE) != sha:
        raise ValueError("freeze identity changed")
    value = json.loads((project / FREEZE).read_bytes())
    if any(value.get(k) != v for k, v in protocol_identity(project).items()):
        raise ValueError("source, prompt or protocol differs from freeze")
    return value


def attach_catalog_audit(record, audit):
    """Persist raw-v3/catalog/mapped-v2 separately, including failed conversion."""
    value = audit.to_dict()
    record["catalog_audit"] = value
    record["mapped_v2_raw_text"] = value["converted_raw_text"]
    record["mapped_v2_sha256"] = value["converted_response_sha256"]


def verify_record_request(record, prepared, *, required):
    """A saved request must exactly match this row, not merely share an alias."""
    expected = {
        "messages": observer.messages(prepared),
        "catalog": observer.catalog(prepared),
    }
    expected["messages_sha256"] = input_sha(expected["messages"])
    expected["catalog_sha256"] = input_sha(expected["catalog"])
    present = set(expected) & set(record)
    if required or present:
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("saved catalog/messages or their SHA differs from prepared input")
    if "raw_content" in record and (
        type(record["raw_content"]) is not str
        or record.get("raw_response_sha256")
        != hashlib.sha256(record["raw_content"].encode("utf-8")).hexdigest()
    ):
        raise ValueError("raw catalog response SHA differs")


def verify_catalog_replay(record, audit):
    expected = audit.to_dict()
    if (
        record.get("catalog_audit") != expected
        or record.get("mapped_v2_raw_text") != expected["converted_raw_text"]
        or record.get("mapped_v2_sha256") != expected["converted_response_sha256"]
    ):
        raise ValueError("offline catalog audit/mapped-v2 replay differs")


def observe_rows(rows, client, directory, guard, *, terminal=None):
    """Each row gets one request. Return full completion/missing accounting."""
    terminal = [] if terminal is None else terminal
    stop = None
    for index, row in enumerate(rows):
        if stop:
            terminal.append({"row_id": row["row_id"], "status": "not_attempted"})
            continue
        record = {
            "row_id": row["row_id"],
            "sequence": index,
            "trace_id": f"{RUN_ID}/row/{index:02d}",
            "input_sha256": input_sha(row["input"]),
            "status": "started",
        }
        try:
            guard()
            prepared = prepare_payload(row["input"])
            messages = observer.messages(prepared)
            record["messages"] = messages
            record["messages_sha256"] = input_sha(messages)
            record["catalog"] = observer.catalog(prepared)
            record["catalog_sha256"] = input_sha(record["catalog"])
            response = client.complete(
                messages, trace_id=record["trace_id"], prompt_version=PROMPT_VERSION
            )
            record["raw_content"] = response.content
            record["raw_response_sha256"] = hashlib.sha256(
                response.content.encode("utf-8")
            ).hexdigest()
            report, audit = observer.resolve(response.content, prepared)
            audit.verify_report(report)
            attach_catalog_audit(record, audit)
            record["report"] = report.to_dict()
            record["status"] = "completed"
            guard()
        except BaseException as error:
            record["status"] = "failed"
            record["failure_type"] = type(error).__name__
            if isinstance(error, observer.CatalogContractError) and error.audit is not None:
                attach_catalog_audit(record, error.audit)
            stop = (
                "observer_contract_failure"
                if isinstance(error, ConditionContractError)
                else type(error).__name__
            )
        path = directory / f"row_{index:02d}.json"
        entry = {
            "row_id": row["row_id"],
            "status": "failed",
            "path": path.name,
            "artifact_status": "not_sealed",
        }
        terminal.append(entry)  # Caller retains this prefix even if filesystem/stdout fails.
        write_json(path, record)
        entry.pop("artifact_status")
        entry.update(status=record["status"], sha256=_sha(path))
        print(
            json.dumps({"a1_catalog_row": index + 1, "status": record["status"], "stop": stop}),
            flush=True,
        )
    return terminal, stop


def final_budget_report(client, cap):
    """An interrupted HTTP attempt without a completed ledger row is not free."""
    if client is None:
        return {
            "api_requests": 0,
            "calls": [],
            "reserved_cny": 0,
            "estimated_actual_cny": 0,
            "limits": asdict(PriceLimits(budget_cny=cap)),
        }
    report = client.report()
    calls = report["calls"]
    if report["api_requests"] != len(calls) or not math.isclose(
        report["reserved_cny"], sum(c["reserved_cny"] for c in calls), abs_tol=1e-10
    ):
        report.update(
            estimated_actual_cny=None,
            input_tokens=None,
            output_tokens=None,
            reconciliation_required=True,
            block_reason="unaccounted_interrupted_attempt",
        )
    return report


def run(project, sha, *, allow_network=False):
    frozen = load_freeze(project, sha)
    rows = load_fixture(project)
    if not allow_network:
        return {"dry_run": True, "planned_rows": len(rows), "api_calls": 0}
    output = project / OUTPUT
    claim = project / "runs" / f"{RUN_ID}.claim.json"
    if output.exists() or claim.exists():
        raise FileExistsError("catalog batch already claimed; no retry")
    if _git_state()["worktree_dirty"]:
        raise ValueError("commit code before paid calls")
    with serial_lock(project / "runs"):
        history = reviewed_history(project / "runs")
        cap = min(
            CONFIG["series_cap_cny"], CONFIG["project_cap_cny"] - history["prior_reserved_cny"]
        )
        if cap <= 0:
            raise ValueError("project conservative budget exhausted")
        settings = read_local_bailian_settings(project / "qwenAPI.md")
        host = urlsplit(settings.base_url).hostname or ""
        if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
            raise ValueError("endpoint does not match declared Beijing price")
        plan = {
            **CONFIG,
            "run_id": RUN_ID,
            "freeze_sha256": sha,
            "prior_reserved_cny": history["prior_reserved_cny"],
            "subcap_cny": cap,
            "question_ids": [],
            "row_ids": frozen["row_ids"],
        }
        write_json(claim, {**plan, "plan_sha256": fingerprint(plan)})
        output.mkdir(exist_ok=False)
        write_json(output / "launch_plan.json", plan)
        write_json(output / "prior_budget.json", history)
        write_json(output / "source_snapshot.json", source_snapshot(project))
        client, stop, terminal = None, None, []
        previous = os.environ.get(KEY_VARIABLE)
        try:
            os.environ[KEY_VARIABLE] = settings.api_key
            config = ChatConfig(
                settings.base_url,
                PILOT_MODEL,
                KEY_VARIABLE,
                max_calls=32,
                max_output_tokens=2048,
                timeout_seconds=60,
                output_limit_parameter="max_tokens",
                enable_thinking=False,
                temperature=0,
                top_p=1,
                json_object_mode=True,
            )
            client = DurableBudgetClient(
                LiveChatClient(config, output / "api_audit", allow_network=True),
                PriceLimits(budget_cny=cap, max_prompt_bytes=30000, max_elapsed_seconds=1800),
                output / "request_journal",
            )
            terminal, stop = observe_rows(
                rows, client, output, lambda: load_freeze(project, sha), terminal=terminal
            )
        except BaseException as error:
            stop = type(error).__name__
        finally:
            if previous is None:
                os.environ.pop(KEY_VARIABLE, None)
            else:
                os.environ[KEY_VARIABLE] = previous
            terminal.extend(
                {"row_id": r["row_id"], "status": "not_attempted"} for r in rows[len(terminal) :]
            )
            budget = final_budget_report(client, cap)
            write_json(output / "final_budget.json", budget)
            value = {
                "protocol": PROTOCOL,
                "status": "stopped" if stop else "completed",
                "stop_reason": stop,
                "rows": terminal,
                "freeze_sha256": sha,
                "budget_sha256": _sha(output / "final_budget.json"),
                "launch_sha256": _sha(output / "launch_plan.json"),
                "prior_sha256": _sha(output / "prior_budget.json"),
                "source_snapshot_sha256": _sha(output / "source_snapshot.json"),
                "claim_sha256": _sha(claim),
                "gold_loaded": False,
                "memory_updated": False,
            }
            write_json(output / "TERMINAL.json", value)
    return {
        "status": value["status"],
        "stop_reason": stop,
        "terminal_sha256": _sha(output / "TERMINAL.json"),
        "output": OUTPUT,
    }


def gate_summary(rows, reports):
    """Grade nominal synthetic policy choices, not RAG correctness or causal gain."""
    details, match, retained, rejected, uncertain = [], 0, 0, 0, 0
    for row in rows:
        report = reports.get(row["row_id"])
        if report is None:
            details.append({"row_id": row["row_id"], "status": "missing"})
            continue
        main = decide(report).to_dict()["decision"]
        expected = row["expected"]["contradiction_only"]
        match += main == expected
        retained += row["variant"] == "A" and main != "reject"
        rejected += row["variant"] == "B" and expected == "reject" and main == "reject"
        uncertain += (
            row["variant"] == "B" and expected == "allow_uncertain" and main == "allow_uncertain"
        )
        details.append(
            {
                "row_id": row["row_id"],
                "status": "scored",
                "expected": row["expected"],
                "observed": report.to_dict(),
                "policies": {
                    **{
                        kind: decide(report, dimensions=(kind,)).to_dict()
                        for kind in ("T", "R", "P0", "P1")
                    },
                    "all": decide(report).to_dict(),
                    "strict": decide(report, strict_unknown=True).to_dict(),
                    "ungated": decide(report, dimensions=()).to_dict(),
                },
                "main_matches": main == expected,
            }
        )
    gate = {
        "completed": len(reports),
        "main_matches": match,
        "A_retained": retained,
        "conflict_B_rejected": rejected,
        "unknown_B_preserved": uncertain,
    }
    gate["passed"] = (
        len(reports) == 32 and match >= 29 and retained >= 15 and rejected >= 12 and uncertain == 3
    )
    return gate, details


def audit_journal(directory, ledger, captured, track, *, refusal_prefixes=()):
    """A1 journal contract; never dispatch its prompt to an A0 schema registry."""
    journal = directory / "request_journal"
    present = {p.name for p in journal.glob("*.json")} if journal.exists() else set()
    names = {f"{i:04d}_{s}.json" for i in range(len(captured)) for s in ("intent", "after")}
    tail = {f"{len(captured):04d}_{s}.json" for s in ("intent", "after")}
    refusal = present == names | tail
    if present != names and not refusal:
        raise ValueError("journal coverage mismatch")
    prior, after = 0.0, None
    for i, (call, request, _) in enumerate(captured):
        intent = json.loads(track(journal / f"{i:04d}_intent.json").read_bytes())
        after = json.loads(track(journal / f"{i:04d}_after.json").read_bytes())
        if (
            intent.get("trace_id") != call["trace_id"]
            or intent.get("prompt_version") != PROMPT_VERSION
            or intent.get("request_fingerprint") != fingerprint(request["messages"])
            or intent.get("status") != "pending_no_automatic_retry"
            or after.get("calls") != ledger["calls"][: i + 1]
        ):
            raise ValueError("journal request identity differs")
        _same_amount(intent.get("prior_reserved_cny"), prior, "journal prior differs")
        _same_amount(intent.get("potential_reserved_cny"), call["reserved_cny"], "reserve differs")
        values = totals(after["calls"])
        for key in ("api_requests", "reserved_cny", "estimated_actual_cny"):
            _same_amount(after.get(key), values[key], "journal total differs")
        prior += call["reserved_cny"]
    if refusal:
        intent = json.loads(track(journal / f"{len(captured):04d}_intent.json").read_bytes())
        after = json.loads(track(journal / f"{len(captured):04d}_after.json").read_bytes())
        if (
            after.get("block_reason")
            not in {"elapsed_time_limit", "estimated_budget_limit", "prompt_size_limit"}
            or after.get("calls") != ledger["calls"]
            or intent.get("prompt_version") != PROMPT_VERSION
            or intent.get("status") != "pending_no_automatic_retry"
            or intent.get("trace_id") != f"{RUN_ID}/row/{len(captured):02d}"
            or not re.fullmatch(r"[0-9a-f]{64}", intent.get("request_fingerprint", ""))
        ):
            raise ValueError("unaccounted no-request refusal")
        _same_amount(intent.get("prior_reserved_cny"), prior, "refusal prior differs")
        _same_amount(after.get("reserved_cny"), prior, "unsent request charged")
        _same_amount(after.get("api_requests"), len(captured), "refusal HTTP count changed")
    if after is not None and after != ledger:
        raise ValueError("last journal differs from final ledger")
    return refusal


def score(project, freeze_sha, terminal_sha):
    """Rebuild all observations from sealed raw responses before reading expected."""
    frozen = load_freeze(project, freeze_sha)
    output = project / OUTPUT
    if _sha(output / "TERMINAL.json") != terminal_sha:
        raise ValueError("external terminal SHA mismatch")
    terminal = json.loads((output / "TERMINAL.json").read_bytes())
    ledger = json.loads((output / "final_budget.json").read_bytes())
    if (
        terminal["budget_sha256"] != _sha(output / "final_budget.json")
        or terminal["freeze_sha256"] != freeze_sha
    ):
        raise ValueError("terminal budget/freeze identity differs")
    if (
        terminal.get("protocol") != PROTOCOL
        or terminal.get("status") not in {"stopped", "completed"}
        or terminal.get("gold_loaded") is not False
        or terminal.get("memory_updated") is not False
        or (terminal["status"] == "completed") != (terminal.get("stop_reason") is None)
    ):
        raise ValueError("invalid terminal metadata")
    for name, key in (
        ("launch_plan.json", "launch_sha256"),
        ("prior_budget.json", "prior_sha256"),
        ("source_snapshot.json", "source_snapshot_sha256"),
    ):
        if _sha(output / name) != terminal.get(key):
            raise ValueError("terminal metadata seal differs")
    claim_path = project / "runs" / f"{RUN_ID}.claim.json"
    if _sha(claim_path) != terminal.get("claim_sha256"):
        raise ValueError("claim seal differs")
    plan = json.loads((output / "launch_plan.json").read_bytes())
    if any(plan.get(k) != v for k, v in CONFIG.items()):
        raise ValueError("launch configuration drift")
    if (
        plan.get("run_id") != RUN_ID
        or plan.get("freeze_sha256") != freeze_sha
        or plan.get("row_ids") != frozen["row_ids"]
        or plan.get("question_ids") != []
        or json.loads(claim_path.read_bytes()) != {**plan, "plan_sha256": fingerprint(plan)}
    ):
        raise ValueError("claim/launch binding differs")
    prior = json.loads((output / "prior_budget.json").read_bytes())
    _same_amount(plan.get("prior_reserved_cny"), prior["prior_reserved_cny"], "prior differs")
    _same_amount(
        plan.get("subcap_cny"),
        min(CONFIG["series_cap_cny"], CONFIG["project_cap_cny"] - prior["prior_reserved_cny"]),
        "subcap differs",
    )
    snapshot = json.loads((output / "source_snapshot.json").read_bytes())
    if (
        snapshot.get("sha256") != frozen["source_sha256"]
        or fingerprint(snapshot["files"]) != snapshot["sha256"]
    ):
        raise ValueError("source snapshot differs")
    values = totals(ledger["calls"])
    for key in ("api_requests", "reserved_cny", "estimated_actual_cny"):
        _same_amount(ledger.get(key), values[key], "final ledger total differs")
    if values["reserved_cny"] > plan["subcap_cny"] or len(ledger["calls"]) > 32:
        raise ValueError("series cap/count exceeded")
    hashes = {}

    def track(path):
        hashes[path.relative_to(project).as_posix()] = _sha(path)
        return path

    seen = set()
    captured = [audit_http(c, output, plan, track, seen, RUN_ID + "/") for c in ledger["calls"]]
    refusal = audit_journal(output, ledger, captured, track)
    reports = {}
    rows = load_fixture(project)
    calls_by_trace = {c[0]["trace_id"]: c for c in captured}
    used = set()
    failed = False
    if [r["row_id"] for r in terminal["rows"]] != [r["row_id"] for r in rows]:
        raise ValueError("terminal row sequence changed")
    for index, (row, entry) in enumerate(zip(rows, terminal["rows"], strict=True)):
        if entry.get("status") not in {"completed", "failed", "not_attempted"}:
            raise ValueError("unknown row status")
        if entry["status"] == "not_attempted":
            if set(entry) != {"row_id", "status"}:
                raise ValueError("unattempted row has fabricated result")
            failed = True
            continue
        if failed:
            raise ValueError("rows executed after stop")
        if entry.get("path") != f"row_{index:02d}.json":
            raise ValueError("row path differs from planned sequence")
        path = track(output / entry["path"])
        if _sha(path) != entry["sha256"]:
            raise ValueError("row seal differs")
        record = json.loads(path.read_bytes())
        prepared = prepare_payload(row["input"])
        if record["input_sha256"] != input_sha(row["input"]) or record["row_id"] != row["row_id"]:
            raise ValueError("row/input binding differs")
        if record.get("sequence") != index or record.get("trace_id") != f"{RUN_ID}/row/{index:02d}":
            raise ValueError("row sequence/trace binding differs")
        call = calls_by_trace.get(record["trace_id"])
        if call is not None:
            used.add(record["trace_id"])
            if (
                call[1]["messages"] != observer.messages(prepared)
                or call[0]["prompt_version"] != PROMPT_VERSION
            ):
                raise ValueError("observer saw a different input/prompt")
        verify_record_request(record, prepared, required=call is not None)
        if call is not None and call[2] is not None and "raw_content" in record:
            if record["raw_content"] != call[2]:
                raise ValueError("saved raw catalog output differs from HTTP")
        if entry["status"] != record["status"]:
            raise ValueError("row terminal differs")
        if record["status"] == "completed":
            if call is None or call[2] != record["raw_content"]:
                raise ValueError("raw HTTP does not support completed observer")
            if (
                any(
                    call[0].get(k) is None
                    for k in ("input_tokens", "output_tokens", "estimated_actual_cny")
                )
                or call[0].get("validation_status") is not None
            ):
                raise ValueError("unknown or rejected usage cannot support completed observer")
            report, audit = observer.resolve(call[2], prepared)
            audit.verify_report(report)
            verify_catalog_replay(record, audit)
            if report.to_dict() != record["report"]:
                raise ValueError("offline observation replay differs")
            reports[row["row_id"]] = report
        else:
            failed = True
            if record.get("failure_type") in {"CatalogContractError", "ConditionContractError"}:
                if call is None or call[2] != record.get("raw_content"):
                    raise ValueError("contract failure missing complete raw response")
                try:
                    observer.resolve(call[2], prepared)
                except observer.CatalogContractError as error:
                    if error.audit is None:
                        raise ValueError(
                            "failed catalog response lacks deterministic audit"
                        ) from error
                    verify_catalog_replay(record, error.audit)
                else:
                    raise ValueError("claimed contract failure not reproduced")
    if used != set(calls_by_trace):
        raise ValueError("unattributed API calls")
    if (terminal["status"] == "completed") != (not failed and len(reports) == 32):
        raise ValueError("batch status differs from row completion")
    if terminal["status"] == "completed" and (ledger.get("block_reason") or refusal):
        raise ValueError("completed batch has a budget stop")
    gate, details = gate_summary(rows, reports)
    agreement = {kind: {"matched": 0, "observed": 0} for kind in ("T", "R", "P0", "P1")}
    for row in details:
        if row["status"] != "scored":
            continue
        expected = {(c["kind"], c["slot"]): c["status"] for c in row["expected"]["checks"]}
        for check in row["observed"]["checks"]:
            kind = check["kind"]
            agreement[kind]["observed"] += 1
            agreement[kind]["matched"] += check["status"] == expected[kind, check["slot"]]
    target = project / FEEDBACK
    target.mkdir(exist_ok=False)
    write_json(target / "per_row.json", details)
    write_json(
        target / "audit.json",
        {"input_sha256": hashes, "http_requests": len(captured), "offline_replay": True},
    )
    summary = {
        "protocol": PROTOCOL,
        "gate": gate,
        "dimension_nominal_agreement": agreement,
        "totals": totals(ledger["calls"]),
        "api_calls": 0,
        "gold_loaded": False,
        "memory_updated": False,
        "notice": (
            "32 reused development rows/30 unique inputs; prior exposure disclosed; "
            "nominal annotation agreement, not a new blind test or natural utility"
        ),
        "freeze_sha256": freeze_sha,
        "terminal_sha256": terminal_sha,
    }
    write_json(target / "SUMMARY.json", summary)
    write_json(
        target / "feedback_frozen.json", {p.name: _sha(p) for p in sorted(target.glob("*.json"))}
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--expected-freeze-sha256")
    parser.add_argument("--expected-terminal-sha256")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze and (args.score or args.allow_network or args.expected_freeze_sha256):
        parser.error("freeze is standalone")
    if args.score and (args.allow_network or not args.expected_terminal_sha256):
        parser.error("scoring is offline and requires external terminal SHA")
    project = Path.cwd().resolve(strict=True)
    result = (
        freeze(project)
        if args.freeze
        else score(project, args.expected_freeze_sha256, args.expected_terminal_sha256)
        if args.score
        else run(project, args.expected_freeze_sha256, allow_network=args.allow_network)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
