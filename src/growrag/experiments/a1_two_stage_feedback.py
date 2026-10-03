"""Sealed, offline two-stage A1 audit; never retrieves or opens natural gold.

Batch completion means every planned row was attempted, not that every model
observation was valid. Only raw HTTP-backed, replayed observations enter the
unchanged catalog development gate. Failed and unsent requests remain separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from . import a1_two_stage_observer as observer
from . import a1_two_stage_study as study
from .a0_v2_feedback import _same_amount, audit_http, totals
from .a1_catalog_study import gate_summary
from .a1_conditions import prepare_payload
from .fresh_dev_manifest import _sha
from .pre_pilot import write_json
from .representation_runner import fingerprint

_STAGES = ("locate", "judge")
_REFUSALS = {"elapsed_time_limit", "estimated_budget_limit", "prompt_size_limit"}
_VALIDATIONS = {"unknown_usage", "model_snapshot_mismatch", "reservation_assumption_exceeded"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _raw_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _version(stage):
    return observer.LOCATOR_PROMPT_VERSION if stage == "locate" else observer.JUDGE_PROMPT_VERSION


def audit_journal(directory, ledger, captured, track, *, refused_stage=None):
    """Match journal ordinals to actual stage traces, never to row numbers."""
    journal = directory / "request_journal"
    present = {p.name for p in journal.glob("*.json")} if journal.exists() else set()
    names = {f"{i:04d}_{s}.json" for i in range(len(captured)) for s in ("intent", "after")}
    tail = {f"{len(captured):04d}_{s}.json" for s in ("intent", "after")}
    refusal = present == names | tail
    _require(present == names or refusal, "journal coverage mismatch")
    _require(refusal == (refused_stage is not None), "unsent stage/journal refusal differs")
    prior, after = 0.0, None
    for number, (call, request, _) in enumerate(captured):
        intent = json.loads(track(journal / f"{number:04d}_intent.json").read_bytes())
        after = json.loads(track(journal / f"{number:04d}_after.json").read_bytes())
        _require(
            intent.get("trace_id") == call["trace_id"]
            and intent.get("prompt_version") == call["prompt_version"]
            and intent.get("request_fingerprint") == fingerprint(request["messages"])
            and intent.get("status") == "pending_no_automatic_retry"
            and after.get("calls") == ledger["calls"][: number + 1],
            "journal request identity differs",
        )
        _same_amount(intent.get("prior_reserved_cny"), prior, "journal prior differs")
        _same_amount(intent.get("potential_reserved_cny"), call["reserved_cny"], "reserve differs")
        values = totals(after["calls"])
        for key in ("api_requests", "reserved_cny", "estimated_actual_cny"):
            _same_amount(after.get(key), values[key], "journal total differs")
        prior += call["reserved_cny"]
    if refusal:
        intent = json.loads(track(journal / f"{len(captured):04d}_intent.json").read_bytes())
        after = json.loads(track(journal / f"{len(captured):04d}_after.json").read_bytes())
        _require(
            after.get("block_reason") in _REFUSALS
            and after.get("calls") == ledger["calls"]
            and intent.get("trace_id") == refused_stage["trace_id"]
            and intent.get("prompt_version") == refused_stage["prompt_version"]
            and intent.get("request_fingerprint") == fingerprint(refused_stage["messages"])
            and intent.get("status") == "pending_no_automatic_retry",
            "unaccounted no-request refusal",
        )
        reserve = intent.get("potential_reserved_cny")
        _require(
            type(reserve) in (int, float) and math.isfinite(reserve) and reserve >= 0,
            "invalid refusal reservation",
        )
        _same_amount(intent.get("prior_reserved_cny"), prior, "refusal prior differs")
        _same_amount(after.get("reserved_cny"), prior, "unsent request charged")
        _same_amount(after.get("api_requests"), len(captured), "refusal HTTP count changed")
    if after is not None:
        _require(after == ledger, "last journal differs from final ledger")
    return refusal


def _metadata(project, freeze_sha, terminal_sha, track):
    for value in (freeze_sha, terminal_sha):
        _require(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "external freeze and terminal SHA256 required",
        )
    frozen = study.load_freeze(project, freeze_sha)
    directory = project / study.OUTPUT
    path = track(directory / "TERMINAL.json")
    _require(_sha(path) == terminal_sha, "external terminal SHA mismatch")
    terminal = json.loads(path.read_bytes())
    ledger = json.loads(track(directory / "final_budget.json").read_bytes())
    _require(
        terminal.get("budget_sha256") == _sha(directory / "final_budget.json")
        and terminal.get("freeze_sha256") == freeze_sha,
        "terminal budget/freeze identity differs",
    )
    _require(
        terminal.get("protocol") == study.PROTOCOL
        and terminal.get("status") in {"stopped", "completed"}
        and terminal.get("gold_loaded") is False
        and terminal.get("memory_updated") is False
        and (terminal["status"] == "completed") == (terminal.get("stop_reason") is None)
        and (
            terminal["status"] == "completed"
            or isinstance(terminal.get("stop_reason"), str)
            and bool(terminal["stop_reason"])
        ),
        "invalid terminal metadata",
    )
    for name, key in (
        ("launch_plan.json", "launch_sha256"),
        ("prior_budget.json", "prior_sha256"),
        ("source_snapshot.json", "source_snapshot_sha256"),
    ):
        _require(
            _sha(track(directory / name)) == terminal.get(key), "terminal metadata seal differs"
        )
    claim_path = track(project / "runs" / f"{study.RUN_ID}.claim.json")
    _require(_sha(claim_path) == terminal.get("claim_sha256"), "claim seal differs")
    plan = json.loads((directory / "launch_plan.json").read_bytes())
    _require(all(plan.get(k) == v for k, v in study.CONFIG.items()), "launch configuration drift")
    _require(
        plan.get("run_id") == study.RUN_ID
        and plan.get("freeze_sha256") == freeze_sha
        and plan.get("row_ids") == frozen["row_ids"]
        and plan.get("question_ids") == []
        and json.loads(claim_path.read_bytes()) == {**plan, "plan_sha256": fingerprint(plan)},
        "claim/launch binding differs",
    )
    prior = json.loads((directory / "prior_budget.json").read_bytes())
    _same_amount(plan.get("prior_reserved_cny"), prior["prior_reserved_cny"], "prior differs")
    _same_amount(
        plan.get("subcap_cny"),
        study.available_subcap(prior["prior_reserved_cny"]),
        "subcap differs",
    )
    snapshot = json.loads((directory / "source_snapshot.json").read_bytes())
    _require(
        snapshot.get("sha256") == frozen["source_sha256"]
        and fingerprint(snapshot["files"]) == snapshot["sha256"],
        "source snapshot differs",
    )
    values = totals(ledger["calls"])
    for key in ("api_requests", "reserved_cny", "estimated_actual_cny"):
        _same_amount(ledger.get(key), values[key], "final ledger total differs")
    for key in ("input_tokens", "output_tokens"):
        if key in ledger:
            _same_amount(ledger[key], values[key], "final ledger usage differs")
    _require(
        values["reserved_cny"] is not None
        and values["reserved_cny"] <= plan["subcap_cny"]
        and len(ledger["calls"]) <= 64,
        "series cap/count exceeded",
    )
    return frozen, directory, terminal, ledger, plan


def _stage_request(stage, expected, name, index, call):
    if stage is None:
        _require(call is None, "unattempted stage has HTTP/result")
        return
    _require(
        stage.get("trace_id") == f"{study.RUN_ID}/row/{index:02d}/{name}"
        and stage.get("prompt_version") == _version(name)
        and stage.get("status") in {"completed", "failed"},
        "stage sequence/trace/prompt/status differs",
    )
    if expected is not None:
        _require(
            stage.get("messages") == expected
            and stage.get("messages_sha256") == study.input_sha(expected),
            "stage saved request differs from prepared input/dependency",
        )
    if call is not None:
        _require(
            expected is not None
            and call[1]["messages"] == expected
            and call[0]["prompt_version"] == _version(name)
            and stage.get("status") in {"completed", "failed"},
            "observer saw a different stage input/prompt",
        )
    if "raw_content" in stage:
        _require(
            type(stage["raw_content"]) is str
            and stage.get("raw_response_sha256") == _raw_sha(stage["raw_content"])
            and call is not None
            and call[2] == stage["raw_content"],
            "stage saved raw output differs from HTTP/SHA",
        )
    else:
        _require("raw_response_sha256" not in stage, "stage raw hash lacks raw content")
    if stage["status"] == "completed":
        _require(
            call is not None
            and call[2] is not None
            and stage.get("raw_content") == call[2]
            and call[0].get("validation_status") is None
            and all(
                call[0].get(k) is not None
                for k in ("input_tokens", "output_tokens", "estimated_actual_cny")
            ),
            "unknown or rejected usage cannot support completed stage",
        )


def _replay_row(record, prepared, index, calls_by_trace, ledger):
    stages = record.get("stages")
    _require(type(stages) is dict and set(stages) <= set(_STAGES), "two-stage structure differs")
    locate, judge = stages.get("locate"), stages.get("judge")
    _require(all(type(s) is dict for s in stages.values()), "invalid stage record")
    calls = {name: calls_by_trace.get(f"{study.RUN_ID}/row/{index:02d}/{name}") for name in _STAGES}
    _stage_request(locate, observer.locator_messages(prepared), "locate", index, calls["locate"])
    location, report, audit, contract = None, None, None, None
    if locate is not None and "raw_content" in locate and locate["status"] == "completed":
        try:
            location = observer.resolve_location(locate["raw_content"], prepared)
        except observer.TwoStageContractError as error:
            contract = error
    if location is not None:
        _require(record.get("location") == location.to_dict(), "offline location replay differs")
    else:
        _require("location" not in record, "failed/unattempted locator fabricated location")
    expected = observer.judge_messages(prepared, location) if location is not None else None
    _stage_request(judge, expected, "judge", index, calls["judge"])
    if location is None:
        _require(judge is None, "judge executed before a valid locator")
    elif judge is not None and "raw_content" in judge and judge["status"] == "completed":
        try:
            report, audit = observer.resolve_judgment(judge["raw_content"], prepared, location)
            audit.verify_report(report)
        except observer.TwoStageContractError as error:
            contract = error
    used = [calls[name][0]["trace_id"] for name in _STAGES if calls[name] is not None]
    refused = None
    if record["status"] == "completed":
        _require(
            contract is None
            and report is not None
            and set(stages) == set(_STAGES)
            and all(stages[name]["status"] == "completed" for name in _STAGES)
            and record.get("report") == report.to_dict()
            and record.get("two_stage_audit") == audit.to_dict()
            and record.get("observation_status") == "valid"
            and record.get("fatal") is False
            and not {"failure_type", "failure_code", "failure_stage"} & set(record),
            "offline observation/status replay differs",
        )
        return report, used, refused
    if "report" in record:
        _require(
            record.get("fatal") is True
            and report is not None
            and record["report"] == report.to_dict(),
            "failed observation fabricated a business report",
        )
    if contract is not None:
        _require(
            record.get("failure_type") == "TwoStageContractError"
            and record.get("failure_code") == contract.code
            and record.get("observation_status") == contract.observation_status
            and (
                record.get("fatal") is False
                or record.get("fatal") is True
                and isinstance(record.get("integrity_failure_type"), str)
            )
            and stages[contract.stage]["status"] == "completed"
            and record.get("two_stage_audit")
            == (contract.audit.to_dict() if contract.audit is not None else None),
            "local contract failure classification/audit differs",
        )
        _require(record.get("failure_stage") == contract.stage, "failure stage differs")
    else:
        _require(
            isinstance(record.get("failure_type"), str)
            and record["failure_type"] != "TwoStageContractError"
            and record.get("fatal") is True
            and record.get("observation_status") == "integrity_error"
            and record.get("failure_stage") in {*_STAGES, "guard"}
            and isinstance(record.get("failure_code"), str),
            "unreproduced or nonfatal execution failure",
        )
        _require(
            "two_stage_audit" not in record
            or audit is not None
            and record["two_stage_audit"] == audit.to_dict(),
            "fatal failure audit differs",
        )
        failed_stages = [
            name for name in _STAGES if name in stages and stages[name]["status"] == "failed"
        ]
        _require(len(failed_stages) <= 1, "multiple failed stages")
        for name in failed_stages:
            _require(record["failure_stage"] == name, "fatal stage differs")
            _require(
                stages[name].get("failure_type") == record["failure_type"],
                "fatal exception type differs from stage",
            )
            call = calls[name]
            if call is None:
                if ledger.get("block_reason") in _REFUSALS:
                    _require("messages" in stages[name], "refused stage lacks actual request")
                    _require(
                        record["failure_code"] == ledger["block_reason"],
                        "refusal failure classification differs",
                    )
                    refused = stages[name]
                else:
                    _require(
                        record["failure_code"] in {"integrity_failure", "execution_failure"},
                        "unaccounted non-HTTP stage failure",
                    )
            else:
                expected_code = (
                    (call[0].get("validation_status") or "transport_failure")
                    if call[2] is None
                    else "execution_failure"
                )
                _require(
                    record["failure_code"] == expected_code, "fatal HTTP classification differs"
                )
        if not failed_stages:
            _require(
                record["failure_code"] in {"integrity_failure", "execution_failure"},
                "unreproduced non-stage failure",
            )
    return None, used, refused


def score(project, freeze_sha, terminal_sha):
    """Rebuild both stages from original HTTP before scoring fixture annotations."""
    project = Path(project).resolve(strict=True)
    hashes = {}

    def track(path):
        path = Path(path).resolve(strict=True)
        _require(path.is_relative_to(project), "audit artifact escapes project")
        hashes[path.relative_to(project).as_posix()] = _sha(path)
        return path

    frozen, directory, terminal, ledger, plan = _metadata(project, freeze_sha, terminal_sha, track)
    seen = set()
    captured = [
        audit_http(c, directory, plan, track, seen, study.RUN_ID + "/") for c in ledger["calls"]
    ]
    _require(
        {p.name for p in (directory / "api_audit").glob("*.json")}
        == {hashlib.sha256(c[0]["trace_id"].encode()).hexdigest() + ".json" for c in captured},
        "unattributed raw HTTP artifact",
    )
    calls_by_trace = {c[0]["trace_id"]: c for c in captured}
    rows, reports, failures, used, refused = study.load_fixture(project), {}, {}, [], None
    _require(
        [r["row_id"] for r in terminal["rows"]] == [r["row_id"] for r in rows] == frozen["row_ids"],
        "terminal row sequence changed",
    )
    stopped, attempted = False, 0
    for index, (row, entry) in enumerate(zip(rows, terminal["rows"], strict=True)):
        _require(
            entry.get("status") in {"completed", "failed", "not_attempted"}, "unknown row status"
        )
        if entry["status"] == "not_attempted":
            _require(set(entry) == {"row_id", "status"}, "unattempted row has fabricated result")
            stopped = True
            continue
        _require(not stopped, "rows executed after stop")
        _require(
            entry.get("path") == f"row_{index:02d}.json", "row path differs from planned sequence"
        )
        path = track(directory / entry["path"])
        _require(_sha(path) == entry.get("sha256"), "row seal differs")
        record = json.loads(path.read_bytes())
        _require(
            record.get("row_id") == row["row_id"]
            and record.get("sequence") == index
            and record.get("input_sha256") == study.input_sha(row["input"])
            and record.get("status") == entry["status"],
            "row/input/terminal binding differs",
        )
        report, row_used, row_refused = _replay_row(
            record, prepare_payload(row["input"]), index, calls_by_trace, ledger
        )
        used.extend(row_used)
        if row_refused is not None:
            _require(refused is None, "multiple unsent refusals")
            refused = row_refused
        if report is not None:
            reports[row["row_id"]] = report
        else:
            failures[row["row_id"]] = {
                key: record[key]
                for key in ("failure_type", "failure_code", "observation_status", "fatal")
            }
        attempted += 1
        if record.get("fatal") is True:
            stopped = True
    _require(
        used == [c[0]["trace_id"] for c in captured], "unattributed or out-of-order API stages"
    )
    _require(
        (terminal["status"] == "completed") == (attempted == 32 and not stopped),
        "batch status differs from planned attempts/fatal stop",
    )
    refusal = audit_journal(directory, ledger, captured, track, refused_stage=refused)
    if terminal["status"] == "completed":
        _require(
            not ledger.get("block_reason") and not refusal, "completed batch has a budget stop"
        )
    if any(
        c[0].get("validation_status") in _VALIDATIONS or c[0]["status"] == "failed"
        for c in captured
    ):
        _require(
            terminal["status"] == "stopped", "unsafe transport/usage failure did not stop batch"
        )
    gate, details = gate_summary(rows, reports)
    agreement = {kind: {"matched": 0, "observed": 0} for kind in ("T", "R", "P0", "P1")}
    for detail in details:
        if detail["status"] != "scored":
            if detail["row_id"] in failures:
                detail["observation_failure"] = failures[detail["row_id"]]
            continue
        expected = {(c["kind"], c["slot"]): c["status"] for c in detail["expected"]["checks"]}
        for check in detail["observed"]["checks"]:
            agreement[check["kind"]]["observed"] += 1
            agreement[check["kind"]]["matched"] += (
                check["status"] == expected[check["kind"], check["slot"]]
            )
    target = project / study.FEEDBACK
    target.mkdir(exist_ok=False)
    write_json(target / "per_row.json", details)
    write_json(
        target / "audit.json",
        {
            "input_sha256": hashes,
            "http_requests": len(captured),
            "offline_replay": True,
            "stages": used,
        },
    )
    summary = {
        "protocol": study.PROTOCOL,
        "gate": gate,
        "batch_status": terminal["status"],
        "attempted_rows": attempted,
        "observation_failures": failures,
        "dimension_nominal_agreement": agreement,
        "totals": totals(ledger["calls"]),
        "api_calls": 0,
        "gold_loaded": False,
        "memory_updated": False,
        "notice": (
            "32 reused development rows/30 unique inputs; prior exposure disclosed; "
            "nominal annotation agreement, not a new blind test or natural utility; "
            "two-stage budget differs from one-stage"
        ),
        "freeze_sha256": freeze_sha,
        "terminal_sha256": terminal_sha,
    }
    write_json(target / "SUMMARY.json", summary)
    write_json(
        target / "feedback_frozen.json", {p.name: _sha(p) for p in sorted(target.glob("*.json"))}
    )
    return summary
