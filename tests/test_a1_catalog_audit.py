"""Adversarial offline A1 audit fixtures; no network or real experiment labels."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from growrag.experiments import a1_catalog_study as study
from growrag.experiments.budget import PriceLimits

PROJECT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "a" * 64


def catalog_archive(tmp_path, monkeypatch, *, tamper=None, stop=None, real_parser=False):
    rows = study.load_fixture(PROJECT)
    source_hash = study.fingerprint({})
    frozen = {"row_ids": [r["row_id"] for r in rows], "source_sha256": source_hash}
    monkeypatch.setattr(study, "load_freeze", lambda *args: frozen)
    monkeypatch.setattr(study, "load_fixture", lambda *args: deepcopy(rows))
    directory = tmp_path / study.OUTPUT
    directory.mkdir(parents=True)
    plan = {
        **study.CONFIG,
        "run_id": study.RUN_ID,
        "freeze_sha256": FREEZE_SHA,
        "prior_reserved_cny": 0,
        "subcap_cny": 1,
        "row_ids": frozen["row_ids"],
        "question_ids": [],
    }
    study.write_json(directory / "launch_plan.json", plan)
    study.write_json(directory / "prior_budget.json", {"prior_reserved_cny": 0})
    study.write_json(directory / "source_snapshot.json", {"files": {}, "sha256": source_hash})
    claim = tmp_path / "runs" / f"{study.RUN_ID}.claim.json"
    study.write_json(claim, {**plan, "plan_sha256": study.fingerprint(plan)})
    records, calls, captures = [], [], []
    unknown = stop == "unknown_usage" or tamper == "unknown_usage_completed"
    for index, row in enumerate(rows):
        if stop and index > 0:
            records.append(None)
            continue
        trace = f"{study.RUN_ID}/row/{index:02d}"
        prepared = study.prepare_payload(row["input"])
        messages = study.observer.messages(prepared)
        # Unknown-only mock output is independent of fixture expected annotations.
        reasons = {
            "T": "insufficient_type_evidence",
            "R": "ambiguous_subgoal",
            "P1": "missing_role_evidence",
        }
        content = json.dumps(
            {
                "checks": [
                    {
                        "kind": kind,
                        "slot": slot,
                        "status": "unknown",
                        "reason": reasons[kind],
                        "ref_ids": [],
                    }
                    for kind, slot in prepared.semantic_checks
                ]
            }
        )
        if stop == "contract":
            content = "invalid contract fixture"
        record = {
            "row_id": row["row_id"],
            "sequence": index,
            "trace_id": trace,
            "input_sha256": study.input_sha(row["input"]),
            "messages": messages,
            "messages_sha256": study.input_sha(messages),
            "catalog": study.observer.catalog(prepared),
            "catalog_sha256": study.input_sha(study.observer.catalog(prepared)),
            "raw_content": content,
            "raw_response_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "status": "completed",
        }
        if not stop or stop in {"contract", "false_contract"}:
            try:
                report, audit = study.observer.resolve(content, prepared)
                record["report"] = report.to_dict()
            except study.observer.CatalogContractError as error:
                audit = error.audit
            study.attach_catalog_audit(record, audit)
        if stop:
            record.update(
                status="failed",
                failure_type="CatalogContractError"
                if stop in {"contract", "false_contract"}
                else "APIRequestError",
            )
            record.pop("report", None)
        if stop == "refusal":
            record.pop("raw_content")
            records.append(record)
            continue
        call = {
            "trace_id": trace,
            "prompt_version": study.PROMPT_VERSION,
            "status": "completed",
            "api_requests": 1,
            "input_tokens": None if unknown else 10,
            "output_tokens": 5,
            "reserved_cny": 0.01,
            "estimated_actual_cny": None if unknown else 0.000006,
            "returned_model": plan["model"],
            "audit_path": hashlib.sha256(trace.encode()).hexdigest() + ".json",
        }
        if stop == "unknown_usage":
            call["validation_status"] = "unknown_usage"
            record.pop("raw_content")
        request = {
            "model": plan["model"],
            "max_tokens": plan["max_output_tokens"],
            "stream": False,
            "enable_thinking": False,
            "temperature": 0,
            "top_p": 1,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        raw = {
            "trace_id": trace,
            "prompt_version": study.PROMPT_VERSION,
            "transport_source": "live_api",
            "network_attempted": True,
            "api_requests": 1,
            "retry_count": 0,
            "status": "completed",
            "http_status": 200,
            "finish_reason": "stop",
            "input_tokens": call["input_tokens"],
            "output_tokens": call["output_tokens"],
            "response_redacted": False,
            "request": request,
            "request_sha256": hashlib.sha256(
                json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
            ).hexdigest(),
            "response": {
                "model": plan["model"],
                "usage": {"prompt_tokens": call["input_tokens"], "completion_tokens": 5},
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            },
        }
        records.append(record)
        calls.append(call)
        captures.append((call, request, raw))
    if tamper == "duplicate_trace":
        seen = {}
        for index, row in enumerate(rows):
            digest = study.fingerprint(row["input"])
            if digest in seen:
                records[index]["trace_id"] = records[seen[digest]]["trace_id"]
                del calls[index]
                del captures[index]
                break
            seen[digest] = index
        else:
            pytest.fail("expected one duplicate synthetic input pair")
    if tamper == "sequence":
        records[0]["sequence"] = 9
    if tamper == "report":
        records[0]["report"]["decision"] = "reject"
    if tamper == "http_content":
        captures[0][2]["response"]["choices"][0]["message"]["content"] = "changed HTTP output"
    if tamper == "http_request_hash":
        captures[0][2]["request_sha256"] = "b" * 64
    if tamper == "not_live":
        captures[0][2]["transport_source"] = "mock"
    if tamper == "catalog":
        records[0]["catalog"][0]["quote"] = "forged question"
        records[0]["catalog_sha256"] = study.input_sha(records[0]["catalog"])
    if tamper == "catalog_audit":
        records[0]["catalog_audit"]["catalog"][0]["quote"] = "forged question"
    if tamper == "mapped_v2":
        records[0]["mapped_v2_raw_text"] = '{"checks":[]}'
        records[0]["mapped_v2_sha256"] = hashlib.sha256(
            records[0]["mapped_v2_raw_text"].encode()
        ).hexdigest()
    if tamper == "messages":
        records[0]["messages"][0]["content"] = "forged system prompt"
        records[0]["messages_sha256"] = study.input_sha(records[0]["messages"])
    if tamper == "raw_sha":
        records[0]["raw_response_sha256"] = "b" * 64
    if tamper == "not_attempted":
        records[-1] = None
        del calls[-1]
        del captures[-1]
    api_audit, journal = directory / "api_audit", directory / "request_journal"
    api_audit.mkdir()
    journal.mkdir()
    ledger = None
    for number, (call, request, raw) in enumerate(captures):
        if tamper != "missing_http" or number != 0:
            study.write_json(api_audit / call["audit_path"], raw)
        values = study.totals(calls[: number + 1])
        ledger = {
            "calls": calls[: number + 1],
            "limits": asdict(PriceLimits(budget_cny=1)),
            "block_reason": "unknown_usage" if stop == "unknown_usage" else None,
            **{k: values[k] for k in ("api_requests", "reserved_cny", "estimated_actual_cny")},
        }
        study.write_json(
            journal / f"{number:04d}_intent.json",
            {
                "trace_id": call["trace_id"],
                "prompt_version": study.PROMPT_VERSION,
                "request_fingerprint": study.fingerprint(request["messages"]),
                "status": "pending_no_automatic_retry",
                "potential_reserved_cny": call["reserved_cny"],
                "prior_reserved_cny": sum(c["reserved_cny"] for c in calls[:number]),
            },
        )
        study.write_json(journal / f"{number:04d}_after.json", ledger)
    if stop == "refusal":
        ledger = {
            "calls": [],
            "api_requests": 0,
            "reserved_cny": 0,
            "estimated_actual_cny": 0,
            "block_reason": "estimated_budget_limit",
        }
        study.write_json(
            journal / "0000_intent.json",
            {
                "trace_id": records[0]["trace_id"],
                "prompt_version": study.PROMPT_VERSION,
                "request_fingerprint": study.fingerprint(records[0]["messages"]),
                "status": "pending_no_automatic_retry",
                "potential_reserved_cny": 0.01,
                "prior_reserved_cny": 0,
            },
        )
        study.write_json(journal / "0000_after.json", ledger)
    if tamper == "zero_unknown_cost":
        ledger["estimated_actual_cny"] = 0
    study.write_json(directory / "final_budget.json", ledger)
    entries = []
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        entry = {"row_id": row["row_id"], "status": "not_attempted"}
        if record is not None:
            path = directory / f"row_{index:02d}.json"
            study.write_json(path, record)
            entry.update(status=record["status"], path=path.name, sha256=study._sha(path))
        entries.append(entry)
    if tamper == "row_path":
        entries[0]["path"] = "../row_00.json"
    if tamper == "row_status":
        entries[0]["status"] = "success"
    terminal = {
        "protocol": study.PROTOCOL,
        "status": "stopped" if stop else "completed",
        "stop_reason": "observer_contract_failure"
        if stop in {"contract", "false_contract"}
        else "APIRequestError"
        if stop
        else None,
        "rows": entries,
        "freeze_sha256": FREEZE_SHA,
        "budget_sha256": study._sha(directory / "final_budget.json"),
        "launch_sha256": study._sha(directory / "launch_plan.json"),
        "prior_sha256": study._sha(directory / "prior_budget.json"),
        "source_snapshot_sha256": study._sha(directory / "source_snapshot.json"),
        "claim_sha256": study._sha(claim),
        "gold_loaded": False,
        "memory_updated": False,
    }
    if tamper == "terminal_status":
        terminal.update(status="stopped", stop_reason="invented stop")
    if tamper == "terminal_protocol":
        terminal["protocol"] = "other"
    if tamper == "terminal_gold":
        terminal["gold_loaded"] = True
    if tamper == "claim_seal":
        terminal["claim_sha256"] = "b" * 64
    path = directory / "TERMINAL.json"
    study.write_json(path, terminal)
    return study._sha(path)


def test_full_sealed_catalog_http_replay_keeps_unknown_not_nominal_pass(tmp_path, monkeypatch):
    terminal_sha = catalog_archive(tmp_path, monkeypatch)
    result = study.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert result["gate"]["completed"] == 32
    assert result["gate"]["passed"] is False
    assert result["totals"]["api_requests"] == 32
    assert result["api_calls"] == 0 and result["gold_loaded"] is False
    with pytest.raises(FileExistsError):
        study.score(tmp_path, FREEZE_SHA, terminal_sha)


def test_actual_catalog_contract_replays_raw_v3_through_v2_gate(tmp_path, monkeypatch):
    terminal_sha = catalog_archive(tmp_path, monkeypatch, real_parser=True)
    result = study.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert result["gate"]["completed"] == 32
    assert result["gate"]["passed"] is False  # Deliberately unknown, not model quality evidence.
    assert result["totals"]["api_requests"] == 32
    details = json.loads((tmp_path / study.FEEDBACK / "per_row.json").read_bytes())
    rows = study.load_fixture(PROJECT)
    for row, detail in zip(rows, details, strict=True):
        report = detail["observed"]
        assert report["input_sha256"] == study.input_sha(row["input"])
        assert set(json.loads(report["raw_text"])) == {"checks"}
        assert "refs" in json.loads(report["raw_text"])["checks"][0]
        assert all(c["status"] == "unknown" for c in report["checks"] if c["producer"] == "model")


@pytest.mark.parametrize(
    "tamper",
    [
        "duplicate_trace",
        "sequence",
        "report",
        "http_content",
        "http_request_hash",
        "not_live",
        "missing_http",
        "not_attempted",
        "row_path",
        "row_status",
        "terminal_status",
        "terminal_protocol",
        "terminal_gold",
        "claim_seal",
        "unknown_usage_completed",
        "catalog",
        "catalog_audit",
        "mapped_v2",
        "messages",
        "raw_sha",
    ],
)
def test_resealed_forged_completion_is_rejected(tmp_path, monkeypatch, tamper):
    terminal_sha = catalog_archive(tmp_path, monkeypatch, tamper=tamper)
    with pytest.raises((ValueError, FileNotFoundError)):
        study.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert not (tmp_path / study.FEEDBACK).exists()


@pytest.mark.parametrize("stop", ["contract", "unknown_usage", "refusal"])
def test_failed_call_or_unsent_refusal_stays_missing_not_a_rerun(tmp_path, monkeypatch, stop):
    terminal_sha = catalog_archive(tmp_path, monkeypatch, stop=stop)
    result = study.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert result["gate"]["completed"] == 0
    assert result["gate"]["passed"] is False
    assert result["totals"]["api_requests"] == (0 if stop == "refusal" else 1)
    assert result["totals"]["reserved_cny"] == (0 if stop == "refusal" else 0.01)
    if stop == "unknown_usage":
        assert result["totals"]["estimated_actual_cny"] is None
        assert result["totals"]["unknown_cost_requests"] == 1


@pytest.mark.parametrize("tamper", ["zero_unknown_cost", "false_contract"])
def test_failed_ledger_cannot_zero_unknown_cost_or_invent_parser_failure(
    tmp_path, monkeypatch, tamper
):
    terminal_sha = catalog_archive(
        tmp_path,
        monkeypatch,
        stop="unknown_usage" if tamper == "zero_unknown_cost" else tamper,
        tamper=tamper,
    )
    with pytest.raises(ValueError):
        study.score(tmp_path, FREEZE_SHA, terminal_sha)


def test_terminal_external_seal_is_required_before_feedback(tmp_path, monkeypatch):
    catalog_archive(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="terminal SHA"):
        study.score(tmp_path, FREEZE_SHA, "b" * 64)
