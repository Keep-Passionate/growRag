"""Adversarial offline A1 audit fixtures; no network or real experiment labels."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import a1_budget as budget
from growrag.experiments import a1_synthetic_study as study
from growrag.experiments import prior_budget
from growrag.experiments.budget import PriceLimits

PROJECT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "a" * 64


class Report:
    def __init__(self, value, checks=()):
        self.value = value
        self.checks = list(checks)

    def to_dict(self):
        return {"decision": self.value, "checks": self.checks}


def resolve_fixture(content, prepared):
    """Isolate audit attribution from the independently tested observer parser."""
    if content == "invalid contract fixture":
        raise study.ConditionContractError("deliberate offline parser rejection")
    value = json.loads(content)
    return Report(value["decision"], value["checks"])


def synthetic_archive(tmp_path, monkeypatch, *, tamper=None, stop=None, real_parser=False):
    rows = study.load_fixture(PROJECT)
    if not real_parser:
        monkeypatch.setattr(study, "resolve_response", resolve_fixture)
        monkeypatch.setattr(study, "decide", lambda report, **kwargs: report)
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
        messages = study.observer_messages(prepared)
        # Fabricated offline transport responses exercise scoring only, not model quality.
        nominal_report = {
            "decision": row["expected"]["contradiction_only"],
            "checks": deepcopy(row["expected"]["checks"]),
        }
        content = json.dumps(nominal_report)
        if real_parser:
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
                            "refs": [],
                        }
                        for kind, slot in prepared.semantic_checks
                    ]
                }
            )
            nominal_report = study.resolve_response(content, prepared).to_dict()
        if stop == "contract":
            content = "invalid contract fixture"
        record = {
            "row_id": row["row_id"],
            "sequence": index,
            "trace_id": trace,
            "input_sha256": study.input_sha(row["input"]),
            "messages": messages,
            "raw_content": content,
            "status": "completed",
            "report": nominal_report,
        }
        if stop:
            record.update(
                status="failed",
                failure_type="ConditionContractError"
                if stop in {"contract", "false_contract"}
                else "APIRequestError",
            )
            record.pop("report")
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


def test_full_sealed_http_replay_can_pass_nominal_gate(tmp_path, monkeypatch):
    terminal_sha = synthetic_archive(tmp_path, monkeypatch)
    result = study.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert result["gate"]["completed"] == 32
    assert result["gate"]["passed"] is True
    assert result["totals"]["api_requests"] == 32
    assert result["api_calls"] == 0 and result["gold_loaded"] is False
    with pytest.raises(FileExistsError):
        study.score(tmp_path, FREEZE_SHA, terminal_sha)


def test_actual_observer_contract_replays_raw_quote_wire_without_mock_parser(tmp_path, monkeypatch):
    terminal_sha = synthetic_archive(tmp_path, monkeypatch, real_parser=True)
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
    ],
)
def test_resealed_forged_completion_is_rejected(tmp_path, monkeypatch, tamper):
    terminal_sha = synthetic_archive(tmp_path, monkeypatch, tamper=tamper)
    with pytest.raises((ValueError, FileNotFoundError)):
        study.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert not (tmp_path / study.FEEDBACK).exists()


@pytest.mark.parametrize("stop", ["contract", "unknown_usage", "refusal"])
def test_failed_call_or_unsent_refusal_stays_missing_not_a_rerun(tmp_path, monkeypatch, stop):
    terminal_sha = synthetic_archive(tmp_path, monkeypatch, stop=stop)
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
    terminal_sha = synthetic_archive(
        tmp_path,
        monkeypatch,
        stop="unknown_usage" if tamper == "zero_unknown_cost" else tamper,
        tamper=tamper,
    )
    with pytest.raises(ValueError):
        study.score(tmp_path, FREEZE_SHA, terminal_sha)


def test_terminal_external_seal_is_required_before_feedback(tmp_path, monkeypatch):
    synthetic_archive(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="terminal SHA"):
        study.score(tmp_path, FREEZE_SHA, "b" * 64)


def test_completed_prefix_survives_later_artifact_write_failure(tmp_path, monkeypatch):
    rows = study.load_fixture(PROJECT)[:3]
    monkeypatch.setattr(study, "resolve_response", lambda *args: Report("allow"))
    original_write = study.write_json

    def fail_second(path, data):
        if path.name == "row_01.json":
            raise OSError("offline disk failure fixture")
        return original_write(path, data)

    monkeypatch.setattr(study, "write_json", fail_second)
    terminal = []
    calls = []

    def complete(messages, **kwargs):
        calls.append(kwargs["trace_id"])
        return SimpleNamespace(content="mock observer")

    with pytest.raises(OSError):
        study.observe_rows(
            rows, SimpleNamespace(complete=complete), tmp_path, lambda: None, terminal=terminal
        )
    assert terminal[0]["status"] == "completed"
    assert terminal[0]["sha256"] == study._sha(tmp_path / "row_00.json")
    assert len(calls) == len(terminal) == 2


def test_interrupted_transport_remains_unknown_reserved_and_cannot_retry(tmp_path, monkeypatch):
    rows = study.load_fixture(PROJECT)[:2]
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(study, "load_fixture", lambda *args: rows)
    monkeypatch.setattr(
        study, "load_freeze", lambda *args: {"row_ids": [r["row_id"] for r in rows]}
    )
    monkeypatch.setattr(
        study, "_git_state", lambda: {"commit": "synthetic", "worktree_dirty": False}
    )
    monkeypatch.setattr(study, "reviewed_history", lambda *args: {"prior_reserved_cny": 0})
    monkeypatch.setattr(
        study, "source_snapshot", lambda *args: {"files": {}, "sha256": study.fingerprint({})}
    )
    monkeypatch.setattr(
        study,
        "read_local_bailian_settings",
        lambda *args: SimpleNamespace(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="offline-placeholder-not-a-credential",
        ),
    )
    calls = []

    class InterruptedOfflineTransport:
        transport_source = "live_api"

        def __init__(self, config, directory, **kwargs):
            self.config, self.attempts = config, 0

        def complete(self, messages, **kwargs):
            calls.append(kwargs["trace_id"])
            self.attempts += 1
            raise KeyboardInterrupt("offline request interruption fixture")

    monkeypatch.setattr(study, "LiveChatClient", InterruptedOfflineTransport)
    monkeypatch.setenv(study.KEY_VARIABLE, "previous-offline-sentinel")
    result = study.run(tmp_path, FREEZE_SHA, allow_network=True)
    assert result["status"] == "stopped" and len(calls) == 1
    output = tmp_path / study.OUTPUT
    ledger = json.loads((output / "final_budget.json").read_bytes())
    assert ledger["api_requests"] == 1 and ledger["calls"] == []
    assert ledger["reserved_cny"] > 0
    assert ledger["estimated_actual_cny"] is None
    assert ledger["input_tokens"] is ledger["output_tokens"] is None
    assert ledger["reconciliation_required"] is True
    assert ledger["block_reason"] == "unaccounted_interrupted_attempt"
    assert study.os.environ[study.KEY_VARIABLE] == "previous-offline-sentinel"
    terminal = json.loads((output / "TERMINAL.json").read_bytes())
    assert [r["status"] for r in terminal["rows"]] == ["failed", "not_attempted"]
    with pytest.raises(FileExistsError, match="already claimed"):
        study.run(tmp_path, FREEZE_SHA, allow_network=True)
    assert len(calls) == 1


def test_reconciled_unknown_history_keeps_its_full_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(prior_budget, "ROOT_LEDGERS", ())
    monkeypatch.setattr(budget.old, "HISTORICAL_ROOTS", ())
    root = tmp_path / (budget.PREFIX + "offline_history_fixture")
    root.mkdir()
    study.write_json(
        root / "launch_plan.json", {"protocol": budget.PROTOCOL, "model": budget.old.PILOT_MODEL}
    )
    call = {
        "trace_id": "synthetic-unknown-call",
        "audit_path": "synthetic-unknown.json",
        "status": "failed",
        "api_requests": 1,
        "input_tokens": None,
        "output_tokens": None,
        "estimated_actual_cny": None,
        "reserved_cny": 0.4,
    }
    ledger = {
        "api_requests": 1,
        "calls": [call],
        "reserved_cny": 0.4,
        "estimated_actual_cny": None,
        "limits": asdict(PriceLimits()),
    }
    study.write_json(root / "final_budget.json", ledger)
    (root / "api_audit").mkdir()
    study.write_json(
        root / "api_audit" / call["audit_path"], {**call, "transport_source": "live_api"}
    )
    value = budget.reviewed_history(tmp_path)
    assert value["prior_reserved_cny"] == 0.4
    assert value["prior_total_actual_cny"] is None
    assert value["prior_unknown_cost_requests"] == 1
    assert value["prior_known_estimated_cny"] == 0


def test_unknown_or_unrecorded_root_usage_fails_closed(tmp_path):
    root = tmp_path / (budget.PREFIX + "offline_interrupted_fixture")
    root.mkdir()
    study.write_json(
        root / "launch_plan.json", {"protocol": budget.PROTOCOL, "model": budget.old.PILOT_MODEL}
    )
    study.write_json(
        root / "final_budget.json", {"api_requests": 0, "calls": [], "reserved_cny": 0.2}
    )
    with pytest.raises(ValueError, match="zero-request reservations"):
        budget.reviewed_history(tmp_path)
