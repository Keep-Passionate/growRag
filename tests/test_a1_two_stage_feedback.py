"""Synthetic sealed HTTP archives; none of these fixtures are API results."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import a1_two_stage_feedback as feedback
from growrag.experiments import a1_two_stage_study as study
from growrag.experiments.api_client import APIRequestError
from growrag.experiments.budget import PriceLimits

PROJECT = Path(__file__).resolve().parents[1]
FREEZE_SHA = "a" * 64


def save_json(path, value):
    """Reseal only generated temporary test artifacts, never real run files."""
    if path.exists():
        path.unlink()
    study.write_json(path, value)


def locator_wire(prepared):
    return json.dumps(
        {
            "locations": [
                {"kind": kind, "slot": slot, "candidate_ref_ids": []}
                for kind, slot in prepared.semantic_checks
            ]
        }
    )


def unknown_wire(prepared):
    reasons = {
        "T": "insufficient_type_evidence",
        "R": "ambiguous_subgoal",
        "P1": "missing_role_evidence",
    }
    return json.dumps(
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


class SealedSyntheticClient:
    """Offline test double with the same raw-HTTP/journal field contracts."""

    def __init__(self, rows, directory, *, bad=None, fatal=None, fatal_stage=(0, "judge")):
        self.rows, self.directory = rows, directory
        self.bad, self.fatal, self.fatal_stage = bad or {}, fatal, fatal_stage
        self.calls, self.requests, self.block_reason = [], {}, None
        (directory / "api_audit").mkdir()
        (directory / "request_journal").mkdir()

    def report(self):
        values = feedback.totals(self.calls)
        return {
            "calls": deepcopy(self.calls),
            "limits": asdict(PriceLimits(budget_cny=1)),
            "block_reason": self.block_reason,
            **{
                k: values[k]
                for k in (
                    "api_requests",
                    "reserved_cny",
                    "estimated_actual_cny",
                    "input_tokens",
                    "output_tokens",
                )
            },
        }

    def intent(self, number, trace, version, messages):
        save_json(
            self.directory / "request_journal" / f"{number:04d}_intent.json",
            {
                "trace_id": trace,
                "prompt_version": version,
                "request_fingerprint": study.fingerprint(messages),
                "status": "pending_no_automatic_retry",
                "potential_reserved_cny": 0.005,
                "prior_reserved_cny": sum(c["reserved_cny"] for c in self.calls[:number]),
            },
        )

    def complete(self, messages, *, trace_id, prompt_version):
        number = len(self.calls)
        index, stage = trace_id.split("/")[-2:]
        key = int(index), stage
        prepared = study.prepare_payload(self.rows[key[0]]["input"])
        content = self.bad.get(
            key, locator_wire(prepared) if stage == "locate" else unknown_wire(prepared)
        )
        self.intent(number, trace_id, prompt_version, messages)
        if self.fatal == "refusal" and key == self.fatal_stage:
            self.block_reason = "estimated_budget_limit"
            save_json(
                self.directory / "request_journal" / f"{number:04d}_after.json", self.report()
            )
            raise APIRequestError("offline unsent refusal")
        unknown = self.fatal in {"unknown_usage", "transport"} and key == self.fatal_stage
        transport = self.fatal == "transport" and key == self.fatal_stage
        name = hashlib.sha256(trace_id.encode()).hexdigest() + ".json"
        call = {
            "trace_id": trace_id,
            "prompt_version": prompt_version,
            "status": "failed" if transport else "completed",
            "api_requests": 1,
            "input_tokens": None if unknown else 10,
            "output_tokens": None if transport else 5,
            "reserved_cny": 0.005,
            "estimated_actual_cny": None if unknown else 0.000006,
            "returned_model": study.CONFIG["model"],
            "audit_path": name,
        }
        if unknown:
            self.block_reason = "transport_failure" if transport else "unknown_usage"
            if not transport:
                call["validation_status"] = "unknown_usage"
        request = {
            "model": study.CONFIG["model"],
            "max_tokens": 2048,
            "stream": False,
            "enable_thinking": False,
            "temperature": 0,
            "top_p": 1,
            "response_format": {"type": "json_object"},
            "messages": deepcopy(messages),
        }
        raw = {
            "trace_id": trace_id,
            "prompt_version": prompt_version,
            "transport_source": "live_api",
            "network_attempted": True,
            "api_requests": 1,
            "retry_count": 0,
            "status": call["status"],
            "http_status": 503 if transport else 200,
            "finish_reason": None if transport else "stop",
            "input_tokens": call["input_tokens"],
            "output_tokens": call["output_tokens"],
            "response_redacted": False,
            "request": request,
            "request_sha256": hashlib.sha256(
                json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
            ).hexdigest(),
            "response": None
            if transport
            else {
                "model": study.CONFIG["model"],
                "usage": {
                    "prompt_tokens": call["input_tokens"],
                    "completion_tokens": call["output_tokens"],
                },
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            },
        }
        if transport:
            raw["error_type"] = "APIRequestError"
        self.calls.append(call)
        self.requests[trace_id] = request
        save_json(self.directory / "api_audit" / name, raw)
        save_json(self.directory / "request_journal" / f"{number:04d}_after.json", self.report())
        if unknown:
            raise APIRequestError("offline rejected usage/transport", api_requests=1)
        return SimpleNamespace(content=content)


def archive(
    tmp_path,
    monkeypatch,
    *,
    bad=None,
    fatal=None,
    fatal_stage=(0, "judge"),
    tamper=None,
    guard_failure_at=None,
    prior_reserved=0,
):
    rows = study.load_fixture(PROJECT)
    source_hash = study.fingerprint({})
    frozen = {"row_ids": [r["row_id"] for r in rows], "source_sha256": source_hash}
    monkeypatch.setattr(study, "load_freeze", lambda *args: deepcopy(frozen))
    monkeypatch.setattr(study, "load_fixture", lambda *args: deepcopy(rows))
    monkeypatch.setattr(study, "progress", lambda *args, **kwargs: None)
    directory = tmp_path / study.OUTPUT
    directory.mkdir(parents=True)
    plan = {
        **study.CONFIG,
        "run_id": study.RUN_ID,
        "freeze_sha256": FREEZE_SHA,
        "prior_reserved_cny": prior_reserved,
        "subcap_cny": study.available_subcap(prior_reserved),
        "row_ids": frozen["row_ids"],
        "question_ids": [],
    }
    study.write_json(directory / "launch_plan.json", plan)
    study.write_json(directory / "prior_budget.json", {"prior_reserved_cny": prior_reserved})
    study.write_json(directory / "source_snapshot.json", {"files": {}, "sha256": source_hash})
    claim = tmp_path / "runs" / f"{study.RUN_ID}.claim.json"
    study.write_json(claim, {**plan, "plan_sha256": study.fingerprint(plan)})
    client = SealedSyntheticClient(rows, directory, bad=bad, fatal=fatal, fatal_stage=fatal_stage)
    guard_calls = 0

    def guard():
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == guard_failure_at:
            raise ValueError("offline identity guard failure")

    entries, stop = study.observe_rows(rows, client, directory, guard)
    if tamper is not None:
        tamper(directory, client, entries)
    # Re-sign row and terminal seals after tampering: cross-object replay must
    # detect the forgery, rather than depending only on stale hashes.
    for entry in entries:
        if "path" in entry:
            entry["sha256"] = study._sha(directory / entry["path"])
    study.write_json(directory / "final_budget.json", client.report())
    terminal = {
        "protocol": study.PROTOCOL,
        "status": "stopped" if stop else "completed",
        "stop_reason": stop,
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
    study.write_json(directory / "TERMINAL.json", terminal)
    return study._sha(directory / "TERMINAL.json")


def alter_record(directory, index, edit):
    path = directory / f"row_{index:02d}.json"
    record = json.loads(path.read_bytes())
    edit(record)
    save_json(path, record)


def test_full_two_stage_archive_replays_64_requests_without_claiming_nominal_pass(
    tmp_path, monkeypatch
):
    terminal_sha = archive(tmp_path, monkeypatch)
    result = feedback.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert result["gate"]["completed"] == 32
    assert result["gate"]["passed"] is False
    assert result["batch_status"] == "completed"
    assert result["attempted_rows"] == 32
    assert result["totals"]["api_requests"] == 64
    assert result["api_calls"] == 0 and result["gold_loaded"] is False
    sealed = json.loads((tmp_path / study.FEEDBACK / "feedback_frozen.json").read_bytes())
    assert set(sealed) == {"SUMMARY.json", "per_row.json", "audit.json"}
    with pytest.raises(FileExistsError):
        feedback.score(tmp_path, FREEZE_SHA, terminal_sha)


def test_uncapped_nonzero_history_launch_and_feedback_share_budget_contract(tmp_path, monkeypatch):
    monkeypatch.setitem(study.CONFIG, "project_cap_cny", None)
    monkeypatch.setitem(study.CONFIG, "series_cap_cny", 1.0)
    terminal_sha = archive(tmp_path, monkeypatch, prior_reserved=53.6229374)
    directory = tmp_path / study.OUTPUT
    prior_path = directory / "prior_budget.json"
    prior_bytes = prior_path.read_bytes()
    plan = json.loads((directory / "launch_plan.json").read_bytes())
    assert plan["subcap_cny"] == study.available_subcap(53.6229374) == 1.0
    assert plan["project_cap_cny"] is None
    result = feedback.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert result["batch_status"] == "completed"
    assert result["totals"]["api_requests"] == 64
    assert result["totals"]["reserved_cny"] == pytest.approx(64 * 0.005)
    assert prior_path.read_bytes() == prior_bytes
    assert json.loads(prior_bytes)["prior_reserved_cny"] == 53.6229374


def test_resealed_archive_cannot_claim_larger_batch_budget_under_uncapped_project(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(study.CONFIG, "project_cap_cny", None)
    archive(tmp_path, monkeypatch, prior_reserved=53.6229374)
    directory = tmp_path / study.OUTPUT
    plan_path = directory / "launch_plan.json"
    plan = json.loads(plan_path.read_bytes())
    plan["subcap_cny"] = 2.0
    save_json(plan_path, plan)
    claim_path = tmp_path / "runs" / f"{study.RUN_ID}.claim.json"
    save_json(claim_path, {**plan, "plan_sha256": study.fingerprint(plan)})
    terminal_path = directory / "TERMINAL.json"
    terminal = json.loads(terminal_path.read_bytes())
    terminal.update(
        launch_sha256=study._sha(plan_path),
        claim_sha256=study._sha(claim_path),
    )
    save_json(terminal_path, terminal)
    with pytest.raises(ValueError, match="subcap differs"):
        feedback.score(tmp_path, FREEZE_SHA, study._sha(terminal_path))
    assert not (tmp_path / study.FEEDBACK).exists()


@pytest.mark.parametrize("stage,count", [("locate", 63), ("judge", 64)])
def test_row_local_output_failure_allows_later_rows_but_never_gate_pass(
    tmp_path, monkeypatch, stage, count
):
    terminal_sha = archive(tmp_path, monkeypatch, bad={(0, stage): "not-json"})
    result = feedback.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert result["batch_status"] == "completed" and result["attempted_rows"] == 32
    assert result["gate"]["completed"] == 31 and result["gate"]["passed"] is False
    assert result["totals"]["api_requests"] == count
    assert result["observation_failures"]["11A"]["observation_status"] == "model_output_error"
    details = json.loads((tmp_path / study.FEEDBACK / "per_row.json").read_bytes())
    assert details[0]["status"] == "missing" and details[1]["status"] == "scored"


def test_intermediate_judgment_missing_current_bridge_is_unverified_not_unknown(
    tmp_path, monkeypatch
):
    rows = study.load_fixture(PROJECT)
    prepared = study.prepare_payload(rows[1]["input"])
    wire = json.loads(unknown_wire(prepared))
    by_source = {
        item["source_id"]: item["ref_id"] for item in study.previous.observer.catalog(prepared)
    }
    relation = next(check for check in wire["checks"] if check["kind"] == "R")
    relation.update(
        status="supported",
        reason="permitted_intermediate_subgoal",
        ref_ids=[by_source["Q"], by_source["A1"]],
    )
    terminal_sha = archive(tmp_path, monkeypatch, bad={(1, "judge"): json.dumps(wire)})
    result = feedback.score(tmp_path, FREEZE_SHA, terminal_sha)
    failure = result["observation_failures"]["07A"]
    assert failure["observation_status"] == "unverified"
    assert failure["failure_code"] == "missing_intermediate_evidence"
    assert result["gate"]["completed"] == 31 and result["batch_status"] == "completed"


@pytest.mark.parametrize("fatal,count", [("unknown_usage", 2), ("transport", 2), ("refusal", 1)])
def test_fatal_judge_usage_transport_or_unsent_refusal_seals_actual_stage(
    tmp_path, monkeypatch, fatal, count
):
    terminal_sha = archive(tmp_path, monkeypatch, fatal=fatal)
    result = feedback.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert result["batch_status"] == "stopped" and result["attempted_rows"] == 1
    assert result["gate"]["completed"] == 0
    assert result["totals"]["api_requests"] == count
    assert result["totals"]["reserved_cny"] == count * 0.005
    if fatal != "refusal":
        assert result["totals"]["estimated_actual_cny"] is None
        assert result["totals"]["unknown_cost_requests"] == 1


def test_unsent_tail_uses_real_row_and_stage_not_http_ordinal(tmp_path, monkeypatch):
    terminal_sha = archive(
        tmp_path,
        monkeypatch,
        bad={(0, "locate"): "not-json"},
        fatal="refusal",
        fatal_stage=(2, "judge"),
    )
    # Row 0 uses one HTTP, row 1 two, row 2 locate one. Journal ordinal 4
    # therefore refers to /row/02/judge, not /row/04 or /row/02/locate.
    result = feedback.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert result["totals"]["api_requests"] == 4
    assert result["gate"]["completed"] == 1
    assert result["attempted_rows"] == 3


@pytest.mark.parametrize("guard_failure_at,count", [(1, 0), (3, 1), (4, 1), (5, 2), (6, 2)])
def test_guard_stop_is_auditable_before_between_or_after_completed_http(
    tmp_path, monkeypatch, guard_failure_at, count
):
    terminal_sha = archive(tmp_path, monkeypatch, guard_failure_at=guard_failure_at)
    result = feedback.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert result["batch_status"] == "stopped" and result["attempted_rows"] == 1
    assert result["totals"]["api_requests"] == count
    assert result["gate"]["completed"] == 0
    if guard_failure_at == 3:
        record = json.loads((tmp_path / study.OUTPUT / "row_00.json").read_bytes())
        assert record["stages"]["locate"]["status"] == "failed"
        assert "raw_content" in record["stages"]["locate"]
        assert "judge" not in record["stages"]


def tampering(kind):
    def edit(directory, client, entries):
        if kind in {
            "messages",
            "raw",
            "location",
            "audit",
            "report",
            "trace",
            "classification",
            "false_contract",
            "stage_dependency",
            "judge_after_invalid_locate",
        }:

            def change(record):
                if kind == "messages":
                    record["stages"]["judge"]["messages"][0]["content"] += " forged"
                    record["stages"]["judge"]["messages_sha256"] = study.input_sha(
                        record["stages"]["judge"]["messages"]
                    )
                elif kind == "raw":
                    record["stages"]["locate"]["raw_content"] += " "
                    record["stages"]["locate"]["raw_response_sha256"] = hashlib.sha256(
                        record["stages"]["locate"]["raw_content"].encode()
                    ).hexdigest()
                elif kind == "location":
                    record["location"]["coverage"][0]["candidate_ref_count"] += 1
                    record["location"]["location_sha256"] = study.input_sha(
                        {k: v for k, v in record["location"].items() if k != "location_sha256"}
                    )
                elif kind == "audit":
                    record["two_stage_audit"]["judge"]["messages_sha256"] = "b" * 64
                    record["two_stage_audit"]["audit_sha256"] = study.input_sha(
                        {k: v for k, v in record["two_stage_audit"].items() if k != "audit_sha256"}
                    )
                elif kind == "report":
                    record["report"]["decision"] = "reject"
                elif kind == "trace":
                    record["stages"]["judge"]["trace_id"] = record["stages"]["locate"]["trace_id"]
                elif kind == "classification":
                    record["failure_code"] = "made_up_code"
                elif kind == "false_contract":
                    record.update(
                        status="failed",
                        fatal=False,
                        failure_type="TwoStageContractError",
                        failure_code="invalid_json",
                        observation_status="model_output_error",
                        failure_stage="judge",
                    )
                    record.pop("report")
                    entries[0]["status"] = "failed"
                elif kind == "stage_dependency":
                    record["stages"].pop("locate")
                    record.pop("location")
                elif kind == "judge_after_invalid_locate":
                    record["stages"]["locate"].update(
                        raw_content="not-json",
                        raw_response_sha256=hashlib.sha256(b"not-json").hexdigest(),
                    )
                    record.pop("location")
                    path = directory / "api_audit" / client.calls[0]["audit_path"]
                    raw = json.loads(path.read_bytes())
                    raw["response"]["choices"][0]["message"]["content"] = "not-json"
                    save_json(path, raw)

            alter_record(directory, 0, change)
        elif kind == "http_order":
            client.calls[0], client.calls[1] = client.calls[1], client.calls[0]
            for number, call in enumerate(client.calls):
                client.intent(
                    number,
                    call["trace_id"],
                    call["prompt_version"],
                    client.requests[call["trace_id"]]["messages"],
                )
                values = feedback.totals(client.calls[: number + 1])
                prefix = {
                    **client.report(),
                    "calls": client.calls[: number + 1],
                    **{
                        k: values[k]
                        for k in (
                            "api_requests",
                            "reserved_cny",
                            "estimated_actual_cny",
                            "input_tokens",
                            "output_tokens",
                        )
                    },
                }
                save_json(directory / "request_journal" / f"{number:04d}_after.json", prefix)
        elif kind == "not_attempted_gap":
            entries[1] = {"row_id": entries[1]["row_id"], "status": "not_attempted"}
        elif kind == "refusal_trace":
            path = directory / "request_journal" / f"{len(client.calls):04d}_intent.json"
            value = json.loads(path.read_bytes())
            value["trace_id"] = f"{study.RUN_ID}/row/{len(client.calls):02d}/locate"
            save_json(path, value)
        elif kind == "zero_unknown_cost":
            client.calls[-1]["estimated_actual_cny"] = 0
        elif kind == "unknown_completed":
            client.calls[0].update(input_tokens=None, estimated_actual_cny=None)
            path = directory / "api_audit" / client.calls[0]["audit_path"]
            value = json.loads(path.read_bytes())
            value["input_tokens"] = None
            value["response"]["usage"]["prompt_tokens"] = None
            save_json(path, value)
            for number in range(len(client.calls)):
                path = directory / "request_journal" / f"{number:04d}_after.json"
                data = json.loads(path.read_bytes())
                data["calls"] = client.calls[: number + 1]
                values = feedback.totals(data["calls"])
                data.update(
                    {
                        k: values[k]
                        for k in (
                            "api_requests",
                            "reserved_cny",
                            "estimated_actual_cny",
                            "input_tokens",
                            "output_tokens",
                        )
                    }
                )
                save_json(path, data)

    return edit


@pytest.mark.parametrize(
    "kind",
    [
        "messages",
        "raw",
        "location",
        "audit",
        "report",
        "trace",
        "false_contract",
        "stage_dependency",
        "judge_after_invalid_locate",
        "http_order",
        "not_attempted_gap",
        "unknown_completed",
    ],
)
def test_resealed_stage_and_dependency_forgery_rejected_before_feedback(
    tmp_path, monkeypatch, kind
):
    terminal_sha = archive(tmp_path, monkeypatch, tamper=tampering(kind))
    with pytest.raises(ValueError):
        feedback.score(tmp_path, FREEZE_SHA, terminal_sha)
    assert not (tmp_path / study.FEEDBACK).exists()


def test_resealed_claimed_failure_code_must_reproduce(tmp_path, monkeypatch):
    terminal_sha = archive(
        tmp_path, monkeypatch, bad={(0, "locate"): "not-json"}, tamper=tampering("classification")
    )
    with pytest.raises(ValueError, match="classification"):
        feedback.score(tmp_path, FREEZE_SHA, terminal_sha)


def test_refusal_tail_wrong_stage_rejected(tmp_path, monkeypatch):
    terminal_sha = archive(
        tmp_path, monkeypatch, fatal="refusal", tamper=tampering("refusal_trace")
    )
    with pytest.raises(ValueError, match="refusal"):
        feedback.score(tmp_path, FREEZE_SHA, terminal_sha)


def test_unknown_billing_cannot_be_zeroed_even_for_failed_observation(tmp_path, monkeypatch):
    terminal_sha = archive(
        tmp_path, monkeypatch, fatal="unknown_usage", tamper=tampering("zero_unknown_cost")
    )
    with pytest.raises(ValueError):
        feedback.score(tmp_path, FREEZE_SHA, terminal_sha)


def test_external_terminal_sha_required(tmp_path, monkeypatch):
    archive(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="terminal SHA"):
        feedback.score(tmp_path, FREEZE_SHA, "b" * 64)
    assert not (tmp_path / study.FEEDBACK).exists()
