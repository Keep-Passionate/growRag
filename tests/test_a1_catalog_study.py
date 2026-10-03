"""Offline catalog-v3 controls, never real model-quality evidence."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import a1_catalog_study as study

PROJECT = Path(__file__).resolve().parents[1]


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


def test_prospective_paths_and_unchanged_development_rows():
    rows = study.load_fixture(PROJECT)
    prior = json.loads((PROJECT / "experiments/fixtures/a1_conditions_v2.json").read_bytes())
    assert rows == prior["rows"]
    assert len(rows) == 32
    assert len({study.input_sha(row["input"]) for row in rows}) == 30
    assert study.OUTPUT == "runs/2026-10-03_a1_catalog_v1"
    assert study.FREEZE == "runs/a1_catalog_freeze_v1.json"
    assert study.FEEDBACK == "runs/a1_catalog_feedback_v1"
    assert study.CONFIG["suite_role"] == "development_interface_regression_not_blind_test"
    assert study.PROTOCOL == "growrag-a1-catalog-observer-v1"
    assert study.PROMPT_VERSION == "growrag-a1-observer-catalog-v3"


def test_only_input_catalog_locations_and_model_checks_reach_request():
    for row in study.load_fixture(PROJECT):
        prepared = study.prepare_payload(row["input"], {"private": "not-for-model"})
        messages = study.observer.messages(prepared)
        request = json.loads(messages[-1]["content"])
        assert set(request) == {"input", "ref_catalog", "model_checks"}
        assert request["input"] == row["input"]
        for ref in request["ref_catalog"]:
            assert set(ref) == {"ref_id", "source_id", "field"}
        assert "not-for-model" not in json.dumps(messages)
        assert not {"row_id", "pair_id", "expected", "variant"} & request.keys()


def test_completed_record_keeps_v3_catalog_mapping_and_real_messages_sha(tmp_path):
    row = study.load_fixture(PROJECT)[0]
    prepared = study.prepare_payload(row["input"])
    raw = unknown_wire(prepared)
    sent = []

    def complete(messages, **kwargs):
        sent.append((messages, kwargs))
        return SimpleNamespace(content=raw)

    terminal, stop = study.observe_rows(
        [row], SimpleNamespace(complete=complete), tmp_path, lambda: None
    )
    assert stop is None and terminal[0]["status"] == "completed"
    record = json.loads((tmp_path / "row_00.json").read_bytes())
    assert record["raw_content"] == raw
    assert record["raw_response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert record["messages"] == sent[0][0]
    assert record["messages_sha256"] == study.input_sha(sent[0][0])
    assert record["catalog"] == study.observer.catalog(prepared)
    assert record["catalog_sha256"] == study.input_sha(record["catalog"])
    assert record["catalog_audit"]["raw_text"] == raw
    assert record["catalog_audit"]["messages_sha256"] == record["messages_sha256"]
    assert record["mapped_v2_raw_text"] == record["report"]["raw_text"]
    assert record["mapped_v2_raw_text"] != raw
    assert (
        record["mapped_v2_sha256"]
        == hashlib.sha256(record["mapped_v2_raw_text"].encode()).hexdigest()
    )
    assert record["report"]["wire_version"] == "growrag-a1-observer-quote-v2"
    assert record["catalog_audit"]["wire_version"] == study.PROMPT_VERSION


@pytest.mark.parametrize("raw_kind", ["malformed", "invalid_ref", "missing_bridge_evidence"])
def test_first_mechanical_failure_stops_without_retry_and_retains_audit(tmp_path, raw_kind):
    rows = study.load_fixture(PROJECT)[:4]
    valid = unknown_wire(study.prepare_payload(rows[0]["input"]))
    prepared = study.prepare_payload(rows[1]["input"])
    if raw_kind == "malformed":
        invalid = "offline malformed catalog fixture"
    else:
        value = json.loads(unknown_wire(prepared))
        rcheck = next(item for item in value["checks"] if item["kind"] == "R")
        if raw_kind == "invalid_ref":
            rcheck["ref_ids"] = ["Q.text"]
        else:
            by_source = {
                item["source_id"]: item["ref_id"] for item in study.observer.catalog(prepared)
            }
            rcheck.update(
                status="supported",
                reason="permitted_intermediate_subgoal",
                ref_ids=[by_source["Q"], by_source["A1"]],
            )
        invalid = json.dumps(value)
    sent = []

    def complete(messages, **kwargs):
        sent.append(kwargs["trace_id"])
        return SimpleNamespace(content=valid if len(sent) == 1 else invalid)

    terminal, stop = study.observe_rows(
        rows, SimpleNamespace(complete=complete), tmp_path, lambda: None
    )
    assert stop == "observer_contract_failure" and len(sent) == 2
    assert [entry["status"] for entry in terminal] == [
        "completed",
        "failed",
        "not_attempted",
        "not_attempted",
    ]
    record = json.loads((tmp_path / "row_01.json").read_bytes())
    assert record["raw_content"] == record["catalog_audit"]["raw_text"] == invalid
    assert record["failure_type"] == "CatalogContractError"
    assert record["catalog_audit"]["outcome"] == "invalid_response"
    assert (record["mapped_v2_raw_text"] is not None) == (raw_kind == "missing_bridge_evidence")
    assert not (tmp_path / "row_02.json").exists()


def test_guard_failure_never_sends(tmp_path):
    def guard():
        raise ValueError("offline freeze change")

    def complete(*args, **kwargs):
        pytest.fail("transport is forbidden")

    terminal, stop = study.observe_rows(
        study.load_fixture(PROJECT)[:2], SimpleNamespace(complete=complete), tmp_path, guard
    )
    assert stop == "ValueError"
    assert [entry["status"] for entry in terminal] == ["failed", "not_attempted"]


def test_deep_request_and_mapping_tamper_is_rejected(tmp_path):
    row = study.load_fixture(PROJECT)[0]
    prepared = study.prepare_payload(row["input"])
    raw = unknown_wire(prepared)
    study.observe_rows(
        [row],
        SimpleNamespace(complete=lambda *a, **k: SimpleNamespace(content=raw)),
        tmp_path,
        lambda: None,
    )
    original = json.loads((tmp_path / "row_00.json").read_bytes())
    for key in ["messages", "catalog", "messages_sha256", "catalog_sha256", "raw_response_sha256"]:
        changed = deepcopy(original)
        changed[key] = [] if isinstance(original[key], list) else "changed"
        with pytest.raises(ValueError):
            study.verify_record_request(changed, prepared, required=True)
    _, audit = study.observer.resolve(raw, prepared)
    for key in ["catalog_audit", "mapped_v2_raw_text", "mapped_v2_sha256"]:
        changed = deepcopy(original)
        changed[key] = None
        with pytest.raises(ValueError, match="replay differs"):
            study.verify_catalog_replay(changed, audit)


def test_dryrun_does_not_read_key_budget_or_transport(monkeypatch):
    monkeypatch.setattr(study, "load_freeze", lambda *a: {})

    def forbidden(*a, **k):
        pytest.fail("dry run must not read key/budget or call transport")

    monkeypatch.setattr(study, "read_local_bailian_settings", forbidden)
    monkeypatch.setattr(study, "reviewed_history", forbidden)
    monkeypatch.setattr(study, "LiveChatClient", forbidden)
    assert study.run(PROJECT, "a" * 64) == {"dry_run": True, "planned_rows": 32, "api_calls": 0}


@pytest.mark.parametrize("existing", ["output", "claim"])
def test_claim_or_output_prevents_retry(monkeypatch, tmp_path, existing):
    monkeypatch.setattr(study, "load_freeze", lambda *a: {})
    monkeypatch.setattr(study, "load_fixture", lambda *a: [])
    if existing == "output":
        (tmp_path / study.OUTPUT).mkdir(parents=True)
    else:
        path = tmp_path / "runs" / (study.RUN_ID + ".claim.json")
        path.parent.mkdir()
        study.write_json(path, {})
    with pytest.raises(FileExistsError, match="already claimed"):
        study.run(tmp_path, "a" * 64, allow_network=True)


def test_changed_fixture_cannot_prepare_request(tmp_path):
    path = tmp_path / study.FIXTURE
    path.parent.mkdir(parents=True)
    fixture = json.loads((PROJECT / study.FIXTURE).read_bytes())
    fixture["rows"][0]["input"]["original_question"] += " changed"
    study.write_json(path, fixture)
    with pytest.raises(ValueError, match="fixture changed"):
        study.load_fixture(tmp_path)


class NominalOnlyReport:
    """Gate arithmetic fixture; never represented as a model response."""

    def __init__(self, decision):
        self.decision = decision

    def to_dict(self):
        return {"decision": self.decision}


def test_gate_thresholds_are_unchanged_and_unknown_preservation_independent(monkeypatch):
    monkeypatch.setattr(study, "decide", lambda report, **kwargs: report)
    rows = study.load_fixture(PROJECT)
    reports = {
        row["row_id"]: NominalOnlyReport(row["expected"]["contradiction_only"]) for row in rows
    }
    gate, _ = study.gate_summary(rows, reports)
    assert gate == {
        "completed": 32,
        "main_matches": 32,
        "A_retained": 16,
        "conflict_B_rejected": 13,
        "unknown_B_preserved": 3,
        "passed": True,
    }
    reports["14B"] = NominalOnlyReport("reject")
    gate, _ = study.gate_summary(rows, reports)
    assert gate["main_matches"] == 31 and not gate["passed"]
    del reports["14B"]
    gate, details = study.gate_summary(rows, reports)
    assert gate["completed"] == 31 and not gate["passed"]
    assert next(row for row in details if row["row_id"] == "14B")["status"] == "missing"


def test_unaccounted_interruption_is_not_free():
    report = study.final_budget_report(
        SimpleNamespace(
            report=lambda: {
                "api_requests": 1,
                "calls": [],
                "reserved_cny": 0.2,
                "estimated_actual_cny": 0,
            }
        ),
        1,
    )
    assert report["estimated_actual_cny"] is None
    assert report["reserved_cny"] == 0.2 and report["reconciliation_required"]
    assert study.final_budget_report(None, 0.5)["api_requests"] == 0
