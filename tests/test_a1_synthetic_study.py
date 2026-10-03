"""Offline controls: mock transport is never a scientific model result."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import a1_budget as budget
from growrag.experiments import a1_synthetic_study as study

PROJECT = Path(__file__).resolve().parents[1]


def test_fixture_integrity_and_no_expected_in_messages():
    rows = study.load_fixture(PROJECT)
    assert len(rows) == 32
    assert len({study.fingerprint(r["input"]) for r in rows}) == 30
    for row in rows:
        prepared = study.prepare_payload(row["input"])
        messages = study.observer_messages(prepared)
        assert json.loads(messages[-1]["content"]) == row["input"]
        for key in ("row_id", "pair_id", "expected", "nominal_vector"):
            assert key not in messages[-1]["content"]


class Report:
    def __init__(self, decision):
        self.decision = decision

    def to_dict(self):
        return {"decision": self.decision}


def install_fake_decision(monkeypatch):
    monkeypatch.setattr(study, "decide", lambda report, **kwargs: report)


def test_nominal_gate_and_missing_are_different(monkeypatch):
    install_fake_decision(monkeypatch)
    rows = study.load_fixture(PROJECT)
    reports = {r["row_id"]: Report(r["expected"]["contradiction_only"]) for r in rows}
    gate, detail = study.gate_summary(rows, reports)
    assert gate == {
        "completed": 32,
        "main_matches": 32,
        "A_retained": 16,
        "conflict_B_rejected": 13,
        "unknown_B_preserved": 3,
        "passed": True,
    }
    assert len(detail) == 32
    del reports[rows[0]["row_id"]]
    gate, detail = study.gate_summary(rows, reports)
    assert not gate["passed"]
    assert gate["completed"] == 31
    assert detail[0]["status"] == "missing"


@pytest.mark.parametrize(
    "changed,decision", [("14B", "reject"), ("15B", "allow"), ("16B", "allow")]
)
def test_unknown_preservation_is_an_independent_gate(monkeypatch, changed, decision):
    install_fake_decision(monkeypatch)
    rows = study.load_fixture(PROJECT)
    reports = {r["row_id"]: Report(r["expected"]["contradiction_only"]) for r in rows}
    reports[changed] = Report(decision)
    gate, _ = study.gate_summary(rows, reports)
    assert gate["main_matches"] == 31
    assert not gate["passed"]


def test_all_reject_is_not_a_success(monkeypatch):
    install_fake_decision(monkeypatch)
    rows = study.load_fixture(PROJECT)
    gate, _ = study.gate_summary(rows, {r["row_id"]: Report("reject") for r in rows})
    assert not gate["passed"] and gate["A_retained"] == 0


def mock_observation(monkeypatch, *, fail_at=None):
    counter = []

    def resolve(content, prepared):
        counter.append(content)
        if len(counter) == fail_at:
            raise study.ConditionContractError("deliberate offline contract fixture")
        return Report("allow")

    monkeypatch.setattr(study, "resolve_response", resolve)
    return counter


def test_first_contract_failure_preserved_and_no_resend(monkeypatch, tmp_path):
    rows = study.load_fixture(PROJECT)[:4]
    count = mock_observation(monkeypatch, fail_at=2)
    sent = []

    def complete(messages, **kwargs):
        sent.append((messages, kwargs))
        return SimpleNamespace(content="mock raw output")

    terminal, stop = study.observe_rows(
        rows, SimpleNamespace(complete=complete), tmp_path, lambda: None
    )
    assert stop == "observer_contract_failure"
    assert [r["status"] for r in terminal] == [
        "completed",
        "failed",
        "not_attempted",
        "not_attempted",
    ]
    assert len(sent) == len(count) == 2
    failed = json.loads((tmp_path / "row_01.json").read_bytes())
    assert failed["raw_content"] == "mock raw output"
    assert failed["failure_type"] == "ConditionContractError"
    assert not (tmp_path / "row_02.json").exists()
    assert sent[0][1]["trace_id"].endswith("/row/00")


def test_guard_failure_never_sends(monkeypatch, tmp_path):
    mock_observation(monkeypatch)

    def fail():
        raise ValueError("synthetic frozen identity changed")

    def forbidden(*args, **kwargs):
        pytest.fail("transport must not be reached")

    terminal, stop = study.observe_rows(
        study.load_fixture(PROJECT)[:2], SimpleNamespace(complete=forbidden), tmp_path, fail
    )
    assert stop == "ValueError"
    assert [r["status"] for r in terminal] == ["failed", "not_attempted"]


def test_semantic_disagreement_is_not_a_contract_stop(monkeypatch, tmp_path):
    mock_observation(monkeypatch)
    client = SimpleNamespace(complete=lambda *args, **kwargs: SimpleNamespace(content="mock"))
    terminal, stop = study.observe_rows(
        study.load_fixture(PROJECT)[:3], client, tmp_path, lambda: None
    )
    assert stop is None and all(r["status"] == "completed" for r in terminal)


def test_dryrun_cannot_read_key_or_reconcile_paid_budget(monkeypatch):
    monkeypatch.setattr(study, "load_freeze", lambda *args: {})

    def forbidden(*args, **kwargs):
        pytest.fail("dryrun must not read credentials, budget or contact transport")

    monkeypatch.setattr(study, "read_local_bailian_settings", forbidden)
    monkeypatch.setattr(study, "reviewed_history", forbidden)
    assert study.run(PROJECT, "a" * 64) == {"dry_run": True, "planned_rows": 32, "api_calls": 0}


def test_no_retry_when_output_or_claim_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "load_freeze", lambda *args: {})
    monkeypatch.setattr(study, "load_fixture", lambda *args: [])
    (tmp_path / study.OUTPUT).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already claimed"):
        study.run(tmp_path, "a" * 64, allow_network=True)


def test_budget_inherits_both_a0_generations_and_registers_a1(monkeypatch, tmp_path):
    roots = [
        (budget.old.OLD_A0_PREFIX + "x", budget.old.OLD_A0_PROTOCOL),
        (budget.old.PREFIX + "x", budget.old.PROTOCOL),
        (budget.PREFIX + "x", budget.PROTOCOL),
    ]
    for name, protocol in roots:
        root = tmp_path / name
        root.mkdir()
        study.write_json(
            root / "launch_plan.json", {"protocol": protocol, "model": budget.old.PILOT_MODEL}
        )
        study.write_json(root / "final_budget.json", {"api_requests": 1, "calls": [{}]})
    monkeypatch.setattr(budget.old, "reconcile_history", lambda runs, **kwargs: kwargs)
    result = budget.reviewed_history(tmp_path)["reviewed_extra_ledgers"]
    assert all(f"{name}/final_budget.json" in result for name, _ in roots)
    assert all(path in result for path in budget.old.HISTORICAL_ROOTS)
    assert len(result) == len(set(result))


def test_unfinished_budget_cannot_be_silently_omitted(tmp_path):
    journal = tmp_path / (budget.PREFIX + "interrupted") / "request_journal"
    journal.mkdir(parents=True)
    study.write_json(journal / "0000_intent.json", {"status": "pending_no_automatic_retry"})
    with pytest.raises(ValueError, match="unfinished"):
        budget.reviewed_history(tmp_path)


def test_interrupted_unledgered_attempt_is_unknown_not_zero():
    client = SimpleNamespace(
        report=lambda: {
            "api_requests": 1,
            "calls": [],
            "reserved_cny": 0.1,
            "estimated_actual_cny": 0,
        }
    )
    value = study.final_budget_report(client, 1.0)
    assert value["estimated_actual_cny"] is None
    assert value["reconciliation_required"] is True
    assert value["reserved_cny"] == 0.1
    assert value["api_requests"] == 1 and value["calls"] == []


def test_no_transport_client_is_genuinely_zero():
    value = study.final_budget_report(None, 0.5)
    assert value["api_requests"] == 0 and value["estimated_actual_cny"] == 0
    assert value["limits"]["budget_cny"] == 0.5


def test_freeze_changed_fixture_refuses_before_any_api(tmp_path):
    target = tmp_path / study.FIXTURE
    target.parent.mkdir(parents=True)
    raw = deepcopy(json.loads((PROJECT / study.FIXTURE).read_bytes()))
    raw["rows"][0]["input"]["original_question"] += "changed"
    study.write_json(target, raw)
    with pytest.raises(ValueError, match="fixture changed"):
        study.load_fixture(tmp_path)
