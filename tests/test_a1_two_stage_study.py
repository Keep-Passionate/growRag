"""离线运行器合同测试：假响应不是模型效果，也不产生付费调用。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import a1_two_stage_study as study

PROJECT = Path(__file__).resolve().parents[1]


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


class OfflineClient:
    def __init__(self, rows, bad=None, fatal=None):
        self.rows, self.sent, self.bad, self.fatal = rows, [], bad or {}, fatal

    def complete(self, messages, *, trace_id, prompt_version):
        self.sent.append((messages, trace_id, prompt_version))
        if self.fatal is not None and len(self.sent) == self.fatal:
            raise ValueError("offline fatal transport fixture")
        index, stage = trace_id.split("/")[-2:]
        prepared = study.prepare_payload(self.rows[int(index)]["input"])
        raw = self.bad.get((int(index), stage))
        if raw is None:
            raw = locator_wire(prepared) if stage == "locate" else unknown_wire(prepared)
        return SimpleNamespace(content=raw)


def test_manifest_reuses_parent_and_new_paths():
    rows = study.load_fixture(PROJECT)
    assert rows == study.previous.load_fixture(PROJECT)
    assert len(rows) == 32
    assert len({study.input_sha(row["input"]) for row in rows}) == 30
    assert study.OUTPUT == "runs/2026-10-03_a1ts_synthetic_v1"
    assert study.CONFIG["max_calls"] == 64
    assert study.CONFIG["retry"] is False
    assert study.CONFIG["suite_role"] == "development_interface_regression_not_blind_test"


def test_success_two_actual_stages_and_audit_binding(tmp_path):
    rows = study.load_fixture(PROJECT)[:1]
    client = OfflineClient(rows)
    terminal, stop = study.observe_rows(rows, client, tmp_path, lambda: None)
    assert stop is None and len(client.sent) == 2
    assert terminal[0]["status"] == "completed"
    record = json.loads((tmp_path / "row_00.json").read_bytes())
    assert record["observation_status"] == "valid"
    assert record["fatal"] is False
    assert set(record["stages"]) == {"locate", "judge"}
    for i, stage in enumerate(("locate", "judge")):
        sent, trace, version = client.sent[i]
        value = record["stages"][stage]
        assert value["messages"] == sent
        assert value["trace_id"] == trace
        assert value["prompt_version"] == version
        assert value["messages_sha256"] == study.input_sha(sent)
        assert value["status"] == "completed"
    prepared = study.prepare_payload(rows[0]["input"])
    location = study.observer.resolve_location(record["stages"]["locate"]["raw_content"], prepared)
    report, audit = study.observer.resolve_judgment(
        record["stages"]["judge"]["raw_content"], prepared, location
    )
    assert audit.to_dict() == record["two_stage_audit"]
    assert report.to_dict() == record["report"]


@pytest.mark.parametrize("stage,count", [("locate", 5), ("judge", 6)])
def test_local_failure_continues_independent_rows_without_retry(tmp_path, stage, count):
    rows = study.load_fixture(PROJECT)[:3]
    client = OfflineClient(rows, bad={(0, stage): "not-json"})
    terminal, stop = study.observe_rows(rows, client, tmp_path, lambda: None)
    assert stop is None
    assert [x["status"] for x in terminal] == ["failed", "completed", "completed"]
    assert len(client.sent) == count
    assert len({x[1] for x in client.sent}) == count
    record = json.loads((tmp_path / "row_00.json").read_bytes())
    assert record["fatal"] is False
    assert record["failure_stage"] == stage
    assert record["observation_status"] == "model_output_error"
    assert "two_stage_audit" in record
    if stage == "locate":
        assert "judge" not in record["stages"]


def test_fatal_transport_stops_and_keeps_missing_suffix(tmp_path):
    rows = study.load_fixture(PROJECT)[:3]
    client = OfflineClient(rows, fatal=2)
    terminal, stop = study.observe_rows(rows, client, tmp_path, lambda: None)
    assert stop == "ValueError"
    assert len(client.sent) == 2
    assert [x["status"] for x in terminal] == ["failed", "not_attempted", "not_attempted"]
    record = json.loads((tmp_path / "row_00.json").read_bytes())
    assert record["fatal"] is True
    assert record["stages"]["judge"]["status"] == "failed"
    assert "raw_content" not in record["stages"]["judge"]


def test_guard_failure_sends_nothing(tmp_path):
    rows = study.load_fixture(PROJECT)[:2]
    client = OfflineClient(rows)

    def guard():
        raise ValueError("offline source changed")

    terminal, stop = study.observe_rows(rows, client, tmp_path, guard)
    assert not client.sent and stop == "ValueError"
    assert [x["status"] for x in terminal] == ["failed", "not_attempted"]


def test_complete_suite_never_more_than_64_calls(tmp_path):
    rows = study.load_fixture(PROJECT)
    client = OfflineClient(rows)
    terminal, stop = study.observe_rows(rows, client, tmp_path, lambda: None)
    assert len(client.sent) == 64 and stop is None
    assert len(terminal) == 32


def test_dryrun_never_reads_key_or_budget(monkeypatch):
    monkeypatch.setattr(study, "load_freeze", lambda *a: {})

    def forbidden(*a, **k):
        pytest.fail("offline dry run must not read key/budget/transport")

    for name in ("read_local_bailian_settings", "reviewed_history", "LiveChatClient"):
        monkeypatch.setattr(study, name, forbidden)
    result = study.run(PROJECT, "a" * 64)
    assert result["api_calls"] == 0 and result["max_calls"] == 64


@pytest.mark.parametrize("existing", ["output", "claim"])
def test_claim_prevents_paid_reentry(monkeypatch, tmp_path, existing):
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


def test_unverified_budget_stops_before_key_or_claim(monkeypatch, tmp_path):
    monkeypatch.setitem(study.CONFIG, "project_cap_cny", 50.0)
    monkeypatch.setattr(study, "load_freeze", lambda *a: {})
    monkeypatch.setattr(study, "load_fixture", lambda *a: [])
    monkeypatch.setattr(study, "_git_state", lambda: {"worktree_dirty": False})
    monkeypatch.setattr(study, "reviewed_history", lambda *a: {"prior_reserved_cny": 53.6229374})
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(
        study, "read_local_bailian_settings", lambda *a: pytest.fail("must not read key")
    )
    with pytest.raises(ValueError, match="authorization needed"):
        study.run(tmp_path, "a" * 64, allow_network=True)
    assert not (tmp_path / study.OUTPUT).exists()


def test_uncapped_project_preserves_finite_batch_cap_with_nonzero_history(monkeypatch):
    monkeypatch.setitem(study.CONFIG, "project_cap_cny", None)
    monkeypatch.setitem(study.CONFIG, "series_cap_cny", 1.0)
    assert study.available_subcap(53.6229374) == 1.0
    assert study.available_subcap(0) == 1.0
    assert study.CONFIG["handoff_cap_cny"] is None
    encoded = json.dumps(study.CONFIG, allow_nan=False)
    restored = json.loads(encoded)
    assert restored["project_cap_cny"] is None
    assert restored["handoff_cap_cny"] is None


@pytest.mark.parametrize("prior", [None, True, "53.62", -1, float("nan"), float("inf")])
def test_uncapped_project_still_rejects_invalid_historical_reservations(monkeypatch, prior):
    monkeypatch.setitem(study.CONFIG, "project_cap_cny", None)
    with pytest.raises(ValueError, match="historical reservation"):
        study.available_subcap(prior)


@pytest.mark.parametrize("cap", [True, "50", 0, -1, float("nan"), float("inf")])
def test_invalid_finite_project_cap_is_not_treated_as_uncapped(monkeypatch, cap):
    monkeypatch.setitem(study.CONFIG, "project_cap_cny", cap)
    with pytest.raises(ValueError, match="project cap"):
        study.available_subcap(53.6229374)


@pytest.mark.parametrize("cap", [None, True, "1", 0, -1, float("nan"), float("inf")])
def test_uncapped_project_still_requires_a_positive_finite_batch_cap(monkeypatch, cap):
    monkeypatch.setitem(study.CONFIG, "project_cap_cny", None)
    monkeypatch.setitem(study.CONFIG, "series_cap_cny", cap)
    with pytest.raises(ValueError, match="batch reservation limit"):
        study.available_subcap(53.6229374)


def test_finite_project_contract_remains_available(monkeypatch):
    monkeypatch.setitem(study.CONFIG, "project_cap_cny", 50.0)
    monkeypatch.setitem(study.CONFIG, "series_cap_cny", 1.0)
    assert study.available_subcap(53.6229374) < 0
    assert study.available_subcap(49.5) == 0.5
    assert study.available_subcap(48) == 1.0


def test_freeze_anchors_authorization_and_json_null_contract(monkeypatch, tmp_path):
    """Only synthetic temporary files are written; no actual freeze is changed."""
    rows = study.load_fixture(PROJECT)[:1]
    monkeypatch.setattr(study, "load_fixture", lambda *a: rows)
    monkeypatch.setattr(study, "source_snapshot", lambda *a: {"sha256": "offline-source"})
    monkeypatch.setattr(study, "_git_state", lambda: {"commit": "offline", "worktree_dirty": False})
    monkeypatch.setitem(study.CONFIG, "project_cap_cny", None)
    for path, content in (
        (tmp_path / study.NOTE, "offline protocol note"),
        (tmp_path / study.AUTHORIZATION, "offline user authorization fixture"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (tmp_path / "runs").mkdir()
    result = study.freeze(tmp_path)
    frozen_path = tmp_path / study.FREEZE
    frozen = json.loads(frozen_path.read_bytes())
    assert frozen["budget_authorization_sha256"] == study._sha(tmp_path / study.AUTHORIZATION)
    assert frozen["configuration"]["project_cap_cny"] is None
    assert study.load_freeze(tmp_path, result["freeze_sha256"]) == frozen
    (tmp_path / study.AUTHORIZATION).write_text("changed authorization", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol differs"):
        study.load_freeze(tmp_path, result["freeze_sha256"])


def test_uncapped_launch_preserves_prior_ledger_and_uses_one_yuan_batch(monkeypatch, tmp_path):
    """Real budget/journal wrapper, synthetic transport: never a network request."""
    rows = study.load_fixture(PROJECT)[:1]
    history = {"prior_reserved_cny": 53.6229374, "offline_existing_ledger": "unchanged"}
    original_history = json.dumps(history, sort_keys=True)
    frozen = {"row_ids": [row["row_id"] for row in rows]}
    delegates = []

    class FakeTransport:
        transport_source = "offline_test"

        def __init__(self, config, audit_path, **kwargs):
            self.config, self.attempts = config, 0
            delegates.append(self)

        def complete(self, messages, **kwargs):
            self.attempts += 1
            prepared = study.prepare_payload(rows[0]["input"])
            stage = kwargs["trace_id"].rsplit("/", 1)[-1]
            return SimpleNamespace(
                content=locator_wire(prepared) if stage == "locate" else unknown_wire(prepared),
                input_tokens=10,
                output_tokens=5,
                returned_model=self.config.model,
                audit_path=tmp_path / "offline_response.json",
                request_id="offline_test_only",
            )

    monkeypatch.setitem(study.CONFIG, "project_cap_cny", None)
    monkeypatch.setitem(study.CONFIG, "series_cap_cny", 1.0)
    monkeypatch.setattr(study, "load_freeze", lambda *a: frozen)
    monkeypatch.setattr(study, "load_fixture", lambda *a: rows)
    monkeypatch.setattr(study, "_git_state", lambda: {"worktree_dirty": False})
    monkeypatch.setattr(study, "reviewed_history", lambda *a: history)
    monkeypatch.setattr(study, "source_snapshot", lambda *a: {"files": {}, "sha256": "offline"})
    monkeypatch.setattr(study, "LiveChatClient", FakeTransport)
    monkeypatch.setattr(
        study,
        "read_local_bailian_settings",
        lambda *a: SimpleNamespace(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key="offline-dummy"
        ),
    )
    (tmp_path / "runs").mkdir()
    prior_path = tmp_path / "runs" / "offline_prior_ledger.json"
    study.write_json(prior_path, history)
    prior_bytes = prior_path.read_bytes()
    result = study.run(tmp_path, "a" * 64, allow_network=True)
    assert result["status"] == "completed"
    directory = tmp_path / study.OUTPUT
    plan = json.loads((directory / "launch_plan.json").read_bytes())
    ledger = json.loads((directory / "final_budget.json").read_bytes())
    assert delegates[0].attempts == ledger["api_requests"] == 2
    assert plan["project_cap_cny"] is None
    assert plan["prior_reserved_cny"] == history["prior_reserved_cny"]
    assert plan["subcap_cny"] == study.available_subcap(history["prior_reserved_cny"]) == 1.0
    assert ledger["limits"]["budget_cny"] == 1.0
    assert 0 < ledger["reserved_cny"] < 1.0
    assert json.dumps(history, sort_keys=True) == original_history
    assert prior_path.read_bytes() == prior_bytes


def test_cli_score_cannot_enable_network():
    with pytest.raises(SystemExit):
        study.main(["--score", "--allow-network", "--expected-terminal-sha256", "a" * 64])


@pytest.mark.parametrize("failure", ["post_guard", "unknown_usage", "interrupted_attempt"])
def test_run_finally_seals_paid_attempt_even_when_it_cannot_finish(monkeypatch, tmp_path, failure):
    """真实budget/journal与fake transport接线，不会连接外部API。"""
    rows = study.load_fixture(PROJECT)[:2]
    frozen = {"row_ids": [row["row_id"] for row in rows]}
    delegates = []

    class FakeTransport:
        transport_source = "offline_test"

        def __init__(self, config, audit_path, **kwargs):
            self.config, self.attempts = config, 0
            delegates.append(self)

        def complete(self, messages, **kwargs):
            self.attempts += 1
            if failure == "interrupted_attempt":
                raise KeyboardInterrupt("offline interruption")
            return SimpleNamespace(
                content=locator_wire(study.prepare_payload(rows[0]["input"])),
                input_tokens=None if failure == "unknown_usage" else 10,
                output_tokens=5,
                returned_model=self.config.model,
                audit_path=tmp_path / "offline_response.json",
                request_id="offline_test_only",
            )

    def guard(*args):
        if failure == "post_guard" and delegates and delegates[0].attempts:
            raise ValueError("offline frozen source drift")
        return frozen

    monkeypatch.setattr(study, "load_freeze", guard)
    monkeypatch.setattr(study, "load_fixture", lambda *a: rows)
    monkeypatch.setattr(study, "_git_state", lambda: {"worktree_dirty": False})
    monkeypatch.setattr(study, "reviewed_history", lambda *a: {"prior_reserved_cny": 0})
    monkeypatch.setattr(study, "source_snapshot", lambda *a: {"files": {}, "sha256": "offline"})
    monkeypatch.setattr(study, "LiveChatClient", FakeTransport)
    monkeypatch.setattr(
        study,
        "read_local_bailian_settings",
        lambda *a: SimpleNamespace(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key="offline-dummy"
        ),
    )
    (tmp_path / "runs").mkdir()
    result = study.run(tmp_path, "a" * 64, allow_network=True)
    assert result["status"] == "stopped"
    directory = tmp_path / study.OUTPUT
    terminal = json.loads((directory / "TERMINAL.json").read_bytes())
    ledger = json.loads((directory / "final_budget.json").read_bytes())
    row = json.loads((directory / "row_00.json").read_bytes())
    assert delegates[0].attempts == ledger["api_requests"] == 1
    assert ledger["reserved_cny"] > 0
    assert [x["status"] for x in terminal["rows"]] == ["failed", "not_attempted"]
    assert "judge" not in row["stages"]
    assert (directory / "request_journal/0000_intent.json").exists()
    assert (directory / "request_journal/0000_after.json").exists()
    if failure == "post_guard":
        assert "raw_content" in row["stages"]["locate"]
        assert ledger["estimated_actual_cny"] > 0
    elif failure == "unknown_usage":
        assert ledger["estimated_actual_cny"] is None
        assert ledger["block_reason"] == "unknown_usage"
    else:
        assert ledger["estimated_actual_cny"] is None
        assert ledger["reconciliation_required"] is True
        assert ledger["calls"] == []
