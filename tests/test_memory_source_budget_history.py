"""The independent source collection shares, never resets, project spending."""

import json

import pytest

from growrag.experiments import run_shared_s2g as runner


def source_root(tmp_path, protocol="growrag-independent-memory-source-collection-v1"):
    root = tmp_path / "2026-09-27_memory_source_v1_0000_0008"
    root.mkdir()
    (root / "launch_plan.json").write_text(
        json.dumps({"protocol": protocol, "model": runner.PILOT_MODEL})
    )
    return root


def test_source_ledger_added_to_project_history(tmp_path, monkeypatch):
    root = source_root(tmp_path)
    (root / "final_budget.json").write_text(json.dumps({"api_requests": 1, "calls": [1]}))
    captured = {}

    def reconcile(runs, *, reviewed_extra_ledgers):
        captured["roots"] = reviewed_extra_ledgers
        return {"authorized_total_cny": 50}

    monkeypatch.setattr(runner, "reconcile_history", reconcile)
    assert runner.reviewed_history(tmp_path)["authorized_total_cny"] == 50
    assert f"{root.name}/final_budget.json" in captured["roots"]


def test_unreviewed_source_protocol_rejected(tmp_path, monkeypatch):
    root = source_root(tmp_path, "unreviewed")
    (root / "final_budget.json").write_text(json.dumps({"api_requests": 1, "calls": [1]}))
    monkeypatch.setattr(runner, "reconcile_history", lambda *a, **k: pytest.fail("unsafe"))
    with pytest.raises(ValueError, match="unreviewed"):
        runner.reviewed_history(tmp_path)


def test_pending_source_request_blocks_every_series(tmp_path):
    root = source_root(tmp_path)
    (root / "request_journal").mkdir()
    (root / "request_journal" / "0000_intent.json").write_text("{}")
    with pytest.raises(ValueError, match="unfinished"):
        runner.reviewed_history(tmp_path)
