"""No network: a new baseline must not reset the project's paid-call ledger."""

import json

import pytest

from growrag.experiments import run_shared_s2g as runner


def setup_root(tmp_path, protocol="growrag-reformer-public-qwen-v1"):
    root = tmp_path / "2026-09-27_reformer_hotpot_v1_0000_0025"
    root.mkdir()
    (root / "launch_plan.json").write_text(
        json.dumps({"protocol": protocol, "model": runner.PILOT_MODEL})
    )
    return root


def test_reformer_ledger_is_added_to_same_history(tmp_path, monkeypatch):
    root = setup_root(tmp_path)
    (root / "final_budget.json").write_text(json.dumps({"api_requests": 3, "calls": [1, 2, 3]}))
    captured = {}

    def reconcile(runs, *, reviewed_extra_ledgers):
        captured["roots"] = reviewed_extra_ledgers
        return {"authorized_total_cny": 50}

    monkeypatch.setattr(runner, "reconcile_history", reconcile)
    assert runner.reviewed_history(tmp_path)["authorized_total_cny"] == 50
    assert captured["roots"][-1] == f"{root.name}/final_budget.json"
    assert captured["roots"][: len(runner.HISTORICAL_ROOTS)] == runner.HISTORICAL_ROOTS


def test_reformer_protocol_cannot_silently_change(tmp_path, monkeypatch):
    root = setup_root(tmp_path, "unreviewed")
    (root / "final_budget.json").write_text(json.dumps({"api_requests": 1, "calls": [1]}))
    monkeypatch.setattr(runner, "reconcile_history", lambda *a, **k: pytest.fail("unsafe"))
    with pytest.raises(ValueError, match="unreviewed"):
        runner.reviewed_history(tmp_path)


def test_reformer_interrupted_journal_blocks_other_baselines(tmp_path):
    root = setup_root(tmp_path)
    (root / "request_journal").mkdir()
    (root / "request_journal" / "0000_intent.json").write_text("{}")
    with pytest.raises(ValueError, match="unfinished"):
        runner.reviewed_history(tmp_path)
