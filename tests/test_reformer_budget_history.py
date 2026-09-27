"""No network: a new baseline must not reset the project's paid-call ledger."""

import json

import pytest

from growrag.experiments import run_shared_s2g as runner


def setup_root(
    tmp_path, protocol="growrag-reformer-public-qwen-v1", prefix="2026-09-27_reformer_hotpot_v1_"
):
    root = tmp_path / f"{prefix}0000_0025"
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


def test_control_ledger_is_in_same_project_budget(tmp_path, monkeypatch):
    root = setup_root(
        tmp_path, "growrag-reformer-id-examples-v1", "2026-09-27_reformer_control_v1_"
    )
    (root / "final_budget.json").write_text(
        json.dumps({"api_requests": 5, "calls": [1, 2, 3, 4, 5]})
    )
    captured = {}

    def reconcile(runs, *, reviewed_extra_ledgers):
        captured["roots"] = reviewed_extra_ledgers
        return {"authorized_total_cny": 50}

    monkeypatch.setattr(runner, "reconcile_history", reconcile)
    assert runner.reviewed_history(tmp_path)["authorized_total_cny"] == 50
    assert f"{root.name}/final_budget.json" in captured["roots"]


def test_control_wrong_protocol_cannot_reset_budget(tmp_path, monkeypatch):
    root = setup_root(tmp_path, "unreviewed", "2026-09-27_reformer_control_v1_")
    (root / "final_budget.json").write_text(json.dumps({"api_requests": 1, "calls": [1]}))
    monkeypatch.setattr(runner, "reconcile_history", lambda *a, **k: pytest.fail("unsafe"))
    with pytest.raises(ValueError, match="unreviewed"):
        runner.reviewed_history(tmp_path)


def test_control_interruption_blocks_other_series(tmp_path):
    root = setup_root(
        tmp_path, "growrag-reformer-id-examples-v1", "2026-09-27_reformer_control_v1_"
    )
    (root / "request_journal").mkdir()
    (root / "request_journal" / "0000_intent.json").write_text("{}")
    with pytest.raises(ValueError, match="unfinished"):
        runner.reviewed_history(tmp_path)
