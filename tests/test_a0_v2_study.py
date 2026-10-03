"""A0收费编排的离线保护测试；不读取密钥、真实题或金标。"""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from growrag.experiments import a0_v2_budget
from growrag.experiments import a0_v2_study as study
from growrag.experiments.pre_pilot import write_json
from growrag.experiments.protocol import RuntimeQuestion


def accounted_call(**changes):
    return {"status": "completed", "api_requests": 1, "estimated_actual_cny": 0.001, **changes}


def test_only_accounted_local_errors_can_continue():
    client = SimpleNamespace(calls=[accounted_call()], block_reason=None)
    assert study.recoverable_local_failure(study.A0LocalOutputError("bad output"), client, 0)
    assert not study.recoverable_local_failure(ValueError("retriever changed IDs"), client, 0)
    assert not study.recoverable_local_failure(OSError("transport"), client, 0)
    assert not study.recoverable_local_failure(study.A0LocalOutputError(), client, 1)
    client.block_reason = "budget exhausted"
    assert not study.recoverable_local_failure(study.A0LocalOutputError(), client, 0)
    assert client.block_reason == "budget exhausted"


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "failed"},
        {"api_requests": 0},
        {"estimated_actual_cny": None},
        {"estimated_actual_cny": float("nan")},
        {"estimated_actual_cny": -1},
        {"estimated_actual_cny": True},
        {"validation_status": "invalid_usage"},
    ],
)
def test_unknown_or_invalid_cost_never_allows_continuation(changes):
    client = SimpleNamespace(calls=[accounted_call(**changes)], block_reason=None)
    assert not study.recoverable_local_failure(study.A0LocalOutputError(), client, 0)


def prediction(qid, method, **changes):
    return {
        "question_id": qid,
        "method": method,
        "status": "completed",
        "gold_loaded": False,
        "memory_updated": False,
        **changes,
    }


def test_terminal_preserves_failed_and_unattempted(tmp_path):
    write_json(tmp_path / "q_base.json", prediction("q", "base", status="failed"))
    rows, hashes = study.terminal_rows(tmp_path, [RuntimeQuestion("q", "Question?")])
    assert rows[0]["arms"]["base"]["status"] == "failed"
    assert rows[0]["arms"]["fresh_original"] == {"status": "not_attempted", "path": None}
    assert set(hashes) == {"q_base.json"}


@pytest.mark.parametrize(
    "changes",
    [
        {"question_id": "other"},
        {"method": "other"},
        {"status": "started"},
        {"gold_loaded": True},
        {"memory_updated": True},
    ],
)
def test_terminal_rejects_wrong_identity_or_unsealed_rows(tmp_path, changes):
    write_json(tmp_path / "q_base.json", {**prediction("q", "base"), **changes})
    with pytest.raises(ValueError):
        study.terminal_rows(tmp_path, [RuntimeQuestion("q", "Question?")])


def test_claim_overlap_is_rechecked_under_lock(tmp_path):
    write_json(tmp_path / "prior.claim.json", {"question_ids": ["old"]})
    study.reject_claim_overlap(tmp_path, ["new"])
    with pytest.raises(ValueError, match="overlaps"):
        study.reject_claim_overlap(tmp_path, ["old"])


def fake_cohort(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    questions = tuple(RuntimeQuestion(f"q{i}", f"Question {i}?") for i in range(100))
    frozen = {"question_ids": [q.question_id for q in questions]}
    monkeypatch.setattr(study, "load_freeze", lambda *a: (frozen, {}, questions, None))
    monkeypatch.setattr(study, "_git_state", lambda: {"worktree_dirty": False})
    monkeypatch.setattr(study, "serial_lock", lambda *a: nullcontext())
    return questions


def test_interface_gate_stops_without_loading_gold_and_seals_all100(tmp_path, monkeypatch):
    fake_cohort(tmp_path, monkeypatch)
    batches = []

    def batch(project, frozen, manifest, questions, corpus, start, count, digest, **kwargs):
        batches.append((start, count))
        output = project / "runs" / f"{study.PREFIX}v1_{start:04d}_{start + count:04d}"
        output.mkdir()
        summary = {
            "status": "completed",
            "stop_reason": None,
            "complete_paired_questions": 4,
            "api_requests": 19,
            "trailing_consecutive_failed_arms": 0,
        }
        write_json(output / "SUMMARY.json", summary)
        return summary

    monkeypatch.setattr(study, "run_batch", batch)
    result = study.run(tmp_path, "f" * 64, allow_network=True)
    assert result["stop_reason"] == "first5_interface_gate_failed"
    assert batches == [(0, 5)]
    final = json.loads((tmp_path / study.OUTPUT / "TERMINAL.json").read_bytes())
    assert len(final["terminal"]) == 100
    assert final["gold_loaded"] is False
    assert all(
        a["status"] == "not_attempted"
        for row in final["terminal"][5:]
        for a in row["arms"].values()
    )
    with pytest.raises(FileExistsError):
        study.run(tmp_path, "f" * 64, allow_network=True)


def test_dryrun_does_not_read_credentials_or_start_batches(tmp_path, monkeypatch):
    fake_cohort(tmp_path, monkeypatch)
    monkeypatch.setattr(study, "read_local_bailian_settings", lambda *a: pytest.fail("key read"))
    monkeypatch.setattr(study, "run_batch", lambda *a: pytest.fail("paid batch"))
    assert study.run(tmp_path, "f" * 64)["api_calls"] == 0
    assert not (tmp_path / study.OUTPUT).exists()


def test_budget_registry_adds_only_reviewed_root_ledger(tmp_path, monkeypatch):
    root = tmp_path / f"{a0_v2_budget.PREFIX}synthetic"
    root.mkdir()
    write_json(
        root / "launch_plan.json",
        {"protocol": a0_v2_budget.PROTOCOL, "model": a0_v2_budget.PILOT_MODEL},
    )
    write_json(root / "final_budget.json", {"api_requests": 1, "calls": [{}]})
    captured = {}

    def reconcile(runs, *, reviewed_extra_ledgers):
        captured["roots"] = reviewed_extra_ledgers
        return {"synthetic": True}

    monkeypatch.setattr(a0_v2_budget, "reconcile_history", reconcile)
    assert a0_v2_budget.reviewed_history(tmp_path) == {"synthetic": True}
    assert f"{root.name}/final_budget.json" in captured["roots"]


def test_budget_registry_refuses_unfinished_request(tmp_path, monkeypatch):
    root = tmp_path / f"{a0_v2_budget.PREFIX}synthetic"
    journal = root / "request_journal"
    journal.mkdir(parents=True)
    write_json(journal / "intent.json", {"pending": True})
    monkeypatch.setattr(a0_v2_budget, "reconcile_history", lambda *a, **k: pytest.fail("reconcile"))
    with pytest.raises(ValueError, match="unfinished"):
        a0_v2_budget.reviewed_history(tmp_path)


def test_budget_inherits_both_old_a0_and_v2_without_refund(tmp_path, monkeypatch):
    roots = []
    for prefix, protocol in (
        (a0_v2_budget.OLD_A0_PREFIX, a0_v2_budget.OLD_A0_PROTOCOL),
        (a0_v2_budget.PREFIX, a0_v2_budget.PROTOCOL),
    ):
        root = tmp_path / f"{prefix}synthetic"
        root.mkdir()
        write_json(root / "launch_plan.json", {"protocol": protocol, "model": study.PILOT_MODEL})
        write_json(root / "final_budget.json", {"api_requests": 1, "calls": [{}]})
        roots.append(f"{root.name}/final_budget.json")
    monkeypatch.setattr(
        a0_v2_budget, "reconcile_history", lambda runs, **k: k["reviewed_extra_ledgers"]
    )
    registered = a0_v2_budget.reviewed_history(tmp_path)
    assert all(registered.count(root) == 1 for root in roots)


def test_frozen_guard_rejects_new_source_and_changed_asset(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    guard = object.__new__(study.FrozenInputGuard)
    guard.project, guard.sources, guard.hashes, guard.stamps = tmp_path, set(), {}, {}
    guard.check()
    write_json(source / "new.json", {"new": True})
    with pytest.raises(ValueError, match="source file set"):
        guard.check()
    guard.sources = {"src/new.json"}
    guard.hashes = {"src/new.json": "0" * 64}
    with pytest.raises(ValueError, match="bytes changed"):
        guard.check()


def test_guard_runs_before_any_budget_or_network_call(monkeypatch):
    class BrokenGuard:
        def check(self):
            raise ValueError("frozen source changed")

    client = object.__new__(study.GuardedBudgetClient)
    client.input_guard = BrokenGuard()
    monkeypatch.setattr(
        study.DurableBudgetClient, "complete", lambda *a, **k: pytest.fail("request")
    )
    with pytest.raises(ValueError, match="frozen"):
        client.complete([], trace_id="synthetic", prompt_version="synthetic")
