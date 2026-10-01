"""Synthetic project ledgers only; no user data, keys, or network."""

import hashlib
import json

import pytest

from growrag.experiments import history_budget as budget
from growrag.experiments import prior_budget, run_shared_s2g


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _root(tmp_path, *, prefix=budget.PREFIX, protocol=budget.PROTOCOL, model=None, suffix="0001"):
    root = tmp_path / f"{prefix}{suffix}"
    _write(
        root / "launch_plan.json",
        {"protocol": protocol, "model": budget.PILOT_MODEL if model is None else model},
    )
    return root


def _minimal_ledger(root, *, requests=1, calls=None, reserved=0.01):
    _write(
        root / "final_budget.json",
        {
            "api_requests": requests,
            "calls": [{}] if calls is None else calls,
            "reserved_cny": reserved,
        },
    )


def _real_ledger(path, trace, *, unknown=False):
    audit = path.parent / "api_audit" / f"{hashlib.sha256(trace.encode()).hexdigest()}.json"
    row = {
        "trace_id": trace,
        "audit_path": str(audit),
        "reserved_cny": 0.01,
        "estimated_actual_cny": None if unknown else 0.000001,
        "status": "failed" if unknown else "completed",
        "input_tokens": None if unknown else 1,
        "output_tokens": None if unknown else 1,
        "api_requests": 1,
    }
    _write(audit, {**row, "transport_source": "live_api"})
    _write(
        path,
        {
            "calls": [row],
            "api_requests": 1,
            "reserved_cny": 0.01,
            "limits": {"input_per_million_cny": 0.2, "output_per_million_cny": 0.8},
        },
    )


def _legacy_roots(tmp_path):
    for index, relative in enumerate((*prior_budget.ROOT_LEDGERS, *budget.HISTORICAL_ROOTS)):
        _real_ledger(tmp_path / relative, f"legacy-{index}")


def test_collects_every_reviewed_series_once_without_mutating_legacy_registry(
    tmp_path, monkeypatch
):
    before = {key: set(value) for key, value in run_shared_s2g.REVIEWED_OTHER_SERIES.items()}
    series = {
        budget.SHARED_PREFIX: budget.REVIEWED_PROTOCOLS,
        **budget.REVIEWED_OTHER_SERIES,
        budget.PREFIX: {budget.PROTOCOL},
    }
    expected = []
    for prefix, protocols in series.items():
        root = _root(tmp_path, prefix=prefix, protocol=sorted(protocols)[0])
        _minimal_ledger(root)
        expected.append(f"{root.name}/final_budget.json")
        _write(root / "cumulative_budget.json", {"reserved_cny": 99999})
    calls = []
    result = {"prior_reserved_cny": 7.5, "prior_unknown_cost_requests": 2}

    def reconcile(runs, *, reviewed_extra_ledgers):
        calls.append((runs, reviewed_extra_ledgers))
        return result

    monkeypatch.setattr(budget, "reconcile_history", reconcile)
    assert budget.reviewed_history(tmp_path) is result
    assert len(calls) == 1
    assert calls[0] == (tmp_path.resolve(), (*budget.HISTORICAL_ROOTS, *expected))
    assert run_shared_s2g.REVIEWED_OTHER_SERIES == before
    assert budget.PREFIX not in run_shared_s2g.REVIEWED_OTHER_SERIES


def test_v1_and_v2_share_one_cumulative_reconciliation(tmp_path, monkeypatch):
    for version in ("v1", "v2"):
        root = _root(tmp_path, protocol=f"growrag-history-calibration-{version}", suffix=version)
        _minimal_ledger(root)
    received = []
    monkeypatch.setattr(
        budget,
        "reconcile_history",
        lambda runs, *, reviewed_extra_ledgers: received.append(reviewed_extra_ledgers) or {},
    )
    budget.reviewed_history(tmp_path)
    assert received == [
        (
            *budget.HISTORICAL_ROOTS,
            f"{budget.PREFIX}v1/final_budget.json",
            f"{budget.PREFIX}v2/final_budget.json",
        )
    ]


@pytest.mark.parametrize(
    ("protocol", "model"),
    [("unreviewed", None), (budget.PROTOCOL, "unreviewed-model")],
)
def test_new_series_protocol_and_model_are_exact(tmp_path, monkeypatch, protocol, model):
    root = _root(tmp_path, protocol=protocol, model=model)
    _minimal_ledger(root)
    monkeypatch.setattr(budget, "reconcile_history", lambda *a, **k: pytest.fail("unsafe"))
    with pytest.raises(ValueError, match="unreviewed"):
        budget.reviewed_history(tmp_path)


@pytest.mark.parametrize("prefix", [budget.PREFIX, "2026-09-30_operator_v3_"])
def test_unfinished_old_or_new_request_journal_blocks_launch(tmp_path, monkeypatch, prefix):
    root = _root(tmp_path, prefix=prefix)
    _write(root / "request_journal" / "0000_intent.json", {"potential_reserved_cny": 0.03})
    monkeypatch.setattr(budget, "reconcile_history", lambda *a, **k: pytest.fail("unsafe"))
    with pytest.raises(ValueError, match="unfinished"):
        budget.reviewed_history(tmp_path)


@pytest.mark.parametrize("requests", [None, True, False, -1, 1.0, "1"])
def test_request_count_never_coerces_invalid_types(tmp_path, monkeypatch, requests):
    root = _root(tmp_path)
    _minimal_ledger(root, requests=requests)
    monkeypatch.setattr(budget, "reconcile_history", lambda *a, **k: pytest.fail("unsafe"))
    with pytest.raises(ValueError, match="invalid root"):
        budget.reviewed_history(tmp_path)


def test_zero_request_pending_calls_are_not_free(tmp_path, monkeypatch):
    root = _root(tmp_path)
    _minimal_ledger(root, requests=0)
    monkeypatch.setattr(budget, "reconcile_history", lambda *a, **k: pytest.fail("unsafe"))
    with pytest.raises(ValueError, match="pending"):
        budget.reviewed_history(tmp_path)


@pytest.mark.parametrize("reserved", [None, False, 0.02, -1, float("nan")])
def test_zero_request_reservation_must_be_explicit_zero(tmp_path, monkeypatch, reserved):
    root = _root(tmp_path)
    _minimal_ledger(root, requests=0, calls=[], reserved=reserved)
    monkeypatch.setattr(budget, "reconcile_history", lambda *a, **k: pytest.fail("unsafe"))
    with pytest.raises(ValueError, match="unexplained reservation"):
        budget.reviewed_history(tmp_path)


def test_completed_zero_request_root_does_not_create_billed_calls(tmp_path, monkeypatch):
    root = _root(tmp_path)
    _minimal_ledger(root, requests=0, calls=[], reserved=0)
    collected = []

    def reconcile(runs, *, reviewed_extra_ledgers):
        collected.extend(reviewed_extra_ledgers)
        return {"prior_reserved_cny": 1}

    monkeypatch.setattr(budget, "reconcile_history", reconcile)
    assert budget.reviewed_history(tmp_path)["prior_reserved_cny"] == 1
    assert tuple(collected) == budget.HISTORICAL_ROOTS


def test_real_reconciliation_preserves_unknown_reservations_and_cumulative_scope(tmp_path):
    _legacy_roots(tmp_path)
    root = _root(tmp_path)
    _real_ledger(root / "final_budget.json", "new-history-failure", unknown=True)
    _write(root / "cumulative_budget.json", {"reserved_cny": 999})
    result = budget.reviewed_history(tmp_path)
    count = len(prior_budget.ROOT_LEDGERS) + len(budget.HISTORICAL_ROOTS) + 1
    assert result["prior_api_requests"] == count
    assert result["prior_reserved_cny"] == pytest.approx(count * 0.01)
    assert result["prior_unknown_cost_requests"] == 1
    assert result["prior_total_actual_cny"] is None
    assert result["prior_known_estimated_cny"] == pytest.approx((count - 1) * 0.000001)
    # This historical field is not rewritten into a new authorization by this module.
    assert result["authorized_total_cny"] == 50.0


def test_real_reconciliation_rejects_duplicate_request_identity_across_old_and_new(tmp_path):
    _legacy_roots(tmp_path)
    root = _root(tmp_path)
    _real_ledger(root / "final_budget.json", "legacy-0")
    with pytest.raises(ValueError, match="overlapping"):
        budget.reviewed_history(tmp_path)


def test_unknown_live_protocol_audit_cannot_escape_project_accounting(tmp_path):
    _legacy_roots(tmp_path)
    _real_ledger(tmp_path / "unreviewed-series" / "final_budget.json", "unreviewed-call")
    with pytest.raises(ValueError, match="unaccounted"):
        budget.reviewed_history(tmp_path)


def test_new_history_audit_without_final_root_blocks_launch(tmp_path):
    _legacy_roots(tmp_path)
    root = _root(tmp_path)
    _write(
        root / "api_audit" / "orphan.json",
        {"trace_id": "orphan", "transport_source": "live_api"},
    )
    with pytest.raises(ValueError, match="unaccounted"):
        budget.reviewed_history(tmp_path)


def test_real_reconciliation_checks_actual_ledger_against_audit(tmp_path):
    _legacy_roots(tmp_path)
    root = _root(tmp_path)
    _real_ledger(root / "final_budget.json", "new-history")
    path = root / "final_budget.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["calls"][0]["input_tokens"] = 2
    _write(path, value)
    with pytest.raises(ValueError, match="usage differs"):
        budget.reviewed_history(tmp_path)
