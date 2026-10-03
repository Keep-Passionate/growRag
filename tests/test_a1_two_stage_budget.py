"""Offline accounting fixtures only: no keys, API calls, or research questions."""

import hashlib
import json
from pathlib import Path

import pytest

from growrag.experiments import a1_two_stage_budget as budget
from growrag.experiments import prior_budget


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def root_fixture(runs, *, prefix=budget.PREFIX, protocol=budget.PROTOCOL, suffix="run", model=None):
    root = runs / f"{prefix}{suffix}"
    write(
        root / "launch_plan.json",
        {"protocol": protocol, "model": budget.catalog.old.PILOT_MODEL if model is None else model},
    )
    return root


def minimal_ledger(root, *, count=1, calls=None, reserved=0.01, **extra):
    write(
        root / "final_budget.json",
        {
            "api_requests": count,
            "calls": [{}] if calls is None else calls,
            "reserved_cny": reserved,
            **extra,
        },
    )


def real_ledger(path, trace, *, unknown=False):
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
    write(audit, {**row, "transport_source": "live_api"})
    write(
        path,
        {
            "calls": [row],
            "api_requests": 1,
            "reserved_cny": 0.01,
            "limits": {"input_per_million_cny": 0.2, "output_per_million_cny": 0.8},
        },
    )


def legacy_roots(runs):
    for index, relative in enumerate(
        (*prior_budget.ROOT_LEDGERS, *budget.catalog.old.HISTORICAL_ROOTS)
    ):
        real_ledger(runs / relative, f"legacy-{index}")


def test_all_old_protocols_and_new_series_registered_once_without_registry_mutation(
    tmp_path, monkeypatch
):
    old = budget.catalog.old
    before = {prefix: set(protocols) for prefix, protocols in old.REVIEWED_OTHER_SERIES.items()}
    old_a1 = budget.catalog.A1_PROTOCOLS
    series = {
        old.SHARED_PREFIX: old.REVIEWED_PROTOCOLS,
        **old.REVIEWED_OTHER_SERIES,
        old.HISTORY_PREFIX: old.REVIEWED_HISTORY_PROTOCOLS,
        old.CANDIDATE_PREFIX: old.CANDIDATE_PROTOCOLS,
        old.OLD_A0_PREFIX: {old.OLD_A0_PROTOCOL},
        old.PREFIX: {old.PROTOCOL},
        budget.catalog.PREFIX: old_a1,
        budget.PREFIX: {budget.PROTOCOL},
    }
    expected = []
    for prefix, protocols in series.items():
        for index, protocol in enumerate(sorted(protocols)):
            root = root_fixture(tmp_path, prefix=prefix, protocol=protocol, suffix=str(index))
            minimal_ledger(root)
            write(root / "cumulative_budget.json", {"reserved_cny": 99999})
            expected.append(f"{root.name}/final_budget.json")
    captured = []
    result = {"prior_reserved_cny": 2, "prior_total_actual_cny": None}

    def reconcile(runs, *, reviewed_extra_ledgers):
        captured.append((runs, reviewed_extra_ledgers))
        return result

    monkeypatch.setattr(budget, "reconcile_history", reconcile)
    monkeypatch.setattr(
        budget.catalog, "reviewed_history", lambda *args: pytest.fail("no separate legacy scan")
    )
    assert budget.reviewed_history(tmp_path) is result
    assert captured == [(tmp_path.resolve(), (*old.HISTORICAL_ROOTS, *expected))]
    assert len(captured[0][1]) == len(set(captured[0][1]))
    assert old.REVIEWED_OTHER_SERIES == before
    assert budget.catalog.A1_PROTOCOLS is old_a1
    assert budget.PROTOCOL not in old_a1
    assert budget.PREFIX not in old.REVIEWED_OTHER_SERIES
    assert not budget.PREFIX.startswith(budget.catalog.PREFIX)


@pytest.mark.parametrize(
    ("prefix", "protocol", "model"),
    [
        (budget.PREFIX, "unreviewed", None),
        (budget.catalog.PREFIX, budget.PROTOCOL, None),
        (budget.PREFIX, budget.catalog.PROTOCOL, None),
        (budget.PREFIX, budget.PROTOCOL, "unreviewed-model"),
    ],
)
def test_unknown_or_cross_series_protocol_and_model_rejected(tmp_path, prefix, protocol, model):
    root = root_fixture(tmp_path, prefix=prefix, protocol=protocol, model=model)
    minimal_ledger(root)
    with pytest.raises(ValueError, match="unregistered"):
        budget.reviewed_history(tmp_path)


@pytest.mark.parametrize(
    "prefix", [budget.PREFIX, budget.catalog.PREFIX, "2026-09-30_operator_v3_"]
)
def test_unfinished_old_or_new_journal_blocks_before_reconciliation(tmp_path, monkeypatch, prefix):
    root = root_fixture(tmp_path, prefix=prefix)
    write(root / "request_journal" / "0000_intent.json", {"potential_reserved_cny": 0.03})
    monkeypatch.setattr(budget, "reconcile_history", lambda *args, **kwargs: pytest.fail("unsafe"))
    with pytest.raises(ValueError, match="unfinished"):
        budget.reviewed_history(tmp_path)


def test_unresolved_final_ledger_is_not_a_completed_zero_cost_run(tmp_path):
    root = root_fixture(tmp_path)
    minimal_ledger(root, count=0, calls=[], reserved=0, reconciliation_required=True)
    with pytest.raises(ValueError, match="unresolved"):
        budget.reviewed_history(tmp_path)


@pytest.mark.parametrize("count", [None, True, False, -1, 1.0, "1"])
def test_request_counts_must_be_nonnegative_integers(tmp_path, count):
    root = root_fixture(tmp_path)
    minimal_ledger(root, count=count)
    with pytest.raises(ValueError, match="invalid root"):
        budget.reviewed_history(tmp_path)


@pytest.mark.parametrize("calls", [None, {}, 0, "pending"])
def test_calls_must_be_an_explicit_list(tmp_path, calls):
    root = root_fixture(tmp_path)
    write(root / "final_budget.json", {"api_requests": 1, "calls": calls, "reserved_cny": 0.01})
    with pytest.raises(ValueError, match="invalid root"):
        budget.reviewed_history(tmp_path)


def test_zero_requests_with_pending_calls_not_omitted(tmp_path):
    root = root_fixture(tmp_path)
    minimal_ledger(root, count=0)
    with pytest.raises(ValueError, match="pending"):
        budget.reviewed_history(tmp_path)


@pytest.mark.parametrize("reserved", [None, False, True, 0.02, -1, float("nan")])
def test_zero_request_reservation_must_be_explicit_numeric_zero(tmp_path, reserved):
    root = root_fixture(tmp_path)
    minimal_ledger(root, count=0, calls=[], reserved=reserved)
    with pytest.raises(ValueError, match="unexplained"):
        budget.reviewed_history(tmp_path)


def test_valid_zero_request_root_does_not_add_a_billed_ledger(tmp_path, monkeypatch):
    root = root_fixture(tmp_path)
    minimal_ledger(root, count=0, calls=[], reserved=0)
    selected = []
    monkeypatch.setattr(
        budget,
        "reconcile_history",
        lambda runs, **kwargs: selected.append(kwargs["reviewed_extra_ledgers"]) or {},
    )
    budget.reviewed_history(tmp_path)
    assert selected == [budget.catalog.old.HISTORICAL_ROOTS]


@pytest.mark.parametrize("target", ["root", "final_budget.json", "launch_plan.json"])
def test_resolved_root_and_ledger_paths_cannot_escape_runs(tmp_path, monkeypatch, target):
    root = root_fixture(tmp_path)
    minimal_ledger(root)
    escaped = root if target == "root" else root / target
    resolve = Path.resolve

    def guarded_resolve(path, *args, **kwargs):
        if path == escaped:
            return tmp_path.parent / "outside-runs"
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    with pytest.raises(ValueError, match="escapes runs"):
        budget.reviewed_history(tmp_path)


def test_real_reconciliation_keeps_both_failed_old_a1_and_new_unknown_reservations(tmp_path):
    legacy_roots(tmp_path)
    roots = [
        root_fixture(
            tmp_path,
            prefix=budget.catalog.PREFIX,
            protocol=budget.catalog.QUOTE_PROTOCOL,
            suffix="quote",
        ),
        root_fixture(tmp_path, prefix=budget.catalog.PREFIX, protocol=budget.catalog.PROTOCOL),
        root_fixture(tmp_path),
    ]
    for index, root in enumerate(roots):
        real_ledger(root / "final_budget.json", f"a1-failed-{index}", unknown=True)
        write(root / "cumulative_budget.json", {"reserved_cny": 99999})
    result = budget.reviewed_history(tmp_path)
    count = len(prior_budget.ROOT_LEDGERS) + len(budget.catalog.old.HISTORICAL_ROOTS) + len(roots)
    assert result["prior_api_requests"] == count
    assert result["prior_reserved_cny"] == pytest.approx(count * 0.01)
    assert result["prior_known_estimated_cny"] == pytest.approx((count - 3) * 0.000001)
    assert result["prior_unknown_cost_requests"] == 3
    assert result["prior_total_actual_cny"] is None
    assert result["authorized_total_cny"] == 50.0
    assert result["authorization_date"] == "2026-09-20"


def test_duplicate_old_and_new_request_identity_rejected(tmp_path):
    legacy_roots(tmp_path)
    root = root_fixture(tmp_path)
    real_ledger(root / "final_budget.json", "legacy-0")
    with pytest.raises(ValueError, match="overlapping"):
        budget.reviewed_history(tmp_path)


@pytest.mark.parametrize("prefix", [budget.PREFIX, "unreviewed-series_"])
def test_live_audits_without_registered_final_root_cannot_escape_accounting(tmp_path, prefix):
    legacy_roots(tmp_path)
    root = tmp_path / f"{prefix}orphan"
    write(
        root / "api_audit" / "orphan.json", {"transport_source": "live_api", "trace_id": "orphan"}
    )
    with pytest.raises(ValueError, match="unaccounted"):
        budget.reviewed_history(tmp_path)


def test_real_reconciliation_rejects_ledger_usage_different_from_audit(tmp_path):
    legacy_roots(tmp_path)
    root = root_fixture(tmp_path)
    final = root / "final_budget.json"
    real_ledger(final, "a1ts-usage")
    ledger = json.loads(final.read_bytes())
    ledger["calls"][0]["input_tokens"] = 2
    write(final, ledger)
    with pytest.raises(ValueError, match="usage differs"):
        budget.reviewed_history(tmp_path)
