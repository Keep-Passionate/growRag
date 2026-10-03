"""Catalog budget registration tests, offline and without touching legacy modules."""

from pathlib import Path

import pytest

from growrag.experiments import a1_catalog_budget as budget
from growrag.experiments.pre_pilot import write_json


def root_fixture(runs, suffix, protocol, ledger=None):
    root = runs / (budget.PREFIX + suffix)
    root.mkdir()
    write_json(root / "launch_plan.json", {"protocol": protocol, "model": budget.old.PILOT_MODEL})
    write_json(root / "final_budget.json", ledger or {"api_requests": 1, "calls": [{}]})
    return root


def test_old_and_new_a1_ledgers_registered_once_with_all_older_roots(tmp_path, monkeypatch):
    old = root_fixture(tmp_path, "synthetic_v1", budget.QUOTE_PROTOCOL)
    new = root_fixture(tmp_path, "catalog_v1", budget.PROTOCOL)
    earlier = [
        (budget.old.OLD_A0_PREFIX + "earlier", budget.old.OLD_A0_PROTOCOL),
        (budget.old.PREFIX + "earlier", budget.old.PROTOCOL),
    ]
    for name, protocol in earlier:
        root = tmp_path / name
        root.mkdir()
        write_json(
            root / "launch_plan.json", {"protocol": protocol, "model": budget.old.PILOT_MODEL}
        )
        write_json(root / "final_budget.json", {"api_requests": 1, "calls": [{}]})
    captured = []

    def reconcile(runs, **kwargs):
        captured.append(kwargs["reviewed_extra_ledgers"])
        return {
            "prior_reserved_cny": 0.4,
            "prior_total_actual_cny": None,
            "prior_unknown_cost_requests": 1,
        }

    monkeypatch.setattr(budget, "reconcile_history", reconcile)
    value = budget.reviewed_history(tmp_path)
    selected = captured[0]
    assert all(
        str((root / "final_budget.json").relative_to(tmp_path).as_posix()) in selected
        for root in [old, new]
    )
    assert all(f"{name}/final_budget.json" in selected for name, _ in earlier)
    assert set(budget.old.HISTORICAL_ROOTS) <= set(selected)
    assert len(selected) == len(set(selected))
    assert value["prior_reserved_cny"] == 0.4
    assert value["prior_total_actual_cny"] is None


def test_unknown_a1_protocol_not_whitelisted_by_broad_prefix(tmp_path):
    root_fixture(tmp_path, "foreign", "unreviewed-other-observer")
    with pytest.raises(ValueError, match="unregistered"):
        budget.reviewed_history(tmp_path)


def test_unfinished_journal_cannot_reset_budget(tmp_path):
    journal = tmp_path / (budget.PREFIX + "unfinished") / "request_journal"
    journal.mkdir(parents=True)
    write_json(journal / "0000_intent.json", {"status": "pending_no_automatic_retry"})
    with pytest.raises(ValueError, match="unfinished"):
        budget.reviewed_history(tmp_path)


@pytest.mark.parametrize(
    "ledger",
    [
        {"api_requests": 0, "calls": [], "reserved_cny": 0.1},
        {"api_requests": 0, "calls": [{}], "reserved_cny": 0},
        {"api_requests": True, "calls": [], "reserved_cny": 0},
    ],
)
def test_unexplained_reservations_and_invalid_counts_are_not_omitted(tmp_path, ledger):
    root_fixture(tmp_path, "invalid", budget.PROTOCOL, ledger)
    with pytest.raises(ValueError):
        budget.reviewed_history(tmp_path)


def test_old_budget_source_is_not_modified_or_reconfigured():
    from growrag.experiments import a1_budget as old_a1

    assert old_a1.PROTOCOL == budget.QUOTE_PROTOCOL
    assert old_a1.PROTOCOL != budget.PROTOCOL
    assert budget.PREFIX == old_a1.PREFIX
    assert budget.A1_PROTOCOLS == {old_a1.PROTOCOL, budget.PROTOCOL}
    assert Path(old_a1.__file__).name == "a1_budget.py"
