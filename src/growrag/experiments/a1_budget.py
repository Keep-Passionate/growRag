"""A1 adds registered ledgers without dropping any older project expense."""

import json
from pathlib import Path

from . import a0_v2_budget as old

PREFIX = "2026-10-03_a1_"
PROTOCOL = "growrag-a1-synthetic-observer-v1"


def reviewed_history(runs: Path) -> dict:
    """No credit for failed requests or differences between estimates and reserves."""
    runs = runs.resolve(strict=True)
    series = {
        old.SHARED_PREFIX: frozenset(old.REVIEWED_PROTOCOLS),
        **{p: frozenset(v) for p, v in old.REVIEWED_OTHER_SERIES.items()},
        old.HISTORY_PREFIX: old.REVIEWED_HISTORY_PROTOCOLS,
        old.CANDIDATE_PREFIX: old.CANDIDATE_PROTOCOLS,
        old.OLD_A0_PREFIX: frozenset({old.OLD_A0_PROTOCOL}),
        old.PREFIX: frozenset({old.PROTOCOL}),
        PREFIX: frozenset({PROTOCOL}),
    }
    extra = list(old.HISTORICAL_ROOTS)
    for prefix, protocols in series.items():
        for root in sorted(runs.glob(f"{prefix}*")):
            if not root.is_dir():
                continue
            if not root.resolve(strict=True).is_relative_to(runs):
                raise ValueError("ledger root escapes runs")
            final, journal = root / "final_budget.json", root / "request_journal"
            if journal.is_dir() and any(journal.iterdir()) and not final.is_file():
                raise ValueError("unfinished request journal requires reconciliation")
            if not final.is_file():
                continue
            plan = json.loads((root / "launch_plan.json").read_bytes())
            if plan.get("protocol") not in protocols or plan.get("model") != old.PILOT_MODEL:
                raise ValueError("unregistered ledger protocol or model")
            ledger = json.loads(final.read_bytes())
            count, calls = ledger.get("api_requests"), ledger.get("calls")
            if type(count) is not int or count < 0 or type(calls) is not list:
                raise ValueError("invalid root ledger")
            if count:
                extra.append(final.relative_to(runs).as_posix())
            elif calls or ledger.get("reserved_cny") != 0:
                raise ValueError("unexplained zero-request reservations")
    return old.reconcile_history(runs, reviewed_extra_ledgers=tuple(extra))
