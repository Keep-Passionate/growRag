"""Catalog-v3 ledgers inherit every earlier expense, including quote-v2 A1.

The old A1 budget module is intentionally unchanged. Its shared date prefix also
covers this new series, so this scanner explicitly registers both protocols.
No failed-request or unused-reservation credits are taken.
"""

import json
from pathlib import Path

from . import a0_v2_budget as old
from .a1_budget import PREFIX
from .a1_budget import PROTOCOL as QUOTE_PROTOCOL
from .prior_budget import reconcile_history

PROTOCOL = "growrag-a1-catalog-observer-v1"
A1_PROTOCOLS = frozenset({QUOTE_PROTOCOL, PROTOCOL})


def reviewed_history(runs: Path) -> dict:
    """Reconcile the complete project ledger without resetting the project cap."""
    runs = Path(runs).resolve(strict=True)
    series = {
        old.SHARED_PREFIX: frozenset(old.REVIEWED_PROTOCOLS),
        **{p: frozenset(v) for p, v in old.REVIEWED_OTHER_SERIES.items()},
        old.HISTORY_PREFIX: old.REVIEWED_HISTORY_PROTOCOLS,
        old.CANDIDATE_PREFIX: old.CANDIDATE_PROTOCOLS,
        old.OLD_A0_PREFIX: frozenset({old.OLD_A0_PROTOCOL}),
        old.PREFIX: frozenset({old.PROTOCOL}),
        PREFIX: A1_PROTOCOLS,
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
    return reconcile_history(runs, reviewed_extra_ledgers=tuple(extra))
