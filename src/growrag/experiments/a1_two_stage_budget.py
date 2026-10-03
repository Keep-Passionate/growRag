"""Register two-stage A1 expenses without changing any frozen legacy registry.

Every root ledger is reconciled together, including failed quote/catalog calls.
Unknown historical charges retain their reservations. Historical authorization
metadata is preserved; this module cannot grant a higher project spending cap.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import a1_catalog_budget as catalog
from .prior_budget import reconcile_history

PREFIX = "2026-10-03_a1ts_"
PROTOCOL = "growrag-a1-two-stage-observer-v1"


def reviewed_history(runs: Path, *, reviewed_other_series=None) -> dict:
    """Review all registered roots once, not cumulative copies or refunded costs.

    Call under the project serial lock before paid work. New unknown charges or
    unresolved journals still require the runner to stop; known legacy unknowns
    remain in the same conservative project total, never a fresh allocation.

    Optional new registrations are explicit caller-reviewed names, not a grant
    of spending authority. They cannot replace, overlap or mutate the legacy
    registry; all new roots still require the exact frozen pilot model.
    """
    runs = Path(runs).resolve(strict=True)
    old = catalog.old
    series = {
        old.SHARED_PREFIX: frozenset(old.REVIEWED_PROTOCOLS),
        **{prefix: frozenset(protocols) for prefix, protocols in old.REVIEWED_OTHER_SERIES.items()},
        old.HISTORY_PREFIX: old.REVIEWED_HISTORY_PROTOCOLS,
        old.CANDIDATE_PREFIX: old.CANDIDATE_PROTOCOLS,
        old.OLD_A0_PREFIX: frozenset({old.OLD_A0_PROTOCOL}),
        old.PREFIX: frozenset({old.PROTOCOL}),
        catalog.PREFIX: catalog.A1_PROTOCOLS,
        PREFIX: frozenset({PROTOCOL}),
    }
    if reviewed_other_series is not None:
        if type(reviewed_other_series) is not dict:
            raise ValueError("reviewed series must be an explicit mapping")
        known_protocols = set().union(*series.values())
        for prefix, protocols in sorted(
            reviewed_other_series.items(), key=lambda item: str(item[0])
        ):
            if (
                type(prefix) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*_", prefix) is None
            ):
                raise ValueError("invalid reviewed series prefix")
            if any(prefix.startswith(name) or name.startswith(prefix) for name in series):
                raise ValueError("reviewed series prefix overlaps registered roots")
            if type(protocols) not in (set, frozenset, tuple, list) or not protocols:
                raise ValueError("reviewed protocols must be an explicit nonempty collection")
            if any(
                type(protocol) is not str
                or re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", protocol) is None
                for protocol in protocols
            ):
                raise ValueError("invalid reviewed protocol name")
            registered = frozenset(protocols)
            if len(registered) != len(protocols) or registered & known_protocols:
                raise ValueError("reviewed protocols duplicate or overlap registered protocols")
            series[prefix] = registered
            known_protocols.update(registered)
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
                continue  # Unaccounted live audits are rejected by the shared reconciler.
            launch = root / "launch_plan.json"
            if any(not path.resolve(strict=True).is_relative_to(runs) for path in (final, launch)):
                raise ValueError("ledger or launch plan escapes runs")
            plan = json.loads(launch.read_bytes())
            if plan.get("protocol") not in protocols or plan.get("model") != old.PILOT_MODEL:
                raise ValueError("unregistered ledger protocol or model")
            ledger = json.loads(final.read_bytes())
            if ledger.get("reconciliation_required"):
                raise ValueError("unresolved root ledger requires reconciliation")
            count, calls = ledger.get("api_requests"), ledger.get("calls")
            if type(count) is not int or count < 0 or type(calls) is not list:
                raise ValueError("invalid root ledger")
            if count:
                extra.append(final.relative_to(runs).as_posix())
            elif calls:
                raise ValueError("zero-request ledger has pending calls")
            else:
                reserved = ledger.get("reserved_cny")
                if type(reserved) not in (int, float) or reserved != 0:
                    raise ValueError("zero-request ledger has unexplained reservations")
    return reconcile_history(runs, reviewed_extra_ledgers=tuple(extra))
