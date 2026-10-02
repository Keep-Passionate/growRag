"""A0账本登记：继承全部旧调用，只新增本系列明确注册的根账本。"""

from __future__ import annotations

import json
from pathlib import Path

from .history_budget import (
    CANDIDATE_PREFIX,
    CANDIDATE_PROTOCOLS,
    REVIEWED_HISTORY_PROTOCOLS,
)
from .history_budget import (
    PREFIX as HISTORY_PREFIX,
)
from .prior_budget import reconcile_history
from .run_shared_s2g import (
    HISTORICAL_ROOTS,
    PILOT_MODEL,
    REVIEWED_OTHER_SERIES,
    REVIEWED_PROTOCOLS,
)
from .run_shared_s2g import (
    PREFIX as SHARED_PREFIX,
)

PREFIX = "2026-10-02_a0_"
PROTOCOL = "growrag-a0-query-construction-v1"


def reviewed_history(runs: Path) -> dict:
    """核对原始用量和未知费用，不把已有保守预留退回可用额度。"""
    runs = Path(runs).resolve(strict=True)
    series = {
        SHARED_PREFIX: frozenset(REVIEWED_PROTOCOLS),
        **{p: frozenset(v) for p, v in REVIEWED_OTHER_SERIES.items()},
        HISTORY_PREFIX: REVIEWED_HISTORY_PROTOCOLS,
        CANDIDATE_PREFIX: CANDIDATE_PROTOCOLS,
        PREFIX: frozenset({PROTOCOL}),
    }
    extra = list(HISTORICAL_ROOTS)
    for prefix, protocols in series.items():
        for root in sorted(runs.glob(f"{prefix}*")):
            if not root.is_dir():
                continue
            if not root.resolve(strict=True).is_relative_to(runs):
                raise ValueError("ledger root escapes project runs")
            final, journal = root / "final_budget.json", root / "request_journal"
            if journal.is_dir() and any(journal.iterdir()) and not final.is_file():
                raise ValueError("unfinished request journal requires reconciliation")
            if not final.is_file():
                continue
            plan = json.loads((root / "launch_plan.json").read_bytes())
            if plan.get("protocol") not in protocols or plan.get("model") != PILOT_MODEL:
                raise ValueError("unregistered ledger protocol or model")
            ledger = json.loads(final.read_bytes())
            count, calls = ledger.get("api_requests"), ledger.get("calls")
            if type(count) is not int or count < 0 or type(calls) is not list:
                raise ValueError("invalid root ledger")
            if count:
                extra.append(final.relative_to(runs).as_posix())
            elif calls or ledger.get("reserved_cny") != 0:
                raise ValueError("zero-request ledger has unexplained reservations")
    return reconcile_history(runs, reviewed_extra_ledgers=tuple(extra))
