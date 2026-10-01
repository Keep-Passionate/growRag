"""独立历史基底实验仍沿用项目累计账本；不改旧实验的协议注册表。

只收集每次运行的根final_budget，不重复相加cumulative副本。所有根一起交给
旧核账器核对API审计、重复身份和未知费用预留；不是重新授予一笔实验额度。
调用方须在项目串行锁内执行，并用另行确认的累计上限减prior_reserved_cny。
本模块不读密钥、题目数据集或gold，不调用API，也不修改任何文件。
API审计沿用旧核账器，仅使用请求身份、状态和usage字段。
"""

from __future__ import annotations

import json
from pathlib import Path

from .prior_budget import reconcile_history
from .run_shared_s2g import (
    HISTORICAL_ROOTS,
    PILOT_MODEL,
    REVIEWED_OTHER_SERIES,
    REVIEWED_PROTOCOLS,
)
from .run_shared_s2g import PREFIX as SHARED_PREFIX

PREFIX = "2026-10-01_history_base_"
PROTOCOL = "growrag-history-calibration-v1"
REVIEWED_HISTORY_PROTOCOLS = frozenset({PROTOCOL, "growrag-history-calibration-v2"})


def reviewed_history(runs: Path) -> dict:
    """一次核账覆盖旧项目与新history运行；未知/未终结请求不能视为免费。

    不先调用旧reviewed_history再相加：旧函数会在扫描audit时拒绝新协议。
    新字典只复制既有规则，不原地扩写旧全局注册表。返回的历史授权信息原样
    保留；它不能代替本次runner必须显式记录的project_cap_cny授权。
    """
    runs = Path(runs).resolve(strict=True)
    series = {
        SHARED_PREFIX: frozenset(REVIEWED_PROTOCOLS),
        **{prefix: frozenset(protocols) for prefix, protocols in REVIEWED_OTHER_SERIES.items()},
        PREFIX: REVIEWED_HISTORY_PROTOCOLS,
    }
    extra = list(HISTORICAL_ROOTS)
    for prefix, protocols in series.items():
        for directory in sorted(runs.glob(f"{prefix}*")):
            if not directory.is_dir():
                continue
            if not directory.resolve(strict=True).is_relative_to(runs):
                raise ValueError("historical run must remain inside runs")
            final = directory / "final_budget.json"
            journal = directory / "request_journal"
            if journal.is_dir() and any(journal.iterdir()) and not final.is_file():
                raise ValueError("unfinished request journal needs offline budget reconciliation")
            if not final.is_file():
                continue  # A live audit without a root ledger is rejected by reconcile_history.
            launch = json.loads((directory / "launch_plan.json").read_bytes())
            if launch.get("protocol") not in protocols or launch.get("model") != PILOT_MODEL:
                raise ValueError("unreviewed historical protocol/model")
            ledger = json.loads(final.read_bytes())
            requests, calls = ledger.get("api_requests"), ledger.get("calls")
            if type(requests) is not int or requests < 0 or type(calls) is not list:
                raise ValueError("invalid root request accounting")
            if requests:
                extra.append(final.relative_to(runs).as_posix())
            elif calls:
                raise ValueError("zero-request run has uncertain pending calls")
            else:
                reserved = ledger.get("reserved_cny")
                if type(reserved) not in (int, float) or reserved != 0:
                    raise ValueError("zero-request run has an unexplained reservation")
    # The shared reconciler rejects duplicate roots/traces/audits and retains all
    # reservations for failed requests with unknown usage; never replace None by 0.
    return reconcile_history(runs, reviewed_extra_ledgers=tuple(extra))
