"""Bounded independent-source BASE+S2G collection, then separate offline scoring.

默认只打印计划。采集不读取标签、不建卡；全部64題封存后使用 --score-only
单独评分。真实API须 --allow-network，仍共享累计50元、系列3元预算。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .pre_pilot import write_json
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_pilot import PILOT_MODEL, _git_state
from .run_s2g_author_pilot import ProgressLog
from .run_shared_s2g import (
    GENERATION_PROFILE,
    SHARED_OUTPUT_CAPS,
    SharedBudgetClient,
    reviewed_history,
)
from .s2g_author_api import S2GAuthorAPI
from .source_collection_core import (
    PLAN_SHA,
    PREFIX,
    PROTOCOL,
    SERIES_CAP_CNY,
    batch_identity,
    collect_batch,
    load_source_questions,
    load_source_runtime,
    recheck_exposure,
    score_collection,
)

KEY_VARIABLE = "GROWRAG_MEMORY_SOURCE_API_KEY"


def check_no_replay(runs, ids):
    for path in [
        *Path(runs).glob(f"{PREFIX}*/launch_plan.json"),
        *Path(runs).glob(f"{PREFIX}*.claim.json"),
    ]:
        old = json.loads(path.read_bytes())
        if old.get("protocol") != PROTOCOL or old.get("manifest_sha256") != PLAN_SHA:
            raise ValueError("source series protocol/manifest changed")
        existing = old.get("question_ids")
        if not isinstance(existing, list) or not existing or len(existing) != len(set(existing)):
            raise ValueError("unreadable source claim needs offline audit")
        if set(ids) & set(existing):
            raise ValueError("source already claimed; no automatic replay")


def series_reserved(runs):
    runs, total = Path(runs), 0.0
    for claim in runs.glob(f"{PREFIX}*.claim.json"):
        if not (runs / claim.name.removesuffix(".claim.json") / "final_budget.json").is_file():
            raise ValueError("unfinished source claim needs offline reconciliation")
    for path in runs.glob(f"{PREFIX}*/launch_plan.json"):
        final = path.parent / "final_budget.json"
        if not final.is_file():
            raise ValueError("unfinished source run needs offline reconciliation")
        launch, ledger = json.loads(path.read_bytes()), json.loads(final.read_bytes())
        if launch.get("protocol") != PROTOCOL or launch.get("model") != PILOT_MODEL:
            raise ValueError("unreviewed source budget")
        value = ledger.get("reserved_cny")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("invalid source reservation; missing is not free")
        total += value
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("manifest", "corpus-manifest", "index-path", "upstream", "runs-root"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--api-config", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--allow-network", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    parser.add_argument("--raw-train", type=Path)
    parser.add_argument("--score-output", type=Path)
    args = parser.parse_args(argv)
    questions, source = load_source_questions(args.manifest)
    runs = args.runs_root.resolve()
    recheck_exposure(runs, source)
    if args.score_only:
        if args.raw_train is None or args.score_output is None:
            raise ValueError("offline scoring needs --raw-train and a fresh --score-output")
        runtime = load_source_runtime(args.manifest, args.corpus_manifest, args.index_path)
        try:
            reports = score_collection(
                runs, args.manifest, runtime, args.raw_train, args.score_output
            )
            print(
                json.dumps(
                    {
                        "scored_sources": len(reports),
                        "memory_cards_created": 0,
                        "network_used": False,
                    }
                )
            )
        finally:
            runtime.close()
        return 0
    run_id = batch_identity(args.start, args.count)
    output = runs / run_id
    if output.exists() or (runs / f"{run_id}.claim.json").exists():
        raise FileExistsError("preserve existing source run; no automatic replay")
    ids = [q.question_id for q in questions[args.start : args.start + args.count]]
    check_no_replay(runs, ids)
    history, prior_series = reviewed_history(runs), series_reserved(runs)
    subcap = min(SERIES_CAP_CNY - prior_series, 50.0 - history["prior_reserved_cny"])
    if subcap <= 0:
        raise ValueError("project or source series budget exhausted")
    # Resolve from cwd: this program is executed only from the project root.
    project = Path.cwd().resolve()
    snapshot, git = source_snapshot(project), _git_state()
    probe = S2GAuthorAPI(
        args.upstream, None, lambda q, k: (), gap_profile="paper_k1", remove_repeat_docs=True
    )
    author = probe.provenance
    author["backend_generation_settings"] = GENERATION_PROFILE
    plan = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "manifest_sha256": PLAN_SHA,
        "manifest_path": str(args.manifest.resolve()),
        "question_ids": ids,
        "start": args.start,
        "count": args.count,
        "created_utc": datetime.now(UTC).isoformat(),
        "git": git,
        "source_sha256": snapshot["sha256"],
        "author": author,
        "model": PILOT_MODEL,
        "generation_profile": GENERATION_PROFILE,
        "backend_output_caps": SHARED_OUTPUT_CAPS,
        "temperature": 0,
        "top_p": 1,
        "enable_thinking": False,
        "max_retrieval_rounds": 4,
        "top_docs": 6,
        "gap_profile": "paper_k1",
        "index_path": str(args.index_path.resolve()),
        "corpus_manifest_path": str(args.corpus_manifest.resolve()),
        "historical_budget": history,
        "project_cap_cny": 50,
        "series_cap_cny": SERIES_CAP_CNY,
        "series_prior_reserved_cny": prior_series,
        "subcap_cny": subcap,
        "max_calls": 11 * args.count,
        "timeout_seconds": 1800,
        "price_checked_date": "2026-09-27",
        "price_source": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
        "price_input_cny_per_million": 0.2,
        "price_output_cny_per_million": 0.8,
        "no_training_no_memory_updates": True,
        "official_dev_test_used": False,
        "gold_policy": "none during collection; globally seal all64 before one offline label scan",
        "notice": "Independent source trajectory collection, "
        "not a learned memory bank or test evaluation.",
    }
    if not args.allow_network:
        print(
            json.dumps(
                {k: v for k, v in plan.items() if k != "historical_budget"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not args.api_config or not git.get("commit") or git["worktree_dirty"]:
        raise ValueError("live collection requires config and a clean committed checkout")
    if Path(__file__).resolve().parent != project / "src" / "growrag" / "experiments":
        raise ValueError(
            "ignored draft is not authorized for live requests; integrate and test first"
        )
    runtime = load_source_runtime(args.manifest, args.corpus_manifest, args.index_path)
    previous, client, failed = os.environ.get(KEY_VARIABLE), None, False
    try:
        chosen = runtime.questions[args.start : args.start + args.count]
        if [q.question_id for q in chosen] != ids:
            raise ValueError("source runtime order differs from frozen plan")
        settings = read_local_bailian_settings(args.api_config)
        host = urlsplit(settings.base_url).hostname or ""
        if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
            raise ValueError("ordinary Beijing endpoint required")
        if source_snapshot(project)["sha256"] != snapshot["sha256"]:
            raise ValueError("code changed before source launch")
        recheck_exposure(runs, source)
        plan["runtime_metadata"] = runtime.metadata
        runs.mkdir(parents=True, exist_ok=True)
        with (runs / f"{run_id}.claim.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "protocol": PROTOCOL,
                    "manifest_sha256": PLAN_SHA,
                    "question_ids": ids,
                    "plan_sha256": fingerprint(plan),
                    "no_retry": True,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "launch_plan.json", plan)
        write_json(output / "source_snapshot.json", snapshot)
        write_json(
            output / "process.json",
            {"pid": os.getpid(), "argv": sys.argv, "started_utc": datetime.now(UTC).isoformat()},
        )
        progress = ProgressLog(output)
        progress({"kind": "launch", "pid": os.getpid(), "model": PILOT_MODEL})
        os.environ[KEY_VARIABLE] = settings.api_key
        client = SharedBudgetClient(
            LiveChatClient(
                ChatConfig(
                    settings.base_url,
                    PILOT_MODEL,
                    KEY_VARIABLE,
                    max_calls=11 * args.count,
                    max_output_tokens=256,
                    output_limit_parameter="max_tokens",
                    enable_thinking=False,
                    temperature=0,
                    top_p=1,
                    allow_s2g_answer_prefix_on_length=True,
                ),
                output / "api_audit",
                allow_network=True,
            ),
            PriceLimits(budget_cny=subcap, max_elapsed_seconds=1800, max_prompt_bytes=30000),
            output / "request_journal",
        )
        client.schema_stages, client.qwen_output_caps = True, True
        collect_batch(
            chosen,
            runtime,
            client,
            args.upstream,
            output,
            progress,
            run_id=run_id,
            start=args.start,
        )
        return int(bool(client.block_reason))
    except Exception as error:
        failed = True
        if client is None:
            raise
        client.block_reason = client.block_reason or "source_runner_failure"
        write_json(
            output / "interrupted.json", {"error_type": type(error).__name__, "no_retry": True}
        )
        progress({"kind": "interrupted", "error_type": type(error).__name__})
        return 1
    finally:
        runtime.close()
        if client is not None:
            write_json(output / "final_budget.json", client.report())
            write_json(
                output / "cumulative_budget.json",
                {
                    "prior_reserved_cny": history["prior_reserved_cny"],
                    "new_reserved_cny": client.reserved_cny,
                    "cumulative_reserved_cny": history["prior_reserved_cny"] + client.reserved_cny,
                    "prior_unknown_cost_requests": history["prior_unknown_cost_requests"],
                    "project_cap_cny": 50,
                },
            )
            progress(
                {
                    "kind": "exit",
                    "requests": client.attempts,
                    "status": "failed" if failed or client.block_reason else "completed",
                }
            )
        if previous is None:
            os.environ.pop(KEY_VARIABLE, None)
        else:
            os.environ[KEY_VARIABLE] = previous


if __name__ == "__main__":
    raise SystemExit(main())
