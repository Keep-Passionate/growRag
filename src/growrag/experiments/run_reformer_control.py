"""Paired ReFormeR control: one selected ID, then with/without canonical examples.

中文：每题只选一次模式，让两条分支使用同一规则；仅改变是否提供原库示例。
偶数题先运行带示例分支，奇数题顺序反转。全批预测封存后才加载答案评分。
默认只打印计划，不读取密钥、不联网；真实请求一旦失败就停止，不自动重试。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from urllib.parse import urlsplit

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_benchmark import call_totals
from .pre_pilot import write_json
from .reformer_control import ReFormeRControlAPI
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_pilot import PILOT_MODEL, _git_state
from .run_reformer_pilot import FROZEN_MANIFEST_SHA, frozen_ids
from .run_reformer_pilot import batch_identity as baseline_batch_identity
from .run_s2g_author_pilot import ProgressLog
from .run_shared_s2g import SharedBudgetClient, reviewed_history

PROTOCOL = "growrag-reformer-id-examples-v1"
PREFIX = "2026-09-27_reformer_control_v1_"
ARMS = ("WITH_EXAMPLES", "WITHOUT_EXAMPLES")
KEY_VARIABLE = "GROWRAG_REFORMER_CONTROL_API_KEY"
GENERATION_PROFILE = "reformer_id512_plain512_s2g_answer1024_temperature0_top_p1"
SERIES_CAP_CNY = 3.0


def batch_identity(start, count):
    """Reuse the reviewed first-100 bounds, but never its claims or output roots."""
    baseline_batch_identity(start, count)
    return f"{PREFIX}{start:04d}_{start + count:04d}"


def arm_order(offset):
    return list(ARMS if offset % 2 == 0 else reversed(ARMS))


def check_no_replay(runs, manifest_sha, question_ids):
    wanted = set(question_ids)
    for path in [
        *Path(runs).glob(f"{PREFIX}*/launch_plan.json"),
        *Path(runs).glob(f"{PREFIX}*.claim.json"),
    ]:
        old = json.loads(path.read_bytes())
        if old.get("protocol") != PROTOCOL or old.get("manifest_sha256") != manifest_sha:
            raise ValueError("series cannot silently change protocol or dataset")
        ids = old.get("question_ids")
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError("unreadable prior claim requires offline audit")
        if wanted & set(ids):
            raise ValueError("question already claimed; no automatic replay")


def series_reserved(runs):
    """An unfinished request might be billed: never infer that it was free."""
    runs, total = Path(runs), 0.0
    for claim in runs.glob(f"{PREFIX}*.claim.json"):
        if not (runs / claim.name.removesuffix(".claim.json") / "final_budget.json").is_file():
            raise ValueError("unfinished ReFormeR control claim needs offline reconciliation")
    for path in runs.glob(f"{PREFIX}*/launch_plan.json"):
        final = path.parent / "final_budget.json"
        if not final.is_file():
            raise ValueError("unfinished ReFormeR control run needs offline reconciliation")
        launch, ledger = json.loads(path.read_bytes()), json.loads(final.read_bytes())
        if launch.get("protocol") != PROTOCOL or launch.get("model") != PILOT_MODEL:
            raise ValueError("unreviewed ReFormeR control series budget")
        value = ledger.get("reserved_cny")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("series reservation must be an explicit numeric amount")
        if not math.isfinite(value) or value < 0:
            raise ValueError("series reservation must be finite and nonnegative")
        total += value
    return total


def _not_started():
    return {"status": "not_started", "calls": [], "elapsed_seconds": 0.0, "feedback": None}


def execute_question(
    question, index, client, reformer, reader, directory, progress, *, run_id, offset
):
    """Prediction-only interface; gold, cached baselines and feedback cannot enter."""
    directory.mkdir(parents=True, exist_ok=False)
    adapter = None

    def execute_stage(stage, action):
        progress.arm = stage
        progress({"kind": "arm_start", "question_id": question.question_id})
        started, first_call = perf_counter(), len(client.calls)
        try:
            result = deepcopy(action())
            # Selection is a hash-sealed adapter input: never rename or append
            # its fields before handing the identical snapshot to both arms.
            if stage != "SELECTION" and "question_id" in result:
                result["execution_trace_id"] = result["question_id"]
                result["question_id"] = question.question_id
            row = {"status": "completed", "result": result, "feedback": None}
        except Exception as error:
            client.block_reason = client.block_reason or "reformer_control_component_failure"
            row = {
                "status": "failed",
                "error_type": type(error).__name__,
                "events": deepcopy(adapter.events) if adapter is not None else [],
                "feedback": None,
            }
            progress({"kind": "arm_failed", "error_type": type(error).__name__})
        row.update(
            calls=deepcopy(client.calls[first_call:]), elapsed_seconds=perf_counter() - started
        )
        write_json(directory / f"{stage}_execution.json", row)
        return row

    def select():
        nonlocal adapter
        adapter = ReFormeRControlAPI(reformer, reader, client, index, event_callback=progress)
        return adapter.select(question.text, f"{run_id}/{question.question_id}/SELECTION")

    selection = execute_stage("SELECTION", select)
    order, arms = arm_order(offset), {arm: _not_started() for arm in ARMS}
    for arm in order:
        if client.block_reason or selection["status"] != "completed":
            break
        arms[arm] = execute_stage(
            arm,
            lambda arm=arm: adapter.run_selected(
                question.text,
                f"{run_id}/{question.question_id}/{arm}",
                deepcopy(selection["result"]),
                include_examples=arm == "WITH_EXAMPLES",
            ),
        )
    return {
        "question_id": question.question_id,
        "question": question.text,
        "offset": offset,
        "arm_order": order,
        "selection": selection,
        "arms": arms,
        "scoring_status": "pending"
        if any(r["status"] == "completed" for r in arms.values())
        else "not_scored",
    }


def summarize(reports, calls, *, run_id, planned, stop_reason, started):
    arm_summaries = {}
    for arm in ARMS:
        completed = [r["arms"][arm] for r in reports if r["arms"][arm]["status"] == "completed"]
        scored = [r for r in completed if (r.get("feedback") or {}).get("answer_em") is not None]
        arm_summaries[arm] = {
            "completed": len(completed),
            "scored": len(scored),
            "em": mean(r["feedback"]["answer_em"] for r in scored) if scored else None,
            "f1": mean(r["feedback"]["answer_f1"] for r in scored) if scored else None,
        }
    return {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "planned": planned,
        "started": started,
        "started_without_report": started - len(reports),
        "completed_pairs": sum(
            all(r["arms"][a]["status"] == "completed" for a in ARMS) for r in reports
        ),
        "scored_pairs": sum(
            all(r["arms"][a].get("feedback") is not None for a in ARMS) for r in reports
        ),
        "arms": arm_summaries,
        "stop_reason": stop_reason,
        "all_actual_calls_including_interrupted": call_totals(calls),
        "benchmark_reproduction": False,
        "notice": "One shared ID selection, two rewrites and two answers; "
        "examples are the paired intervention. Each deployable path must include "
        "the FULL selector cost, although it was executed once here. "
        "Compared with the old public-code run, the ID-only selector is also changed; "
        "not a pure ID ablation.",
    }


def execute_batch(
    questions,
    runtime,
    client,
    reformer,
    reader,
    manifest_path,
    manifest_sha,
    output,
    progress,
    *,
    run_id,
    start,
):
    """Seal every prediction before feedback; preserve any one-arm partial success."""
    from .shared_s2g_corpus import load_gold_after_execution, score_result

    reports, started = [], 0
    try:
        for offset, question in enumerate(questions, start):
            if client.block_reason:
                break
            started += 1
            report = execute_question(
                question,
                runtime.index,
                client,
                reformer,
                reader,
                output / "questions" / f"{offset:04d}",
                progress,
                run_id=run_id,
                offset=offset,
            )
            reports.append(report)
            write_json(output / "questions" / f"{offset:04d}" / "prediction_report.json", report)
            progress(
                {
                    "kind": "question_complete",
                    "question_id": question.question_id,
                    "status": "completed"
                    if all(r["status"] == "completed" for r in report["arms"].values())
                    else "partial_or_failed",
                    "requests": client.attempts,
                }
            )
        write_json(
            output / "predictions_frozen.json",
            {
                "run_id": run_id,
                "completed_question_ids": [
                    r["question_id"]
                    for r in reports
                    if all(r["arms"][a]["status"] == "completed" for a in ARMS)
                ],
                "completed_arm_ids": [
                    {"question_id": r["question_id"], "arm": a}
                    for r in reports
                    for a in ARMS
                    if r["arms"][a]["status"] == "completed"
                ],
                "reports_sha256_before_scoring": fingerprint(reports),
                "created_utc": datetime.now(UTC).isoformat(),
            },
        )
        for report in reports:
            complete = [arm for arm in ARMS if report["arms"][arm]["status"] == "completed"]
            if complete:
                try:
                    gold = load_gold_after_execution(
                        manifest_path,
                        completed_question_ids=[report["question_id"]],
                        expected_manifest_sha256=manifest_sha,
                    )[report["question_id"]]
                    # Stage all feedback first: scoring cannot leave half-committed labels.
                    feedback = {
                        arm: score_result(
                            report["arms"][arm]["result"], gold, runtime.index, retained_mode="raw"
                        )
                        for arm in complete
                    }
                    initial = score_result(
                        {
                            **report["arms"][complete[0]]["result"],
                            "retrieved_documents": report["selection"]["result"][
                                "initial_selector_documents"
                            ],
                        },
                        gold,
                        runtime.index,
                        retained_mode="raw",
                    )
                    report["selection"]["initial_feedback"] = {
                        key: initial.get(key)
                        for key in (
                            "raw_support_recall",
                            "raw_support_hits",
                            "gold_support_count",
                            "coverage_notice",
                        )
                    }
                    report["selection"]["initial_feedback"].update(
                        stage="selector_initial_top3",
                        document_count=len(
                            report["selection"]["result"]["initial_selector_documents"]
                        ),
                    )
                    for arm in complete:
                        report["arms"][arm]["feedback"] = feedback[arm]
                    report["offline_gold_answers"] = list(gold.answers)
                    report["scoring_status"] = (
                        "completed" if len(complete) == len(ARMS) else "partial_completed"
                    )
                except Exception as error:
                    report["scoring_status"] = "failed"
                    report["scoring_error_type"] = type(error).__name__
                    client.block_reason = client.block_reason or "offline_scoring_failure"
            write_json(output / "questions" / f"{report['offset']:04d}" / "report.json", report)
    except Exception:
        client.block_reason = client.block_reason or "execution_or_scoring_failure"
        raise
    finally:
        write_json(output / "reports.json", reports)
        summary = summarize(
            reports,
            client.calls,
            run_id=run_id,
            planned=len(questions),
            stop_reason=client.block_reason,
            started=started,
        )
        write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "reformer-snapshot", "reader-snapshot", "index-path", "runs-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", default=FROZEN_MANIFEST_SHA)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    run_id = batch_identity(args.start, args.count)
    runs, output = args.runs_root.resolve(), args.runs_root.resolve() / run_id
    if output.exists() or (runs / f"{run_id}.claim.json").exists():
        raise FileExistsError("batch already exists; preserve previous work")
    _, ids = frozen_ids(args.manifest, args.expected_manifest_sha256, args.start, args.count)
    check_no_replay(runs, args.expected_manifest_sha256, ids)
    history, series_prior = reviewed_history(runs), series_reserved(runs)
    subcap = min(SERIES_CAP_CNY - series_prior, 50.0 - history["prior_reserved_cny"])
    if subcap <= 0:
        raise ValueError("project or ReFormeR control series budget exhausted")
    project = Path(__file__).resolve().parents[3]
    snapshot, git = source_snapshot(project), _git_state()
    probe = ReFormeRControlAPI(args.reformer_snapshot, args.reader_snapshot, None, None)
    plan = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "arms": list(ARMS),
        "created_utc": datetime.now(UTC).isoformat(),
        "git": git,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": args.expected_manifest_sha256,
        "question_ids": ids,
        "start": args.start,
        "count": args.count,
        "cohort_policy": "first100_frozen_shared500_ids_not_selected_by_outcomes",
        "arm_order_policy": "even_offset_WITH_EXAMPLES_first_odd_offset_WITHOUT_EXAMPLES_first",
        "shared_selection": True,
        "deployable_cost_policy": "full_selection_plus_one_arm",
        "model": PILOT_MODEL,
        "generation_profile": GENERATION_PROFILE,
        "selector_and_rewrite_output_tokens": 512,
        "answer_output_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "enable_thinking": False,
        "initial_top_docs": 3,
        "answer_top_docs": 6,
        "retrieval_rounds_per_deployable_path": 2,
        "author": probe.provenance,
        "source_sha256": snapshot["sha256"],
        "index_path": str(args.index_path.resolve()),
        "subcap_cny": subcap,
        "project_cap_cny": 50,
        "max_calls": 5 * args.count,
        "series_cap_cny": SERIES_CAP_CNY,
        "series_prior_reserved_cny": series_prior,
        "timeout_seconds": 1800,
        "historical_budget": history,
        "price_checked_date": "2026-09-27",
        "price_source": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
        "price_input_cny_per_million": 0.2,
        "price_output_cny_per_million": 0.8,
        "no_training_no_memory_updates": True,
        "official_dev_test_used": False,
        "baseline_results_access": "offline analysis only after batch prediction seal",
        "failure_policy": "First API/protocol failure stops predictions; "
        "no retries or automatic replay.",
        "notice": "Controlled examples contrast under one ID-only selection. "
        "Not paper-trained selection or dynamic memory; "
        "old-run comparisons also change selector prompt.",
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
    if not args.api_config or not git["commit"] or git["worktree_dirty"]:
        raise ValueError("live launch needs config path and a clean committed checkout")
    from .shared_s2g_corpus import load_runtime

    runtime = load_runtime(
        args.manifest,
        index_path=args.index_path,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    previous, client, failed = os.environ.get(KEY_VARIABLE), None, False
    try:
        questions = runtime.questions[args.start : args.start + args.count]
        if [q.question_id for q in questions] != ids:
            raise ValueError("runtime order differs from frozen plan")
        settings = read_local_bailian_settings(args.api_config)
        host = urlsplit(settings.base_url).hostname or ""
        if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
            raise ValueError("ordinary Beijing endpoint required for frozen price")
        if source_snapshot(project)["sha256"] != snapshot["sha256"]:
            raise ValueError("source changed before live launch")
        plan["runtime_metadata"] = runtime.metadata
        runs.mkdir(parents=True, exist_ok=True)
        with (runs / f"{run_id}.claim.json").open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "protocol": PROTOCOL,
                    "manifest_sha256": args.expected_manifest_sha256,
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
                    max_calls=5 * args.count,
                    max_output_tokens=512,
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
        execute_batch(
            questions,
            runtime,
            client,
            args.reformer_snapshot,
            args.reader_snapshot,
            args.manifest,
            args.expected_manifest_sha256,
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
        client.block_reason = client.block_reason or "runner_failure"
        write_json(
            output / "interrupted.json", {"error_type": type(error).__name__, "no_retry": True}
        )
        progress({"kind": "interrupted", "error_type": type(error).__name__})
        return 1
    finally:
        runtime.index.close()
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
