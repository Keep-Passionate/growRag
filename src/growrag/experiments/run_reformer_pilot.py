"""Bounded, auditable ReFormeR public-code migration on frozen Hotpot development IDs.

中文：先检索3篇给模式选择器，再按规则改写，最后检索6篇给同一个S2G回答器。
本模块不训练、不写历史记忆；全部预测结束并落盘之后，才读取gold进行离线评分。
默认仅打印计划，不读取密钥、不联网。旧失败保留，禁止自动重试或覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
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
from .reformer_api import ReFormeRAPI
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_pilot import PILOT_MODEL, _git_state
from .run_s2g_author_pilot import ProgressLog
from .run_shared_s2g import SharedBudgetClient, reviewed_history

PROTOCOL = "growrag-reformer-public-qwen-v1"
PREFIX = "2026-09-27_reformer_hotpot_v1_"
ARM = "REFORMER_PUBLIC_QWEN"
FROZEN_MANIFEST_SHA = "2fb117c03b64be3ab12398a297c82ce3315600c14e33f5dcd56c5c6c5c59e064"
COHORT_SIZE = 100
MAX_BATCH = 25
KEY_VARIABLE = "GROWRAG_REFORMER_PILOT_API_KEY"
GENERATION_PROFILE = "reformer_plain512_s2g_answer1024_temperature0_top_p1"
SERIES_CAP_CNY = 3.0


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def batch_identity(start, count):
    """Only the first 100 frozen IDs, not a score-selected sample, are authorized."""
    if (
        type(start) is not int
        or type(count) is not int
        or start < 0
        or not 1 <= count <= MAX_BATCH
        or start + count > COHORT_SIZE
    ):
        raise ValueError("batch must contain 1-25 questions within frozen offsets 0-99")
    return f"{PREFIX}{start:04d}_{start + count:04d}"


def frozen_ids(manifest_path, expected_sha, start, count):
    batch_identity(start, count)
    if expected_sha != FROZEN_MANIFEST_SHA or _sha(manifest_path) != expected_sha:
        raise ValueError("only the reviewed shared500 frozen manifest is authorized")
    manifest = json.loads(Path(manifest_path).read_bytes())
    if manifest.get("official_split") != "train" or manifest.get("role") != "development":
        raise ValueError("only official train development data may enter this pilot")
    all_ids = manifest.get("question_ids", [])
    if len(all_ids) != 500 or len(set(all_ids)) != 500:
        raise ValueError("expected exactly 500 unique frozen development IDs")
    return manifest, all_ids[start : start + count]


def check_no_replay(runs, manifest_sha, question_ids):
    """A prior plan or pre-launch claim consumes its IDs, even if it failed.

    Untouched continuation is deliberately not inferred from missing answers: a
    request may have been billed before interruption. It needs a separate audit.
    """
    wanted = set(question_ids)
    paths = [
        *Path(runs).glob(f"{PREFIX}*/launch_plan.json"),
        *Path(runs).glob(f"{PREFIX}*.claim.json"),
    ]
    for path in paths:
        old = json.loads(path.read_bytes())
        if old.get("protocol") != PROTOCOL or old.get("manifest_sha256") != manifest_sha:
            raise ValueError("series cannot silently change protocol or dataset")
        ids = old.get("question_ids")
        if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
            raise ValueError("unreadable prior claim requires offline audit")
        if wanted & set(ids):
            raise ValueError("question already claimed; no automatic replay")


def series_reserved(runs):
    """All batches share one 3-yuan series envelope; unfinished runs block launches."""
    runs, total = Path(runs), 0.0
    for claim in runs.glob(f"{PREFIX}*.claim.json"):
        directory = runs / claim.name.removesuffix(".claim.json")
        if not (directory / "final_budget.json").is_file():
            raise ValueError("unfinished ReFormeR claim needs offline budget reconciliation")
    for path in runs.glob(f"{PREFIX}*/launch_plan.json"):
        final = path.parent / "final_budget.json"
        if not final.is_file():
            raise ValueError("unfinished ReFormeR run needs offline budget reconciliation")
        launch, ledger = json.loads(path.read_bytes()), json.loads(final.read_bytes())
        if launch.get("protocol") != PROTOCOL or launch.get("model") != PILOT_MODEL:
            raise ValueError("unreviewed ReFormeR series budget")
        value = ledger.get("reserved_cny")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("series reservation must be an explicit numeric amount")
        if not math.isfinite(value) or value < 0:
            raise ValueError("series reservation must be finite and nonnegative")
        total += value
    return total


def execute_question(question, index, client, reformer, reader, directory, progress, *, run_id):
    """Prediction authority: no labels, question types, baseline outputs or scores."""
    directory.mkdir(parents=True, exist_ok=False)
    progress.arm = ARM
    progress({"kind": "arm_start", "question_id": question.question_id})
    started, first_call, adapter = perf_counter(), len(client.calls), None
    try:
        adapter = ReFormeRAPI(reformer, reader, client, index, event_callback=progress)
        result = adapter.run(question.text, f"{run_id}/{question.question_id}/{ARM}")
        result["execution_trace_id"] = result["question_id"]
        result["question_id"] = question.question_id
        row = {"status": "completed", "result": result, "feedback": None}
    except Exception as error:
        client.block_reason = client.block_reason or "reformer_component_failure"
        row = {
            "status": "failed",
            "error_type": type(error).__name__,
            "events": adapter.events if adapter is not None else [],
            "feedback": None,
        }
        progress({"kind": "arm_failed", "error_type": type(error).__name__})
    row.update(calls=list(client.calls[first_call:]), elapsed_seconds=perf_counter() - started)
    write_json(directory / f"{ARM}_execution.json", row)
    return row


def summarize(reports, calls, *, run_id, planned, stop_reason, started):
    completed = [r for r in reports if r["outcome"]["status"] == "completed"]
    scored = [
        r for r in completed if (r["outcome"].get("feedback") or {}).get("answer_em") is not None
    ]
    return {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "arm": ARM,
        "planned": planned,
        "started": started,
        "started_without_report": started - len(reports),
        "completed": len(completed),
        "scored": len(scored),
        "unscored_completed": len(completed) - len(scored),
        "stop_reason": stop_reason,
        "em": mean(r["outcome"]["feedback"]["answer_em"] for r in scored) if scored else None,
        "f1": mean(r["outcome"]["feedback"]["answer_f1"] for r in scored) if scored else None,
        "completed_retrieval_rounds": sum(
            r["outcome"]["result"]["retrieval_rounds"] for r in completed
        ),
        "all_actual_calls_including_interrupted": call_totals(calls),
        "benchmark_reproduction": False,
        "notice": "Qwen/Hotpot closed-corpus public-code migration, not paper TREC reproduction. "
        "Static released patterns and a prompted selector; no trained selector or dynamic memory. "
        "BASE1, ReFormeR2 and S2G4 use unequal budgets; contrasts are not memory attribution.",
    }


def execute_batch(
    questions, runtime, client, reformer, reader, manifest_path, manifest_sha, output, progress,
    *, run_id, start,
):
    """Freeze all batch predictions before any gold is loaded, then score offline."""
    from .shared_s2g_corpus import load_gold_after_execution, score_result

    reports, started = [], 0
    try:
        for offset, question in enumerate(questions, start):
            if client.block_reason:
                break
            started += 1
            directory = output / "questions" / f"{offset:04d}"
            outcome = execute_question(
                question, runtime.index, client, reformer, reader, directory, progress,
                run_id=run_id,
            )
            report = {
                "question_id": question.question_id,
                "question": question.text,
                "offset": offset,
                "outcome": outcome,
                "scoring_status": "pending" if outcome["status"] == "completed" else "not_scored",
            }
            reports.append(report)
            write_json(directory / "prediction_report.json", report)
            progress({
                "kind": "question_complete", "question_id": question.question_id,
                "status": outcome["status"], "requests": client.attempts,
            })
        # A batch-level runtime seal makes the prediction/feedback boundary visible.
        write_json(output / "predictions_frozen.json", {
            "run_id": run_id,
            "completed_question_ids": [
                r["question_id"] for r in reports if r["outcome"]["status"] == "completed"
            ],
            "reports_sha256_before_scoring": fingerprint(reports),
            "created_utc": datetime.now(UTC).isoformat(),
        })
        for report in reports:
            if report["outcome"]["status"] != "completed":
                write_json(output / "questions" / f"{report['offset']:04d}" / "report.json", report)
                continue
            try:
                gold = load_gold_after_execution(
                    manifest_path, completed_question_ids=[report["question_id"]],
                    expected_manifest_sha256=manifest_sha,
                )[report["question_id"]]
                feedback = score_result(
                    report["outcome"]["result"], gold, runtime.index, retained_mode="raw",
                )
                initial = score_result(
                    {
                        **report["outcome"]["result"],
                        "retrieved_documents": report["outcome"]["result"][
                            "initial_selector_documents"
                        ],
                    },
                    gold, runtime.index, retained_mode="raw",
                )
                # These three documents informed selection but not the answer.
                # Do not report answer accuracy as if it came from this context.
                report["outcome"]["initial_feedback"] = {
                    key: initial.get(key) for key in (
                        "raw_support_recall", "raw_support_hits", "gold_support_count",
                        "coverage_notice",
                    )
                }
                report["outcome"]["initial_feedback"].update(
                    stage="selector_initial_top3",
                    document_count=len(report["outcome"]["result"]["initial_selector_documents"]),
                    comparison_notice="Three selector documents, not BASE1's six reader documents.",
                )
                report["outcome"]["feedback"] = feedback
                report["offline_gold_answers"] = list(gold.answers)
                report["scoring_status"] = "completed"
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
            reports, client.calls, run_id=run_id, planned=len(questions),
            stop_reason=client.block_reason, started=started,
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
    history = reviewed_history(runs)
    series_prior = series_reserved(runs)
    subcap = min(SERIES_CAP_CNY - series_prior, 50.0 - history["prior_reserved_cny"])
    if subcap <= 0:
        raise ValueError("project or ReFormeR series budget exhausted")
    project = Path(__file__).resolve().parents[3]
    snapshot, git = source_snapshot(project), _git_state()
    probe = ReFormeRAPI(args.reformer_snapshot, args.reader_snapshot, None, None)
    plan = {
        "run_id": run_id, "protocol": PROTOCOL, "arm": ARM,
        "created_utc": datetime.now(UTC).isoformat(), "git": git,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": args.expected_manifest_sha256,
        "question_ids": ids, "start": args.start, "count": args.count,
        "cohort_policy": "first100_frozen_shared500_ids_not_selected_by_outcomes",
        "model": PILOT_MODEL, "generation_profile": GENERATION_PROFILE,
        "selector_and_rewrite_output_tokens": 512, "answer_output_tokens": 1024,
        "temperature": 0, "top_p": 1, "enable_thinking": False,
        "initial_top_docs": 3, "answer_top_docs": 6, "retrieval_rounds": 2,
        "author": probe.provenance, "source_sha256": snapshot["sha256"],
        "index_path": str(args.index_path.resolve()),
        "subcap_cny": subcap, "project_cap_cny": 50, "max_calls": 3 * args.count,
        "series_cap_cny": SERIES_CAP_CNY, "series_prior_reserved_cny": series_prior,
        "timeout_seconds": 1800, "historical_budget": history,
        "price_checked_date": "2026-09-27",
        "price_source": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
        "price_input_cny_per_million": 0.2, "price_output_cny_per_million": 0.8,
        "no_training_no_memory_updates": True, "official_dev_test_used": False,
        "baseline_results_access": "separate offline analysis after prediction; never runtime",
        "failure_policy": "First API/protocol failure stops predictions; no retries. "
        "Unused claimed IDs require a separate continuation audit.",
        "notice": "Public-code prompted pattern selector, not paper-trained selector. "
        "Final retrieval uses paper-inspired original+rewrite hybrid. "
        "Frozen source examples are retained including imperfections. "
        "BASE1/ReFormeR2/S2G4 have unequal budgets; not a causal memory ablation.",
    }
    if not args.allow_network:
        print(json.dumps(
            {key: value for key, value in plan.items() if key != "historical_budget"},
            ensure_ascii=False, indent=2,
        ))
        return 0
    if not args.api_config or not git["commit"] or git["worktree_dirty"]:
        raise ValueError("live launch needs config path and a clean committed checkout")
    from .shared_s2g_corpus import load_runtime

    runtime = load_runtime(
        args.manifest, index_path=args.index_path,
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
        # Exclusive create protects a duplicated launch of this exact batch.
        runs.mkdir(parents=True, exist_ok=True)
        with (runs / f"{run_id}.claim.json").open("x", encoding="utf-8") as handle:
            json.dump({
                "protocol": PROTOCOL, "manifest_sha256": args.expected_manifest_sha256,
                "question_ids": ids, "plan_sha256": fingerprint(plan), "no_retry": True,
            }, handle, ensure_ascii=False, indent=2)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "launch_plan.json", plan)
        write_json(output / "source_snapshot.json", snapshot)
        write_json(output / "process.json", {
            "pid": os.getpid(), "argv": sys.argv, "started_utc": datetime.now(UTC).isoformat(),
        })
        progress = ProgressLog(output)
        progress({"kind": "launch", "pid": os.getpid(), "model": PILOT_MODEL})
        os.environ[KEY_VARIABLE] = settings.api_key
        client = SharedBudgetClient(
            LiveChatClient(ChatConfig(
                settings.base_url, PILOT_MODEL, KEY_VARIABLE,
                max_calls=3 * args.count, max_output_tokens=512,
                output_limit_parameter="max_tokens", enable_thinking=False,
                temperature=0, top_p=1, allow_s2g_answer_prefix_on_length=True,
            ), output / "api_audit", allow_network=True),
            PriceLimits(budget_cny=subcap, max_elapsed_seconds=1800, max_prompt_bytes=30000),
            output / "request_journal",
        )
        client.schema_stages, client.qwen_output_caps = True, True
        execute_batch(
            questions, runtime, client, args.reformer_snapshot, args.reader_snapshot,
            args.manifest, args.expected_manifest_sha256, output, progress,
            run_id=run_id, start=args.start,
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
            write_json(output / "cumulative_budget.json", {
                "prior_reserved_cny": history["prior_reserved_cny"],
                "new_reserved_cny": client.reserved_cny,
                "cumulative_reserved_cny": history["prior_reserved_cny"] + client.reserved_cny,
                "prior_unknown_cost_requests": history["prior_unknown_cost_requests"],
                "project_cap_cny": 50,
            })
            progress({
                "kind": "exit", "requests": client.attempts,
                "status": "failed" if failed or client.block_reason else "completed",
            })
        if previous is None:
            os.environ.pop(KEY_VARIABLE, None)
        else:
            os.environ[KEY_VARIABLE] = previous


if __name__ == "__main__":
    raise SystemExit(main())
