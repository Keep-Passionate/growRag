"""Frozen shared-corpus S2G API batches; no training or automatic request retries.

中文：一个共享磁盘索引，两路真正执行后才评分；分批为防中断丢日志，
不是按分数筛选数据。旧运行不可覆盖，所有已消费请求继承到50元总账。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
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
from .prior_budget import reconcile_history
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_pilot import PILOT_MODEL, _git_state
from .run_s2g_author_pilot import (
    ARMS,
    CAPACITY_RUN_ID,
    HISTORY_ROOTS,
    JSON_RUN_ID,
    QWEN_OUTPUT_CAPS,
    RUN_ID,
    SCHEMA_RUN_ID,
    AuthorBudgetClient,
    ProgressLog,
)
from .s2g_author_api import PROMPT_VERSIONS, S2GAuthorAPI

PROTOCOL = "growrag-s2g-shared-qwen-capacity-v3"
REVIEWED_PROTOCOLS = {
    PROTOCOL,
    "growrag-s2g-shared-qwen-capacity-v1",
    "growrag-s2g-shared-qwen-capacity-v2",
}
SHARED_OUTPUT_CAPS = {**QWEN_OUTPUT_CAPS, "answer": 1024}
GENERATION_PROFILE = "qwen_capacity_v5_judge768_extract128_answer1024"
PREFIX = "2026-09-27_s2g_shared"
KEY_VARIABLE = "GROWRAG_SHARED_S2G_API_KEY"
MAX_BATCH = 50
HISTORICAL_ROOTS = (
    *HISTORY_ROOTS,
    *(
        f"{name}/final_budget.json"
        for name in (RUN_ID, JSON_RUN_ID, SCHEMA_RUN_ID, CAPACITY_RUN_ID)
    ),
)
# 新的模式库实验仍消费同一项目预算；不是重新获得一个50元额度。
REVIEWED_OTHER_SERIES = {
    "2026-09-30_operator_v1_": {"growrag-operator-study-v1"},
    "2026-09-30_operator_v2_": {"growrag-operator-study-structured-v2"},
    "2026-09-27_reformer_hotpot_v1_": {"growrag-reformer-public-qwen-v1"},
    "2026-09-27_reformer_control_v1_": {"growrag-reformer-id-examples-v1"},
    "2026-09-27_memory_source_v1_": {"growrag-independent-memory-source-collection-v1"},
}


class SharedBudgetClient(AuthorBudgetClient):
    """Explicit v5 capacity change; original v4 runs and transport are unchanged."""

    @property
    def generation_profile(self):
        return GENERATION_PROFILE

    def author_output_cap(self, prompt_version, requested_cap):
        original = super().author_output_cap(prompt_version, requested_cap)
        if self.qwen_output_caps and prompt_version == PROMPT_VERSIONS["answer"]:
            return SHARED_OUTPUT_CAPS["answer"]
        return original


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def reviewed_history(runs: Path) -> dict:
    """Declared S2G/ReFormeR roots share one budget; unknown audits still fail."""
    extra = list(HISTORICAL_ROOTS)
    for prefix, protocols in {PREFIX: REVIEWED_PROTOCOLS, **REVIEWED_OTHER_SERIES}.items():
        for directory in runs.glob(f"{prefix}*/request_journal"):
            if any(directory.iterdir()) and not (directory.parent / "final_budget.json").exists():
                raise ValueError("unfinished request journal needs offline budget reconciliation")
        for ledger_path in sorted(runs.glob(f"{prefix}*/final_budget.json")):
            launch = json.loads((ledger_path.parent / "launch_plan.json").read_bytes())
            if launch.get("protocol") not in protocols or launch.get("model") != PILOT_MODEL:
                raise ValueError("unreviewed shared-run protocol/model")
            ledger = json.loads(ledger_path.read_bytes())
            if ledger.get("api_requests", 0):
                extra.append(ledger_path.relative_to(runs).as_posix())
            elif ledger.get("calls"):
                raise ValueError("zero-request run has uncertain pending calls")
    return reconcile_history(runs, reviewed_extra_ledgers=tuple(extra))


def batch_identity(series: str, start: int, count: int) -> str:
    if not re.fullmatch(r"(?:32|500)_v[1-9][0-9]*", series):
        raise ValueError("explicit reviewed 32/500 series identity required")
    if type(start) is not int or start < 0 or type(count) is not int or not 1 <= count <= MAX_BATCH:
        raise ValueError("invalid frozen batch bounds")
    return f"{PREFIX}{series}_{start:04d}_{start + count:04d}"


def check_no_replay(
    runs: Path,
    series: str,
    manifest_sha: str,
    question_ids: list[str],
    *,
    continuation_of: str | None = None,
) -> dict | None:
    proof = None
    if continuation_of is not None:
        from .shared_continuation import verify_unstarted_continuation

        if not continuation_of.startswith(f"{PREFIX}{series}_"):
            raise ValueError("continuation must remain in the same frozen series")
        proof = verify_unstarted_continuation(
            runs,
            continuation_of,
            manifest_sha256=manifest_sha,
            question_ids=question_ids,
            generation_profile=GENERATION_PROFILE,
            protocol=PROTOCOL,
            model=PILOT_MODEL,
        )
    wanted = set(question_ids)
    for path in runs.glob(f"{PREFIX}{series}_*/launch_plan.json"):
        old = json.loads(path.read_bytes())
        if old.get("manifest_sha256") != manifest_sha:
            raise ValueError("series cannot silently change dataset")
        if wanted & set(old.get("question_ids", [])):
            if proof is not None and path.parent.name in proof.get(
                "validated_prior_runs", [continuation_of]
            ):
                continue
            raise ValueError("question already claimed in this series; no automatic replay")
    return proof


def execute_pair(question, index, client, upstream, directory, progress, *, run_id):
    """Runtime half: accepts NO gold object or question type, returns durable outputs."""
    directory.mkdir(parents=True, exist_ok=False)
    outcomes = {}
    for arm in ARMS:
        if client.block_reason:
            outcomes[arm] = {"status": "not_executed", "feedback": None, "calls": []}
            continue
        progress.arm = arm
        started, first_call = perf_counter(), len(client.calls)
        adapter = None
        progress({"kind": "arm_start", "question_id": question.question_id})
        try:
            adapter = S2GAuthorAPI(
                upstream,
                client,
                index,
                max_turns=4,
                top_docs=6,
                gap_profile="paper_k1",
                remove_repeat_docs=True,
                event_callback=progress,
            )
            trace = f"{run_id}/{question.question_id}/{arm}"
            if arm == ARMS[0]:
                before_retrieval = perf_counter()
                docs = index(question.text, 6)
                retrieval_seconds = perf_counter() - before_retrieval
                progress(
                    {
                        "kind": "retrieval",
                        "round": 1,
                        "query": question.text,
                        "documents": [asdict(d) for d in docs],
                        "elapsed_seconds": retrieval_seconds,
                    }
                )
                context = adapter.scope["concat_raw_retrieved_docs"](
                    [d.title for d in docs], [d.text for d in docs]
                )
                result = adapter.answer_once(question.text, context, trace)
                result.update(
                    retrieved_documents=[asdict(d) for d in docs],
                    retrieval_rounds=1,
                    retrieval_seconds=retrieval_seconds,
                )
            else:
                result = adapter.run(question.text, trace)
            # API events keep their unique run/arm trace. Scoring uses the dataset ID.
            result["execution_trace_id"] = result["question_id"]
            result["question_id"] = question.question_id
            row = {"status": "completed", "result": result, "feedback": None}
        except Exception as error:
            client.block_reason = client.block_reason or "author_component_failure"
            row = {
                "status": "failed",
                "error_type": type(error).__name__,
                "events": adapter.events if adapter is not None else [],
                "feedback": None,
            }
            progress({"kind": "arm_failed", "error_type": type(error).__name__})
        row.update(calls=list(client.calls[first_call:]), elapsed_seconds=perf_counter() - started)
        outcomes[arm] = row
        write_json(directory / f"{arm}_execution.json", row)
    return outcomes


def summarize(reports, calls, *, run_id, planned, stop_reason, started=None):
    complete = [r for r in reports if r["complete_pair"]]
    scorable = [
        r
        for r in complete
        if all((r["arms"][a].get("feedback") or {}).get("answer_em") is not None for a in ARMS)
    ]
    summary = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "planned": planned,
        "started": len(reports) if started is None else started,
        "started_without_report": 0 if started is None else started - len(reports),
        "complete_pairs": len(complete),
        "scored_pairs": len(scorable),
        "unscored_complete_pairs": len(complete) - len(scorable),
        "unscorable_annotation_pairs": sum(
            any(
                (r["arms"][a].get("feedback") or {}).get("unscorable_annotation", False)
                for a in ARMS
            )
            for r in complete
        ),
        "scoring_failed_pairs": sum(r.get("scoring_status") == "failed" for r in complete),
        "stop_reason": stop_reason,
        "benchmark_reproduction": False,
        "arms": {},
        "all_actual_calls_including_interrupted": call_totals(calls),
        "notice": "Qwen migration; closed shared corpus, not original fullwiki/LoRA. "
        "BASE1 vs S2G4 differs in budget and evidence extraction; not a single-factor ablation.",
    }
    for arm in ARMS:
        rows = [r["arms"][arm] for r in scorable]
        summary["arms"][arm] = {
            "n": len(rows),
            "em": mean(r["feedback"]["answer_em"] for r in rows) if rows else None,
            "f1": mean(r["feedback"]["answer_f1"] for r in rows) if rows else None,
            "retrieval_rounds_total": sum(r["result"]["retrieval_rounds"] for r in rows),
            "all_started_cost": call_totals([c for r in reports for c in r["arms"][arm]["calls"]]),
        }
    summary["paired_em"] = {
        "repairs": sum(
            r["arms"][ARMS[0]]["feedback"]["answer_em"] == 0
            and r["arms"][ARMS[1]]["feedback"]["answer_em"] == 1
            for r in scorable
        ),
        "harms": sum(
            r["arms"][ARMS[0]]["feedback"]["answer_em"] == 1
            and r["arms"][ARMS[1]]["feedback"]["answer_em"] == 0
            for r in scorable
        ),
    }
    return summary


def execute_batch(
    questions,
    runtime,
    client,
    upstream,
    manifest_path,
    manifest_sha,
    output,
    progress,
    *,
    run_id,
    start,
    question_types,
):
    from .shared_s2g_corpus import load_gold_after_execution, score_result

    reports, started = [], 0
    try:
        for offset, question in enumerate(questions, start):
            if client.block_reason:
                break
            started += 1
            directory = output / "questions" / f"{offset:04d}"
            outcomes = execute_pair(
                question, runtime.index, client, upstream, directory, progress, run_id=run_id
            )
            complete = all(outcomes[a]["status"] == "completed" for a in ARMS)
            report = {
                "question_id": question.question_id,
                "question": question.text,
                "question_type": question_types[question.question_id],
                "offset": offset,
                "arms": outcomes,
                "complete_pair": complete,
            }
            reports.append(report)
            # No gold was constructed/read by the runtime: both branch outputs now exist.
            if complete:
                try:
                    gold = load_gold_after_execution(
                        manifest_path,
                        completed_question_ids=[question.question_id],
                        expected_manifest_sha256=manifest_sha,
                    )[question.question_id]
                    feedback = {
                        arm: score_result(
                            outcomes[arm]["result"],
                            gold,
                            runtime.index,
                            retained_mode="raw" if arm == ARMS[0] else "sources",
                        )
                        for arm in ARMS
                    }
                    for arm in ARMS:
                        outcomes[arm]["feedback"] = feedback[arm]
                    report["offline_gold_answers"] = list(gold.answers)
                    report["scoring_status"] = "completed"
                except Exception as error:
                    report["scoring_status"] = "failed"
                    report["scoring_error_type"] = type(error).__name__
                    client.block_reason = client.block_reason or "offline_scoring_failure"
            write_json(directory / "report.json", report)
            write_json(
                output / f"summary_after_{offset:04d}.json",
                summarize(
                    reports,
                    client.calls,
                    run_id=run_id,
                    planned=len(questions),
                    stop_reason=client.block_reason,
                    started=started,
                ),
            )
            progress(
                {
                    "kind": "question_complete",
                    "question_id": question.question_id,
                    "status": "completed" if complete else "failed",
                    "requests": client.attempts,
                }
            )
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
    for name in ("manifest", "upstream", "index-path", "runs-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--series", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--continue-unstarted-from")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    run_id = batch_identity(args.series, args.start, args.count)
    runs = args.runs_root.resolve()
    output = runs / run_id
    if output.exists() or (runs / f"{run_id}.claim.json").exists():
        raise FileExistsError("batch already exists; preserve all previous work")
    if _sha(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("frozen manifest SHA mismatch")
    manifest = json.loads(args.manifest.read_bytes())
    if manifest.get("official_split") != "train" or manifest.get("role") != "development":
        raise ValueError("only frozen train development protocol is authorized here")
    ids = manifest["question_ids"][args.start : args.start + args.count]
    if len(ids) != args.count or len(set(ids)) != len(ids):
        raise ValueError("batch extends beyond frozen dataset")
    continuation_proof = check_no_replay(
        runs,
        args.series,
        args.expected_manifest_sha256,
        ids,
        continuation_of=args.continue_unstarted_from,
    )
    history = reviewed_history(runs)
    max_calls = 11 * args.count
    subcap = min(5.0, 50.0 - history["prior_reserved_cny"])
    if subcap <= 0:
        raise ValueError("project budget exhausted")
    project = Path(__file__).resolve().parents[3]
    snapshot, git = source_snapshot(project), _git_state()
    probe = S2GAuthorAPI(
        args.upstream, None, lambda q, k: (), gap_profile="paper_k1", remove_repeat_docs=True
    )
    author_provenance = probe.provenance
    author_provenance["backend_generation_settings"] = GENERATION_PROFILE
    plan = {
        "run_id": run_id,
        "protocol": PROTOCOL,
        "series": args.series,
        "continuation_of": args.continue_unstarted_from,
        "continuation_proof": continuation_proof,
        "created_utc": datetime.now(UTC).isoformat(),
        "git": git,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": args.expected_manifest_sha256,
        "question_ids": ids,
        "start": args.start,
        "count": args.count,
        "model": PILOT_MODEL,
        "backend_output_caps": SHARED_OUTPUT_CAPS,
        "generation_profile": GENERATION_PROFILE,
        "answer_length_policy": "accept complete Answer field before Rationale delimiter; "
        "preserve and flag truncated rationale; other stages remain strict",
        "max_retrieval_rounds": 4,
        "top_docs": 6,
        "gap_profile": "paper_k1",
        "arms": ARMS,
        "author": author_provenance,
        "source_sha256": snapshot["sha256"],
        "subcap_cny": subcap,
        "project_cap_cny": 50,
        "max_calls": max_calls,
        "timeout_seconds": 1800,
        "price_checked_date": "2026-09-27",
        "price_source": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
        "price_input_cny_per_million": 0.2,
        "price_output_cny_per_million": 0.8,
        "historical_budget": history,
        "index_path": str(args.index_path.resolve()),
        "no_training_no_memory_updates": True,
        "official_dev_test_used": False,
        "failure_policy": "First API/protocol failure stops batch; no request replay. "
        "Independent later batches require audited diagnosis before launch.",
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
    questions = runtime.questions[args.start : args.start + args.count]
    if [q.question_id for q in questions] != ids:
        runtime.index.close()
        raise ValueError("runtime question order differs from launch plan")
    settings = read_local_bailian_settings(args.api_config)
    host = urlsplit(settings.base_url).hostname or ""
    if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
        runtime.index.close()
        raise ValueError("ordinary Beijing endpoint required for frozen price")
    if source_snapshot(project)["sha256"] != snapshot["sha256"]:
        runtime.index.close()
        raise ValueError("source changed before live launch")
    plan["runtime_metadata"] = runtime.metadata
    write_json(runs / f"{run_id}.claim.json", {"plan_sha256": fingerprint(plan), "no_retry": True})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch_plan.json", plan)
    write_json(output / "source_snapshot.json", snapshot)
    write_json(
        output / "process.json",
        {"pid": os.getpid(), "argv": sys.argv, "started_utc": datetime.now(UTC).isoformat()},
    )
    progress = ProgressLog(output)
    progress({"kind": "launch", "pid": os.getpid(), "model": PILOT_MODEL})
    previous, client, failed = os.environ.get(KEY_VARIABLE), None, False
    os.environ[KEY_VARIABLE] = settings.api_key
    try:
        client = SharedBudgetClient(
            LiveChatClient(
                ChatConfig(
                    settings.base_url,
                    PILOT_MODEL,
                    KEY_VARIABLE,
                    max_calls=max_calls,
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
        execute_batch(
            questions,
            runtime,
            client,
            args.upstream,
            args.manifest,
            args.expected_manifest_sha256,
            output,
            progress,
            run_id=run_id,
            start=args.start,
            question_types=manifest["question_types"],
        )
        return int(bool(client.block_reason))
    except Exception as error:
        failed = True
        if client is not None:
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
