"""One-use eight-question BASE versus bounded non-FT DualRAG development pilot.

Default: write a plan, zero API calls. Live execution requires explicit opt-in,
the pinned ordinary Beijing API profile, complete historical budget audit and
a new output directory. Old failures are never retried or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from urllib.parse import urlsplit

from growrag.outer_loop import RagRequest, RetrieverReaderBackend

from . import dualrag_adapter as dualrag
from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_benchmark import call_totals
from .hotpot import evaluate_layered_feedback
from .llm_adapters import _execution_kind
from .pre_pilot import _IndexAdapter, write_json
from .prior_budget import reconcile_history
from .protocol import Answer, BackendCallError
from .representation_runner import fingerprint
from .run_bounded_system import READER_PROMPT, READER_VERSION, APIShortAnswerReader, plain
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient, load_debug

SCHEMA = "growrag-dualrag-debug-pilot-v1"
RUN_ID = "2026-09-24_dualrag_debug8_v1"
CLAIM_NAME = "2026-09-24_dualrag_debug8.claim.json"
ARMS = ("BASE1", "DUALRAG2")
EXTRA_ROOTS = tuple(
    f"{name}/final_budget.json"
    for name in (
        "2026-09-20_pre_opportunity_8_v1",
        "2026-09-23_bounded_system_8_v1",
        "2026-09-23_bounded_system_unstarted7_v2",
        "2026-09-23_judge_paired_stability_v1",
    )
)
KEY_VARIABLE = "GROWRAG_DUALRAG_PILOT_API_KEY"


def source_snapshot(project: Path) -> dict:
    """Snapshot source text only; no credentials, data, artifacts or bytecode."""
    files = {}
    for path in sorted((project / "src").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to((project / "src").resolve()):
            raise ValueError("source snapshot cannot follow an external link")
        raw = path.read_bytes()
        files[path.relative_to(project).as_posix()] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text": raw.decode("utf-8"),
        }
    if not files:
        raise ValueError("empty source snapshot")
    return {"files": files, "sha256": fingerprint(files)}


def prompts() -> dict:
    values = {READER_VERSION: READER_PROMPT}
    for name, text in (
        ("reason", dualrag.REASON_PROMPT),
        ("entities", dualrag.ENTITIES_PROMPT),
        ("summarize", dualrag.SUMMARY_PROMPT),
        ("answer", dualrag.ANSWER_PROMPT),
    ):
        values[dualrag.PROMPT_VERSIONS[name]] = text
    return {
        version: {"text": text, "sha256": fingerprint(text)} for version, text in values.items()
    }


def run_question(example, client, directory: Path) -> dict:
    """Both runtime paths end before gold is accessed; incomplete pairs stay unscored."""
    directory.mkdir(parents=True, exist_ok=False)
    question = example.question
    outcomes, runtime = {}, {}
    # Fixed order is a declared engineering pilot condition, not optimized by outcomes.
    for arm in ARMS:
        if client.block_reason:
            outcomes[arm] = {"status": "not_executed", "feedback": None}
            continue
        start, clock = len(client.calls), perf_counter()
        try:
            index = _IndexAdapter(example.candidate_context, _execution_kind(client))
            if arm == "BASE1":
                result = RetrieverReaderBackend(index, APIShortAnswerReader(client), top_k=4).run(
                    RagRequest(question, question.text)
                )
                reply = result.value
                runtime[arm] = (reply.answer, reply.evidence or (), reply.evidence or ())
                row = {
                    "status": "completed",
                    "result": plain(result),
                    "answer": plain(reply.answer),
                    "stop_reason": "single_retrieval",
                    "retrieval_calls": 1,
                    "evidence_scope": "actual raw reader context",
                }
            else:
                result = dualrag.DualRAGAdapter(client, index).run(question)
                sources = {ref for k in result.knowledge for ref in k.evidence_ids}
                active_sources = tuple(
                    e for e in result.observed_evidence if e.evidence_id in sources
                )
                if result.status == "completed":
                    runtime[arm] = (result.answer, active_sources, result.observed_evidence)
                row = {
                    "status": result.status,
                    "result": plain(result),
                    "answer": plain(result.answer),
                    "stop_reason": result.stop_reason,
                    "retrieval_calls": sum(e.operation == "retrieve" for e in result.events),
                    "evidence_scope": "source IDs retained in summaries; final answer sees "
                    "summaries, not every raw retrieved sentence",
                }
        except Exception as error:
            row = {"status": "failed", "answer": None, "error_type": type(error).__name__}
            if isinstance(error, BackendCallError):
                row["failure_call"] = {
                    field: plain(getattr(error, field))
                    for field in (
                        "usage",
                        "provider",
                        "model",
                        "request_id",
                        "audit_path",
                        "transport_source",
                    )
                }
                row["component_events"] = plain(getattr(error, "component_events", ()))
        row.update(
            feedback=None,
            calls=list(client.calls[start:]),
            call_cost=call_totals(client.calls[start:]),
            elapsed_seconds=perf_counter() - clock,
        )
        outcomes[arm] = row
        write_json(directory / f"{arm}_execution.json", row)
        if row["status"] != "completed":
            client.block_reason = client.block_reason or f"{arm.lower()}_component_failure"
    complete = all(outcomes[arm]["status"] == "completed" for arm in ARMS)
    if complete and example.gold is not None:
        for arm in ARMS:
            answer, active_sources, all_retrieved = runtime[arm]
            outcomes[arm]["feedback"] = asdict(
                evaluate_layered_feedback(
                    question,
                    (),
                    active_sources,
                    answer,
                    example.gold,
                )
            )
            # Raw retrieval coverage and summary retention are not interchangeable.
            outcomes[arm]["all_retrieved_support_recall"] = evaluate_layered_feedback(
                question,
                (),
                all_retrieved,
                Answer(""),
                example.gold,
            ).retrieved_gold_support_recall
    report = {
        "question_id": question.question_id,
        "question": question.text,
        "execution_kind": _execution_kind(client).value,
        "arms": outcomes,
        "complete_pair": complete,
        "arm_order": ARMS,
        "scoring_notice": "Gold accessed only after both successful paths finish. "
        "For DUALRAG2, feedback support recall is summary-source coverage, not "
        "verified summary entailment; all-retrieved coverage is separate.",
    }
    write_json(directory / "report.json", report)
    return report


def summarize(reports: list[dict], *, planned: int, started: int) -> dict:
    complete = [r for r in reports if all(r["arms"][a].get("feedback") is not None for a in ARMS)]
    result = {
        "schema_version": SCHEMA,
        "planned_questions": planned,
        "started_questions": started,
        "not_started_questions": planned - started,
        "reported_questions": len(reports),
        "complete_scored_pairs": len(complete),
        "arms": {},
        "notice": "Previously exposed training-split debug questions only, not official "
        "dev/test performance, statistical significance, or published-score reproduction.",
    }
    for arm in ARMS:
        rows = [r["arms"][arm] for r in complete]
        result["arms"][arm] = {
            "n": len(rows),
            "answer_em": mean(r["feedback"]["answer_em"] for r in rows) if rows else None,
            "answer_f1": mean(r["feedback"]["answer_f1"] for r in rows) if rows else None,
            "mean_retrieval_calls": mean(r["retrieval_calls"] for r in rows) if rows else None,
            "cost_including_incomplete": call_totals(
                [call for r in reports for call in r["arms"][arm].get("calls", [])]
            ),
        }
    result["paired"] = {
        "n": len(complete),
        "mean_f1_delta": mean(
            r["arms"]["DUALRAG2"]["feedback"]["answer_f1"]
            - r["arms"]["BASE1"]["feedback"]["answer_f1"]
            for r in complete
        )
        if complete
        else None,
        "em_rescues": sum(
            r["arms"]["BASE1"]["feedback"]["answer_em"] == 0
            and r["arms"]["DUALRAG2"]["feedback"]["answer_em"] == 1
            for r in complete
        ),
        "em_harms": sum(
            r["arms"]["BASE1"]["feedback"]["answer_em"] == 1
            and r["arms"]["DUALRAG2"]["feedback"]["answer_em"] == 0
            for r in complete
        ),
    }
    return result


def execute(client, examples, output: Path) -> dict:
    reports, started = [], 0
    try:
        for index, example in enumerate(examples):
            if client.block_reason:
                break
            started += 1
            reports.append(run_question(example, client, output / "questions" / f"{index:03d}"))
            print(
                f"DualRAG debug {index + 1}: complete={reports[-1]['complete_pair']}; "
                f"actual requests={client.attempts}; stop={client.block_reason}",
                flush=True,
            )
            if index == 1:
                write_json(
                    output / "first_two_checkpoint.json",
                    {
                        "completed_pairs": sum(r["complete_pair"] for r in reports),
                        "continue_only_if_no_protocol_failure": client.block_reason is None,
                        "outcome_based_prompt_tuning": False,
                    },
                )
    finally:
        write_json(output / "reports.json", reports)
        result = summarize(reports, planned=len(examples), started=started)
        write_json(output / "summary.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    root, output = args.runs_root.resolve(), args.output.resolve()
    if output == root or not output.is_relative_to(root) or output.exists():
        raise ValueError("output must be new and strictly inside runs_root")
    if args.allow_network and output.name != RUN_ID:
        raise ValueError("live execution requires the frozen one-use run name")
    claim = root / CLAIM_NAME
    if args.allow_network and claim.exists():
        raise FileExistsError("DualRAG pilot already claimed; no resume or retry")
    examples, data = load_debug(args.manifest)
    history = reconcile_history(root, reviewed_extra_ledgers=EXTRA_ROOTS)
    subcap = min(3.0, 50.0 - history["prior_reserved_cny"])
    if subcap <= 0:
        raise ValueError("no conservative project budget remains")
    source = source_snapshot(Path(__file__).resolve().parents[3])
    limits = dualrag.DualRAGLimits()
    plan = {
        "schema_version": SCHEMA,
        "date": "2026-09-24",
        "git": _git_state(),
        "data": data,
        "source_snapshot_sha256": source["sha256"],
        "prompts": prompts(),
        "arms": ARMS,
        "model": PILOT_MODEL,
        "temperature": 0,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
        "json_schema_mode": False,
        "max_output_tokens": 1024,
        "max_api_calls": 80,
        "max_elapsed_seconds": 1800,
        "subcap_cny": subcap,
        "authorized_project_cap_cny": 50,
        "historical_budget": history,
        "method_limits": asdict(limits),
        "worst_case_model_calls": len(examples) * (limits.max_model_calls + 1),
        "input_unit_cny_per_million": 0.2,
        "output_unit_cny_per_million": 0.8,
        "price_source": "https://help.aliyun.com/zh/model-studio/model-pricing",
        "price_checked_date": "2026-09-24",
        "source_paper": dualrag.SOURCE_URL,
        "source_code_commit": dualrag.SOURCE_CODE_COMMIT,
        "adaptation_notes": dualrag.ADAPTATION_NOTES,
        "initial_checkpoint": "first two pairs; remaining six only if protocol succeeds; "
        "do not select continuation by correctness or tune prompts mid-run",
        "no_training_or_memory": True,
        "check24_used": False,
        "official_dev_test_used": False,
    }
    if plan["worst_case_model_calls"] > plan["max_api_calls"]:
        raise ValueError("planned calls exceed frozen API-call cap")
    if not args.allow_network:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        write_json(output / "source_snapshot.json", source)
        print(f"Plan only, zero API calls: {output / 'plan.json'}")
        return 0
    if not args.api_config or not plan["git"]["commit"]:
        raise ValueError("live execution requires local config and a recorded Git commit")
    # Dirty documentation is allowed; exact source bytes are snapshotted and rechecked.
    try:
        settings = read_local_bailian_settings(args.api_config)
    except (OSError, ValueError):
        print("Invalid local API configuration; no secrets shown or calls sent.")
        return 2
    host = urlsplit(settings.base_url).hostname or ""
    if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
        raise ValueError("frozen price profile requires ordinary Beijing endpoint")
    if source_snapshot(Path(__file__).resolve().parents[3])["sha256"] != source["sha256"]:
        raise ValueError("source changed during launch; review before running")
    write_json(claim, {"output": str(output), "plan_sha256": fingerprint(plan), "no_retry": True})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch_plan.json", plan)
    write_json(output / "source_snapshot.json", source)
    previous, client = os.environ.get(KEY_VARIABLE), None
    os.environ[KEY_VARIABLE] = settings.api_key
    try:
        client = DurableBudgetClient(
            LiveChatClient(
                ChatConfig(
                    settings.base_url,
                    PILOT_MODEL,
                    KEY_VARIABLE,
                    max_calls=80,
                    max_output_tokens=1024,
                    timeout_seconds=45,
                    output_limit_parameter="max_tokens",
                    enable_thinking=False,
                    temperature=0,
                    json_object_mode=True,
                ),
                output / "api_audit",
                allow_network=True,
            ),
            PriceLimits(
                budget_cny=subcap,
                max_elapsed_seconds=1800,
            ),
            output / "request_journal",
        )
        execute(client, examples, output)
        return 1 if client.block_reason else 0
    except Exception as error:
        write_json(
            output / "interrupted.json", {"error_type": type(error).__name__, "no_retry": True}
        )
        print("Stopped; all partial records retained, no retry.")
        return 1
    finally:
        if client is not None:
            budget = client.report()
            write_json(output / "final_budget.json", budget)
            write_json(
                output / "cumulative_budget.json",
                {
                    "prior": history,
                    "new_calls": budget,
                    "authorized_total_cny": 50,
                    "cumulative_reserved_cny": history["prior_reserved_cny"]
                    + budget["reserved_cny"],
                    "notice": "Declared-price estimate, not invoice; "
                    "historical unknown stays unknown.",
                },
            )
        if previous is None:
            os.environ.pop(KEY_VARIABLE, None)
        else:
            os.environ[KEY_VARIABLE] = previous


if __name__ == "__main__":
    raise SystemExit(main())
