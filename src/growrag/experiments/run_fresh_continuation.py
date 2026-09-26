"""New train-development questions: BASE, guarded DualRAG and sentence-pointer FRESH.

中文：这是完整流程诊断，预算和上下文形式不同，不是单因素消融。
先保存所有臂的真实执行，再用gold评分；不训练、不写历史经验、不重试。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter
from urllib.parse import urlsplit

from growrag.outer_loop import RagRequest, RetrieverReaderBackend

from . import dualrag_adapter as dualrag
from . import s2g_adapter as s2g
from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_benchmark import call_totals
from .fresh_dev_manifest import load_fresh_dev
from .hotpot import evaluate_layered_feedback
from .llm_adapters import _execution_kind
from .pre_pilot import _IndexAdapter, write_json
from .prior_budget import reconcile_history
from .protocol import Answer, BackendCallError
from .representation_runner import fingerprint
from .run_bounded_system import READER_PROMPT, READER_VERSION, APIShortAnswerReader, plain
from .run_dualrag_pilot import EXTRA_ROOTS, source_snapshot
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient

ARMS = ("BASE1", "DUAL_GUARDED2", "S2G_POINTER2")
RUN_ID = "2026-09-26_fresh_new12_v1"
SCHEMA = "growrag-fresh-new12-v1"
HISTORY_ROOTS = (*EXTRA_ROOTS, "2026-09-24_dualrag_debug8_v1/final_budget.json")
KEY_VARIABLE = "GROWRAG_FRESH_CONTINUATION_KEY"
LIVE_PROTOCOL_ENABLED = False  # 用户改为先审计/迁移作者 S2G；独立实现只保留离线检查。


def arm_order(question_id: str) -> tuple[str, ...]:
    return tuple(sorted(ARMS, key=lambda a: fingerprint([20260926, question_id, a])))


def prompt_manifest() -> dict:
    values = {READER_VERSION: READER_PROMPT}
    for name, prompt in (
        ("reason", dualrag.REASON_PROMPT),
        ("entities", dualrag.ENTITIES_PROMPT),
        ("summarize", dualrag.SUMMARY_PROMPT),
        ("answer", dualrag.ANSWER_PROMPT),
    ):
        values[dualrag.GUARDED_PROMPT_VERSIONS[name]] = prompt
    values.update(
        {
            s2g.PROMPT_VERSIONS["extract"]: s2g.EXTRACT_PROMPT,
            s2g.PROMPT_VERSIONS["judge"]: s2g.JUDGE_PROMPT,
        }
    )
    return {key: {"text": text, "sha256": fingerprint(text)} for key, text in values.items()}


def execute_arm(arm, question, evidence_pool, client) -> tuple[dict, dict]:
    """No GoldRecord or HotpotExample is accepted by this execution boundary."""
    if arm not in ARMS:
        raise ValueError("unknown frozen comparison arm")
    index = _IndexAdapter(evidence_pool, _execution_kind(client))
    if arm == "BASE1":
        result = RetrieverReaderBackend(index, APIShortAnswerReader(client), top_k=4).run(
            RagRequest(question, question.text)
        )
        answer, observed = result.value.answer, result.value.evidence or ()
        retained = provided = observed
        row = {
            "status": "completed",
            "result": plain(result),
            "answer": plain(answer),
            "stop_reason": "single_retrieval",
            "retrieval_calls": 1,
        }
    else:
        adapter = (
            dualrag.DualRAGAdapter(client, index, guarded_v2=True)
            if arm == "DUAL_GUARDED2"
            else s2g.S2GAdapter(client, index)
        )
        result = adapter.run(question)
        answer, observed = result.answer, result.observed_evidence
        if arm == "DUAL_GUARDED2":
            retained_ids = {ref for k in result.knowledge for ref in k.evidence_ids}
            retained = tuple(e for e in observed if e.evidence_id in retained_ids)
            supplied_ids = {
                e["evidence_id"]
                for step in result.steps
                if step["operation"] == "summarize"
                for e in step["input"]["evidence"]
            }
            provided = tuple(e for e in observed if e.evidence_id in supplied_ids)
        else:
            retained = result.evidence_context
            # Only successful extractor calls certify the input was processed.
            # Failed/blocked request details remain in raw steps/events, not scored.
            supplied_ids = {
                e["evidence_id"]
                for step in result.steps
                if step["operation"] == "extract" and step.get("status") == "ok"
                for e in step["input"]["evidence"]
            }
            provided = tuple(e for e in observed if e.evidence_id in supplied_ids)
        row = {
            "status": result.status,
            "result": plain(result),
            "answer": plain(answer),
            "stop_reason": result.stop_reason,
            "retrieval_calls": sum(e.operation == "retrieve" for e in result.events),
        }
    lineage = {"retrieved": observed, "provided": provided, "retained": retained}
    row["evidence_ids_by_stage"] = {
        stage: [e.evidence_id for e in evidence] for stage, evidence in lineage.items()
    }
    row["citation_ids"] = list(answer.cited_evidence_ids) if answer else []
    row["support_notice"] = (
        "ID coverage is not entailment. Dual retained means summary-source coverage, "
        "not verbatim facts read by the final model."
    )
    return row, {"answer": answer, **lineage}


def run_question(example, client, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    question, outcomes, runtime = example.question, {}, {}
    order = arm_order(question.question_id)
    for arm in order:
        if client.block_reason:
            outcomes[arm] = {"status": "not_executed", "feedback": None, "calls": []}
            write_json(directory / f"{arm}_execution.json", outcomes[arm])
            continue
        start, clock = len(client.calls), perf_counter()
        try:
            row, state = execute_arm(arm, question, example.candidate_context, client)
            if row["status"] == "completed":
                runtime[arm] = state
        except Exception as error:
            row = {"status": "failed", "answer": None, "error_type": type(error).__name__}
            if isinstance(error, BackendCallError):
                row["failure_call"] = {
                    key: plain(getattr(error, key))
                    for key in ("usage", "request_id", "audit_path", "transport_source")
                }
        row.update(
            feedback=None,
            calls=list(client.calls[start:]),
            cost=call_totals(client.calls[start:]),
            elapsed_seconds=perf_counter() - clock,
        )
        outcomes[arm] = row
        write_json(directory / f"{arm}_execution.json", row)
        if row["status"] != "completed":
            client.block_reason = client.block_reason or f"{arm.lower()}_failure"
    complete = all(outcomes[a]["status"] == "completed" for a in ARMS)
    # Gold is inspected only after all three independently executed paths finish.
    if complete and example.gold is not None:
        for arm in ARMS:
            state = runtime[arm]
            outcomes[arm]["feedback"] = asdict(
                evaluate_layered_feedback(
                    question,
                    (),
                    state["retained"],
                    state["answer"],
                    example.gold,
                )
            )
            outcomes[arm]["gold_support_recall_by_stage"] = {
                stage: evaluate_layered_feedback(
                    question,
                    (),
                    state[stage],
                    Answer(""),
                    example.gold,
                ).retrieved_gold_support_recall
                for stage in ("retrieved", "provided", "retained")
            }
    report = {
        "question_id": question.question_id,
        "question": question.text,
        "question_type": getattr(example, "question_type", None),
        "execution_kind": _execution_kind(client).value,
        "arms": outcomes,
        "arm_order": order,
        "complete_triplet": complete,
    }
    write_json(directory / "report.json", report)
    return report


def summarize(reports, *, planned, started):
    complete = [r for r in reports if all(r["arms"][a].get("feedback") for a in ARMS)]
    methods = {}
    for arm in ARMS:
        rows = [r["arms"][arm] for r in complete]
        methods[arm] = {
            "n": len(rows),
            "em": mean(r["feedback"]["answer_em"] for r in rows) if rows else None,
            "f1": mean(r["feedback"]["answer_f1"] for r in rows) if rows else None,
            "retrieval_calls": sum(r["retrieval_calls"] for r in rows),
            "abstentions": sum(not r["answer"]["text"] for r in rows),
            "cost_including_failures": call_totals(
                [c for r in reports for c in r["arms"][arm].get("calls", [])]
            ),
            "em_rescues_vs_base": sum(
                r["arms"][arm]["feedback"]["answer_em"] == 1
                and r["arms"]["BASE1"]["feedback"]["answer_em"] == 0
                for r in complete
            ),
            "em_harms_vs_base": sum(
                r["arms"][arm]["feedback"]["answer_em"] == 0
                and r["arms"]["BASE1"]["feedback"]["answer_em"] == 1
                for r in complete
            ),
        }
    return {
        "schema_version": SCHEMA,
        "planned": planned,
        "started": started,
        "not_started": planned - started,
        "reported": len(reports),
        "complete_scored_triplets": len(complete),
        "methods": methods,
        "notice": "New balanced train-development sample; unequal method budgets; "
        "not official benchmark performance, significance, or single-factor attribution.",
    }


def execute(client, examples, output):
    reports, started = [], 0
    try:
        for i, example in enumerate(examples):
            if client.block_reason:
                break
            started += 1
            reports.append(run_question(example, client, output / "questions" / f"{i:03d}"))
            print(
                f"Fresh {i + 1}/{len(examples)} complete={reports[-1]['complete_triplet']} "
                f"requests={client.attempts} stop={client.block_reason}",
                flush=True,
            )
    finally:
        write_json(output / "reports.json", reports)
        write_json(
            output / "summary.json",
            summarize(
                reports,
                planned=len(examples),
                started=started,
            ),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "runs-root", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if args.allow_network and not LIVE_PROTOCOL_ENABLED:
        raise ValueError("live protocol superseded: audit and port the author's S2G first")
    root, output = args.runs_root.resolve(), args.output.resolve()
    if output == root or not output.is_relative_to(root) or output.exists():
        raise ValueError("new output strictly inside runs required")
    claim = root / f"{RUN_ID}.claim.json"
    if args.allow_network and (output.name != RUN_ID or claim.exists()):
        raise ValueError("one-use live run identity already used or incorrect")
    examples, data = load_fresh_dev(args.manifest)
    if len(examples) != 12:
        raise ValueError("exactly 12 frozen development questions required")
    history = reconcile_history(root, reviewed_extra_ledgers=HISTORY_ROOTS)
    subcap = min(5.0, 50 - history["prior_reserved_cny"])
    if subcap <= 0:
        raise ValueError("project budget exhausted")
    project = Path(__file__).resolve().parents[3]
    source = source_snapshot(project)
    dual_limits, s2g_limits = dualrag.DualRAGLimits(), s2g.S2GLimits()
    worst_case_calls = len(examples) * (
        1 + dual_limits.max_model_calls + s2g_limits.max_model_calls
    )
    if worst_case_calls != 180:
        raise ValueError("adapter budget changed; revise the protocol before running")
    plan = {
        "schema_version": SCHEMA,
        "date": "2026-09-26",
        "git": _git_state(),
        "data": data,
        "model": PILOT_MODEL,
        "arms": ARMS,
        "source_snapshot_sha256": source["sha256"],
        "prompts": prompt_manifest(),
        "arm_orders": {e.question.question_id: arm_order(e.question.question_id) for e in examples},
        "dual_limits": asdict(dual_limits),
        "s2g_limits": asdict(s2g_limits),
        "dual_guard_version": dualrag.GUARD_VERSION,
        "dual_baseline_id": dualrag.GUARDED_BASELINE_ID,
        "dual_guard_notes": [
            "Empty knowledge cannot stop as sufficient; discard ungrounded information need.",
            "Reject explicit unfilled entity slots; keep valid queries or use original question.",
            "Original prompt texts are unchanged; guard identity is separately versioned.",
        ],
        "method_notes": [*dualrag.ADAPTATION_NOTES, *s2g.ADAPTATION_NOTES],
        "historical_budget": history,
        "subcap_cny": subcap,
        "project_cap_cny": 50,
        "max_api_calls": worst_case_calls,
        "max_elapsed_seconds": 1800,
        "max_output_tokens": 1024,
        "temperature": 0,
        "enable_thinking": False,
        "response_format": "json_object",
        "prices_cny_per_million": {"input": 0.2, "output": 0.8},
        "price_verified_date": "2026-09-26",
        "price_source": "https://help.aliyun.com/zh/model-studio/model-pricing",
        "no_retries": True,
        "no_training_or_memory_updates": True,
        "check24_used": False,
        "official_dev_test_used": False,
    }
    if not args.allow_network:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        write_json(output / "source_snapshot.json", source)
        print(f"Plan only; zero API requests: {output}")
        return 0
    if not args.api_config or not plan["git"]["commit"]:
        raise ValueError("live run requires recorded commit and local config")
    try:
        settings = read_local_bailian_settings(args.api_config)
    except (ValueError, OSError):
        print("Invalid local API settings; no requests or secret output.")
        return 2
    host = urlsplit(settings.base_url).hostname or ""
    if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
        raise ValueError("ordinary Beijing endpoint required by frozen price profile")
    if source_snapshot(project)["sha256"] != source["sha256"]:
        raise ValueError("source changed during launch")
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
                    max_calls=worst_case_calls,
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
            PriceLimits(budget_cny=subcap, max_elapsed_seconds=1800),
            output / "request_journal",
        )
        execute(client, examples, output)
        return 1 if client.block_reason else 0
    except Exception as error:
        write_json(output / "interrupted.json", {"error_type": type(error).__name__})
        print("Stopped without retry; partial artifacts preserved.")
        return 1
    finally:
        if client is not None:
            write_json(output / "final_budget.json", client.report())
            write_json(
                output / "cumulative_budget.json",
                {
                    "prior": history,
                    "current": client.report(),
                    "cap_cny": 50,
                    "reserved_cny": history["prior_reserved_cny"] + client.reserved_cny,
                    "notice": "Estimates, not provider invoice; old unknown usage remains unknown.",
                },
            )
        if previous is None:
            os.environ.pop(KEY_VARIABLE, None)
        else:
            os.environ[KEY_VARIABLE] = previous


if __name__ == "__main__":
    raise SystemExit(main())
