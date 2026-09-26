"""One-use, visible S2G author-control API migration on eight exposed train questions.

中文：不是完整维基复现。文档级小池检索、本题状态、真实API与离线gold评分分开。
默认只生成计划；真调用需要显式开关。日志滚动显示，首错停止，不自动重试。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter
from urllib.parse import urlsplit

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_benchmark import call_totals
from .hotpot import evaluate_layered_feedback
from .lexical_retriever import BM25SentenceRetriever
from .output_schemas import response_format_for
from .pre_pilot import write_json
from .prior_budget import reconcile_history
from .protocol import Answer, Evidence
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_fresh_continuation import HISTORY_ROOTS
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient, load_debug
from .s2g_author_api import PROMPT_VERSIONS, AuthorDocument, S2GAuthorAPI

RUN_ID = "2026-09-27_s2g_author_api_debug8_v1"
JSON_RUN_ID = "2026-09-27_s2g_author_api_json8_v2"
SCHEMA_RUN_ID = "2026-09-27_s2g_author_api_schema8_v3"
CAPACITY_RUN_ID = "2026-09-27_s2g_author_api_capacity8_v4"
AUTHOR_OUTPUT_CAPS = {"judge": 256, "extract": 64, "answer": 128}
QWEN_OUTPUT_CAPS = {"judge": 768, "extract": 128, "answer": 256}
KEY_VARIABLE = "GROWRAG_S2G_AUTHOR_PILOT_KEY"
ARMS = ("BASE1_AUTHOR_READER", "S2G_AUTHOR_API4")
MAX_CALLS = 100


def validate_author_response(event):
    """Sidecar protocol check, not a replacement of the author's decision/parser.

    Invalid API formats stop this migration pilot instead of becoming EM=0.
    This additional fail-closed operational behavior is disclosed in the plan.
    No semantic correctness is asserted by passing this check.
    """
    if event.get("kind") != "api_response":
        return
    stage, text = event.get("stage"), event.get("content")
    if not isinstance(text, str):
        raise ValueError("author response must be text")
    if stage == "answer":
        # Match the pinned author's case-sensitive field delimiters exactly.
        match = re.search(r"Answer:\s*(.*?)\s*Rationale:\s*(.*)", text, re.S)
        if match is None or not match.group(1).strip():
            raise ValueError("author answer format missing")
        return
    if stage not in {"judge", "extract"}:
        raise ValueError("unknown author stage")
    # Fenced JSON is accepted just as in the upstream parser. No repair/retry.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(text)
    except ValueError:
        raise ValueError("author JSON response invalid") from None
    if not isinstance(value, dict):
        raise ValueError("author JSON response must be an object")
    if stage == "judge":
        if type(value.get("sufficient")) is not bool or not isinstance(
            value.get("gap_items"), list
        ):
            raise ValueError("author judge fields invalid")
        if any(not isinstance(gap, dict) for gap in value["gap_items"]):
            raise ValueError("author gaps must be objects")
    elif not isinstance(value.get("evidence_global_ids"), list) or any(
        type(item) is not int for item in value["evidence_global_ids"]
    ):
        raise ValueError("author pointer fields invalid")


class ProgressLog:
    """Append-only human-readable lines plus complete structured events; no secrets."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.arm = "setup"

    def __call__(self, event):
        event = {"logged_at_utc": datetime.now(UTC).isoformat(), "arm": self.arm, **event}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
        brief = {
            k: event[k]
            for k in (
                "logged_at_utc",
                "arm",
                "kind",
                "question_id",
                "stage",
                "backend_max_output_tokens",
                "round",
                "query",
                "verdicts",
                "stop_reason",
                "answer",
                "returned_model",
                "input_tokens",
                "output_tokens",
                "elapsed_seconds",
                "requests",
                "error_type",
                "status",
            )
            if k in event
        }
        line = json.dumps(brief, ensure_ascii=False)
        with (self.directory / "live.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
        print(line, flush=True)
        # Log the real malformed response first, then stop; do not silently fix it.
        validate_author_response(event)


class AuthorBudgetClient(DurableBudgetClient):
    """One ledger; author caps by default, explicit Qwen capacity profile in v4.

    中文：只改变最大输出容量，不更改原作者提示、缺口、选句或停止策略。
    作者请求的cap与实际HTTP cap分开记录，不能把迁移设置称作原设置。
    """

    json_stages = False
    schema_stages = False
    qwen_output_caps = False

    @property
    def generation_profile(self):
        return (
            "qwen_capacity_v4_judge768_extract128_answer256"
            if self.qwen_output_caps
            else "author_per_stage_limits_via_complete_author"
        )

    def author_output_cap(self, prompt_version, requested_cap):
        if type(requested_cap) is not int or requested_cap not in {64, 128, 256}:
            raise ValueError("unreviewed author output cap")
        if not self.qwen_output_caps:
            return requested_cap
        stage = next((k for k, v in PROMPT_VERSIONS.items() if v == prompt_version), None)
        if stage is None or requested_cap != AUTHOR_OUTPUT_CAPS[stage]:
            raise ValueError("Qwen capacity requires exact reviewed author stage/cap")
        if not self.schema_stages or self.json_stages:
            raise ValueError("Qwen capacity requires strict schema stages only")
        return QWEN_OUTPUT_CAPS[stage]

    def complete_author(
        self, messages, *, trace_id, prompt_version, max_output_tokens, temperature, top_p
    ):
        actual_cap = self.author_output_cap(prompt_version, max_output_tokens)
        previous = self.config
        self.config = replace(
            previous,
            max_output_tokens=actual_cap,
            temperature=temperature,
            top_p=top_p,
            json_object_mode=self.json_stages
            and not self.schema_stages
            and prompt_version in {PROMPT_VERSIONS["judge"], PROMPT_VERSIONS["extract"]},
            json_schema_mode=self.schema_stages
            and prompt_version in {PROMPT_VERSIONS["judge"], PROMPT_VERSIONS["extract"]},
        )
        self.delegate.config = self.config
        try:
            return self.complete(messages, trace_id=trace_id, prompt_version=prompt_version)
        finally:
            self.config = previous
            self.delegate.config = previous


class LocalDocumentIndex:
    """Reuse the lexical scorer with one complete paragraph per index item.

    The scorer's class name contains Sentence, but each input here is a whole
    document. This is explicitly NOT Pyserini/Lucene or a fullwiki index.
    """

    def __init__(self, evidence_pool):
        grouped = {}
        for item in evidence_pool:
            grouped.setdefault(item.title, []).append(item)
        self.documents = {}
        for title, sentences in grouped.items():
            text = "\n\n".join(x.text for x in sorted(sentences, key=lambda e: e.sentence_id))
            identity = fingerprint([title, text])
            self.documents[identity] = AuthorDocument(identity, title, text)
        self.index = BM25SentenceRetriever(
            tuple(Evidence(d.doc_id, d.title, 0, d.text) for d in self.documents.values())
        )

    def __call__(self, query, k):
        return tuple(
            self.documents[e.evidence_id] for e in self.index.retrieve(query, top_k=k).value
        )


def aligned_sources(pool, sources):
    """Exact text alignment only: author split indices are NOT Hotpot sentence IDs."""

    def normalize(value):
        return " ".join(value.split())

    selected = {(s["title"], normalize(s["text"])) for s in sources}
    return tuple(e for e in pool if (e.title, normalize(e.text)) in selected)


def execute_question(example, client, upstream, directory, progress, *, run_id=RUN_ID):
    directory.mkdir(parents=True, exist_ok=False)
    outcomes = {}
    question, pool = example.question, example.candidate_context
    index = LocalDocumentIndex(pool)
    runtime = {}
    for arm in ARMS:
        progress.arm = arm
        if client.block_reason:
            outcomes[arm] = {"status": "not_executed", "feedback": None, "calls": []}
            continue
        start_calls, started = len(client.calls), perf_counter()
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
        progress({"kind": "arm_start", "question_id": question.question_id})
        try:
            # Unique execution IDs distinguish BASE/S2G; the original question is unchanged.
            trace = f"{run_id}/{question.question_id}/{arm}"
            if arm == ARMS[0]:
                docs = index(question.text, 6)
                context = adapter.scope["concat_raw_retrieved_docs"](
                    [d.title for d in docs], [d.text for d in docs]
                )
                progress(
                    {
                        "kind": "retrieval",
                        "query": question.text,
                        "documents": [asdict(d) for d in docs],
                        "round": 1,
                    }
                )
                result = adapter.answer_once(question.text, context, trace)
                result["retrieval_rounds"] = 1
                result["retrieved_documents"] = [asdict(d) for d in docs]
                observed = tuple(e for e in pool if e.title in {d.title for d in docs})
                retained = observed
            else:
                result = adapter.run(question.text, trace)
                observed = tuple(
                    e
                    for e in pool
                    if e.title in {d["title"] for d in result["retrieved_documents"]}
                )
                retained = aligned_sources(pool, result["sources"])
            runtime[arm] = (Answer(result["answer"]), observed, retained)
            row = {"status": "completed", "result": result, "feedback": None}
        except Exception as error:
            client.block_reason = client.block_reason or "author_component_failure"
            row = {
                "status": "failed",
                "error_type": type(error).__name__,
                "events": adapter.events,
                "feedback": None,
            }
            progress({"kind": "arm_failed", "error_type": type(error).__name__})
        row["calls"] = list(client.calls[start_calls:])
        row["elapsed_seconds"] = perf_counter() - started
        outcomes[arm] = row
        write_json(directory / f"{arm}_execution.json", row)
    complete = all(outcomes[a]["status"] == "completed" for a in ARMS)
    # Only now are gold labels consulted. No feedback is supplied to either branch.
    if complete and example.gold is not None:
        for arm, (answer, observed, retained) in runtime.items():
            outcomes[arm]["feedback"] = asdict(
                evaluate_layered_feedback(question, (), retained, answer, example.gold)
            )
            outcomes[arm]["raw_retrieved_support_recall"] = evaluate_layered_feedback(
                question, (), observed, Answer(""), example.gold
            ).retrieved_gold_support_recall
    report = {
        "question_id": question.question_id,
        "question": question.text,
        "question_type": example.question_type,
        "arms": outcomes,
        "complete_pair": complete,
        "notice": "Exposed official-train debug; per-question paragraph pool, not fullwiki. "
        "S2G support recall is exact original-sentence alignment, not semantic entailment. "
        "BASE1 and S2G4 have unequal budgets; this is NOT a single-factor ablation.",
    }
    if complete and example.gold is not None:
        report["offline_gold_answers"] = list(example.gold.answers)
    write_json(directory / "report.json", report)
    return report


def execute(examples, client, upstream, output, progress, *, run_id=RUN_ID):
    reports = []
    started = 0
    try:
        for i, example in enumerate(examples):
            if client.block_reason:
                break
            started += 1
            reports.append(
                execute_question(
                    example,
                    client,
                    upstream,
                    output / "questions" / f"{i:03d}",
                    progress,
                    run_id=run_id,
                )
            )
            progress(
                {
                    "kind": "question_complete",
                    "question_id": example.question.question_id,
                    "status": "completed" if reports[-1]["complete_pair"] else "failed",
                    "requests": client.attempts,
                }
            )
            if i == 1:
                write_json(
                    output / "first_two_checkpoint.json",
                    {
                        "protocol_success": all(r["complete_pair"] for r in reports),
                        "continue_by_protocol_not_answer_correctness": True,
                    },
                )
    except Exception:
        client.block_reason = client.block_reason or "execution_failure"
        raise
    finally:
        complete = [r for r in reports if all(r["arms"][a].get("feedback") for a in ARMS)]
        summary = {
            "run_id": run_id,
            "planned": len(examples),
            "started": started,
            "started_without_report": started - len(reports),
            "complete_pairs": len(complete),
            "arms": {},
            "stop_reason": client.block_reason,
            "benchmark_reproduction": False,
            "all_actual_calls_including_interrupted": call_totals(client.calls),
        }
        for arm in ARMS:
            summary["arms"][arm] = {
                "n": len(complete),
                "em": mean(r["arms"][arm]["feedback"]["answer_em"] for r in complete)
                if complete
                else None,
                "f1": mean(r["arms"][arm]["feedback"]["answer_f1"] for r in complete)
                if complete
                else None,
                "all_started_cost": call_totals(
                    [c for r in reports for c in r["arms"][arm]["calls"]]
                ),
            }
        write_json(output / "reports.json", reports)
        write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "manifest", "runs-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    format_options = parser.add_mutually_exclusive_group()
    format_options.add_argument(
        "--json-stages",
        action="store_true",
        help="Authorized v2: JSON mode for judge/extract, plain final answer",
    )
    format_options.add_argument(
        "--schema-stages",
        action="store_true",
        help="v3: strict author output schemas for judge/extract; plain final answer",
    )
    format_options.add_argument(
        "--qwen-capacity",
        action="store_true",
        help="v4: strict schemas with explicit Qwen output caps 768/128/256",
    )
    args = parser.parse_args(argv)
    root, output = args.runs_root.resolve(), args.output.resolve()
    strict_stages = args.schema_stages or args.qwen_capacity
    run_id = (
        CAPACITY_RUN_ID
        if args.qwen_capacity
        else SCHEMA_RUN_ID
        if args.schema_stages
        else JSON_RUN_ID
        if args.json_stages
        else RUN_ID
    )
    claim = root / f"{run_id}.claim.json"
    if output == root or not output.is_relative_to(root) or output.exists():
        raise ValueError("new output strictly inside runs required")
    if args.allow_network and (output.name != run_id or claim.exists()):
        raise ValueError("one-use author pilot already claimed or wrong output identity")
    examples, data = load_debug(args.manifest)
    historical_roots = HISTORY_ROOTS
    if args.json_stages or strict_stages:
        historical_roots = (*historical_roots, f"{RUN_ID}/final_budget.json")
    if strict_stages:
        historical_roots = (*historical_roots, f"{JSON_RUN_ID}/final_budget.json")
    if args.qwen_capacity:
        historical_roots = (*historical_roots, f"{SCHEMA_RUN_ID}/final_budget.json")
    history = reconcile_history(root, reviewed_extra_ledgers=historical_roots)
    subcap = min(5.0, 50.0 - history["prior_reserved_cny"])
    if subcap <= 0:
        raise ValueError("project budget exhausted")
    project = Path(__file__).resolve().parents[3]
    snapshot = source_snapshot(project)
    probe = S2GAuthorAPI(
        args.upstream, None, lambda q, k: (), gap_profile="paper_k1", remove_repeat_docs=True
    )
    caps = QWEN_OUTPUT_CAPS if args.qwen_capacity else AUTHOR_OUTPUT_CAPS
    generation_profile = (
        "qwen_capacity_v4_judge768_extract128_answer256"
        if args.qwen_capacity
        else "author_per_stage_limits_via_complete_author"
    )
    author_provenance = probe.provenance
    author_provenance["backend_generation_settings"] = generation_profile
    plan = {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "git": _git_state(),
        "data": data,
        "author": author_provenance,
        "source_sha256": snapshot["sha256"],
        "model": PILOT_MODEL,
        "actual_backend_generation_settings": generation_profile,
        "author_requested_output_caps": AUTHOR_OUTPUT_CAPS,
        "decoding": {
            "judge_max_tokens": caps["judge"],
            "extract_max_tokens": caps["extract"],
            "answer_max_tokens": caps["answer"],
            "temperature": 0,
            "top_p": 1,
            "enable_thinking": False,
            "json_object_mode": "judge/extract only" if args.json_stages else False,
            "json_schema_mode": "judge/extract only" if strict_stages else False,
        },
        "output_schemas": {
            stage: response_format_for(PROMPT_VERSIONS[stage]) for stage in ("judge", "extract")
        }
        if strict_stages
        else {},
        "arms": ARMS,
        "max_calls": MAX_CALLS,
        "worst_case_calls": len(examples) * 11,
        "subcap_cny": subcap,
        "project_cap_cny": 50,
        "historical_budget": history,
        "price_input_cny_per_million": 0.2,
        "price_output_cny_per_million": 0.8,
        "price_source": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
        "price_checked_date": "2026-09-27",
        "timeout_seconds": 1800,
        "retrieval": "Local positive-score BM25 over per-question complete documents; "
        "NOT author Pyserini/fullwiki. Upstream regex sentence fallback.",
        "no_training_no_memory_updates": True,
        "official_dev_test_used": False,
        "stop_policy": "First transport/protocol failure; no retry; after 2 questions "
        "continue only by protocol success, not correctness. No mid-run prompt edits.",
        "protocol_sidecar": "Original prompts/parser/control preserved; outside loop, "
        "malformed JSON fields or missing Answer/Rationale stops pilot, unscored. "
        "This is stricter than author's silent parsing fallback, not a method gain.",
        "v2_authorization": "User chose agent recommendation after v1's extractor length "
        "failure. New full eight-question protocol; v1 immutable and separately reported."
        if args.json_stages
        else None,
        "v3_decision": "User delegated API compatibility choice. v2 JSON was syntactically "
        "valid but used specific instead of sufficient. Official provider documentation "
        "confirms strict schema support; enforce only author's existing fields. "
        "No correction of past outputs; fresh full eight-question protocol, first error stops."
        if args.schema_stages
        else None,
        "v4_decision": "User authorized next step after v3 truncation. Only raise backend "
        "output capacity to judge768/extract128/answer256; strict schemas and author "
        "prompts/control unchanged. All v1-v3 fees retained. Engineering acceptance, "
        "not a new method, independent test set, or trained-author-model reproduction."
        if args.qwen_capacity
        else None,
        "schema_support_source": "https://help.aliyun.com/zh/model-studio/qwen-structured-output"
        if strict_stages
        else None,
    }
    if plan["worst_case_calls"] > MAX_CALLS:
        raise ValueError("call envelope exceeds cap")
    if not args.allow_network:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        print(f"Plan only; no API/config access: {output / 'plan.json'}", flush=True)
        return 0
    if not args.api_config or not plan["git"]["commit"]:
        raise ValueError("live launch requires local config and Git version")
    settings = read_local_bailian_settings(args.api_config)
    host = urlsplit(settings.base_url).hostname or ""
    if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
        raise ValueError("frozen price profile requires ordinary Beijing endpoint")
    if source_snapshot(project)["sha256"] != snapshot["sha256"]:
        raise ValueError("source changed before launch")
    write_json(claim, {"output": str(output), "plan_sha256": fingerprint(plan), "no_retry": True})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch_plan.json", plan)
    write_json(output / "source_snapshot.json", snapshot)
    write_json(
        output / "process.json",
        {
            "pid": os.getpid(),
            "python": sys.executable,
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "started_utc": datetime.now(UTC).isoformat(),
            "notice": "Only API config PATH is public; credentials are never command arguments.",
        },
    )
    progress = ProgressLog(output)
    progress({"kind": "launch", "pid": os.getpid(), "model": PILOT_MODEL})
    previous, client, run_failed = os.environ.get(KEY_VARIABLE), None, False
    os.environ[KEY_VARIABLE] = settings.api_key
    try:
        client = AuthorBudgetClient(
            LiveChatClient(
                ChatConfig(
                    settings.base_url,
                    PILOT_MODEL,
                    KEY_VARIABLE,
                    max_calls=MAX_CALLS,
                    max_output_tokens=256,
                    output_limit_parameter="max_tokens",
                    enable_thinking=False,
                    temperature=0,
                    top_p=1,
                ),
                output / "api_audit",
                allow_network=True,
            ),
            PriceLimits(budget_cny=subcap, max_elapsed_seconds=1800, max_prompt_bytes=30000),
            output / "request_journal",
        )
        client.json_stages = args.json_stages
        client.schema_stages = strict_stages
        client.qwen_output_caps = args.qwen_capacity
        execute(examples, client, args.upstream, output, progress, run_id=run_id)
        return 1 if client.block_reason else 0
    except Exception as error:
        run_failed = True
        if client is not None:
            client.block_reason = client.block_reason or "runner_failure"
        write_json(
            output / "interrupted.json", {"error_type": type(error).__name__, "no_retry": True}
        )
        progress({"kind": "interrupted", "error_type": type(error).__name__})
        return 1
    finally:
        if client is not None:
            budget = client.report()
            write_json(output / "final_budget.json", budget)
            write_json(
                output / "cumulative_budget.json",
                {
                    "prior": history,
                    "new": budget,
                    "project_cap_cny": 50,
                    "cumulative_reserved_cny": history["prior_reserved_cny"]
                    + budget["reserved_cny"],
                    "notice": "Estimate, not provider invoice; historical unknown remains unknown.",
                },
            )
            progress(
                {
                    "kind": "exit",
                    "requests": client.attempts,
                    "status": "failed" if run_failed or client.block_reason else "completed",
                }
            )
        if previous is None:
            os.environ.pop(KEY_VARIABLE, None)
        else:
            os.environ[KEY_VARIABLE] = previous


if __name__ == "__main__":
    raise SystemExit(main())
