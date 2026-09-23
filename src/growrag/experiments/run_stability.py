"""Reviewed synthetic gate, then only the untouched four bounded-loop debug items.

中文：不是旧失败请求的重试。先过小型判断检查，再执行预先确定的后缀；
保留全部旧费用和日志。默认只写计划，显式网络参数才允许真实调用。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from growrag import controller
from growrag.controller import APIEvidenceAssessor

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .bounded_amendment import review_amended_unstarted
from .budget import PriceLimits
from .fresh_benchmark import call_totals
from .judge_smoke import evaluate_case, smoke_cases, smoke_manifest
from .pre_pilot import write_json
from .prior_budget import reconcile_history
from .representation_runner import fingerprint
from .run_bounded_system import ARMS, plain, prompt_manifest, run_question, summarize
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient, load_candidates, load_debug

SCHEMA = "growrag-judge-paired-stability-v1"
PRIOR_RUNS = (
    "2026-09-23_bounded_system_8_v1",
    "2026-09-23_bounded_system_unstarted7_v2",
)
EXTRA_ROOTS = (
    "2026-09-20_pre_opportunity_8_v1/final_budget.json",
    *(f"{name}/final_budget.json" for name in PRIOR_RUNS),
)
REVISIONS = {
    "reason": "Explicit protocol amendment after two judge schema failures and identical "
    "Reader requests yielding different answers. No attempted question is retried.",
    "json_object_mode": {"old": False, "new": True},
    "judge_prompt": controller.ASSESS_PROMPT_VERSION,
    "judge_parser": controller.ASSESS_PARSER_VERSION,
    "paired_execution": "Same-question exact Reader and judge inputs share validated results; "
    "rewriters and routers remain independent. Replays are not new API calls.",
    "semantic_gate": "All prespecified synthetic checks must pass before Hotpot calls.",
}
KEY_VARIABLE = "GROWRAG_STABILITY_API_KEY"


def execute(client, examples, views, output: Path) -> dict:
    """No hidden resume. Gold stays downstream of all runtime arm decisions."""
    smoke, reports, attempted = [], [], 0
    (output / "smoke").mkdir(exist_ok=False)
    try:
        for index, case in enumerate(smoke_cases()):
            assessor = APIEvidenceAssessor(client)
            start = len(client.calls)
            row = {"case_id": case.case_id, "status": "started"}
            try:
                result = assessor(case.state, case.reply)
                row.update(
                    status="completed",
                    assessment=plain(assessor.latest),
                    response=plain(result),
                    evaluation=evaluate_case(case, assessor.latest),
                )
            except Exception as error:
                # Only safe type names; neither secret nor arbitrary HTTP text.
                row.update(status="failed", error_type=type(error).__name__)
                client.block_reason = client.block_reason or "synthetic_component_failure"
            row.update(records=plain(assessor.records), calls=list(client.calls[start:]))
            smoke.append(row)
            write_json(output / "smoke" / f"{index:03d}.json", row)
            print(f"Synthetic check {index + 1}: {row['status']}", flush=True)
            if client.block_reason:
                break
        passed = len(smoke) == len(smoke_cases()) and all(
            row.get("evaluation", {}).get("semantic_pass") is True for row in smoke
        )
        gate = {
            "passed": passed,
            "cases": smoke,
            "cost": call_totals([call for row in smoke for call in row["calls"]]),
            "notice": "Prespecified basic smoke checks, NOT a general judge accuracy estimate.",
        }
        write_json(output / "smoke_gate.json", gate)
        if not passed:
            client.block_reason = client.block_reason or "synthetic_semantic_gate_failed"
        if passed and not client.block_reason:
            for index, example in enumerate(examples):
                attempted += 1
                report = run_question(
                    example,
                    views,
                    client,
                    output / "questions" / f"{index:03d}",
                    share_execution=True,
                )
                reports.append(report)
                print(
                    f"Bounded debug {index + 1}: actual requests={client.attempts}, "
                    f"stop={client.block_reason}",
                    flush=True,
                )
                if client.block_reason:
                    break
        return {"smoke_passed": passed, "reported_questions": len(reports)}
    finally:
        write_json(output / "reports.json", reports)
        summary = summarize(reports, planned=len(examples), attempted=attempted)
        summary["protocol"] = SCHEMA
        summary["synthetic_cases_completed"] = sum(r["status"] == "completed" for r in smoke)
        summary["synthetic_cost"] = call_totals([c for r in smoke for c in r["calls"]])
        summary["cost_notice"] += " Synthetic gate calls are separate, included in final_budget."
        write_json(output / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    root, output = args.runs_root.resolve(), args.output.resolve()
    if output == root or not output.is_relative_to(root) or output.exists():
        raise ValueError("output must be a new directory strictly inside runs_root")
    examples, data = load_debug(args.manifest)
    pool, source = load_candidates(args.source_dir)
    examples, amendment, ledgers = review_amended_unstarted(
        tuple(root / name for name in PRIOR_RUNS),
        root,
        examples,
        data=data,
        source=source,
        prompts=prompt_manifest(),
        model=PILOT_MODEL,
        revisions=REVISIONS,
    )
    if len(examples) != 4 or set(ledgers) != set(EXTRA_ROOTS[1:]):
        raise ValueError("the reviewed four-question suffix or ledgers changed")
    history = reconcile_history(root, reviewed_extra_ledgers=EXTRA_ROOTS)
    subcap = min(1.0, 50.0 - history["prior_reserved_cny"])
    if subcap <= 0:
        raise ValueError("no conservative budget remains")
    plan = {
        "schema_version": SCHEMA,
        "date": "2026-09-23",
        "git": _git_state(),
        "data": data,
        "source": source,
        "amendment": amendment,
        "planned_question_ids": [e.question.question_id for e in examples],
        "smoke": smoke_manifest(),
        "arms": ARMS,
        "model": PILOT_MODEL,
        "temperature": 0,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
        "max_output_tokens": 1536,
        "max_api_calls": 150,
        "max_elapsed_seconds": 1800,
        "subcap_cny": subcap,
        "historical_budget": history,
        "prompts": prompt_manifest(),
        "retrieval_top_k": 4,
        "reader_max_evidence": 8,
        "rag_calls_per_path_max": 2,
        "paired_execution": REVISIONS["paired_execution"],
        "scope": "Diagnostic amended protocol; not comparable as one cohort with old runs. "
        "Frozen six PRE candidates; no new cards, training, test exposure, or promotion.",
    }
    if not args.allow_network:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
        print(f"Plan only, zero API calls: {output / 'plan.json'}")
        return 0
    if not args.api_config or not plan["git"]["commit"] or plan["git"]["worktree_dirty"]:
        raise ValueError("live execution requires committed clean code and local API config")
    # Resolve the fixed one-use claim before opening the secret file.
    claim = root / "2026-09-23_judge_paired_stability.claim.json"
    if claim.exists():
        raise FileExistsError("this amended protocol is already claimed; no retry")
    try:
        settings = read_local_bailian_settings(args.api_config)
    except (ValueError, OSError):
        print("Invalid local configuration; no secrets shown or requests sent.")
        return 2
    hostname = urlsplit(settings.base_url).hostname or ""
    if hostname != "dashscope.aliyuncs.com" and not hostname.endswith(
        ".cn-beijing.maas.aliyuncs.com"
    ):
        raise ValueError("frozen price profile requires Beijing endpoint")
    write_json(claim, {"output": str(output), "plan_sha256": fingerprint(plan), "no_retry": True})
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "launch_plan.json", plan)
    write_json(
        output / "memory_snapshot.json",
        {"views": [asdict(c.view) for c in pool], "production_enabled": False, "source": source},
    )
    previous = os.environ.get(KEY_VARIABLE)
    os.environ[KEY_VARIABLE] = settings.api_key
    client = None
    try:
        client = DurableBudgetClient(
            LiveChatClient(
                ChatConfig(
                    settings.base_url,
                    PILOT_MODEL,
                    KEY_VARIABLE,
                    max_calls=150,
                    max_output_tokens=1536,
                    timeout_seconds=45,
                    temperature=0,
                    output_limit_parameter="max_tokens",
                    enable_thinking=False,
                    json_object_mode=True,
                ),
                output / "api_audit",
                allow_network=True,
            ),
            PriceLimits(budget_cny=subcap, max_elapsed_seconds=1800),
            output / "request_journal",
        )
        execute(client, examples, tuple(c.view for c in pool), output)
        return 1 if client.block_reason else 0
    except Exception as error:
        write_json(
            output / "interrupted.json", {"error_type": type(error).__name__, "no_retry": True}
        )
        print("Stopped. All partial audits retained; no retry.")
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
                    "notice": "Declared-price estimates, not invoices; historical unknown remains.",
                },
            )
        if previous is None:
            os.environ.pop(KEY_VARIABLE, None)
        else:
            os.environ[KEY_VARIABLE] = previous


if __name__ == "__main__":
    raise SystemExit(main())
