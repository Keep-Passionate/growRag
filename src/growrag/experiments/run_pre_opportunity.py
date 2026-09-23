"""Eight frozen PRE debug questions: does any historical action beat fresh rewriting?

中文：这不是训练，也不自动相信卡片。先只看原问题选卡，再独立运行每个候选，
比较是否存在值得选的经验。旧24检查题不构造GoldRecord、不发送模型。
V_RULE_EXAMPLE/V_PLUS_CONDITIONS是新的机械投影视图，不改旧M1–M5定义。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from growrag.experience.cards import ActivationStage
from growrag.experience.query_views import CardMemoryView
from growrag.outer_loop import RetrieverReaderBackend, run_outer_loop
from growrag.query_actions import RewriteDecision
from growrag.query_operators import RewriteForm

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import BudgetedChatClient, PriceLimits, request_input_bytes
from .data_protocol import normalize_question
from .fresh_baselines import BASELINE_SPECS, baseline_generator
from .fresh_benchmark import call_totals
from .hotpot import evaluate_layered_feedback, parse_hotpot_example
from .llm_adapters import APIReader, _execution_kind
from .pre_pilot import PRE_INTENT, _IndexAdapter, write_json
from .prior_budget import reconcile_history
from .protocol import Action, ExecutionKind, RuntimeQuestion
from .representation_runner import (
    CANONICAL_VERSION,
    SELECTOR_VERSION,
    APICanonicalQueryGenerator,
    APIRepresentationSelector,
    fingerprint,
)
from .run_pilot import PILOT_MODEL, _git_state

VIEWS = ("V_RULE_EXAMPLE", "V_PLUS_CONDITIONS")
MAX_CALLS = 150
KEY_VARIABLE = "GROWRAG_PRE_OPPORTUNITY_API_KEY"


@dataclass(frozen=True)
class FrozenCandidate:
    view: CardMemoryView
    source_question: str
    executed_query: str
    source_fingerprint: str


def decode_view(raw: dict) -> CardMemoryView:
    """Typed, validated runtime projection; never deserialize historical answers."""
    return CardMemoryView(
        **{
            **raw,
            "source_query_ids": tuple(raw["source_query_ids"]),
            "source_question_hashes": tuple(raw["source_question_hashes"]),
            "form": RewriteForm(raw["form"]),
            "stage": ActivationStage(raw["stage"]),
        }
    )


def load_candidates(source_dir: Path) -> tuple[tuple[FrozenCandidate, ...], dict]:
    path = source_dir / "source_library.json"
    raw = path.read_bytes()
    library = json.loads(raw)
    if library.get("execution_kind") != "real" or len(library.get("views", [])) != 6:
        raise ValueError("this pilot requires the frozen six real PRE candidate cards")
    sources = {}
    source_hashes = {}
    for path in sorted((source_dir / "sources").glob("*/source_record.json")):
        value = json.loads(path.read_bytes())
        expected = value.pop("source_fingerprint")
        if fingerprint(value) != expected or value.get("execution_kind") != "real":
            raise ValueError("source execution fingerprint mismatch")
        sources[value["question"]["question_id"]] = value
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidates = []
    for view_raw in library["views"]:
        view = decode_view(view_raw)
        view.check_stage(after_retrieval=False, evidence=())
        source = sources[view.source_query_id]
        if len(source["fresh"]["state"]["rounds"]) != 1:
            raise ValueError("a single PRE source execution is required")
        question = source["question"]["text"]
        if view.source_question_hashes != (
            hashlib.sha256(normalize_question(question).encode()).hexdigest(),
        ):
            raise ValueError("source text does not match the card")
        candidates.append(
            FrozenCandidate(
                view,
                question,
                source["fresh"]["state"]["rounds"][0]["search_query"],
                fingerprint(source),
            )
        )
    if len({c.view.memory_id for c in candidates}) != 6:
        raise ValueError("duplicate frozen candidate")
    return tuple(candidates), {
        "library_path": str(source_dir / "source_library.json"),
        "library_sha256": hashlib.sha256(raw).hexdigest(),
        "source_record_sha256": source_hashes,
        "candidate_ids": [c.view.memory_id for c in candidates],
        "lifecycle": "candidate_not_trusted",
    }


def shortlist(question: RuntimeQuestion, pool: tuple[FrozenCandidate, ...]):
    """Same source-query Jaccard idea as existing representation_views; not a gate."""
    wanted = set(normalize_question(question.text).split())
    ranked = []
    for candidate in pool:
        if candidate.view.is_source(question):
            continue
        terms = set(normalize_question(candidate.source_question).split())
        score = len(wanted & terms) / len(wanted | terms)
        if score > 0:
            ranked.append((score, candidate.view.memory_id, candidate))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(
        (f"c{i + 1}", score, candidate) for i, (score, _, candidate) in enumerate(ranked[:3])
    )


def selector_payload(question: RuntimeQuestion, candidates, representation: str) -> dict:
    if representation not in VIEWS:
        raise ValueError("unknown opportunity view")
    projected = []
    for slot, _, candidate in candidates:
        value = json.loads(candidate.view.text)
        text = {
            "procedure": value["body"],
            "example": {
                "source_query": candidate.source_question,
                "executed_query": candidate.executed_query,
            },
        }
        if representation == "V_PLUS_CONDITIONS":
            text["inferred_conditions_not_verified_guarantees"] = value["conditions"]
        projected.append({"candidate_id": slot, "text": json.dumps(text, ensure_ascii=False)})
    return {"original_question": question.text, "candidates": projected}


def load_debug(manifest_path: Path):
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    selected = manifest["selected"]
    ids = selected["debug"]
    if (
        manifest.get("official_split") != "train"
        or manifest.get("synthetic_data") is not False
        or len(ids) != 8
        or len(set(ids)) != 8
        or set(ids) & (set(selected["check"]) | set(manifest["source_expansion_order"]))
    ):
        raise ValueError("only the original eight independent debug IDs are permitted")
    records_path = manifest_path.parent / manifest["selected_records_file"]
    records_raw = records_path.read_bytes()
    if hashlib.sha256(records_raw).hexdigest() != manifest["selected_records_sha256"]:
        raise ValueError("selected dataset bytes changed")
    # The source file contains 288 raw records. Project the eight IDs before
    # constructing examples/gold; never inspect, print or score the 24 check labels.
    selected_records = {r["_id"]: r for r in json.loads(records_raw) if r["_id"] in set(ids)}
    examples = tuple(
        parse_hotpot_example(selected_records[qid], dataset=manifest["input_dataset_label"])
        for qid in ids
    )
    if len({normalize_question(e.question.text) for e in examples}) != len(ids):
        raise ValueError("duplicate normalized debug questions")
    return examples, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "selected_records_sha256": manifest["selected_records_sha256"],
        "question_ids": ids,
        "role": "original_eight_debug_only",
        "check_question_labels_projected": False,
        "official_dev_test_used": False,
    }


class DurableBudgetClient(BudgetedChatClient):
    """Write an intent BEFORE transport; unfinished intent retains its reserve.

    New directories and a one-child authorization claim prevent crash restart
    from silently resending calls. This runner does not implement automatic resume.
    """

    def __init__(self, delegate, limits, directory):
        super().__init__(delegate, limits)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.intent_number = 0

    def complete(self, messages, *, trace_id, prompt_version):
        reserve = self._price(
            request_input_bytes(self.config, messages, prompt_version=prompt_version) + 1024,
            self.config.max_output_tokens,
        )
        intent = self.intent_number
        self.intent_number += 1
        write_json(
            self.directory / f"{intent:04d}_intent.json",
            {
                "trace_id": trace_id,
                "prompt_version": prompt_version,
                "request_fingerprint": fingerprint(messages),
                "potential_reserved_cny": reserve,
                "prior_reserved_cny": self.reserved_cny,
                "status": "pending_no_automatic_retry",
            },
        )
        try:
            return super().complete(messages, trace_id=trace_id, prompt_version=prompt_version)
        finally:
            write_json(self.directory / f"{intent:04d}_after.json", self.report())


def run_question(example, pool, client, directory: Path, *, allow_real: bool = False) -> dict:
    """Runtime uses only q, candidates and actual evidence; gold scoring is LAST."""
    kind = _execution_kind(client)
    if kind is ExecutionKind.REAL and not allow_real:
        raise ValueError("real opportunity execution requires explicit opt-in")
    directory.mkdir(parents=True, exist_ok=False)
    question = example.question
    candidates = shortlist(question, pool)
    candidate_map = {slot: candidate for slot, _, candidate in candidates}
    planned = ("BASE", "FRESH", "RRR_KEYWORDS", *candidate_map)
    write_json(
        directory / "candidates.json",
        {
            "original_question": asdict(question),
            "candidates": [
                {
                    "slot": slot,
                    "score": score,
                    "memory_id": c.view.memory_id,
                    "source_fingerprint": c.source_fingerprint,
                }
                for slot, score, c in candidates
            ],
        },
    )
    selections = {}
    for name in VIEWS:
        payload = selector_payload(question, candidates, name)
        write_json(directory / f"{name}_input.json", payload)
        start = len(client.calls)
        response, error_type = None, None
        status = "not_executed_after_error" if client.block_reason else "no_candidates"
        if candidates and not client.block_reason:
            try:
                response = APIRepresentationSelector(client).select(payload)
                status = "refused" if response.value == "FRESH" else "selected"
            except Exception as error:
                # Persist partial question provenance, but never arbitrary error text.
                error_type = type(error).__name__
                status = "selector_error"
                client.block_reason = client.block_reason or "selector_output_failure"
        selections[name] = {
            "selected_id": response.value if response else "FRESH",
            "status": status,
            "error_type": error_type,
            "call_cost": call_totals(client.calls[start:]),
            "response": asdict(response) if response else None,
        }
        write_json(directory / f"{name}_selection.json", selections[name])
    # Both choices are durable before any branch outcome exists.
    write_json(directory / "selections.json", selections)
    order = sorted(planned, key=lambda route: fingerprint([42, question.question_id, route]))
    outcomes, runs = {}, {}
    for route in order:
        if client.block_reason:
            break
        backend = RetrieverReaderBackend(
            _IndexAdapter(example.candidate_context, kind), APIReader(client), top_k=4
        )
        if route == "BASE":
            decision, generator = RewriteDecision(), None
        elif route == "RRR_KEYWORDS":
            decision = BASELINE_SPECS[route].decision()
            generator = baseline_generator(client, route)
        else:
            decision = RewriteDecision(
                Action.FRESH if route == "FRESH" else Action.REUSE,
                RewriteForm.PARAPHRASE,
                PRE_INTENT,
                candidate_map[route].view if route != "FRESH" else None,
            )
            generator = APICanonicalQueryGenerator(client)
        start = len(client.calls)
        result = run_outer_loop(
            question,
            backend,
            generator=generator,
            policy=lambda state, fixed=decision: fixed,
            max_rag_calls=1,
        )
        runs[route] = result
        completed = bool(result.state.rounds) and not any(
            e.status == "error" for e in result.events
        )
        last = result.state.rounds[-1] if result.state.rounds else None
        row = {
            "status": "completed" if completed else "failed",
            "planned_route": route,
            "planned_action": decision.action.value,
            "effective_action": last.decision.action.value if last else None,
            "effective_route": "BASE"
            if last and last.decision.action is Action.BASE
            else route
            if last
            else None,
            "fallback_reason": "unchanged_query"
            if last and decision.action is not Action.BASE and last.decision.action is Action.BASE
            else None,
            "query": last.search_query if last else None,
            "answer": last.reply.answer.text if last else None,
            "evidence": [asdict(e) for e in last.reply.evidence or ()] if last else [],
            "stop_reason": result.stop_reason,
            "calls": client.calls[start:],
            "call_cost": call_totals(client.calls[start:]),
            "result": asdict(result),
        }
        outcomes[route] = row
        write_json(directory / f"{route}_execution.json", row)
        if not completed:
            client.block_reason = client.block_reason or "component_output_failure"
            break
    # Explicit missing rows, never silently turn unavailable or failed into zero.
    for route in planned:
        outcomes.setdefault(
            route,
            {
                "status": "not_executed",
                "feedback": None,
                "planned_route": route,
                "effective_action": None,
                "effective_route": None,
                "fallback_reason": None,
            },
        )
    for route, result in runs.items():
        row = outcomes[route]
        last = result.state.rounds[-1] if result.state.rounds else None
        row["feedback"] = (
            asdict(
                evaluate_layered_feedback(
                    question, (), last.reply.evidence or (), last.reply.answer, example.gold
                )
            )
            if row["status"] == "completed" and last and example.gold
            else None
        )
    report = {
        "execution_kind": kind.value,
        "synthetic_not_model_quality": kind is ExecutionKind.MOCK,
        "question_id": question.question_id,
        "question": question.text,
        "selections": selections,
        "outcomes": outcomes,
        "route_order": order,
    }
    write_json(directory / "report.json", report)
    lines = [f"# {question.question_id}", "", question.text, "", "## 事前候选与选择", ""]
    for slot, score, candidate in candidates:
        lines.append(
            f"- {slot}：{candidate.view.memory_id}；来源问题词重叠={score:.4f}。"
            "仅召回，不代表适用。"
        )
    for view, selection in selections.items():
        lines.append(f"- {view} → {selection['selected_id']}；只有原问题和经验投影，未看分支答案。")
    lines.extend(["", "## 实际执行（评分在所有分支执行之后）", ""])
    for route in order:
        row = outcomes[route]
        lines.extend(
            [
                f"### {route}",
                "",
                f"状态：{row['status']}。实际动作：{row.get('effective_action')}；回退原因：{row.get('fallback_reason')}。",
                "",
            ]
        )
        if row["status"] == "not_executed":
            continue
        lines.extend([f"实际查询：{row.get('query')}", "", f"回答：{row.get('answer')}", ""])
        if row.get("feedback"):
            feedback = row["feedback"]
            lines.append(
                f"EM={feedback['answer_em']:.3f}，F1={feedback['answer_f1']:.3f}，支持召回={feedback['retrieved_gold_support_recall']}。"
            )
        lines.extend(["", "实际Reader证据：", ""])
        lines.extend(f"- {e['title']}[{e['sentence_id']}]：{e['text']}" for e in row["evidence"])
        lines.extend(["", f"费用记录：{json.dumps(row['call_cost'], ensure_ascii=False)}", ""])
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def summarize_opportunities(
    reports: list[dict], *, planned_questions: int = 8, started_questions: int | None = None
) -> dict:
    """Paired complete-cohort diagnostics, never treat an unrun branch as zero."""
    complete = [
        r for r in reports if all(row.get("feedback") is not None for row in r["outcomes"].values())
    ]
    started = len(reports) if started_questions is None else started_questions
    if not len(reports) <= started <= planned_questions:
        raise ValueError("invalid planned/started/report question counts")
    summary = {
        "planned_questions": planned_questions,
        "started_questions": started,
        "not_started_questions": planned_questions - started,
        "started_without_report": started - len(reports),
        "questions_with_report": len(reports),
        "complete_questions": len(complete),
        "incomplete_started_questions": started - len(complete),
        "total_not_complete_questions": planned_questions - len(complete),
        "baselines": {},
        "opportunity": {},
        "policies": {},
        "notice": "Eight development questions, not a significance claim. Oracle is post-hoc "
        "best of executed candidates, not a deployable selector. Failure is missing, not wrong.",
    }
    for reference in ("BASE", "FRESH", "RRR_KEYWORDS"):
        feedback = [r["outcomes"][reference]["feedback"] for r in complete]
        summary["baselines"][reference] = {
            "n": len(feedback),
            "em": sum(f["answer_em"] for f in feedback) / len(feedback) if feedback else None,
            "f1": sum(f["answer_f1"] for f in feedback) / len(feedback) if feedback else None,
        }
    for reference in ("FRESH", "RRR_KEYWORDS"):
        with_candidates = [r for r in complete if any(key.startswith("c") for key in r["outcomes"])]
        gains = [
            max(
                row["feedback"]["answer_f1"]
                for key, row in r["outcomes"].items()
                if key.startswith("c")
            )
            - r["outcomes"][reference]["feedback"]["answer_f1"]
            for r in with_candidates
        ]
        summary["opportunity"][reference] = {
            "n_with_candidates": len(gains),
            "any_candidate_route_better_including_fallback": sum(g > 0 for g in gains),
            "all_candidate_routes_worse_including_fallback": sum(g < 0 for g in gains),
            "best_candidate_route_mean_delta_including_fallback": sum(gains) / len(gains)
            if gains
            else None,
        }
        executed_gains = []
        for row in with_candidates:
            actual = [
                v["feedback"]["answer_f1"]
                for k, v in row["outcomes"].items()
                if k.startswith("c") and v.get("effective_action") == "REUSE"
            ]
            if actual:
                executed_gains.append(
                    max(actual) - row["outcomes"][reference]["feedback"]["answer_f1"]
                )
        summary["opportunity"][reference].update(
            n_with_executed_reuse=len(executed_gains),
            any_executed_reuse_better=sum(g > 0 for g in executed_gains),
            best_executed_reuse_mean_delta=sum(executed_gains) / len(executed_gains)
            if executed_gains
            else None,
        )
    for view in VIEWS:
        values = {}
        for reference in ("BASE", "FRESH", "RRR_KEYWORDS"):
            pairs = [
                (
                    r["outcomes"][r["selections"][view]["selected_id"]]["feedback"],
                    r["outcomes"][reference]["feedback"],
                )
                for r in complete
            ]
            values[reference] = {
                "n": len(pairs),
                "f1_delta": sum(a["answer_f1"] - b["answer_f1"] for a, b in pairs) / len(pairs)
                if pairs
                else None,
                "rescues": sum(a["answer_em"] == 1 and b["answer_em"] == 0 for a, b in pairs),
                "harms": sum(a["answer_em"] == 0 and b["answer_em"] == 1 for a, b in pairs),
            }
        summary["policies"][view] = {
            "reuse_selected": sum(
                r["selections"][view]["selected_id"] != "FRESH" for r in complete
            ),
            "reuse_executed": sum(
                r["outcomes"][r["selections"][view]["selected_id"]].get("effective_action")
                == "REUSE"
                for r in complete
            ),
            "selected_memory_fell_back_to_base": sum(
                r["selections"][view]["selected_id"] != "FRESH"
                and r["outcomes"][r["selections"][view]["selected_id"]].get("effective_action")
                == "BASE"
                for r in complete
            ),
            "comparisons": values,
        }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("new output directory required; no automatic retry/resume")
    examples, data = load_debug(args.manifest)
    pool, library = load_candidates(args.source_dir)
    history = reconcile_history(args.runs_root)
    subcap = min(3.0, 50.0 - history["prior_reserved_cny"])
    if subcap <= 0:
        raise ValueError("no conservatively available project budget")
    plan = {
        "schema_version": "growrag-pre-opportunity-plan-v1",
        "data": data,
        "library": library,
        "git": _git_state(),
        "model": PILOT_MODEL,
        "views": VIEWS,
        "source_to_candidate_map": {
            e.question.question_id: [
                {"slot": slot, "score": score, "memory_id": c.view.memory_id}
                for slot, score, c in shortlist(e.question, pool)
            ]
            for e in examples
        },
        "historical_budget": history,
        "new_run_subcap_cny": subcap,
        "max_api_calls": MAX_CALLS,
        "planned_max_api_calls": sum(
            5 + 2 * len(shortlist(e.question, pool)) + (2 if shortlist(e.question, pool) else 0)
            for e in examples
        ),
        "max_elapsed_seconds": 1800,
        "max_output_tokens": 768,
        "top_k": 4,
        "temperature": 0,
        "enable_thinking": False,
        "canonical_prompt_version": CANONICAL_VERSION,
        "selector_prompt_version": SELECTOR_VERSION,
        "rrr": asdict(BASELINE_SPECS["RRR_KEYWORDS"]),
        "pricing_verified_date": "2026-09-20",
        "pricing_source": "https://help.aliyun.com/zh/model-studio/model-pricing",
        "no_automatic_retry": True,
        "check24_used": False,
        "notice": "Eight development questions only; candidate opportunity and condition audit, "
        "not trained policy, not full paper reproduction, not a final strong-FRESH conclusion. "
        "Canonical FRESH remains unchanged; independent RRR_KEYWORDS is also executed.",
    }
    if not args.allow_network:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "plan.json", plan)
        print(
            json.dumps(
                {
                    "plan": str(args.output / "plan.json"),
                    "actual_api_calls": 0,
                    "planned_max_api_calls": plan["planned_max_api_calls"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.api_config is None:
        parser.error("--api-config required for live execution")
    if not plan["git"]["commit"] or plan["git"]["worktree_dirty"]:
        raise ValueError("review and commit code before live execution")
    try:
        settings = read_local_bailian_settings(args.api_config)
    except (OSError, ValueError):
        print("Local API configuration invalid; no request sent, no secrets shown.")
        return 2
    host = urlsplit(settings.base_url).hostname or ""
    if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
        raise ValueError("this frozen price profile requires the Beijing endpoint")
    claim = args.runs_root / "2026-09-20_budget50_pre_opportunity.claim.json"
    write_json(
        claim,
        {
            "output": str(args.output.resolve()),
            "history_fingerprint": fingerprint(history),
            "total_cap_cny": 50,
            "subcap_cny": subcap,
            "no_resume": True,
        },
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "launch_plan.json", plan)
    previous = os.environ.get(KEY_VARIABLE)
    os.environ[KEY_VARIABLE] = settings.api_key
    client, reports, started_questions = None, [], 0
    try:
        client = DurableBudgetClient(
            LiveChatClient(
                ChatConfig(
                    settings.base_url,
                    PILOT_MODEL,
                    KEY_VARIABLE,
                    max_calls=MAX_CALLS,
                    max_output_tokens=768,
                    timeout_seconds=45,
                    output_limit_parameter="max_tokens",
                    enable_thinking=False,
                    temperature=0,
                ),
                args.output / "api_audit",
                allow_network=True,
            ),
            PriceLimits(budget_cny=subcap, max_elapsed_seconds=1800),
            args.output / "request_journal",
        )
        for index, example in enumerate(examples):
            started_questions = index + 1
            reports.append(
                run_question(
                    example,
                    pool,
                    client,
                    args.output / "questions" / f"{index:03d}",
                    allow_real=True,
                )
            )
            if client.block_reason:
                break
        write_json(args.output / "reports.json", reports)
        summary = summarize_opportunities(reports, started_questions=started_questions)
        write_json(args.output / "summary.json", summary)
        (args.output / "summary.md").write_text(
            "# PRE经验机会诊断（开发集）\n\n"
            + "这是来源冻结、候选先定、两种视图先选择、随后穷举候选的配对诊断。\n\n"
            + "FRESH是原canonical提示；RRR_KEYWORDS是另一个无历史对照，不宣称最强已定。\n\n"
            + "```json\n"
            + json.dumps(summary, ensure_ascii=False, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "questions_with_report": len(reports),
                    "api_requests": client.attempts,
                    "block_reason": client.block_reason,
                },
                ensure_ascii=False,
            )
        )
        return 1 if client.block_reason else 0
    except Exception as error:
        write_json(
            args.output / "interrupted.json",
            {
                "error_type": type(error).__name__,
                "notice": "Stopped without retry; inspect safe audit metadata, not raw errors.",
            },
        )
        print("Experiment stopped; completed audits retained. No automatic retry.")
        return 1
    finally:
        if not (args.output / "reports.json").exists():
            write_json(args.output / "reports.json", reports)
        if not (args.output / "summary.json").exists():
            write_json(
                args.output / "summary.json",
                summarize_opportunities(reports, started_questions=started_questions),
            )
        if client is not None:
            budget = client.report()
            write_json(args.output / "final_budget.json", budget)
            write_json(
                args.output / "cumulative_budget.json",
                {
                    "prior": history,
                    "new_calls": budget,
                    "cumulative_reserved_cny": history["prior_reserved_cny"]
                    + budget["reserved_cny"],
                    "authorized_total_cny": 50,
                    "billing_notice": "Unknown old cost remains unknown.",
                },
            )
        if previous is None:
            os.environ.pop(KEY_VARIABLE, None)
        else:
            os.environ[KEY_VARIABLE] = previous


if __name__ == "__main__":
    raise SystemExit(main())
