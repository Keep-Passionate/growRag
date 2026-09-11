"""Gold-based evaluation AFTER frozen selection and isolated canonical execution."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .comparison_report import _sum_usage
from .data_protocol import normalize_question
from .hotpot import evaluate_layered_feedback
from .protocol import Action, ExecutionKind, GoldRecord
from .representation_runner import RepresentationRun


def representation_report(run: RepresentationRun, *, gold: GoldRecord | None = None) -> dict:
    if not isinstance(run, RepresentationRun):
        raise TypeError("a completed typed representation run is required")
    question = run.spec.candidates.question
    if gold is not None and (
        not isinstance(gold, GoldRecord) or gold.question_id != question.question_id
    ):
        raise ValueError("gold must belong to this target")
    rows, actual_usage = {}, []
    for outcome in run.outcomes:
        result = outcome.result
        last = result.state.rounds[-1] if result is not None and result.state.rounds else None
        events = result.events if result is not None else ()
        actual_usage.extend(event.usage for event in events)
        feedback = None
        if outcome.status == "completed" and last is not None and gold is not None:
            feedback = asdict(
                evaluate_layered_feedback(
                    question, (), last.reply.evidence or (), last.reply.answer, gold
                )
            )
            if last.reply.evidence is None:
                for key in (
                    "retrieved_gold_support_recall",
                    "new_gold_support",
                    "cited_gold_support_precision",
                    "cited_gold_support_recall",
                    "citation_ids_resolve",
                ):
                    feedback[key] = None
        rows[outcome.route_id] = {
            "status": outcome.status,
            "cache_key": outcome.cache_key,
            "query": last.search_query if last else None,
            "answer": last.reply.answer.text if last else None,
            "executed_action": last.decision.action.value if last else None,
            "evidence": asdict(last.reply)["evidence"] if last else None,
            "feedback": feedback,
            "stop_reason": result.stop_reason if result is not None else "not_executed",
            "usage": _sum_usage(tuple(event.usage for event in events)),
            "rag_calls": sum(event.operation == "rag" for event in events),
            "rewrite_calls": sum(event.operation == "rewrite" for event in events),
        }
    if set(rows) != set(run.spec.route_ids):
        raise ValueError("the outcome table must retain every planned route")
    fresh = rows["FRESH"]
    candidates = [rows[c.candidate_id] for c in run.spec.candidates.candidates]
    pool_complete = fresh["feedback"] is not None and all(
        row["feedback"] is not None for row in candidates
    )
    best_f1 = (
        max(row["feedback"]["answer_f1"] for row in [fresh, *candidates]) if pool_complete else None
    )
    beneficial = (
        any(row["feedback"]["answer_f1"] > fresh["feedback"]["answer_f1"] for row in candidates)
        if pool_complete
        else None
    )
    policies = {}
    for selection in run.selections:
        row = rows[selection.selected_id]
        selected, baseline = row["feedback"], fresh["feedback"]
        comparable = selected is not None and baseline is not None
        selector_usage = () if selection.event is None else (selection.event.usage,)
        actual_usage.extend(selector_usage)
        chosen_result = next(
            item.result for item in run.outcomes if item.route_id == selection.selected_id
        )
        chosen_usage = tuple(event.usage for event in chosen_result.events) if chosen_result else ()
        reuse_selected = selection.selected_id != "FRESH"
        reuse_attempted = (
            reuse_selected
            and chosen_result is not None
            and any(event.operation == "rewrite" for event in chosen_result.events)
        )
        reuse_executed = (
            reuse_selected
            and chosen_result is not None
            and any(step.decision.action is Action.REUSE for step in chosen_result.state.rounds)
        )
        different = (
            normalize_question(row["query"]) != normalize_question(fresh["query"])
            if comparable
            else None
        )
        policy = {
            "selected_id": selection.selected_id,
            "selection_status": selection.status,
            "selector_error": selection.status == "selector_error",
            "reused": reuse_selected,
            "reuse_selected": reuse_selected,
            "reuse_attempted": reuse_attempted,
            "reuse_executed": reuse_executed,
            "selected_path_completed": row["status"] == "completed",
            "paired_available": comparable,
            "answer_f1_delta_vs_fresh": selected["answer_f1"] - baseline["answer_f1"]
            if comparable
            else None,
            "answer_em": selected["answer_em"] if selected else None,
            "rescued_vs_fresh": baseline["answer_em"] == 0 and selected["answer_em"] == 1
            if comparable
            else None,
            "harmed_vs_fresh": baseline["answer_em"] == 1 and selected["answer_em"] == 0
            if comparable
            else None,
            "different_executed_query": different,
            "oracle_regret_f1": best_f1 - selected["answer_f1"]
            if best_f1 is not None and selected
            else None,
            "missed_beneficial_candidate": beneficial and selection.selected_id == "FRESH"
            if pool_complete
            else None,
            "path_usage_from_audit": _sum_usage((*selector_usage, *chosen_usage)),
            "cost_note": "Reconstructed path, not online latency; source building excluded.",
        }
        a = selected["retrieved_gold_support_recall"] if selected else None
        b = baseline["retrieved_gold_support_recall"] if baseline else None
        policy["support_recall_delta_vs_fresh"] = a - b if a is not None and b is not None else None
        policies[selection.representation.value] = policy
    return {
        "schema_version": "growrag-representation-report-v1",
        "question_id": question.question_id,
        "spec_fingerprint": run.spec.fingerprint,
        "execution_kind": run.execution_kind.value,
        "synthetic_demo": run.execution_kind is ExecutionKind.MOCK,
        "notice": "Paired diagnostic only. Same-query answer variation is not rewrite credit. "
        "MOCK outputs must never be reported as model performance.",
        "gold_available": gold is not None,
        "candidate_count": len(candidates),
        "candidate_pool_complete": pool_complete,
        "candidate_pool_has_benefit_vs_fresh": beneficial,
        "offline_candidate_oracle_f1": best_f1,
        "oracle_note": "Post-hoc fixed-pool bound for this realization; never selector input.",
        "outcomes": rows,
        "policies": policies,
        "actual_audit_usage": _sum_usage(tuple(actual_usage)),
        "selector_call_count": sum(selection.event is not None for selection in run.selections),
        "reuse_metrics_note": "reused is a deprecated alias of reuse_selected (selection intent). "
        "reuse_attempted counts a recorded rewrite control attempt, not necessarily a network "
        "request. reuse_executed requires a completed round with a REUSE decision; an unchanged "
        "rewrite falling back to BASE does not qualify.",
        "cost_scope": "Executed selector calls plus attempted canonical route calls. "
        "No double-counting component events; source/representation generation excluded here.",
        "representation_lengths": [
            {
                "candidate_id": candidate.candidate_id,
                "views": {
                    view.kind.value: {"characters": view.char_count, "tokens": None}
                    for view in candidate.bundle.views
                },
            }
            for candidate in run.spec.candidates.candidates
        ],
    }


def save_representation_run(run: RepresentationRun, directory: Path, *, gold=None) -> dict:
    """Exclusive final artifacts. Intermediate choices/outcomes belong to callbacks."""
    report = representation_report(run, gold=gold)
    encoded = {"run.json": run.to_dict(), "report.json": report}
    serialized = {
        name: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        for name, value in encoded.items()
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in serialized.items():
        with (directory / name).open("x", encoding="utf-8") as handle:
            handle.write(content + "\n")
    return report
