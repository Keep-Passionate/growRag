"""Offline factual audit of bounded system reports; no clients, prompts or API calls.

Missing scores remain missing. Shared-prefix cost is added only for hypothetical
standalone paths, never charged again in the observed-call ledger. Gold matching
is not a proof that a judge or a cited passage is right/wrong.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

METRICS = ("answer_em", "answer_f1", "retrieved_gold_support_recall")
COST_FIELDS = ("api_requests", "input_tokens", "output_tokens", "estimated_actual_cny")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sum_known(rows, field):
    values = [row.get(field) for row in rows]
    return sum(values) if all(_number(value) for value in values) else None


def _cost(rows):
    return {"recorded_attempts": len(rows), **{key: _sum_known(rows, key) for key in COST_FIELDS}}


def _score(feedback):
    return {key: feedback.get(key) if isinstance(feedback, dict) else None for key in METRICS}


def _audit_root(path: Path) -> Path:
    for parent in (path, *path.parents):
        if (parent / "api_audit").is_dir():
            return parent
    return path


def _reader_identity(event, root: Path):
    """Read only in-scope api_audit JSON, exporting a strict metadata whitelist."""
    result = {
        "request_sha256": None,
        "model": event.get("model"),
        "prompt_version": None,
        "audit_status": "unavailable",
        "call_identity": None,
    }
    raw = event.get("audit_path")
    if not isinstance(raw, str):
        return result
    original = Path(raw)
    choices = (original, root / original, root / "api_audit" / original.name)
    for candidate in choices:
        candidate = candidate.resolve()
        if (
            candidate.suffix != ".json"
            or "api_audit" not in candidate.parts
            or not candidate.is_relative_to(root.resolve())
            or not candidate.is_file()
        ):
            continue
        try:
            audit = _read(candidate)
        except (OSError, ValueError):
            continue
        digest = audit.get("request_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            result["audit_status"] = "invalid_request_hash"
            return result
        # Never export request/response bodies, endpoints, headers, or credentials.
        result.update(
            request_sha256=digest,
            prompt_version=audit.get("prompt_version"),
            audit_status="resolved",
            call_identity=candidate.stem,
        )
        return result
    return result


def _project_arm(name: str, row: dict, root: Path):
    result = row.get("result", {})
    source_rounds = result.get("state", {}).get("rounds", [])
    round_scores = row.get("round_feedback", [])
    rounds = []
    for index, step in enumerate(source_rounds):
        reply, decision = step["reply"], step["decision"]
        readers = [
            event
            for event in reply.get("component_events", [])
            if event.get("operation") == "rag.answer"
        ]
        feedback = round_scores[index] if index < len(round_scores) else None
        rounds.append(
            {
                "round": index + 1,
                "action": decision.get("action"),
                "memory_id": (decision.get("memory") or {}).get("memory_id"),
                "query": step.get("search_query"),
                "raw_answer": reply["answer"].get("text"),
                "judge_sufficient": step.get("feedback", {}).get("sufficient"),
                "judge_useful_gain": step.get("feedback", {}).get("useful_gain"),
                "score": _score(feedback),
                "new_gold_support": feedback.get("new_gold_support") if feedback else None,
                "reader": _reader_identity(readers[-1], root) if readers else None,
            }
        )
    prefix = row.get("shared_prefix")
    calls = row.get("calls", [])
    incremental = _cost(calls) if "calls" in row else None
    path_cost = None
    if incremental:
        own = incremental["estimated_actual_cny"]
        shared = (prefix or {}).get("cost", {}).get("estimated_actual_cny", 0)
        path_cost = own + shared if _number(own) and _number(shared) else None
    scored = row.get("status") == "completed" and isinstance(row.get("feedback"), dict)
    return {
        "arm": name,
        "status": row.get("status", "unknown"),
        "scored": scored,
        "first_action": rounds[0]["action"] if rounds else None,
        "final_action": rounds[-1]["action"] if rounds else None,
        "rounds": rounds,
        "stop_reason": result.get("stop_reason"),
        "final_score": _score(row.get("feedback") if scored else None),
        "released_score": _score(row.get("released_feedback") if scored else None),
        "allowed_by_judge": row.get("answer_release", {}).get("allowed_by_judge"),
        "issued_reuse_route_count": sum(
            record.get("action") == "REUSE" for record in row.get("route_records", [])
        ),
        "issued_memory_ids": [
            record.get("selected_memory_id")
            for record in row.get("route_records", [])
            if record.get("action") == "REUSE"
        ],
        "executed_reuse_round_count": sum(step["action"] == "REUSE" for step in rounds),
        "incremental_cost": incremental,
        "shared_prefix": {
            key: prefix.get(key)
            for key in ("source_arm", "state_fingerprint", "cost", "actually_collected_once")
        }
        if prefix
        else None,
        "shadow_standalone_estimated_cny": path_cost,
    }


def _paired(questions, arm, baseline):
    pairs = []
    for question in questions:
        left, right = question["arms"].get(arm), question["arms"].get(baseline)
        if left and right and left["scored"] and right["scored"]:
            pairs.append((question["question_id"], left["final_score"], right["final_score"]))
    result = {"baseline": baseline, "scored_intersection_n": len(pairs)}
    for metric in METRICS:
        values = [
            (qid, a[metric], b[metric])
            for qid, a, b in pairs
            if _number(a[metric]) and _number(b[metric])
        ]
        result[metric] = {
            "n": len(values),
            "mean_delta": mean(a - b for _, a, b in values) if values else None,
            "improved_ids": [qid for qid, a, b in values if a > b],
            "equal_ids": [qid for qid, a, b in values if a == b],
            "worse_ids": [qid for qid, a, b in values if a < b],
        }
    em = [(qid, a["answer_em"], b["answer_em"]) for qid, a, b in pairs]
    result["rescues_em_0_to_1"] = [qid for qid, a, b in em if a == 1 and b == 0]
    result["harms_em_1_to_0"] = [qid for qid, a, b in em if a == 0 and b == 1]
    return result


def analyze(reports_with_roots, *, sources=(), budgets=()):
    """Pure report projection, plus read-only whitelisted transport hash lookup."""
    questions, calls, seen_questions, seen_calls = [], [], set(), {}
    for report, root in reports_with_roots:
        qid = report["question"]["question_id"]
        if qid in seen_questions:
            raise ValueError("duplicate question reports: analyze separate attempts separately")
        seen_questions.add(qid)
        questions.append(
            {
                "question_id": qid,
                "question": report["question"]["text"],
                "arms": {
                    name: _project_arm(name, row, root) for name, row in report["arms"].items()
                },
            }
        )
        for row in report["arms"].values():
            for call in row.get("calls", []):
                # Identical requests made twice still cost twice: dedup by CALL identity,
                # never by the request_sha256 body used for reader-output comparison.
                key = call.get("audit_path") or call.get("trace_id")
                if not key:
                    raise ValueError("call metadata lacks a unique audit_path/trace_id")
                projection = {field: call.get(field) for field in COST_FIELDS}
                if key in seen_calls and seen_calls[key] != projection:
                    raise ValueError("inconsistent duplicated call cost")
                if key not in seen_calls:
                    seen_calls[key] = projection
                    calls.append(projection)
    arms = {}
    names = sorted({name for q in questions for name in q["arms"]})
    for name in names:
        rows = [(q["question_id"], q["arms"][name]) for q in questions if name in q["arms"]]
        scored = [(qid, row) for qid, row in rows if row["scored"]]
        module = {
            "initial_insufficient_with_second_round_n": 0,
            "second_round_added_gold_support_ids": [],
            "answer_f1_improved_ids": [],
            "answer_f1_equal_ids": [],
            "answer_f1_worse_ids": [],
            "transition_unscored_ids": [],
            "gold_matching_raw_answer_withheld_ids": [],
            "sufficient_signal_with_gold_nonmatching_raw_answer_ids": [],
        }
        for qid, row in rows:
            rounds = row["rounds"]
            if len(rounds) > 1 and rounds[0]["judge_sufficient"] is False:
                module["initial_insufficient_with_second_round_n"] += 1
                first, last = rounds[0]["score"], rounds[-1]["score"]
                a, b = first["answer_f1"], last["answer_f1"]
                if _number(a) and _number(b):
                    suffix = "improved" if b > a else "worse" if b < a else "equal"
                    module[f"answer_f1_{suffix}_ids"].append(qid)
                else:
                    module["transition_unscored_ids"].append(qid)
                if rounds[-1]["new_gold_support"]:
                    module["second_round_added_gold_support_ids"].append(qid)
            if row["scored"] and rounds:
                em = row["final_score"]["answer_em"]
                if em == 1 and row["allowed_by_judge"] is False:
                    module["gold_matching_raw_answer_withheld_ids"].append(qid)
                if em == 0 and rounds[-1]["judge_sufficient"] is True:
                    module["sufficient_signal_with_gold_nonmatching_raw_answer_ids"].append(qid)
        means = {}
        for metric in METRICS:
            values = [
                r["final_score"][metric] for _, r in scored if _number(r["final_score"][metric])
            ]
            means[metric] = {"n": len(values), "mean": mean(values) if values else None}
        arms[name] = {
            "reported_n": len(rows),
            "scored_n": len(scored),
            "scores": means,
            "attempted_question_n": sum(row["status"] != "not_executed" for _, row in rows),
            "status_counts": dict(Counter(row["status"] for _, row in rows)),
            "stop_counts": dict(Counter(row["stop_reason"] for _, row in rows)),
            "issued_reuse_route_count": sum(row["issued_reuse_route_count"] for _, row in rows),
            "issued_reuse_question_ids": [
                qid for qid, row in rows if row["issued_reuse_route_count"]
            ],
            "issued_reuse_question_rate_among_attempted": (
                sum(bool(row["issued_reuse_route_count"]) for _, row in rows)
                / sum(row["status"] != "not_executed" for _, row in rows)
                if any(row["status"] != "not_executed" for _, row in rows)
                else None
            ),
            "executed_reuse_round_count": sum(row["executed_reuse_round_count"] for _, row in rows),
            "modules": module,
            "comparisons": {
                baseline: _paired(questions, name, baseline)
                for baseline in ("BASE1", "ADAPTIVE_NO_MEMORY2")
                if baseline != name
            },
        }
    groups = defaultdict(list)
    for question in questions:
        for name, row in question["arms"].items():
            for step in row["rounds"]:
                reader = step["reader"]
                if reader and reader["request_sha256"]:
                    key = (
                        question["question_id"],
                        reader["request_sha256"],
                        reader["model"],
                        reader["prompt_version"],
                    )
                    groups[key].append(
                        {
                            "arm": name,
                            "round": step["round"],
                            "raw_answer": step["raw_answer"],
                            "call_identity": reader["call_identity"],
                        }
                    )
    divergent = [
        {
            "question_id": key[0],
            "request_sha256": key[1],
            "model": key[2],
            "prompt_version": key[3],
            "occurrences": values,
        }
        for key, values in groups.items()
        if len({value["raw_answer"] for value in values}) > 1
    ]
    return {
        "schema_version": "growrag-bounded-offline-analysis-v1",
        "sources": list(sources),
        "reported_question_count": len(questions),
        "arms": arms,
        "questions": questions,
        "observed_unique_reported_call_cost": _cost(calls),
        "authoritative_final_budget_totals": {
            field: _sum_known(budgets, field) if budgets else None
            for field in (*COST_FIELDS, "reserved_cny")
        },
        "same_reader_request_different_answers": divergent,
        "notice": "Descriptive development audit only. Missing scores are not zeros. "
        "Gold answer mismatch is not proof of a judge error. Shared-prefix shadow "
        "path costs are NOT added to actual observed-call costs. Differences between "
        "adaptive arms are total-policy contrasts, not isolated causal memory effects.",
    }


def load_inputs(paths):
    reports, sources, budgets = [], [], []
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_dir():
            required = [
                path / name for name in ("reports.json", "summary.json", "final_budget.json")
            ]
            if not all(item.is_file() for item in required):
                raise ValueError("run directory must have reports, summary and final_budget")
            loaded, summary, budget = (_read(item) for item in required)
            budgets.append(budget)
            source = {
                "path": str(path),
                "input_type": "finalized_run",
                **{
                    key: summary.get(key)
                    for key in ("planned", "started", "reported", "complete", "not_started")
                },
                "block_reason": budget.get("block_reason"),
            }
        else:
            loaded = _read(path)
            source = {
                "path": str(path),
                "input_type": "report_files",
                "completion_and_budget": "not established from report files alone",
            }
        items = loaded if isinstance(loaded, list) else [loaded]
        if any(
            not isinstance(item, dict) or "question" not in item or "arms" not in item
            for item in items
        ):
            raise ValueError("expected bounded question report(s)")
        root = _audit_root(path if path.is_dir() else path.parent)
        reports.extend((item, root) for item in items)
        sources.append(source)
    return reports, sources, budgets


def markdown(data):
    lines = ["# Bounded system factual audit", "", data["notice"], "", "## Coverage", ""]
    lines.extend(f"- {source}" for source in data["sources"])
    lines.extend(
        [
            "",
            "## Actual costs (shared prefix counted once)",
            "",
            f"Reported calls: {data['observed_unique_reported_call_cost']}",
            "",
            f"Final budgets: {data['authoritative_final_budget_totals']}",
            "",
            "## Arm scores (each metric has its own observed denominator)",
            "",
        ]
    )
    for name, row in data["arms"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"Statuses: {row['status_counts']}",
                "",
                f"Scores: {row['scores']}",
                "",
                f"Modules: {row['modules']}",
                "",
            ]
        )
        lines.extend(
            f"- vs {baseline}: {comparison}" for baseline, comparison in row["comparisons"].items()
        )
        lines.append("")
    lines.extend(
        [
            "## Same Reader request, different raw answers",
            "",
            json.dumps(data["same_reader_request_different_answers"], ensure_ascii=False, indent=2),
            "",
            "## Question-level execution",
            "",
        ]
    )
    for question in data["questions"]:
        lines.extend([f"### {question['question_id']}", "", question["question"], ""])
        for name, row in question["arms"].items():
            lines.extend(
                [
                    f"#### {name}: {row['status']}",
                    "",
                    f"Stop: {row['stop_reason']}; score: {row['final_score']}",
                    "",
                    f"Cost: {row['incremental_cost']}; standalone shadow: "
                    f"{row['shadow_standalone_estimated_cny']}",
                    "",
                ]
            )
            for step in row["rounds"]:
                lines.extend(
                    [
                        f"- Round {step['round']} {step['action']}, memory={step['memory_id']}",
                        f"  - Query: {step['query']}",
                        f"  - Raw answer: {step['raw_answer']}",
                        f"  - Judge sufficient: {step['judge_sufficient']}; score: {step['score']}",
                    ]
                )
            lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    reports, sources, budgets = load_inputs(args.inputs)
    data = analyze(reports, sources=sources, budgets=budgets)
    # Exclusive output directory: never overwrite source reports or prior audits.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "analysis.md").write_text(markdown(data), encoding="utf-8")
    print(f"Offline analysis written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
