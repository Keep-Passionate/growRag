"""Descriptive paired summaries from already sealed/scored evaluation records.

只处理调用者给定的评分与执行记录，不读取文件、标签、模型或网络。失败不补零；
各方法自己的可评分集合、全部方法共同集合、两两共同集合分别报告，避免混分母。
这里没有显著性检验或因果推断，也不把一次选中但未执行的算子算作真实复用。
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean

from .representation_runner import fingerprint
from .run_operator_study import ARMS

METRICS = ("answer_em", "answer_f1", "raw_support_recall", "visible_support_recall")
STATUSES = {"completed", "failed", "failure_induced_unstarted"}
USAGE_KEYS = (
    "api_requests",
    "input_tokens",
    "output_tokens",
    "estimated_actual_cny",
    "reserved_cny",
)


def _number(value, *, bounded=False, integer=False):
    if value is None:
        return None
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (bounded and value > 1)
        or (integer and type(value) is not int)
    ):
        raise ValueError("invalid nonnegative finite metric or usage")
    return value


def _usage(calls):
    result = {"recorded_calls": len(calls)}
    for key in USAGE_KEYS:
        values = [_number(call.get(key), integer=key in USAGE_KEYS[:3]) for call in calls]
        known = [value for value in values if value is not None]
        missing = len(values) - len(known)
        result[key] = {
            "total": None if missing else sum(known),
            "known_subtotal": sum(known),
            "unknown_calls": missing,
        }
    return result


def _estimate(values):
    return {"n": len(values), "mean": mean(values) if values else None}


def _denominator(ids):
    return {"n": len(ids), "question_ids": ids, "question_ids_sha256": fingerprint(ids)}


def _execution(arm_report):
    """Use completed search events, never merely proposed/chosen actions.

    失败发生在episode返回之前时，该对象可能为None。此时即便之前已有检索，
    仅凭这里的输入也不能知道次数或复用情况，返回未知而不是假装0次。
    """
    episode = arm_report.get("episode")
    if episode is None:
        return None
    if type(episode) is not dict:
        raise ValueError("episode must be a recorded object or unknown")
    searches, proposals = episode.get("searches"), episode.get("proposals")
    if type(searches) is not list or type(proposals) is not list:
        raise ValueError("episode lacks searches/proposals")
    if any(type(proposal) is not dict for proposal in proposals):
        raise ValueError("proposal must be an object")
    reuse_queries = 0
    executed_reuse = set()
    for event in searches:
        if type(event) is not dict or type(event.get("step")) is not int:
            raise ValueError("search event lacks a valid proposal step")
        step = event["step"]
        if not 0 <= step <= len(proposals):
            raise ValueError("search references an absent proposal")
        if step and proposals[step - 1].get("origin") == "reuse":
            if type(proposals[step - 1].get("spec")) is not dict:
                raise ValueError("executed reuse has no operator specification")
            reuse_queries += 1
            executed_reuse.add(step)
    return {
        "retrievals": len(searches),
        "reuse_queries": reuse_queries,
        "reuse_actions": len(executed_reuse),
        "reused": bool(executed_reuse),
        "plan_rejected": episode.get("stop_reason") == "plan_rejected",
    }


def summarize_evaluation(reports, feedback, *, arms=ARMS):
    """Summarize a caller-verified set; the scorer, not this function, enforces 500.

    Small synthetic sets are permitted for unit tests. No path, dataset or hidden
    label is loaded, and the supplied objects are never modified.
    """
    arms = tuple(arms)
    if arms != ARMS or type(reports) is not list or not reports:
        raise ValueError("nonempty records and all seven ordered arms required")
    ids = [report.get("question_id") for report in reports]
    if (
        any(type(qid) is not str or not qid for qid in ids)
        or len(set(ids)) != len(ids)
        or type(feedback) is not dict
        or set(feedback) != set(ids)
    ):
        raise ValueError("unique questions and exact feedback coverage required")
    indexed, traces = {}, set()
    for report in reports:
        qid = report["question_id"]
        if type(report.get("arms")) is not dict or set(report["arms"]) != set(arms):
            raise ValueError("execution must explicitly include every arm terminal status")
        if type(feedback[qid]) is not dict or set(feedback[qid]) != set(arms):
            raise ValueError("feedback must explicitly include every arm")
        indexed[qid] = report["arms"]
        for arm in arms:
            row, scores = report["arms"][arm], feedback[qid][arm]
            if type(row) is not dict or row.get("status") not in STATUSES:
                raise ValueError("invalid terminal execution status")
            if type(scores) is not dict or not set(METRICS) <= set(scores):
                raise ValueError("four explicit metrics required, including unknowns")
            if "status" in scores and scores["status"] != row["status"]:
                raise ValueError("feedback and execution statuses disagree")
            for key in METRICS:
                value = _number(scores[key], bounded=True)
                if key == "answer_em" and value not in (None, 0, 1):
                    raise ValueError("EM must be binary or unknown")
                if row["status"] != "completed" and value is not None:
                    raise ValueError("failed or unstarted execution cannot have a quality score")
                if scores.get("annotation_status") == "invalid" and value is not None:
                    raise ValueError("invalid annotation cannot have a quality score")
            if type(row.get("calls")) is not list:
                raise ValueError("request accounting must be explicit")
            if row["status"] == "failure_induced_unstarted" and (
                row["calls"] or row.get("episode") is not None or row.get("reader") is not None
            ):
                raise ValueError("unstarted arm cannot own API calls or predictions")
            if row["status"] == "completed" and (
                type(row.get("reader")) is not dict
                or type(row["reader"].get("answer")) is not str
                or row.get("episode") is None
            ):
                raise ValueError("completed arm requires a reader answer and episode")
            for call in row["calls"]:
                if type(call) is not dict or not isinstance(call.get("trace_id"), str):
                    raise ValueError("request must have a trace identity")
                if not call["trace_id"] or call["trace_id"] in traces:
                    raise ValueError("duplicate or empty request trace; do not double count cost")
                traces.add(call["trace_id"])

    common = {
        key: [qid for qid in ids if all(feedback[qid][arm][key] is not None for arm in arms)]
        for key in METRICS
    }
    by_arm = {}
    for arm in arms:
        rows = [indexed[qid][arm] for qid in ids]
        executions = [_execution(row) for row in rows]
        known_executions = [value for value in executions if value is not None]
        reuse_questions = sum(value["reused"] for value in known_executions)
        statuses = Counter(row["status"] for row in rows)
        attempted = sum(row["status"] != "failure_induced_unstarted" for row in rows)
        by_arm[arm] = {
            "planned_questions": len(ids),
            "attempted_questions": attempted,
            "status_counts": {key: statuses[key] for key in sorted(STATUSES)},
            "completed_empty_answers": sum(
                row["status"] == "completed" and not row["reader"]["answer"].strip() for row in rows
            ),
            "annotation_status_counts": dict(
                Counter(feedback[qid][arm].get("annotation_status", "unknown") for qid in ids)
            ),
            "own_scorable": {
                key: _estimate(
                    [feedback[qid][arm][key] for qid in ids if feedback[qid][arm][key] is not None]
                )
                for key in METRICS
            },
            "all_arms_common_scorable": {
                key: _estimate([feedback[qid][arm][key] for qid in common[key]]) for key in METRICS
            },
            "execution": {
                "known_episode_questions": len(known_executions),
                "unknown_episode_questions": attempted - len(known_executions),
                "unstarted_questions": statuses["failure_induced_unstarted"],
                "retrievals_known_subtotal": sum(value["retrievals"] for value in known_executions),
                "executed_reuse_questions": reuse_questions,
                "reuse_rate_known_episode_denominator": len(known_executions),
                "reuse_rate_known_episodes": (
                    reuse_questions / len(known_executions) if known_executions else None
                ),
                "confirmed_reuse_coverage_of_planned": reuse_questions / len(ids),
                "reuse_queries_known_subtotal": sum(
                    value["reuse_queries"] for value in known_executions
                ),
                "reuse_actions_known_subtotal": sum(
                    value["reuse_actions"] for value in known_executions
                ),
                "plan_rejected_known_count": sum(
                    value["plan_rejected"] for value in known_executions
                ),
                "fallback_reason_counts": dict(
                    Counter(
                        row["memory_context"]["fallback_reason"]
                        for row in rows
                        if type(row.get("memory_context")) is dict
                        and row["memory_context"].get("fallback_reason") is not None
                    )
                ),
            },
            "all_attempted_cost": _usage([call for row in rows for call in row["calls"]]),
        }

    comparisons = [("fresh", "base"), ("static", "base")]
    comparisons.extend(
        (arm, ref) for arm in arms if arm.startswith("memory") for ref in ("fresh", "static")
    )
    paired = {}
    for candidate, reference in comparisons:
        metrics = {}
        for key in METRICS:
            paired_ids = [
                qid
                for qid in ids
                if feedback[qid][candidate][key] is not None
                and feedback[qid][reference][key] is not None
            ]
            pairs = [
                (feedback[qid][candidate][key], feedback[qid][reference][key]) for qid in paired_ids
            ]
            values = [left - right for left, right in pairs]
            metrics[key] = {
                **_estimate(values),
                "denominator": _denominator(paired_ids),
                "positive": sum(value > 0 for value in values),
                "negative": sum(value < 0 for value in values),
                "unchanged": sum(value == 0 for value in values),
            }
            if key == "answer_em":
                repair = sum(left == 1 and right == 0 for left, right in pairs)
                harm = sum(left == 0 and right == 1 for left, right in pairs)
                reference_wrong = sum(right == 0 for _, right in pairs)
                reference_correct = sum(right == 1 for _, right in pairs)
                metrics[key].update(
                    repairs=repair,
                    harms=harm,
                    both_correct=sum(left == right == 1 for left, right in pairs),
                    both_wrong=sum(left == right == 0 for left, right in pairs),
                    reference_wrong_n=reference_wrong,
                    reference_correct_n=reference_correct,
                    repair_rate=repair / reference_wrong if reference_wrong else None,
                    harm_rate=harm / reference_correct if reference_correct else None,
                )
        paired[f"{candidate}_minus_{reference}"] = {
            "candidate": candidate,
            "reference": reference,
            "metrics": metrics,
        }
    return {
        "schema_version": "growrag-operator-evaluation-summary-v1",
        "question_count": len(ids),
        "arms": by_arm,
        "all_arms_common_counts": {key: len(value) for key, value in common.items()},
        "denominators": {
            "all_arms_common": {key: _denominator(value) for key, value in common.items()},
            "own_scorable": {
                arm: {
                    key: _denominator([qid for qid in ids if feedback[qid][arm][key] is not None])
                    for key in METRICS
                }
                for arm in arms
            },
        },
        "paired": paired,
        "all_branches_evaluation_cost": _usage(
            [call for qid in ids for arm in arms for call in indexed[qid][arm]["calls"]]
        ),
        "scope": "Descriptive frozen-evaluation comparisons only. Pair-specific, method-specific "
        "and seven-way common denominators are different. Missing quality/usage is never zero. "
        "Costs include failed evaluation calls, exclude source/calibration costs, and are not "
        "provider invoices. All-branch collection cost is not a deployed single-route cost. "
        "Reuse is counted only for completed search events tied to a historical proposal; "
        "failed episodes without an episode object remain unknown. "
        "No significance or causal claim.",
    }
