"""Explicit read-only review before starting a stopped run's untouched suffix.

中文：这不是失败重试器。已经开始的题（包括失败题）全部排除，只允许
原计划顺序中从未开始的连续后缀。旧轨迹与费用保持原样，不改判成功。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protocol import RuntimeQuestion

SCHEMA = "growrag-bounded-system-v1"


def _read_object(path: Path, root: Path) -> tuple[dict, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("reviewed file escapes the prior run")
    raw = resolved.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("reviewed JSON must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _same_json(left: object, right: object) -> bool:
    # A launch plan has JSON lists where a caller may use immutable tuples.
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(
        right, sort_keys=True, ensure_ascii=False
    )


def review_unstarted(
    prior: Path,
    runs_root: Path,
    examples: tuple | list,
    *,
    data: dict,
    source: dict,
    prompts: dict,
    model: str,
) -> tuple[tuple, dict, str]:
    """Validate a completed audit boundary; return only unstarted questions.

    No model calls, dataset gold access, retries, or writes occur here. The caller
    must explicitly request this review and separately include the returned root
    ledger in cumulative-budget reconciliation before authorizing new calls.
    """
    root, previous = Path(runs_root).resolve(), Path(prior).resolve()
    if previous == root or not previous.is_relative_to(root) or not previous.is_dir():
        raise ValueError("prior run must be a directory strictly inside runs_root")
    if not isinstance(examples, (tuple, list)) or len(examples) < 2:
        raise TypeError("review requires the full original ordered example sequence")
    questions = tuple(getattr(example, "question", None) for example in examples)
    if not all(isinstance(question, RuntimeQuestion) for question in questions):
        raise TypeError("examples must expose gold-free RuntimeQuestion values")
    identifiers = [question.question_id for question in questions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate question ID in the original sequence")
    if not all(isinstance(value, dict) for value in (data, source, prompts)):
        raise TypeError("data, source and prompts must be frozen metadata objects")
    if data.get("question_ids") != identifiers:
        raise ValueError("provided examples do not match the full frozen data order")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("review requires a fixed model identifier")

    plan, plan_hash = _read_object(previous / "launch_plan.json", previous)
    if plan.get("schema_version") != SCHEMA:
        raise ValueError("prior launch plan has an incompatible schema")
    for name, expected in (
        ("data", data),
        ("source", source),
        ("prompts", prompts),
        ("model", model),
    ):
        if not _same_json(plan.get(name), expected):
            raise ValueError(f"prior {name} differs from the current frozen configuration")
    summary, summary_hash = _read_object(previous / "summary.json", previous)
    started, reported = summary.get("started"), summary.get("reported")
    if (
        type(started) is not int
        or type(reported) is not int
        or not 0 < started < len(examples)
        or reported != started
        or summary.get("planned") != len(examples)
        or summary.get("not_started") != len(examples) - started
    ):
        raise ValueError("prior run must have a completely reported nonempty attempted prefix")
    complete = summary.get("complete")
    if type(complete) is not int or not 0 <= complete <= started:
        raise ValueError("prior complete-cohort count is invalid")

    question_root = (previous / "questions").resolve()
    if not question_root.is_relative_to(previous) or not question_root.is_dir():
        raise ValueError("prior question audit directory is missing or outside the run")
    directories = sorted(path for path in question_root.iterdir() if path.is_dir())
    expected_names = [f"{i:03d}" for i in range(started)]
    if [path.name for path in directories] != expected_names:
        raise ValueError("question audit directories are not the exact contiguous attempted prefix")
    report_hashes = {}
    trace_ids = []
    actual_complete = 0
    for i, directory in enumerate(directories):
        report, digest = _read_object(directory / "report.json", previous)
        if report.get("schema_version") != SCHEMA or report.get("execution_kind") != "real":
            raise ValueError("prior question report must preserve real execution provenance")
        question = report.get("question")
        if not isinstance(question, dict) or (
            question.get("question_id"),
            question.get("text"),
        ) != (questions[i].question_id, questions[i].text):
            raise ValueError("prior question report does not match the original attempted prefix")
        arms = report.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(plan.get("arms", ())):
            raise ValueError("prior question report has an incomplete or changed arm matrix")
        if not arms or any(not isinstance(arm, dict) for arm in arms.values()):
            raise ValueError("prior arm audit is malformed")
        actual_complete += all(arm.get("feedback") is not None for arm in arms.values())
        for arm in arms.values():
            calls = arm.get("calls", [])
            if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
                raise ValueError("prior arm calls must be explicit audit records")
            trace_ids.extend(call.get("trace_id") for call in calls)
        report_hashes[str((directory / "report.json").relative_to(previous))] = digest
    if actual_complete != complete:
        raise ValueError("prior complete count differs from question reports")

    ledger_path = previous / "final_budget.json"
    ledger, ledger_hash = _read_object(ledger_path, previous)
    calls, count = ledger.get("calls"), ledger.get("api_requests")
    if (
        not isinstance(calls, list)
        or not calls
        or any(not isinstance(call, dict) for call in calls)
        or type(count) is not int
        or count != len(calls)
        or not isinstance(ledger.get("block_reason"), str)
        or not ledger["block_reason"].strip()
    ):
        raise ValueError("prior final budget must preserve a stopped run and all attempted calls")
    ledger_traces = [call.get("trace_id") for call in calls]
    if (
        any(not isinstance(trace, str) or not trace for trace in (*trace_ids, *ledger_traces))
        or len(set(trace_ids)) != len(trace_ids)
        or len(set(ledger_traces)) != len(ledger_traces)
        or set(trace_ids) != set(ledger_traces)
    ):
        raise ValueError(
            "prior question calls do not account for the stopped run's unique requests"
        )

    relative_ledger = ledger_path.resolve().relative_to(root).as_posix()
    remaining = tuple(examples[started:])
    provenance = {
        "schema_version": "growrag-explicit-unstarted-suffix-review-v1",
        "prior_run": str(previous),
        "prior_schema": SCHEMA,
        "prior_started": started,
        "prior_complete": complete,
        "prior_api_requests": count,
        "prior_block_reason": ledger["block_reason"],
        "excluded_attempted_question_ids": identifiers[:started],
        "remaining_question_ids": identifiers[started:],
        "reviewed_extra_ledger": relative_ledger,
        "sha256": {
            "launch_plan.json": plan_hash,
            "summary.json": summary_hash,
            "final_budget.json": ledger_hash,
            **report_hashes,
        },
        "notice": "Explicitly reviewed unstarted suffix only, in original order. No retries "
        "of any attempted question or prior API call. Prior failures stay failed, "
        "prior fees remain charged, and no old result is promoted to success.",
    }
    return remaining, provenance, relative_ledger
