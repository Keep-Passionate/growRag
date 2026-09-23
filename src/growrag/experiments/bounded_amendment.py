"""Review a stopped experiment chain before a declared protocol amendment.

中文：保留旧失败和费用，仅返回原开发题顺序中从未启动的后缀。新提示词、
JSON 模式和配对共享属于新协议，不把新旧结果冒充同一批可合并的实验。
"""

from __future__ import annotations

from pathlib import Path

from .bounded_continuation import SCHEMA, _read_object, _same_json
from .protocol import RuntimeQuestion


def review_amended_unstarted(
    priors: tuple[Path, ...],
    runs_root: Path,
    examples: tuple | list,
    *,
    data: dict,
    source: dict,
    prompts: dict,
    model: str,
    revisions: dict,
) -> tuple[tuple, dict, tuple[str, ...]]:
    """Return unattempted examples plus immutable-chain provenance; never read gold.

    This function reviews report/call identities and prior file hashes, not cost
    arithmetic. The caller MUST separately use ``reconcile_history`` with every
    returned ledger and the other historical roots before any paid invocation.
    Each prior must be a fully reported stopped prefix of the remaining suffix.
    """
    root = Path(runs_root).resolve()
    if not isinstance(priors, tuple) or not priors:
        raise TypeError("explicit nonempty tuple of prior runs required")
    prior_paths = tuple(Path(path).resolve() for path in priors)
    if len(set(prior_paths)) != len(prior_paths):
        raise ValueError("duplicate prior run")
    if any(path == root or not path.is_relative_to(root) for path in prior_paths):
        raise ValueError("prior runs must remain strictly inside runs_root")
    if not isinstance(examples, (tuple, list)) or len(examples) < 2:
        raise TypeError("full original ordered example sequence required")
    questions = tuple(getattr(example, "question", None) for example in examples)
    if not all(isinstance(question, RuntimeQuestion) for question in questions):
        raise TypeError("examples must expose gold-free RuntimeQuestion values")
    identifiers = [question.question_id for question in questions]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate original question ID")
    if not all(isinstance(value, dict) for value in (data, source, prompts, revisions)):
        raise TypeError("frozen metadata and declared revisions must be objects")
    if data.get("question_ids") != identifiers:
        raise ValueError("examples differ from full frozen data order")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("fixed model required")
    if not isinstance(revisions.get("reason"), str) or not revisions["reason"].strip():
        raise ValueError("protocol amendment requires an explicit reason")

    offset, reviews, ledgers, all_traces = 0, [], [], set()
    frozen_arms = None
    for index, prior in enumerate(prior_paths):
        plan, plan_hash = _read_object(prior / "launch_plan.json", prior)
        if plan.get("schema_version") != SCHEMA:
            raise ValueError("prior plan schema differs")
        for name, expected in (("data", data), ("source", source), ("model", model)):
            if not _same_json(plan.get(name), expected):
                raise ValueError(f"prior {name} differs from frozen configuration")
        old_prompts = plan.get("prompts")
        if not isinstance(old_prompts, dict) or not old_prompts:
            raise ValueError("prior frozen prompts missing")
        arms = plan.get("arms")
        if (
            not isinstance(arms, list)
            or not arms
            or any(not isinstance(arm, str) or not arm for arm in arms)
            or len(set(arms)) != len(arms)
        ):
            raise ValueError("prior arm matrix invalid")
        if frozen_arms is None:
            frozen_arms = arms
        elif arms != frozen_arms:
            raise ValueError("arm order differs across prior chain")
        planned = identifiers[offset:]
        if plan.get("planned_question_ids", identifiers if not index else None) != planned:
            raise ValueError("prior plan is not the untouched suffix in original order")
        if index:
            continuation = plan.get("continuation")
            previous_review = reviews[-1]
            if not isinstance(continuation, dict) or (
                Path(continuation.get("prior_run", "")).resolve() != prior_paths[index - 1]
                or continuation.get("remaining_question_ids") != planned
                or continuation.get("excluded_attempted_question_ids")
                != previous_review["attempted_question_ids"]
                or continuation.get("reviewed_extra_ledger") != ledgers[-1]
                or continuation.get("prior_started") != previous_review["started"]
                or continuation.get("prior_complete") != previous_review["complete"]
                or continuation.get("prior_api_requests") != previous_review["api_requests"]
                or not _same_json(continuation.get("sha256"), previous_review["sha256"])
            ):
                raise ValueError("prior continuation does not authenticate the preceding run")
        elif plan.get("continuation") is not None:
            raise ValueError("first prior must be the original run, not an omitted prefix")

        summary, summary_hash = _read_object(prior / "summary.json", prior)
        started, complete = summary.get("started"), summary.get("complete")
        if (
            type(started) is not int
            or not 0 < started < len(planned)
            or type(complete) is not int
            or not 0 <= complete <= started
            or summary.get("reported") != started
            or summary.get("planned") != len(planned)
            or summary.get("not_started") != len(planned) - started
        ):
            raise ValueError("prior must have a fully reported nonempty stopped prefix")
        question_root = (prior / "questions").resolve()
        if not question_root.is_relative_to(prior) or not question_root.is_dir():
            raise ValueError("prior question directory missing or outside run")
        directories = sorted(path for path in question_root.iterdir() if path.is_dir())
        if [path.name for path in directories] != [f"{i:03d}" for i in range(started)]:
            raise ValueError("question directories differ from exact attempted prefix")
        hashes = {"launch_plan.json": plan_hash, "summary.json": summary_hash}
        report_calls, actual_complete = {}, 0
        for local_index, directory in enumerate(directories):
            report, digest = _read_object(directory / "report.json", prior)
            if report.get("schema_version") != SCHEMA or report.get("execution_kind") != "real":
                raise ValueError("prior question lacks real execution provenance")
            expected = questions[offset + local_index]
            question = report.get("question")
            if not isinstance(question, dict) or (
                question.get("question_id"),
                question.get("text"),
            ) != (expected.question_id, expected.text):
                raise ValueError("question report differs from original attempted prefix")
            matrix = report.get("arms")
            if not isinstance(matrix, dict) or set(matrix) != set(arms):
                raise ValueError("reported arm matrix differs")
            if any(not isinstance(arm, dict) for arm in matrix.values()):
                raise ValueError("reported arm is malformed")
            actual_complete += all(arm.get("feedback") is not None for arm in matrix.values())
            for arm in matrix.values():
                rows = arm.get("calls", [])
                if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                    raise ValueError("reported calls must be explicit records")
                for row in rows:
                    trace = row.get("trace_id")
                    if not isinstance(trace, str) or not trace or trace in report_calls:
                        raise ValueError("missing or duplicate report request identity")
                    report_calls[trace] = row
            hashes[str((directory / "report.json").relative_to(prior))] = digest
        if actual_complete != complete:
            raise ValueError("complete count differs from reports")

        ledger, ledger_hash = _read_object(prior / "final_budget.json", prior)
        rows = ledger.get("calls")
        if (
            not isinstance(rows, list)
            or not rows
            or any(not isinstance(row, dict) for row in rows)
            or type(ledger.get("api_requests")) is not int
            or ledger["api_requests"] != len(rows)
            or not isinstance(ledger.get("block_reason"), str)
            or not ledger["block_reason"].strip()
        ):
            raise ValueError("prior final ledger must preserve the stopped run")
        seen, audit_names = set(), set()
        for row in rows:
            trace = row.get("trace_id")
            if (
                not isinstance(trace, str)
                or not trace
                or trace in seen
                or trace in all_traces
                or trace not in report_calls
                or not _same_json(row, report_calls[trace])
            ):
                raise ValueError("reports do not account for unique identical ledger calls")
            seen.add(trace)
            audit_name = Path(row.get("audit_path", "")).name
            if not audit_name or audit_name in audit_names:
                raise ValueError("missing or duplicate audit filename")
            audit_names.add(audit_name)
            audit, _ = _read_object(prior / "api_audit" / audit_name, prior)
            if audit.get("transport_source") != "live_api" or audit.get("trace_id") != trace:
                raise ValueError("ledger call differs from live audit identity")
            for field in ("status", "api_requests", "input_tokens", "output_tokens"):
                if audit.get(field) != row.get(field):
                    raise ValueError("ledger call differs from live audit usage/status")
            if audit.get("request", {}).get("model") != model:
                raise ValueError("prior request used a different model")
        if seen != set(report_calls):
            raise ValueError("reported call missing from final ledger")
        if {path.name for path in (prior / "api_audit").glob("*.json")} != audit_names:
            raise ValueError("unaccounted audit file in prior run")
        all_traces.update(seen)
        hashes["final_budget.json"] = ledger_hash
        ledgers.append((prior / "final_budget.json").relative_to(root).as_posix())
        reviews.append(
            {
                "prior_run": str(prior),
                "started": started,
                "complete": complete,
                "api_requests": len(rows),
                "block_reason": ledger["block_reason"],
                "attempted_question_ids": planned[:started],
                "sha256": hashes,
                "old_prompts": old_prompts,
                "changed_prompt_keys": sorted(
                    key
                    for key in set(old_prompts) | set(prompts)
                    if not _same_json(old_prompts.get(key), prompts.get(key))
                ),
            }
        )
        offset += started

    return (
        tuple(examples[offset:]),
        {
            "schema_version": "growrag-explicit-protocol-amendment-v1",
            "prior_runs": reviews,
            "excluded_attempted_question_ids": identifiers[:offset],
            "remaining_question_ids": identifiers[offset:],
            "reviewed_extra_ledgers": list(ledgers),
            "new_prompts": prompts,
            "declared_revisions": revisions,
            "notice": "New reviewed protocol on untouched bounded-loop debug suffix only; "
            "not unseen/blind data. No retries, no rewriting old failures or fees, "
            "no pooling old/new protocol scores, no promotion or memory update. "
            "All cumulative root ledgers still require separate reconciliation.",
        },
        tuple(ledgers),
    )
