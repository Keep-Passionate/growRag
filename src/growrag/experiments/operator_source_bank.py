"""Pure source-prefix extraction, not single-operator causal attribution.

先切来源前缀，再读取反馈和聚合。整条 FRESH 轨迹胜 STATIC 只允许保留其中
实际执行过的规格；不把整题收益分配给每个动作。validated 仅表示探索性发布
检查通过，绝不表示跨题可信。此模块无文件读写、API、gold 文本或在线更新。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from math import isfinite
from string import Formatter

from growrag.macro_operators import GoalContract, GroundedBinding, OperatorRegistry, RuntimeState
from growrag.operator_bank import (
    MemoryBuilder,
    OperatorRecord,
    operator_from_dict,
    operator_to_dict,
)

from .operator_model import canonical_operator
from .protocol import Evidence

SOURCE_PROTOCOL = "growrag-operator-study-v1"
METRICS = ("answer_em", "answer_f1", "raw_support_recall")
_ARMS = ("base", "fresh", "static")
_QUESTION_WORDS = frozenset(
    (
        "Which What Where When Who Whose How Is Are Was Were Did Do Does Has Have Had "
        "The A An In At On Of And Or To From For Before After More Less Both Either Same Different"
    ).split()
)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _metric(value: object, name: str) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid source metric: {name}")
    if name == "answer_em" and value not in (0, 1):
        raise ValueError("answer_em must be 0, 1 or unknown")
    return float(value)


def _scores(feedback: Mapping, qid: str) -> dict:
    row = feedback.get(qid, {})
    if not isinstance(row, Mapping):
        raise TypeError("feedback must map each question to arm metrics")
    result = {}
    for arm in _ARMS:
        values = row.get(arm, {})
        if not isinstance(values, Mapping) or set(values) - set(METRICS):
            raise ValueError("feedback accepts only the three preregistered scalar metrics")
        result[arm] = {name: _metric(values.get(name), name) for name in METRICS}
    return result


def _complete_arms(report: dict) -> bool:
    arms = report.get("arms", {})
    return isinstance(arms, dict) and all(
        isinstance(arms.get(arm), dict)
        and arms[arm].get("status") == "completed"
        and isinstance(arms[arm].get("episode"), dict)
        for arm in _ARMS
    )


def _trajectory(episode: dict, qid: str, budget: int):
    """Recover pre-action states from final immutable evidence and recorded search IDs."""
    if episode.get("question_id") != qid:
        raise ValueError("episode question mismatch")
    searches, proposals = episode.get("searches"), episode.get("proposals")
    if type(searches) is not list or not searches or type(proposals) is not list:
        raise ValueError("episode requires search and proposal arrays")
    if any(type(row) is not dict for row in searches):
        raise ValueError("search events must be objects")
    if len(searches) > budget or searches[0].get("step") != 0:
        raise ValueError("invalid episode search budget or initial search")
    pool = tuple(Evidence(**row) for row in episode.get("evidence", []))
    if len({e.evidence_id for e in pool}) != len(pool):
        raise ValueError("duplicate final evidence IDs")
    by_id = {e.evidence_id: e for e in pool}
    previous_step, seen, queries = 0, set(), set()
    for index, search in enumerate(searches):
        step, ids = search.get("step"), search.get("evidence_ids")
        if type(step) is not int or step < previous_step or not 0 <= step <= len(proposals):
            raise ValueError("search refers to invalid proposal step")
        if index and step == 0:
            raise ValueError("multiple initial searches")
        if type(ids) is not list or any(type(key) is not str for key in ids):
            raise ValueError("search IDs must be string arrays")
        if len(set(ids)) != len(ids) or not set(ids) <= set(by_id):
            raise ValueError("search cites duplicate or absent final evidence")
        added = search.get("new_evidence_ids")
        if type(added) is not list or added != [key for key in ids if key not in seen]:
            raise ValueError("search new-evidence IDs disagree with recorded history")
        query = search.get("query")
        if type(query) is not str or not query.strip():
            raise ValueError("invalid executed query")
        normalized = " ".join(query.casefold().split())
        if normalized in queries:
            raise ValueError("duplicate executed query against fixed index")
        queries.add(normalized)
        previous_step = step
        seen.update(ids)
    if seen != set(by_id):
        raise ValueError("unretrieved evidence injected into final pool")
    return searches, proposals, by_id


def _executed_proposal(raw: dict, number: int, question: str, searches, by_id, budget):
    if type(raw) is not dict or set(raw) != {"goal", "spec", "gap", "bindings", "origin", "reason"}:
        raise ValueError("unexpected source proposal fields")
    executed = [row for row in searches if row["step"] == number]
    if not executed:
        return None
    if (
        raw["origin"] != "fresh"
        or type(raw["gap"]) is not dict
        or type(raw["bindings"]) is not list
    ):
        raise ValueError("only fresh executed specifications can become source candidates")
    goal_data = raw["goal"]
    if type(goal_data) is not dict or set(goal_data) != {
        "original_question",
        "intent",
        "constraints",
    }:
        raise ValueError("invalid goal fields")
    if type(goal_data["constraints"]) is not list or goal_data["original_question"] != question:
        raise ValueError("source original question changed")
    goal = GoalContract(question, goal_data["intent"], tuple(goal_data["constraints"]))
    spec = canonical_operator(operator_from_dict(raw["spec"]))
    bindings = []
    for row in raw["bindings"]:
        if (
            type(row) is not dict
            or set(row) != {"name", "value", "evidence_ids"}
            or type(row["evidence_ids"]) is not list
        ):
            raise ValueError("invalid binding fields")
        bindings.append(GroundedBinding(row["name"], row["value"], tuple(row["evidence_ids"])))
    earlier = [row for row in searches if row["step"] < number]
    ids = tuple(dict.fromkeys(key for row in earlier for key in row["evidence_ids"]))
    state = RuntimeState(tuple(by_id[key] for key in ids), tuple(bindings), budget - len(earlier))
    registry = OperatorRegistry()
    registry.register(spec)
    plan = registry.plan(spec.operator_id, spec.version, goal=goal, gap=raw["gap"], state=state)
    if [r.query for r in plan.requests] != [row["query"] for row in executed]:
        raise ValueError("compiled source plan differs from actually executed query batch")
    audit = {
        "proposal_number": number,
        "executed_queries": [r.query for r in plan.requests],
        "pre_action_evidence_ids": list(ids),
        "remaining_retrievals": state.remaining_retrievals,
        "state_fingerprint": _hash(
            {
                "question": question,
                "evidence": [asdict(by_id[key]) for key in ids],
                "earlier_queries": [r["query"] for r in earlier],
            }
        ),
        "individual_operator_effect": "unknown",
        "attribution_scope": "episode_association",
    }
    return spec, audit


def _generalization_flags(spec, question: str) -> list[str]:
    """Conservative lexical heuristic; passing does NOT establish semantic safety."""
    literals = [
        literal for step in spec.steps for literal, _, _, _ in Formatter().parse(step.template)
    ]
    literals.extend(str(c.value) for step in spec.steps for c in step.when if type(c.value) is str)
    text = " ".join(literals).casefold()
    names = set(re.findall(r"\b[A-Z][A-Za-z'-]*\b", question)) - _QUESTION_WORDS
    numbers = set(re.findall(r"\b\d+\b", question))
    flags = []
    if any(
        re.search(rf"(?<!\w){re.escape(word.casefold())}(?!\w)", text) for word in names | numbers
    ):
        flags.append("possible_source_entity_or_number_hardcoding")
    if any(
        not any(field for _, field, _, _ in Formatter().parse(step.template)) for step in spec.steps
    ):
        flags.append("literal_only_query_template")
    return flags


def build_source_bank(
    source_prefix: tuple[str, ...],
    reports: Iterable[dict],
    feedback: Mapping,
    *,
    protocol_id: str = SOURCE_PROTOCOL,
    retrieval_budget: int = 3,
):
    """Return (immutable bank, audit dict); no input mutation or persistent writes.

    Reports outside the prefix are not scored or aggregated. Metadata for every
    supplied report must still declare the source phase and this exact protocol.
    A published candidate is experimentally executable, never automatically trusted.
    """
    if protocol_id != SOURCE_PROTOCOL or type(retrieval_budget) is not int or retrieval_budget != 3:
        raise ValueError("this extractor requires the frozen source protocol and budget 3")
    builder = MemoryBuilder(source_prefix, protocol_id)
    if not isinstance(feedback, Mapping):
        raise TypeError("feedback must be a separate scalar mapping")
    allowed, selected = set(source_prefix), {}
    for report in reports:
        if (
            type(report) is not dict
            or report.get("phase") != "source"
            or report.get("protocol") != protocol_id
        ):
            raise ValueError("only source-phase reports with the frozen protocol are allowed")
        qid = report.get("question_id")
        if type(qid) is not str or not qid:
            raise ValueError("invalid source question identity")
        if qid in allowed:
            if qid in selected:
                raise ValueError("duplicate source question report")
            selected[qid] = report
    audits = {
        "protocol_id": protocol_id,
        "source_prefix_ids": list(source_prefix),
        "metrics": list(METRICS),
        "questions": {},
        "operators": {},
    }
    aggregate = {}
    for qid in source_prefix:
        report = selected.get(qid)
        qa = {"status": "missing_report", "retained": 0}
        audits["questions"][qid] = qa
        if report is None or not _complete_arms(report):
            qa["status"] = "missing_report" if report is None else "incomplete_arms"
            continue
        scores = _scores(feedback, qid)
        delta = {
            name: None
            if scores["fresh"][name] is None or scores["static"][name] is None
            else scores["fresh"][name] - scores["static"][name]
            for name in METRICS
        }
        qa.update(status="no_observed_gain", scores=scores, episode_delta_vs_static=delta)
        wins = [name for name, value in delta.items() if value is not None and value > 0]
        if not wins:
            continue
        episode = report["arms"]["fresh"]["episode"]
        try:
            searches, proposals, by_id = _trajectory(episode, qid, retrieval_budget)
            question = searches[0]["query"]
            for arm in _ARMS:
                record = report["arms"][arm]
                if record.get("question_id") != qid or record.get("arm") != arm:
                    raise ValueError("source arm identity mismatch")
                other = record["episode"]
                control_searches, _, control_evidence = _trajectory(
                    other, qid, 1 if arm == "base" else retrieval_budget
                )
                if (
                    control_searches[0]["query"] != question
                    or control_searches[0]["evidence_ids"] != searches[0]["evidence_ids"]
                    or any(
                        control_evidence[key] != by_id[key] for key in searches[0]["evidence_ids"]
                    )
                ):
                    raise ValueError("source arms did not start from the same question/evidence")
            actual = [
                _executed_proposal(raw, i, question, searches, by_id, retrieval_budget)
                for i, raw in enumerate(proposals, 1)
            ]
        except (ValueError, TypeError, KeyError) as error:
            qa.update(status="invalid_source_trajectory", error=str(error))
            continue
        qa["status"] = "episode_gain_observed"
        for item in actual:
            if item is None:
                continue
            spec, event = item
            flags = _generalization_flags(spec, question)
            if (
                episode.get("stop_reason") == "plan_rejected"
                or episode.get("rejected_error") is not None
            ):
                flags.append("episode_had_rejected_plan")
            if scores["fresh"]["answer_em"] != 1:
                flags.append("source_answer_not_exactly_correct_or_unknown")
            event.update(
                question_id=qid,
                source_episode_metrics=scores,
                episode_delta_vs_static=delta,
                publication_flags=flags,
                publication_eligible=not flags,
            )
            key = spec.operator_id
            if key not in aggregate:
                aggregate[key] = {"spec": spec, "events": []}
            aggregate[key]["events"].append(event)
            qa["retained"] += 1
    namespace = _hash(sorted(source_prefix))[:16]
    for identity, item in sorted(aggregate.items()):
        events = item["events"]
        blockers = sorted(
            {
                flag
                for event in events
                for flag in event["publication_flags"]
                if flag
                in {"possible_source_entity_or_number_hardcoding", "literal_only_query_template"}
            }
        )
        published = not blockers and any(event["publication_eligible"] for event in events)
        ref = f"operator_source/{namespace}/{identity}"
        audits["operators"][ref] = {
            "spec": operator_to_dict(item["spec"]),
            "source_events": events,
            "attribution_scope": "episode_association",
            "individual_operator_effect": "unknown",
            "trusted": False,
            "publication": {
                "eligible": published,
                "blocking_flags": blockers,
                "validation_scope": "source_schema_sanity_only",
                "transfer_evidence": "single_source_unverified"
                if len({e["question_id"] for e in events}) == 1
                else "multiple_source_associations_not_transfer_validation",
                "scope": "exploratory_candidate_evaluation",
                "trusted": False,
                "semantic_drift_absence_proven": False,
            },
        }
        builder.add(
            OperatorRecord(
                item["spec"],
                tuple(sorted({e["question_id"] for e in events})),
                protocol_id,
                ref,
                "validated" if published else "candidate",
                ref + "#publication" if published else None,
            )
        )
    return builder.freeze(), audits
