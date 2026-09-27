"""Prove an old shared-run claim was never executed before explicitly transferring it.

中文：只继承旧批次已计划但从未触及的题。失败题即使没有答案也不可重跑。
此离线证明依据最终账本、完整报告、exit封口日志及执行文件，不查询或终止进程。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_RUN_NAME = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}_s2g_shared(?:32|500)_v[1-9][0-9]*_[0-9]{4,}_[0-9]{4,}"
)
_ARMS = {"BASE1_AUTHOR_READER", "S2G_AUTHOR_API4"}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in continuation evidence")
        result[key] = value
    return result


def _read(path):
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=_unique), hashlib.sha256(raw).hexdigest()


def _safe_file(directory, name):
    path = directory / name
    resolved = path.resolve(strict=True)
    if resolved.parent != directory or not resolved.is_file():
        raise ValueError("continuation evidence path escapes prior run")
    return resolved


def _event_mentions(value, wanted):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"question_id", "trace_id"} and isinstance(item, str):
                if item in wanted or any(f"/{qid}/" in f"/{item}/" for qid in wanted):
                    return True
            if isinstance(item, (list, dict)) and _event_mentions(item, wanted):
                return True
    elif isinstance(value, list):
        return any(_event_mentions(item, wanted) for item in value)
    return False


def _request_evidence(directory, budget, wanted):
    """核对原始请求意图与结果；未进入最终摘要的请求同样不得重放。"""
    file_hashes, journal, audits = {}, {}, {}
    calls = budget["calls"]
    canonical = {call["trace_id"]: call for call in calls}
    for folder_name in ("request_journal", "api_audit"):
        folder = (directory / folder_name).resolve(strict=True)
        if folder.parent != directory or not folder.is_dir():
            raise ValueError("prior request evidence directory escapes run")
        for path in sorted(folder.iterdir()):
            resolved = path.resolve(strict=True)
            if resolved.parent != folder or not resolved.is_file() or path.suffix != ".json":
                raise ValueError("unknown or unsafe prior request evidence artifact")
            value, digest = _read(resolved)
            file_hashes[f"{folder_name}/{path.name}"] = digest
            if _event_mentions(value, wanted):
                raise ValueError("continuation target appears in prior request evidence")
            if folder_name == "request_journal":
                match = re.fullmatch(r"([0-9]{4,})_(intent|after)\.json", path.name)
                if match is None:
                    raise ValueError("unknown prior journal filename")
                index, kind = int(match[1]), match[2]
                pair = journal.setdefault(index, {})
                if kind in pair:
                    raise ValueError("duplicate prior journal sequence")
                pair[kind] = value
            else:
                trace = value.get("trace_id")
                if trace not in canonical or trace in audits:
                    raise ValueError("unaccounted or duplicate prior raw API audit")
                audits[trace] = value
    if set(journal) != set(range(len(calls))):
        raise ValueError("prior request journal does not match final ledger")
    for index, call in enumerate(calls):
        pair = journal[index]
        if set(pair) != {"intent", "after"}:
            raise ValueError("prior request intent lacks corresponding after record")
        if pair["intent"].get("trace_id") != call["trace_id"]:
            raise ValueError("prior request intent trace does not match final ledger")
        if pair["after"].get("calls") != calls[: index + 1]:
            raise ValueError("prior after ledger does not reconcile request prefix")
        if call.get("api_requests", 1) != 0 and call["trace_id"] not in audits:
            raise ValueError("prior network attempt lacks raw API audit")
    if calls and journal[len(calls) - 1]["after"] != budget:
        raise ValueError("last prior after record differs from final budget")
    digest = hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()
    return {
        "file_count": len(file_hashes),
        "intent_after_pairs": len(journal),
        "raw_audit_count": len(audits),
        "files_sha256": digest,
    }


def verify_unstarted_continuation(
    runs_root,
    prior_run_id,
    *,
    manifest_sha256,
    question_ids,
    generation_profile,
    protocol,
    model,
    _visited=(),
):
    """Return proof only for a closed run's untouched subset; never grants an API retry."""
    if not isinstance(prior_run_id, str) or not _RUN_NAME.fullmatch(prior_run_id):
        raise ValueError("continuation requires an exact safe prior run basename")
    if prior_run_id in _visited or len(_visited) >= 16:
        raise ValueError("continuation ancestry cycle or depth limit")
    root = Path(runs_root).resolve(strict=True)
    directory = (root / prior_run_id).resolve(strict=True)
    if directory.parent != root or not directory.is_dir():
        raise ValueError("prior run escapes runs root")
    wanted = tuple(question_ids)
    if (
        not wanted
        or any(not isinstance(qid, str) or not qid for qid in wanted)
        or len(wanted) != len(set(wanted))
    ):
        raise ValueError("explicit unique continuation question IDs required")
    paths = {
        name: _safe_file(directory, name)
        for name in ("launch_plan.json", "reports.json", "final_budget.json", "events.jsonl")
    }
    launch, launch_sha = _read(paths["launch_plan.json"])
    expected = {
        "manifest_sha256": manifest_sha256,
        "generation_profile": generation_profile,
        "protocol": protocol,
        "model": model,
        "run_id": prior_run_id,
    }
    if any(
        not isinstance(value, str) or not value or launch.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError("continuation prior protocol/manifest/profile/model mismatch")
    old_ids = launch.get("question_ids")
    if (
        not isinstance(old_ids, list)
        or not old_ids
        or len(old_ids) != len(set(old_ids))
        or not set(wanted) <= set(old_ids)
        or type(launch.get("start")) is not int
        or launch["start"] < 0
        or launch.get("count") != len(old_ids)
    ):
        raise ValueError("continuation IDs are not a valid subset of prior frozen plan")
    reports, reports_sha = _read(paths["reports.json"])
    budget, budget_sha = _read(paths["final_budget.json"])
    if not isinstance(reports, list) or not isinstance(budget.get("calls"), list):
        raise ValueError("prior final report/ledger must be complete lists")
    seen = set()
    report_calls = {}
    for report in reports:
        qid = report.get("question_id")
        if qid in wanted:
            raise ValueError("continuation target already has a prior report")
        if qid not in old_ids or qid in seen or set(report.get("arms", {})) != _ARMS:
            raise ValueError("invalid prior question report")
        seen.add(qid)
        for outcome in report["arms"].values():
            for call in outcome.get("calls", []):
                trace = call.get("trace_id")
                if not isinstance(trace, str) or trace in report_calls:
                    raise ValueError("invalid/duplicate prior report call trace")
                report_calls[trace] = call
    canonical = {}
    for call in budget["calls"]:
        trace = call.get("trace_id")
        if not isinstance(trace, str) or trace in canonical:
            raise ValueError("invalid/duplicate prior final call trace")
        pieces = trace.split("/")
        if (
            len(pieces) < 4
            or pieces[0] != prior_run_id
            or pieces[1] not in old_ids
            or pieces[2] not in _ARMS
        ):
            raise ValueError("unattributable prior call prevents continuation proof")
        if pieces[1] in wanted:
            raise ValueError("continuation target already has a prior call")
        canonical[trace] = call
    if any(
        trace not in canonical or canonical[trace] != call for trace, call in report_calls.items()
    ):
        raise ValueError("prior final ledger does not reconcile all reported attempts")
    request_evidence = _request_evidence(directory, budget, set(wanted))
    events_sha, last_event = hashlib.sha256(), None
    with paths["events.jsonl"].open("rb") as handle:
        for raw in handle:
            events_sha.update(raw)
            if not raw.strip():
                continue
            event = json.loads(raw, object_pairs_hook=_unique)
            if not isinstance(event, dict):
                raise ValueError("invalid prior event record")
            if last_event is not None and last_event.get("kind") == "exit":
                raise ValueError("prior run has events after terminal exit")
            if _event_mentions(event, set(wanted)):
                raise ValueError("continuation target already appears in prior events")
            last_event = event
    if (
        last_event is None
        or last_event.get("kind") != "exit"
        or last_event.get("status") not in {"completed", "failed"}
    ):
        raise ValueError("prior run is not closed by a terminal exit event")
    checked_offsets = []
    for qid in wanted:
        offset = launch["start"] + old_ids.index(qid)
        question_dir = directory / "questions" / f"{offset:04d}"
        if question_dir.exists():
            raise ValueError("continuation target has a prior question directory")
        checked_offsets.append(offset)
    proof = {
        "schema_version": "growrag-proven-unstarted-continuation-v1",
        "prior_run_id": prior_run_id,
        "question_ids": list(wanted),
        "prior_manifest_sha256": manifest_sha256,
        "prior_record_sha256": {
            "launch_plan.json": launch_sha,
            "reports.json": reports_sha,
            "final_budget.json": budget_sha,
            "events.jsonl": events_sha.hexdigest(),
        },
        "checked_global_offsets": checked_offsets,
        "prior_terminal_status": last_event["status"],
        "prior_request_evidence": request_evidence,
        "proof_scope": "Offline closed-log evidence of untouched questions; "
        "never a failed-call retry.",
    }
    proof["validated_prior_runs"] = [prior_run_id]
    if launch.get("continuation_of") is not None:
        ancestor = verify_unstarted_continuation(
            runs_root,
            launch["continuation_of"],
            manifest_sha256=manifest_sha256,
            question_ids=wanted,
            generation_profile=generation_profile,
            protocol=protocol,
            model=model,
            _visited=(*_visited, prior_run_id),
        )
        proof["validated_prior_runs"].extend(ancestor["validated_prior_runs"])
        proof["ancestor_proof"] = ancestor
    return proof
