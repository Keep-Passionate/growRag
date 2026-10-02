"""冻结50×5预测后评分；默认仅无标签预检，不调用API、不改变记忆。

主指标按同一道题、同一真实Reader输入，以预注册method顺序中的首次回答
规范化；原始回答和全部实发成本始终保留。不是部署缓存，不声称节约这些
请求的费用。开发题反馈不能用来重跑、更新本批记忆或冒充官方测试成绩。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from growrag.history_library import FrozenHistoryLibrary, card_view
from growrag.operator_loop import run_operator_episode

from . import history_candidate_study as study
from .history_candidate_runtime import METHODS as CANDIDATE_METHODS
from .history_candidate_runtime import CandidateHistoryPlanner
from .history_context import (
    REWRITE_PROMPT_VERSION,
    SELECT_PROMPT_VERSION,
    prepare_history_context,
    prepare_rule_rewrite,
    resolve_history_selection,
)
from .history_rank_diagnostics import rank_cards
from .history_runtime import HistoryContractClient, shortlist_cards
from .history_runtime_v2 import FILL_VERSION, _fill_request, schema_for_v2
from .operator_model import READER_PROMPT, _messages, strict_object, visible_evidence
from .operator_model_v3 import ModelOperatorPlannerV3, planner_prompt
from .operator_schemas import READER_VERSION, validate_wire_shape
from .operator_schemas_v3 import PLANNER_VERSIONS
from .prepare_history_opportunity import OUTPUT as BUNDLE
from .prepare_history_opportunity import load_bundle
from .protocol import Evidence, RuntimeQuestion
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .score_history_calibration import _inside, _money, _read, _require, _sha, _totals
from .score_operator_sources import score_source_report

OUTPUT = "runs/history_candidates50_feedback_v1"
PROBE = "runs/2026-10-02_history_candidate_fillcheck_v1/SUMMARY.json"
COUNT = 50
METRICS = ("answer_em", "answer_f1", "raw_support_recall", "visible_support_recall")
PRIMARY = "controlled_reader"


def _ranking(question, library, method):
    policy, limit = CANDIDATE_METHODS[method]
    if method == "history_legacy3":
        return shortlist_cards(question, library)
    return tuple(
        {"card_id": row["card_id"], "lexical_score": row["score"]}
        for row in rank_cards(question.text, library, policy, limit)
    )


def _http(call, directory, plan, inputs, seen, prefix):
    """来自旧_audit_arm的逐请求合同；没有放宽模型、成本或正常终态要求。"""
    trace = call.get("trace_id", "")
    _require(trace.startswith(prefix) and trace not in seen, "duplicate/wrong API trace")
    seen.add(trace)
    name = hashlib.sha256(trace.encode()).hexdigest() + ".json"
    path = _inside(directory, directory / "api_audit" / name)
    _require(Path(call.get("audit_path", "")).name == name, "ledger audit identity mismatch")
    data = _read(path)
    inputs[str(path)] = _sha(path)
    normal = {
        "trace_id": trace,
        "status": "completed",
        "transport_source": "live_api",
        "network_attempted": True,
        "api_requests": 1,
        "http_status": 200,
        "retry_count": 0,
        "finish_reason": "stop",
        "response_redacted": False,
        "prompt_version": call.get("prompt_version"),
    }
    _require(all(data.get(k) == v for k, v in normal.items()), "abnormal raw HTTP terminal")
    _require(
        call.get("status") == "completed"
        and call.get("api_requests") == 1
        and call.get("validation_status") is None,
        "abnormal ledger request",
    )
    for key in ("input_tokens", "output_tokens"):
        _require(
            type(call.get(key)) is int and call[key] >= 0 and data.get(key) == call[key],
            "ledger/HTTP token count mismatch",
        )
    request, response = data["request"], data["response"]
    expected = {
        "model": plan["model"],
        "max_tokens": plan["max_output_tokens"],
        "stream": False,
        "enable_thinking": False,
        "temperature": 0,
        "top_p": 1,
        "response_format": {"type": "json_object"},
    }
    _require(
        set(request) == {*expected, "messages"}
        and all(request[k] == v for k, v in expected.items()),
        "request model/config drift",
    )
    raw = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
    _require(
        data.get("request_sha256") == hashlib.sha256(raw).hexdigest(),
        "request bytes do not match saved hash",
    )
    _require(
        response.get("model") == plan["model"] == call.get("returned_model"),
        "returned model differs from pinned model",
    )
    usage = response.get("usage", {})
    _require(
        usage.get("prompt_tokens") == call["input_tokens"]
        and usage.get("completion_tokens") == call["output_tokens"],
        "raw usage mismatch",
    )
    choices = response.get("choices", [])
    _require(
        len(choices) == 1 and choices[0].get("finish_reason") == "stop",
        "completion is truncated or not unique",
    )
    value = strict_object(choices[0]["message"]["content"])
    validate_wire_shape(value, schema_for_v2(call["prompt_version"]))
    cost = (call["input_tokens"] * 0.2 + call["output_tokens"] * 0.8) / 1_000_000
    _require(
        math.isclose(_money(call.get("estimated_actual_cny")), cost, abs_tol=1e-10)
        and cost <= _money(call.get("reserved_cny")),
        "cost estimate mismatch",
    )
    return request, value, choices[0]["message"]["content"]


def _contract_replay(report, captured, prefix, library):
    """纯合同审计：固定原HTTP内容/原检索，不调用模型、索引或标签。

    每次规划器请求必须与原messages/trace/version逐字一致。只排除检索耗时
    elapsed_seconds，不能排除查询、轮次、证据顺序、gap、停止或错误字段。
    """
    original = report["episode"]
    method = report["method"]
    planned = captured[:-1]  # Reader已有独立审核，不参与规划器回放。

    class RecordedClient:
        def __init__(self):
            self.position, self.calls, self.block_reason = 0, [], None
            self.config = SimpleNamespace(json_object_mode=True, json_schema_mode=False)

        def complete(self, messages, *, trace_id, prompt_version):
            _require(self.position < len(planned), "replay requested an unrecorded response")
            call, request, content = planned[self.position]
            _require(
                messages == request["messages"]
                and trace_id == call["trace_id"]
                and prompt_version == call["prompt_version"],
                "replay request differs from original HTTP",
            )
            self.position += 1
            self.calls.append(call)
            return SimpleNamespace(content=content)

    client = RecordedClient()
    contract = HistoryContractClient(client)
    question = RuntimeQuestion(**report["question"])
    planner = (
        CandidateHistoryPlanner(
            contract, library, method=method, trace_prefix=prefix.removesuffix("/")
        )
        if method in CANDIDATE_METHODS
        else ModelOperatorPlannerV3(contract, mode="fresh", trace_prefix=prefix.removesuffix("/"))
    )
    evidence = {e["evidence_id"]: Evidence(**e) for e in original["evidence"]}
    search_position = 0

    def recorded_retrieval(query, top_k):
        nonlocal search_position
        _require(search_position < len(original["searches"]), "replay has an unrecorded retrieval")
        search = original["searches"][search_position]
        _require(
            query == search["query"] and top_k == 6,
            "compiled query/top-k differs from original search",
        )
        search_position += 1
        return tuple(evidence[eid] for eid in search["evidence_ids"])

    result = run_operator_episode(
        question,
        recorded_retrieval,
        planner,
        retrieval_budget=1 if method == "base" else 3,
        max_decisions=2,
        top_k=6,
    )
    replayed = json.loads(json.dumps(asdict(result)))
    saved = deepcopy(original)
    for value in (replayed, saved):
        for search in value["searches"]:
            search.pop("elapsed_seconds", None)
    _require(
        replayed == saved,
        "offline contract replay differs from proposals/searches/evidence/stop/error",
    )
    _require(
        client.position == len(planned) and search_position == len(original["searches"]),
        "offline replay left unused responses or retrievals",
    )


def _history_state(report, decision):
    """重建当时已获得的证据，不把后轮证据塞入首轮选择器。"""
    searches = [s for s in report["episode"]["searches"] if s["step"] < decision]
    _require(searches and searches[0]["step"] == 0, "selection lacks initial retrieval")
    evidence = {row["evidence_id"]: Evidence(**row) for row in report["episode"]["evidence"]}
    ordered = dict.fromkeys(eid for search in searches for eid in search["evidence_ids"])
    _require(set(ordered) <= set(evidence), "search cites absent episode evidence")
    return (
        tuple(evidence[eid] for eid in ordered),
        tuple(s["query"] for s in searches),
        3 - len(searches),
    )


def _audit_method(report, directory, plan, events, library, inputs, seen):
    """泛化旧_audit_arm，仅注册新method/容量；额外核验真实排名和选择上下文。

    不删除SELECT日志，不重命名trace，不修改旧审计模块。实际检索归因使用
    execution.decision与search.step，而不是把第几条execution当作检索轮次。
    """
    qid, method = report["question_id"], report["method"]
    _require(method in study.METHODS, "unregistered method")
    prefix = f"{plan['run_id']}/{qid}/{method}/"
    calls = report.get("calls")
    _require(type(calls) is list and calls, "completed method has no API records")
    local = [e for e in events if e.get("question_id") == qid and e.get("arm") == method]
    candidate_events = [e for e in local if e.get("kind") == "history_candidates"]
    cards = {r.card.card_id: r for r in library.records if r.status == "published"}
    choices, used_candidates, reader_sha, selection_contexts = [], [], None, {}
    captured = []
    question = RuntimeQuestion(**report["question"])
    searches = report["episode"]["searches"]
    _require(
        type(searches) is list
        and 1 <= len(searches) <= (1 if method == "base" else 3)
        and searches[0]["step"] == 0
        and searches[0]["query"] == question.text,
        "initial query/retrieval budget drift",
    )
    for number, call in enumerate(calls):
        trace, version = call["trace_id"], call["prompt_version"]
        request, value, content = _http(call, directory, plan, inputs, seen, prefix)
        captured.append((call, request, content))
        if trace == prefix + "reader":
            _require(reader_sha is None and version == READER_VERSION, "duplicate/wrong Reader")
            evidence = tuple(Evidence(**row) for row in report["episode"]["evidence"])
            visible = visible_evidence(evidence)
            payload = {
                "original_question": question.text,
                "evidence": visible,
                "evidence_window_omitted_count": len(evidence) - len(visible),
            }
            _require(
                request["messages"] == _messages(READER_PROMPT, payload),
                "Reader input differs from current evidence/original question",
            )
            _require(
                value == {k: report["reader"][k] for k in ("answer", "supported", "evidence_ids")},
                "saved Reader differs from HTTP response",
            )
            _require(
                report["reader"].get("evidence_windows")
                == [{"evidence_id": row["evidence_id"], **row["window"]} for row in visible]
                and report["reader"].get("visible_evidence_ids")
                == [row["evidence_id"] for row in visible],
                "Reader visible windows mismatch",
            )
            reader_sha = fingerprint(request["messages"])
        elif version == SELECT_PROMPT_VERSION:
            _require(method in CANDIDATE_METHODS, "selection in non-history method")
            suffix = trace.removeprefix(prefix).split("/")
            _require(
                len(suffix) == 3
                and suffix[0] == "history"
                and suffix[2] == "select"
                and suffix[1] in {"1", "2"},
                "invalid selection trace",
            )
            decision = int(suffix[1])
            _require(decision not in selection_contexts, "duplicate selection decision")
            ranking = _ranking(question, library, method)
            evidence, previous, remaining = _history_state(report, decision)
            context = prepare_history_context(
                library,
                question,
                offered_ids=tuple(r["card_id"] for r in ranking),
                representation="examples",
                evidence=evidence,
                previous_queries=previous,
                remaining_retrievals=remaining,
            )
            _require(
                request["messages"] == context.messages(),
                "SELECT input/ranking/policy differs from frozen method",
            )
            matches = [e for e in candidate_events if e.get("decision") == decision]
            _require(
                len(matches) == 1
                and matches[0].get("ranking") == list(ranking)
                and matches[0].get("payload") == context.payload
                and matches[0].get("context_fingerprint") == context.fingerprint,
                "candidate event differs from actual selection context",
            )
            selected = resolve_history_selection(json.dumps(value), context, library)
            selection_contexts[decision] = (context, selected)
            choices.append(value)
            used_candidates.append(
                {
                    "decision": decision,
                    "ranking": list(ranking),
                    "cards": [
                        {"card_id": r["card_id"], "source_kind": cards[r["card_id"]].source_kind}
                        for r in ranking
                    ],
                }
            )
            if selected is not None:
                tail, prompt = (
                    ("fill", FILL_VERSION)
                    if selected.operator_spec
                    else ("rewrite", REWRITE_PROMPT_VERSION)
                )
                _require(
                    number + 1 < len(calls)
                    and calls[number + 1]["trace_id"] == trace.removesuffix("select") + tail
                    and calls[number + 1]["prompt_version"] == prompt,
                    "selected card lacks matching execution request",
                )
        elif version in {FILL_VERSION, REWRITE_PROMPT_VERSION}:
            _require(method in CANDIDATE_METHODS, "history executor in non-history method")
            pieces = trace.removeprefix(prefix).split("/")
            _require(
                len(pieces) == 3 and pieces[0] == "history" and pieces[1] in {"1", "2"},
                "invalid execution trace",
            )
            context, card = selection_contexts.get(int(pieces[1]), (None, None))
            _require(card is not None, "execution has no prior selected card")
            if version == REWRITE_PROMPT_VERSION:
                _require(
                    pieces[2] == "rewrite"
                    and card.operator_spec is None
                    and request["messages"] == prepare_rule_rewrite(library, context, card.card_id),
                    "rule request differs from frozen selected card",
                )
            else:
                from .history_context import _bounded_messages
                from .history_runtime import FILL_PROMPT

                payload = {
                    "original_question": question.text,
                    "evidence": context.payload["evidence"],
                    "previous_queries": context.payload["previous_queries"],
                    "remaining_retrievals": context.payload["remaining_retrievals"],
                    "selected_card": card_view(card, "examples"),
                }
                expected, _, _ = _fill_request(_bounded_messages(FILL_PROMPT, payload))
                _require(
                    pieces[2] == "fill"
                    and card.operator_spec is not None
                    and request["messages"] == expected,
                    "FILL-v2 request/card scope drift",
                )
        else:
            _require(
                method == "fresh" and version == PLANNER_VERSIONS["fresh"],
                "unregistered prompt or planner in wrong method",
            )
            pieces = trace.removeprefix(prefix).split("/")
            _require(
                len(pieces) == 2 and pieces[0] == "plan" and pieces[1] in {"1", "2"},
                "invalid fresh planner trace",
            )
            evidence, previous, remaining = _history_state(report, int(pieces[1]))
            visible = visible_evidence(evidence)
            payload = {
                "mode": "fresh",
                "original_question": question.text,
                "evidence": visible,
                "evidence_window_omitted_count": len(evidence) - len(visible),
                "previous_queries": list(previous),
                "remaining_retrievals": remaining,
                "candidate_specs": [],
                "candidate_shortlist_omitted_count": 0,
            }
            _require(
                request["messages"] == _messages(planner_prompt("fresh"), payload),
                "FRESH prompt/input differs from frozen action-list planner",
            )
    _require(
        reader_sha is not None and calls[-1]["trace_id"] == prefix + "reader",
        "missing/final Reader request mismatch",
    )
    observed = [
        {k: e[k] for k in ("selected_card_id", "reason")}
        for e in local
        if e.get("kind") == "history_selection"
    ]
    executions = [e for e in local if e.get("kind") == "history_execution"]
    _require(
        observed == choices
        and len(candidate_events) == len(used_candidates)
        and [e["selected_card_id"] for e in executions]
        == [v["selected_card_id"] for v in choices if v["selected_card_id"] is not None],
        "selection/execution event differs from HTTP selection",
    )
    _contract_replay(report, captured, prefix, library)
    executed = []
    for e in executions:
        record = cards[e["selected_card_id"]]
        _require(
            e["selected_card_version"] == record.card.version
            and e["action_kind"] == record.card.action_kind
            and selection_contexts.get(e["decision"], (None, None))[1] == record.card
            and e.get("proposal") == report["episode"]["proposals"][e["decision"] - 1],
            "executed card identity/version/decision drift",
        )
        searches = [s for s in report["episode"]["searches"] if s["step"] == e["decision"]]
        executed.append(
            {
                "card_id": record.card.card_id,
                "version": record.card.version,
                "action_kind": record.card.action_kind,
                "source_kind": record.source_kind,
                "source_qids": list(record.source_qids),
                "decision": e["decision"],
                "search_executed": bool(searches),
                "queries": [s["query"] for s in searches],
                "new_evidence_ids": [
                    eid for s in searches for eid in s.get("new_evidence_ids", [])
                ],
            }
        )
    return {
        **_totals(calls),
        "reader_input_sha256": reader_sha,
        "selections": choices,
        "candidate_offers": used_candidates,
        "executed_cards": executed,
        "offline_contract_replay": "passed",
        "retrieval_calls": len(report["episode"]["searches"]),
        "stop_reason": report["episode"].get("stop_reason"),
    }


def collect(project):
    """完整250输出、方法、源、模型与账本未通过，绝不打开标签投影。"""
    project = Path(project).resolve(strict=True)
    inputs = {}

    def tracked(relative):
        path = _inside(project, project / relative)
        inputs[str(path)] = _sha(path)
        return path

    manifest, questions, corpus = load_bundle(project, expected_sha=study.BUNDLE_SHA256)
    ids = manifest["question_ids"]
    _require(
        len(ids) == len(set(ids)) == COUNT and [q.question_id for q in questions] == ids,
        "registered study must contain exactly 50 runtime IDs",
    )
    tracked(f"{BUNDLE}/manifest.json")
    tracked(f"{BUNDLE}/manifest.sha256")
    tracked(f"{BUNDLE}/{manifest['runtime_artifact']['path']}")
    tracked(corpus.relative_to(project))
    for key in ("source_manifest", "background_manifest", "library"):
        tracked(manifest[key]["path"])
    library = FrozenHistoryLibrary.from_json(
        (project / manifest["library"]["path"]).read_text(encoding="utf-8")
    )
    _require(
        library.fingerprint == manifest["library"]["fingerprint"]
        and not set(ids) & set(library.allowed_source_ids)
        and all(not set(r.source_qids) & set(ids) for r in library.records),
        "history library/source role overlaps development",
    )
    snapshot = source_snapshot(project)
    for relative in snapshot["files"]:
        tracked(relative)
    snapshot_sha = snapshot["sha256"]
    runs = project / "runs"
    expected_runs = {f"{study.PREFIX}v1_{s:04d}_{s + n:04d}" for s, n in study.BATCHES}
    for path in runs.rglob("*.claim.json"):
        claim = _read(_inside(project, path))
        if set(ids) & set(claim.get("question_ids", [])):
            _require(
                path.parent == runs and path.name.removesuffix(".claim.json") in expected_runs,
                "development IDs claimed again under another run",
            )
    for path in runs.glob("*/launch_plan.json"):
        if set(ids) & set(_read(_inside(project, path)).get("question_ids", [])):
            _require(path.parent.name in expected_runs, "duplicate question launch")
    probe_path = tracked(PROBE)
    probe = _read(probe_path)
    _require(
        probe.get("status") == "completed" and probe.get("source_sha256") == snapshot_sha,
        "required real FILL probe not completed under this source snapshot",
    )
    _require(study.verify_probe(runs, snapshot_sha) == _sha(probe_path), "probe seal mismatch")
    probe_root = probe_path.parent
    tracked(f"runs/{probe_root.name}.claim.json")
    for path in probe_root.rglob("*"):
        if path.is_file():
            tracked(path.relative_to(project))
    records, seen, all_calls, prior = [], set(), [], {}
    by_id = {q.question_id: asdict(q) for q in questions}
    for start, count in study.BATCHES:
        run_id = f"{study.PREFIX}v1_{start:04d}_{start + count:04d}"
        directory = runs / run_id
        plan = _read(tracked(f"runs/{run_id}/launch_plan.json"))
        claim = _read(tracked(f"runs/{run_id}.claim.json"))
        summary_path = tracked(f"runs/{run_id}/SUMMARY.json")
        summary = _read(summary_path)
        saved = _read(tracked(f"runs/{run_id}/source_snapshot.json"))
        ledger = _read(tracked(f"runs/{run_id}/final_budget.json"))
        events = [
            json.loads(line)
            for line in tracked(f"runs/{run_id}/events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        expected_ids = ids[start : start + count]
        expected = {
            **study.CONFIGURATION,
            "protocol": study.PROTOCOL,
            "phase": "candidate_development",
            "run_id": run_id,
            "start": start,
            "count": count,
            "methods": list(study.METHODS),
            "question_ids": expected_ids,
            "bundle_sha256": study.BUNDLE_SHA256,
            "library_sha256": manifest["library"]["sha256"],
            "library_fingerprint": library.fingerprint,
            "source_sha256": snapshot_sha,
            "gold_loaded": False,
            "memory_updates": False,
            "official_split": "train",
            "probe_summary_sha256": _sha(probe_path),
            "prior_batch_summary_sha256": prior,
        }
        _require(
            all(plan.get(k) == v for k, v in expected.items())
            and claim == {**plan, "plan_sha256": fingerprint(plan)},
            "launch/claim drift",
        )
        _require(
            saved.get("sha256") == snapshot_sha == fingerprint(saved.get("files"))
            and saved == snapshot,
            "saved/current source snapshots differ",
        )
        normal = {
            "protocol": study.PROTOCOL,
            "run_id": run_id,
            "status": "completed",
            "failure_type": None,
            "planned_questions": count,
            "completed_questions": count,
            "source_sha256": snapshot_sha,
            "gold_loaded": False,
            "memory_updated": False,
            "new_operator_induction": False,
        }
        _require(
            all(summary.get(k) == v for k, v in normal.items()),
            "all 250 predictions must complete normally before gold access",
        )
        terminal = [
            {
                "question_id": qid,
                "methods": {
                    m: {"status": "completed", "path": f"{qid}_{m}.json"} for m in study.METHODS
                },
            }
            for qid in expected_ids
        ]
        names = {f"{qid}_{m}.json" for qid in expected_ids for m in study.METHODS}
        _require(
            summary.get("terminal") == terminal
            and set(summary.get("prediction_sha256", {})) == names,
            "terminal/seal coverage mismatch",
        )
        _require(
            [e["question_id"] for e in events if e.get("kind") == "question_completed"]
            == expected_ids
            and not any(e.get("kind") == "batch_stopped" for e in events),
            "completion sequence mismatch",
        )
        batch_calls = []
        for qid in expected_ids:
            row = {"question_id": qid, "question": by_id[qid]["text"], "methods": {}, "usage": {}}
            for method in study.METHODS:
                name = f"{qid}_{method}.json"
                path = tracked(f"runs/{run_id}/{name}")
                report = _read(path)
                raw_path = tracked(f"runs/{run_id}/raw_execution/{name}")
                _require(
                    _read(raw_path) == {k: v for k, v in report.items() if k != "method"},
                    "top report differs from immutable underlying execution",
                )
                arm = method if method in {"base", "fresh"} else "history"
                _require(
                    _sha(path) == summary["prediction_sha256"][name]
                    and report.get("status") == "completed"
                    and report.get("question_id") == qid
                    and report.get("arm") == arm
                    and report.get("method") == method
                    and report.get("question") == by_id[qid]
                    and report.get("gold_loaded") is False
                    and report.get("memory_updated") is False
                    and report.get("feedback") is None
                    and report["episode"]["question_id"] == qid,
                    "prediction hash/status/identity/feedback mismatch",
                )
                row["usage"][method] = _audit_method(
                    report, directory, plan, events, library, inputs, seen
                )
                row["methods"][method] = report
                batch_calls.extend(report["calls"])
            records.append(row)
        _require(
            ledger.get("calls") == batch_calls and ledger.get("block_reason") is None,
            "ledger has omitted/duplicate/failed requests",
        )
        for key, value in _totals(batch_calls).items():
            _require(
                math.isclose(_money(ledger.get(key)), value, abs_tol=1e-10), "ledger total mismatch"
            )
            _require(
                math.isclose(_money(summary.get(key)), value, abs_tol=1e-10),
                "summary cost mismatch",
            )
        all_calls.extend(batch_calls)
        prior = {**prior, run_id: _sha(summary_path)}
    _require(
        len(records) == COUNT and [r["question_id"] for r in records] == ids,
        "missing/duplicate registered study batch",
    )
    reconciled = study.reviewed_history(runs)
    audited = {
        c["trace_id"]
        for c in reconciled["calls"]
        if c["trace_id"].split("/", 1)[0] in expected_runs
    }
    _require(audited == seen, "project audit and prediction request coverage differ")
    return {
        "manifest": manifest,
        "ids": ids,
        "records": records,
        "inputs": inputs,
        "totals": _totals(all_calls),
        "source_sha256": snapshot_sha,
        "model": study.CONFIGURATION["model"],
        "protocol": study.PROTOCOL,
    }


def _load_gold(project, manifest, ids):
    """仅collect成功后投影固定50 train ID；不读任何source/dev答案。"""
    import pyarrow.dataset as ds

    project = Path(project).resolve(strict=True)
    _require(
        ids == manifest["question_ids"]
        and len(ids) == len(set(ids)) == COUNT
        and manifest["official_split"] == "train",
        "gold projection must be fixed train50",
    )
    entry = manifest["source_provenance"]
    path = _inside(project, project / entry["path"])
    _require(_sha(path) == entry["sha256"], "train provenance hash mismatch")
    provenance = _read(path)
    _require(provenance.get("official_split") == "train", "gold is not official train")
    recorded = {r["path"].replace("\\", "/"): r["sha256"] for r in manifest["parquet_inputs"]}
    _require(len(recorded) == len(manifest["parquet_inputs"]), "duplicate declared train shard")
    paths, inputs = [], {str(path): _sha(path)}
    for shard in provenance["shards"]:
        part = _inside(project, path.parent / shard["file"])
        digest = _sha(part)
        _require(
            part.parent == path.parent
            and digest == shard["sha256"] == recorded.get(part.relative_to(project).as_posix()),
            "train shard/provenance/manifest mismatch",
        )
        paths.append(str(part))
        inputs[str(part)] = digest
    _require(
        paths and len(paths) == len(set(paths)) == len(recorded), "missing/duplicate train shards"
    )
    rows = (
        ds.dataset(paths, format="parquet")
        .to_table(
            columns=["id", "answer", "context", "supporting_facts"],
            filter=ds.field("id").isin(ids),
        )
        .to_pylist()
    )
    _require(
        len(rows) == COUNT and {r["id"] for r in rows} == set(ids), "gold ID coverage mismatch"
    )
    return {r["id"]: r for r in rows}, inputs


def _mean(values):
    known = [v for v in values if v is not None]
    return {"mean": mean(known) if known else None, "valid_n": len(known)}


def summarize(records, gold):
    """纯评分；规范化回答只存在于评分副本，不改变原预测或实发成本。"""
    rows = []
    for item in records:
        qid = item["question_id"]
        canonical, raw, controlled, normalized_by = {}, {}, {}, {}
        for method in study.METHODS:
            report = item["methods"][method]
            key = item["usage"][method]["reader_input_sha256"]
            canonical.setdefault(key, method)  # 必须按注册method序，不按分数挑回答。
            first = canonical[key]
            normalized_by[method] = first
            raw[method] = score_source_report(report, gold[qid])
            view = deepcopy(report)
            view["reader"] = deepcopy(item["methods"][first]["reader"])
            controlled[method] = score_source_report(view, gold[qid])
        pairs = {}
        for method in CANDIDATE_METHODS:
            pairs[method] = {}
            for other in ("base", "fresh"):
                same = (
                    item["usage"][method]["reader_input_sha256"]
                    == item["usage"][other]["reader_input_sha256"]
                )
                values = {}
                for label, scores in (("raw", raw), (PRIMARY, controlled)):
                    left, right = scores[method], scores[other]
                    values[label] = {
                        "repaired": (right["answer_em"], left["answer_em"]) == (0.0, 1.0),
                        "harmed": (right["answer_em"], left["answer_em"]) == (1.0, 0.0),
                        "delta": {
                            m: None if left[m] is None or right[m] is None else left[m] - right[m]
                            for m in METRICS
                        },
                    }
                pairs[method][other] = {
                    "reader_input_identical": same,
                    "changed_reader_input": not same,
                    "raw_answer_changed": item["methods"][method]["reader"]["answer"]
                    != item["methods"][other]["reader"]["answer"],
                    **values,
                }
        oracle = {
            label: {
                metric: max(known) if known else None
                for metric in METRICS
                for known in [
                    [scores[m][metric] for m in CANDIDATE_METHODS if scores[m][metric] is not None]
                ]
            }
            for label, scores in (("raw", raw), (PRIMARY, controlled))
        }
        rows.append(
            {
                "question_id": qid,
                "question": item["question"],
                "reference_answer": gold[qid]["answer"],
                "answers": {m: item["methods"][m]["reader"]["answer"] for m in study.METHODS},
                "canonical_reader_method": normalized_by,
                "raw_scores": raw,
                "controlled_reader_scores": controlled,
                "history_vs": pairs,
                "usage": item["usage"],
                "executed_policy_oracle": oracle,
            }
        )
    methods = {}
    for method in study.METHODS:
        stats = {
            label: {metric: _mean([r[key][method][metric] for r in rows]) for metric in METRICS}
            for label, key in (("raw", "raw_scores"), (PRIMARY, "controlled_reader_scores"))
        }
        for key in (
            "api_requests",
            "input_tokens",
            "output_tokens",
            "estimated_actual_cny",
            "reserved_cny",
            "retrieval_calls",
        ):
            stats[key] = sum(r["usage"][method][key] for r in rows)
        stats["prepared_card_questions"] = sum(
            bool(r["usage"][method]["executed_cards"]) for r in rows
        )
        stats["actually_searched_card_questions"] = sum(
            any(c["search_executed"] for c in r["usage"][method]["executed_cards"]) for r in rows
        )
        stats["actually_searched_learned_card_questions"] = sum(
            any(
                c["source_kind"] == "learned" and c["search_executed"]
                for c in r["usage"][method]["executed_cards"]
            )
            for r in rows
        )
        stats["actually_searched_learned_card_actions"] = sum(
            c["source_kind"] == "learned" and c["search_executed"]
            for r in rows
            for c in r["usage"][method]["executed_cards"]
        )
        methods[method] = stats
    paired = {}
    for method in CANDIDATE_METHODS:
        paired[method] = {}
        for other in ("base", "fresh"):
            values = [r["history_vs"][method][other] for r in rows]
            paired[method][other] = {
                "changed_reader_input": sum(v["changed_reader_input"] for v in values),
                "identical_reader_input": sum(v["reader_input_identical"] for v in values),
                "same_input_raw_answer_changed": sum(
                    v["reader_input_identical"] and v["raw_answer_changed"] for v in values
                ),
                **{
                    label: {
                        "repaired": sum(v[label]["repaired"] for v in values),
                        "harmed": sum(v[label]["harmed"] for v in values),
                        "delta": {
                            m: _mean([v[label]["delta"][m] for v in values]) for m in METRICS
                        },
                    }
                    for label in ("raw", PRIMARY)
                },
            }
    return rows, {
        "questions": len(rows),
        "completed_methods": len(rows) * len(study.METHODS),
        "primary_metric_policy": PRIMARY,
        "methods": methods,
        "history_vs": paired,
        "executed_policy_oracle": {
            label: {
                m: _mean([r["executed_policy_oracle"][label][m] for r in rows]) for m in METRICS
            }
            for label in ("raw", PRIMARY)
        },
        "oracle_notice": (
            "Per-metric upper bound among only the three executed history policies; "
            "not a 44-action oracle, not an online routing result."
        ),
        "reader_control_notice": (
            "First same-input Reader by fixed METHODS order; raw predictions and every actual "
            "API/retrieval cost retained. No deployment cache or cost savings."
        ),
    }


def _markdown(rows):
    lines = [
        "# 候选检索50题开发反馈",
        "",
        "主指标：同题同Reader输入采用固定method顺序的首次回答。原回答单列，所有实发费用保留。",
        "仅train开发诊断；选择/准备/真正发出检索分别记录。三个已执行策略的oracle不是44动作上限或线上结果。",
        "",
    ]
    for row in rows:
        lines += [
            f"## {row['question_id']}",
            "",
            row["question"],
            "",
            f"参考答案：{row['reference_answer']}",
            "",
            "| method | 原EM/F1 | 控制EM/F1 | 可见证据覆盖 | API/检索 | 实发估算元 |",
            "|---|---|---|---|---|---|",
        ]
        for m in study.METHODS:
            raw, controlled, usage = (
                row["raw_scores"][m],
                row["controlled_reader_scores"][m],
                row["usage"][m],
            )
            lines.append(
                f"| {m} | {raw['answer_em']}/{raw['answer_f1']} | "
                f"{controlled['answer_em']}/{controlled['answer_f1']} | "
                f"{raw['visible_support_recall']} | "
                f"{usage['api_requests']}/{usage['retrieval_calls']} | "
                f"{usage['estimated_actual_cny']:.6f} |"
            )
        lines += [
            "",
            "```json",
            json.dumps(
                {k: row[k] for k in ("answers", "canonical_reader_method", "usage", "history_vs")},
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    return "\n".join(lines)


def run(project, *, score=False):
    project = Path(project).resolve(strict=True)
    output = (project / OUTPUT).resolve()
    _require(output.is_relative_to(project / "runs"), "feedback output escapes runs")
    if score and output.exists():
        raise FileExistsError("feedback output exists; no overwrite, subset or rescoring")
    checked = collect(project)
    public = {
        "protocol": checked["protocol"],
        "questions": COUNT,
        "methods": list(study.METHODS),
        "status": "preflight_passed",
        "gold_loaded": False,
        "api_calls": 0,
        "source_sha256": checked["source_sha256"],
        "totals": checked["totals"],
    }
    if not score:
        return public
    gold, gold_inputs = _load_gold(project, checked["manifest"], checked["ids"])
    rows, summary = summarize(checked["records"], gold)
    inputs = {**checked["inputs"], **gold_inputs}
    _require(
        all(_sha(Path(p)) == digest for p, digest in inputs.items()),
        "inputs changed during scoring",
    )
    summary.update(
        protocol=checked["protocol"],
        model=checked["model"],
        totals=checked["totals"],
        source_sha256=checked["source_sha256"],
        split="official_train_independent_candidate_development50",
        api_calls=0,
        memory_updated=False,
        prediction_modified=False,
        labels_opened_after_full_seal=True,
        policy=(
            "Development diagnosis only; no rerun, label-based within-batch update "
            "or causal attribution."
        ),
    )
    output.mkdir(exist_ok=False)
    for name, value in (
        ("per_question.json", rows),
        ("SUMMARY.json", summary),
        ("audit.json", {"inputs": inputs}),
    ):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / "per_question.md").open("x", encoding="utf-8") as handle:
        handle.write(_markdown(rows))
    with (output / "feedback_frozen.json").open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "files": {
                    p.name: _sha(p) for p in output.iterdir() if p.name != "feedback_frozen.json"
                },
                "api_calls": 0,
                "primary_metric_policy": PRIMARY,
            },
            handle,
            indent=2,
        )
    return {**public, "status": "scored", "gold_loaded": True, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score",
        action="store_true",
        help="after all 250 sealed predictions, project train50 labels",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(Path.cwd(), score=args.score), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
