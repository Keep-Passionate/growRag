"""50道独立train开发题的五路候选对照；默认预检，预测封存前不读gold。

只改变历史候选排序/数量，不改变选择、填参、检索、Reader或卡片。三个批次
串行推进，先通过真实FILL接口探针，后续批次须有同一冻结源码的完整前批。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from growrag.history_library import FrozenHistoryLibrary

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_dev_manifest import _sha
from .history_budget import reviewed_history
from .history_calibration import LIBRARY_FINGERPRINT, LIBRARY_SHA256, execute_arm
from .history_calibration import check_claims as check_history_claims
from .history_candidate_runtime import METHODS as CANDIDATE_METHODS
from .history_candidate_runtime import planner_factory
from .history_fill_probe import PROTOCOL as PROBE_PROTOCOL
from .history_fill_probe import RUN_ID as PROBE_RUN_ID
from .history_fill_probe import cases as probe_cases
from .history_fill_probe import compile_probe, probe_request
from .history_runtime import HistoryContractClient, HistoryPlanner
from .history_runtime_v2 import FILL_VERSION
from .pre_pilot import write_json
from .prepare_history_opportunity import OUTPUT as BUNDLE_PATH
from .prepare_history_opportunity import load_bundle
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_operator_study import OperatorLog, serial_lock
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient
from .shared_s2g_corpus import SharedBM25Index

PREFIX = "2026-10-02_history_candidate_"
PROTOCOL = "growrag-history-candidate-study-v1"
METHODS = ("base", "fresh", "history_legacy3", "history_body3", "history_body8")
BATCHES = ((0, 5), (5, 20), (25, 25))
# 真实bundle生成后，由外部审阅固定；缺省不能自行信任manifest旁的hash文件。
BUNDLE_SHA256 = "e8aa9cb2fe6323a2823d9085acecb6b4b70418cf2a82075413f02e0621e3dbef"
KEY_VARIABLE = "GROWRAG_HISTORY_CANDIDATE_API_KEY"
CONFIGURATION = {
    "model": PILOT_MODEL,
    "max_output_tokens": 2048,
    "output_limit_parameter": "max_tokens",
    "enable_thinking": False,
    "temperature": 0,
    "top_p": 1,
    "json_object_mode": True,
    "json_schema_mode": False,
    "retrieval_budget": 3,
    "max_decisions": 2,
    "top_k": 6,
    "representation": "examples",
    "hybrid_rule_query": True,
    "runtime_memory_updates": False,
    "fill_prompt_version": FILL_VERSION,
    "candidate_methods": {
        method: {"policy": policy, "limit": limit}
        for method, (policy, limit) in CANDIDATE_METHODS.items()
    },
    "transport_contract": "json-object-plus-local-validation-all-arms-v1",
    "reader_control": "first-identical-messages-by-fixed-method-order-analysis-only-v1",
}


def run_id(start, count):
    return f"{PREFIX}v1_{start:04d}_{start + count:04d}"


def underlying_arm(method):
    if method not in METHODS:
        raise ValueError("unregistered candidate method")
    return method if method in {"base", "fresh"} else "history"


def prepare(project, *, start, count):
    if type(start) is not int or type(count) is not int or (start, count) not in BATCHES:
        raise ValueError("only preregistered candidate batches are allowed")
    if not isinstance(BUNDLE_SHA256, str) or not re.fullmatch(r"[0-9a-f]{64}", BUNDLE_SHA256):
        raise ValueError("external bundle SHA must be pinned before preflight or execution")
    manifest, questions, corpus = load_bundle(project, BUNDLE_SHA256)
    if len(questions) != 50 or [q.question_id for q in questions] != manifest["question_ids"]:
        raise ValueError("candidate bundle must contain exactly the frozen ordered 50 questions")
    info = manifest["library"]
    path = project / info["path"]
    if info["sha256"] != LIBRARY_SHA256 or _sha(path) != LIBRARY_SHA256:
        raise ValueError("frozen history library bytes changed")
    library = FrozenHistoryLibrary.from_json(path.read_text(encoding="utf-8"))
    if library.fingerprint != LIBRARY_FINGERPRINT or len(library.published_cards) != 44:
        raise ValueError("fixed 44-card history library changed")
    return manifest, questions[start : start + count], library, corpus


def check_claims(runs, ids):
    """旧方法和新方法的声明均排除；接口失败也不能自动换名字重试。"""
    check_history_claims(runs, ids)
    paths = (*runs.glob(f"{PREFIX}*.claim.json"), *runs.glob(f"{PREFIX}*/launch_plan.json"))
    for path in paths:
        if not path.resolve().is_relative_to(runs.resolve()):
            raise ValueError("candidate claim escapes runs directory")
        claim = json.loads(path.read_bytes())
        if claim.get("protocol") == PROBE_PROTOCOL:
            if claim.get("run_id") != PROBE_RUN_ID or "question_ids" in claim:
                raise ValueError("unreviewed probe claim")
            continue
        old_ids = claim.get("question_ids")
        if (
            claim.get("protocol") != PROTOCOL
            or type(old_ids) is not list
            or not old_ids
            or any(not isinstance(qid, str) or not qid for qid in old_ids)
            or len(set(old_ids)) != len(old_ids)
            or claim.get("methods") != list(METHODS)
        ):
            raise ValueError("unreviewed candidate claim")
        if set(ids) & set(old_ids):
            raise ValueError("question previously claimed; no automatic replay")


def _sealed_metadata(root, snapshot_sha):
    launch = json.loads((root / "launch_plan.json").read_bytes())
    claim = json.loads(root.with_name(root.name + ".claim.json").read_bytes())
    snapshot = json.loads((root / "source_snapshot.json").read_bytes())
    if (
        claim != {**launch, "plan_sha256": fingerprint(launch)}
        or launch.get("source_sha256") != snapshot_sha
        or snapshot.get("sha256") != snapshot_sha
        or fingerprint(snapshot.get("files")) != snapshot_sha
    ):
        raise ValueError("sealed claim/launch/source snapshot mismatch")
    return launch


def _probe_library(runs):
    path = runs / "history_foundation_20261001_v1/combined.json"
    if _sha(path) != LIBRARY_SHA256:
        raise ValueError("frozen probe library bytes changed")
    library = FrozenHistoryLibrary.from_json(path.read_text(encoding="utf-8"))
    if library.fingerprint != LIBRARY_FINGERPRINT:
        raise ValueError("frozen probe library fingerprint changed")
    return library


def verify_probe(runs, snapshot_sha):
    """五次真实FILL及消费者编译成功，不把合成探针当效果评价。"""
    root = runs / PROBE_RUN_ID
    launch = _sealed_metadata(root, snapshot_sha)
    summary = json.loads((root / "SUMMARY.json").read_bytes())
    expected = {f"{case['case_id']}.json": case for case in probe_cases()}
    if (
        launch.get("protocol") != PROBE_PROTOCOL
        or launch.get("run_id") != PROBE_RUN_ID
        or launch.get("model") != PILOT_MODEL
        or launch.get("library_sha256") != LIBRARY_SHA256
        or launch.get("cases_sha256") != fingerprint(probe_cases())
        or launch.get("cases") != list(probe_cases())
        or summary.get("protocol") != PROBE_PROTOCOL
        or summary.get("status") != "completed"
        or summary.get("planned_cases") != 5
        or summary.get("completed_cases") != 5
        or summary.get("api_requests") != 5
        or summary.get("source_sha256") != snapshot_sha
        or summary.get("library_sha256") != LIBRARY_SHA256
        or set(summary.get("prediction_sha256", {})) != set(expected)
    ):
        raise ValueError("five successful frozen FILL probes are required")
    records = {}
    for name, case in expected.items():
        path = root / name
        record = json.loads(path.read_bytes())
        if (
            _sha(path) != summary["prediction_sha256"][name]
            or record.get("status") != "completed"
            or record.get("case") != case
            or not isinstance(record.get("raw_content"), str)
            or len(record.get("compiled", {}).get("requests", [])) != 1
        ):
            raise ValueError("probe prediction missing, failed or changed")
        records[case["case_id"]] = record
    budget = json.loads((root / "final_budget.json").read_bytes())
    calls = budget.get("calls", [])
    expected_traces = {f"{PROBE_RUN_ID}/{case['case_id']}/fill" for case in probe_cases()}
    if (
        budget.get("api_requests") != 5
        or len(calls) != 5
        or {call.get("trace_id") for call in calls} != expected_traces
        or any(
            call.get("status") != "completed"
            or call.get("prompt_version") != FILL_VERSION
            or call.get("returned_model") != PILOT_MODEL
            or call.get("api_requests") != 1
            or any(
                type(call.get(k)) is not int or call[k] < 0
                for k in ("input_tokens", "output_tokens")
            )
            for call in calls
        )
    ):
        raise ValueError("probe must preserve five real completed FILL calls")
    library = _probe_library(runs)
    cases_by_trace = {f"{PROBE_RUN_ID}/{case['case_id']}/fill": case for case in probe_cases()}
    for call in calls:
        audit_path = root / "api_audit" / Path(call.get("audit_path", "")).name
        if not audit_path.resolve().is_relative_to(root.resolve()):
            raise ValueError("probe audit escapes its run")
        audit = json.loads(audit_path.read_bytes())
        if (
            audit.get("transport_source") != "live_api"
            or audit.get("http_status") != 200
            or audit.get("status") != "completed"
            or audit.get("prompt_version") != FILL_VERSION
            or audit.get("request", {}).get("model") != PILOT_MODEL
            or audit.get("response", {}).get("model") != PILOT_MODEL
            or any(
                audit.get(k) != call[k]
                for k in ("trace_id", "api_requests", "input_tokens", "output_tokens")
            )
        ):
            raise ValueError("probe HTTP audit/model/usage differs from completed FILL ledger")
        case = cases_by_trace[call["trace_id"]]
        record = records[case["case_id"]]
        choices = audit.get("response", {}).get("choices", [])
        if (
            len(choices) != 1
            or choices[0].get("message", {}).get("content") != record["raw_content"]
        ):
            raise ValueError("probe saved content differs from its actual HTTP response")
        _, evidence, context, _ = probe_request(case, library)
        compiled = compile_probe(case, library, context, evidence, record["raw_content"])
        if json.loads(json.dumps(compiled)) != record["compiled"]:
            raise ValueError("probe saved compilation differs from frozen consumer output")
    return _sha(root / "SUMMARY.json")


def verify_prior_batches(runs, manifest, start, snapshot_sha, probe_sha):
    """必须完成所有此前批次，不依据答案/分数开闸。"""
    hashes = {}
    for prior_start, count in BATCHES:
        if prior_start >= start:
            break
        root = runs / run_id(prior_start, count)
        launch = _sealed_metadata(root, snapshot_sha)
        summary = json.loads((root / "SUMMARY.json").read_bytes())
        ids = manifest["question_ids"][prior_start : prior_start + count]
        if (
            launch.get("protocol") != PROTOCOL
            or launch.get("run_id") != root.name
            or launch.get("phase") != "candidate_development"
            or launch.get("question_ids") != ids
            or launch.get("methods") != list(METHODS)
            or launch.get("start") != prior_start
            or launch.get("count") != count
            or launch.get("bundle_sha256") != BUNDLE_SHA256
            or launch.get("library_sha256") != LIBRARY_SHA256
            or launch.get("probe_summary_sha256") != probe_sha
            or launch.get("prior_batch_summary_sha256") != hashes
            or any(launch.get(key) != value for key, value in CONFIGURATION.items())
            or summary.get("protocol") != PROTOCOL
            or summary.get("status") != "completed"
            or summary.get("planned_questions") != count
            or summary.get("completed_questions") != count
            or summary.get("source_sha256") != snapshot_sha
        ):
            raise ValueError("all prior batches must be complete under the same frozen method")
        expected = {f"{qid}_{method}.json" for qid in ids for method in METHODS}
        if set(summary.get("prediction_sha256", {})) != expected:
            raise ValueError("prior batch prediction set is incomplete")
        terminal = []
        for qid in ids:
            item = {"question_id": qid, "methods": {}}
            for method in METHODS:
                name = f"{qid}_{method}.json"
                path = root / name
                report = json.loads(path.read_bytes())
                if (
                    _sha(path) != summary["prediction_sha256"][name]
                    or report.get("status") != "completed"
                    or report.get("question_id") != qid
                    or report.get("method") != method
                    or report.get("arm") != underlying_arm(method)
                    or report.get("gold_loaded") is not False
                    or report.get("memory_updated") is not False
                ):
                    raise ValueError("prior prediction failed, changed or has invalid identity")
                item["methods"][method] = {"status": "completed", "path": name}
            terminal.append(item)
        if summary.get("terminal") != terminal or not (root / "final_budget.json").is_file():
            raise ValueError("prior terminal coverage or budget seal is incomplete")
        hashes[root.name] = _sha(root / "SUMMARY.json")
    return hashes


def execute_method(question, method, index, client, *, library, trace, log, target):
    """旧执行器原报告另存，顶层只添加method身份，不覆盖原始失败记录。"""
    arm = underlying_arm(method)
    raw = target.parent / "raw_execution" / target.name
    if target.exists() or raw.exists():
        raise FileExistsError("method already has a prediction; never replay")
    raw.parent.mkdir(exist_ok=True)
    try:
        execute_arm(
            question,
            arm,
            index,
            client,
            library=library,
            reference=None,
            trace=trace,
            log=log,
            target=raw,
            planner_class=planner_factory(method) if arm == "history" else HistoryPlanner,
        )
    finally:
        if raw.exists():
            report = json.loads(raw.read_bytes())
            write_json(target, {**report, "method": method})
    return json.loads(target.read_bytes())


def _runtime_unchanged(project, manifest, corpus):
    if (
        _sha(project / BUNDLE_PATH / "manifest.json") != BUNDLE_SHA256
        or _sha(corpus) != manifest["corpus_ref"]["sha256"]
        or _sha(project / manifest["library"]["path"]) != LIBRARY_SHA256
    ):
        raise ValueError("runtime bundle/corpus/library bytes changed")


def run(args):
    project = Path.cwd().resolve(strict=True)
    runs = project / "runs"
    manifest, chosen, library, corpus = prepare(project, start=args.start, count=args.count)
    ids = [question.question_id for question in chosen]
    check_claims(runs, ids)
    snapshot = source_snapshot(project)
    identity = run_id(args.start, args.count)
    output = runs / identity
    if output.exists() or (runs / f"{identity}.claim.json").exists():
        raise FileExistsError("run already exists; never overwrite or replay")
    plan = {
        **CONFIGURATION,
        "protocol": PROTOCOL,
        "run_id": identity,
        "phase": "candidate_development",
        "question_ids": ids,
        "methods": list(METHODS),
        "start": args.start,
        "count": args.count,
        "bundle_sha256": BUNDLE_SHA256,
        "bundle_path": f"{BUNDLE_PATH}/manifest.json",
        "library_sha256": LIBRARY_SHA256,
        "library_fingerprint": library.fingerprint,
        "source_sha256": snapshot["sha256"],
        "git": _git_state(),
        "gold_loaded": False,
        "memory_updates": False,
        "official_split": "train",
        "project_cap_cny": 200.0,
        "series_cap_cny": 5.0,
        "price_profile": "declared input0.2/output0.8 CNY per million, not invoice",
        "price_checked_date": "2026-10-02",
        "price_source": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
    }
    if not args.allow_network:
        print(json.dumps({"dry_run": True, "plan": plan}, ensure_ascii=False, indent=2))
        return 0
    with serial_lock(runs):
        check_claims(runs, ids)
        if output.exists() or (runs / f"{identity}.claim.json").exists():
            raise FileExistsError("run already exists; never overwrite or replay")
        plan["probe_summary_sha256"] = verify_probe(runs, snapshot["sha256"])
        plan["prior_batch_summary_sha256"] = verify_prior_batches(
            runs, manifest, args.start, snapshot["sha256"], plan["probe_summary_sha256"]
        )
        history = reviewed_history(runs)
        series_reserved = sum(
            row["reserved_cny"] for row in history["calls"] if row["trace_id"].startswith(PREFIX)
        )
        subcap = min(5.0 - series_reserved, 200.0 - history["prior_reserved_cny"])
        if subcap <= 0:
            raise ValueError("series or cumulative project reservation exhausted")
        settings = read_local_bailian_settings(project / "qwenAPI.md")
        host = urlsplit(settings.base_url).hostname or ""
        if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
            raise ValueError("declared price profile requires the Beijing endpoint")
        if source_snapshot(project)["sha256"] != snapshot["sha256"]:
            raise ValueError("source changed before execution")
        _runtime_unchanged(project, manifest, corpus)
        plan.update(prior_reserved_cny=history["prior_reserved_cny"], subcap_cny=subcap)
        write_json(runs / f"{identity}.claim.json", {**plan, "plan_sha256": fingerprint(plan)})
        output.mkdir(exist_ok=False)
        write_json(output / "launch_plan.json", plan)
        write_json(output / "source_snapshot.json", snapshot)
        write_json(output / "prior_budget.json", history)
        write_json(output / "process.json", {"pid": os.getpid()})
        log = OperatorLog(output)
        client, index, failure = None, None, None
        previous_key = os.environ.get(KEY_VARIABLE)
        limits = PriceLimits(budget_cny=subcap, max_prompt_bytes=30000, max_elapsed_seconds=1800)
        try:
            info = manifest["corpus_ref"]
            index = SharedBM25Index(
                project / info["index_path"], corpus, info["sha256"], info["rows"]
            )
            os.environ[KEY_VARIABLE] = settings.api_key
            config = ChatConfig(
                settings.base_url,
                PILOT_MODEL,
                KEY_VARIABLE,
                max_calls=len(chosen) * 19,
                max_output_tokens=2048,
                timeout_seconds=60,
                output_limit_parameter="max_tokens",
                enable_thinking=False,
                temperature=0,
                top_p=1,
                json_object_mode=True,
            )
            client = DurableBudgetClient(
                LiveChatClient(config, output / "api_audit", allow_network=True),
                limits,
                output / "request_journal",
            )
            contract = HistoryContractClient(client, on_record=log)
            for question in chosen:
                log.question_id, log.arm = question.question_id, "setup"
                log({"kind": "question_start"})
                for method in METHODS:
                    log.arm = method
                    _runtime_unchanged(project, manifest, corpus)
                    execute_method(
                        question,
                        method,
                        index,
                        contract,
                        library=library,
                        trace=f"{identity}/{question.question_id}/{method}",
                        log=log,
                        target=output / f"{question.question_id}_{method}.json",
                    )
                log({"kind": "question_completed"})
            if source_snapshot(project)["sha256"] != snapshot["sha256"]:
                raise ValueError("source changed during execution; preserve predictions for audit")
        except BaseException as error:
            failure = type(error).__name__
            if client is not None and not client.block_reason:
                client.block_reason = "experiment_execution_failure"
            log({"kind": "batch_stopped", "error_type": failure})
        finally:
            # 清理索引失败也必须保存费用，并恢复原有环境变量。
            try:
                if index is not None:
                    index.close()
            except BaseException as error:
                failure = failure or type(error).__name__
            finally:
                if previous_key is None:
                    os.environ.pop(KEY_VARIABLE, None)
                else:
                    os.environ[KEY_VARIABLE] = previous_key
                budget = (
                    client.report()
                    if client is not None
                    else {
                        "api_requests": 0,
                        "calls": [],
                        "reserved_cny": 0,
                        "estimated_actual_cny": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "limits": asdict(limits),
                    }
                )
                write_json(output / "final_budget.json", budget)
            terminal, hashes = [], {}
            for question in chosen:
                row = {"question_id": question.question_id, "methods": {}}
                for method in METHODS:
                    path = output / f"{question.question_id}_{method}.json"
                    if path.exists():
                        saved = json.loads(path.read_bytes())
                        hashes[path.name] = _sha(path)
                        row["methods"][method] = {"status": saved["status"], "path": path.name}
                    else:
                        row["methods"][method] = {"status": "not_attempted", "path": None}
                terminal.append(row)
            summary = {
                "protocol": PROTOCOL,
                "run_id": identity,
                "status": "failed" if failure else "completed",
                "failure_type": failure,
                "planned_questions": len(chosen),
                "completed_questions": sum(
                    all(item["status"] == "completed" for item in row["methods"].values())
                    for row in terminal
                ),
                "terminal": terminal,
                "prediction_sha256": hashes,
                "source_sha256": snapshot["sha256"],
                "bundle_sha256": BUNDLE_SHA256,
                "library_sha256": LIBRARY_SHA256,
                "gold_loaded": False,
                "memory_updated": False,
                "new_operator_induction": False,
                "api_requests": budget["api_requests"],
                "reserved_cny": budget["reserved_cny"],
                "estimated_actual_cny": budget["estimated_actual_cny"],
                "input_tokens": budget.get("input_tokens"),
                "output_tokens": budget.get("output_tokens"),
            }
            write_json(output / "SUMMARY.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 1 if failure else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--allow-network", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
