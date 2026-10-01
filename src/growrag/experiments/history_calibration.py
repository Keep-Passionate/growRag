"""独立的train校准入口：历史基底接通、四路真实对照、全部预测先封存。

此入口不读gold、不更新记忆、不开放evaluation。默认只预检；显式网络开关
沿用已授权普通按量API与累计项目账本。每个qid/arm只声明一次，失败不重试。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from growrag.history_library import FrozenHistoryLibrary
from growrag.operator_loop import run_operator_episode

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_dev_manifest import _sha
from .history_budget import PREFIX, PROTOCOL, REVIEWED_HISTORY_PROTOCOLS, reviewed_history
from .history_runtime import HistoryContractClient, HistoryPlanner
from .history_runtime_v2 import FILL_VERSION as FILL_VERSION_V2
from .history_runtime_v2 import HistoryPlannerV2
from .operator_model_v3 import ModelOperatorPlannerV3, answer_episode
from .pre_pilot import write_json
from .protocol import Evidence
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_operator_study import OperatorLog, artifact, check_unstarted, load_inputs, serial_lock
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient
from .shared_s2g_corpus import SharedBM25Index

ARMS = ("base", "fresh", "static_rules", "history")
KEY_VARIABLE = "GROWRAG_HISTORY_API_KEY"
LIBRARY_FINGERPRINT = "8b6aa7f14baa54a7048d539ec791c7884e1029c0aa20afbce0693a55ce7934c7"
LIBRARY_SHA256 = "73eab3f46d9d6edaf20c497cc39e6082086f1ab4c43afe45b606abf35397dd88"
MANIFEST_SHA256 = "afc54ffb487b095b9cabf751cc3139fe0b84df93b7577770cb44462766488618"
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
    "shortlist_size": 3,
    "representation": "examples",
    "hybrid_rule_query": True,
    "runtime_memory_updates": False,
    "selection_policy": "original-question-jaccard-canonical-no-examples-v1",
    "transport_contract": "json-object-plus-local-validation-all-arms-v1",
}


def profile_for(version="v1"):
    """显式版本决定题目范围和填参接口；失败旧题不因修接口而重跑。"""
    if version == "v1":
        return {
            "protocol": PROTOCOL,
            "batches": ((40, 5), (45, 20)),
            "configuration": dict(CONFIGURATION),
            "planner_class": HistoryPlanner,
        }
    if version == "v2":
        return {
            "protocol": "growrag-history-calibration-v2",
            "batches": ((65, 5), (70, 20)),
            "configuration": {**CONFIGURATION, "fill_prompt_version": FILL_VERSION_V2},
            "planner_class": HistoryPlannerV2,
        }
    raise ValueError("unregistered history calibration version")


def prepare(project, *, start, count, version="v1"):
    """锁住角色和字节后才解码无标签校准题；本版本只允许既定25题。"""
    if (
        type(start) is not int
        or type(count) is not int
        or (start, count) not in profile_for(version)["batches"]
    ):
        raise ValueError("this version predeclares only its registered calibration batches")
    manifest_path = project / "data/hotpotqa/operator_scale_action_v3/manifest.json"
    library_path = project / "runs/history_foundation_20261001_v1/combined.json"
    if _sha(manifest_path) != MANIFEST_SHA256 or _sha(library_path) != LIBRARY_SHA256:
        raise ValueError("manifest/history bytes changed from externally pinned artifacts")
    manifest, questions = load_inputs(manifest_path, "calibration")
    library = FrozenHistoryLibrary.from_json(library_path.read_text(encoding="utf-8"))
    if library.fingerprint != LIBRARY_FINGERPRINT:
        raise ValueError("history fingerprint changed")
    source_ids = set(manifest["roles"]["source"])
    protected_ids = set(manifest["roles"]["calibration"] + manifest["roles"]["evaluation"])
    if set(library.allowed_source_ids) != source_ids or any(
        set(record.source_qids) - source_ids or set(record.source_qids) & protected_ids
        for record in library.records
    ):
        raise ValueError("library source role overlap or missing source provenance")
    reference = FrozenHistoryLibrary(
        library.protocol_id,
        library.allowed_source_ids,
        tuple(record for record in library.records if record.source_kind == "reference"),
    )
    chosen = questions[start : start + count]
    corpus = artifact(manifest_path.parent, manifest, "corpus.jsonl")
    return manifest, chosen, library, reference, corpus


def check_claims(runs, ids):
    """新协议的所有旧声明也保留；旧方法已有预留不能用新名字重跑。"""
    check_unstarted(runs, "calibration", list(ids), ("base", "fresh"))
    for path in runs.glob(f"{PREFIX}*.claim.json"):
        claim = json.loads(path.read_bytes())
        if claim.get("protocol") not in REVIEWED_HISTORY_PROTOCOLS or not isinstance(
            claim.get("question_ids"), list
        ):
            raise ValueError("unreviewed history claim")
        if set(ids) & set(claim["question_ids"]):
            raise ValueError("question previously claimed by history series; no automatic replay")


def verify_first_batch(runs, snapshot_sha, version="v1"):
    """扩题前核对完整预测字节；不能只相信一个手写completed字段。"""
    profile = profile_for(version)
    start, count = profile["batches"][0]
    root = runs / f"{PREFIX}{version}_{start:04d}_{start + count:04d}"
    previous = json.loads((root / "SUMMARY.json").read_bytes())
    launch = json.loads((root / "launch_plan.json").read_bytes())
    if (
        previous.get("status") != "completed"
        or previous.get("completed_questions") != 5
        or previous.get("source_sha256") != snapshot_sha
        or launch.get("protocol") != profile["protocol"]
        or launch.get("manifest_sha256") != MANIFEST_SHA256
        or launch.get("library_sha256") != LIBRARY_SHA256
        or launch.get("source_sha256") != snapshot_sha
        or len(launch.get("question_ids", [])) != 5
        or launch.get("arms") != list(ARMS)
        or any(launch.get(key) != value for key, value in profile["configuration"].items())
    ):
        raise ValueError("first five must complete under the same frozen method before expansion")
    expected = {f"{qid}_{arm}.json" for qid in launch["question_ids"] for arm in ARMS}
    if set(previous.get("prediction_sha256", {})) != expected:
        raise ValueError("first batch prediction set is incomplete")
    for name in expected:
        path = root / name
        report = json.loads(path.read_bytes())
        if _sha(path) != previous["prediction_sha256"][name] or report.get("status") != "completed":
            raise ValueError("first batch prediction changed or failed")
    return _sha(root / "SUMMARY.json")


def execute_arm(
    question,
    arm,
    index,
    client,
    *,
    library,
    reference,
    trace,
    log,
    target,
    planner_class=HistoryPlanner,
):
    """四路共享检索与Reader；异常先落盘再向上停止整个批次。"""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    start_call = len(client.calls)
    report = {
        "status": "started",
        "question_id": question.question_id,
        "question": asdict(question),
        "arm": arm,
        "episode": None,
        "reader": None,
        "memory_updated": False,
        "gold_loaded": False,
    }
    try:
        if arm in {"base", "fresh"}:
            planner = ModelOperatorPlannerV3(
                client,
                mode="fresh",
                trace_prefix=trace,
                on_record=lambda event: log({"kind": "planner_record", **event}),
            )
        else:
            planner = planner_class(
                client,
                reference if arm == "static_rules" else library,
                trace_prefix=trace,
                origin="static" if arm == "static_rules" else "reuse",
                on_record=log,
            )

        def retrieve(query, top_k):
            return tuple(
                Evidence(doc.doc_id, doc.title, 0, doc.text) for doc in index(query, top_k)
            )

        result = run_operator_episode(
            question,
            retrieve,
            planner,
            retrieval_budget=1 if arm == "base" else 3,
            max_decisions=2,
            top_k=6,
            on_event=log,
        )
        report["episode"] = asdict(result)
        report["reader"] = answer_episode(
            client,
            question,
            result.evidence,
            trace_id=f"{trace}/reader",
            on_record=lambda event: log({"kind": "reader_record", **event}),
        )
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        report["calls"] = list(client.calls[start_call:])
        write_json(target, report)
    return report


def run(args):
    project = Path.cwd().resolve(strict=True)
    runs = project / "runs"
    version = getattr(args, "version", "v1")
    profile = profile_for(version)
    manifest, chosen, library, reference, corpus = prepare(
        project, start=args.start, count=args.count, version=version
    )
    ids = [question.question_id for question in chosen]
    check_claims(runs, ids)
    snapshot = source_snapshot(project)
    run_id = f"{PREFIX}{version}_{args.start:04d}_{args.start + args.count:04d}"
    plan = {
        **profile["configuration"],
        "protocol": profile["protocol"],
        "run_id": run_id,
        "phase": "calibration",
        "question_ids": ids,
        "arms": list(ARMS),
        "start": args.start,
        "count": args.count,
        "manifest_sha256": MANIFEST_SHA256,
        "library_sha256": LIBRARY_SHA256,
        "library_fingerprint": library.fingerprint,
        "reference_fingerprint": reference.fingerprint,
        "source_sha256": snapshot["sha256"],
        "git": _git_state(),
        "gold_loaded": False,
        "memory_updates": False,
        "official_split": "train",
        "project_cap_cny": 200.0,
        "series_cap_cny": 3.0,
        "price_profile": "declared historical input0.2/output0.8 CNY per million, not invoice",
        "price_checked_date": "2026-10-01",
        "price_source": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
    }
    if not args.allow_network:
        print(json.dumps({"dry_run": True, "plan": plan}, ensure_ascii=False, indent=2))
        return 0
    with serial_lock(runs):
        check_claims(runs, ids)
        history = reviewed_history(runs)
        series_reserved = sum(
            row["reserved_cny"] for row in history["calls"] if row["trace_id"].startswith(PREFIX)
        )
        subcap = min(3.0 - series_reserved, 200.0 - history["prior_reserved_cny"])
        if subcap <= 0:
            raise ValueError("series or cumulative project reservation exhausted")
        if (args.start, args.count) == profile["batches"][1]:
            plan["first_batch_summary_sha256"] = verify_first_batch(
                runs, snapshot["sha256"], version
            )
        settings = read_local_bailian_settings(project / "qwenAPI.md")
        endpoint_host = urlsplit(settings.base_url).hostname or ""
        if endpoint_host != "dashscope.aliyuncs.com" and not endpoint_host.endswith(
            ".cn-beijing.maas.aliyuncs.com"
        ):
            raise ValueError("declared price profile is Beijing; configured region differs")
        plan.update(prior_reserved_cny=history["prior_reserved_cny"], subcap_cny=subcap)
        output = runs / run_id
        if output.exists():
            raise FileExistsError("run already exists; never overwrite")
        write_json(runs / f"{run_id}.claim.json", {**plan, "plan_sha256": fingerprint(plan)})
        output.mkdir(exist_ok=False)
        write_json(output / "launch_plan.json", plan)
        write_json(output / "source_snapshot.json", snapshot)
        write_json(output / "prior_budget.json", history)
        write_json(output / "process.json", {"pid": os.getpid()})
        log = OperatorLog(output)
        client, index = None, None
        previous_key = os.environ.get(KEY_VARIABLE)
        reports, failure = [], None
        prediction_hashes = {}
        try:
            info = manifest["artifacts"]["corpus.jsonl"]
            index = SharedBM25Index(
                corpus.parent / "index.sqlite3", corpus, info["sha256"], info["rows"]
            )
            os.environ[KEY_VARIABLE] = settings.api_key
            config = ChatConfig(
                settings.base_url,
                PILOT_MODEL,
                KEY_VARIABLE,
                max_calls=len(chosen) * 16,
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
                PriceLimits(budget_cny=subcap, max_prompt_bytes=30000, max_elapsed_seconds=1800),
                output / "request_journal",
            )
            contract = HistoryContractClient(client, on_record=log)
            for question in chosen:
                log.question_id = question.question_id
                log({"kind": "question_start"})
                for arm in ARMS:
                    log.arm = arm
                    target = output / f"{question.question_id}_{arm}.json"
                    if (
                        _sha(corpus) != info["sha256"]
                        or _sha(project / "runs/history_foundation_20261001_v1/combined.json")
                        != LIBRARY_SHA256
                    ):
                        raise ValueError("runtime corpus/library bytes changed")
                    report = execute_arm(
                        question,
                        arm,
                        index,
                        contract,
                        library=library,
                        reference=reference,
                        trace=f"{run_id}/{question.question_id}/{arm}",
                        log=log,
                        target=target,
                        planner_class=profile["planner_class"],
                    )
                    reports.append(report)
                    prediction_hashes[target.name] = _sha(target)
                log({"kind": "question_completed"})
            if source_snapshot(project)["sha256"] != snapshot["sha256"]:
                raise ValueError("source changed during execution; preserve predictions for audit")
        except BaseException as error:
            failure = type(error).__name__
            if client is not None and not client.block_reason:
                client.block_reason = "experiment_execution_failure"
            log({"kind": "batch_stopped", "error_type": failure})
        finally:
            if index is not None:
                index.close()
            if previous_key is None:
                os.environ.pop(KEY_VARIABLE, None)
            else:
                os.environ[KEY_VARIABLE] = previous_key
            # 补登记已保存失败臂；未执行臂保持not_attempted，而非虚构零分。
            terminal = []
            for question in chosen:
                item = {"question_id": question.question_id, "arms": {}}
                for arm in ARMS:
                    path = output / f"{question.question_id}_{arm}.json"
                    if path.exists():
                        saved = json.loads(path.read_bytes())
                        prediction_hashes[path.name] = _sha(path)
                        item["arms"][arm] = {"status": saved["status"], "path": path.name}
                    else:
                        item["arms"][arm] = {"status": "not_attempted", "path": None}
                terminal.append(item)
            budget = (
                client.report()
                if client is not None
                else {
                    "api_requests": 0,
                    "calls": [],
                    "reserved_cny": 0,
                    "estimated_actual_cny": 0,
                    "limits": asdict(PriceLimits(budget_cny=subcap)),
                }
            )
            summary = {
                "protocol": profile["protocol"],
                "run_id": run_id,
                "status": "failed" if failure else "completed",
                "failure_type": failure,
                "planned_questions": len(chosen),
                "completed_questions": sum(
                    all(a["status"] == "completed" for a in row["arms"].values())
                    for row in terminal
                ),
                "terminal": terminal,
                "prediction_sha256": prediction_hashes,
                "source_sha256": snapshot["sha256"],
                "gold_loaded": False,
                "memory_updated": False,
                "new_operator_induction": False,
                "api_requests": budget["api_requests"],
                "reserved_cny": budget["reserved_cny"],
                "estimated_actual_cny": budget["estimated_actual_cny"],
            }
            write_json(output / "final_budget.json", budget)
            write_json(output / "SUMMARY.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 1 if failure else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--allow-network", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
