"""A0：先冻结100题和方法，再按5/20/25/25/25连续运行四路。

运行只接触问题和文档。首次5题是接口门，禁止按答案效果决定扩批。
已完成HTTP的局部格式错误保留为失败臂；不重试，可按预登记继续下一臂。
传输/用量异常、连续3个失败臂或接口门未通过时停止并封存全部缺失状态。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from growrag.history_library import FrozenHistoryLibrary

from .a0_budget import PREFIX, PROTOCOL, reviewed_history
from .a0_runtime import (
    FOCUSED_PLANNER_VERSION,
    METHODS,
    SHORT_READER_VERSION,
    A0ContractClient,
    A0LocalOutputError,
    execute_arm,
)
from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_dev_manifest import _sha
from .pre_pilot import write_json
from .prepare_a0_data import COUNT, load_bundle
from .prepare_a0_data import OUTPUT as BUNDLE
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_operator_study import OperatorLog, serial_lock
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient
from .shared_s2g_corpus import SharedBM25Index

BATCHES = ((0, 5), (5, 20), (25, 25), (50, 25), (75, 25))
FREEZE = "runs/a0_query_construction_freeze_v1.json"
OUTPUT = "runs/a0_query_construction_v1"
PROTOCOL_NOTE = "knowledge/experiments/2026-10-02_A0_查询构造对照预登记.md"
KEY_VARIABLE = "GROWRAG_A0_API_KEY"
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
    "reader_version": SHORT_READER_VERSION,
    "focused_version": FOCUSED_PLANNER_VERSION,
    "history_policy": "action_body_bm25_top8",
    "history_representation": "examples",
    "memory_updates": False,
    "project_cap_cny": 200.0,
    "series_cap_cny": 5.0,
    "primary_metric": "controlled_reader_first_by_fixed_method_order",
    "local_failure_policy": "completed_http_continue_no_retry_stop_after_3_consecutive_arms",
    "interface_gate": "first5_all4_methods_complete_before_expansion",
    "price_profile": "declared input0.2/output0.8 CNY per million; not invoice",
    "price_checked_date": "2026-10-02",
    "price_source": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
}


def freeze(project: Path) -> dict:
    """保存外部冻结证书；不含密钥、不读标签、不请求模型。"""
    project = project.resolve(strict=True)
    if (project / FREEZE).exists() or (project / OUTPUT).exists():
        raise FileExistsError("A0 protocol already frozen or started")
    git = _git_state()
    if not git["commit"] or git["worktree_dirty"]:
        raise ValueError("commit reviewed implementation before freezing")
    bundle_sha = _sha(project / BUNDLE / "manifest.json")
    manifest, _, _ = load_bundle(project, expected_sha=bundle_sha)
    snapshot = source_snapshot(project)
    value = {
        "protocol": PROTOCOL,
        "configuration": CONFIGURATION,
        "methods": list(METHODS),
        "batches": [list(pair) for pair in BATCHES],
        "bundle_sha256": bundle_sha,
        "question_ids": manifest["question_ids"],
        "library_sha256": manifest["library"]["sha256"],
        "source_sha256": snapshot["sha256"],
        "protocol_note_sha256": _sha(project / PROTOCOL_NOTE),
        "git": git,
        "gold_loaded": False,
        "api_calls": 0,
    }
    write_json(project / FREEZE, value)
    return {"freeze_path": FREEZE, "freeze_sha256": _sha(project / FREEZE), **value}


def load_freeze(project: Path, expected_sha: str):
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ValueError("externally recorded freeze SHA required")
    path = project / FREEZE
    if _sha(path) != expected_sha:
        raise ValueError("freeze certificate changed")
    value = json.loads(path.read_bytes())
    if (
        value.get("protocol") != PROTOCOL
        or value.get("configuration") != CONFIGURATION
        or value.get("methods") != list(METHODS)
        or value.get("batches") != [list(p) for p in BATCHES]
        or value.get("gold_loaded") is not False
        or value.get("source_sha256") != source_snapshot(project)["sha256"]
        or value.get("protocol_note_sha256") != _sha(project / PROTOCOL_NOTE)
    ):
        raise ValueError("method or prelabel protocol differs from freeze")
    manifest, questions, corpus = load_bundle(project, expected_sha=value["bundle_sha256"])
    if (
        value["question_ids"] != manifest["question_ids"]
        or len(questions) != COUNT
        or value["library_sha256"] != manifest["library"]["sha256"]
    ):
        raise ValueError("frozen cohort identity changed")
    return value, manifest, questions, corpus


def recoverable_local_failure(error, client, start_call):
    """仅限已计费、已记录的本地模型输出失败；不能清除预算/传输阻断。"""
    calls = client.calls[start_call:]
    return (
        isinstance(error, A0LocalOutputError)
        and not client.block_reason
        and bool(calls)
        and all(
            c.get("status") == "completed"
            and c.get("api_requests") == 1
            and type(c.get("estimated_actual_cny")) in (int, float)
            and math.isfinite(c["estimated_actual_cny"])
            and c["estimated_actual_cny"] >= 0
            and c.get("validation_status") is None
            for c in calls
        )
    )


def terminal_rows(output, questions):
    rows, hashes = [], {}
    for q in questions:
        arms = {}
        for method in METHODS:
            path = output / f"{q.question_id}_{method}.json"
            if path.is_file():
                row = json.loads(path.read_bytes())
                if row["question_id"] != q.question_id or row["method"] != method:
                    raise ValueError("terminal report identity mismatch")
                if (
                    row.get("status") not in {"completed", "failed"}
                    or row.get("gold_loaded") is not False
                    or row.get("memory_updated") is not False
                ):
                    raise ValueError("invalid terminal status or runtime isolation flags")
                hashes[path.name] = _sha(path)
                arms[method] = {"status": row["status"], "path": path.name}
            else:
                arms[method] = {"status": "not_attempted", "path": None}
        rows.append({"question_id": q.question_id, "arms": arms})
    return rows, hashes


def reject_claim_overlap(runs, question_ids):
    """检查冻结以后别的运行是否占用了本批题；只看无标签claim元数据。"""
    selected = set(question_ids)
    for path in Path(runs).glob("*.claim.json"):
        value = json.loads(path.read_bytes())
        ids = value.get("question_ids", [])
        if type(ids) is not list or any(type(qid) is not str for qid in ids):
            raise ValueError("invalid prior claim identity metadata")
        if selected.intersection(ids):
            raise ValueError("A0 cohort overlaps an already claimed paid run")


class FrozenInputGuard:
    """每次调用前检查文件集合/属性；有改动就重新核SHA，结束时全量核SHA。"""

    def __init__(self, project, frozen, manifest, freeze_sha):
        self.project = project
        snapshot = source_snapshot(project)
        if snapshot["sha256"] != frozen["source_sha256"]:
            raise ValueError("source changed before guarded run")
        self.sources = set(snapshot["files"])
        self.hashes = {name: row["sha256"] for name, row in snapshot["files"].items()}
        self.hashes.update(
            {
                FREEZE: freeze_sha,
                PROTOCOL_NOTE: frozen["protocol_note_sha256"],
                f"{BUNDLE}/manifest.json": frozen["bundle_sha256"],
                f"{BUNDLE}/{manifest['runtime_artifact']['path']}": manifest["runtime_artifact"][
                    "sha256"
                ],
                manifest["corpus_ref"]["path"]: manifest["corpus_ref"]["sha256"],
                manifest["corpus_ref"]["index_path"]: manifest["corpus_ref"]["index_sha256"],
                manifest["library"]["path"]: frozen["library_sha256"],
            }
        )
        self.stamps = {}
        self.check(force=True)

    def check(self, *, force=False):
        sources = {
            p.relative_to(self.project).as_posix()
            for p in (self.project / "src").rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}
        }
        if sources != self.sources:
            raise ValueError("source file set changed during guarded run")
        for name, digest in self.hashes.items():
            path = self.project / name
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(self.project):
                raise ValueError("frozen input escapes project")
            stat = path.stat()
            stamp = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
            if force or self.stamps.get(name) != stamp:
                if _sha(path) != digest:
                    raise ValueError("frozen input bytes changed during guarded run")
                self.stamps[name] = stamp


class GuardedBudgetClient(DurableBudgetClient):
    def __init__(self, *args, input_guard, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_guard = input_guard

    def complete(self, messages, *, trace_id, prompt_version):
        self.input_guard.check()
        return super().complete(messages, trace_id=trace_id, prompt_version=prompt_version)


def run_batch(
    project,
    frozen,
    manifest,
    questions,
    corpus,
    start,
    count,
    freeze_sha,
    *,
    previous_consecutive_failures=0,
):
    """锁内串行一次性执行，所有终态与实际请求账本即使异常也落盘。"""
    runs = project / "runs"
    run_id = f"{PREFIX}v1_{start:04d}_{start + count:04d}"
    output = runs / run_id
    claim_path = runs / f"{run_id}.claim.json"
    if output.exists() or claim_path.exists():
        raise FileExistsError("batch already claimed; no replay")
    chosen = questions[start : start + count]
    history = reviewed_history(runs)
    reserved = sum(c["reserved_cny"] for c in history["calls"] if c["trace_id"].startswith(PREFIX))
    cap = min(
        CONFIGURATION["series_cap_cny"] - reserved,
        CONFIGURATION["project_cap_cny"] - history["prior_reserved_cny"],
    )
    if cap <= 0:
        raise ValueError("registered series/project budget exhausted")
    settings = read_local_bailian_settings(project / "qwenAPI.md")
    host = urlsplit(settings.base_url).hostname or ""
    if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
        raise ValueError("Beijing price profile does not match endpoint")
    plan = {
        **CONFIGURATION,
        "protocol": PROTOCOL,
        "run_id": run_id,
        "phase": "a0_development",
        "question_ids": [q.question_id for q in chosen],
        "methods": list(METHODS),
        "start": start,
        "count": count,
        "freeze_sha256": freeze_sha,
        "source_sha256": frozen["source_sha256"],
        "bundle_sha256": frozen["bundle_sha256"],
        "library_sha256": frozen["library_sha256"],
        "gold_loaded": False,
        "prior_reserved_cny": history["prior_reserved_cny"],
        "subcap_cny": cap,
    }
    write_json(claim_path, {**plan, "plan_sha256": fingerprint(plan)})
    output.mkdir(exist_ok=False)
    write_json(output / "launch_plan.json", plan)
    write_json(output / "source_snapshot.json", source_snapshot(project))
    write_json(output / "prior_budget.json", history)
    write_json(output / "process.json", {"pid": os.getpid()})
    log = OperatorLog(output)
    client, index, stop = None, None, None
    consecutive = previous_consecutive_failures
    previous_key = os.environ.get(KEY_VARIABLE)
    try:
        guard = FrozenInputGuard(project, frozen, manifest, freeze_sha)
        library = FrozenHistoryLibrary.from_json(
            (project / manifest["library"]["path"]).read_text(encoding="utf-8")
        )
        index = SharedBM25Index(
            corpus.parent / "index.sqlite3", corpus, manifest["corpus_ref"]["sha256"], 76691
        )
        os.environ[KEY_VARIABLE] = settings.api_key
        config = ChatConfig(
            settings.base_url,
            PILOT_MODEL,
            KEY_VARIABLE,
            max_calls=count * 16,
            max_output_tokens=2048,
            timeout_seconds=60,
            output_limit_parameter="max_tokens",
            enable_thinking=False,
            temperature=0,
            top_p=1,
            json_object_mode=True,
        )
        client = GuardedBudgetClient(
            LiveChatClient(config, output / "api_audit", allow_network=True),
            PriceLimits(budget_cny=cap, max_prompt_bytes=30000, max_elapsed_seconds=3600),
            output / "request_journal",
            input_guard=guard,
        )
        contract = A0ContractClient(client, on_record=log)
        for question in chosen:
            log.question_id = question.question_id
            log({"kind": "question_start"})
            for method in METHODS:
                log.arm = method
                before = len(client.calls)
                try:
                    guard.check()
                    execute_arm(
                        question,
                        method,
                        index,
                        contract,
                        library=library,
                        trace=f"{run_id}/{question.question_id}/{method}",
                        log=log,
                        target=output / f"{question.question_id}_{method}.json",
                    )
                    consecutive = 0
                except Exception as error:
                    consecutive += 1
                    log({"kind": "arm_failed", "error_type": type(error).__name__})
                    if not recoverable_local_failure(error, client, before) or consecutive >= 3:
                        stop = client.block_reason or "consecutive_or_nonlocal_failure"
                        break
            if stop:
                break
            log({"kind": "question_terminal"})
        guard.check(force=True)
    except BaseException as error:
        stop = type(error).__name__
        log({"kind": "batch_stopped", "error_type": stop})
    finally:
        if index is not None:
            index.close()
        if previous_key is None:
            os.environ.pop(KEY_VARIABLE, None)
        else:
            os.environ[KEY_VARIABLE] = previous_key
        terminal, hashes = terminal_rows(output, chosen)
        budget = (
            client.report()
            if client
            else {
                "api_requests": 0,
                "calls": [],
                "reserved_cny": 0,
                "estimated_actual_cny": 0,
                "limits": asdict(PriceLimits(budget_cny=cap)),
            }
        )
        summary = {
            "protocol": PROTOCOL,
            "run_id": run_id,
            "status": "stopped" if stop else "completed",
            "stop_reason": stop,
            "trailing_consecutive_failed_arms": consecutive,
            "terminal": terminal,
            "prediction_sha256": hashes,
            "complete_paired_questions": sum(
                all(a["status"] == "completed" for a in r["arms"].values()) for r in terminal
            ),
            "source_sha256": frozen["source_sha256"],
            "gold_loaded": False,
            "memory_updated": False,
            "api_requests": budget["api_requests"],
            "reserved_cny": budget["reserved_cny"],
            "estimated_actual_cny": budget["estimated_actual_cny"],
        }
        write_json(output / "final_budget.json", budget)
        write_json(output / "SUMMARY.json", summary)
    return summary


def run(project: Path, expected_sha: str, *, allow_network=False):
    frozen, manifest, questions, corpus = load_freeze(project, expected_sha)
    if not allow_network:
        return {"dry_run": True, "planned_questions": COUNT, "methods": METHODS, "api_calls": 0}
    aggregate = project / OUTPUT
    if aggregate.exists():
        raise FileExistsError("cohort already started; no implicit resume")
    if _git_state()["worktree_dirty"]:
        raise ValueError("commit implementation before paid calls")
    summaries, stop, consecutive = {}, None, 0
    with serial_lock(project / "runs"):
        reject_claim_overlap(project / "runs", frozen["question_ids"])
        aggregate.mkdir(exist_ok=False)
        write_json(
            aggregate / "launch_plan.json",
            {
                "protocol": PROTOCOL,
                "question_ids": frozen["question_ids"],
                "methods": list(METHODS),
                "freeze_sha256": expected_sha,
                "gold_loaded": False,
            },
        )
        try:
            for start, count in BATCHES:
                load_freeze(project, expected_sha)
                summary = run_batch(
                    project,
                    frozen,
                    manifest,
                    questions,
                    corpus,
                    start,
                    count,
                    expected_sha,
                    previous_consecutive_failures=consecutive,
                )
                consecutive = summary["trailing_consecutive_failed_arms"]
                name = f"{PREFIX}v1_{start:04d}_{start + count:04d}"
                summaries[name] = _sha(project / "runs" / name / "SUMMARY.json")
                print(
                    json.dumps(
                        {
                            "batch": name,
                            "complete_paired_questions": summary["complete_paired_questions"],
                            "status": summary["status"],
                            "api_requests": summary["api_requests"],
                        }
                    ),
                    flush=True,
                )
                if (
                    summary["status"] != "completed"
                    or start == 0
                    and summary["complete_paired_questions"] != 5
                ):
                    stop = summary["stop_reason"] or "first5_interface_gate_failed"
                    break
        except BaseException as error:
            stop = type(error).__name__
        finally:
            terminal = []
            for start, count in BATCHES:
                name = f"{PREFIX}v1_{start:04d}_{start + count:04d}"
                rows, _ = terminal_rows(project / "runs" / name, questions[start : start + count])
                terminal.extend(rows)
            final = {
                "protocol": PROTOCOL,
                "planned_questions": COUNT,
                "status": "stopped" if stop else "completed",
                "stop_reason": stop,
                "freeze_sha256": expected_sha,
                "batch_summary_sha256": summaries,
                "terminal": terminal,
                "gold_loaded": False,
                "memory_updated": False,
            }
            write_json(aggregate / "TERMINAL.json", final)
    return {"status": final["status"], "stop_reason": stop, "output": OUTPUT}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--expected-freeze-sha256")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze and (args.allow_network or args.expected_freeze_sha256):
        parser.error("freeze is an offline standalone step")
    project = Path.cwd().resolve(strict=True)
    result = (
        freeze(project)
        if args.freeze
        else run(project, args.expected_freeze_sha256, allow_network=args.allow_network)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 1 if result.get("status") == "stopped" else 0


if __name__ == "__main__":
    raise SystemExit(main())
