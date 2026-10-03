"""Once-only, two-stage A1 development diagnostics; no natural questions or gold.

定位和判断是两个实际请求，分别留账。局部输出错误不吞掉后续独立样本，
但不重试、不修补引用、不把失败算作 unknown。旧批次与旧评分不修改。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

from . import a1_catalog_study as previous
from . import a1_two_stage_observer as observer
from .a1_conditions import prepare_payload
from .a1_two_stage_budget import PREFIX, PROTOCOL, reviewed_history
from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_dev_manifest import _sha
from .pre_pilot import write_json
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_operator_study import serial_lock
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient

FIXTURE = "experiments/fixtures/a1_conditions_two_stage_v1.json"
# The manifest references the immutable parent instead of duplicating its 32 rows.
FIXTURE_SHA = "5343a17d2c63f1cf8577eca730a8d209c3c54481ab3d7bc79a098453b3fbf1fc"
NOTE = "docs/experiments/2026-10-03_A1两阶段观察预登记.md"
FREEZE = "runs/a1_two_stage_freeze_v1.json"
RUN_ID = PREFIX + "synthetic_v1"
OUTPUT = "runs/" + RUN_ID
FEEDBACK = "runs/a1_two_stage_feedback_v1"
KEY_VARIABLE = "GROWRAG_A1_TWO_STAGE_API_KEY"
CONFIG = {
    "protocol": PROTOCOL,
    "model": PILOT_MODEL,
    "max_output_tokens": 2048,
    "temperature": 0,
    "top_p": 1,
    "enable_thinking": False,
    "json_object_mode": True,
    "json_schema_mode": False,
    "output_limit_parameter": "max_tokens",
    "locator_prompt_version": observer.LOCATOR_PROMPT_VERSION,
    "judge_prompt_version": observer.JUDGE_PROMPT_VERSION,
    "project_cap_cny": 50.0,
    "series_cap_cny": 1.0,
    "handoff_cap_cny": 5.0,
    "planned_rows": 32,
    "unique_inputs": 30,
    "max_calls": 64,
    "retry": False,
    "gold_loaded": False,
    "memory_updated": False,
    "cost_notice": "Declared input0.2/output0.8 CNY/million; not provider invoice",
    "suite_role": "development_interface_regression_not_blind_test",
    "collection_policy": "local_output_failure_continue_fatal_stop_v1",
}

input_sha = previous.input_sha
final_budget_report = previous.final_budget_report


def load_fixture(project):
    """Only the known development suite is read; labels never enter messages."""
    path = project / FIXTURE
    if _sha(path) != FIXTURE_SHA:
        raise ValueError("registered two-stage fixture manifest changed")
    manifest = json.loads(path.read_bytes())
    if (
        manifest["parent_fixture"] != previous.FIXTURE
        or manifest["parent_fixture_sha256"] != previous.FIXTURE_SHA
        or manifest["planned_rows"] != 32
        or manifest["unique_inputs"] != 30
    ):
        raise ValueError("parent fixture identity differs")
    return previous.load_fixture(project)


def protocol_identity(project):
    rows = load_fixture(project)
    return {
        "configuration": CONFIG,
        "fixture_sha256": FIXTURE_SHA,
        "parent_fixture_sha256": previous.FIXTURE_SHA,
        "source_sha256": source_snapshot(project)["sha256"],
        "protocol_note_sha256": _sha(project / NOTE),
        "locator_prompt_sha256": observer.LOCATOR_PROMPT_SHA256,
        "judge_prompt_sha256": observer.JUDGE_PROMPT_SHA256,
        "row_ids": [r["row_id"] for r in rows],
        "locator_messages_sha256": {
            r["row_id"]: input_sha(observer.locator_messages(prepare_payload(r["input"])))
            for r in rows
        },
    }


def freeze(project):
    if (project / FREEZE).exists() or (project / OUTPUT).exists():
        raise FileExistsError("two-stage batch already frozen/started")
    git = _git_state()
    if not git["commit"] or git["worktree_dirty"]:
        raise ValueError("commit reviewed implementation before freezing")
    write_json(project / FREEZE, {**protocol_identity(project), "git": git, "api_calls": 0})
    return {"freeze_path": FREEZE, "freeze_sha256": _sha(project / FREEZE)}


def load_freeze(project, sha):
    if type(sha) is not str or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("external freeze SHA required")
    if _sha(project / FREEZE) != sha:
        raise ValueError("freeze identity changed")
    value = json.loads((project / FREEZE).read_bytes())
    if any(value.get(k) != v for k, v in protocol_identity(project).items()):
        raise ValueError("source, prompts, fixture or protocol differs from freeze")
    return value


def progress(directory, event, client=None, *, started=None, **fields):
    """进度不包含密钥；原始请求另由持久账本保存。"""
    budget = client.report() if client is not None and hasattr(client, "report") else {}
    if budget.get("api_requests", 0) != len(budget.get("calls", [])):
        budget["estimated_actual_cny"] = None  # Interrupted attempt is not a zero-cost success.
    value = {
        "time_unix": time.time(),
        "run_id": RUN_ID,
        "event": event,
        "api_requests": budget.get("api_requests", 0),
        "estimated_actual_cny": budget.get("estimated_actual_cny"),
        "reserved_cny": budget.get("reserved_cny", 0),
        "elapsed_seconds": None if started is None else time.monotonic() - started,
        **fields,
    }
    with (directory / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def stage_request(record, stage, messages, client, guard, directory, started):
    """每个stage只有一次physical调用。请求前后都核对冻结身份。"""
    guard()
    version = (
        observer.LOCATOR_PROMPT_VERSION if stage == "locate" else observer.JUDGE_PROMPT_VERSION
    )
    value = {
        "trace_id": f"{RUN_ID}/row/{record['sequence']:02d}/{stage}",
        "prompt_version": version,
        "messages": messages,
        "messages_sha256": input_sha(messages),
        "status": "started",
    }
    record["stages"][stage] = value
    progress(
        directory, "stage_started", client, started=started, row_id=record["row_id"], stage=stage
    )
    try:
        response = client.complete(messages, trace_id=value["trace_id"], prompt_version=version)
        value["raw_content"] = response.content
        value["raw_response_sha256"] = hashlib.sha256(response.content.encode()).hexdigest()
        value["status"] = "completed"
        guard()
    except BaseException as error:
        value["status"] = "failed"
        value["failure_type"] = type(error).__name__
        raise
    return response.content


def observe_rows(rows, client, directory, guard, *, terminal=None):
    terminal = [] if terminal is None else terminal
    stop, started = None, time.monotonic()
    for index, row in enumerate(rows):
        if stop:
            terminal.append({"row_id": row["row_id"], "status": "not_attempted"})
            continue
        record = {
            "row_id": row["row_id"],
            "sequence": index,
            "input_sha256": input_sha(row["input"]),
            "status": "started",
            "stages": {},
        }
        try:
            guard()
            prepared = prepare_payload(row["input"])
            raw = stage_request(
                record,
                "locate",
                observer.locator_messages(prepared),
                client,
                guard,
                directory,
                started,
            )
            location = observer.resolve_location(raw, prepared)
            record["location"] = location.to_dict()
            raw = stage_request(
                record,
                "judge",
                observer.judge_messages(prepared, location),
                client,
                guard,
                directory,
                started,
            )
            report, audit = observer.resolve_judgment(raw, prepared, location)
            audit.verify_report(report)
            record.update(
                report=report.to_dict(),
                two_stage_audit=audit.to_dict(),
                observation_status="valid",
                status="completed",
                fatal=False,
            )
            guard()
        except observer.TwoStageContractError as error:
            # HTTP已经成功且费用由transport核验；这里只继续独立后续样本。
            record.update(
                status="failed",
                failure_type=type(error).__name__,
                failure_code=error.code,
                observation_status=error.observation_status,
                failure_stage=error.stage,
                fatal=False,
            )
            if error.audit is not None:
                record["two_stage_audit"] = error.audit.to_dict()
            try:
                guard()  # 不能以局部输出错误掩盖同时发生的冻结身份漂移。
            except BaseException as fatal:
                record.update(fatal=True, integrity_failure_type=type(fatal).__name__)
                stop = type(fatal).__name__
        except BaseException as error:
            record.update(
                status="failed",
                failure_type=type(error).__name__,
                fatal=True,
                failure_stage=next(reversed(record["stages"]), "guard"),
                failure_code=getattr(client, "block_reason", None) or "execution_failure",
                observation_status="integrity_error",
            )
            stop = type(error).__name__
        path = directory / f"row_{index:02d}.json"
        entry = {
            "row_id": row["row_id"],
            "status": "failed",
            "path": path.name,
            "artifact_status": "not_sealed",
        }
        terminal.append(entry)
        write_json(path, record)
        entry.pop("artifact_status")
        entry.update(status=record["status"], sha256=_sha(path))
        progress(
            directory,
            "row_terminal",
            client,
            started=started,
            row_id=row["row_id"],
            sequence=index,
            status=record["status"],
            stop=stop,
        )
    return terminal, stop


def run(project, sha, *, allow_network=False):
    frozen = load_freeze(project, sha)
    rows = load_fixture(project)
    if not allow_network:
        return {"dry_run": True, "planned_rows": len(rows), "max_calls": 64, "api_calls": 0}
    output = project / OUTPUT
    claim = project / "runs" / f"{RUN_ID}.claim.json"
    if output.exists() or claim.exists():
        raise FileExistsError("two-stage batch already claimed; no retry")
    if _git_state()["worktree_dirty"]:
        raise ValueError("commit reviewed implementation before paid calls")
    with serial_lock(project / "runs"):
        # 保守50元为当前代码默认值；只有真实用户授权核实后才能新登记更高值。
        print(json.dumps({"event": "budget_reconciliation_started", "run_id": RUN_ID}), flush=True)
        history = reviewed_history(project / "runs")
        cap = min(
            CONFIG["series_cap_cny"], CONFIG["project_cap_cny"] - history["prior_reserved_cny"]
        )
        if cap <= 0:
            raise ValueError("verified project conservative budget exhausted; authorization needed")
        settings = read_local_bailian_settings(project / "qwenAPI.md")
        host = urlsplit(settings.base_url).hostname or ""
        if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
            raise ValueError("endpoint does not match declared Beijing price")
        plan = {
            **CONFIG,
            "run_id": RUN_ID,
            "freeze_sha256": sha,
            "prior_reserved_cny": history["prior_reserved_cny"],
            "subcap_cny": cap,
            "question_ids": [],
            "row_ids": frozen["row_ids"],
        }
        write_json(claim, {**plan, "plan_sha256": fingerprint(plan)})
        output.mkdir(exist_ok=False)
        write_json(output / "launch_plan.json", plan)
        write_json(output / "prior_budget.json", history)
        write_json(output / "source_snapshot.json", source_snapshot(project))
        progress(
            output,
            "budget_reconciliation_completed",
            prior_reserved_cny=history["prior_reserved_cny"],
        )
        client, stop, terminal = None, None, []
        previous_key = os.environ.get(KEY_VARIABLE)
        try:
            os.environ[KEY_VARIABLE] = settings.api_key
            config = ChatConfig(
                settings.base_url,
                PILOT_MODEL,
                KEY_VARIABLE,
                max_calls=64,
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
                PriceLimits(budget_cny=cap, max_prompt_bytes=30000, max_elapsed_seconds=1800),
                output / "request_journal",
            )
            terminal, stop = observe_rows(
                rows, client, output, lambda: load_freeze(project, sha), terminal=terminal
            )
        except BaseException as error:
            stop = type(error).__name__
        finally:
            if previous_key is None:
                os.environ.pop(KEY_VARIABLE, None)
            else:
                os.environ[KEY_VARIABLE] = previous_key
            terminal.extend(
                {"row_id": r["row_id"], "status": "not_attempted"} for r in rows[len(terminal) :]
            )
            write_json(output / "final_budget.json", final_budget_report(client, cap))
            value = {
                "protocol": PROTOCOL,
                "status": "stopped" if stop else "completed",
                "stop_reason": stop,
                "rows": terminal,
                "freeze_sha256": sha,
                "budget_sha256": _sha(output / "final_budget.json"),
                "launch_sha256": _sha(output / "launch_plan.json"),
                "prior_sha256": _sha(output / "prior_budget.json"),
                "source_snapshot_sha256": _sha(output / "source_snapshot.json"),
                "claim_sha256": _sha(claim),
                "gold_loaded": False,
                "memory_updated": False,
            }
            write_json(output / "TERMINAL.json", value)
            progress(output, "batch_terminal", client, status=value["status"], stop=stop)
    return {
        "status": value["status"],
        "stop_reason": stop,
        "terminal_sha256": _sha(output / "TERMINAL.json"),
        "output": OUTPUT,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--expected-freeze-sha256")
    parser.add_argument("--expected-terminal-sha256")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze and (args.score or args.allow_network or args.expected_freeze_sha256):
        parser.error("freeze is standalone")
    if args.score and (args.allow_network or not args.expected_terminal_sha256):
        parser.error("scoring is offline and requires external terminal SHA")
    project = Path.cwd().resolve(strict=True)
    if args.freeze:
        result = freeze(project)
    elif args.score:
        from .a1_two_stage_feedback import score

        result = score(project, args.expected_freeze_sha256, args.expected_terminal_sha256)
    else:
        result = run(project, args.expected_freeze_sha256, allow_network=args.allow_network)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
