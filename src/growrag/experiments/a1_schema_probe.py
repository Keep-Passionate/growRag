"""One genuine locator compatibility probe; never a substitute for the A1 gate.

代码在独立worktree，账本仍在原项目。两个根显式分离，避免把新代码错记为
原源码。只运行旧固定顺序第一行一次：不判断适用性、不检索、不读自然gold。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from . import a1_two_stage_observer as observer
from . import a1_two_stage_study as parent
from .a0_v2_feedback import audit_http, totals
from .a1_conditions import prepare_payload
from .a1_two_stage_budget import reviewed_history
from .a1_two_stage_feedback import audit_journal
from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_dev_manifest import _sha
from .output_schemas import response_format_for
from .pre_pilot import write_json
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_operator_study import serial_lock
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient

CODE_PROJECT = Path(__file__).resolve().parents[3]
PREFIX = "2026-10-03_a1schema_"
PROTOCOL = "growrag-a1-schema-probe-v1"
RUN_ID = PREFIX + "locator_probe_v1"
OUTPUT = "runs/" + RUN_ID
FREEZE = "runs/a1_schema_probe_freeze_v1.json"
NOTE = "docs/experiments/2026-10-03_A路线_结构输出兼容探针.md"
KEY_VARIABLE = "GROWRAG_A1_SCHEMA_PROBE_API_KEY"
CONFIG = {
    "protocol": PROTOCOL,
    "model": PILOT_MODEL,
    "max_output_tokens": 2048,
    "max_calls": 1,
    "temperature": 0,
    "top_p": 1,
    "enable_thinking": False,
    "json_schema_mode": True,
    "json_object_mode": False,
    "output_limit_parameter": "max_tokens",
    "project_cap_cny": None,
    "batch_cap_cny": 1.0,
    "retry": False,
    "purpose": "one_known_development_input_format_compatibility_not_semantic_accuracy",
    "gold_loaded": False,
    "memory_updated": False,
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_bytes())


def probe_input():
    """First old registered row, not cherry-picking a newly opened test example."""
    row = parent.load_fixture(CODE_PROJECT)[0]
    prepared = prepare_payload(row["input"])
    return row["row_id"], prepared, observer.locator_messages(prepared)


def identity(artifact_project):
    artifact_project = Path(artifact_project).resolve(strict=True)
    row_id, _, messages = probe_input()
    return {
        "configuration": CONFIG,
        "code_project": str(CODE_PROJECT),
        "artifact_project": str(artifact_project),
        "source_sha256": source_snapshot(CODE_PROJECT)["sha256"],
        "fixture_sha256": parent.FIXTURE_SHA,
        "parent_fixture_sha256": parent.previous.FIXTURE_SHA,
        "protocol_note_sha256": _sha(CODE_PROJECT / NOTE),
        "budget_authorization_sha256": _sha(CODE_PROJECT / parent.AUTHORIZATION),
        "locator_prompt_sha256": observer.LOCATOR_PROMPT_SHA256,
        "row_id": row_id,
        "messages_sha256": fingerprint(messages),
        "response_format": response_format_for(observer.LOCATOR_PROMPT_VERSION),
    }


def freeze(artifact_project):
    project = Path(artifact_project).resolve(strict=True)
    _require(Path.cwd().resolve() == CODE_PROJECT, "run from the code worktree, not artifact root")
    git = _git_state()
    _require(git["commit"] and not git["worktree_dirty"], "commit code/protocol before freezing")
    _require(not (project / OUTPUT).exists(), "probe already started")
    value = {**identity(project), "git": git, "api_calls": 0}
    write_json(project / FREEZE, value)
    return {"freeze_path": str(project / FREEZE), "freeze_sha256": _sha(project / FREEZE)}


def load_freeze(artifact_project, expected_sha):
    project = Path(artifact_project).resolve(strict=True)
    _require(type(expected_sha) is str and len(expected_sha) == 64, "external freeze SHA required")
    _require(_sha(project / FREEZE) == expected_sha, "freeze SHA differs")
    value = _read(project / FREEZE)
    _require(
        all(value.get(key) == content for key, content in identity(project).items()),
        "probe code, schema, fixture, authority or input changed",
    )
    return value


def run(artifact_project, expected_sha, *, allow_network=False):
    project = Path(artifact_project).resolve(strict=True)
    frozen = load_freeze(project, expected_sha)
    if not allow_network:
        return {"dry_run": True, "max_calls": 1, "row_id": frozen["row_id"], "api_calls": 0}
    _require(Path.cwd().resolve() == CODE_PROJECT, "paid run requires the explicit code worktree")
    _require(not _git_state()["worktree_dirty"], "commit implementation before paid run")
    output, claim = project / OUTPUT, project / "runs" / (RUN_ID + ".claim.json")
    _require(not output.exists() and not claim.exists(), "once-only probe already claimed")
    with serial_lock(project / "runs"):
        print(json.dumps({"event": "budget_reconciliation_started", "run_id": RUN_ID}), flush=True)
        # 注册新独立protocol，旧账/20未知完整继承；不能从新worktree空runs起新账。
        prior = reviewed_history(
            project / "runs", reviewed_other_series={PREFIX: frozenset({PROTOCOL})}
        )
        settings = read_local_bailian_settings(project / "qwenAPI.md")
        host = urlsplit(settings.base_url).hostname or ""
        _require(
            host == "dashscope.aliyuncs.com" or host.endswith(".cn-beijing.maas.aliyuncs.com"),
            "endpoint differs from declared Beijing price",
        )
        plan = {
            **CONFIG,
            "run_id": RUN_ID,
            "freeze_sha256": expected_sha,
            "row_ids": [frozen["row_id"]],
            "question_ids": [],
            "prior_reserved_cny": prior["prior_reserved_cny"],
            "subcap_cny": CONFIG["batch_cap_cny"],
            "code_project": str(CODE_PROJECT),
            "artifact_project": str(project),
            "output_schema_sha256": fingerprint(frozen["response_format"]),
        }
        write_json(claim, {**plan, "plan_sha256": fingerprint(plan)})
        output.mkdir(exist_ok=False)
        write_json(output / "launch_plan.json", plan)
        write_json(output / "prior_budget.json", prior)
        write_json(output / "source_snapshot.json", source_snapshot(CODE_PROJECT))
        _, prepared, messages = probe_input()
        client, error_type, record = None, None, None
        previous_key = os.environ.get(KEY_VARIABLE)
        try:
            os.environ[KEY_VARIABLE] = settings.api_key
            config = ChatConfig(
                settings.base_url,
                PILOT_MODEL,
                KEY_VARIABLE,
                max_calls=1,
                max_output_tokens=2048,
                timeout_seconds=60,
                output_limit_parameter="max_tokens",
                enable_thinking=False,
                temperature=0,
                top_p=1,
                json_schema_mode=True,
            )
            client = DurableBudgetClient(
                LiveChatClient(config, output / "api_audit", allow_network=True),
                PriceLimits(budget_cny=1.0, max_prompt_bytes=30000, max_elapsed_seconds=1800),
                output / "request_journal",
            )
            load_freeze(project, expected_sha)
            response = client.complete(
                messages,
                trace_id=RUN_ID + "/locator",
                prompt_version=observer.LOCATOR_PROMPT_VERSION,
            )
            record = {
                "row_id": frozen["row_id"],
                "messages": messages,
                "raw_content": response.content,
                "status": "completed",
            }
            try:
                location = observer.resolve_location(response.content, prepared)
                record["location_report"] = location.to_dict()
            except observer.TwoStageContractError as error:
                record.update(status="model_output_error", failure_code=error.code)
        except BaseException as error:
            error_type = type(error).__name__
            record = {"row_id": frozen["row_id"], "status": "stopped", "error_type": error_type}
        finally:
            if previous_key is None:
                os.environ.pop(KEY_VARIABLE, None)
            else:
                os.environ[KEY_VARIABLE] = previous_key
            write_json(output / "probe.json", record)
            write_json(output / "final_budget.json", parent.final_budget_report(client, 1.0))
            terminal = {
                "protocol": PROTOCOL,
                "status": record["status"],
                "error_type": error_type,
                "freeze_sha256": expected_sha,
                "record_sha256": _sha(output / "probe.json"),
                "budget_sha256": _sha(output / "final_budget.json"),
                "plan_sha256": _sha(output / "launch_plan.json"),
                "prior_sha256": _sha(output / "prior_budget.json"),
                "source_snapshot_sha256": _sha(output / "source_snapshot.json"),
                "claim_sha256": _sha(claim),
                "gold_loaded": False,
                "memory_updated": False,
            }
            write_json(output / "TERMINAL.json", terminal)
    return {
        "output": str(output),
        "status": terminal["status"],
        "terminal_sha256": _sha(output / "TERMINAL.json"),
    }


def audit(artifact_project, freeze_sha, terminal_sha):
    """Read original strict HTTP; never normalize it into a JSON-object request."""
    project = Path(artifact_project).resolve(strict=True)
    frozen = load_freeze(project, freeze_sha)
    directory = project / OUTPUT
    _require(_sha(directory / "TERMINAL.json") == terminal_sha, "external terminal SHA differs")
    terminal = _read(directory / "TERMINAL.json")
    for name, key in (
        ("probe.json", "record_sha256"),
        ("final_budget.json", "budget_sha256"),
        ("launch_plan.json", "plan_sha256"),
        ("prior_budget.json", "prior_sha256"),
        ("source_snapshot.json", "source_snapshot_sha256"),
    ):
        _require(_sha(directory / name) == terminal[key], "probe artifact seal differs")
    _require(
        terminal["freeze_sha256"] == freeze_sha and terminal["protocol"] == PROTOCOL,
        "terminal identity differs",
    )
    ledger, plan, record = (
        _read(directory / name) for name in ("final_budget.json", "launch_plan.json", "probe.json")
    )
    _require(
        all(plan.get(key) == value for key, value in CONFIG.items()), "plan configuration differs"
    )
    prior = _read(directory / "prior_budget.json")
    _require(
        plan.get("run_id") == RUN_ID
        and plan.get("freeze_sha256") == freeze_sha
        and plan.get("row_ids") == [frozen["row_id"]]
        and plan.get("question_ids") == []
        and plan.get("code_project") == frozen["code_project"] == str(CODE_PROJECT)
        and plan.get("artifact_project") == frozen["artifact_project"] == str(project)
        and plan.get("subcap_cny") == CONFIG["batch_cap_cny"]
        and plan.get("prior_reserved_cny") == prior.get("prior_reserved_cny")
        and plan.get("output_schema_sha256") == fingerprint(frozen["response_format"])
        and record.get("row_id") == frozen["row_id"]
        and record.get("status") == terminal.get("status")
        and terminal.get("gold_loaded") is False
        and terminal.get("memory_updated") is False,
        "plan, budget history, row or terminal identity differs",
    )
    claim = project / "runs" / (RUN_ID + ".claim.json")
    _require(
        _sha(claim) == terminal["claim_sha256"]
        and _read(claim) == {**plan, "plan_sha256": fingerprint(plan)},
        "claim differs",
    )
    snapshot = _read(directory / "source_snapshot.json")
    _require(
        snapshot["sha256"] == frozen["source_sha256"] == fingerprint(snapshot["files"]),
        "executed source snapshot differs",
    )
    calls = ledger["calls"]
    _require(len(calls) == ledger["api_requests"] == 1, "not exactly one real probe request")
    amounts = totals(calls)
    for key in (
        "api_requests",
        "reserved_cny",
        "estimated_actual_cny",
        "input_tokens",
        "output_tokens",
    ):
        _require(ledger[key] == amounts[key], "probe ledger totals differ")
    _require(amounts["reserved_cny"] <= CONFIG["batch_cap_cny"], "probe exceeds reservation cap")
    hashes = {}

    def track(path):
        path = Path(path).resolve(strict=True)
        _require(path.is_relative_to(directory), "probe audit escapes its output")
        hashes[path.relative_to(directory).as_posix()] = _sha(path)
        return path

    captured = audit_http(
        calls[0],
        directory,
        plan,
        track,
        set(),
        RUN_ID + "/",
        expected_response_format=frozen["response_format"],
    )
    _require(
        {path.name for path in (directory / "api_audit").glob("*.json")}
        == {Path(captured[0]["audit_path"]).name},
        "unattributed probe HTTP exists",
    )
    audit_journal(directory, ledger, [captured], track)
    _require(captured[0]["trace_id"] == RUN_ID + "/locator", "probe trace differs")
    _, prepared, messages = probe_input()
    _require(captured[1]["messages"] == messages, "actual probe input differs")
    valid = False
    if captured[2] is not None:
        _require(record["messages"] == messages, "saved probe input differs")
        _require(captured[2] == record["raw_content"], "probe raw content differs")
        try:
            location = observer.resolve_location(captured[2], prepared)
        except observer.TwoStageContractError as error:
            _require(
                record["status"] == "model_output_error" and record["failure_code"] == error.code,
                "failed structure replay differs",
            )
        else:
            _require(
                record["status"] == "completed" and location.to_dict() == record["location_report"],
                "probe location report differs",
            )
            valid = True
    else:
        _require(
            record["status"] == "stopped" and ledger.get("block_reason"),
            "unknown or rejected probe was not stopped",
        )
    return {
        "status": "format_compatibility_verified" if valid else "probe_failed_audited",
        "row_id": frozen["row_id"],
        "format_valid": valid,
        "totals": amounts,
        "audited_input_sha256": hashes,
        "api_calls": 0,
        "files_written": 0,
        "gold_loaded": False,
        "memory_updated": False,
        "natural_prepare_allowed": False,
        "notice": "One exposed locator input, not judge/semantic accuracy or a passed A1 gate.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-project", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--expected-freeze-sha256")
    parser.add_argument("--expected-terminal-sha256")
    args = parser.parse_args(argv)
    if args.freeze and (args.audit or args.allow_network or args.expected_freeze_sha256):
        parser.error("freeze must be standalone")
    if args.audit and (args.allow_network or not args.expected_terminal_sha256):
        parser.error("offline audit requires terminal pin and no network")
    if args.freeze:
        result = freeze(args.artifact_project)
    elif args.audit:
        result = audit(
            args.artifact_project, args.expected_freeze_sha256, args.expected_terminal_sha256
        )
    else:
        result = run(
            args.artifact_project, args.expected_freeze_sha256, allow_network=args.allow_network
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
