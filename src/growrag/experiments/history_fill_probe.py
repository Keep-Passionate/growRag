"""五个合成场景的真实FILL分支探针，不执行选择器、不读Hotpot答案。

探针由实验者显式指定卡片，只检验原动作的参数传输/类型/编译兼容。
不报告准确率、检索收益或学习模板适用性。任何失败保留原响应并停止。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from growrag.history_library import FrozenHistoryLibrary, card_view, resolve_card
from growrag.macro_operators import GoalContract, GroundedBinding, RuntimeState

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import PriceLimits
from .fresh_dev_manifest import _sha
from .history_budget import CANDIDATE_PREFIX, reviewed_history
from .history_calibration import LIBRARY_FINGERPRINT, LIBRARY_SHA256
from .history_context import _bounded_messages, plan_selected_template, prepare_history_context
from .history_runtime import FILL_PROMPT, FILL_VERSION, HistoryContractClient
from .history_runtime_v2 import FillWireV2Client
from .operator_model import strict_object
from .pre_pilot import write_json
from .protocol import Evidence, RuntimeQuestion
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_operator_study import OperatorLog, serial_lock
from .run_pilot import PILOT_MODEL, _git_state
from .run_pre_opportunity import DurableBudgetClient

PROTOCOL = "growrag-history-candidate-fill-probe-v1"
RUN_ID = CANDIDATE_PREFIX + "fillcheck_v1"
KEY_VARIABLE = "GROWRAG_HISTORY_FILL_PROBE_KEY"
LIBRARY_PATH = "runs/history_foundation_20261001_v1/combined.json"


def cases():
    """虚构名字、无答案的固定输入；expected_gap只是接口测试期望。"""
    return (
        {
            "case_id": "population_integer",
            "card_id": "OP_d031917d14319bca",
            "question": "What was the population of Linden in 2012?",
            "evidence": "Linden is the city under discussion; the requested year is 2012.",
            "expected_gap": {"city": "Linden", "year": 2012},
        },
        {
            "case_id": "person_birth",
            "card_id": "OP_06a401fc825c2327",
            "question": "In which year was Avery Lin born?",
            "evidence": (
                "Avery Lin is a fictional singer. This passage does not state a birth year."
            ),
            "expected_gap": {"person_name": "Avery Lin"},
        },
        {
            "case_id": "series_creator",
            "card_id": "OP_46346786999106e7",
            "question": "Who created the fictional television series River Bells?",
            "evidence": "River Bells is a fictional television series; its creator is not given.",
            "expected_gap": {"series_name": "River Bells"},
        },
        {
            "case_id": "entity_property",
            "card_id": "OP_fd457a0cd835fa75",
            "question": "What is the height of the fictional Old Gate?",
            "evidence": "Old Gate is a fictional tower. Its height is not given here.",
            "expected_gap": {"entity": "Old Gate", "property": "height"},
        },
        {
            "case_id": "actor_film",
            "card_id": "OP_e8e689c3513a8365",
            "question": (
                "In which year was the fictional film River Bells starring Mira Chen released?"
            ),
            "evidence": (
                "Mira Chen stars in the fictional film River Bells; no release year is stated."
            ),
            "expected_gap": {"actor_name": "Mira Chen", "film_title": "River Bells"},
        },
    )


def probe_request(case, library):
    question = RuntimeQuestion(case["case_id"], case["question"], "synthetic_fill_probe")
    evidence = (
        Evidence("probe_current_passage", "Synthetic current passage", 0, case["evidence"]),
    )
    context = prepare_history_context(
        library,
        question,
        offered_ids=(case["card_id"],),
        representation="examples",
        evidence=evidence,
        remaining_retrievals=1,
    )
    card = resolve_card(library, context.offered_ids, case["card_id"])
    payload = {
        "original_question": question.text,
        "evidence": context.payload["evidence"],
        "previous_queries": [],
        "remaining_retrievals": 1,
        "selected_card": card_view(card, "examples"),
    }
    return question, evidence, context, _bounded_messages(FILL_PROMPT, payload)


def compile_probe(case, library, context, evidence, response_text):
    """复用正式消费者，不删除非法字段、不字符串转整数、不补猜实体。"""
    value = strict_object(response_text)
    gap = {}
    for entry in value["gap_entries"]:
        if entry["name"] in gap:
            raise ValueError("duplicate probe gap")
        gap[entry["name"]] = entry["value"]
    bindings = tuple(
        GroundedBinding(row["name"], row["value"], tuple(row["evidence_ids"]))
        for row in value["bindings"]
    )
    plan = plan_selected_template(
        library,
        context,
        case["card_id"],
        goal=GoalContract(case["question"], value["intent"], tuple(value["constraints"])),
        gap=gap,
        state=RuntimeState(evidence, bindings, 1),
    )
    if gap != case["expected_gap"] or bindings or len(plan.requests) != 1:
        raise ValueError("synthetic probe parameters differ from preregistered inputs")
    return asdict(plan)


def run(allow_network=False):
    project = Path.cwd().resolve(strict=True)
    runs = project / "runs"
    path = project / LIBRARY_PATH
    if _sha(path) != LIBRARY_SHA256:
        raise ValueError("frozen library bytes changed")
    library = FrozenHistoryLibrary.from_json(path.read_text(encoding="utf-8"))
    if library.fingerprint != LIBRARY_FINGERPRINT:
        raise ValueError("frozen library identity changed")
    snapshot = source_snapshot(project)
    plan = {
        "protocol": PROTOCOL,
        "run_id": RUN_ID,
        "phase": "synthetic_transport_probe",
        "model": PILOT_MODEL,
        "is_hotpot_evaluation": False,
        "gold_loaded": False,
        "memory_updates": False,
        "selector_executed": False,
        "cases": cases(),
        "cases_sha256": fingerprint(cases()),
        "library_sha256": LIBRARY_SHA256,
        "source_sha256": snapshot["sha256"],
        "git": _git_state(),
        "project_cap_cny": 200.0,
        "series_cap_cny": 5.0,
        "price_checked_date": "2026-10-02",
        "price_source": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
    }
    if not allow_network:
        print(json.dumps({"dry_run": True, "plan": plan}, ensure_ascii=False, indent=2))
        return 0
    with serial_lock(runs):
        output = runs / RUN_ID
        if output.exists() or (runs / f"{RUN_ID}.claim.json").exists():
            raise FileExistsError("probe already claimed; no automatic replay")
        history = reviewed_history(runs)
        series_reserved = sum(
            c["reserved_cny"]
            for c in history["calls"]
            if c["trace_id"].startswith(CANDIDATE_PREFIX)
        )
        subcap = min(5.0 - series_reserved, 200.0 - history["prior_reserved_cny"])
        if subcap <= 0:
            raise ValueError("cumulative budget exhausted")
        # 全项目审计可能耗时数分钟；开始付费前再确认并行编辑没有改变方法。
        if source_snapshot(project)["sha256"] != snapshot["sha256"]:
            raise ValueError("source changed before probe execution")
        if _sha(path) != LIBRARY_SHA256:
            raise ValueError("library changed before probe execution")
        settings = read_local_bailian_settings(project / "qwenAPI.md")
        host = urlsplit(settings.base_url).hostname or ""
        if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
            raise ValueError("probe requires the declared Beijing price region")
        plan.update(prior_reserved_cny=history["prior_reserved_cny"], subcap_cny=subcap)
        write_json(runs / f"{RUN_ID}.claim.json", {**plan, "plan_sha256": fingerprint(plan)})
        output.mkdir(exist_ok=False)
        write_json(output / "launch_plan.json", plan)
        write_json(output / "source_snapshot.json", snapshot)
        write_json(output / "prior_budget.json", history)
        write_json(output / "process.json", {"pid": os.getpid()})
        log = OperatorLog(output)
        prior_key = os.environ.get(KEY_VARIABLE)
        os.environ[KEY_VARIABLE] = settings.api_key
        client, failure, completed, hashes = None, None, 0, {}
        try:
            config = ChatConfig(
                settings.base_url,
                PILOT_MODEL,
                KEY_VARIABLE,
                max_calls=5,
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
                PriceLimits(budget_cny=subcap, max_prompt_bytes=30000, max_elapsed_seconds=600),
                output / "request_journal",
            )
            wire = FillWireV2Client(HistoryContractClient(client, on_record=log))
            for case in cases():
                log.question_id, log.arm = case["case_id"], "fill_probe"
                record = {"status": "started", "case": case, "selection": "experimenter_fixed"}
                target = output / f"{case['case_id']}.json"
                try:
                    if _sha(path) != LIBRARY_SHA256:
                        raise ValueError("library changed during probe")
                    _, evidence, context, messages = probe_request(case, library)
                    log({"kind": "probe_case_start", "card_id": case["card_id"]})
                    response = wire.complete(
                        messages,
                        trace_id=f"{RUN_ID}/{case['case_id']}/fill",
                        prompt_version=FILL_VERSION,
                    )
                    record["raw_content"] = response.content
                    record["compiled"] = compile_probe(
                        case,
                        library,
                        context,
                        evidence,
                        response.content,
                    )
                    record["status"] = "completed"
                    completed += 1
                    log({"kind": "probe_case_completed"})
                except BaseException as error:
                    record.update(status="failed", error_type=type(error).__name__)
                    raise
                finally:
                    write_json(target, record)
                    hashes[target.name] = _sha(target)
            if source_snapshot(project)["sha256"] != snapshot["sha256"]:
                raise ValueError("source changed during probe")
        except BaseException as error:
            failure = type(error).__name__
            if client is not None and not client.block_reason:
                client.block_reason = "probe_execution_failure"
            log({"kind": "probe_stopped", "error_type": failure})
        finally:
            if prior_key is None:
                os.environ.pop(KEY_VARIABLE, None)
            else:
                os.environ[KEY_VARIABLE] = prior_key
            budget = (
                client.report()
                if client
                else {
                    "api_requests": 0,
                    "calls": [],
                    "reserved_cny": 0,
                    "estimated_actual_cny": 0,
                }
            )
            summary = {
                "protocol": PROTOCOL,
                "run_id": RUN_ID,
                "status": "failed" if failure else "completed",
                "failure_type": failure,
                "planned_cases": 5,
                "completed_cases": completed,
                "source_sha256": snapshot["sha256"],
                "library_sha256": LIBRARY_SHA256,
                "prediction_sha256": hashes,
                "gold_loaded": False,
                "memory_updated": False,
                "selector_executed": False,
                "api_requests": budget["api_requests"],
                "estimated_actual_cny": budget["estimated_actual_cny"],
                "reserved_cny": budget["reserved_cny"],
            }
            write_json(output / "final_budget.json", budget)
            write_json(output / "SUMMARY.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 1 if failure else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-network", action="store_true")
    return run(parser.parse_args(argv).allow_network)


if __name__ == "__main__":
    raise SystemExit(main())
