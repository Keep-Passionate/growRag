"""25题开发评分：先核验两批100个正常封存预测，再打开train标签。

默认只预检，不读取gold、不写文件；--score才生成独立反馈目录。失败、漏题、
重跑或账本异常均停止，不能把失败改成零分，也不能据结果重跑本批。
这不是官方测试集成绩；同Reader输入的答案波动不归因给历史卡。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from growrag.history_library import FrozenHistoryLibrary, card_view

from .fresh_dev_manifest import _sha
from .history_budget import reviewed_history
from .history_calibration import (
    ARMS,
    LIBRARY_FINGERPRINT,
    LIBRARY_SHA256,
    MANIFEST_SHA256,
    PREFIX,
    profile_for,
)
from .history_calibration import (
    CONFIGURATION as CONFIGURATION,
)
from .history_calibration import (
    PROTOCOL as PROTOCOL,
)
from .history_context import REWRITE_PROMPT_VERSION, SELECT_PROMPT_VERSION
from .history_runtime import FILL_VERSION, schema_for
from .history_runtime_v2 import schema_for_v2
from .operator_model import READER_PROMPT, _messages, strict_object, visible_evidence
from .operator_schemas import READER_VERSION, validate_wire_shape
from .protocol import Evidence
from .representation_runner import fingerprint
from .run_dualrag_pilot import source_snapshot
from .run_operator_study import artifact
from .score_operator_sources import score_source_report

MANIFEST = "data/hotpotqa/operator_scale_action_v3/manifest.json"
LIBRARY = "runs/history_foundation_20261001_v1/combined.json"
OUTPUT = "runs/history_base25_feedback_v1"
BATCHES = ((40, 5), (45, 20))
METRICS = ("answer_em", "answer_f1", "raw_support_recall", "visible_support_recall")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_bytes())


def _inside(root, path):
    path = path.resolve(strict=True)
    _require(path.is_relative_to(root) and path.is_file(), "audit input escapes project")
    return path


def _money(value):
    _require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        "unknown or invalid accounting amount",
    )
    return value


def _totals(calls):
    return {
        "api_requests": len(calls),
        **{
            key: sum(_money(call[key]) for call in calls)
            for key in ("input_tokens", "output_tokens", "estimated_actual_cny", "reserved_cny")
        },
    }


def _audit_arm(report, directory, plan, events, library, inputs, seen, *, version="v1"):
    """逐臂对齐原HTTP、原回答与选择事件；不是重新运行模型。"""
    configuration = profile_for(version)["configuration"]
    fill_version = configuration.get("fill_prompt_version", FILL_VERSION)
    wire_schema = schema_for_v2 if version == "v2" else schema_for
    qid, arm = report["question_id"], report["arm"]
    prefix = f"{plan['run_id']}/{qid}/{arm}/"
    calls = report.get("calls")
    _require(type(calls) is list and calls, "completed arm has no API records")
    selected, reader_sha = [], None
    cards = {r.card.card_id: r for r in library.records if r.status == "published"}
    for number, call in enumerate(calls):
        trace = call.get("trace_id", "")
        _require(trace.startswith(prefix) and trace not in seen, "duplicate/wrong API trace")
        seen.add(trace)
        name = hashlib.sha256(trace.encode()).hexdigest() + ".json"
        path = _inside(directory, directory / "api_audit" / name)
        _require(Path(call.get("audit_path", "")).name == name, "ledger audit identity mismatch")
        data = _read(path)
        inputs[str(path)] = _sha(path)
        _require(
            all(
                data.get(key) == value
                for key, value in {
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
                }.items()
            ),
            "HTTP request is not a normal unretried terminal response",
        )
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
            "raw response usage differs from ledger",
        )
        choices = response.get("choices", [])
        _require(
            len(choices) == 1 and choices[0].get("finish_reason") == "stop",
            "completion is truncated or not unique",
        )
        value = strict_object(choices[0]["message"]["content"])
        validate_wire_shape(value, wire_schema(call["prompt_version"]))
        cost = (call["input_tokens"] * 0.2 + call["output_tokens"] * 0.8) / 1_000_000
        _require(
            math.isclose(_money(call.get("estimated_actual_cny")), cost, abs_tol=1e-10)
            and cost <= _money(call.get("reserved_cny")),
            "cost estimate mismatch",
        )
        if trace == prefix + "reader":
            _require(
                reader_sha is None and call["prompt_version"] == READER_VERSION,
                "duplicate/wrong Reader",
            )
            evidence = tuple(Evidence(**item) for item in report["episode"]["evidence"])
            visible = visible_evidence(evidence)
            payload = {
                "original_question": report["question"]["text"],
                "evidence": visible,
                "evidence_window_omitted_count": len(evidence) - len(visible),
            }
            _require(
                request["messages"] == _messages(READER_PROMPT, payload),
                "Reader request is not original question plus current evidence",
            )
            _require(
                value
                == {key: report["reader"][key] for key in ("answer", "supported", "evidence_ids")},
                "saved Reader answer differs from HTTP response",
            )
            _require(
                report["reader"].get("evidence_windows")
                == [{"evidence_id": e["evidence_id"], **e["window"]} for e in visible]
                and report["reader"].get("visible_evidence_ids")
                == [e["evidence_id"] for e in visible],
                "saved Reader windows differ from actual visible input",
            )
            reader_sha = fingerprint(request["messages"])
        elif call["prompt_version"] == SELECT_PROMPT_VERSION:
            _require(arm in {"history", "static_rules"}, "selection in non-history arm")
            payload = json.loads(request["messages"][1]["content"])
            offered = payload["candidate_cards"]
            _require(
                0 < len(offered) <= 3 and len({v["card_id"] for v in offered}) == len(offered),
                "invalid offered identities",
            )
            for view in offered:
                record = cards.get(view["card_id"])
                _require(
                    record is not None
                    and view == card_view(record.card, "examples")
                    and (arm != "static_rules" or record.source_kind == "reference"),
                    "offered card differs from frozen library",
                )
            identity = value["selected_card_id"]
            _require(
                identity is None or identity in {v["card_id"] for v in offered},
                "selected card was not offered",
            )
            if identity is not None:
                template = cards[identity].card.operator_spec is not None
                suffix, prompt_version = (
                    ("fill", fill_version) if template else ("rewrite", REWRITE_PROMPT_VERSION)
                )
                _require(
                    number + 1 < len(calls)
                    and calls[number + 1]["trace_id"] == trace.removesuffix("select") + suffix
                    and calls[number + 1]["prompt_version"] == prompt_version,
                    "selected card has no matching fill/rewrite request",
                )
            selected.append(value)
    _require(
        reader_sha is not None and calls[-1]["trace_id"] == prefix + "reader",
        "missing/final Reader request mismatch",
    )
    local = [e for e in events if e.get("question_id") == qid and e.get("arm") == arm]
    choices = [
        {k: e[k] for k in ("selected_card_id", "reason")}
        for e in local
        if e.get("kind") == "history_selection"
    ]
    executed = [e for e in local if e.get("kind") == "history_execution"]
    _require(
        choices == selected
        and [e["selected_card_id"] for e in executed]
        == [v["selected_card_id"] for v in selected if v["selected_card_id"] is not None],
        "selection/execution event differs from actual HTTP selection",
    )
    _require(
        all(
            e["selected_card_version"] == cards[e["selected_card_id"]].card.version
            and e["action_kind"] == cards[e["selected_card_id"]].card.action_kind
            for e in executed
        ),
        "executed card identity/version differs from frozen library",
    )
    return {
        **_totals(calls),
        "reader_input_sha256": reader_sha,
        "selections": selected,
        "executed_cards": [
            {
                "card_id": e["selected_card_id"],
                "version": e["selected_card_version"],
                "action_kind": e["action_kind"],
                "source_kind": cards[e["selected_card_id"]].source_kind,
                "search_executed": any(
                    search["step"] == number for search in report["episode"]["searches"]
                ),
            }
            for number, e in enumerate(executed, start=1)
        ],
        "retrieval_calls": len(report["episode"]["searches"]),
    }


def _scope(manifest, version):
    """题段来自runner显式登记，不允许通过评分CLI自选题/拼接旧失败批次。"""
    profile = profile_for(version)
    ids = [
        qid
        for start, count in profile["batches"]
        for qid in manifest["roles"]["calibration"][start : start + count]
    ]
    _require(len(ids) == len(set(ids)) == 25, "registered calibration scope must contain 25 IDs")
    return ids


def collect(project, version="v1"):
    """只读取预测/无标签题/账本。全25题正常才返回，失败时绝不打开gold。"""
    profile = profile_for(version)
    batches, configuration, protocol = (
        profile[k] for k in ("batches", "configuration", "protocol")
    )
    project = Path(project).resolve(strict=True)
    inputs = {}

    def tracked(relative):
        path = _inside(project, project / relative)
        inputs[str(path)] = _sha(path)
        return path

    manifest_path, library_path = tracked(MANIFEST), tracked(LIBRARY)
    _require(
        _sha(manifest_path) == MANIFEST_SHA256 and _sha(library_path) == LIBRARY_SHA256,
        "manifest/library pinned bytes changed",
    )
    manifest = _read(manifest_path)
    roles = manifest["roles"]
    ids = _scope(manifest, version)
    _require(
        len(roles["calibration"]) == 100
        and len(ids) == len(set(ids)) == 25
        and manifest["official_splits"]["calibration"] == "train"
        and not set(ids) & set(roles["source"] + roles["evaluation"]),
        "calibration identity/split overlap",
    )
    runtime = artifact(manifest_path.parent, manifest, "calibration_runtime_questions.jsonl")
    inputs[str(runtime)] = _sha(runtime)
    questions = [json.loads(line) for line in runtime.read_text(encoding="utf-8").splitlines()]
    _require(
        [q["question_id"] for q in questions] == roles["calibration"]
        and all(set(q) == {"question_id", "text", "dataset"} for q in questions),
        "runtime questions changed or contain non-runtime fields",
    )
    questions = {q["question_id"]: q for q in questions}
    corpus = artifact(manifest_path.parent, manifest, "corpus.jsonl")
    inputs[str(corpus)] = _sha(corpus)
    library = FrozenHistoryLibrary.from_json(library_path.read_text(encoding="utf-8"))
    _require(
        library.fingerprint == LIBRARY_FINGERPRINT
        and set(library.allowed_source_ids) == set(roles["source"]),
        "library identity changed",
    )
    reference = FrozenHistoryLibrary(
        library.protocol_id,
        library.allowed_source_ids,
        tuple(r for r in library.records if r.source_kind == "reference"),
    )
    runs = project / "runs"
    expected_runs = {
        f"{PREFIX}{version}_{start:04d}_{start + count:04d}" for start, count in batches
    }
    for claim_path in runs.glob("*.claim.json"):
        claim = _read(_inside(project, claim_path))
        if set(ids) & set(claim.get("question_ids", [])):
            _require(
                claim_path.name.removesuffix(".claim.json") in expected_runs,
                "calibration IDs were claimed again under another run",
            )
    for launch in runs.glob(f"{PREFIX}*/launch_plan.json"):
        if set(ids) & set(_read(_inside(project, launch)).get("question_ids", [])):
            _require(launch.parent.name in expected_runs, "duplicate question launch")
    snapshot_sha = source_snapshot(project)["sha256"]
    records, seen, all_calls = [], set(), []
    for batch_number, (start, count) in enumerate(batches):
        run_id = f"{PREFIX}{version}_{start:04d}_{start + count:04d}"
        directory = runs / run_id
        plan = _read(tracked(f"runs/{run_id}/launch_plan.json"))
        claim = _read(tracked(f"runs/{run_id}.claim.json"))
        summary = _read(tracked(f"runs/{run_id}/SUMMARY.json"))
        snapshot = _read(tracked(f"runs/{run_id}/source_snapshot.json"))
        ledger = _read(tracked(f"runs/{run_id}/final_budget.json"))
        events = [
            json.loads(line)
            for line in tracked(f"runs/{run_id}/events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        expected_ids = roles["calibration"][start : start + count]
        expected = {
            **configuration,
            "protocol": protocol,
            "phase": "calibration",
            "run_id": run_id,
            "start": start,
            "count": count,
            "arms": list(ARMS),
            "question_ids": expected_ids,
            "manifest_sha256": MANIFEST_SHA256,
            "library_sha256": LIBRARY_SHA256,
            "library_fingerprint": LIBRARY_FINGERPRINT,
            "reference_fingerprint": reference.fingerprint,
            "official_split": "train",
            "source_sha256": snapshot_sha,
            "gold_loaded": False,
            "memory_updates": False,
        }
        if batch_number:
            first_start, first_count = batches[0]
            first_run = f"{PREFIX}{version}_{first_start:04d}_{first_start + first_count:04d}"
            expected["first_batch_summary_sha256"] = _sha(runs / first_run / "SUMMARY.json")
        _require(
            all(plan.get(k) == v for k, v in expected.items())
            and claim == {**plan, "plan_sha256": fingerprint(plan)},
            "launch/claim drift",
        )
        _require(
            snapshot.get("sha256") == snapshot_sha == fingerprint(snapshot.get("files")),
            "saved/current source snapshots differ",
        )
        _require(
            all(
                summary.get(k) == v
                for k, v in {
                    "protocol": protocol,
                    "run_id": run_id,
                    "status": "completed",
                    "failure_type": None,
                    "planned_questions": count,
                    "completed_questions": count,
                    "source_sha256": snapshot_sha,
                    "gold_loaded": False,
                    "memory_updated": False,
                    "new_operator_induction": False,
                }.items()
            ),
            "all 25 predictions must complete normally before gold access",
        )
        terminal = [
            {
                "question_id": qid,
                "arms": {arm: {"status": "completed", "path": f"{qid}_{arm}.json"} for arm in ARMS},
            }
            for qid in expected_ids
        ]
        names = {f"{qid}_{arm}.json" for qid in expected_ids for arm in ARMS}
        _require(
            summary.get("terminal") == terminal
            and set(summary.get("prediction_sha256", {})) == names,
            "terminal/seal coverage mismatch",
        )
        _require(
            [e["question_id"] for e in events if e.get("kind") == "question_completed"]
            == expected_ids
            and not any(e.get("kind") == "batch_stopped" for e in events),
            "question completion sequence mismatch",
        )
        batch_calls = []
        for qid in expected_ids:
            row = {"question_id": qid, "question": questions[qid]["text"], "arms": {}, "usage": {}}
            for arm in ARMS:
                name = f"{qid}_{arm}.json"
                path = tracked(f"runs/{run_id}/{name}")
                report = _read(path)
                _require(
                    _sha(path) == summary["prediction_sha256"][name]
                    and report.get("status") == "completed"
                    and report.get("question_id") == qid
                    and report.get("arm") == arm
                    and report.get("question") == questions[qid]
                    and report.get("gold_loaded") is False
                    and report.get("memory_updated") is False
                    and report.get("feedback") is None
                    and report["episode"]["question_id"] == qid,
                    "prediction hash/status/identity/feedback mismatch",
                )
                row["usage"][arm] = _audit_arm(
                    report, directory, plan, events, library, inputs, seen, version=version
                )
                row["arms"][arm] = report
                batch_calls.extend(report["calls"])
            records.append(row)
        _require(
            ledger.get("calls") == batch_calls and ledger.get("block_reason") is None,
            "ledger has omitted/duplicate/failed requests",
        )
        totals = _totals(batch_calls)
        for key, value in totals.items():
            _require(
                math.isclose(_money(ledger.get(key)), value, abs_tol=1e-10), "ledger total mismatch"
            )
            if key in summary:
                _require(
                    math.isclose(_money(summary[key]), value, abs_tol=1e-10),
                    "summary cost mismatch",
                )
        all_calls.extend(batch_calls)
    reconciled = reviewed_history(runs)  # 复用项目审计：检测漏登的HTTP请求与跨run重复trace。
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
        "model": configuration["model"],
        "protocol": protocol,
        "version": version,
    }


def _load_calibration_gold(project, manifest, ids, *, version="v1"):
    """仅在collect成功后调用；标签投影固定25个train ID，不读source/dev答案。"""
    import pyarrow.dataset as ds

    _require(
        ids == _scope(manifest, version) and len(ids) == len(set(ids)) == 25,
        "gold projection must be exactly predeclared calibration 25",
    )
    recorded = {x["path"].replace("\\", "/"): x["sha256"] for x in manifest["input_files"]}
    relative = "data/hotpotqa/official_train_v1_1/mirror_provenance.json"
    provenance_path = _inside(project, project / relative)
    _require(recorded.get(relative) == _sha(provenance_path), "train provenance hash mismatch")
    provenance = _read(provenance_path)
    _require(provenance.get("official_split") == "train", "gold is not official train")
    paths, inputs = [], {str(provenance_path): _sha(provenance_path)}
    for shard in provenance["shards"]:
        path = _inside(project, provenance_path.parent / shard["file"])
        digest = _sha(path)
        _require(
            digest == shard["sha256"] == recorded.get(path.relative_to(project).as_posix()),
            "train shard differs from provenance/manifest",
        )
        paths.append(str(path))
        inputs[str(path)] = digest
    _require(paths and len(paths) == len(set(paths)), "missing/duplicate train shards")
    rows = (
        ds.dataset(paths, format="parquet")
        .to_table(
            columns=["id", "answer", "supporting_facts", "context"],
            filter=ds.field("id").isin(ids),
        )
        .to_pylist()
    )
    _require(
        len(rows) == 25 and {row["id"] for row in rows} == set(ids), "gold ID coverage mismatch"
    )
    return {row["id"]: row for row in rows}, inputs


def summarize(records, gold):
    """纯函数：配对差值不是因果证明；未知证据分数保持None并报告有效分母。"""
    rows = []
    for item in records:
        qid = item["question_id"]
        scores = {arm: score_source_report(item["arms"][arm], gold[qid]) for arm in ARMS}
        comparisons = {}
        for other in ARMS[:-1]:
            left, right = scores["history"], scores[other]
            same = (
                item["usage"]["history"]["reader_input_sha256"]
                == item["usage"][other]["reader_input_sha256"]
            )
            ems = (right["answer_em"], left["answer_em"])
            comparisons[other] = {
                "reader_input_identical": same,
                "answer_changed": item["arms"]["history"]["reader"]["answer"]
                != item["arms"][other]["reader"]["answer"],
                "repaired": ems == (0.0, 1.0),
                "harmed": ems == (1.0, 0.0),
                "delta": {
                    m: None if left[m] is None or right[m] is None else left[m] - right[m]
                    for m in METRICS
                },
            }
        rows.append(
            {
                "question_id": qid,
                "question": item["question"],
                "scores": scores,
                "answers": {arm: item["arms"][arm]["reader"]["answer"] for arm in ARMS},
                "reference_answer": gold[qid]["answer"],
                "usage": item["usage"],
                "history_vs": comparisons,
            }
        )
    by_arm = {}
    for arm in ARMS:
        stats = {}
        for metric in METRICS:
            values = [
                r["scores"][arm][metric] for r in rows if r["scores"][arm][metric] is not None
            ]
            stats[metric] = {"mean": mean(values) if values else None, "valid_n": len(values)}
        for key in (
            "api_requests",
            "input_tokens",
            "output_tokens",
            "estimated_actual_cny",
            "retrieval_calls",
        ):
            stats[key] = sum(r["usage"][arm][key] for r in rows)
        stats["card_preparation_questions"] = sum(
            bool(r["usage"][arm]["executed_cards"]) for r in rows
        )
        stats["card_execution_questions"] = sum(
            any(c.get("search_executed", False) for c in r["usage"][arm]["executed_cards"])
            for r in rows
        )
        stats["learned_card_questions"] = sum(
            any(
                c["source_kind"] == "learned" and c.get("search_executed", False)
                for c in r["usage"][arm]["executed_cards"]
            )
            for r in rows
        )
        by_arm[arm] = stats
    paired = {}
    for other in ARMS[:-1]:
        values = [r["history_vs"][other] for r in rows]
        paired[other] = {
            "repaired": sum(v["repaired"] for v in values),
            "harmed": sum(v["harmed"] for v in values),
            "identical_reader_input": sum(v["reader_input_identical"] for v in values),
            "same_input_answer_changed": sum(
                v["reader_input_identical"] and v["answer_changed"] for v in values
            ),
            "same_input_repaired": sum(
                v["reader_input_identical"] and v["repaired"] for v in values
            ),
            "same_input_harmed": sum(v["reader_input_identical"] and v["harmed"] for v in values),
            "delta": {
                m: {"mean": mean(known) if known else None, "valid_n": len(known)}
                for m in METRICS
                for known in [[v["delta"][m] for v in values if v["delta"][m] is not None]]
            },
        }
    return rows, {
        "questions": len(rows),
        "completed_arms": len(rows) * len(ARMS),
        "arms": by_arm,
        "history_vs": paired,
    }


def _markdown(rows):
    lines = [
        "# 历史基底25题开发反馈",
        "",
        "仅官方train开发诊断，不是独立测试成绩。EM为完全匹配，F1为答案词重合分数。",
        "同Reader输入的答案变化单列；卡被选择不等于补充了证据，也不等于因果收益。费用为申报价估算，非账单。",
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
            "| 路径 | EM | F1 | 可见证据覆盖 | 请求/检索 | 元 | 卡片 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for arm in ARMS:
            score, usage = row["scores"][arm], row["usage"][arm]
            cards = (
                ", ".join(
                    x["card_id"]
                    + ("（已检索）" if x.get("search_executed", False) else "（仅准备）")
                    for x in usage["executed_cards"]
                )
                or "无"
            )
            lines.append(
                f"| {arm} | {score['answer_em']} | {score['answer_f1']} | "
                f"{score['visible_support_recall']} | "
                f"{usage['api_requests']}/{usage['retrieval_calls']} | "
                f"{usage['estimated_actual_cny']:.6f} | {cards} |"
            )
        lines += [
            "",
            "```json",
            json.dumps(
                {
                    "answers": row["answers"],
                    "history_vs": row["history_vs"],
                    "selection_and_usage": row["usage"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    return "\n".join(lines)


def run(project, *, score=False, version="v1"):
    profile = profile_for(version)
    project = Path(project).resolve(strict=True)
    output_name = OUTPUT if version == "v1" else "runs/history_base25_feedback_v2"
    output = (project / output_name).resolve()
    _require(output.is_relative_to(project / "runs"), "feedback output escapes runs")
    if score and output.exists():
        raise FileExistsError("feedback output exists; never overwrite or choose another subset")
    checked = collect(project, version=version)
    public = {
        "protocol": checked["protocol"],
        "version": version,
        "questions": 25,
        "arms": list(ARMS),
        "status": "preflight_passed",
        "gold_loaded": False,
        "api_calls": 0,
        "source_sha256": checked["source_sha256"],
        "totals": checked["totals"],
    }
    if not score:
        return public
    gold, gold_inputs = _load_calibration_gold(
        project, checked["manifest"], checked["ids"], version=version
    )
    rows, summary = summarize(checked["records"], gold)
    inputs = {**checked["inputs"], **gold_inputs}
    _require(
        all(_sha(Path(path)) == digest for path, digest in inputs.items()),
        "inputs changed during offline scoring; no output published",
    )
    summary.update(
        protocol=checked["protocol"],
        version=version,
        split=f"official_train_calibration_{profile['batches'][0][0]}_"
        f"{sum(profile['batches'][-1])}",
        model=checked["model"],
        totals=checked["totals"],
        api_calls=0,
        memory_updated=False,
        prediction_modified=False,
        labels_opened_after_full_seal=True,
        source_sha256=checked["source_sha256"],
        library_sha256=LIBRARY_SHA256,
        policy="Development diagnosis only; no rerun, gold-based method update, or causal claim.",
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
            },
            handle,
            indent=2,
        )
    return {**public, "status": "scored", "gold_loaded": True, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    parser.add_argument(
        "--score",
        action="store_true",
        help="after complete preflight, read train25 gold and write feedback",
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(Path.cwd(), score=args.score, version=args.version), ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
