"""Synthetic, offline reporting fixtures; no labels, real API, or experiment replay."""

import json
from pathlib import Path

import pytest

from growrag.experiments.operator_run_report import PROTOCOL, _fingerprint, build_report, main


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def call(number=1):
    return {
        "api_requests": number,
        "input_tokens": 10 * number,
        "output_tokens": 2 * number,
        "estimated_actual_cny": 0.001 * number,
        "reserved_cny": 0.002 * number,
    }


def arm(qid, name, *, failed=False):
    if failed:
        return {
            "status": "failed",
            "question_id": qid,
            "arm": name,
            "episode": None,
            "reader": None,
            "error_type": "ValueError",
            "feedback": None,
            "memory_updated": False,
            "calls": [call()],
        }
    searches = [{"query": f"Question {qid}?", "step": 0}]
    proposals = []
    if name != "base":
        searches.append({"query": f"Query {qid} birthplace", "step": 1})
        proposals.append(
            {
                "spec": {"operator_id": "OP", "version": "1"},
                "origin": "fresh" if name == "fresh" else "static",
                "reason": "LONG_REASON_DO_NOT_RENDER",
            }
        )
    return {
        "status": "completed",
        "question_id": qid,
        "arm": name,
        "feedback": None,
        "calls": [call(1 if name == "base" else 2)],
        "reader": {"answer": "PREDICTED_ANSWER_DO_NOT_RENDER"},
        "episode": {"searches": searches, "proposals": proposals, "stop_reason": "decision_limit"},
    }


def freeze(root, predictions):
    dump(root / "predictions.json", predictions)
    dump(
        root / "predictions_frozen.json",
        {
            "sha256": _fingerprint(predictions),
            "phase": "calibration",
            "gold_loaded": False,
            "question_ids": [p["question_id"] for p in predictions],
        },
    )


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "runs" / "synthetic-calibration"
    root.mkdir(parents=True)
    ids = [f"q{i}" for i in range(8)]
    arms = ["base", "fresh", "static"]
    dump(
        root / "launch_plan.json",
        {
            "phase": "calibration",
            "protocol": PROTOCOL,
            "question_ids": ids,
            "arms": arms,
            "model": "fake-model",
        },
    )
    predictions = [
        {
            "question_id": qid,
            "arms": {a: arm(qid, a, failed=qid == "q1" and a == "static") for a in arms},
        }
        for qid in ids[:2]
    ]
    freeze(root, predictions)
    dump(
        root / "final_budget.json",
        {
            "api_requests": 9,
            "input_tokens": 90,
            "output_tokens": 18,
            "estimated_actual_cny": 0.009,
            "reserved_cny": 0.018,
        },
    )
    events = [
        {
            "question_id": "q1",
            "arm": "static",
            "kind": "operator_search",
            "search": {"query": "Question q1?", "step": 0},
        },
        {
            "question_id": "q1",
            "arm": "static",
            "kind": "planner_record",
            "model_output": {"reason": "LONG_REASON_DO_NOT_RENDER"},
        },
        {"question_id": "q1", "arm": "static", "kind": "failure", "error_type": "ValueError"},
    ]
    (root / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    return root


def test_report_distinguishes_planned_involved_complete_and_failed(run):
    report = build_report(run)
    assert "计划 8 题 × 3 臂 = 24 条路径" in report
    assert "实际涉及 2/8 题，全部臂完成 1 题" in report
    assert "完成 5，失败 1，中断/无完整报告 0，未启动 18" in report
    assert "批次 API 请求：9" in report
    assert "输入 tokens：90" in report
    assert "0.00900000 元" in report
    assert "0.01800000 元" in report
    assert "| q1 | 完成 | 完成 | 失败 |" in report
    assert "| q7 | 未启动 | 未启动 | 未启动 |" in report


def test_actual_queries_and_origins_appear_without_reason_or_answer(run):
    report = build_report(run)
    assert "原 query / 初检：Question q0?" in report
    assert "实际查询 2：Query q0 birthplace" in report
    assert "已执行算子：OP@1；来源：fresh" in report
    assert "已执行算子：OP@1；来源：static" in report
    assert "停止/异常标识：ValueError" in report
    assert "LONG_REASON_DO_NOT_RENDER" not in report
    assert "PREDICTED_ANSWER_DO_NOT_RENDER" not in report
    assert "未判断任何答案正确与否" in report
    assert not (run / "REPORT.md").exists()


def test_missing_budget_and_missing_arm_calls_are_unknown(run):
    (run / "final_budget.json").unlink()
    predictions = json.loads((run / "predictions.json").read_text())
    predictions[0]["arms"]["base"].pop("calls")
    freeze(run, predictions)
    report = build_report(run)
    assert "批次 API 请求：未知" in report
    assert "批次估算费用：未知；保守预留：未知" in report
    assert "API请求：未知（无账单记录）" in report


def test_empty_budget_does_not_implicitly_become_zero(run):
    dump(run / "final_budget.json", {})
    report = build_report(run)
    assert "批次 API 请求：未知" in report
    assert "批次估算费用：未知" in report


def test_partial_call_usage_is_labelled_not_underreported_as_complete(run):
    predictions = json.loads((run / "predictions.json").read_text())
    predictions[0]["arms"]["base"]["calls"] = [call(), {"api_requests": 1, "input_tokens": None}]
    freeze(run, predictions)
    report = build_report(run)
    assert "10（部分已知；另 1 项未知）" in report
    assert "0.00100000 元（部分已知；另 1 项未知）" in report


@pytest.mark.parametrize("change", ["fingerprint", "phase", "gold_loaded", "question_ids"])
def test_frozen_contract_mismatch_prevents_reporting(run, change):
    frozen = json.loads((run / "predictions_frozen.json").read_text())
    frozen[change if change != "fingerprint" else "sha256"] = "bad"
    dump(run / "predictions_frozen.json", frozen)
    with pytest.raises(ValueError, match="frozen"):
        build_report(run)


def test_mutated_predictions_cannot_pass_seal(run):
    predictions = json.loads((run / "predictions.json").read_text())
    predictions[0]["arms"]["base"]["status"] = "failed"
    dump(run / "predictions.json", predictions)
    with pytest.raises(ValueError, match="fingerprint"):
        build_report(run)


def test_evaluation_is_rejected_before_reading_predictions_or_other_files(run, monkeypatch):
    launch = json.loads((run / "launch_plan.json").read_text())
    launch["phase"] = "evaluation"
    dump(run / "launch_plan.json", launch)
    seen = []
    original = Path.read_text

    def checked(path, *args, **kwargs):
        seen.append(path.name)
        if path.name != "launch_plan.json":
            raise AssertionError("evaluation payload must stay unopened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", checked)
    with pytest.raises(ValueError, match="evaluation reporting is locked"):
        build_report(run)
    assert seen == ["launch_plan.json"]


def test_reporter_opens_only_whitelisted_run_artifacts(run, monkeypatch):
    original = Path.open
    names = []

    def checked(path, *args, **kwargs):
        names.append(path.name)
        assert path.name in {
            "launch_plan.json",
            "predictions.json",
            "predictions_frozen.json",
            "final_budget.json",
            "events.jsonl",
        }
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked)
    build_report(run)
    assert "predictions_frozen.json" in names


def test_cli_defaults_to_preview_and_explicit_new_write_never_overwrites(run, capsys):
    assert main([str(run)]) == 0
    assert "运行报告" in capsys.readouterr().out
    target = run / "REPORT.md"
    assert not target.exists()
    assert main([str(run), "--write-new"]) == 0
    first = target.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        main([str(run), "--write-new"])
    assert target.read_text(encoding="utf-8") == first


def test_failed_arm_queries_and_executed_operators_can_come_from_events(run):
    rows = [
        {
            "question_id": "q1",
            "arm": "static",
            "kind": "operator_proposal",
            "decision": 1,
            "proposal": {
                "spec": {"operator_id": "EVENT_OP", "version": "2"},
                "origin": "reuse",
                "reason": "PRIVATE_LONG_REASON",
            },
        },
        {
            "question_id": "q1",
            "arm": "static",
            "kind": "operator_search",
            "search": {"query": "Executed before failure", "step": 1},
        },
        {
            "question_id": "q1",
            "arm": "static",
            "kind": "operator_proposal",
            "decision": 2,
            "proposal": {
                "spec": {"operator_id": "NOT_EXECUTED", "version": "1"},
                "origin": "reuse",
            },
        },
    ]
    with (run / "events.jsonl").open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write("\n" + json.dumps(row))
    report = build_report(run)
    assert "Executed before failure" in report
    assert "EVENT_OP@2；来源：reuse" in report
    assert "NOT_EXECUTED" not in report
    assert "PRIVATE_LONG_REASON" not in report


def test_search_start_without_end_is_not_reported_as_executed_query(run):
    with (run / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n"
            + json.dumps(
                {
                    "question_id": "q2",
                    "arm": "base",
                    "kind": "operator_search_start",
                    "query": "NOT_COMPLETED_QUERY",
                }
            )
        )
    report = build_report(run)
    assert "| q2 | 中断/无完整报告 | 未启动 | 未启动 |" in report
    assert "NOT_COMPLETED_QUERY" not in report


def test_real_failure_shape_with_explicit_null_episode_falls_back_to_events(run):
    predictions = json.loads((run / "predictions.json").read_text())
    failed = predictions[1]["arms"]["static"]
    assert failed["episode"] is None and failed["reader"] is None
    assert set(failed) == {
        "status",
        "question_id",
        "arm",
        "episode",
        "reader",
        "feedback",
        "memory_updated",
        "error_type",
        "calls",
    }
    report = build_report(run)
    failed_section = report.split("### q1", 1)[1].split("#### static：失败", 1)[1]
    assert "原 query / 初检：Question q1?" in failed_section
    assert "停止/异常标识：ValueError" in failed_section
    assert "API请求：1" in failed_section


def test_scored_feedback_is_not_accepted_as_prediction_only(run):
    predictions = json.loads((run / "predictions.json").read_text())
    predictions[0]["arms"]["base"]["feedback"] = {"answer_em": 1}
    freeze(run, predictions)
    with pytest.raises(ValueError, match="prediction-only"):
        build_report(run)


def test_query_text_cannot_inject_markdown_or_html(run):
    predictions = json.loads((run / "predictions.json").read_text())
    predictions[0]["arms"]["base"]["episode"]["searches"][0]["query"] = "<img>\n# heading|`code`"
    freeze(run, predictions)
    report = build_report(run)
    assert "<img>" not in report
    assert "&lt;img&gt; # heading\\|&#96;code&#96;" in report


def test_source_phase_is_supported_without_loading_any_score_file(run):
    for filename in ("launch_plan.json", "predictions_frozen.json"):
        value = json.loads((run / filename).read_text())
        value["phase"] = "source"
        dump(run / filename, value)
    assert "阶段：source" in build_report(run)


def test_empty_frozen_batch_is_not_misreported_as_completed(run):
    freeze(run, [])
    (run / "events.jsonl").write_text("", encoding="utf-8")
    dump(run / "final_budget.json", {})
    report = build_report(run)
    assert "实际涉及 0/8 题，全部臂完成 0 题" in report
    assert "未启动 24" in report
    assert "批次 API 请求：未知" in report


@pytest.mark.parametrize("filename", ["launch_plan.json", "predictions_frozen.json"])
def test_non_object_manifest_is_rejected(run, filename):
    dump(run / filename, [])
    with pytest.raises(ValueError):
        build_report(run)
