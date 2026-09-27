"""Synthetic local logs only; never call an API or read project credentials."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "offline_audit",
    Path(__file__).resolve().parents[1] / "scripts" / "audits" / "audit_shared500.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def create_run(root, *, suffix="0000_0001", qid="q1", failed=False):
    directory = root / f"2026-09-27_s2g_shared500_v1_{suffix}"
    trace = f"{directory.name}/{qid}/BASE1_AUTHOR_READER/01-answer"
    prompt = "s2g-author-5d842a6-answer-api-v1"
    request = {"model": "pinned", "messages": [{"role": "user", "content": "synthetic"}]}
    audit_path = directory / "api_audit" / f"{MODULE.sha(trace.encode())}.json"
    cost = None if failed else (10 * 0.2 + 5 * 0.8) / 1_000_000
    call = {
        "trace_id": trace,
        "prompt_version": prompt,
        "reserved_cny": 0.1,
        "status": "failed" if failed else "completed",
        "api_requests": 1,
        "input_tokens": None if failed else 10,
        "output_tokens": None if failed else 5,
        "audit_path": str(audit_path),
        "estimated_actual_cny": cost,
    }
    if not failed:
        call["returned_model"] = "pinned"
    audit = {
        **call,
        "request": request,
        "request_sha256": MODULE.sha(json.dumps(request, ensure_ascii=False).encode()),
        "transport_source": "live_api",
        "network_attempted": True,
        "retry_count": 0,
    }
    if not failed:
        audit.update(
            http_status=200,
            request_id="request-one",
            finish_reason="stop",
            response={
                "model": "pinned",
                "id": "response-one",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"finish_reason": "stop"}],
            },
        )
    launch = {
        "series": "500_v1",
        "run_id": directory.name,
        "manifest_sha256": MODULE.EXPECTED_MANIFEST,
        "model": "pinned",
        "protocol": "v3",
        "generation_profile": "v5",
        "question_ids": [qid],
        "price_input_cny_per_million": 0.2,
        "price_output_cny_per_million": 0.8,
        "historical_budget": {"estimated_actual_cny": 90000},
    }
    budget = {
        "calls": [call],
        "api_requests": 1,
        "reserved_cny": 0.1,
        "input_tokens": call["input_tokens"],
        "output_tokens": call["output_tokens"],
        "estimated_actual_cny": cost,
    }
    dump(directory / "launch_plan.json", launch)
    dump(
        directory / "reports.json",
        [{"question_id": qid, "arms": {"BASE1_AUTHOR_READER": {"calls": [call]}}}],
    )
    dump(directory / "final_budget.json", budget)
    dump(
        directory / "request_journal" / "0000_intent.json",
        {
            "trace_id": trace,
            "prompt_version": prompt,
            "prior_reserved_cny": 0.0,
            "potential_reserved_cny": 0.1,
            "request_fingerprint": MODULE.sha(
                json.dumps(request["messages"], ensure_ascii=False, sort_keys=True).encode()
            ),
        },
    )
    dump(directory / "request_journal" / "0000_after.json", budget)
    dump(audit_path, audit)
    (directory / "events.jsonl").write_text(
        json.dumps({"kind": "exit", "status": "failed" if failed else "completed"}) + "\n"
    )
    return directory


def test_known_cost_is_counted_once_without_historical_budget(tmp_path):
    create_run(tmp_path)
    summary, rows = MODULE.audit_series(tmp_path)
    assert summary["issues"] == []
    assert summary["known_estimate_cny_subtotal"] == pytest.approx(0.000006)
    assert summary["total_estimate_cny"] == pytest.approx(0.000006)
    assert summary["api_requests"] == 1
    assert len(rows) == 1


def test_failed_transport_keeps_unknown_cost_not_zero(tmp_path):
    create_run(tmp_path, failed=True)
    summary, _ = MODULE.audit_series(tmp_path)
    assert summary["issues"] == []
    assert summary["unknown_cost_call_count"] == 1
    assert summary["total_estimate_cny"] is None
    assert summary["unknown_cost_reserved_cny"] == 0.1


def test_journal_request_disagreement_is_detected(tmp_path):
    directory = create_run(tmp_path)
    path = directory / "request_journal" / "0000_intent.json"
    value = json.loads(path.read_text())
    value["request_fingerprint"] = "tampered"
    dump(path, value)
    summary, _ = MODULE.audit_series(tmp_path)
    assert "intent_request_fingerprint_mismatch" in {i["code"] for i in summary["issues"]}


def test_provider_ids_cannot_repeat_across_batches(tmp_path):
    create_run(tmp_path)
    create_run(tmp_path, suffix="0001_0002", qid="q2")
    summary, _ = MODULE.audit_series(tmp_path)
    codes = {i["code"] for i in summary["issues"]}
    assert {"duplicate_provider_request_id", "duplicate_provider_response_id"} <= codes


def test_unclosed_batch_excluded_before_request_data_is_read(tmp_path):
    directory = tmp_path / "2026-09-27_s2g_shared500_v1_0000_0025"
    directory.mkdir()
    (directory / "launch_plan.json").write_text("not yet valid JSON")
    summary, rows = MODULE.audit_series(tmp_path)
    assert summary["issues"] == []
    assert len(summary["excluded_unclosed_batches"]) == 1
    assert not rows


@pytest.mark.parametrize("change_cost", [False, True])
def test_post_api_component_failure_is_not_transport_failure_or_cost_change(tmp_path, change_cost):
    directory = create_run(tmp_path)
    after_path = directory / "request_journal" / "0000_after.json"
    after = json.loads(after_path.read_text())
    after["block_reason"] = None
    dump(after_path, after)
    final = {**after, "block_reason": "author_component_failure"}
    if change_cost:
        final["reserved_cny"] = 100
    dump(directory / "final_budget.json", final)
    report_path = directory / "reports.json"
    reports = json.loads(report_path.read_text())
    reports[0]["arms"]["BASE1_AUTHOR_READER"].update(status="failed", error_type="ValueError")
    dump(report_path, reports)
    tail = [
        {"kind": "api_response", "trace_id": after["calls"][0]["trace_id"]},
        {"kind": "arm_failed", "error_type": "ValueError"},
        {"kind": "question_complete", "question_id": "q1", "status": "failed"},
        {"kind": "exit", "status": "failed"},
    ]
    for event in tail:
        event["arm"] = "BASE1_AUTHOR_READER"
    (directory / "events.jsonl").write_text("\n".join(json.dumps(e) for e in tail))
    summary, rows = MODULE.audit_series(tmp_path)
    if change_cost:
        assert "last_after_final_mismatch" in {i["code"] for i in summary["issues"]}
    else:
        assert summary["issues"] == []
        assert summary["post_transport_component_failure_batches"] == [directory.name]
        assert rows[0]["status"] == "completed"
        assert summary["unknown_cost_call_count"] == 0
