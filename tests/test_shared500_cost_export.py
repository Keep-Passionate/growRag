"""Synthetic accounting tests: no API, corpus, key, or author code required."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "cost_export",
    Path(__file__).resolve().parents[1] / "scripts/audits/export_shared500_costs.py",
)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)
BASE, S2G = EXPORT.ARMS


def call(qid="q1", arm=BASE, role="answer", index=1, unknown=False):
    return {
        "run_id": "run",
        "question_id": qid,
        "arm": arm,
        "role": role,
        "trace_id": f"run/{qid}/{arm}/{index}-{role}",
        "status": "failed" if unknown else "completed",
        "api_requests": 1,
        "requested_model": "fixed-model",
        "returned_model": None if unknown else "fixed-model",
        "reserved_cny": 0.01,
        "input_tokens": None if unknown else 100,
        "output_tokens": None if unknown else 10,
        "recomputed_estimate_cny": None if unknown else 0.001,
        "elapsed_seconds": 2.0,
    }


def fixture():
    calls = [call(), call(arm=S2G, role="judge"), call(arm=S2G, index=2)]
    outcomes = {
        arm: {
            "status": "completed",
            "calls": [c for c in calls if c["arm"] == arm],
            "elapsed_seconds": 5.0,
            "result": {"retrieval_rounds": 1},
        }
        for arm in EXPORT.ARMS
    }
    events = []
    for arm in EXPORT.ARMS:
        events.extend(
            [
                {"kind": "arm_start", "question_id": "q1", "arm": arm},
                {"kind": "retrieval", "arm": arm, "round": 1},
            ]
        )
    return [
        {
            "run_id": "run",
            "events": events,
            "reports": [
                {"question_id": "q1", "offset": 0, "complete_pair": True, "arms": outcomes}
            ],
        }
    ], calls


def test_known_costs_roles_and_complete_pairs():
    batches, calls = fixture()
    rows, summary = EXPORT.assemble(["q1"], batches, calls)
    assert summary["usage_all_attempts"]["api_requests"] == 3
    assert summary["usage_all_attempts"]["total_input_tokens"] == 300
    assert summary["usage_all_attempts"]["total_estimated_cost_cny"] == pytest.approx(0.003)
    assert rows[0]["arms"][BASE]["roles"]["judge"]["usage"] is None
    assert rows[0]["arms"][S2G]["roles"]["judge"]["usage"]["api_requests"] == 1
    assert summary["arms"][S2G]["usage_complete_pairs"]["api_requests"] == 2


def test_failed_call_is_counted_unknown_is_not_zero():
    batches, calls = fixture()
    failed = call(arm=S2G, index=2, unknown=True)
    calls[-1] = failed
    report = batches[0]["reports"][0]
    report["complete_pair"] = False
    report["arms"][S2G].update(status="failed", calls=calls[1:], error_type="Timeout")
    rows, summary = EXPORT.assemble(["q1"], batches, calls)
    total = summary["usage_all_attempts"]
    assert total["api_requests"] == 3
    assert total["known_input_tokens_subtotal"] == 200
    assert total["total_input_tokens"] is None
    assert total["total_estimated_cost_cny"] is None
    assert total["unknown_estimated_cost_cny_calls"] == 1
    assert total["reserved_cny"] == pytest.approx(0.03)
    assert rows[0]["arms"][S2G]["retrieval_rounds_observed"] == 1
    assert rows[0]["arms"][S2G]["retrieval_rounds_complete"] is False
    assert summary["arms"][BASE]["usage_complete_pairs"]["api_requests"] == 0


@pytest.mark.parametrize("explicit", [True, False])
def test_unexecuted_arm_has_null_usage_not_zero_cost_answer(explicit):
    batches, calls = fixture()
    report = batches[0]["reports"][0]
    report["complete_pair"] = False
    del report["arms"][S2G]
    if explicit:
        report["arms"][S2G] = {"status": "not_executed", "calls": []}
    batches[0]["events"] = batches[0]["events"][:2]
    rows, summary = EXPORT.assemble(["q1"], batches, calls[:1])
    assert rows[0]["arms"][S2G] == {"status": "not_executed", "usage": None}
    assert summary["arms"][S2G]["status_counts"] == {"not_executed": 1}


def test_complete_pair_must_match_outcomes():
    batches, calls = fixture()
    batches[0]["reports"][0]["complete_pair"] = False
    with pytest.raises(ValueError, match="complete_pair"):
        EXPORT.assemble(["q1"], batches, calls)


@pytest.mark.parametrize("case", ["duplicate", "orphan", "moved", "missing"])
def test_every_call_belongs_to_one_reported_arm(case):
    batches, calls = fixture()
    if case == "duplicate":
        calls.append(copy.deepcopy(calls[0]))
    elif case == "orphan":
        calls.append(call(qid="q2"))
    elif case == "moved":
        calls[-1] = {**calls[-1], "arm": BASE}
    else:
        calls.pop()
    with pytest.raises(ValueError):
        EXPORT.assemble(["q1"], batches, calls)


@pytest.mark.parametrize("case", ["missing", "duplicate", "wrong_id", "negative", "round"])
def test_exact_cohort_and_rounds_required(case):
    batches, calls = fixture()
    report = batches[0]["reports"][0]
    ids = ["q1"]
    if case == "missing":
        ids.append("q2")
    elif case == "duplicate":
        batches[0]["reports"].append(copy.deepcopy(report))
    elif case == "wrong_id":
        ids = ["other"]
    elif case == "negative":
        report["offset"] = -1
    else:
        report["arms"][S2G]["result"]["retrieval_rounds"] = 2
    with pytest.raises(ValueError):
        EXPORT.assemble(ids, batches, calls)


def test_projection_does_not_include_question_answer_or_messages():
    batches, calls = fixture()
    report = batches[0]["reports"][0]
    report.update(question="SECRET_CONTENT", offline_gold_answers=["SECRET_CONTENT"])
    report["arms"][BASE]["result"]["answer"] = "SECRET_CONTENT"
    rows, summary = EXPORT.assemble(["q1"], batches, calls)
    assert "SECRET_CONTENT" not in json.dumps([rows, summary])


def test_output_cannot_overwrite_or_escape_runs(tmp_path):
    runs = tmp_path / "runs"
    existing = runs / "old"
    existing.mkdir(parents=True)
    for target in (runs, tmp_path / "elsewhere"):
        with pytest.raises(ValueError, match="inside runs"):
            EXPORT.export(runs, tmp_path / "missing.json", target)
    with pytest.raises(FileExistsError):
        EXPORT.export(runs, tmp_path / "missing.json", existing)


def test_invalid_manifest_rejected_before_loading_records(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"question_ids": ["q1"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen shared500"):
        EXPORT.export(tmp_path / "runs", manifest, tmp_path / "runs" / "new")
