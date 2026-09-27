"""Synthetic accounting and seal tests: no network, key, or benchmark execution."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "reformer_control_analysis",
    Path(__file__).resolve().parents[1] / "scripts/audits/analyze_reformer_control.py",
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def feedback(em, f1=None):
    return {"answer_em": em, "answer_f1": em if f1 is None else f1}


def call(qid, arm, role):
    return {
        "trace_id": f"run/{qid}/{arm}/{role}",
        "prompt_version": f"reformer-{role}-v1",
        "api_requests": 1,
        "input_tokens": 100,
        "output_tokens": 10,
        "reserved_cny": 0.01,
        "estimated_actual_cny": 0.001,
        "status": "completed",
    }


def fixture():
    reports, cached, calls = [], [], []
    for offset, qid in enumerate(("q1", "q2")):
        question = f"Question {qid}?"
        pattern = {
            "pattern_name": "Clarify",
            "transformation_rule": "clarify",
            "examples": [{"original": "x", "rewritten": "y"}],
        }
        shared = {
            "question": question,
            "question_id": f"run/{qid}/SELECTION",
            "pattern_id": "p0",
            "selection_sha256": f"sha-{qid}",
            "selected_pattern": pattern,
            "initial_selector_documents": [{"doc_id": "d1"}],
        }
        shared["selection_sha256"] = M.selection_digest(shared)
        selected_call = call(qid, "SELECTION", "select")
        selection = {
            "status": "completed",
            "result": shared,
            "calls": [selected_call],
            "elapsed_seconds": 1,
            "feedback": None,
        }
        arms = {}
        calls.append(selected_call)
        for arm, em in ((M.WITH, 1 - offset), (M.WITHOUT, offset)):
            selected = copy.deepcopy(pattern)
            if arm == M.WITHOUT:
                selected["examples"] = []
            result = {
                "question": question,
                "question_id": qid,
                "selected_pattern_id": "p0",
                "shared_selection_sha256": shared["selection_sha256"],
                "shared_selection_question_id": f"run/{qid}/SELECTION",
                "initial_selector_documents": [{"doc_id": "d1"}],
                "selected_pattern": selected,
                "include_examples": arm == M.WITH,
                "example_count": len(selected["examples"]),
                "answer": f"{qid}-{arm}",
                "rewritten_query": question if offset == 0 else f"{question} {arm}",
                "retrieval_query": question,
                "retrieved_documents": [{"doc_id": "d1"}],
            }
            requests = [call(qid, arm, role) for role in ("rewrite", "answer")]
            calls.extend(requests)
            arms[arm] = {
                "status": "completed",
                "result": result,
                "calls": requests,
                "elapsed_seconds": 2,
                "feedback": feedback(em),
            }
        reports.append(
            {
                "offset": offset,
                "question_id": qid,
                "question": question,
                "arm_order": M.expected_order(offset),
                "selection": selection,
                "arms": arms,
                "scoring_status": "completed",
            }
        )
        cached.append(
            {
                "offset": offset,
                "question_id": qid,
                "question": question,
                "arms": {
                    a: {"status": "completed", "feedback": feedback(offset)}
                    for a in (M.BASE, M.S2G, M.OLD)
                },
                "cached_usage": {a: M.usage([call(qid, a, "answer")]) for a in (M.BASE, M.S2G)},
                "new_usage": M.usage([call(qid, M.OLD, "answer")]),
            }
        )
    budget = {"calls": calls, "api_requests": 10, "reserved_cny": 0.1}
    return ["q1", "q2"], cached, [{"run_id": "run", "reports": reports, "budget": budget}]


def test_two_denominators_actual_shared_cost_and_hypothetical_paths():
    rows, summary = M.assemble(*fixture())
    assert summary["new_pair_complete_n"] == summary["all_five_complete_n"] == 2
    assert summary["paired_scores"][M.WITH]["answer_em"] == 0.5
    assert summary["with_vs_without"]["em_repairs"] == 1
    assert summary["with_vs_without"]["em_harms"] == 1
    assert summary["actual_new_usage_all_attempts"]["api_requests"] == 10
    assert summary["actual_new_role_usage"]["pattern_selection"]["api_requests"] == 2
    for arm in M.ARMS:
        assert summary["hypothetical_one_path_usage_on_complete_pairs"][arm]["api_requests"] == 6
        assert summary["hypothetical_one_path_mean_elapsed_seconds_on_complete_pairs"][arm] == 3
    assert summary["oracle_opportunity_not_a_router"][M.WITH]["extra_answer_em"] == 0.5
    assert summary["diagnostics"]["same_rewritten_query_n"] == 1
    assert summary["diagnostics"]["order_counts"] == {M.WITH: 1, M.WITHOUT: 1}
    assert rows[0]["actual_new_usage"]["api_requests"] == 5


def test_failed_cached_s2g_does_not_shrink_new_pair_denominator():
    args = fixture()
    args[1][1]["arms"][M.S2G] = {"status": "failed", "feedback": None}
    _, summary = M.assemble(*args)
    assert summary["new_pair_complete_n"] == 2
    assert summary["all_five_complete_n"] == 1
    assert summary["paired_scores"][M.WITH]["answer_em"] == 0.5
    assert summary["all_five_scores"][M.WITH]["answer_em"] == 1


@pytest.mark.parametrize(
    "mutation",
    ["pattern", "examples", "source", "question", "selection_ref", "id", "flag", "count", "order"],
)
def test_pair_coherence_rejected(mutation):
    args = fixture()
    report = args[2][0]["reports"][0]
    result = report["arms"][M.WITH]["result"]
    if mutation == "pattern":
        result["selected_pattern"]["transformation_rule"] = "other"
    elif mutation == "examples":
        result["selected_pattern"]["examples"] = []
    elif mutation == "source":
        result["initial_selector_documents"] = []
    elif mutation == "question":
        result["question"] = "other"
    elif mutation == "selection_ref":
        result["shared_selection_sha256"] = "other"
    elif mutation == "id":
        result["selected_pattern_id"] = "other"
    elif mutation == "flag":
        result["include_examples"] = False
    elif mutation == "count":
        result["example_count"] = 99
    else:
        report["arm_order"].reverse()
    with pytest.raises(ValueError):
        M.assemble(*args)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "orphan", "missing", "owner", "role", "total", "cost", "replay", "qid"],
)
def test_ledger_and_frozen_cohort_rejected(mutation):
    args = fixture()
    batch = args[2][0]
    calls, reports = batch["budget"]["calls"], batch["reports"]
    if mutation == "duplicate":
        calls.append(copy.deepcopy(calls[0]))
    elif mutation == "orphan":
        calls.append(call("q3", "SELECTION", "select"))
    elif mutation == "missing":
        reports[0]["selection"]["calls"] = []
    elif mutation == "owner":
        calls[0]["trace_id"] = "run/q2/SELECTION/select"
    elif mutation == "role":
        calls[0]["prompt_version"] = "reformer-answer-v1"
    elif mutation == "total":
        batch["budget"]["api_requests"] = 9
    elif mutation == "cost":
        batch["budget"]["reserved_cny"] = 0
    elif mutation == "replay":
        reports.append(copy.deepcopy(reports[0]))
    else:
        reports[0]["question_id"] = "not-in-cohort"
    with pytest.raises(ValueError):
        M.assemble(*args)


def test_failed_new_arm_still_owns_cost_and_unknown_usage_not_zero():
    args = fixture()
    outcome = args[2][0]["reports"][1]["arms"][M.WITH]
    outcome["status"], outcome["feedback"] = "failed", None
    outcome["calls"][-1]["estimated_actual_cny"] = None
    rows, summary = M.assemble(*args)
    assert summary["new_pair_complete_n"] == 1
    assert summary["actual_new_usage_all_attempts"]["api_requests"] == 10
    assert summary["actual_new_usage_all_attempts"]["total_estimated_cost_cny"] is None
    assert rows[1]["actual_new_usage"]["unknown_estimated_cost_cny_calls"] == 1


def test_unstarted_not_implicitly_completed_or_costed():
    ids, cache, _ = fixture()
    rows, summary = M.assemble(ids, cache, [])
    assert summary["new_pair_complete_n"] == 0
    assert summary["paired_scores"][M.WITH]["answer_em"] is None
    assert rows[0]["actual_new_usage"] is None
    assert summary["status_counts"][M.WITH] == {"not_executed": 2}


def test_f1_opportunity_separate_from_exact_match():
    args = fixture()
    args[2][0]["reports"][0]["arms"][M.WITH]["feedback"] = feedback(0, 0.5)
    _, summary = M.assemble(*args)
    oracle = summary["oracle_opportunity_not_a_router"][M.WITH]
    assert oracle["extra_answer_em"] == 0
    assert oracle["extra_answer_f1"] == 0.25


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def sealed_batch(tmp_path):
    ids, cache, batches = fixture()
    batch = batches[0]
    directory = tmp_path / "run"
    snapshot = {"files": {"src/test.py": {"sha256": hashlib.sha256(b"x").hexdigest(), "text": "x"}}}
    snapshot["sha256"] = M.fingerprint(snapshot["files"])
    launch = {
        "run_id": "run",
        "protocol": M.PROTOCOL,
        "manifest_sha256": M.EXPECTED_MANIFEST,
        "start": 0,
        "count": 2,
        "max_calls": 10,
        "question_ids": ids,
        "source_sha256": snapshot["sha256"],
        "model": "test-model",
    }
    predictions = []
    for report in batch["reports"]:
        for outcome in [report["selection"], *report["arms"].values()]:
            for request in outcome["calls"]:
                audit_path = (
                    directory
                    / "api_audit"
                    / f"{hashlib.sha256(request['trace_id'].encode()).hexdigest()}.json"
                )
                request["audit_path"] = str(audit_path)
                audit = {
                    k: request[k]
                    for k in (
                        "trace_id",
                        "prompt_version",
                        "api_requests",
                        "input_tokens",
                        "output_tokens",
                    )
                }
                audit.update(request={"model": "test-model"}, retry_count=0, finish_reason="stop")
                dump(audit_path, audit)
        prediction = copy.deepcopy(report)
        prediction["scoring_status"] = "pending"
        for outcome in [prediction["selection"], *prediction["arms"].values()]:
            outcome["feedback"] = None
        predictions.append(prediction)
        path = directory / "questions" / f"{report['offset']:04d}"
        dump(path / "prediction_report.json", prediction)
        dump(path / "report.json", report)
    dump(directory / "source_snapshot.json", snapshot)
    dump(directory / "launch_plan.json", launch)
    dump(directory / "reports.json", batch["reports"])
    dump(directory / "final_budget.json", batch["budget"])
    dump(
        directory / "predictions_frozen.json",
        {
            "run_id": "run",
            "completed_question_ids": ids,
            "completed_arm_ids": [{"question_id": qid, "arm": a} for qid in ids for a in M.ARMS],
            "reports_sha256_before_scoring": M.fingerprint(predictions),
        },
    )
    (directory / "events.jsonl").write_text('{"kind":"exit"}\n', encoding="utf-8")
    return directory, ids, cache


def test_sealed_batch_validated(tmp_path):
    directory, ids, _ = sealed_batch(tmp_path)
    batch = M.read_batch(directory, lambda p: json.loads(p.read_bytes()), {}, ids)
    assert len(batch["reports"]) == 2
    assert len(batch["audit_finish_reasons"]) == 10


@pytest.mark.parametrize(
    "mutation", ["source_text", "seal", "gold", "prediction", "launch", "audit", "exit"]
)
def test_tampered_artifacts_rejected(tmp_path, mutation):
    directory, ids, _ = sealed_batch(tmp_path)
    if mutation == "exit":
        (directory / "events.jsonl").write_text('{"kind":"launch"}\n', encoding="utf-8")
    else:
        filename = {
            "source_text": "source_snapshot.json",
            "seal": "predictions_frozen.json",
            "gold": "questions/0000/prediction_report.json",
            "prediction": "questions/0000/report.json",
            "launch": "launch_plan.json",
            "audit": "final_budget.json",
        }[mutation]
        path = directory / filename
        value = json.loads(path.read_bytes())
        if mutation == "source_text":
            value["files"]["src/test.py"]["text"] = "changed"
        elif mutation == "seal":
            value["reports_sha256_before_scoring"] = "wrong"
        elif mutation == "gold":
            value["arms"][M.WITH]["feedback"] = feedback(1)
        elif mutation == "prediction":
            value["arms"][M.WITH]["result"]["answer"] = "changed"
        elif mutation == "launch":
            value["question_ids"].reverse()
        else:
            value["calls"][0]["input_tokens"] = 999
        dump(path, value)
    with pytest.raises(ValueError):
        M.read_batch(directory, lambda p: json.loads(p.read_bytes()), {}, ids)


def test_readable_outputs_all_cases_and_caveats(tmp_path):
    rows, summary = M.assemble(*fixture())
    M.write_markdown(tmp_path, rows, summary)
    questions = (tmp_path / "QUESTIONS.md").read_text(encoding="utf-8")
    assert "q1" in questions and "q2" in questions
    assert "Oracle" in (tmp_path / "README.md").read_text(encoding="utf-8")
    json.dumps(summary, allow_nan=False)


def test_output_overwrite_or_escape_forbidden(tmp_path):
    runs = tmp_path / "runs"
    (runs / "old").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        M.analyze(runs, tmp_path / "none", runs / "old")
    with pytest.raises(ValueError, match="inside runs"):
        M.analyze(runs, tmp_path / "none", tmp_path / "outside")
