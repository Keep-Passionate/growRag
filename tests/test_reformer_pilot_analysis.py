"""Small synthetic accounting/paired-denominator tests; never use a model or gold file."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "reformer_analysis",
    Path(__file__).resolve().parents[1] / "scripts/audits/analyze_reformer_pilot.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BASE, S2G, REFORMER = MODULE.BASE, MODULE.S2G, MODULE.REFORMER
ARM = "REFORMER_PUBLIC_QWEN"


def feedback(em, f1=None):
    return {"answer_em": em, "answer_f1": em if f1 is None else f1, "raw_support_recall": 0.5}


def call(qid="q1", role="select", unknown=False):
    return {
        "trace_id": f"run/{qid}/{ARM}/reformer/01-{role}",
        "prompt_version": f"reformer-{role}-v1",
        "api_requests": 1,
        "reserved_cny": 0.01,
        "status": "failed" if unknown else "completed",
        "input_tokens": None if unknown else 100,
        "output_tokens": None if unknown else 10,
        "estimated_actual_cny": None if unknown else 0.001,
    }


def fixture():
    baseline, costs, reports, calls = [], [], [], []
    # q1 is repaired against BOTH; q2 harmed against BASE but not S2G.
    for offset, (qid, b, s, r) in enumerate((("q1", 0, 0, 1), ("q2", 1, 0, 0))):
        baseline.append(
            {
                "question_id": qid,
                "offset": offset,
                "question_type": "bridge",
                "arms": {
                    BASE: {"status": "completed", "feedback": feedback(b), "elapsed_seconds": 1},
                    S2G: {"status": "completed", "feedback": feedback(s), "elapsed_seconds": 4},
                },
            }
        )
        costs.append(
            {
                "question_id": qid,
                "offset": offset,
                "arms": {
                    name: {"usage": MODULE.usage([call(qid, "answer")])} for name in (BASE, S2G)
                },
            }
        )
        request = call(qid)
        calls.append(request)
        reports.append(
            {
                "question_id": qid,
                "offset": offset,
                "question": "Original query?",
                "outcome": {
                    "status": "completed",
                    "elapsed_seconds": 2,
                    "calls": [request],
                    "feedback": feedback(r),
                    "initial_feedback": {"raw_support_recall": 0.25},
                    "result": {
                        "selected_pattern": {"pattern_name": "Test pattern"},
                        "canonical_library_match": qid == "q1",
                        "rewritten_query": "Original query?" if qid == "q1" else "Changed?",
                        "retrieval_query": "Original query? Changed?",
                    },
                },
            }
        )
    batches = [
        {
            "run_id": "run",
            "reports": reports,
            "budget": {"calls": calls, "api_requests": 2, "reserved_cny": 0.02},
        }
    ]
    return ["q1", "q2"], baseline, costs, batches


def test_same_case_metrics_costs_oracle_and_diagnostics():
    rows, summary = MODULE.assemble(*fixture())
    assert len(rows) == summary["same_complete_case_n"] == 2
    assert summary["paired_scores"][BASE]["answer_em"] == 0.5
    assert summary["paired_scores"][REFORMER]["answer_em"] == 0.5
    assert summary["paired_vs_base"]["em_repairs"] == 1
    assert summary["paired_vs_base"]["em_harms"] == 1
    assert summary["oracle_opportunity_not_a_router"]["three_arm_em"] == 1
    assert summary["oracle_opportunity_not_a_router"]["reformer_repairs_both_others_wrong"] == 1
    assert summary["new_api_usage_all_attempts"]["api_requests"] == 2
    assert summary["cached_usage_all_selected_questions"][BASE]["api_requests"] == 2
    assert summary["rewrite_diagnostics"]["library_object_mismatch_rate"] == 0.5
    assert "library_mutation_rate" not in summary["rewrite_diagnostics"]
    assert summary["rewrite_diagnostics"]["repeat_rewrites"] == 1
    assert summary["exact_support_coverage"]["initial3_support_recall"] == 0.25
    oracle = summary["oracle_opportunity_not_a_router"]
    assert oracle["base_s2g_em"] == oracle["base_s2g_f1"] == 0.5
    assert oracle["extra_over_base_s2g_em"] == oracle["extra_over_base_s2g_f1"] == 0.5
    resources = summary["same_complete_case_online_resources"]
    assert resources["n"] == 2
    assert resources["arms"][REFORMER]["usage"]["api_requests"] == 2
    assert resources["arms"][BASE]["elapsed_seconds"]["mean"] == 1
    assert resources["arms"][S2G]["elapsed_seconds"]["total"] == 8


def test_failed_cached_arm_excludes_question_for_every_method():
    args = fixture()
    args[1][1]["arms"][S2G] = {"status": "failed", "feedback": None}
    _, summary = MODULE.assemble(*args)
    assert summary["same_complete_case_n"] == 1
    assert summary["paired_scores"][BASE]["answer_em"] == 0
    assert summary["paired_scores"][REFORMER]["answer_em"] == 1
    assert summary["paired_vs_base"]["em_harms"] == 0
    assert summary["new_api_usage_all_attempts"]["api_requests"] == 2
    resources = summary["same_complete_case_online_resources"]
    assert resources["n"] == 1
    assert resources["arms"][REFORMER]["usage"]["api_requests"] == 1
    assert resources["arms"][S2G]["elapsed_seconds"]["total"] == 4


def test_no_new_oracle_opportunity_when_new_arm_duplicates_existing_successes():
    args = fixture()
    args[1][0]["arms"][S2G]["feedback"] = feedback(1)
    _, summary = MODULE.assemble(*args)
    oracle = summary["oracle_opportunity_not_a_router"]
    assert oracle["base_s2g_em"] == oracle["three_arm_em"] == 1
    assert oracle["extra_over_base_s2g_em"] == oracle["extra_over_base_s2g_f1"] == 0


def test_oracle_f1_opportunity_not_conflated_with_em_repairs():
    args = fixture()
    args[1][0]["arms"][S2G]["feedback"] = feedback(0, 0.5)
    args[3][0]["reports"][0]["outcome"]["feedback"] = feedback(0, 0.75)
    _, summary = MODULE.assemble(*args)
    oracle = summary["oracle_opportunity_not_a_router"]
    assert oracle["extra_over_base_s2g_em"] == 0
    assert oracle["extra_over_base_s2g_f1"] == 0.125


def test_paired_missing_cost_or_duration_is_unknown_not_zero():
    args = fixture()
    args[2][0]["arms"][S2G]["usage"] = None
    args[1][0]["arms"][BASE]["elapsed_seconds"] = None
    _, summary = MODULE.assemble(*args)
    resources = summary["same_complete_case_online_resources"]["arms"]
    assert resources[S2G]["usage_missing_n"] == 1
    assert resources[S2G]["usage"] is None
    assert resources[BASE]["elapsed_seconds"]["known_subtotal"] == 1
    assert resources[BASE]["elapsed_seconds"]["mean"] is None
    assert resources[BASE]["elapsed_seconds"]["total"] is None


def test_failure_cost_unknown_does_not_become_zero_or_disappear():
    args = fixture()
    batch = args[3][0]
    bad = call("q2", unknown=True)
    batch["budget"]["calls"][1] = bad
    batch["reports"][1]["outcome"] = {"status": "failed", "calls": [bad], "feedback": None}
    rows, summary = MODULE.assemble(*args)
    assert summary["same_complete_case_n"] == 1
    assert summary["new_api_usage_all_attempts"]["api_requests"] == 2
    assert summary["new_api_usage_all_attempts"]["known_estimated_cost_cny_subtotal"] == 0.001
    assert summary["new_api_usage_all_attempts"]["total_estimated_cost_cny"] is None
    assert rows[1]["new_usage"]["unknown_estimated_cost_cny_calls"] == 1


def test_unstarted_questions_explicit_not_zero_cost_success():
    args = fixture()
    batch = args[3][0]
    batch["reports"].pop()
    batch["budget"]["calls"].pop()
    batch["budget"].update(api_requests=1, reserved_cny=0.01)
    rows, summary = MODULE.assemble(*args)
    assert rows[1]["new_usage"] is None
    assert rows[1]["arms"][REFORMER]["status"] == "not_executed"
    assert summary["new_status_counts"] == {"completed": 1, "not_executed": 1}


@pytest.mark.parametrize(
    "mode",
    [
        "duplicate_trace",
        "orphan",
        "wrong_owner",
        "missing",
        "request_sum",
        "reserve_sum",
        "changed_call",
    ],
)
def test_root_ledger_exact_ownership(mode):
    args = fixture()
    batch = args[3][0]
    if mode == "duplicate_trace":
        batch["budget"]["calls"].append(copy.deepcopy(batch["budget"]["calls"][0]))
    elif mode == "orphan":
        batch["budget"]["calls"].append(call("q3"))
    elif mode == "wrong_owner":
        batch["reports"][0]["outcome"]["calls"][0]["trace_id"] = "run/q2/REFORMER_PUBLIC_QWEN/x"
    elif mode == "missing":
        batch["reports"][0]["outcome"]["calls"] = []
    elif mode == "request_sum":
        batch["budget"]["api_requests"] = 1
    elif mode == "reserve_sum":
        batch["budget"]["reserved_cny"] = 0
    else:
        batch["reports"][0]["outcome"]["calls"] = [dict(call(), output_tokens=999)]
    with pytest.raises(ValueError):
        MODULE.assemble(*args)


@pytest.mark.parametrize("mode", ["id", "offset", "duplicate_question", "missing_cached"])
def test_frozen_cohort_enforced(mode):
    args = fixture()
    if mode == "id":
        args[3][0]["reports"][0]["question_id"] = "q3"
    elif mode == "offset":
        args[3][0]["reports"][0]["offset"] = 1
    elif mode == "duplicate_question":
        args[3][0]["reports"].append(copy.deepcopy(args[3][0]["reports"][0]))
    else:
        args[1].pop()
    with pytest.raises(ValueError):
        MODULE.assemble(*args)


@pytest.mark.parametrize("value", [-1, float("nan"), True])
def test_invalid_costs_rejected(value):
    with pytest.raises(ValueError):
        MODULE.usage([dict(call(), estimated_actual_cny=value)])


def test_empty_analysis_is_not_evidence_of_zero_accuracy():
    ids, baseline, costs, _ = fixture()
    _, summary = MODULE.assemble(ids, baseline, costs, [])
    assert summary["same_complete_case_n"] == 0
    assert summary["paired_scores"][REFORMER]["answer_em"] is None
    assert summary["oracle_opportunity_not_a_router"]["three_arm_em"] is None


def test_readable_outputs_include_all_questions_and_caveats(tmp_path):
    rows, summary = MODULE.assemble(*fixture())
    MODULE.write_markdown(tmp_path, rows, summary)
    assert "q1" in (tmp_path / "QUESTIONS.md").read_text(encoding="utf-8")
    assert "q2" in (tmp_path / "QUESTIONS.md").read_text(encoding="utf-8")
    assert "Oracle" in (tmp_path / "README.md").read_text(encoding="utf-8")
    json.dumps(summary, allow_nan=False)


def test_outputs_cannot_escape_or_overwrite(tmp_path):
    runs = tmp_path / "runs"
    (runs / "old").mkdir(parents=True)
    with pytest.raises(ValueError, match="inside runs"):
        MODULE.analyze(runs, tmp_path / "no.json", tmp_path / "outside")
    with pytest.raises(FileExistsError):
        MODULE.analyze(runs, tmp_path / "no.json", runs / "old")
