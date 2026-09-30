"""Synthetic descriptive summaries: no dataset, provider, I/O or trained memory."""

from copy import deepcopy

import pytest

from growrag.experiments.operator_evaluation_summary import ARMS, METRICS, summarize_evaluation


@pytest.fixture
def data():
    reports, feedback = [], {}
    for n in range(5):
        qid = f"synthetic-{n}"
        reports.append({"question_id": qid, "arms": {}})
        feedback[qid] = {}
        for arm in ARMS:
            reports[-1]["arms"][arm] = {
                "status": "completed",
                "reader": {"answer": "synthetic prediction"},
                "episode": {
                    "searches": [{"step": 0}],
                    "proposals": [],
                    "stop_reason": "retrieval_budget",
                },
                "calls": [
                    {
                        "trace_id": f"{qid}/{arm}/reader",
                        "api_requests": 1,
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "estimated_actual_cny": 0.001,
                        "reserved_cny": 0.02,
                    }
                ],
            }
            feedback[qid][arm] = {**dict.fromkeys(METRICS, 0.0), "annotation_status": "valid"}
    feedback["synthetic-0"]["memory50"]["answer_em"] = 1.0
    feedback["synthetic-1"]["fresh"]["answer_em"] = 1.0
    return reports, feedback


def fail(data, n, arm, *, unstarted=False):
    reports, feedback = data
    row = reports[n]["arms"][arm]
    row.update(
        status="failure_induced_unstarted" if unstarted else "failed", episode=None, reader=None
    )
    if unstarted:
        row["calls"] = []
    feedback[f"synthetic-{n}"][arm].update(dict.fromkeys(METRICS))


def test_repairs_harms_are_paired_with_explicit_denominators(data):
    before = deepcopy(data)
    result = summarize_evaluation(*data)
    em = result["paired"]["memory50_minus_fresh"]["metrics"]["answer_em"]
    assert em["repairs"] == em["harms"] == 1
    assert em["both_correct"] == 0 and em["both_wrong"] == 3
    assert em["mean"] == 0 and em["n"] == 5
    assert em["repair_rate"] == 0.25 and em["harm_rate"] == 1
    assert em["denominator"]["question_ids"] == [f"synthetic-{n}" for n in range(5)]
    assert len(em["denominator"]["question_ids_sha256"]) == 64
    assert data == before


def test_seven_way_and_pairwise_and_own_denominators_are_distinct(data):
    fail(data, 4, "memory500")
    result = summarize_evaluation(*data)
    assert result["all_arms_common_counts"]["answer_em"] == 4
    assert result["arms"]["base"]["own_scorable"]["answer_em"]["n"] == 5
    assert result["arms"]["base"]["all_arms_common_scorable"]["answer_em"]["n"] == 4
    assert result["paired"]["memory50_minus_fresh"]["metrics"]["answer_em"]["n"] == 5
    assert result["paired"]["memory500_minus_fresh"]["metrics"]["answer_em"]["n"] == 4


def test_empty_denominator_returns_unknown_not_zero(data):
    for n in range(5):
        fail(data, n, "memory500")
    result = summarize_evaluation(*data)
    em = result["paired"]["memory500_minus_fresh"]["metrics"]["answer_em"]
    assert em["n"] == 0 and em["mean"] is None
    assert em["repair_rate"] is None and em["harm_rate"] is None
    assert result["arms"]["memory500"]["execution"]["reuse_rate_known_episodes"] is None


def test_failures_unstarted_abstentions_and_invalid_annotations_are_separate(data):
    reports, feedback = data
    fail(data, 0, "memory500")
    fail(data, 1, "memory500", unstarted=True)
    reports[2]["arms"]["memory500"]["reader"]["answer"] = ""
    feedback["synthetic-3"]["memory500"].update(
        **dict.fromkeys(METRICS), annotation_status="invalid"
    )
    arm = summarize_evaluation(*data)["arms"]["memory500"]
    assert arm["status_counts"] == {"completed": 3, "failed": 1, "failure_induced_unstarted": 1}
    assert arm["attempted_questions"] == 4 and arm["completed_empty_answers"] == 1
    assert arm["annotation_status_counts"] == {"valid": 4, "invalid": 1}
    assert arm["own_scorable"]["answer_em"]["n"] == 2
    assert arm["execution"]["unknown_episode_questions"] == 1


def test_completed_plan_rejection_still_scores(data):
    data[0][0]["arms"]["memory50"]["episode"]["stop_reason"] = "plan_rejected"
    arm = summarize_evaluation(*data)["arms"]["memory50"]
    assert arm["own_scorable"]["answer_em"]["n"] == 5
    assert arm["execution"]["plan_rejected_known_count"] == 1


def test_reuse_requires_actual_search_not_merely_selection_or_arm_name(data):
    reports, _ = data
    for n in (0, 1):
        reports[n]["arms"]["memory50"]["episode"]["proposals"] = [
            {"origin": "reuse", "spec": {"operator_id": "synthetic"}}
        ]
    reports[0]["arms"]["memory50"]["episode"]["searches"] += [{"step": 1}, {"step": 1}]
    reports[2]["arms"]["memory50"]["episode"]["proposals"] = [
        {"origin": "fresh", "spec": {"operator_id": "synthetic"}}
    ]
    reports[2]["arms"]["memory50"]["episode"]["searches"] += [{"step": 1}]
    report = summarize_evaluation(*data)["arms"]["memory50"]["execution"]
    assert report["executed_reuse_questions"] == report["reuse_actions_known_subtotal"] == 1
    assert report["reuse_queries_known_subtotal"] == 2
    assert report["reuse_rate_known_episodes"] == 0.2


def test_reader_failure_does_not_erase_executed_reuse(data):
    reports, feedback = data
    row = reports[0]["arms"]["memory50"]
    row.update(status="failed", reader=None)
    feedback["synthetic-0"]["memory50"].update(dict.fromkeys(METRICS))
    row["episode"].update(
        proposals=[{"origin": "reuse", "spec": {}}], searches=[{"step": 0}, {"step": 1}]
    )
    result = summarize_evaluation(*data)
    assert result["arms"]["memory50"]["execution"]["executed_reuse_questions"] == 1
    assert result["arms"]["memory50"]["own_scorable"]["answer_em"]["n"] == 4


def test_unknown_usage_preserves_known_subtotal_and_failed_costs(data):
    fail(data, 0, "memory500")
    call = data[0][0]["arms"]["memory500"]["calls"][0]
    call.update(estimated_actual_cny=None, output_tokens=None)
    result = summarize_evaluation(*data)
    own = result["arms"]["memory500"]["all_attempted_cost"]
    assert own["estimated_actual_cny"] == {
        "total": None,
        "known_subtotal": 0.004,
        "unknown_calls": 1,
    }
    assert own["reserved_cny"]["total"] == pytest.approx(0.1)
    all_cost = result["all_branches_evaluation_cost"]
    assert all_cost["recorded_calls"] == 35
    assert all_cost["estimated_actual_cny"]["total"] is None
    assert all_cost["estimated_actual_cny"]["known_subtotal"] == pytest.approx(0.034)


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_metric_refused(data, invalid):
    data[1]["synthetic-0"]["base"]["answer_em"] = invalid
    with pytest.raises(ValueError):
        summarize_evaluation(*data)


def test_duplicate_requests_not_counted_twice(data):
    row = data[0][0]["arms"]["base"]
    row["calls"].append(deepcopy(row["calls"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        summarize_evaluation(*data)


def test_failed_execution_cannot_be_given_a_score(data):
    fail(data, 0, "memory500")
    data[1]["synthetic-0"]["memory500"]["answer_em"] = 0
    with pytest.raises(ValueError, match="quality score"):
        summarize_evaluation(*data)


def test_complete_scaled_fixture_contains_500_distinct_questions(data):
    reports, feedback = [], {}
    for n in range(500):
        qid = f"fixture-scale-{n}"
        row = deepcopy(data[0][0])
        row["question_id"] = qid
        for arm in ARMS:
            row["arms"][arm]["calls"][0]["trace_id"] = f"{qid}/{arm}/reader"
        reports.append(row)
        feedback[qid] = deepcopy(data[1]["synthetic-0"])
    result = summarize_evaluation(reports, feedback)
    assert result["question_count"] == 500
    assert result["all_branches_evaluation_cost"]["recorded_calls"] == 3500
    assert all(n == 500 for n in result["all_arms_common_counts"].values())
