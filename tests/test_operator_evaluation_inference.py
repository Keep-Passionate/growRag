"""Synthetic fixed-policy inference; never reads dataset labels or invokes an API."""

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from growrag.experiments import operator_evaluation_inference as module
from growrag.experiments.operator_evaluation_summary import ARMS, METRICS, summarize_evaluation


def _data(n=5):
    reports, feedback = [], {}
    for index in range(n):
        qid = f"synthetic-{index}"
        reports.append({"question_id": qid, "arms": {}})
        feedback[qid] = {}
        for arm in ARMS:
            reports[-1]["arms"][arm] = {
                "status": "completed",
                "reader": {"answer": "synthetic"},
                "episode": {"searches": [{"step": 0}], "proposals": []},
                "calls": [],
            }
            feedback[qid][arm] = {**dict.fromkeys(METRICS, 0.0), "annotation_status": "valid"}
    return reports, feedback


def _comparison(result, name="fresh_minus_base", metric="answer_em"):
    return result["metrics"][metric]["comparisons"][name]


def test_policy_is_fixed_json_compatible_and_returns_independent_copies():
    policy = module.analysis_policy()
    assert policy["bootstrap"]["resamples"] == 10000
    assert policy["bootstrap"]["seed"] == 20260930
    assert policy["bootstrap"]["bit_generator"] == "PCG64"
    assert policy["bootstrap"]["batch_size_max"] == 256
    assert policy["bootstrap"]["quantile_method"] == "linear"
    assert policy["em_test"]["family_size"] == len(policy["comparisons"]) == 10
    assert json.loads(json.dumps(policy)) == policy
    policy["bootstrap"]["resamples"] = 1
    policy["comparisons"][0]["candidate"] = "mutated"
    assert module.analysis_policy()["bootstrap"]["resamples"] == 10000
    assert module.analysis_policy()["comparisons"][0]["candidate"] == "fresh"


def test_empty_common_set_keeps_all_tests_unknown_without_imputation():
    reports, feedback = _data()
    for values in feedback.values():
        values["memory500"].update(dict.fromkeys(METRICS))
    result = module.infer_evaluation(reports, feedback)
    for metric in METRICS:
        info = result["metrics"][metric]
        assert info["denominator"]["n"] == 0 and info["missing_questions"] == 5
        assert info["bootstrap_resamples_generated"] == 0
        assert info["shared_resampling_indices_sha256"] is None
        for row in info["comparisons"].values():
            assert row["mean_delta"] is None and row["ci95"] is None
            if metric == "answer_em":
                assert row["p_exact"] is None and row["p_holm"] is None
                assert row["reject_holm_0_05"] is None


def test_one_question_has_no_interval_but_exact_test_retains_definition():
    reports, feedback = _data(1)
    feedback["synthetic-0"]["fresh"]["answer_em"] = 1
    result = module.infer_evaluation(reports, feedback)
    row = _comparison(result)
    assert row["mean_delta"] == 1 and row["ci95"] is None
    assert row["p_exact"] == row["p_holm"] == 1
    assert row["warnings"] == ["fewer_than_two_common_questions_ci_unavailable"]


def test_equal_scores_have_degenerate_interval_not_zero_uncertainty():
    result = module.infer_evaluation(*_data())
    row = _comparison(result)
    assert row["mean_delta"] == 0
    assert row["ci95"] == {"lower": 0.0, "upper": 0.0}
    assert row["p_exact"] == row["p_holm"] == 1 and not row["reject_holm_0_05"]
    assert "constant_observed_paired_differences" in row["warnings"]
    assert "degenerate_percentile_interval_is_not_zero_uncertainty" in row["warnings"]


def test_five_all_repairs_ci_excludes_zero_but_exact_and_holm_do_not_reject():
    reports, feedback = _data(5)
    for values in feedback.values():
        values["fresh"].update(dict.fromkeys(METRICS, 1.0))
    row = _comparison(module.infer_evaluation(reports, feedback))
    assert row["mean_delta"] == 1 and row["ci95"] == {"lower": 1.0, "upper": 1.0}
    assert row["repairs"] == 5 and row["harms"] == 0
    assert row["p_exact"] == 0.0625
    assert row["p_holm"] == 0.625 and row["reject_holm_0_05"] is False


def test_repairs_harms_and_mean_are_computed_on_same_pairs():
    reports, feedback = _data(5)
    for index, (candidate, reference) in enumerate(
        zip([1, 1, 0, 1, 0], [0, 0, 1, 1, 0], strict=True)
    ):
        feedback[f"synthetic-{index}"]["fresh"]["answer_em"] = candidate
        feedback[f"synthetic-{index}"]["base"]["answer_em"] = reference
    row = _comparison(module.infer_evaluation(reports, feedback))
    assert row["repairs"] == 2 and row["harms"] == 1
    assert row["discordant_pairs"] == 3 and row["mean_delta"] == 0.2
    assert row["p_exact"] == 1


def test_inference_uses_common7_not_larger_pairwise_descriptive_denominators():
    reports, feedback = _data(3)
    feedback["synthetic-2"]["fresh"]["answer_em"] = 1
    feedback["synthetic-2"]["memory500"]["answer_em"] = None
    summary = summarize_evaluation(reports, feedback)
    before = deepcopy(summary)
    result = module.infer_evaluation(reports, feedback)
    assert summary == before
    assert summary["paired"]["fresh_minus_base"]["metrics"]["answer_em"]["n"] == 3
    assert result["metrics"]["answer_em"]["denominator"]["question_ids"] == [
        "synthetic-0",
        "synthetic-1",
    ]
    assert _comparison(result)["n"] == 2 and _comparison(result)["mean_delta"] == 0
    assert result["metrics"]["answer_em"]["missing_by_arm"]["memory500"] == 1
    assert result["metrics"]["answer_f1"]["denominator"]["n"] == 3


def test_every_contrast_shares_identical_resampling_indices():
    deltas = np.array([[-1, -2, 1], [0, 0, 0], [1, 2, -1]], dtype=float)
    means, digest = module._bootstrap_means(deltas, resamples=19, chunk_size=3)
    rng = np.random.Generator(np.random.PCG64(20260930))
    indices = rng.integers(0, 3, size=(19, 3), dtype=np.int64)
    assert np.array_equal(means, deltas[indices].mean(axis=1))
    assert np.array_equal(means[:, 1], 2 * means[:, 0])
    assert np.array_equal(means[:, 2], -means[:, 0])
    assert digest == hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()


def test_bootstrap_chunking_does_not_change_resamples_and_is_bounded():
    deltas = np.arange(18).reshape(6, 3) / 20
    a, a_sha = module._bootstrap_means(deltas, resamples=513, chunk_size=1)
    b, b_sha = module._bootstrap_means(deltas, resamples=513, chunk_size=256)
    assert np.array_equal(a, b) and a_sha == b_sha
    with pytest.raises(ValueError, match="chunk<=256"):
        module._bootstrap_means(deltas, chunk_size=257)


def test_fixed_public_policy_is_reproducible_preserves_inputs_and_global_rng():
    reports, feedback = _data(8)
    for index in range(8):
        feedback[f"synthetic-{index}"]["fresh"]["answer_em"] = index % 2
        feedback[f"synthetic-{index}"]["fresh"]["answer_f1"] = index / 8
    before = deepcopy((reports, feedback))
    global_before = np.random.get_state()
    first = module.infer_evaluation(reports, feedback)
    second = module.infer_evaluation(reports, feedback)
    global_after = np.random.get_state()
    assert first == second and (reports, feedback) == before
    assert global_before[0] == global_after[0]
    assert np.array_equal(global_before[1], global_after[1])
    assert global_before[2:] == global_after[2:]
    assert first["numpy_version"] == np.__version__
    assert first["policy_sha256"] == module.fingerprint(module.analysis_policy())
    assert first["independence_verified"] is False
    assert len({row["shared_resampling_indices_sha256"] for row in first["metrics"].values()}) == 1
    assert json.loads(json.dumps(first, allow_nan=False)) == first


def test_percentile_method_is_explicit_linear(monkeypatch):
    observed = []
    original = module.np.quantile

    def capture(values, q, *, axis, method):
        observed.append((values.shape, q, axis, method))
        return original(values, q, axis=axis, method=method)

    monkeypatch.setattr(module.np, "quantile", capture)
    module.infer_evaluation(*_data(2))
    assert observed == [((10000, 10), [0.025, 0.975], 0, "linear")] * 4


@pytest.mark.parametrize(
    "repairs,harms,expected", [(0, 0, 1), (5, 0, 0.0625), (9, 1, 22 / 1024), (3, 3, 1)]
)
def test_exact_mcnemar_hand_computed(repairs, harms, expected):
    assert module._mcnemar_exact(repairs, harms) == expected
    assert module._mcnemar_exact(harms, repairs) == expected


@pytest.mark.parametrize("repairs,harms", [(0, 500), (100, 120), (20, 31), (1, 1), (7, 0)])
def test_exact_mcnemar_matches_independent_scipy_reference(repairs, harms):
    binomtest = pytest.importorskip("scipy.stats").binomtest

    expected = binomtest(repairs, repairs + harms, p=0.5, alternative="two-sided").pvalue
    assert module._mcnemar_exact(repairs, harms) == pytest.approx(expected, rel=1e-12)


def test_holm_hand_calculation_keeps_planned_ten_hypothesis_family():
    values = [0.01, 0.04, 0.03, 0.002, 0.05, 0.05, 0.12, None, 1, 0.001]
    expected = [0.08, 0.24, 0.21, 0.018, 0.25, 0.25, 0.36, None, 1, 0.01]
    actual = module._holm(values)
    for result, wanted in zip(actual, expected, strict=True):
        assert result is None if wanted is None else result == pytest.approx(wanted)
    assert module._holm([0.001, 0.01] + [None] * 8) == [0.01, 0.09] + [None] * 8
    assert module._holm([None] * 10) == [None] * 10


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, 0.5])
def test_invalid_or_nonbinary_em_is_rejected_by_existing_summary(value):
    reports, feedback = _data(2)
    feedback["synthetic-0"]["fresh"]["answer_em"] = value
    with pytest.raises(ValueError):
        module.infer_evaluation(reports, feedback)


def test_production_500_question_size_runs_all_fixed_comparisons():
    reports, feedback = _data(500)
    for index in range(500):
        for offset, arm in enumerate(ARMS):
            feedback[f"synthetic-{index}"][arm].update(
                answer_em=int((index + offset) % 3 == 0),
                answer_f1=((index + offset) % 11) / 10,
                raw_support_recall=((index + offset) % 7) / 6,
                visible_support_recall=((index + offset) % 5) / 4,
            )
    result = module.infer_evaluation(reports, feedback)
    assert result["question_count"] == 500
    for info in result["metrics"].values():
        assert info["denominator"]["n"] == 500
        assert len(info["denominator"]["question_ids_sha256"]) == 64
        assert info["missing_questions"] == 0
        assert len(info["comparisons"]) == 10
        assert info["bootstrap_resamples_generated"] == 10000
        assert all(row["ci95"] is not None for row in info["comparisons"].values())
