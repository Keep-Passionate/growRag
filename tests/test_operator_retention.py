"""Synthetic policy tests: none of these numbers are HotpotQA experiment results."""

from dataclasses import replace

import pytest

from growrag.operator_retention import MetricObservation as M
from growrag.operator_retention import OperatorOutcome, assess_retention


def outcome(name, value, **kwargs):
    return OperatorOutcome(
        name, "same-state", "same-protocol", name + "/trace", (M("f1", value),), **kwargs
    )


def test_better_than_history_but_worse_than_fresh_is_retained_not_selected():
    result = assess_retention(
        outcome("new@1", 0.7), outcome("old@1", 0.55), fresh=outcome("fresh", 0.8)
    )
    assert result.retain_candidate
    assert result.improved_dimensions == ("f1",)
    assert result.fresh_deltas[0].benefit < 0
    assert not result.trusted and not result.execution_authorized


def test_unobserved_fresh_is_not_fabricated_and_not_a_retention_gate():
    result = assess_retention(outcome("new@1", 0.7), outcome("old@1", 0.55))
    assert result.retain_candidate and result.fresh_deltas is None


def test_quality_cost_tradeoff_is_recorded_not_summed():
    new = replace(outcome("new@1", 1), metrics=(M("f1", 1), M("cost", 2, False, "CNY")))
    old = replace(outcome("old@1", 0.5), metrics=(M("f1", 0.5), M("cost", 1, False, "CNY")))
    decision = assess_retention(new, old)
    assert decision.retain_candidate
    assert decision.improved_dimensions == ("f1",)
    assert decision.worsened_dimensions == ("cost",)


def test_only_cost_improvement_can_be_recorded():
    new = replace(outcome("new@1", 1), metrics=(M("cost", 1, False, "CNY"),))
    old = replace(outcome("old@1", 1), metrics=(M("cost", 2, False, "CNY"),))
    assert assess_retention(new, old).improved_dimensions == ("cost",)


@pytest.mark.parametrize("new,old", [(None, 0.5), (0.8, None), (0.5, 0.5), (0.4, 0.5)])
def test_no_observed_gain_does_not_qualify(new, old):
    assert not assess_retention(outcome("new@1", new), outcome("old@1", old)).retain_candidate


@pytest.mark.parametrize("failed_side", ["candidate", "historical"])
def test_failed_path_does_not_turn_into_zero_score(failed_side):
    new, old = outcome("new@1", 1), outcome("old@1", 0)
    if failed_side == "candidate":
        new = replace(new, completed=False)
    else:
        old = replace(old, completed=False)
    result = assess_retention(new, old)
    assert not result.retain_candidate and result.historical_deltas[0].benefit is None


def test_proxy_is_not_labelled_measured_success():
    result = assess_retention(
        outcome("new@1", 0.7, feedback_source="proxy"),
        outcome("old@1", 0.5, feedback_source="proxy"),
    )
    assert result.retain_candidate and result.evidence_status == "local_gain_estimated"


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_id", "other"),
        ("protocol_id", "other"),
        ("feedback_source", "human"),
        ("trace_ref", "new@1/trace"),
    ],
)
def test_incomparable_runs_are_rejected(field, value):
    with pytest.raises(ValueError):
        assess_retention(outcome("new@1", 0.7), replace(outcome("old@1", 0.5), **{field: value}))


@pytest.mark.parametrize("metric", [M("f1", 0.5, False), M("f1", 0.5, True, "percent")])
def test_incompatible_units_or_direction_are_rejected(metric):
    with pytest.raises(ValueError):
        assess_retention(outcome("new@1", 0.7), replace(outcome("old@1", 0.5), metrics=(metric,)))


def test_missing_metric_is_unknown_not_zero():
    result = assess_retention(outcome("new@1", 0.7), replace(outcome("old@1", 0.5), metrics=()))
    assert not result.retain_candidate and result.historical_deltas[0].raw_delta is None


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1"])
def test_invalid_metric_values(value):
    with pytest.raises(ValueError):
        M("x", value)


def test_duplicate_metric_is_rejected():
    with pytest.raises(ValueError):
        replace(outcome("new@1", 0.7), metrics=(M("f1", 0.7), M("f1", 0.8)))


def test_same_version_rerun_is_not_operator_growth():
    with pytest.raises(ValueError):
        assess_retention(outcome("old@1", 0.7), replace(outcome("old@1", 0.5), trace_ref="other"))
