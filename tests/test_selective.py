from __future__ import annotations

from dataclasses import replace

import pytest

from growrag.evaluation import PairedOutcome
from growrag.evaluation.selective import (
    OperatingPoint,
    SelectiveCandidate,
    evaluate_operating_point,
    sweep_operating_points,
)
from growrag.selection import CandidateGateTrace


def trace(
    experience: str,
    rank: int,
    applicability: float,
    risk: float,
    *,
    environment_score: float = 1.0,
    rejection_reasons: tuple[str, ...] = (),
) -> CandidateGateTrace:
    return CandidateGateTrace(
        experience_id=experience,
        recall_rank=rank,
        state="active",
        environment_compatible="environment_mismatch" not in rejection_reasons,
        environment_score=environment_score,
        environment_mode="strict",
        environment_hard_mismatches=(
            ("corpus_id",) if "environment_mismatch" in rejection_reasons else ()
        ),
        environment_soft_mismatches=(),
        provenance_valid="missing_provenance" not in rejection_reasons,
        reliability_observations=3,
        reliability_benefits=1,
        reliability_mean_gain=0.5,
        risk_upper_bound=risk,
        risk_measure="wilson_95_conditional_breakage_upper_bound",
        reliability_valid=not any(
            reason
            in {
                "insufficient_reliability_observations",
                "insufficient_reliability_benefits",
                "reliability_mean_gain_too_low",
                "risk_upper_bound_too_high",
            }
            for reason in rejection_reasons
        ),
        query_plan_valid="missing_prepared_query_plan" not in rejection_reasons,
        applicability_score=applicability,
        applicability_scorer_id="signature-coverage-v2",
        applicability_matched_signatures=("intent",),
        applicability_required_signatures=("intent",),
        applicability_matched_contraindications=(
            ("answer_already_explicit",)
            if "applicability_contraindication_matched" in rejection_reasons
            else ()
        ),
        applicability_valid=not any(
            reason
            in {
                "applicability_contraindication_matched",
                "missing_applicability_estimate",
                "missing_applicability_features",
                "missing_applicability_scorer",
                "applicability_below_threshold",
            }
            for reason in rejection_reasons
        ),
        passed=not rejection_reasons,
        first_failure=rejection_reasons[0] if rejection_reasons else None,
        rejection_reasons=rejection_reasons,
    )


def candidate(
    target: str,
    experience: str,
    rank: int,
    applicability: float,
    risk: float,
    outcome: PairedOutcome,
    *,
    environment_score: float = 1.0,
    rejection_reasons: tuple[str, ...] = (),
) -> SelectiveCandidate:
    scores = {
        PairedOutcome.BENEFIT: (0.0, 1.0),
        PairedOutcome.HARM: (1.0, 0.0),
        PairedOutcome.BOTH_GOOD: (1.0, 1.0),
        PairedOutcome.BOTH_BAD: (0.0, 0.0),
    }
    direct, reuse = scores[outcome]
    return SelectiveCandidate(
        target_query_id=target,
        gate_trace=trace(
            experience,
            rank,
            applicability,
            risk,
            environment_score=environment_score,
            rejection_reasons=rejection_reasons,
        ),
        direct_score=direct,
        reuse_score=reuse,
        good_threshold=0.5,
    )


def test_operating_point_exposes_safety_coverage_tradeoff() -> None:
    rows = (
        candidate("q1", "safe-benefit", 1, 0.9, 0.1, PairedOutcome.BENEFIT),
        candidate("q2", "risky-harm", 1, 0.8, 0.6, PairedOutcome.HARM),
    )

    strict, permissive = sweep_operating_points(
        rows,
        (
            OperatingPoint(minimum_applicability=0.7, maximum_risk_upper_bound=0.2),
            OperatingPoint(minimum_applicability=0.7, maximum_risk_upper_bound=0.7),
        ),
        target_query_ids=("q1", "q2"),
    )

    assert strict.coverage == 0.5
    assert strict.benefit_count == 1
    assert strict.harm_count == 0
    assert permissive.coverage == 1.0
    assert permissive.harm_count == 1
    assert permissive.conditional_breakage_rate == 1.0


def test_selection_matches_gate_order_without_using_outcome() -> None:
    rows = (
        candidate("q1", "near-risky", 1, 0.8, 0.2, PairedOutcome.BOTH_BAD),
        candidate("q1", "fit-safe", 2, 0.9, 0.1, PairedOutcome.BENEFIT),
    )

    result = evaluate_operating_point(
        rows,
        OperatingPoint(minimum_applicability=0.5, maximum_risk_upper_bound=0.5),
        target_query_ids=("q1",),
    )

    assert result.selected_pairs == (("q1", "fit-safe"),)
    assert result.benefit_count == 1
    assert result.mean_gain == 1.0


def test_duplicate_rank_within_target_is_rejected() -> None:
    rows = (
        candidate("q1", "a", 1, 0.8, 0.2, PairedOutcome.BENEFIT),
        candidate("q1", "b", 1, 0.9, 0.1, PairedOutcome.BENEFIT),
    )

    with pytest.raises(ValueError, match="duplicate recall rank"):
        evaluate_operating_point(
            rows,
            OperatingPoint(minimum_applicability=0.5, maximum_risk_upper_bound=0.5),
            target_query_ids=("q1",),
        )


def test_empty_sweep_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        sweep_operating_points((), (), target_query_ids=("q1",))


def test_coverage_denominator_includes_targets_with_no_candidate() -> None:
    rows = (candidate("q1", "a", 1, 0.9, 0.1, PairedOutcome.BENEFIT),)

    result = evaluate_operating_point(
        rows,
        OperatingPoint(minimum_applicability=0.5, maximum_risk_upper_bound=0.5),
        target_query_ids=("q1", "q2"),
    )

    assert result.target_count == 2
    assert result.accepted_count == 1
    assert result.coverage == 0.5


def test_contraindication_is_hard_veto_at_zero_threshold() -> None:
    rows = (
        candidate(
            "q1",
            "blocked",
            1,
            0.0,
            0.0,
            PairedOutcome.BENEFIT,
            rejection_reasons=("applicability_contraindication_matched",),
        ),
    )

    result = evaluate_operating_point(
        rows,
        OperatingPoint(minimum_applicability=0.0, maximum_risk_upper_bound=1.0),
        target_query_ids=("q1",),
    )

    assert result.accepted_count == 0
    assert result.coverage == 0.0


def test_environment_score_uses_the_same_tie_break_as_gate() -> None:
    rows = (
        candidate(
            "q1",
            "soft-mismatch",
            1,
            0.9,
            0.1,
            PairedOutcome.BOTH_BAD,
            environment_score=0.9,
        ),
        candidate(
            "q1",
            "exact-environment",
            2,
            0.9,
            0.1,
            PairedOutcome.BENEFIT,
            environment_score=1.0,
        ),
    )

    result = evaluate_operating_point(
        rows,
        OperatingPoint(minimum_applicability=0.5, maximum_risk_upper_bound=0.5),
        target_query_ids=("q1",),
    )

    assert result.selected_pairs == (("q1", "exact-environment"),)
    assert result.benefit_count == 1


def test_outcome_is_derived_and_target_direct_score_must_be_consistent() -> None:
    first = candidate("q1", "a", 1, 0.9, 0.1, PairedOutcome.BENEFIT)
    second = SelectiveCandidate(
        target_query_id="q1",
        gate_trace=trace("b", 2, 0.8, 0.1),
        direct_score=1.0,
        reuse_score=1.0,
        good_threshold=0.5,
    )

    assert first.outcome is PairedOutcome.BENEFIT
    with pytest.raises(ValueError, match="share direct_score"):
        evaluate_operating_point(
            (first, second),
            OperatingPoint(minimum_applicability=0.5, maximum_risk_upper_bound=0.5),
            target_query_ids=("q1",),
        )


@pytest.mark.parametrize(
    "invariant_reason",
    [
        "environment_mismatch",
        "missing_prepared_query_plan",
        "missing_provenance",
        "insufficient_reliability_observations",
    ],
)
def test_non_scanned_gate_invariant_can_never_be_accepted(invariant_reason: str) -> None:
    row = candidate(
        "q1",
        "blocked",
        1,
        1.0,
        0.0,
        PairedOutcome.BENEFIT,
        rejection_reasons=(invariant_reason,),
    )

    result = evaluate_operating_point(
        (row,),
        OperatingPoint(minimum_applicability=0.0, maximum_risk_upper_bound=1.0),
        target_query_ids=("q1",),
    )

    assert row.invariant_eligible is False
    assert row.invariant_rejection_reasons == (invariant_reason,)
    assert result.accepted_count == 0


def test_only_risk_and_applicability_threshold_failures_are_rescanned() -> None:
    row = candidate("q1", "rescanned", 1, 0.8, 0.4, PairedOutcome.BENEFIT)
    row = replace(
        row,
        gate_trace=replace(
            row.gate_trace,
            rejection_reasons=(
                "risk_upper_bound_too_high",
                "applicability_below_threshold",
            ),
            passed=False,
            first_failure="risk_upper_bound_too_high",
            reliability_valid=False,
            applicability_valid=False,
        ),
    )

    result = evaluate_operating_point(
        (row,),
        OperatingPoint(minimum_applicability=0.5, maximum_risk_upper_bound=0.5),
        target_query_ids=("q1",),
    )

    assert row.invariant_eligible is True
    assert row.invariant_rejection_reasons == ()
    assert result.accepted_count == 1
