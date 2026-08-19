from __future__ import annotations

import pytest

from growrag.evaluation.paired import PairedOutcome
from growrag.experience import (
    EvidenceRole,
    ExperienceLedger,
    ExperienceState,
    ReliabilitySummary,
    SourceEvidence,
    TransferObservation,
)
from growrag.models import EnvironmentFingerprint, QueryTransformation


def transformation() -> QueryTransformation:
    return QueryTransformation(
        experience_id="exp-1",
        source_query_id="source-1",
        source_query="Who wrote the novel?",
        transformed_query="novel title author",
        atomic_units=("replace question phrase with relation keyword",),
        environment=EnvironmentFingerprint(
            corpus_id="toy",
            corpus_version="1",
            retriever_id="bm25",
            retriever_version="1",
            rewriter_id="prompt",
            rewriter_version="1",
        ),
        provenance="paired source evaluation",
    )


def source_evidence(
    direct_score: float = 0.0,
    reuse_score: float = 1.0,
    *,
    role: EvidenceRole = EvidenceRole.DEVELOPMENT,
) -> SourceEvidence:
    return SourceEvidence(
        direct_score=direct_score,
        reuse_score=reuse_score,
        dataset_id="toy",
        split="train",
        source_group_id="source-family",
        evidence_role=role,
        metric_protocol_id="answer-em-v1",
        applier_id="literal-rewrite",
        applier_version="1",
    )


def observation(
    target_id: str,
    direct_score: float,
    reuse_score: float,
    *,
    group_id: str | None = None,
    role: EvidenceRole = EvidenceRole.CALIBRATION,
    metric_protocol_id: str = "answer-em-v1",
    applier_version: str = "1",
) -> TransferObservation:
    return TransferObservation(
        target_query_id=target_id,
        direct_score=direct_score,
        reuse_score=reuse_score,
        good_threshold=0.5,
        dataset_id="toy",
        split="validation",
        target_group_id=group_id or f"group-{target_id}",
        evidence_role=role,
        metric_protocol_id=metric_protocol_id,
        applier_id="literal-rewrite",
        applier_version=applier_version,
    )


def ledger_with_candidate() -> ExperienceLedger:
    ledger = ExperienceLedger()
    ledger.add_candidate(transformation(), source_evidence())
    return ledger


def test_source_without_strict_net_gain_is_not_written() -> None:
    ledger = ExperienceLedger()

    rejected = ledger.add_candidate(transformation(), source_evidence(0.7, 0.7))

    assert rejected is None
    assert len(ledger) == 0


def test_source_success_only_creates_candidate() -> None:
    ledger = ExperienceLedger()

    record = ledger.add_candidate(transformation(), source_evidence())

    assert record is not None
    assert record.state is ExperienceState.CANDIDATE
    assert record.reliability.n == 0
    assert record.reliability.wilson_95_harm_upper_bound == 1.0
    assert record.reliability.wilson_95_conditional_breakage_upper_bound == 1.0


def test_outcome_is_frozen_and_derived_from_scores() -> None:
    derived = observation("target-harm", 1.0, 0.0)

    assert derived.outcome is PairedOutcome.HARM
    with pytest.raises(TypeError, match="outcome"):
        TransferObservation(
            target_query_id="contradiction",
            direct_score=1.0,
            reuse_score=0.0,
            good_threshold=0.5,
            dataset_id="toy",
            split="validation",
            target_group_id="contradiction-family",
            evidence_role=EvidenceRole.CALIBRATION,
            metric_protocol_id="answer-em-v1",
            applier_id="literal-rewrite",
            applier_version="1",
            outcome=PairedOutcome.BENEFIT,
        )


def test_promotion_requires_benefit_and_direct_good_safety_trials() -> None:
    ledger = ledger_with_candidate()

    after_benefit = ledger.record_transfer("exp-1", observation("benefit", 0.0, 1.0))
    after_one_safe = ledger.record_transfer("exp-1", observation("safe-1", 0.8, 0.9))
    after_two_safe = ledger.record_transfer("exp-1", observation("safe-2", 0.7, 0.8))

    assert after_benefit.state is ExperienceState.CANDIDATE
    assert after_one_safe.state is ExperienceState.CANDIDATE
    assert after_two_safe.state is ExperienceState.ACTIVE
    assert after_two_safe.reliability.direct_good_count == 2
    assert after_two_safe.reliability.benefit_count == 1


def test_only_repairs_do_not_prove_that_reuse_is_safe() -> None:
    ledger = ledger_with_candidate()

    for index in range(5):
        record = ledger.record_transfer(
            "exp-1",
            observation(f"benefit-{index}", 0.0, 1.0),
        )

    assert record.state is ExperienceState.CANDIDATE
    assert record.reliability.direct_good_count == 0
    assert record.reliability.conditional_breakage_rate == 0.0
    assert record.reliability.wilson_95_conditional_breakage_upper_bound == 1.0


def test_harm_immediately_quarantines_an_experience() -> None:
    ledger = ledger_with_candidate()

    record = ledger.record_transfer("exp-1", observation("target-harm", 1.0, 0.0))

    assert record.state is ExperienceState.QUARANTINE
    assert record.reliability.harm_count == 1
    assert record.reliability.harm_rate == 1.0
    assert record.reliability.conditional_breakage_rate == 1.0


def test_test_role_cannot_update_candidate_or_transfer_history() -> None:
    empty = ExperienceLedger()
    with pytest.raises(ValueError, match="test evidence"):
        empty.add_candidate(
            transformation(),
            source_evidence(role=EvidenceRole.TEST),
        )
    assert len(empty) == 0

    ledger = ledger_with_candidate()
    with pytest.raises(ValueError, match="test evidence"):
        ledger.record_transfer(
            "exp-1",
            observation("held-out", 0.0, 1.0, role=EvidenceRole.TEST),
        )
    assert ledger.get("exp-1").reliability.n == 0


def test_source_group_and_duplicate_target_or_group_are_rejected() -> None:
    ledger = ledger_with_candidate()

    with pytest.raises(ValueError, match="source group"):
        ledger.record_transfer(
            "exp-1",
            observation("source-neighbour", 0.0, 1.0, group_id="source-family"),
        )

    ledger.record_transfer(
        "exp-1",
        observation("target-1", 0.0, 1.0, group_id="family-1"),
    )
    with pytest.raises(ValueError, match="duplicate target_query"):
        ledger.record_transfer(
            "exp-1",
            observation("target-1", 0.0, 1.0, group_id="family-2"),
        )
    with pytest.raises(ValueError, match="duplicate target_group"):
        ledger.record_transfer(
            "exp-1",
            observation("target-2", 0.0, 1.0, group_id="family-1"),
        )


def test_mixed_metric_or_applier_protocols_are_not_pooled() -> None:
    ledger = ledger_with_candidate()

    with pytest.raises(ValueError, match="metric protocol"):
        ledger.record_transfer(
            "exp-1",
            observation("metric-change", 0.0, 1.0, metric_protocol_id="f1-v2"),
        )
    with pytest.raises(ValueError, match="applier"):
        ledger.record_transfer(
            "exp-1",
            observation("applier-change", 0.0, 1.0, applier_version="2"),
        )


def test_conditional_breakage_and_repair_rates_use_correct_denominators() -> None:
    observations = (
        observation("benefit", 0.0, 1.0),
        observation("harm", 1.0, 0.0),
        observation("both-good", 0.8, 0.9),
        observation("both-bad", 0.2, 0.1),
    )

    summary = ReliabilitySummary.from_observations(observations)

    assert summary.n == 4
    assert summary.benefit_count == 1
    assert summary.harm_count == 1
    assert summary.mean_gain == pytest.approx(0.0)
    assert summary.harm_rate == 0.25
    assert summary.direct_good_count == 2
    assert summary.conditional_breakage_rate == 0.5
    assert summary.direct_bad_count == 2
    assert summary.repair_rate == 0.5


def test_conditional_wilson_bound_is_conservative_for_small_samples() -> None:
    one_safe = ReliabilitySummary.from_observations((observation("safe-1", 0.8, 0.9),))
    ten_safe = ReliabilitySummary.from_observations(
        tuple(observation(f"safe-{index}", 0.8, 0.9) for index in range(10))
    )

    assert one_safe.conditional_breakage_rate == 0.0
    assert one_safe.wilson_95_conditional_breakage_upper_bound > 0.79
    assert (
        ten_safe.wilson_95_conditional_breakage_upper_bound
        < one_safe.wilson_95_conditional_breakage_upper_bound
    )
