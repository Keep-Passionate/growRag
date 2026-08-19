from __future__ import annotations

import math
from dataclasses import replace

import pytest

from growrag.evaluation import PairedOutcome
from growrag.experience import (
    EvidenceRole,
    ExperienceActivity,
    ExperienceRecord,
    ExperienceState,
    PortfolioPolicy,
    SourceEvidence,
    TransferObservation,
    select_hot_portfolio,
)
from growrag.models import EnvironmentFingerprint, QueryTransformation


def _observation(
    experience_id: str,
    index: int,
    outcome: PairedOutcome,
) -> TransferObservation:
    scores = {
        PairedOutcome.BENEFIT: (0.0, 1.0),
        PairedOutcome.HARM: (1.0, 0.0),
        PairedOutcome.BOTH_GOOD: (1.0, 1.0),
        PairedOutcome.BOTH_BAD: (0.0, 0.0),
    }
    direct_score, reuse_score = scores[outcome]
    return TransferObservation(
        target_query_id=f"{experience_id}-target-{index}",
        direct_score=direct_score,
        reuse_score=reuse_score,
        good_threshold=0.5,
        dataset_id="toy",
        split="calibration",
        target_group_id=f"{experience_id}-group-{index}",
        evidence_role=EvidenceRole.CALIBRATION,
        metric_protocol_id="answer-em-v1",
        applier_id="template",
        applier_version="1",
    )


def _record(
    experience_id: str,
    outcomes: tuple[PairedOutcome, ...],
    *,
    state: ExperienceState = ExperienceState.ACTIVE,
) -> ExperienceRecord:
    transformation = QueryTransformation(
        experience_id=experience_id,
        source_query_id=f"source-{experience_id}",
        source_query="Who wrote the book?",
        transformed_query="book author",
        atomic_units=("author relation",),
        environment=EnvironmentFingerprint(
            corpus_id="toy",
            corpus_version="1",
            retriever_id="bm25",
            retriever_version="1",
            rewriter_id="template",
            rewriter_version="1",
        ),
        provenance="paired calibration",
    )
    source = SourceEvidence(
        direct_score=0.0,
        reuse_score=1.0,
        dataset_id="toy",
        split="development",
        source_group_id=f"source-group-{experience_id}",
        evidence_role=EvidenceRole.DEVELOPMENT,
        metric_protocol_id="answer-em-v1",
        applier_id="template",
        applier_version="1",
    )
    observations = tuple(
        _observation(experience_id, index, outcome)
        for index, outcome in enumerate(outcomes, start=1)
    )
    return ExperienceRecord(
        transformation=transformation,
        state=state,
        source_evidence=source,
        transfer_observations=observations,
    )


GOOD_OUTCOMES = (
    PairedOutcome.BENEFIT,
    PairedOutcome.BENEFIT,
    PairedOutcome.BOTH_GOOD,
    PairedOutcome.BOTH_GOOD,
    PairedOutcome.BOTH_GOOD,
    PairedOutcome.BOTH_GOOD,
)


def test_top_k_uses_decayed_priority_and_sends_overflow_to_cold() -> None:
    records = tuple(_record(item_id, GOOD_OUTCOMES) for item_id in ("new", "mid", "old"))
    activities = (
        ExperienceActivity("new", last_validated_step=100),
        ExperienceActivity("mid", last_validated_step=90),
        ExperienceActivity("old", last_validated_step=80),
    )

    selection = select_hot_portfolio(
        records,
        activities,
        current_step=100,
        policy=PortfolioPolicy(capacity=2, half_life_steps=10, minimum_priority=0.0),
    )

    assert selection.hot == ("new", "mid")
    assert selection.cold == ("old",)
    assert selection.expired == ()
    assert selection.active_ids == ("mid", "new", "old")


def test_one_half_life_halves_priority_but_not_priority_base() -> None:
    records = (_record("recent", GOOD_OUTCOMES), _record("aged", GOOD_OUTCOMES))
    selection = select_hot_portfolio(
        records,
        (
            ExperienceActivity("recent", last_validated_step=20),
            ExperienceActivity("aged", last_validated_step=10),
        ),
        current_step=20,
        policy=PortfolioPolicy(capacity=2, half_life_steps=10),
    )

    recent = selection.score_for("recent")
    aged = selection.score_for("aged")
    assert recent is not None and aged is not None
    assert aged.priority_base == pytest.approx(recent.priority_base)
    assert aged.decay == pytest.approx(0.5)
    assert aged.priority == pytest.approx(recent.priority * 0.5)


def test_harmful_old_and_missing_activity_fail_closed_without_deletion() -> None:
    harmful = _record(
        "harmful",
        (
            PairedOutcome.BENEFIT,
            PairedOutcome.HARM,
            PairedOutcome.BOTH_GOOD,
            PairedOutcome.BOTH_GOOD,
        ),
    )
    old = _record("old", GOOD_OUTCOMES)
    missing = _record("missing", GOOD_OUTCOMES)
    inactive = _record("candidate", GOOD_OUTCOMES, state=ExperienceState.CANDIDATE)
    records = (harmful, old, missing, inactive)
    before = tuple((record.state, record.transfer_observations) for record in records)

    selection = select_hot_portfolio(
        records,
        (
            ExperienceActivity("harmful", last_validated_step=100),
            ExperienceActivity("old", last_validated_step=0),
            ExperienceActivity("candidate", last_validated_step=100),
        ),
        current_step=100,
        policy=PortfolioPolicy(capacity=3, half_life_steps=10, minimum_priority=0.06),
    )

    assert selection.hot == ()
    assert selection.cold == ("missing",)
    assert set(selection.expired) == {"harmful", "old"}
    harmful_score = selection.score_for("harmful")
    old_score = selection.score_for("old")
    assert harmful_score is not None and harmful_score.priority_base < 0.06
    assert old_score is not None and old_score.priority < 0.06
    assert "candidate" not in selection.active_ids
    assert selection.score_for("missing") is None
    assert tuple((record.state, record.transfer_observations) for record in records) == before


def test_unvalidated_last_use_does_not_refresh_validation_age() -> None:
    record = _record("used", GOOD_OUTCOMES)
    selection = select_hot_portfolio(
        (record,),
        (ExperienceActivity("used", last_validated_step=0, last_used_step=9),),
        current_step=10,
        policy=PortfolioPolicy(capacity=1, half_life_steps=10),
    )

    score = selection.score_for("used")
    assert score is not None
    assert score.age_steps == 10
    assert score.decay == pytest.approx(0.5)
    assert record.reliability.n == len(GOOD_OUTCOMES)


def test_equal_priority_has_stable_experience_id_tie_break() -> None:
    records = (_record("zeta", GOOD_OUTCOMES), _record("alpha", GOOD_OUTCOMES))
    activities = (
        ExperienceActivity("zeta", last_validated_step=5),
        ExperienceActivity("alpha", last_validated_step=5),
    )

    selection = select_hot_portfolio(
        records,
        activities,
        current_step=5,
        policy=PortfolioPolicy(capacity=1, half_life_steps=10),
    )

    assert selection.hot == ("alpha",)
    assert selection.cold == ("zeta",)


@pytest.mark.parametrize(
    "policy",
    [
        PortfolioPolicy(capacity=1, half_life_steps=1, minimum_priority=0.0),
    ],
)
def test_future_activity_and_duplicate_ids_are_rejected(policy: PortfolioPolicy) -> None:
    record = _record("one", GOOD_OUTCOMES)
    duplicate = replace(record)

    with pytest.raises(ValueError, match="future"):
        select_hot_portfolio(
            (record,),
            (ExperienceActivity("one", last_validated_step=11),),
            current_step=10,
            policy=policy,
        )
    with pytest.raises(ValueError, match="duplicate experience_id"):
        select_hot_portfolio(
            (record, duplicate),
            (ExperienceActivity("one", last_validated_step=10),),
            current_step=10,
            policy=policy,
        )
    with pytest.raises(ValueError, match="activities contain duplicate"):
        select_hot_portfolio(
            (record,),
            (
                ExperienceActivity("one", last_validated_step=10),
                ExperienceActivity("one", last_validated_step=9),
            ),
            current_step=10,
            policy=policy,
        )


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        {"capacity": 0, "half_life_steps": 1},
        {"capacity": 1, "half_life_steps": 0},
        {"capacity": 1, "half_life_steps": float("nan")},
        {"capacity": 1, "half_life_steps": 1, "minimum_priority": -0.1},
        {"capacity": 1, "half_life_steps": 1, "minimum_priority": math.inf},
    ],
)
def test_policy_rejects_invalid_or_non_finite_values(policy_kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PortfolioPolicy(**policy_kwargs)  # type: ignore[arg-type]


def test_current_step_and_unknown_activity_are_validated() -> None:
    record = _record("one", GOOD_OUTCOMES)
    policy = PortfolioPolicy(capacity=1, half_life_steps=10)

    with pytest.raises(ValueError, match="current_step"):
        select_hot_portfolio((record,), (), current_step=-1, policy=policy)
    with pytest.raises(ValueError, match="unknown experience"):
        select_hot_portfolio(
            (record,),
            (ExperienceActivity("other", last_validated_step=0),),
            current_step=0,
            policy=policy,
        )
