"""Immutable hot-memory portfolio selection for ACTIVE experiences.

Decay changes invocation priority only. It never changes reliability statistics,
lifecycle state, transfer observations, or any other audit record in the ledger.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from growrag.experience.ledger import ExperienceRecord, ExperienceState


@dataclass(frozen=True, slots=True)
class PortfolioPolicy:
    """Capacity and recency policy for the hot-memory view."""

    capacity: int
    half_life_steps: float
    minimum_priority: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise ValueError("capacity must be an integer")
        if self.capacity < 1:
            raise ValueError("capacity must be at least 1")
        if isinstance(self.half_life_steps, bool) or not isinstance(
            self.half_life_steps, (int, float)
        ):
            raise ValueError("half_life_steps must be numeric")
        if not math.isfinite(self.half_life_steps) or self.half_life_steps <= 0:
            raise ValueError("half_life_steps must be finite and positive")
        if isinstance(self.minimum_priority, bool) or not isinstance(
            self.minimum_priority, (int, float)
        ):
            raise ValueError("minimum_priority must be numeric")
        if not math.isfinite(self.minimum_priority) or not 0.0 <= self.minimum_priority <= 1.0:
            raise ValueError("minimum_priority must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class ExperienceActivity:
    """External recency metadata; it is not reliability evidence.

    ``last_used_step`` is retained for audit only. It does not refresh decay;
    only a newly validated outcome may advance ``last_validated_step``.
    """

    experience_id: str
    last_validated_step: int
    last_used_step: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.experience_id, str) or not self.experience_id.strip():
            raise ValueError("experience_id must be a non-empty string")
        object.__setattr__(self, "experience_id", self.experience_id.strip())
        _validate_step(self.last_validated_step, "last_validated_step")
        if self.last_used_step is not None:
            _validate_step(self.last_used_step, "last_used_step")


@dataclass(frozen=True, slots=True)
class PortfolioScore:
    """One immutable heuristic priority, not a probability or confidence bound."""

    priority_base: float
    decay: float
    priority: float
    age_steps: int

    def __post_init__(self) -> None:
        for field_name in ("priority_base", "decay", "priority"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be numeric")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be finite and in [0, 1]")
        _validate_step(self.age_steps, "age_steps")


@dataclass(frozen=True, slots=True)
class PortfolioSelection:
    """A non-destructive partition of ACTIVE experience IDs.

    ``cold`` includes capacity overflow and ACTIVE experiences with no activity
    record. ``expired`` contains scored experiences below ``minimum_priority``.
    Scores are retained for audit; missing-activity experiences have no score.
    """

    hot: tuple[str, ...]
    cold: tuple[str, ...]
    expired: tuple[str, ...]
    scores: tuple[tuple[str, PortfolioScore], ...]

    def __post_init__(self) -> None:
        partitions = (*self.hot, *self.cold, *self.expired)
        if len(partitions) != len(set(partitions)):
            raise ValueError("hot, cold, and expired experience IDs must be disjoint")
        score_ids = [experience_id for experience_id, _ in self.scores]
        if len(score_ids) != len(set(score_ids)):
            raise ValueError("scores must contain unique experience IDs")
        scored = set(score_ids)
        if not set(self.hot).issubset(scored) or not set(self.expired).issubset(scored):
            raise ValueError("hot and expired experience IDs must have scores")

    def score_for(self, experience_id: str) -> PortfolioScore | None:
        """Return one immutable score without creating a mutable lookup table."""

        return next((score for item_id, score in self.scores if item_id == experience_id), None)

    @property
    def active_ids(self) -> tuple[str, ...]:
        """All ACTIVE IDs represented in this view, sorted for stable auditing."""

        return tuple(sorted((*self.hot, *self.cold, *self.expired)))


def select_hot_portfolio(
    records: Iterable[ExperienceRecord],
    activities: Iterable[ExperienceActivity],
    *,
    current_step: int,
    policy: PortfolioPolicy,
) -> PortfolioSelection:
    """Return a deterministic hot/cold view without mutating or deleting records.

    The heuristic priority base is ``benefit_rate * (1 - Wilson
    conditional-breakage upper bound)``. Recency multiplies that heuristic by
    ``0.5 ** (age_steps / half_life_steps)`` only for invocation priority.
    """

    _validate_step(current_step, "current_step")
    if not isinstance(policy, PortfolioPolicy):
        raise TypeError("policy must be a PortfolioPolicy")

    materialized_records = tuple(records)
    materialized_activities = tuple(activities)
    if not all(isinstance(record, ExperienceRecord) for record in materialized_records):
        raise TypeError("records must contain only ExperienceRecord instances")
    if not all(isinstance(activity, ExperienceActivity) for activity in materialized_activities):
        raise TypeError("activities must contain only ExperienceActivity instances")

    record_ids = [record.experience_id for record in materialized_records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("records contain duplicate experience_id values")
    activity_ids = [activity.experience_id for activity in materialized_activities]
    if len(activity_ids) != len(set(activity_ids)):
        raise ValueError("activities contain duplicate experience_id values")
    unknown_activity_ids = set(activity_ids) - set(record_ids)
    if unknown_activity_ids:
        raise ValueError(
            f"activities reference unknown experience IDs: {sorted(unknown_activity_ids)}"
        )

    activity_by_id = {activity.experience_id: activity for activity in materialized_activities}
    for activity in materialized_activities:
        if activity.last_validated_step > current_step:
            raise ValueError(f"last_validated_step is in the future for {activity.experience_id!r}")
        if activity.last_used_step is not None and activity.last_used_step > current_step:
            raise ValueError(f"last_used_step is in the future for {activity.experience_id!r}")

    active_records = tuple(
        record for record in materialized_records if record.state is ExperienceState.ACTIVE
    )
    missing_activity: list[str] = []
    scored: list[tuple[str, PortfolioScore]] = []
    for record in active_records:
        activity = activity_by_id.get(record.experience_id)
        if activity is None:
            missing_activity.append(record.experience_id)
            continue
        scored.append(
            (
                record.experience_id,
                _score_record(
                    record,
                    activity=activity,
                    current_step=current_step,
                    half_life_steps=float(policy.half_life_steps),
                ),
            )
        )

    ranked = sorted(scored, key=lambda item: (-item[1].priority, item[0]))
    eligible = [item for item in ranked if item[1].priority >= policy.minimum_priority]
    expired_pairs = [item for item in ranked if item[1].priority < policy.minimum_priority]
    hot_pairs = eligible[: policy.capacity]
    overflow_pairs = eligible[policy.capacity :]

    return PortfolioSelection(
        hot=tuple(experience_id for experience_id, _ in hot_pairs),
        cold=(
            *(experience_id for experience_id, _ in overflow_pairs),
            *sorted(missing_activity),
        ),
        expired=tuple(experience_id for experience_id, _ in expired_pairs),
        scores=tuple(sorted(scored, key=lambda item: item[0])),
    )


def _score_record(
    record: ExperienceRecord,
    *,
    activity: ExperienceActivity,
    current_step: int,
    half_life_steps: float,
) -> PortfolioScore:
    reliability = record.reliability
    values = (
        reliability.n,
        reliability.benefit_count,
        reliability.wilson_95_conditional_breakage_upper_bound,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"reliability statistics are non-finite for {record.experience_id!r}")
    if reliability.n < 0 or not 0 <= reliability.benefit_count <= reliability.n:
        raise ValueError(f"reliability counts are invalid for {record.experience_id!r}")
    risk_upper = reliability.wilson_95_conditional_breakage_upper_bound
    if not 0.0 <= risk_upper <= 1.0:
        raise ValueError(
            f"conditional breakage upper bound is invalid for {record.experience_id!r}"
        )

    benefit_rate = reliability.benefit_count / reliability.n if reliability.n else 0.0
    priority_base = benefit_rate * (1.0 - risk_upper)
    # Mere exposure must not refresh trust. A use only resets age after its
    # outcome has been validated and ``last_validated_step`` is advanced.
    age_steps = current_step - activity.last_validated_step
    decay = 0.5 ** (age_steps / half_life_steps)
    priority = priority_base * decay
    return PortfolioScore(
        priority_base=priority_base,
        decay=decay,
        priority=priority,
        age_steps=age_steps,
    )


def _validate_step(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
