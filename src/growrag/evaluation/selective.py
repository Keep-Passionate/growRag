"""Offline risk--coverage analysis for the query-side reuse gate.

This module uses paired outcomes only after DIRECT and REUSE have both been
executed on a development/calibration split.  It is not part of the deployable
query-time gate.  Its purpose is to choose transparent operating thresholds
without hiding the safety/coverage trade-off inside one weighted score.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING

from growrag.evaluation.paired import PairedOutcome, evaluate_pair

if TYPE_CHECKING:
    from growrag.selection.gate import CandidateGateTrace


_SCANNED_THRESHOLD_REASONS = frozenset(
    {
        "risk_upper_bound_too_high",
        "applicability_below_threshold",
    }
)


@dataclass(frozen=True, slots=True)
class SelectiveCandidate:
    """One deployable Gate trace paired with its later offline outcome.

    The trace is the single source of truth for all query-time checks.  During
    risk--coverage analysis, only the risk and applicability threshold failures
    are reconsidered; every other Gate rejection remains an invariant veto.
    """

    target_query_id: str
    gate_trace: CandidateGateTrace
    direct_score: float
    reuse_score: float
    good_threshold: float
    invariant_eligible: bool = field(init=False)
    invariant_rejection_reasons: tuple[str, ...] = field(init=False)
    outcome: PairedOutcome = field(init=False)

    def __post_init__(self) -> None:
        from growrag.selection.gate import CandidateGateTrace

        if not isinstance(self.target_query_id, str) or not self.target_query_id.strip():
            raise ValueError("target_query_id must not be empty")
        if not isinstance(self.gate_trace, CandidateGateTrace):
            raise TypeError("gate_trace must be a CandidateGateTrace")
        for value, name in (
            (self.direct_score, "direct_score"),
            (self.reuse_score, "reuse_score"),
            (self.good_threshold, "good_threshold"),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        invariant_reasons = tuple(
            reason
            for reason in self.gate_trace.rejection_reasons
            if reason not in _SCANNED_THRESHOLD_REASONS
        )
        object.__setattr__(self, "invariant_rejection_reasons", invariant_reasons)
        object.__setattr__(self, "invariant_eligible", not invariant_reasons)
        object.__setattr__(
            self,
            "outcome",
            evaluate_pair(
                self.direct_score,
                self.reuse_score,
                good_threshold=self.good_threshold,
            ).outcome,
        )

    @property
    def experience_id(self) -> str:
        return self.gate_trace.experience_id

    @property
    def recall_rank(self) -> int:
        return self.gate_trace.recall_rank

    @property
    def applicability_score(self) -> float | None:
        return self.gate_trace.applicability_score

    @property
    def risk_upper_bound(self) -> float:
        return self.gate_trace.risk_upper_bound

    @property
    def environment_score(self) -> float:
        return self.gate_trace.environment_score

    @property
    def gain(self) -> float:
        return self.reuse_score - self.direct_score


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """Two interpretable knobs controlling how readily memory is reused."""

    minimum_applicability: float
    maximum_risk_upper_bound: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.minimum_applicability, "minimum_applicability"),
            (self.maximum_risk_upper_bound, "maximum_risk_upper_bound"),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    """Observed development-set behavior at one frozen operating point."""

    operating_point: OperatingPoint
    target_count: int
    accepted_count: int
    coverage: float
    benefit_count: int
    harm_count: int
    both_good_count: int
    both_bad_count: int
    accepted_harm_rate: float
    conditional_breakage_rate: float
    repair_rate: float
    mean_gain: float
    selected_pairs: tuple[tuple[str, str], ...]


def evaluate_operating_point(
    candidates: tuple[SelectiveCandidate, ...],
    operating_point: OperatingPoint,
    *,
    target_query_ids: tuple[str, ...],
) -> RiskCoveragePoint:
    """Apply the deployable gate ordering, then measure paired outcomes offline."""

    target_universe = _validate_target_universe(target_query_ids)
    grouped = _validate_and_group(candidates, target_universe=target_universe)
    selected: list[SelectiveCandidate] = []
    for target_query_id in sorted(target_universe):
        eligible = [
            candidate
            for candidate in grouped.get(target_query_id, ())
            if candidate.invariant_eligible
            and candidate.applicability_score is not None
            and candidate.applicability_score >= operating_point.minimum_applicability
            and candidate.risk_upper_bound <= operating_point.maximum_risk_upper_bound
        ]
        if eligible:
            selected.append(min(eligible, key=_selection_key))

    counts = {outcome: 0 for outcome in PairedOutcome}
    for candidate in selected:
        counts[candidate.outcome] += 1

    target_count = len(target_universe)
    accepted_count = len(selected)
    harm_count = counts[PairedOutcome.HARM]
    benefit_count = counts[PairedOutcome.BENEFIT]
    both_good_count = counts[PairedOutcome.BOTH_GOOD]
    both_bad_count = counts[PairedOutcome.BOTH_BAD]
    direct_good_count = harm_count + both_good_count
    direct_bad_count = benefit_count + both_bad_count
    return RiskCoveragePoint(
        operating_point=operating_point,
        target_count=target_count,
        accepted_count=accepted_count,
        coverage=accepted_count / target_count if target_count else 0.0,
        benefit_count=benefit_count,
        harm_count=harm_count,
        both_good_count=both_good_count,
        both_bad_count=both_bad_count,
        accepted_harm_rate=harm_count / accepted_count if accepted_count else 0.0,
        conditional_breakage_rate=(harm_count / direct_good_count if direct_good_count else 0.0),
        repair_rate=benefit_count / direct_bad_count if direct_bad_count else 0.0,
        mean_gain=(
            sum(candidate.gain for candidate in selected) / accepted_count
            if accepted_count
            else 0.0
        ),
        selected_pairs=tuple(
            (candidate.target_query_id, candidate.experience_id) for candidate in selected
        ),
    )


def sweep_operating_points(
    candidates: tuple[SelectiveCandidate, ...],
    operating_points: tuple[OperatingPoint, ...],
    *,
    target_query_ids: tuple[str, ...],
) -> tuple[RiskCoveragePoint, ...]:
    """Evaluate explicit threshold pairs; callers decide which trade-off is acceptable."""

    if not operating_points:
        raise ValueError("at least one operating point is required")
    return tuple(
        evaluate_operating_point(
            candidates,
            operating_point,
            target_query_ids=target_query_ids,
        )
        for operating_point in operating_points
    )


def _validate_and_group(
    candidates: tuple[SelectiveCandidate, ...],
    *,
    target_universe: frozenset[str],
) -> dict[str, tuple[SelectiveCandidate, ...]]:
    grouped_lists: dict[str, list[SelectiveCandidate]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    seen_ranks: set[tuple[str, int]] = set()
    target_protocol: dict[str, tuple[float, float]] = {}
    for candidate in candidates:
        if candidate.target_query_id not in target_universe:
            raise ValueError(
                "candidate target_query_id is absent from target universe: "
                + candidate.target_query_id
            )
        pair = (candidate.target_query_id, candidate.experience_id)
        ranked = (candidate.target_query_id, candidate.recall_rank)
        if pair in seen_pairs:
            raise ValueError(f"duplicate target/experience pair: {pair}")
        if ranked in seen_ranks:
            raise ValueError(f"duplicate recall rank for target: {ranked}")
        seen_pairs.add(pair)
        seen_ranks.add(ranked)
        protocol = (candidate.direct_score, candidate.good_threshold)
        prior_protocol = target_protocol.setdefault(candidate.target_query_id, protocol)
        if prior_protocol != protocol:
            raise ValueError("candidates for one target must share direct_score and good_threshold")
        grouped_lists[candidate.target_query_id].append(candidate)
    return {target: tuple(rows) for target, rows in grouped_lists.items()}


def _selection_key(
    candidate: SelectiveCandidate,
) -> tuple[float, float, float, int, str]:
    applicability = candidate.applicability_score
    assert applicability is not None  # Invariant-filtered candidates have an estimate.
    return (
        -applicability,
        candidate.risk_upper_bound,
        -candidate.environment_score,
        candidate.recall_rank,
        candidate.experience_id,
    )


def _validate_target_universe(target_query_ids: tuple[str, ...]) -> frozenset[str]:
    if not target_query_ids:
        raise ValueError("target_query_ids must contain the complete evaluation universe")
    normalized: list[str] = []
    for target_query_id in target_query_ids:
        if not isinstance(target_query_id, str) or not target_query_id.strip():
            raise ValueError("target_query_ids must contain non-empty strings")
        normalized.append(target_query_id.strip())
    if len(normalized) != len(set(normalized)):
        raise ValueError("target_query_ids must be unique")
    return frozenset(normalized)
