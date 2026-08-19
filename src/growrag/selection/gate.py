"""A query-only, deterministic trust gate for historical transformation reuse.

The gate deliberately accepts no retrieval results, answers, gold labels, or
target outcome scores.  It only decides whether a prepared target-query variant
is safe enough to execute.  Outcome evaluation belongs to the offline evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

from growrag.experience.ledger import ExperienceRecord, ExperienceState
from growrag.models import EnvironmentFingerprint, PreparedReuseCandidate
from growrag.selection.applicability import ApplicabilityEstimate
from growrag.selection.environment import (
    EnvironmentCompatibilityPolicy,
    EnvironmentMatchReport,
)


class DeploymentAction(StrEnum):
    """Actions the deployable query-side gate is allowed to return."""

    DIRECT = "direct"
    REUSE = "reuse"


class DecisionReason(StrEnum):
    """Top-level, machine-readable reason for the final action."""

    REUSE_SELECTED = "reuse_selected"
    NO_CANDIDATES = "no_candidates"
    NO_ELIGIBLE_EXPERIENCE = "no_eligible_experience"
    INVALID_QUERY_PLAN_FALLBACK = "invalid_query_plan_fallback"


@dataclass(frozen=True, slots=True)
class QueryBudget:
    """Hard upper bounds for one prepared REUSE query plan.

    Positive values are required so that ``0`` cannot silently mean either
    "unknown" or "unlimited".  The executor should use the same limits for the
    DIRECT arm; this gate only verifies the proposed REUSE plan.
    """

    max_retrieval_queries: int = 1
    max_top_k: int = 10
    max_context_tokens: int = 4096
    max_query_characters: int = 2048
    max_application_cost: float | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.max_retrieval_queries, "max_retrieval_queries"),
            (self.max_top_k, "max_top_k"),
            (self.max_context_tokens, "max_context_tokens"),
            (self.max_query_characters, "max_query_characters"),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_application_cost is not None and (
            not isfinite(self.max_application_cost) or self.max_application_cost < 0
        ):
            raise ValueError("max_application_cost must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """Frozen thresholds for historical reliability and current applicability."""

    minimum_observations: int = 2
    minimum_benefits: int = 1
    minimum_mean_gain: float = 0.0
    maximum_risk_upper_bound: float = 0.70
    minimum_applicability: float = 0.50

    def __post_init__(self) -> None:
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be at least 1")
        if self.minimum_benefits < 0:
            raise ValueError("minimum_benefits must be non-negative")
        if not isfinite(self.minimum_mean_gain):
            raise ValueError("minimum_mean_gain must be finite")
        for value, name in (
            (self.maximum_risk_upper_bound, "maximum_risk_upper_bound"),
            (self.minimum_applicability, "minimum_applicability"),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class GateCandidate:
    """All query-time evidence the gate may inspect for one recalled experience."""

    record: ExperienceRecord
    recall_rank: int
    prepared: PreparedReuseCandidate | None
    applicability: ApplicabilityEstimate | None

    def __post_init__(self) -> None:
        if self.recall_rank < 1:
            raise ValueError("recall_rank must be at least 1")

    @property
    def experience_id(self) -> str:
        return self.record.experience_id


@dataclass(frozen=True, slots=True)
class CandidateGateTrace:
    """Auditable query-only checks for one recalled experience.

    This schema intentionally contains no target score, answer, retrieval result,
    paired outcome, or gold field.
    """

    experience_id: str
    recall_rank: int
    state: str
    environment_compatible: bool
    environment_score: float
    environment_mode: str
    environment_hard_mismatches: tuple[str, ...]
    environment_soft_mismatches: tuple[str, ...]
    provenance_valid: bool
    reliability_observations: int
    reliability_benefits: int
    reliability_mean_gain: float
    risk_upper_bound: float
    risk_measure: str
    reliability_valid: bool
    query_plan_valid: bool
    applicability_score: float | None
    applicability_scorer_id: str | None
    applicability_matched_signatures: tuple[str, ...]
    applicability_required_signatures: tuple[str, ...]
    applicability_matched_contraindications: tuple[str, ...]
    applicability_valid: bool
    passed: bool
    first_failure: str | None
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Final action plus the intended action before a safe query-plan fallback."""

    target_query_id: str
    intended_action: DeploymentAction
    executed_action: DeploymentAction
    intended_experience_id: str | None
    executed_experience_id: str | None
    executed_query: str
    reason: DecisionReason
    traces: tuple[CandidateGateTrace, ...]

    @property
    def action(self) -> DeploymentAction:
        """Convenient alias for the action that should actually be executed."""

        return self.executed_action


@dataclass(frozen=True, slots=True)
class _EvaluatedCandidate:
    candidate: GateCandidate
    trace: CandidateGateTrace
    semantic_intent_valid: bool


class TrustedReuseGate:
    """Apply the frozen ACTIVE→environment→trust→plan→fit contract."""

    def __init__(
        self,
        *,
        policy: GatePolicy | None = None,
        budget: QueryBudget | None = None,
        environment_policy: EnvironmentCompatibilityPolicy | None = None,
    ) -> None:
        self.policy = policy or GatePolicy()
        self.budget = budget or QueryBudget()
        self.environment_policy = environment_policy or EnvironmentCompatibilityPolicy()

    def match_environment(
        self,
        reference: EnvironmentFingerprint,
        current: EnvironmentFingerprint,
    ) -> EnvironmentMatchReport:
        """Return the exact environment report used by recall filtering and gating."""

        return self.environment_policy.evaluate(reference, current)

    def decide(
        self,
        *,
        target_query_id: str,
        target_query: str,
        current_environment: EnvironmentFingerprint,
        candidates: tuple[GateCandidate, ...],
    ) -> GateDecision:
        """Choose DIRECT or one REUSE candidate using query-time information only."""

        normalized_target_id = target_query_id.strip()
        normalized_target_query = target_query.strip()
        if not normalized_target_id:
            raise ValueError("target_query_id must not be empty")
        if not normalized_target_query:
            raise ValueError("target_query must not be empty")
        self._validate_unique_candidates(candidates)

        if not candidates:
            return GateDecision(
                target_query_id=normalized_target_id,
                intended_action=DeploymentAction.DIRECT,
                executed_action=DeploymentAction.DIRECT,
                intended_experience_id=None,
                executed_experience_id=None,
                executed_query=normalized_target_query,
                reason=DecisionReason.NO_CANDIDATES,
                traces=(),
            )

        evaluated = tuple(
            self._evaluate_candidate(
                candidate,
                target_query_id=normalized_target_id,
                target_query=normalized_target_query,
                current_environment=current_environment,
            )
            for candidate in candidates
        )
        traces = tuple(item.trace for item in evaluated)

        fully_valid = [item for item in evaluated if item.trace.passed]
        if fully_valid:
            selected = min(fully_valid, key=self._selection_key)
            prepared = selected.candidate.prepared
            assert prepared is not None  # Proven by query_plan_valid.
            return GateDecision(
                target_query_id=normalized_target_id,
                intended_action=DeploymentAction.REUSE,
                executed_action=DeploymentAction.REUSE,
                intended_experience_id=selected.candidate.experience_id,
                executed_experience_id=selected.candidate.experience_id,
                executed_query=prepared.transformed_query.strip(),
                reason=DecisionReason.REUSE_SELECTED,
                traces=traces,
            )

        semantic_candidates = [item for item in evaluated if item.semantic_intent_valid]
        if semantic_candidates:
            intended = min(semantic_candidates, key=self._selection_key)
            return GateDecision(
                target_query_id=normalized_target_id,
                intended_action=DeploymentAction.REUSE,
                executed_action=DeploymentAction.DIRECT,
                intended_experience_id=intended.candidate.experience_id,
                executed_experience_id=None,
                executed_query=normalized_target_query,
                reason=DecisionReason.INVALID_QUERY_PLAN_FALLBACK,
                traces=traces,
            )

        return GateDecision(
            target_query_id=normalized_target_id,
            intended_action=DeploymentAction.DIRECT,
            executed_action=DeploymentAction.DIRECT,
            intended_experience_id=None,
            executed_experience_id=None,
            executed_query=normalized_target_query,
            reason=DecisionReason.NO_ELIGIBLE_EXPERIENCE,
            traces=traces,
        )

    @staticmethod
    def _validate_unique_candidates(candidates: tuple[GateCandidate, ...]) -> None:
        experience_ids = [candidate.experience_id for candidate in candidates]
        ranks = [candidate.recall_rank for candidate in candidates]
        if len(experience_ids) != len(set(experience_ids)):
            raise ValueError("candidate experience_id values must be unique")
        if len(ranks) != len(set(ranks)):
            raise ValueError("candidate recall_rank values must be unique")

    def _evaluate_candidate(
        self,
        candidate: GateCandidate,
        *,
        target_query_id: str,
        target_query: str,
        current_environment: EnvironmentFingerprint,
    ) -> _EvaluatedCandidate:
        record = candidate.record
        transformation = record.transformation
        reliability = record.reliability
        risk_upper, risk_measure = _risk_upper_bound(reliability)

        state_valid = record.state is ExperienceState.ACTIVE
        environment_report = self.match_environment(
            transformation.environment,
            current_environment,
        )
        environment_valid = environment_report.compatible
        provenance_valid = _is_specified(transformation.provenance)

        reliability_reasons: list[str] = []
        if reliability.n < self.policy.minimum_observations:
            reliability_reasons.append("insufficient_reliability_observations")
        if reliability.benefit_count < self.policy.minimum_benefits:
            reliability_reasons.append("insufficient_reliability_benefits")
        if not isfinite(reliability.mean_gain) or (
            reliability.mean_gain <= self.policy.minimum_mean_gain
        ):
            reliability_reasons.append("reliability_mean_gain_too_low")
        if not isfinite(risk_upper) or risk_upper > self.policy.maximum_risk_upper_bound:
            reliability_reasons.append("risk_upper_bound_too_high")
        reliability_valid = not reliability_reasons

        plan_reasons = self._query_plan_reasons(
            candidate.prepared,
            experience_id=candidate.experience_id,
            target_query_id=target_query_id,
            target_query=target_query,
            expected_applier_id=getattr(record.source_evidence, "applier_id", ""),
            expected_applier_version=getattr(record.source_evidence, "applier_version", ""),
        )
        query_plan_valid = not plan_reasons

        applicability = candidate.applicability
        applicability_reasons: list[str] = []
        if applicability is None:
            applicability_reasons.append("missing_applicability_estimate")
            applicability_score = None
            applicability_scorer_id = None
            matched_signatures: tuple[str, ...] = ()
            required_signatures: tuple[str, ...] = ()
            matched_contraindications: tuple[str, ...] = ()
        else:
            applicability_score = applicability.score
            applicability_scorer_id = applicability.scorer_id
            matched_signatures = applicability.matched_signatures
            required_signatures = applicability.required_signatures
            matched_contraindications = applicability.matched_contraindications
            if not _is_specified(applicability.scorer_id):
                applicability_reasons.append("missing_applicability_scorer")
            if applicability.matched_contraindications:
                applicability_reasons.append("applicability_contraindication_matched")
            if applicability.missing_features:
                applicability_reasons.append("missing_applicability_features")
            if (
                not isfinite(applicability.score)
                or applicability.score < self.policy.minimum_applicability
            ):
                applicability_reasons.append("applicability_below_threshold")
        applicability_valid = not applicability_reasons

        stage_reasons: list[str] = []
        if not state_valid:
            stage_reasons.append("experience_not_active")
        if not environment_valid:
            stage_reasons.append("environment_mismatch")
        if not provenance_valid:
            stage_reasons.append("missing_provenance")
        stage_reasons.extend(reliability_reasons)
        stage_reasons.extend(plan_reasons)
        stage_reasons.extend(applicability_reasons)

        passed = (
            state_valid
            and environment_valid
            and provenance_valid
            and reliability_valid
            and query_plan_valid
            and applicability_valid
        )
        semantic_intent_valid = (
            state_valid
            and environment_valid
            and provenance_valid
            and reliability_valid
            and applicability_valid
        )
        trace = CandidateGateTrace(
            experience_id=candidate.experience_id,
            recall_rank=candidate.recall_rank,
            state=record.state.value,
            environment_compatible=environment_valid,
            environment_score=environment_report.score,
            environment_mode=environment_report.mode.value,
            environment_hard_mismatches=environment_report.hard_mismatches,
            environment_soft_mismatches=environment_report.soft_mismatches,
            provenance_valid=provenance_valid,
            reliability_observations=reliability.n,
            reliability_benefits=reliability.benefit_count,
            reliability_mean_gain=reliability.mean_gain,
            risk_upper_bound=risk_upper,
            risk_measure=risk_measure,
            reliability_valid=reliability_valid,
            query_plan_valid=query_plan_valid,
            applicability_score=applicability_score,
            applicability_scorer_id=applicability_scorer_id,
            applicability_matched_signatures=matched_signatures,
            applicability_required_signatures=required_signatures,
            applicability_matched_contraindications=matched_contraindications,
            applicability_valid=applicability_valid,
            passed=passed,
            first_failure=stage_reasons[0] if stage_reasons else None,
            rejection_reasons=tuple(stage_reasons),
        )
        return _EvaluatedCandidate(
            candidate=candidate,
            trace=trace,
            semantic_intent_valid=semantic_intent_valid,
        )

    def _query_plan_reasons(
        self,
        prepared: PreparedReuseCandidate | None,
        *,
        experience_id: str,
        target_query_id: str,
        target_query: str,
        expected_applier_id: str,
        expected_applier_version: str,
    ) -> list[str]:
        if prepared is None:
            return ["missing_prepared_query_plan"]

        reasons: list[str] = []
        if prepared.experience_id != experience_id:
            reasons.append("prepared_experience_mismatch")
        if prepared.target_query_id.strip() != target_query_id:
            reasons.append("prepared_target_id_mismatch")
        if prepared.target_query.strip() != target_query:
            reasons.append("prepared_target_query_mismatch")
        transformed = prepared.transformed_query.strip()
        if not transformed:
            reasons.append("empty_transformed_query")
        elif transformed.casefold() == target_query.casefold():
            reasons.append("unchanged_transformed_query")
        if len(transformed) > self.budget.max_query_characters:
            reasons.append("transformed_query_too_long")
        if not _is_specified(prepared.generated_by) or not _is_specified(
            prepared.application_version
        ):
            reasons.append("missing_application_provenance")
        if prepared.generated_by.strip() != expected_applier_id.strip():
            reasons.append("application_applier_mismatch")
        if prepared.application_version.strip() != expected_applier_version.strip():
            reasons.append("application_version_mismatch")
        if not 1 <= prepared.retrieval_query_count <= self.budget.max_retrieval_queries:
            reasons.append("retrieval_query_budget_exceeded")
        if not 1 <= prepared.requested_top_k <= self.budget.max_top_k:
            reasons.append("top_k_budget_invalid")
        if not 1 <= prepared.context_token_budget <= self.budget.max_context_tokens:
            reasons.append("context_token_budget_invalid")
        if not isfinite(prepared.estimated_cost) or prepared.estimated_cost < 0:
            reasons.append("application_cost_invalid")
        elif (
            self.budget.max_application_cost is not None
            and prepared.estimated_cost > self.budget.max_application_cost
        ):
            reasons.append("application_cost_budget_exceeded")
        return reasons

    @staticmethod
    def _selection_key(
        item: _EvaluatedCandidate,
    ) -> tuple[float, float, float, int, str]:
        applicability = item.trace.applicability_score
        assert applicability is not None  # Only valid candidates are ranked.
        return (
            -applicability,
            item.trace.risk_upper_bound,
            -item.trace.environment_score,
            item.trace.recall_rank,
            item.trace.experience_id,
        )


def _risk_upper_bound(reliability: Any) -> tuple[float, str]:
    """Prefer a future breakage bound while remaining compatible with v1 ledger."""

    for field_name in (
        "wilson_95_conditional_breakage_upper_bound",
        "wilson_95_breakage_upper_bound",
        "wilson_95_harm_upper_bound",
    ):
        value = getattr(reliability, field_name, None)
        if value is not None:
            return float(value), field_name
    return 1.0, "missing_risk_upper_bound"


def _is_specified(value: str) -> bool:
    normalized = value.strip().casefold()
    return bool(normalized) and normalized not in {"unset", "unspecified", "unknown"}
