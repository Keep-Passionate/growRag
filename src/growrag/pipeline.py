"""Minimal facade that composes the query-side trusted-reuse components.

This module stops at choosing the query that a downstream retriever should run.
It accepts no gold labels, retrieval results, answers, or target outcome scores.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from growrag.experience import ExperienceLedger, ExperienceRecord, ExperienceState
from growrag.models import EnvironmentFingerprint
from growrag.rewrite import ApplicationError, QueryPlanApplier
from growrag.selection import (
    ApplicabilityScorer,
    CandidateRecallScorer,
    GateCandidate,
    GateDecision,
    LexicalCandidateRecallScorer,
    SignatureApplicabilityScorer,
    TrustedReuseGate,
    rank_experiences,
)


@dataclass(frozen=True, slots=True)
class TrustedReuseResult:
    """A query-only decision plus the candidate-recall audit envelope."""

    decision: GateDecision
    recall_scorer_id: str
    max_candidates: int
    eligible_experience_ids: tuple[str, ...]
    candidate_experience_ids: tuple[str, ...]


class TrustedReuseLayer:
    """Run ACTIVE filtering, recall, application, fit scoring, and the trust gate."""

    def __init__(
        self,
        *,
        applier: QueryPlanApplier,
        recall_scorer: CandidateRecallScorer | None = None,
        applicability_scorer: ApplicabilityScorer | None = None,
        gate: TrustedReuseGate | None = None,
        max_candidates: int = 5,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        self.applier = applier
        self.recall_scorer = recall_scorer or LexicalCandidateRecallScorer()
        self.applicability_scorer = applicability_scorer or SignatureApplicabilityScorer()
        self.gate = gate or TrustedReuseGate()
        self.max_candidates = max_candidates

    def decide(
        self,
        *,
        records: ExperienceLedger | Iterable[ExperienceRecord],
        target_query_id: str,
        target_query: str,
        current_environment: EnvironmentFingerprint,
        current_signatures: frozenset[str],
    ) -> TrustedReuseResult:
        """Return the query-side action without observing downstream outcomes."""

        materialized = _materialize_records(records)
        _validate_unique_experience_ids(materialized)
        eligible = tuple(
            sorted(
                (
                    record
                    for record in materialized
                    if record.state is ExperienceState.ACTIVE
                    and record.transformation.environment.retrieval_compatible_with(
                        current_environment
                    )
                ),
                key=lambda record: record.experience_id,
            )
        )
        records_by_id = {record.experience_id: record for record in eligible}
        ranked = rank_experiences(
            target_query,
            (record.transformation for record in eligible),
            scorer=self.recall_scorer,
            max_candidates=self.max_candidates,
        )

        gate_candidates: list[GateCandidate] = []
        for recalled in ranked:
            experience = recalled.experience
            try:
                prepared = self.applier.prepare(
                    target_query_id=target_query_id,
                    target_query=target_query,
                    experience=experience,
                )
            except ApplicationError:
                prepared = None
            fit = self.applicability_scorer.score(
                target_query,
                experience,
                current_signatures=current_signatures,
            )
            gate_candidates.append(
                GateCandidate(
                    record=records_by_id[experience.experience_id],
                    recall_rank=recalled.rank,
                    prepared=prepared,
                    applicability=fit,
                )
            )

        decision = self.gate.decide(
            target_query_id=target_query_id,
            target_query=target_query,
            current_environment=current_environment,
            candidates=tuple(gate_candidates),
        )
        return TrustedReuseResult(
            decision=decision,
            recall_scorer_id=_component_id(self.recall_scorer),
            max_candidates=self.max_candidates,
            eligible_experience_ids=tuple(record.experience_id for record in eligible),
            candidate_experience_ids=tuple(
                recalled.experience.experience_id for recalled in ranked
            ),
        )


def _materialize_records(
    source: ExperienceLedger | Iterable[ExperienceRecord],
) -> tuple[ExperienceRecord, ...]:
    if isinstance(source, ExperienceLedger):
        return source.records
    return tuple(source)


def _validate_unique_experience_ids(records: tuple[ExperienceRecord, ...]) -> None:
    experience_ids = [record.experience_id for record in records]
    if len(experience_ids) != len(set(experience_ids)):
        raise ValueError("experience_id values must be unique")


def _component_id(component: object) -> str:
    value = getattr(component, "scorer_id", "")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return component.__class__.__name__
