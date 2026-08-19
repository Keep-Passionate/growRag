"""Current-query applicability, kept separate from candidate recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from growrag.models import QueryTransformation


@dataclass(frozen=True, slots=True)
class ApplicabilityEstimate:
    """A pre-retrieval compatibility estimate that never sees target outcomes."""

    score: float
    matched_signatures: tuple[str, ...]
    required_signatures: tuple[str, ...]
    scorer_id: str
    missing_features: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("applicability score must be in [0, 1]")


class ApplicabilityScorer(Protocol):
    """Interface for a current-query versus experience compatibility model."""

    def score(
        self,
        current_query: str,
        experience: QueryTransformation,
        *,
        current_signatures: frozenset[str],
    ) -> ApplicabilityEstimate: ...


@dataclass(frozen=True, slots=True)
class SignatureApplicabilityScorer:
    """Baseline over explicit, typed applicability conditions.

    Candidate recall may use lexical similarity. This scorer deliberately does
    not: it asks whether the current query satisfies the experience card's
    declared structural conditions, such as `comparison` and `two_entity`.
    """

    scorer_id: str = "signature-coverage-v1"

    def score(
        self,
        current_query: str,
        experience: QueryTransformation,
        *,
        current_signatures: frozenset[str],
    ) -> ApplicabilityEstimate:
        del current_query  # Reserved for future calibrated models.
        required = tuple(
            sorted(
                signature.strip().casefold()
                for signature in experience.applicability_signature
                if signature.strip()
            )
        )
        observed = frozenset(
            signature.strip().casefold() for signature in current_signatures if signature.strip()
        )
        if not required or not observed:
            return ApplicabilityEstimate(
                score=0.0,
                matched_signatures=(),
                required_signatures=required,
                scorer_id=self.scorer_id,
                missing_features=True,
            )

        matched = tuple(signature for signature in required if signature in observed)
        return ApplicabilityEstimate(
            score=len(matched) / len(required),
            matched_signatures=matched,
            required_signatures=required,
            scorer_id=self.scorer_id,
        )
