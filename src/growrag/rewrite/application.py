"""Adapters that apply a source experience to a new target query.

An experience stores the *source* pair ``q_s -> q_s'``.  It must never be
mistaken for the query that will be sent to retrieval for a new target.  This
module makes the second object explicit: an applier produces a target-specific
``PreparedReuseCandidate`` and the trust gate decides whether to use it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from growrag.models import PreparedReuseCandidate, QueryTransformation


class ApplicationError(ValueError):
    """Raised when no valid target-specific query plan can be prepared."""


class QueryPlanApplier(Protocol):
    """Interface for a future template, pattern, or frozen-LLM applier."""

    applier_id: str
    application_version: str

    def prepare(
        self,
        *,
        target_query_id: str,
        target_query: str,
        experience: QueryTransformation,
    ) -> PreparedReuseCandidate: ...


@dataclass(slots=True)
class PrecomputedPlanApplier:
    """Use target variants prepared before the trusted-gate experiment.

    Gate 1 deliberately freezes candidate generation.  QueryGym, ReFormeR, a
    prompt, or a human-built fixture may create the variants, but selection sees
    only this versioned output.  The mapping key is ``(target_id, experience_id)``.
    """

    plans: Mapping[tuple[str, str], str]
    applier_id: str = "precomputed-plan"
    application_version: str = "v1"
    retrieval_query_count: int = 1
    requested_top_k: int = 5
    context_token_budget: int = 4096
    estimated_cost: float = 0.0

    def prepare(
        self,
        *,
        target_query_id: str,
        target_query: str,
        experience: QueryTransformation,
    ) -> PreparedReuseCandidate:
        if target_query_id == experience.source_query_id:
            raise ApplicationError("source query cannot be reused as independent target")
        key = (target_query_id, experience.experience_id)
        try:
            transformed_query = self.plans[key]
        except KeyError as exc:
            raise ApplicationError(f"missing precomputed plan for {key!r}") from exc

        return PreparedReuseCandidate(
            target_query_id=target_query_id,
            target_query=target_query,
            experience_id=experience.experience_id,
            transformed_query=transformed_query,
            generated_by=self.applier_id,
            estimated_cost=self.estimated_cost,
            application_version=self.application_version,
            retrieval_query_count=self.retrieval_query_count,
            requested_top_k=self.requested_top_k,
            context_token_budget=self.context_token_budget,
        )
