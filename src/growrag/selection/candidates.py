"""Candidate recall before trust and applicability filtering."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from growrag.models import QueryTransformation

_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def token_jaccard(left: str, right: str) -> float:
    """Transparent lexical similarity used only for candidate recall."""

    left_tokens = frozenset(token.casefold() for token in _TOKEN_RE.findall(left))
    right_tokens = frozenset(token.casefold() for token in _TOKEN_RE.findall(right))
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


@dataclass(frozen=True, slots=True)
class CandidateRecallEstimate:
    score: float
    scorer_id: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("candidate recall score must be in [0, 1]")


class CandidateRecallScorer(Protocol):
    def score(
        self,
        current_query: str,
        experience: QueryTransformation,
    ) -> CandidateRecallEstimate: ...


@dataclass(frozen=True, slots=True)
class LexicalCandidateRecallScorer:
    scorer_id: str = "source-query-jaccard-v1"

    def score(
        self,
        current_query: str,
        experience: QueryTransformation,
    ) -> CandidateRecallEstimate:
        return CandidateRecallEstimate(
            score=token_jaccard(current_query, experience.source_query),
            scorer_id=self.scorer_id,
        )


@dataclass(frozen=True, slots=True)
class RankedExperience:
    rank: int
    experience: QueryTransformation
    recall: CandidateRecallEstimate


def rank_experiences(
    current_query: str,
    experiences: Iterable[QueryTransformation],
    *,
    scorer: CandidateRecallScorer,
    max_candidates: int,
) -> tuple[RankedExperience, ...]:
    """Return a deterministic top-m list using a dedicated recall signal."""

    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1")

    scored = [(scorer.score(current_query, experience), experience) for experience in experiences]
    scored.sort(key=lambda item: (-item[0].score, item[1].experience_id))
    return tuple(
        RankedExperience(rank=rank, experience=experience, recall=estimate)
        for rank, (estimate, experience) in enumerate(scored[:max_candidates], start=1)
    )
