"""Auditable strict and tiered retrieval-environment compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from growrag.models import EnvironmentFingerprint

_HARD_FIELDS = (
    "corpus_id",
    "corpus_version",
    "index_id",
    "index_version",
    "retriever_id",
)
_SOFT_FIELDS = (
    "retriever_version",
    "analyzer_id",
    "analyzer_version",
)
_ALL_RETRIEVAL_FIELDS = (*_HARD_FIELDS, *_SOFT_FIELDS)
_UNSPECIFIED_VALUES = frozenset({"unset", "unspecified", "unknown"})


class EnvironmentCompatibilityMode(StrEnum):
    """How retrieval fingerprint differences are treated."""

    STRICT = "strict"
    TIERED = "tiered"


@dataclass(frozen=True, slots=True)
class EnvironmentMatchReport:
    """Compatibility decision and an inspectable mismatch explanation."""

    compatible: bool
    score: float
    hard_mismatches: tuple[str, ...]
    soft_mismatches: tuple[str, ...]
    mode: EnvironmentCompatibilityMode

    def __post_init__(self) -> None:
        if not isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("environment compatibility score must be in [0, 1]")
        object.__setattr__(self, "mode", EnvironmentCompatibilityMode(self.mode))


@dataclass(frozen=True, slots=True)
class EnvironmentCompatibilityPolicy:
    """Choose strict equality or a fail-closed tiered compatibility rule.

    In tiered mode, corpus, index, and retriever identity are hard constraints.
    Retriever/analyzer version differences are warnings that reduce the score.
    Missing or unknown values are always hard failures, including soft fields.
    """

    mode: EnvironmentCompatibilityMode = EnvironmentCompatibilityMode.STRICT
    minimum_score: float = 0.80
    soft_mismatch_penalty: float = 0.10

    def __post_init__(self) -> None:
        mode = self.mode
        if isinstance(mode, str):
            mode = EnvironmentCompatibilityMode(mode.strip().casefold())
        object.__setattr__(self, "mode", mode)
        for value, name in (
            (self.minimum_score, "minimum_score"),
            (self.soft_mismatch_penalty, "soft_mismatch_penalty"),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

    def evaluate(
        self,
        reference: EnvironmentFingerprint,
        current: EnvironmentFingerprint,
    ) -> EnvironmentMatchReport:
        """Compare a validated reference environment with the current one."""

        unknown_fields = tuple(
            field_name
            for field_name in _ALL_RETRIEVAL_FIELDS
            if not _specified(getattr(reference, field_name))
            or not _specified(getattr(current, field_name))
        )

        if self.mode is EnvironmentCompatibilityMode.STRICT:
            differing = tuple(
                field_name
                for field_name in _ALL_RETRIEVAL_FIELDS
                if getattr(reference, field_name) != getattr(current, field_name)
            )
            hard_mismatches = _ordered_union(unknown_fields, differing)
            compatible = not hard_mismatches
            return EnvironmentMatchReport(
                compatible=compatible,
                score=1.0 if compatible else 0.0,
                hard_mismatches=hard_mismatches,
                soft_mismatches=(),
                mode=self.mode,
            )

        hard_differences = tuple(
            field_name
            for field_name in _HARD_FIELDS
            if getattr(reference, field_name) != getattr(current, field_name)
        )
        hard_mismatches = _ordered_union(unknown_fields, hard_differences)
        soft_mismatches = tuple(
            field_name
            for field_name in _SOFT_FIELDS
            if field_name not in unknown_fields
            and getattr(reference, field_name) != getattr(current, field_name)
        )
        if hard_mismatches:
            score = 0.0
            compatible = False
        else:
            score = max(0.0, 1.0 - self.soft_mismatch_penalty * len(soft_mismatches))
            compatible = score >= self.minimum_score
        return EnvironmentMatchReport(
            compatible=compatible,
            score=score,
            hard_mismatches=hard_mismatches,
            soft_mismatches=soft_mismatches,
            mode=self.mode,
        )


def match_environments(
    reference: EnvironmentFingerprint,
    current: EnvironmentFingerprint,
    *,
    policy: EnvironmentCompatibilityPolicy | None = None,
) -> EnvironmentMatchReport:
    """Evaluate two retrieval environments with an explicit policy."""

    return (policy or EnvironmentCompatibilityPolicy()).evaluate(reference, current)


def _specified(value: str) -> bool:
    normalized = value.strip().casefold()
    return bool(normalized) and normalized not in _UNSPECIFIED_VALUES


def _ordered_union(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*first, *second)))
