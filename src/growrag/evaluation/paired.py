"""Counterfactual-style paired labels for DIRECT versus REUSE."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class PairedOutcome(StrEnum):
    """Four observable cells when both actions are run offline."""

    BENEFIT = "direct_bad_reuse_good"
    HARM = "direct_good_reuse_bad"
    BOTH_GOOD = "direct_good_reuse_good"
    BOTH_BAD = "direct_bad_reuse_bad"


@dataclass(frozen=True, slots=True)
class PairedEvaluation:
    direct_score: float
    reuse_score: float
    difference: float
    outcome: PairedOutcome


def evaluate_pair(
    direct_score: float,
    reuse_score: float,
    *,
    good_threshold: float,
) -> PairedEvaluation:
    """Classify one target query using an explicitly chosen success threshold."""

    direct_good = direct_score >= good_threshold
    reuse_good = reuse_score >= good_threshold

    if direct_good and not reuse_good:
        outcome = PairedOutcome.HARM
    elif not direct_good and reuse_good:
        outcome = PairedOutcome.BENEFIT
    elif direct_good and reuse_good:
        outcome = PairedOutcome.BOTH_GOOD
    else:
        outcome = PairedOutcome.BOTH_BAD

    return PairedEvaluation(
        direct_score=direct_score,
        reuse_score=reuse_score,
        difference=reuse_score - direct_score,
        outcome=outcome,
    )

