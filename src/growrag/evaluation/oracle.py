"""Offline candidate-set oracle analysis for historical query reuse.

This module measures hindsight ceilings and risks. It deliberately omits per-target
best-action and best-experience identifiers from serialized reports, because those
would be oracle labels rather than signals available to a deployable router.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from growrag.models import Action

ACTION_COST_ORDER: tuple[Action, ...] = (Action.DIRECT, Action.REUSE, Action.FRESH)
"""Tie-breaking order from lowest to highest online cost."""


def _validate_id(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _validate_score(value: float, field_name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric, got {value!r}") from exc
    if not math.isfinite(score):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {score}")
    return score


def _validate_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class OracleInputRow:
    """One target/candidate measurement under a shared action budget.

    The three budget fields apply equally to DIRECT, this REUSE candidate, and
    FRESH when present. The CSV schema does not support unequal action budgets.
    """

    target_query_id: str
    experience_id: str
    memory_snapshot_id: str
    candidate_rank: int
    direct_score: float
    reuse_score: float
    retrieval_query_count: int
    requested_top_k: int
    context_token_budget: int
    fresh_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_query_id", _validate_id(self.target_query_id, "target_query_id")
        )
        object.__setattr__(self, "experience_id", _validate_id(self.experience_id, "experience_id"))
        object.__setattr__(
            self,
            "memory_snapshot_id",
            _validate_id(self.memory_snapshot_id, "memory_snapshot_id"),
        )
        object.__setattr__(
            self, "candidate_rank", _validate_positive_int(self.candidate_rank, "candidate_rank")
        )
        object.__setattr__(self, "direct_score", _validate_score(self.direct_score, "direct_score"))
        object.__setattr__(self, "reuse_score", _validate_score(self.reuse_score, "reuse_score"))
        object.__setattr__(
            self,
            "retrieval_query_count",
            _validate_positive_int(self.retrieval_query_count, "retrieval_query_count"),
        )
        object.__setattr__(
            self, "requested_top_k", _validate_positive_int(self.requested_top_k, "requested_top_k")
        )
        object.__setattr__(
            self,
            "context_token_budget",
            _validate_positive_int(self.context_token_budget, "context_token_budget"),
        )
        if self.fresh_score is not None:
            object.__setattr__(
                self, "fresh_score", _validate_score(self.fresh_score, "fresh_score")
            )

    @property
    def execution_budget(self) -> tuple[int, int, int]:
        return (
            self.retrieval_query_count,
            self.requested_top_k,
            self.context_token_budget,
        )


@dataclass(frozen=True, slots=True)
class TargetOracleResult:
    """Offline candidate-set ceiling for one target, without routing labels."""

    target_query_id: str
    memory_snapshot_id: str
    candidate_count: int
    candidate_ranks_evaluated: tuple[int, ...]
    retrieval_query_count: int
    requested_top_k: int
    context_token_budget: int
    direct_score: float
    candidate_set_oracle_reuse_score: float
    fresh_score: float | None
    offline_policy_oracle_score: float

    @property
    def reuse_gain_over_direct(self) -> float:
        return self.candidate_set_oracle_reuse_score - self.direct_score

    @property
    def policy_gain_over_direct(self) -> float:
        return self.offline_policy_oracle_score - self.direct_score

    @property
    def policy_gain_over_reuse(self) -> float:
        return self.offline_policy_oracle_score - self.candidate_set_oracle_reuse_score

    def to_dict(self) -> dict[str, object]:
        """Serialize scores and budgets, intentionally excluding hindsight labels."""

        return {
            "target_query_id": self.target_query_id,
            "memory_snapshot_id": self.memory_snapshot_id,
            "candidate_count": self.candidate_count,
            "candidate_ranks_evaluated": list(self.candidate_ranks_evaluated),
            "candidates_evaluated": [
                {"candidate_rank": rank} for rank in self.candidate_ranks_evaluated
            ],
            "retrieval_query_count": self.retrieval_query_count,
            "requested_top_k": self.requested_top_k,
            "context_token_budget": self.context_token_budget,
            "direct_score": self.direct_score,
            "candidate_set_oracle_reuse_score": self.candidate_set_oracle_reuse_score,
            "fresh_score": self.fresh_score,
            "offline_policy_oracle_score": self.offline_policy_oracle_score,
            "reuse_gain_over_direct": self.reuse_gain_over_direct,
            "policy_gain_over_direct": self.policy_gain_over_direct,
            "policy_gain_over_reuse": self.policy_gain_over_reuse,
        }


@dataclass(frozen=True, slots=True)
class OracleAnalysis:
    """Dataset-level offline ceiling, pair risk, and budget statistics."""

    memory_snapshot_id: str
    good_threshold: float
    target_count: int
    candidate_count: int
    pair_harm_count: int
    helpful_target_count: int
    harmful_target_count: int
    candidate_set_reuse_target_harm_count: int
    direct_mean: float
    candidate_set_oracle_reuse_mean: float
    offline_policy_oracle_mean: float
    offline_policy_oracle_action_counts: dict[str, int]
    targets: tuple[TargetOracleResult, ...]

    @property
    def mean_candidate_count(self) -> float:
        return self.candidate_count / self.target_count

    @property
    def pair_harm_rate(self) -> float:
        return self.pair_harm_count / self.candidate_count

    @property
    def helpful_coverage(self) -> float:
        return self.helpful_target_count / self.target_count

    @property
    def harmful_coverage(self) -> float:
        return self.harmful_target_count / self.target_count

    @property
    def candidate_set_reuse_target_harm_rate(self) -> float:
        return self.candidate_set_reuse_target_harm_count / self.target_count

    @property
    def candidate_set_oracle_gain_over_direct(self) -> float:
        return self.candidate_set_oracle_reuse_mean - self.direct_mean

    @property
    def offline_policy_oracle_gain_over_direct(self) -> float:
        return self.offline_policy_oracle_mean - self.direct_mean

    @property
    def offline_policy_oracle_gain_over_candidate_set_oracle(self) -> float:
        return self.offline_policy_oracle_mean - self.candidate_set_oracle_reuse_mean

    @property
    def oracle_reuse_mean(self) -> float:
        """Compatibility alias for early Gate 1 notebooks."""

        return self.candidate_set_oracle_reuse_mean

    @property
    def policy_oracle_mean(self) -> float:
        """Compatibility alias for early Gate 1 notebooks."""

        return self.offline_policy_oracle_mean

    @property
    def policy_action_counts(self) -> dict[str, int]:
        """Compatibility alias; counts are aggregate, not target routing labels."""

        return self.offline_policy_oracle_action_counts

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_type": "offline_candidate_set_oracle_report",
            "offline_diagnostic_only": True,
            "deployable_routing_labels_emitted": False,
            "warning": (
                "Scores use target outcomes observed after execution. This report is a hindsight "
                "ceiling and must not be treated as an online router or deployment label file."
            ),
            "definitions": {
                "candidate_set_oracle": (
                    "best measured REUSE score among only the supplied candidate rows"
                ),
                "offline_policy_oracle": (
                    "hindsight best score among DIRECT, candidate-set REUSE, and FRESH; "
                    "exact ties use the stated lower-cost order"
                ),
                "action_cost_order": [action.value for action in ACTION_COST_ORDER],
                "routing_label_policy": (
                    "all evaluated candidate_rank values are retained, while the per-target "
                    "best action, experience ID, and winning rank are omitted"
                ),
                "execution_budget_semantics": (
                    "each row's retrieval_query_count, requested_top_k, and "
                    "context_token_budget are a common budget for all compared actions"
                ),
                "helpful_target": "candidate-set oracle REUSE score > DIRECT score",
                "harmful_target": "candidate-set oracle REUSE score < DIRECT score",
                "pair_harm": (
                    "DIRECT score >= good_threshold and candidate REUSE score < good_threshold"
                ),
            },
            "memory_snapshot_id": self.memory_snapshot_id,
            "good_threshold": self.good_threshold,
            "target_count": self.target_count,
            "candidate_count": self.candidate_count,
            "mean_candidate_count": self.mean_candidate_count,
            "pair_harm_count": self.pair_harm_count,
            "pair_harm_rate": self.pair_harm_rate,
            "helpful_target_count": self.helpful_target_count,
            "helpful_coverage": self.helpful_coverage,
            "harmful_target_count": self.harmful_target_count,
            "harmful_coverage": self.harmful_coverage,
            "candidate_set_reuse_target_harm_count": (self.candidate_set_reuse_target_harm_count),
            "candidate_set_reuse_target_harm_rate": (self.candidate_set_reuse_target_harm_rate),
            "direct_mean": self.direct_mean,
            "candidate_set_oracle_reuse_mean": self.candidate_set_oracle_reuse_mean,
            "offline_policy_oracle_mean": self.offline_policy_oracle_mean,
            "candidate_set_oracle_gain_over_direct": (self.candidate_set_oracle_gain_over_direct),
            "offline_policy_oracle_gain_over_direct": (self.offline_policy_oracle_gain_over_direct),
            "offline_policy_oracle_gain_over_candidate_set_oracle": (
                self.offline_policy_oracle_gain_over_candidate_set_oracle
            ),
            "offline_policy_oracle_action_counts": self.offline_policy_oracle_action_counts,
            "targets": [target.to_dict() for target in self.targets],
        }


def read_oracle_csv(path: str | Path) -> list[OracleInputRow]:
    """Read and strictly validate an offline candidate-set oracle CSV."""

    csv_path = Path(path)
    required = {
        "target_query_id",
        "experience_id",
        "memory_snapshot_id",
        "candidate_rank",
        "direct_score",
        "reuse_score",
        "retrieval_query_count",
        "requested_top_k",
        "context_token_budget",
    }
    allowed = required | {"fresh_score"}

    try:
        handle = csv_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"cannot open oracle CSV {csv_path}: {exc}") from exc

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("oracle CSV must contain a header row")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("oracle CSV must not contain duplicate column names")
        fieldnames = set(reader.fieldnames)
        missing = required - fieldnames
        unexpected = fieldnames - allowed
        if missing:
            raise ValueError(f"oracle CSV is missing columns: {sorted(missing)}")
        if unexpected:
            raise ValueError(f"oracle CSV has unexpected columns: {sorted(unexpected)}")

        rows: list[OracleInputRow] = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                fresh_raw = raw.get("fresh_score")
                fresh_score = (
                    None if fresh_raw is None or not fresh_raw.strip() else float(fresh_raw)
                )
                rows.append(
                    OracleInputRow(
                        target_query_id=raw["target_query_id"],
                        experience_id=raw["experience_id"],
                        memory_snapshot_id=raw["memory_snapshot_id"],
                        candidate_rank=int(raw["candidate_rank"]),
                        direct_score=float(raw["direct_score"]),
                        reuse_score=float(raw["reuse_score"]),
                        retrieval_query_count=int(raw["retrieval_query_count"]),
                        requested_top_k=int(raw["requested_top_k"]),
                        context_token_budget=int(raw["context_token_budget"]),
                        fresh_score=fresh_score,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid oracle CSV row at line {line_number}: {exc}") from exc

    if not rows:
        raise ValueError("oracle CSV must contain at least one data row")
    return rows


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return sum(materialized) / len(materialized)


def analyze_oracle(
    rows: Iterable[OracleInputRow],
    *,
    good_threshold: float = 0.5,
) -> OracleAnalysis:
    """Compute an offline oracle over exactly the supplied candidate set.

    Filtering the input rows changes the candidate-set oracle condition. Every
    target's compared actions must share one memory snapshot and execution budget.
    """

    threshold = _validate_score(good_threshold, "good_threshold")
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("analyze_oracle requires at least one row")
    if not all(isinstance(row, OracleInputRow) for row in materialized):
        raise TypeError("all rows must be OracleInputRow instances")

    snapshot_ids = {row.memory_snapshot_id for row in materialized}
    if len(snapshot_ids) != 1:
        raise ValueError("one oracle report must use exactly one memory_snapshot_id")
    memory_snapshot_id = next(iter(snapshot_ids))

    seen_pairs: set[tuple[str, str]] = set()
    grouped: defaultdict[str, list[OracleInputRow]] = defaultdict(list)
    pair_harm_count = 0
    for row in materialized:
        pair = (row.target_query_id, row.experience_id)
        if pair in seen_pairs:
            raise ValueError(
                "duplicate target-query/experience pair: "
                f"target_query_id={row.target_query_id!r}, experience_id={row.experience_id!r}"
            )
        seen_pairs.add(pair)
        grouped[row.target_query_id].append(row)
        if row.direct_score >= threshold and row.reuse_score < threshold:
            pair_harm_count += 1

    target_results: list[TargetOracleResult] = []
    offline_actions: list[Action] = []
    helpful_target_count = 0
    harmful_target_count = 0
    candidate_set_reuse_target_harm_count = 0

    cost_rank = {action: rank for rank, action in enumerate(ACTION_COST_ORDER)}
    for target_query_id in sorted(grouped):
        candidates = grouped[target_query_id]

        direct_scores = {row.direct_score for row in candidates}
        if len(direct_scores) != 1:
            raise ValueError(f"DIRECT score is inconsistent for target {target_query_id!r}")
        direct_score = next(iter(direct_scores))

        fresh_scores = {row.fresh_score for row in candidates}
        if len(fresh_scores) != 1:
            raise ValueError(f"FRESH score/presence is inconsistent for target {target_query_id!r}")
        fresh_score = next(iter(fresh_scores))

        budgets = {row.execution_budget for row in candidates}
        if len(budgets) != 1:
            raise ValueError(
                f"shared DIRECT/REUSE/FRESH execution budget is inconsistent for target "
                f"{target_query_id!r}"
            )
        retrieval_query_count, requested_top_k, context_token_budget = next(iter(budgets))

        candidate_ranks = tuple(sorted(row.candidate_rank for row in candidates))
        if len(candidate_ranks) != len(set(candidate_ranks)):
            raise ValueError(f"candidate_rank is duplicated for target {target_query_id!r}")

        candidate_set_oracle_reuse_score = max(row.reuse_score for row in candidates)
        if candidate_set_oracle_reuse_score > direct_score:
            helpful_target_count += 1
        elif candidate_set_oracle_reuse_score < direct_score:
            harmful_target_count += 1
        if direct_score >= threshold and candidate_set_oracle_reuse_score < threshold:
            candidate_set_reuse_target_harm_count += 1

        action_options = [
            (Action.DIRECT, direct_score),
            (Action.REUSE, candidate_set_oracle_reuse_score),
        ]
        if fresh_score is not None:
            action_options.append((Action.FRESH, fresh_score))
        offline_action, offline_policy_oracle_score = max(
            action_options,
            key=lambda option: (option[1], -cost_rank[option[0]]),
        )
        offline_actions.append(offline_action)

        target_results.append(
            TargetOracleResult(
                target_query_id=target_query_id,
                memory_snapshot_id=memory_snapshot_id,
                candidate_count=len(candidates),
                candidate_ranks_evaluated=candidate_ranks,
                retrieval_query_count=retrieval_query_count,
                requested_top_k=requested_top_k,
                context_token_budget=context_token_budget,
                direct_score=direct_score,
                candidate_set_oracle_reuse_score=candidate_set_oracle_reuse_score,
                fresh_score=fresh_score,
                offline_policy_oracle_score=offline_policy_oracle_score,
            )
        )

    results = tuple(target_results)
    action_counter = Counter(action.value for action in offline_actions)
    action_counts = {action.value: action_counter[action.value] for action in ACTION_COST_ORDER}
    return OracleAnalysis(
        memory_snapshot_id=memory_snapshot_id,
        good_threshold=threshold,
        target_count=len(results),
        candidate_count=len(materialized),
        pair_harm_count=pair_harm_count,
        helpful_target_count=helpful_target_count,
        harmful_target_count=harmful_target_count,
        candidate_set_reuse_target_harm_count=candidate_set_reuse_target_harm_count,
        direct_mean=_mean(result.direct_score for result in results),
        candidate_set_oracle_reuse_mean=_mean(
            result.candidate_set_oracle_reuse_score for result in results
        ),
        offline_policy_oracle_mean=_mean(result.offline_policy_oracle_score for result in results),
        offline_policy_oracle_action_counts=action_counts,
        targets=results,
    )
