"""Offline candidate retention, deliberately separate from runtime routing.

这里比较的是已完成路径的反馈，不是在线选择器的输入。一个新版本可以比历史
版本好、但仍不如 FRESH；此时允许留下候选记录，不自动授权执行或晋升可信。
This module performs no API calls, disk writes, scalar reward fitting or promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


@dataclass(frozen=True, slots=True)
class MetricObservation:
    """None means unobserved, not zero. Keep cost and quality in separate units."""

    name: str
    value: float | None
    higher_is_better: bool = True
    unit: str = "score"

    def __post_init__(self) -> None:
        _text(self.name, "metric name")
        _text(self.unit, "metric unit")
        if type(self.higher_is_better) is not bool:
            raise TypeError("higher_is_better must be bool")
        if self.value is not None and (
            type(self.value) not in (int, float) or not isfinite(self.value)
        ):
            raise ValueError("metric must be finite numeric data or None")


@dataclass(frozen=True, slots=True)
class OperatorOutcome:
    """Audit-only feedback with declared comparison identity and trace pointer.

    Identity checks are necessary, not proof that two runs actually had identical
    budgets or inputs: the real runner must audit those using its stored traces.
    No field from this record should be injected into a runtime plan or router.
    """

    operator_version: str
    state_id: str
    protocol_id: str
    trace_ref: str
    metrics: tuple[MetricObservation, ...]
    feedback_source: str = "gold"
    completed: bool = True

    def __post_init__(self) -> None:
        for name in ("operator_version", "state_id", "protocol_id", "trace_ref"):
            _text(getattr(self, name), name)
        if self.feedback_source not in {"gold", "human", "proxy"}:
            raise ValueError("feedback_source must be gold, human or proxy")
        if type(self.completed) is not bool:
            raise TypeError("completed must be bool")
        if not isinstance(self.metrics, tuple) or any(
            not isinstance(m, MetricObservation) for m in self.metrics
        ):
            raise TypeError("metrics must be an immutable tuple of observations")
        if len({m.name for m in self.metrics}) != len(self.metrics):
            raise ValueError("duplicate metric name")


@dataclass(frozen=True, slots=True)
class MetricDelta:
    name: str
    raw_delta: float | None
    benefit: float | None
    unit: str


@dataclass(frozen=True, slots=True)
class RetentionDecision:
    """Retain a candidate record, not an active operator or a preferred answer."""

    retain_candidate: bool
    evidence_status: str
    candidate_version: str
    historical_version: str
    state_id: str
    protocol_id: str
    trace_refs: tuple[str, ...]
    historical_deltas: tuple[MetricDelta, ...]
    fresh_deltas: tuple[MetricDelta, ...] | None
    improved_dimensions: tuple[str, ...]
    worsened_dimensions: tuple[str, ...]

    @property
    def trusted(self) -> bool:
        return False

    @property
    def execution_authorized(self) -> bool:
        return False


def _compare(candidate: OperatorOutcome, reference: OperatorOutcome) -> tuple[MetricDelta, ...]:
    if (
        candidate.state_id != reference.state_id
        or candidate.protocol_id != reference.protocol_id
        or candidate.feedback_source != reference.feedback_source
    ):
        raise ValueError("comparison requires the same state, protocol and feedback source")
    if candidate.trace_ref == reference.trace_ref:
        raise ValueError("distinct executed paths need distinct trace pointers")
    left, right = ({m.name: m for m in o.metrics} for o in (candidate, reference))
    deltas = []
    for name in sorted(left.keys() | right.keys()):
        a, b = left.get(name), right.get(name)
        if (
            a is not None
            and b is not None
            and (a.higher_is_better != b.higher_is_better or a.unit != b.unit)
        ):
            raise ValueError("metric direction and units must match")
        raw = None
        if (
            candidate.completed
            and reference.completed
            and a is not None
            and b is not None
            and a.value is not None
            and b.value is not None
        ):
            raw = a.value - b.value
        metric = a if a is not None else b
        benefit = raw if raw is None or metric.higher_is_better else -raw
        deltas.append(MetricDelta(name, raw, benefit, metric.unit))
    return tuple(deltas)


def assess_retention(
    candidate: OperatorOutcome,
    historical: OperatorOutcome,
    *,
    fresh: OperatorOutcome | None = None,
) -> RetentionDecision:
    """Any recorded local gain vs history can justify retention, not activation.

    FRESH deltas are reported for diagnosis but NEVER used as a retention gate.
    Tradeoffs are kept, not averaged away. Failed/unexecuted paths stay unknown.
    Proxy superiority is explicitly labelled estimated. This rule is a testable
    initial policy, not evidence that retaining every tiny gain is optimal.
    """
    if candidate.operator_version == historical.operator_version:
        raise ValueError("compare different operator versions, not a rerun of the same one")
    historical_deltas = _compare(candidate, historical)
    fresh_deltas = _compare(candidate, fresh) if fresh is not None else None
    wins = tuple(d.name for d in historical_deltas if d.benefit is not None and d.benefit > 0)
    losses = tuple(d.name for d in historical_deltas if d.benefit is not None and d.benefit < 0)
    status = "no_comparable_local_gain"
    if wins:
        status = (
            "local_gain_estimated"
            if candidate.feedback_source == "proxy"
            else "local_gain_observed"
        )
    return RetentionDecision(
        bool(wins),
        status,
        candidate.operator_version,
        historical.operator_version,
        candidate.state_id,
        candidate.protocol_id,
        (candidate.trace_ref, historical.trace_ref) + ((fresh.trace_ref,) if fresh else ()),
        historical_deltas,
        fresh_deltas,
        wins,
        losses,
    )
