"""Strict TOML configuration loading for the Gate 1 pilot.

Template configurations may contain the explicit placeholder ``UNSET``. Callers
must request runtime validation before executing an experiment; unresolved
environment or application identities then fail closed.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from growrag.experience.ledger import LifecyclePolicy
from growrag.models import EnvironmentFingerprint
from growrag.selection.environment import EnvironmentCompatibilityPolicy
from growrag.selection.gate import GatePolicy, QueryBudget


class ConfigError(ValueError):
    """Raised when a pilot configuration is malformed, ambiguous, or unsafe."""


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    name: str
    seed: int


@dataclass(frozen=True, slots=True)
class ActionSettings:
    direct: bool
    reuse: bool
    fresh: bool


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    good_threshold: float
    max_history_candidates: int
    paired_harm_is_primary: bool


@dataclass(frozen=True, slots=True)
class WriteGateSettings:
    minimum_source_gain: float


@dataclass(frozen=True, slots=True)
class CandidateRecallSettings:
    scorer: str
    maximum_candidates: int


@dataclass(frozen=True, slots=True)
class ApplicabilitySettings:
    scorer: str
    minimum_score: float


@dataclass(frozen=True, slots=True)
class ApplicationSettings:
    applier_id: str
    application_version: str


@dataclass(frozen=True, slots=True)
class PilotConfig:
    """Fully typed Gate 1 configuration assembled from one TOML document."""

    experiment: ExperimentSettings
    actions: ActionSettings
    evaluation: EvaluationSettings
    environment: EnvironmentFingerprint
    environment_compatibility: EnvironmentCompatibilityPolicy
    write_gate: WriteGateSettings
    lifecycle_policy: LifecyclePolicy
    gate_policy: GatePolicy
    candidate_recall: CandidateRecallSettings
    applicability: ApplicabilitySettings
    application: ApplicationSettings
    query_budget: QueryBudget

    def require_runtime_ready(self) -> PilotConfig:
        """Fail closed if any required runtime identity remains unresolved."""

        unresolved: list[str] = []
        for field in fields(EnvironmentFingerprint):
            value = getattr(self.environment, field.name)
            if _is_placeholder(value):
                unresolved.append(f"[environment].{field.name}")
        for field_name in ("applier_id", "application_version"):
            value = getattr(self.application, field_name)
            if _is_placeholder(value):
                unresolved.append(f"[application].{field_name}")
        if unresolved:
            joined = ", ".join(unresolved)
            raise ConfigError(
                "runtime configuration is not ready; replace placeholder values for: " + joined
            )
        return self


_TOP_LEVEL_SECTIONS = {
    "experiment",
    "actions",
    "evaluation",
    "environment",
    "environment_compatibility",
    "write_gate",
    "reliability",
    "candidate_recall",
    "applicability",
    "application",
    "budget",
}

_SECTION_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "experiment": (frozenset({"name", "seed"}), frozenset()),
    "actions": (frozenset({"direct", "reuse", "fresh"}), frozenset()),
    "evaluation": (
        frozenset({"good_threshold", "max_history_candidates", "paired_harm_is_primary"}),
        frozenset(),
    ),
    "environment": (
        frozenset(
            {
                "corpus_id",
                "corpus_version",
                "retriever_id",
                "retriever_version",
                "index_id",
                "index_version",
                "analyzer_id",
                "analyzer_version",
                "rewriter_id",
                "rewriter_version",
            }
        ),
        frozenset(),
    ),
    "environment_compatibility": (
        frozenset({"mode", "minimum_score", "soft_mismatch_penalty"}),
        frozenset(),
    ),
    "write_gate": (frozenset({"minimum_source_gain"}), frozenset()),
    "reliability": (
        frozenset(
            {
                "minimum_transfer_observations",
                "minimum_direct_good_observations",
                "minimum_benefits",
                "minimum_mean_gain",
                "maximum_conditional_breakage_upper_bound",
                "quarantine_at_harm_count",
            }
        ),
        frozenset(),
    ),
    "candidate_recall": (frozenset({"scorer", "maximum_candidates"}), frozenset()),
    "applicability": (frozenset({"scorer", "minimum_score"}), frozenset()),
    "application": (frozenset({"applier_id", "application_version"}), frozenset()),
    "budget": (
        frozenset(
            {
                "maximum_retrieval_queries",
                "top_k",
                "context_tokens",
                "maximum_query_characters",
            }
        ),
        frozenset({"maximum_application_cost"}),
    ),
}

_PLACEHOLDERS = {"unset", "unspecified", "unknown"}


def load_pilot_config(
    path: str | Path,
    *,
    for_runtime: bool = False,
) -> PilotConfig:
    """Load a strict pilot TOML, optionally requiring runtime-ready identities."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {config_path}: {exc}") from exc

    _validate_root(raw)
    tables = {name: _validated_table(raw, name) for name in sorted(_TOP_LEVEL_SECTIONS)}

    experiment = ExperimentSettings(
        name=_string(tables["experiment"], "experiment", "name"),
        seed=_integer(tables["experiment"], "experiment", "seed"),
    )
    actions = ActionSettings(
        direct=_boolean(tables["actions"], "actions", "direct"),
        reuse=_boolean(tables["actions"], "actions", "reuse"),
        fresh=_boolean(tables["actions"], "actions", "fresh"),
    )
    evaluation = EvaluationSettings(
        good_threshold=_bounded_number(
            tables["evaluation"], "evaluation", "good_threshold", minimum=0.0, maximum=1.0
        ),
        max_history_candidates=_positive_integer(
            tables["evaluation"], "evaluation", "max_history_candidates"
        ),
        paired_harm_is_primary=_boolean(
            tables["evaluation"], "evaluation", "paired_harm_is_primary"
        ),
    )

    environment_values = {
        field.name: _string(tables["environment"], "environment", field.name)
        for field in fields(EnvironmentFingerprint)
    }
    environment = EnvironmentFingerprint(**environment_values)
    environment_compatibility = _construct_policy(
        EnvironmentCompatibilityPolicy,
        {
            "mode": _string(
                tables["environment_compatibility"],
                "environment_compatibility",
                "mode",
            ),
            "minimum_score": _bounded_number(
                tables["environment_compatibility"],
                "environment_compatibility",
                "minimum_score",
                minimum=0.0,
                maximum=1.0,
            ),
            "soft_mismatch_penalty": _bounded_number(
                tables["environment_compatibility"],
                "environment_compatibility",
                "soft_mismatch_penalty",
                minimum=0.0,
                maximum=1.0,
            ),
        },
        section="environment_compatibility",
    )

    write_gate = WriteGateSettings(
        minimum_source_gain=_bounded_number(
            tables["write_gate"],
            "write_gate",
            "minimum_source_gain",
            minimum=0.0,
            maximum=1.0,
        )
    )

    reliability = tables["reliability"]
    lifecycle_values = {
        "promotion_min_observations": _positive_integer(
            reliability, "reliability", "minimum_transfer_observations"
        ),
        "promotion_min_direct_good_observations": _positive_integer(
            reliability, "reliability", "minimum_direct_good_observations"
        ),
        "promotion_min_benefits": _non_negative_integer(
            reliability, "reliability", "minimum_benefits"
        ),
        "promotion_min_mean_gain": _number(reliability, "reliability", "minimum_mean_gain"),
        "promotion_max_wilson_breakage_upper_bound": _bounded_number(
            reliability,
            "reliability",
            "maximum_conditional_breakage_upper_bound",
            minimum=0.0,
            maximum=1.0,
        ),
        "quarantine_at_harm_count": _positive_integer(
            reliability, "reliability", "quarantine_at_harm_count"
        ),
    }
    lifecycle_policy = _construct_policy(LifecyclePolicy, lifecycle_values, section="reliability")

    candidate_recall = CandidateRecallSettings(
        scorer=_string(tables["candidate_recall"], "candidate_recall", "scorer"),
        maximum_candidates=_positive_integer(
            tables["candidate_recall"], "candidate_recall", "maximum_candidates"
        ),
    )
    applicability = ApplicabilitySettings(
        scorer=_string(tables["applicability"], "applicability", "scorer"),
        minimum_score=_bounded_number(
            tables["applicability"],
            "applicability",
            "minimum_score",
            minimum=0.0,
            maximum=1.0,
        ),
    )

    gate_values = {
        "minimum_observations": lifecycle_values["promotion_min_observations"],
        "minimum_benefits": lifecycle_values["promotion_min_benefits"],
        "minimum_mean_gain": lifecycle_values["promotion_min_mean_gain"],
        "maximum_risk_upper_bound": lifecycle_values["promotion_max_wilson_breakage_upper_bound"],
        "minimum_applicability": applicability.minimum_score,
    }
    gate_policy = _construct_policy(GatePolicy, gate_values, section="reliability/applicability")

    application = ApplicationSettings(
        applier_id=_string(tables["application"], "application", "applier_id"),
        application_version=_string(tables["application"], "application", "application_version"),
    )

    budget = tables["budget"]
    maximum_application_cost = None
    if "maximum_application_cost" in budget:
        maximum_application_cost = _non_negative_number(
            budget, "budget", "maximum_application_cost"
        )
    query_budget = _construct_policy(
        QueryBudget,
        {
            "max_retrieval_queries": _positive_integer(
                budget, "budget", "maximum_retrieval_queries"
            ),
            "max_top_k": _positive_integer(budget, "budget", "top_k"),
            "max_context_tokens": _positive_integer(budget, "budget", "context_tokens"),
            "max_query_characters": _positive_integer(budget, "budget", "maximum_query_characters"),
            "max_application_cost": maximum_application_cost,
        },
        section="budget",
    )

    if evaluation.max_history_candidates != candidate_recall.maximum_candidates:
        raise ConfigError(
            "[evaluation].max_history_candidates must equal "
            "[candidate_recall].maximum_candidates so the candidate-set condition is unambiguous"
        )

    config = PilotConfig(
        experiment=experiment,
        actions=actions,
        evaluation=evaluation,
        environment=environment,
        environment_compatibility=environment_compatibility,
        write_gate=write_gate,
        lifecycle_policy=lifecycle_policy,
        gate_policy=gate_policy,
        candidate_recall=candidate_recall,
        applicability=applicability,
        application=application,
        query_budget=query_budget,
    )
    return config.require_runtime_ready() if for_runtime else config


def load_runtime_config(path: str | Path) -> PilotConfig:
    """Load a pilot configuration and reject every unresolved runtime identity."""

    return load_pilot_config(path, for_runtime=True)


def _validate_root(raw: dict[str, Any]) -> None:
    unknown = set(raw) - _TOP_LEVEL_SECTIONS
    missing = _TOP_LEVEL_SECTIONS - set(raw)
    if unknown:
        raise ConfigError(f"unknown top-level configuration sections: {sorted(unknown)}")
    if missing:
        raise ConfigError(f"missing top-level configuration sections: {sorted(missing)}")


def _validated_table(raw: dict[str, Any], name: str) -> dict[str, Any]:
    table = raw[name]
    if not isinstance(table, dict):
        raise ConfigError(f"[{name}] must be a TOML table")
    required, optional = _SECTION_FIELDS[name]
    unknown = set(table) - required - optional
    missing = required - set(table)
    if unknown:
        raise ConfigError(f"unknown fields in [{name}]: {sorted(unknown)}")
    if missing:
        raise ConfigError(f"missing fields in [{name}]: {sorted(missing)}")
    return table


def _path(section: str, field_name: str) -> str:
    return f"[{section}].{field_name}"


def _string(table: dict[str, Any], section: str, field_name: str) -> str:
    value = table[field_name]
    if not isinstance(value, str):
        raise ConfigError(
            f"{_path(section, field_name)} must be a string, got {type(value).__name__}"
        )
    normalized = value.strip()
    if not normalized:
        raise ConfigError(f"{_path(section, field_name)} must not be empty")
    return normalized


def _boolean(table: dict[str, Any], section: str, field_name: str) -> bool:
    value = table[field_name]
    if not isinstance(value, bool):
        raise ConfigError(
            f"{_path(section, field_name)} must be a boolean, got {type(value).__name__}"
        )
    return value


def _integer(table: dict[str, Any], section: str, field_name: str) -> int:
    value = table[field_name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{_path(section, field_name)} must be an integer, got {type(value).__name__}"
        )
    return value


def _positive_integer(table: dict[str, Any], section: str, field_name: str) -> int:
    value = _integer(table, section, field_name)
    if value < 1:
        raise ConfigError(f"{_path(section, field_name)} must be at least 1")
    return value


def _non_negative_integer(table: dict[str, Any], section: str, field_name: str) -> int:
    value = _integer(table, section, field_name)
    if value < 0:
        raise ConfigError(f"{_path(section, field_name)} must be non-negative")
    return value


def _number(table: dict[str, Any], section: str, field_name: str) -> float:
    value = table[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{_path(section, field_name)} must be numeric, got {type(value).__name__}"
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise ConfigError(f"{_path(section, field_name)} must be finite")
    return converted


def _non_negative_number(table: dict[str, Any], section: str, field_name: str) -> float:
    value = _number(table, section, field_name)
    if value < 0.0:
        raise ConfigError(f"{_path(section, field_name)} must be non-negative")
    return value


def _bounded_number(
    table: dict[str, Any],
    section: str,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = _number(table, section, field_name)
    if not minimum <= value <= maximum:
        raise ConfigError(
            f"{_path(section, field_name)} must be in [{minimum}, {maximum}], got {value}"
        )
    return value


def _construct_policy(cls: type[Any], values: dict[str, Any], *, section: str) -> Any:
    try:
        return cls(**values)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid [{section}] policy: {exc}") from exc


def _is_placeholder(value: str) -> bool:
    return not value.strip() or value.strip().casefold() in _PLACEHOLDERS
