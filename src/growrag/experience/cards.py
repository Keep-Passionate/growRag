"""Versioned procedural query-repair cards backed by immutable episodes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from growrag.episodes.models import EpisodeTurnRef
from growrag.query_operators import ActionBody, ExpansionBody, ParaphraseBody


class CardLifecycle(StrEnum):
    """Serving lifecycle for a versioned experience card."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    QUARANTINE = "quarantine"
    RETIRED = "retired"


class ActivationStage(StrEnum):
    """Pipeline stage at which a card is allowed to apply."""

    PRE_RETRIEVAL = "pre_retrieval"
    POST_RETRIEVAL = "post_retrieval"


class VerificationTier(StrEnum):
    """Evidence strength used to validate a reusable card."""

    GOLD = "gold"
    HUMAN = "human"
    MULTI_JUDGE = "multi_judge"
    PROXY = "proxy"


@dataclass(frozen=True, slots=True)
class CardActivationPolicy:
    """Explicit, versionable serving gate instead of a hidden hard-coded threshold."""

    allowed_verification_tiers: tuple[VerificationTier, ...] = (
        VerificationTier.GOLD,
        VerificationTier.HUMAN,
    )
    min_independent_episodes: int = 2
    min_independent_documents: int = 2
    min_matched_trials: int = 2
    min_benefit_count: int = 2
    max_conditional_harm_rate: float = 0.1

    def __post_init__(self) -> None:
        tiers = tuple(VerificationTier(tier) for tier in self.allowed_verification_tiers)
        if not tiers or len(tiers) != len(set(tiers)):
            raise ValueError("allowed_verification_tiers must be unique and non-empty")
        object.__setattr__(self, "allowed_verification_tiers", tiers)
        for name, value in (
            ("min_independent_episodes", self.min_independent_episodes),
            ("min_independent_documents", self.min_independent_documents),
            ("min_matched_trials", self.min_matched_trials),
            ("min_benefit_count", self.min_benefit_count),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.min_independent_documents > self.min_independent_episodes:
            raise ValueError("document minimum cannot exceed episode minimum")
        if not (
            isfinite(self.max_conditional_harm_rate)
            and 0.0 <= self.max_conditional_harm_rate <= 1.0
        ):
            raise ValueError("max_conditional_harm_rate must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class CardActivation:
    """Stable rules stored on a card, excluding query-specific fit scores."""

    stage: ActivationStage
    query_pattern: str
    gap_pattern: str | None = None
    preconditions: tuple[str, ...] = ()
    contraindications: tuple[str, ...] = ()
    required_retriever_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", ActivationStage(self.stage))
        _require_text_fields(query_pattern=self.query_pattern)
        if self.gap_pattern is not None:
            _require_text_fields(gap_pattern=self.gap_pattern)
        for field_name in (
            "preconditions",
            "contraindications",
            "required_retriever_capabilities",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_unique_text(getattr(self, field_name), field_name=field_name),
            )


@dataclass(frozen=True, slots=True)
class RepairSpecification:
    """Entity-neutral repair procedure and its observable success contract."""

    operator_type: str
    slot_template: str | None
    evidence_contract: str
    action_body: ActionBody | None = None
    intent: str | None = None

    def __post_init__(self) -> None:
        _require_text_fields(
            operator_type=self.operator_type,
            evidence_contract=self.evidence_contract,
        )
        if self.action_body is None:
            _require_text_fields(slot_template=self.slot_template)
            if self.intent is not None:
                raise ValueError("typed intent requires an action body")
        else:
            if not isinstance(self.action_body, (ParaphraseBody, ExpansionBody)):
                raise TypeError("unsupported action body")
            if self.operator_type != self.action_body.form.value:
                raise ValueError("operator_type must agree with the action body form")
            if self.slot_template is not None:
                raise ValueError("typed body replaces slot_template; do not duplicate instructions")
            _require_text_fields(intent=self.intent)


@dataclass(frozen=True, slots=True)
class CardProvenance:
    """Pointers from an abstraction back to concrete q-to-q-prime turns."""

    source_episode_turn_refs: tuple[EpisodeTurnRef, ...]
    canonical_example_refs: tuple[EpisodeTurnRef, ...]
    independent_episode_count: int
    independent_document_count: int

    def __post_init__(self) -> None:
        _require_unique_turn_refs(
            self.source_episode_turn_refs,
            field_name="source_episode_turn_refs",
        )
        _require_unique_turn_refs(
            self.canonical_example_refs,
            field_name="canonical_example_refs",
        )
        if not self.source_episode_turn_refs:
            raise ValueError("a card must reference at least one source episode turn")
        if not set(self.canonical_example_refs).issubset(self.source_episode_turn_refs):
            raise ValueError("canonical examples must be a subset of source episode turns")
        for name, value in (
            ("independent_episode_count", self.independent_episode_count),
            ("independent_document_count", self.independent_document_count),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.independent_episode_count < 1:
            raise ValueError("independent_episode_count must be at least 1")
        if self.independent_document_count > self.independent_episode_count:
            raise ValueError("independent_document_count cannot exceed independent_episode_count")


@dataclass(frozen=True, slots=True)
class CardValidation:
    """Sufficient statistics; confidence estimates remain recomputable."""

    verification_tier: VerificationTier
    verification_event_ids: tuple[str, ...]
    matched_trials: int
    benefit_count: int
    neutral_count: int
    harm_count: int
    direct_correct_trials: int
    mean_gain: float
    last_validated_at: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verification_tier",
            VerificationTier(self.verification_tier),
        )
        if self.last_validated_at is None:
            if self.matched_trials:
                raise ValueError("validated trials require last_validated_at")
        else:
            _require_text_fields(last_validated_at=self.last_validated_at)
        object.__setattr__(
            self,
            "verification_event_ids",
            _normalize_unique_text(
                self.verification_event_ids,
                field_name="verification_event_ids",
            ),
        )
        for name, value in (
            ("matched_trials", self.matched_trials),
            ("benefit_count", self.benefit_count),
            ("neutral_count", self.neutral_count),
            ("harm_count", self.harm_count),
            ("direct_correct_trials", self.direct_correct_trials),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.benefit_count + self.neutral_count + self.harm_count != self.matched_trials:
            raise ValueError("benefit + neutral + harm counts must equal matched_trials")
        if len(self.verification_event_ids) != self.matched_trials:
            raise ValueError("verification_event_ids must account for every matched trial")
        if self.direct_correct_trials > self.matched_trials:
            raise ValueError("direct_correct_trials cannot exceed matched_trials")
        if self.harm_count > self.direct_correct_trials:
            raise ValueError("harm_count cannot exceed direct_correct_trials")
        if not isfinite(self.mean_gain):
            raise ValueError("mean_gain must be finite")

    @property
    def conditional_harm_rate(self) -> float | None:
        """Observed P(REUSE bad | BASE good), if the condition was observed."""

        if self.direct_correct_trials == 0:
            return None
        return self.harm_count / self.direct_correct_trials


@dataclass(frozen=True, slots=True)
class CardServing:
    """Small cost and use profile; not a substitute for validation evidence."""

    expected_cost: float
    use_count: int = 0
    last_used_at: str | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.expected_cost) or self.expected_cost < 0:
            raise ValueError("expected_cost must be finite and non-negative")
        if self.use_count < 0:
            raise ValueError("use_count must be non-negative")
        if self.last_used_at is not None and not self.last_used_at.strip():
            raise ValueError("last_used_at must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class ExperienceCard:
    """A versioned serving view, never the sole source of historical truth."""

    card_id: str
    version: str
    lifecycle_state: CardLifecycle
    created_at: str
    activation: CardActivation
    repair: RepairSpecification
    provenance: CardProvenance
    validation: CardValidation
    serving: CardServing
    activation_policy: CardActivationPolicy = CardActivationPolicy()
    parent_versioned_ids: tuple[str, ...] = ()
    schema_version: str = "experience_card.v0"

    def __post_init__(self) -> None:
        _require_text_fields(
            card_id=self.card_id,
            version=self.version,
            created_at=self.created_at,
            schema_version=self.schema_version,
        )
        object.__setattr__(self, "lifecycle_state", CardLifecycle(self.lifecycle_state))
        if self.repair.action_body is not None and self.schema_version != "experience_card.v1":
            raise ValueError("typed action bodies require experience_card.v1")
        object.__setattr__(
            self,
            "parent_versioned_ids",
            _normalize_unique_text(
                self.parent_versioned_ids,
                field_name="parent_versioned_ids",
            ),
        )
        if self.lifecycle_state is CardLifecycle.ACTIVE:
            if self.validation.benefit_count < 1 or self.validation.mean_gain <= 0:
                raise ValueError("ACTIVE cards require observed positive paired benefit")
            failures = self.activation_policy_failures()
            if failures:
                raise ValueError(f"ACTIVE card violates activation policy: {'; '.join(failures)}")

    @property
    def versioned_id(self) -> str:
        return f"{self.card_id}@{self.version}"

    def revised_candidate(
        self,
        *,
        version: str,
        created_at: str,
        activation: CardActivation,
        repair: RepairSpecification,
        expected_cost: float,
    ) -> ExperienceCard:
        """A changed/compressed representation does not inherit validation.

        Caller must explicitly supply the revised scope and cost estimate. The
        old card and its evidence remain intact; this child is unvalidated, not
        automatically worse or better. This function is NOT a compressor.
        """
        if version == self.version:
            raise ValueError("revision requires a new version")
        return ExperienceCard(
            card_id=self.card_id,
            version=version,
            lifecycle_state=CardLifecycle.CANDIDATE,
            created_at=created_at,
            activation=activation,
            repair=repair,
            provenance=self.provenance,
            validation=CardValidation(
                VerificationTier.PROXY,
                (),
                0,
                0,
                0,
                0,
                0,
                0.0,
                None,
            ),
            serving=CardServing(expected_cost=expected_cost),
            activation_policy=self.activation_policy,
            parent_versioned_ids=(self.versioned_id,),
            schema_version=(
                "experience_card.v1" if repair.action_body is not None else self.schema_version
            ),
        )

    def activation_policy_failures(self) -> tuple[str, ...]:
        """Explain why the card is not yet trusted under its recorded policy."""

        policy = self.activation_policy
        failures: list[str] = []
        if self.validation.verification_tier not in policy.allowed_verification_tiers:
            failures.append("verification tier is not allowed")
        if self.provenance.independent_episode_count < policy.min_independent_episodes:
            failures.append("too few independent episodes")
        if self.provenance.independent_document_count < policy.min_independent_documents:
            failures.append("too few independent documents")
        if self.validation.matched_trials < policy.min_matched_trials:
            failures.append("too few matched trials")
        if self.validation.benefit_count < policy.min_benefit_count:
            failures.append("too few beneficial transfers")
        harm_rate = self.validation.conditional_harm_rate
        if harm_rate is None:
            failures.append("conditional harm is unobserved")
        elif harm_rate > policy.max_conditional_harm_rate:
            failures.append("conditional harm exceeds the serving ceiling")
        if self.validation.mean_gain <= 0:
            failures.append("mean paired gain is not positive")
        return tuple(failures)


def _require_text_fields(**values: str) -> None:
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")


def _normalize_unique_text(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name} must be a tuple of strings")
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-empty string")
        cleaned = " ".join(value.strip().split())
        if cleaned in normalized:
            raise ValueError(f"duplicate value in {field_name}: {cleaned}")
        normalized.append(cleaned)
    return tuple(normalized)


def _require_unique_turn_refs(
    values: tuple[EpisodeTurnRef, ...],
    *,
    field_name: str,
) -> None:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name} must be a tuple of EpisodeTurnRef values")
    if any(not isinstance(value, EpisodeTurnRef) for value in values):
        raise ValueError(f"{field_name} must contain only EpisodeTurnRef values")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicate references")
