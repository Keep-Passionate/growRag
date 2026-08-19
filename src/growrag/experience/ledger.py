"""Evidence ledger and lifecycle rules for reusable query transformations.

This module answers one deliberately narrow question: *has an experience been
reliable on independent target queries?* It does not decide whether the
experience applies to the current query, and it does not compare environment
fingerprints. Those are read-time responsibilities of the calling layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import sqrt

from growrag.evaluation.paired import PairedOutcome, evaluate_pair
from growrag.models import QueryTransformation


class ExperienceState(StrEnum):
    """Lifecycle states of one stored query-transformation experience."""

    CANDIDATE = "candidate"
    ACTIVE = "active"
    QUARANTINE = "quarantine"
    RETIRED = "retired"


class EvidenceRole(StrEnum):
    """How an observation may be used in the experimental protocol."""

    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    """Paired DIRECT/REUSE evidence observed on the source query.

    A transformation passes the write gate only when REUSE is strictly better
    than DIRECT by more than ``minimum_gain``. Source success is therefore
    evidence for admission to the candidate pool, never immediate activation.
    """

    direct_score: float
    reuse_score: float
    dataset_id: str
    split: str
    source_group_id: str
    evidence_role: EvidenceRole
    metric_protocol_id: str
    applier_id: str
    applier_version: str
    minimum_gain: float = 0.0
    gain: float = field(init=False)
    passes_paired_write_gate: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.minimum_gain < 0:
            raise ValueError("minimum_gain must be non-negative")
        _validate_non_empty_fields(
            dataset_id=self.dataset_id,
            split=self.split,
            source_group_id=self.source_group_id,
            metric_protocol_id=self.metric_protocol_id,
            applier_id=self.applier_id,
            applier_version=self.applier_version,
        )
        object.__setattr__(self, "evidence_role", EvidenceRole(self.evidence_role))
        gain = self.reuse_score - self.direct_score
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "passes_paired_write_gate", gain > self.minimum_gain)


@dataclass(frozen=True, slots=True)
class TransferObservation:
    """Paired outcome when an experience is tried on an independent target."""

    target_query_id: str
    direct_score: float
    reuse_score: float
    good_threshold: float
    dataset_id: str
    split: str
    target_group_id: str
    evidence_role: EvidenceRole
    metric_protocol_id: str
    applier_id: str
    applier_version: str
    outcome: PairedOutcome = field(init=False)

    def __post_init__(self) -> None:
        _validate_non_empty_fields(
            target_query_id=self.target_query_id,
            dataset_id=self.dataset_id,
            split=self.split,
            target_group_id=self.target_group_id,
            metric_protocol_id=self.metric_protocol_id,
            applier_id=self.applier_id,
            applier_version=self.applier_version,
        )
        object.__setattr__(self, "evidence_role", EvidenceRole(self.evidence_role))
        paired = evaluate_pair(
            self.direct_score,
            self.reuse_score,
            good_threshold=self.good_threshold,
        )
        object.__setattr__(self, "outcome", paired.outcome)

    @property
    def gain(self) -> float:
        """Score change caused by reusing the experience."""

        return self.reuse_score - self.direct_score

    @classmethod
    def from_scores(
        cls,
        *,
        target_query_id: str,
        direct_score: float,
        reuse_score: float,
        good_threshold: float,
        dataset_id: str,
        split: str,
        target_group_id: str,
        evidence_role: EvidenceRole,
        metric_protocol_id: str,
        applier_id: str,
        applier_version: str,
    ) -> TransferObservation:
        """Construct an observation using the project's four-cell paired label."""

        return cls(
            target_query_id=target_query_id,
            direct_score=direct_score,
            reuse_score=reuse_score,
            good_threshold=good_threshold,
            dataset_id=dataset_id,
            split=split,
            target_group_id=target_group_id,
            evidence_role=evidence_role,
            metric_protocol_id=metric_protocol_id,
            applier_id=applier_id,
            applier_version=applier_version,
        )


def _validate_non_empty_fields(**values: str) -> None:
    for name, value in values.items():
        if not value:
            raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class ReliabilitySummary:
    """Aggregate transfer evidence for one experience.

    The Wilson upper bound treats HARM as the adverse binary event and all
    transfer observations as trials. Returning 1.0 with no observations is a
    deliberate conservative choice: no evidence must not look safe.
    """

    n: int
    benefit_count: int
    harm_count: int
    mean_gain: float
    harm_rate: float
    wilson_95_harm_upper_bound: float
    direct_good_count: int
    direct_bad_count: int
    conditional_breakage_rate: float
    wilson_95_conditional_breakage_upper_bound: float
    repair_rate: float

    @classmethod
    def from_observations(
        cls,
        observations: tuple[TransferObservation, ...],
    ) -> ReliabilitySummary:
        n = len(observations)
        benefit_count = sum(obs.outcome is PairedOutcome.BENEFIT for obs in observations)
        harm_count = sum(obs.outcome is PairedOutcome.HARM for obs in observations)
        mean_gain = sum(obs.gain for obs in observations) / n if n else 0.0
        harm_rate = harm_count / n if n else 0.0
        both_good_count = sum(obs.outcome is PairedOutcome.BOTH_GOOD for obs in observations)
        both_bad_count = sum(obs.outcome is PairedOutcome.BOTH_BAD for obs in observations)
        direct_good_count = harm_count + both_good_count
        direct_bad_count = benefit_count + both_bad_count
        conditional_breakage_rate = harm_count / direct_good_count if direct_good_count else 0.0
        repair_rate = benefit_count / direct_bad_count if direct_bad_count else 0.0
        return cls(
            n=n,
            benefit_count=benefit_count,
            harm_count=harm_count,
            mean_gain=mean_gain,
            harm_rate=harm_rate,
            wilson_95_harm_upper_bound=_wilson_upper_bound(harm_count, n),
            direct_good_count=direct_good_count,
            direct_bad_count=direct_bad_count,
            conditional_breakage_rate=conditional_breakage_rate,
            wilson_95_conditional_breakage_upper_bound=_wilson_upper_bound(
                harm_count,
                direct_good_count,
            ),
            repair_rate=repair_rate,
        )


def _wilson_upper_bound(successes: int, n: int, *, z: float = 1.959963984540054) -> float:
    """Return the upper end of a two-sided 95% Wilson interval."""

    if n < 0 or successes < 0 or successes > n:
        raise ValueError("successes and n must describe a valid binomial sample")
    if n == 0:
        return 1.0

    proportion = successes / n
    z_squared = z * z
    denominator = 1.0 + z_squared / n
    centre = proportion + z_squared / (2.0 * n)
    margin = z * sqrt(proportion * (1.0 - proportion) / n + z_squared / (4.0 * n * n))
    return min(1.0, (centre + margin) / denominator)


@dataclass(frozen=True, slots=True)
class LifecyclePolicy:
    """Explicit, inspectable rules for promotion and quarantine.

    The defaults are for a smoke-test prototype, not a final research claim.
    One harmless trial cannot promote because its Wilson upper bound is about
    0.79; two harmless trials reduce it to about 0.66, below the default 0.70.
    Any observed HARM quarantines immediately under the safety-first default.
    """

    promotion_min_observations: int = 3
    promotion_min_direct_good_observations: int = 2
    promotion_min_benefits: int = 1
    promotion_min_mean_gain: float = 0.0
    promotion_max_wilson_breakage_upper_bound: float = 0.70
    quarantine_at_harm_count: int = 1

    def __post_init__(self) -> None:
        if self.promotion_min_observations < 1:
            raise ValueError("promotion_min_observations must be at least 1")
        if self.promotion_min_direct_good_observations < 1:
            raise ValueError("promotion_min_direct_good_observations must be at least 1")
        if self.promotion_min_benefits < 0:
            raise ValueError("promotion_min_benefits must be non-negative")
        if self.promotion_min_benefits > self.promotion_min_observations:
            raise ValueError("promotion_min_benefits cannot exceed minimum observations")
        if not 0.0 <= self.promotion_max_wilson_breakage_upper_bound <= 1.0:
            raise ValueError("promotion breakage Wilson bound must be between 0 and 1")
        if self.quarantine_at_harm_count < 1:
            raise ValueError("quarantine_at_harm_count must be at least 1")


@dataclass(frozen=True, slots=True)
class ExperienceRecord:
    """One transformation, its source evidence, and independent transfer log."""

    transformation: QueryTransformation
    state: ExperienceState
    source_evidence: SourceEvidence
    transfer_observations: tuple[TransferObservation, ...] = ()

    @property
    def experience_id(self) -> str:
        return self.transformation.experience_id

    @property
    def reliability(self) -> ReliabilitySummary:
        return ReliabilitySummary.from_observations(self.transfer_observations)


class ExperienceLedger:
    """In-memory evidence ledger with deterministic lifecycle transitions."""

    def __init__(self, *, policy: LifecyclePolicy | None = None) -> None:
        self.policy = policy or LifecyclePolicy()
        self._records: dict[str, ExperienceRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, experience_id: object) -> bool:
        return experience_id in self._records

    @property
    def records(self) -> tuple[ExperienceRecord, ...]:
        """Return an immutable snapshot in insertion order."""

        return tuple(self._records.values())

    def get(self, experience_id: str) -> ExperienceRecord:
        """Return a record, raising ``KeyError`` if it is unknown."""

        return self._records[experience_id]

    def add_candidate(
        self,
        transformation: QueryTransformation,
        source_evidence: SourceEvidence,
    ) -> ExperienceRecord | None:
        """Admit a source-validated transformation to the candidate pool.

        A failed paired write gate is a normal rejection and returns ``None``;
        the ledger remains unchanged. Passing the source gate never creates an
        ACTIVE record.
        """

        experience_id = transformation.experience_id
        if experience_id in self._records:
            raise ValueError(f"duplicate experience_id: {experience_id}")
        if source_evidence.evidence_role is EvidenceRole.TEST:
            raise ValueError("test evidence must not update the experience ledger")
        if not source_evidence.passes_paired_write_gate:
            return None

        record = ExperienceRecord(
            transformation=transformation,
            state=ExperienceState.CANDIDATE,
            source_evidence=source_evidence,
        )
        self._records[experience_id] = record
        return record

    def record_transfer(
        self,
        experience_id: str,
        observation: TransferObservation,
        *,
        evaluate: bool = True,
    ) -> ExperienceRecord:
        """Append an independent target observation and optionally transition state.

        Test evidence, the source query/group, duplicate targets, and duplicate
        groups cannot update the ledger. Metric and applier protocols must also
        match the frozen source protocol so heterogeneous trials are not pooled.
        """

        record = self.get(experience_id)
        if record.state is ExperienceState.RETIRED:
            raise ValueError("cannot add transfer evidence to a retired experience")
        if observation.evidence_role is EvidenceRole.TEST:
            raise ValueError("test evidence must not update the experience ledger")
        if observation.target_query_id == record.transformation.source_query_id:
            raise ValueError("source query is not independent transfer evidence")
        source = record.source_evidence
        if (
            observation.dataset_id == source.dataset_id
            and observation.target_group_id == source.source_group_id
        ):
            raise ValueError("source group is not independent transfer evidence")
        if observation.metric_protocol_id != source.metric_protocol_id:
            raise ValueError("metric protocol differs from source evidence")
        if (observation.applier_id, observation.applier_version) != (
            source.applier_id,
            source.applier_version,
        ):
            raise ValueError("applier identity or version differs from source evidence")
        if any(
            (prior.dataset_id, prior.target_query_id)
            == (observation.dataset_id, observation.target_query_id)
            for prior in record.transfer_observations
        ):
            raise ValueError(f"duplicate target_query_id: {observation.target_query_id}")
        if any(
            (prior.dataset_id, prior.target_group_id)
            == (observation.dataset_id, observation.target_group_id)
            for prior in record.transfer_observations
        ):
            raise ValueError(f"duplicate target_group_id: {observation.target_group_id}")

        updated = replace(
            record,
            transfer_observations=(*record.transfer_observations, observation),
        )
        self._records[experience_id] = updated
        return self.evaluate_state(experience_id) if evaluate else updated

    def evaluate_state(self, experience_id: str) -> ExperienceRecord:
        """Apply the configured safety-first lifecycle policy."""

        record = self.get(experience_id)
        if record.state in {ExperienceState.QUARANTINE, ExperienceState.RETIRED}:
            return record

        summary = record.reliability
        if summary.harm_count >= self.policy.quarantine_at_harm_count:
            next_state = ExperienceState.QUARANTINE
        elif (
            record.state is ExperienceState.CANDIDATE
            and summary.n >= self.policy.promotion_min_observations
            and summary.direct_good_count >= self.policy.promotion_min_direct_good_observations
            and summary.wilson_95_conditional_breakage_upper_bound
            <= self.policy.promotion_max_wilson_breakage_upper_bound
            and summary.benefit_count >= self.policy.promotion_min_benefits
            and summary.mean_gain > self.policy.promotion_min_mean_gain
        ):
            next_state = ExperienceState.ACTIVE
        else:
            next_state = record.state

        if next_state is record.state:
            return record
        updated = replace(record, state=next_state)
        self._records[experience_id] = updated
        return updated

    def retire(self, experience_id: str) -> ExperienceRecord:
        """Irreversibly retire a record from automatic lifecycle evaluation."""

        record = self.get(experience_id)
        if record.state is ExperienceState.RETIRED:
            return record
        updated = replace(record, state=ExperienceState.RETIRED)
        self._records[experience_id] = updated
        return updated
