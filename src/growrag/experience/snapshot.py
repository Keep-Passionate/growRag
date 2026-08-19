"""Strict, deterministic JSON snapshots for the experience ledger.

Snapshots are an audit format, not a permissive interchange format. Every
object has an exact field allow-list, duplicate JSON keys are rejected, and a
snapshot is restored only after all lifecycle and anti-leakage invariants pass.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from growrag.experience.ledger import (
    EvidenceRole,
    ExperienceLedger,
    ExperienceRecord,
    ExperienceState,
    LifecyclePolicy,
    ReliabilitySummary,
    SourceEvidence,
    TransferObservation,
)
from growrag.models import EnvironmentFingerprint, QueryTransformation

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, SCHEMA_VERSION})


class SnapshotValidationError(ValueError):
    """Raised when a snapshot fails closed validation."""


@dataclass(frozen=True, slots=True)
class ExperienceSnapshot:
    """A validated ledger together with its persistent identity and schema."""

    schema_version: int
    memory_snapshot_id: str
    ledger: ExperienceLedger

    def require_runtime_ready(self) -> ExperienceSnapshot:
        """Reject audit-readable legacy cards before deployable selection.

        Schema v1 did not represent contraindications. Treating its missing
        field as an audited empty set would be fail-open, so v1 is available
        only for explicit migration and review.
        """

        if self.schema_version != SCHEMA_VERSION:
            raise SnapshotValidationError(
                f"memory schema v{self.schema_version} is audit/migration-only; "
                f"review the cards and save a new schema v{SCHEMA_VERSION} snapshot "
                "before runtime decisions"
            )
        return self


def dumps_snapshot(ledger: ExperienceLedger, *, memory_snapshot_id: str) -> str:
    """Serialize ``ledger`` to canonical, deterministic JSON."""

    _require_non_empty_string(memory_snapshot_id, "memory_snapshot_id")
    records = sorted(ledger.records, key=lambda record: record.experience_id)
    _validate_unique_experience_ids(tuple(records))
    for record in records:
        _validate_record_invariants(record, ledger.policy)

    payload = {
        "lifecycle_policy": _policy_to_dict(ledger.policy),
        "memory_snapshot_id": memory_snapshot_id,
        "records": [_record_to_dict(record) for record in records],
        "schema_version": SCHEMA_VERSION,
    }
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SnapshotValidationError(f"ledger is not JSON-safe: {exc}") from exc


def loads_snapshot(data: str) -> ExperienceSnapshot:
    """Load canonical snapshot data using strict, fail-closed validation."""

    if not isinstance(data, str):
        raise SnapshotValidationError("snapshot data must be text")
    try:
        raw = json.loads(
            data,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except SnapshotValidationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SnapshotValidationError(f"invalid JSON snapshot: {exc}") from exc

    root = _expect_object(raw, "snapshot")
    _require_exact_keys(
        root,
        {"schema_version", "memory_snapshot_id", "lifecycle_policy", "records"},
        "snapshot",
    )
    schema_version = _expect_int(root["schema_version"], "schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SnapshotValidationError(
            f"unsupported schema_version {schema_version}; "
            f"supported={sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    snapshot_id = _expect_string(root["memory_snapshot_id"], "memory_snapshot_id")
    _require_non_empty_string(snapshot_id, "memory_snapshot_id")
    policy = _policy_from_dict(root["lifecycle_policy"])

    raw_records = _expect_list(root["records"], "records")
    records = tuple(
        _record_from_dict(
            value,
            path=f"records[{index}]",
            schema_version=schema_version,
        )
        for index, value in enumerate(raw_records)
    )
    _validate_unique_experience_ids(records)
    for record in records:
        _validate_record_invariants(record, policy)

    ledger = ExperienceLedger(policy=policy)
    # Restoration is intentionally not replayed through add_candidate and
    # record_transfer: replay would recalculate lifecycle state. We validate all
    # equivalent write/test/leakage invariants first, then restore exact state.
    ledger._records = {record.experience_id: record for record in records}
    return ExperienceSnapshot(
        schema_version=schema_version,
        memory_snapshot_id=snapshot_id,
        ledger=ledger,
    )


def save_snapshot(
    path: str | os.PathLike[str],
    ledger: ExperienceLedger,
    *,
    memory_snapshot_id: str,
) -> Path:
    """Atomically save a canonical UTF-8 snapshot and return its path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = dumps_snapshot(ledger, memory_snapshot_id=memory_snapshot_id)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()
    return target


def load_snapshot(path: str | os.PathLike[str]) -> ExperienceSnapshot:
    """Load and validate a UTF-8 snapshot file."""

    try:
        data = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SnapshotValidationError(f"cannot read snapshot: {exc}") from exc
    return loads_snapshot(data)


def _record_to_dict(record: ExperienceRecord) -> dict[str, Any]:
    transformation = record.transformation
    return {
        "source_evidence": _source_to_dict(record.source_evidence),
        "state": record.state.value,
        "transfer_observations": [
            _observation_to_dict(observation) for observation in record.transfer_observations
        ],
        "transformation": {
            "applicability_signature": list(transformation.applicability_signature),
            "application_note": transformation.application_note,
            "atomic_units": list(transformation.atomic_units),
            "contraindication_signature": list(transformation.contraindication_signature),
            "diagnosed_failure": transformation.diagnosed_failure,
            "environment": _environment_to_dict(transformation.environment),
            "experience_id": transformation.experience_id,
            "provenance": transformation.provenance,
            "gap_categories": list(transformation.gap_categories),
            "source_dataset_id": transformation.source_dataset_id,
            "source_group_id": transformation.source_group_id,
            "source_query": transformation.source_query,
            "source_query_id": transformation.source_query_id,
            "source_split": transformation.source_split,
            "transformation_type": transformation.transformation_type,
            "transformed_query": transformation.transformed_query,
        },
    }


def _record_from_dict(
    value: object,
    *,
    path: str,
    schema_version: int,
) -> ExperienceRecord:
    raw = _expect_object(value, path)
    _require_exact_keys(
        raw,
        {"transformation", "state", "source_evidence", "transfer_observations"},
        path,
    )
    transformation = _transformation_from_dict(
        raw["transformation"],
        f"{path}.transformation",
        schema_version=schema_version,
    )
    try:
        state = ExperienceState(_expect_string(raw["state"], f"{path}.state"))
    except ValueError as exc:
        raise SnapshotValidationError(f"{path}.state is not a known lifecycle state") from exc
    source = _source_from_dict(raw["source_evidence"], f"{path}.source_evidence")
    observations_raw = _expect_list(
        raw["transfer_observations"],
        f"{path}.transfer_observations",
    )
    observations = tuple(
        _observation_from_dict(item, f"{path}.transfer_observations[{index}]")
        for index, item in enumerate(observations_raw)
    )
    return ExperienceRecord(
        transformation=transformation,
        state=state,
        source_evidence=source,
        transfer_observations=observations,
    )


def _transformation_from_dict(
    value: object,
    path: str,
    *,
    schema_version: int,
) -> QueryTransformation:
    raw = _expect_object(value, path)
    v1_fields = {
        "experience_id",
        "source_query_id",
        "source_query",
        "transformed_query",
        "atomic_units",
        "environment",
        "provenance",
        "transformation_type",
        "application_note",
        "applicability_signature",
        "source_dataset_id",
        "source_split",
        "source_group_id",
    }
    v2_fields = v1_fields | {
        "diagnosed_failure",
        "gap_categories",
        "contraindication_signature",
    }
    fields = v1_fields if schema_version == 1 else v2_fields
    _require_exact_keys(raw, fields, path)
    atomic_units = _string_tuple(raw["atomic_units"], f"{path}.atomic_units")
    signature = _string_tuple(
        raw["applicability_signature"],
        f"{path}.applicability_signature",
    )
    diagnosed_failure = ""
    gap_categories: tuple[str, ...] = ()
    contraindications: tuple[str, ...] = ()
    if schema_version == 2:
        diagnosed_failure = _expect_string(
            raw["diagnosed_failure"],
            f"{path}.diagnosed_failure",
        )
        gap_categories = _string_tuple(
            raw["gap_categories"],
            f"{path}.gap_categories",
        )
        contraindications = _string_tuple(
            raw["contraindication_signature"],
            f"{path}.contraindication_signature",
        )
    try:
        return QueryTransformation(
            experience_id=_expect_string(raw["experience_id"], f"{path}.experience_id"),
            source_query_id=_expect_string(
                raw["source_query_id"],
                f"{path}.source_query_id",
            ),
            source_query=_expect_string(raw["source_query"], f"{path}.source_query"),
            transformed_query=_expect_string(
                raw["transformed_query"],
                f"{path}.transformed_query",
            ),
            atomic_units=atomic_units,
            environment=_environment_from_dict(raw["environment"], f"{path}.environment"),
            provenance=_expect_string(raw["provenance"], f"{path}.provenance"),
            transformation_type=_expect_string(
                raw["transformation_type"],
                f"{path}.transformation_type",
            ),
            application_note=_expect_string(
                raw["application_note"],
                f"{path}.application_note",
            ),
            applicability_signature=signature,
            source_dataset_id=_expect_string(
                raw["source_dataset_id"],
                f"{path}.source_dataset_id",
            ),
            source_split=_expect_string(raw["source_split"], f"{path}.source_split"),
            source_group_id=_expect_string(
                raw["source_group_id"],
                f"{path}.source_group_id",
            ),
            diagnosed_failure=diagnosed_failure,
            gap_categories=gap_categories,
            contraindication_signature=contraindications,
        )
    except SnapshotValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise SnapshotValidationError(f"invalid {path}: {exc}") from exc


def _environment_to_dict(environment: EnvironmentFingerprint) -> dict[str, str]:
    return {
        "analyzer_id": environment.analyzer_id,
        "analyzer_version": environment.analyzer_version,
        "corpus_id": environment.corpus_id,
        "corpus_version": environment.corpus_version,
        "index_id": environment.index_id,
        "index_version": environment.index_version,
        "retriever_id": environment.retriever_id,
        "retriever_version": environment.retriever_version,
        "rewriter_id": environment.rewriter_id,
        "rewriter_version": environment.rewriter_version,
    }


def _environment_from_dict(value: object, path: str) -> EnvironmentFingerprint:
    raw = _expect_object(value, path)
    fields = {
        "corpus_id",
        "corpus_version",
        "retriever_id",
        "retriever_version",
        "rewriter_id",
        "rewriter_version",
        "index_id",
        "index_version",
        "analyzer_id",
        "analyzer_version",
    }
    _require_exact_keys(raw, fields, path)
    values = {name: _expect_string(raw[name], f"{path}.{name}") for name in fields}
    return EnvironmentFingerprint(**values)


def _source_to_dict(source: SourceEvidence) -> dict[str, Any]:
    return {
        "applier_id": source.applier_id,
        "applier_version": source.applier_version,
        "dataset_id": source.dataset_id,
        "direct_score": float(source.direct_score),
        "evidence_role": source.evidence_role.value,
        "gain": float(source.gain),
        "metric_protocol_id": source.metric_protocol_id,
        "minimum_gain": float(source.minimum_gain),
        "passes_paired_write_gate": source.passes_paired_write_gate,
        "reuse_score": float(source.reuse_score),
        "source_group_id": source.source_group_id,
        "split": source.split,
    }


def _source_from_dict(value: object, path: str) -> SourceEvidence:
    raw = _expect_object(value, path)
    fields = {
        "direct_score",
        "reuse_score",
        "dataset_id",
        "split",
        "source_group_id",
        "evidence_role",
        "metric_protocol_id",
        "applier_id",
        "applier_version",
        "minimum_gain",
        "gain",
        "passes_paired_write_gate",
    }
    _require_exact_keys(raw, fields, path)
    try:
        source = SourceEvidence(
            direct_score=_expect_number(raw["direct_score"], f"{path}.direct_score"),
            reuse_score=_expect_number(raw["reuse_score"], f"{path}.reuse_score"),
            dataset_id=_expect_string(raw["dataset_id"], f"{path}.dataset_id"),
            split=_expect_string(raw["split"], f"{path}.split"),
            source_group_id=_expect_string(
                raw["source_group_id"],
                f"{path}.source_group_id",
            ),
            evidence_role=_evidence_role(raw["evidence_role"], f"{path}.evidence_role"),
            metric_protocol_id=_expect_string(
                raw["metric_protocol_id"],
                f"{path}.metric_protocol_id",
            ),
            applier_id=_expect_string(raw["applier_id"], f"{path}.applier_id"),
            applier_version=_expect_string(
                raw["applier_version"],
                f"{path}.applier_version",
            ),
            minimum_gain=_expect_number(raw["minimum_gain"], f"{path}.minimum_gain"),
        )
    except SnapshotValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise SnapshotValidationError(f"invalid {path}: {exc}") from exc
    stored_gain = _expect_number(raw["gain"], f"{path}.gain")
    stored_gate = _expect_bool(
        raw["passes_paired_write_gate"],
        f"{path}.passes_paired_write_gate",
    )
    if stored_gain != source.gain or stored_gate is not source.passes_paired_write_gate:
        raise SnapshotValidationError(f"{path} contains inconsistent derived source evidence")
    return source


def _observation_to_dict(observation: TransferObservation) -> dict[str, Any]:
    return {
        "applier_id": observation.applier_id,
        "applier_version": observation.applier_version,
        "dataset_id": observation.dataset_id,
        "direct_score": float(observation.direct_score),
        "evidence_role": observation.evidence_role.value,
        "gain": float(observation.gain),
        "good_threshold": float(observation.good_threshold),
        "metric_protocol_id": observation.metric_protocol_id,
        "outcome": observation.outcome.value,
        "reuse_score": float(observation.reuse_score),
        "split": observation.split,
        "target_group_id": observation.target_group_id,
        "target_query_id": observation.target_query_id,
    }


def _observation_from_dict(value: object, path: str) -> TransferObservation:
    raw = _expect_object(value, path)
    fields = {
        "target_query_id",
        "direct_score",
        "reuse_score",
        "good_threshold",
        "dataset_id",
        "split",
        "target_group_id",
        "evidence_role",
        "metric_protocol_id",
        "applier_id",
        "applier_version",
        "outcome",
        "gain",
    }
    _require_exact_keys(raw, fields, path)
    try:
        observation = TransferObservation(
            target_query_id=_expect_string(
                raw["target_query_id"],
                f"{path}.target_query_id",
            ),
            direct_score=_expect_number(raw["direct_score"], f"{path}.direct_score"),
            reuse_score=_expect_number(raw["reuse_score"], f"{path}.reuse_score"),
            good_threshold=_expect_number(
                raw["good_threshold"],
                f"{path}.good_threshold",
            ),
            dataset_id=_expect_string(raw["dataset_id"], f"{path}.dataset_id"),
            split=_expect_string(raw["split"], f"{path}.split"),
            target_group_id=_expect_string(
                raw["target_group_id"],
                f"{path}.target_group_id",
            ),
            evidence_role=_evidence_role(raw["evidence_role"], f"{path}.evidence_role"),
            metric_protocol_id=_expect_string(
                raw["metric_protocol_id"],
                f"{path}.metric_protocol_id",
            ),
            applier_id=_expect_string(raw["applier_id"], f"{path}.applier_id"),
            applier_version=_expect_string(
                raw["applier_version"],
                f"{path}.applier_version",
            ),
        )
    except SnapshotValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise SnapshotValidationError(f"invalid {path}: {exc}") from exc
    stored_outcome = _expect_string(raw["outcome"], f"{path}.outcome")
    stored_gain = _expect_number(raw["gain"], f"{path}.gain")
    if stored_outcome != observation.outcome.value or stored_gain != observation.gain:
        raise SnapshotValidationError(f"{path} contains inconsistent derived observation")
    return observation


def _policy_to_dict(policy: LifecyclePolicy) -> dict[str, int | float]:
    return {
        "promotion_max_wilson_breakage_upper_bound": (
            float(policy.promotion_max_wilson_breakage_upper_bound)
        ),
        "promotion_min_benefits": policy.promotion_min_benefits,
        "promotion_min_direct_good_observations": (policy.promotion_min_direct_good_observations),
        "promotion_min_mean_gain": float(policy.promotion_min_mean_gain),
        "promotion_min_observations": policy.promotion_min_observations,
        "quarantine_at_harm_count": policy.quarantine_at_harm_count,
    }


def _policy_from_dict(value: object) -> LifecyclePolicy:
    path = "lifecycle_policy"
    raw = _expect_object(value, path)
    fields = {
        "promotion_min_observations",
        "promotion_min_direct_good_observations",
        "promotion_min_benefits",
        "promotion_min_mean_gain",
        "promotion_max_wilson_breakage_upper_bound",
        "quarantine_at_harm_count",
    }
    _require_exact_keys(raw, fields, path)
    try:
        return LifecyclePolicy(
            promotion_min_observations=_expect_int(
                raw["promotion_min_observations"],
                f"{path}.promotion_min_observations",
            ),
            promotion_min_direct_good_observations=_expect_int(
                raw["promotion_min_direct_good_observations"],
                f"{path}.promotion_min_direct_good_observations",
            ),
            promotion_min_benefits=_expect_int(
                raw["promotion_min_benefits"],
                f"{path}.promotion_min_benefits",
            ),
            promotion_min_mean_gain=_expect_number(
                raw["promotion_min_mean_gain"],
                f"{path}.promotion_min_mean_gain",
            ),
            promotion_max_wilson_breakage_upper_bound=_expect_number(
                raw["promotion_max_wilson_breakage_upper_bound"],
                f"{path}.promotion_max_wilson_breakage_upper_bound",
            ),
            quarantine_at_harm_count=_expect_int(
                raw["quarantine_at_harm_count"],
                f"{path}.quarantine_at_harm_count",
            ),
        )
    except ValueError as exc:
        raise SnapshotValidationError(f"invalid lifecycle policy: {exc}") from exc


def _validate_record_invariants(record: ExperienceRecord, policy: LifecyclePolicy) -> None:
    source = record.source_evidence
    transformation = record.transformation
    if not transformation.experience_id.strip():
        raise SnapshotValidationError("experience_id must not be empty")
    if source.evidence_role is EvidenceRole.TEST:
        raise SnapshotValidationError("test source evidence cannot exist in memory")
    if not source.passes_paired_write_gate:
        raise SnapshotValidationError("stored source evidence fails the paired write gate")
    _validate_source_metadata_consistency(transformation, source)

    seen_targets: set[tuple[str, str]] = set()
    seen_groups: set[tuple[str, str]] = set()
    prefixes: list[TransferObservation] = []
    expected_state = ExperienceState.CANDIDATE
    for observation in record.transfer_observations:
        if observation.evidence_role is EvidenceRole.TEST:
            raise SnapshotValidationError("test transfer evidence cannot exist in memory")
        if observation.target_query_id == transformation.source_query_id:
            raise SnapshotValidationError("source query appears as transfer evidence")
        if (
            observation.dataset_id == source.dataset_id
            and observation.target_group_id == source.source_group_id
        ):
            raise SnapshotValidationError("source group appears as transfer evidence")
        if observation.metric_protocol_id != source.metric_protocol_id:
            raise SnapshotValidationError("transfer metric protocol differs from source")
        if (observation.applier_id, observation.applier_version) != (
            source.applier_id,
            source.applier_version,
        ):
            raise SnapshotValidationError("transfer applier differs from source")
        target_key = (observation.dataset_id, observation.target_query_id)
        group_key = (observation.dataset_id, observation.target_group_id)
        if target_key in seen_targets:
            raise SnapshotValidationError("duplicate transfer target")
        if group_key in seen_groups:
            raise SnapshotValidationError("duplicate transfer group")
        seen_targets.add(target_key)
        seen_groups.add(group_key)
        prefixes.append(observation)
        expected_state = _next_state(expected_state, tuple(prefixes), policy)

    if record.state is not ExperienceState.RETIRED and record.state is not expected_state:
        raise SnapshotValidationError(
            f"stored state {record.state.value!r} contradicts its ordered evidence"
        )


def _next_state(
    state: ExperienceState,
    observations: tuple[TransferObservation, ...],
    policy: LifecyclePolicy,
) -> ExperienceState:
    if state in {ExperienceState.QUARANTINE, ExperienceState.RETIRED}:
        return state
    summary = ReliabilitySummary.from_observations(observations)
    if summary.harm_count >= policy.quarantine_at_harm_count:
        return ExperienceState.QUARANTINE
    if (
        state is ExperienceState.CANDIDATE
        and summary.n >= policy.promotion_min_observations
        and summary.direct_good_count >= policy.promotion_min_direct_good_observations
        and summary.wilson_95_conditional_breakage_upper_bound
        <= policy.promotion_max_wilson_breakage_upper_bound
        and summary.benefit_count >= policy.promotion_min_benefits
        and summary.mean_gain > policy.promotion_min_mean_gain
    ):
        return ExperienceState.ACTIVE
    return state


def _validate_source_metadata_consistency(
    transformation: QueryTransformation,
    source: SourceEvidence,
) -> None:
    pairs = (
        (transformation.source_dataset_id, source.dataset_id, "source_dataset_id"),
        (transformation.source_split, source.split, "source_split"),
        (transformation.source_group_id, source.source_group_id, "source_group_id"),
    )
    for stored, evidence, name in pairs:
        if not _is_unspecified(stored) and stored != evidence:
            raise SnapshotValidationError(f"transformation {name} contradicts source evidence")


def _validate_unique_experience_ids(records: tuple[ExperienceRecord, ...]) -> None:
    identifiers = [record.experience_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise SnapshotValidationError("duplicate experience_id in snapshot")


def _is_unspecified(value: str) -> bool:
    return value.strip().casefold() in {"", "unset", "unspecified", "unknown"}


def _evidence_role(value: object, path: str) -> EvidenceRole:
    try:
        return EvidenceRole(_expect_string(value, path))
    except ValueError as exc:
        raise SnapshotValidationError(f"{path} is not a known evidence role") from exc


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SnapshotValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise SnapshotValidationError(f"non-finite JSON number is forbidden: {value}")


def _require_exact_keys(raw: dict[str, object], expected: set[str], path: str) -> None:
    actual = set(raw)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise SnapshotValidationError(
            f"{path} fields differ from schema; unknown={unknown}, missing={missing}"
        )


def _expect_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SnapshotValidationError(f"{path} must be an object")
    return value


def _expect_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise SnapshotValidationError(f"{path} must be an array")
    return value


def _expect_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise SnapshotValidationError(f"{path} must be a string")
    return value


def _expect_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise SnapshotValidationError(f"{path} must be a boolean")
    return value


def _expect_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SnapshotValidationError(f"{path} must be an integer")
    return value


def _expect_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotValidationError(f"{path} must be a number")
    result = float(value)
    if not isfinite(result):
        raise SnapshotValidationError(f"{path} must be finite")
    return result


def _require_non_empty_string(value: str, path: str) -> None:
    if not value.strip():
        raise SnapshotValidationError(f"{path} must not be empty")


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    items = _expect_list(value, path)
    return tuple(_expect_string(item, f"{path}[{index}]") for index, item in enumerate(items))
