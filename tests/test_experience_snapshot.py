from __future__ import annotations

import json

import pytest

from growrag.experience import (
    EvidenceRole,
    ExperienceLedger,
    ExperienceState,
    LifecyclePolicy,
    SnapshotValidationError,
    SourceEvidence,
    TransferObservation,
    dumps_snapshot,
    load_snapshot,
    loads_snapshot,
    save_snapshot,
)
from growrag.models import EnvironmentFingerprint, QueryTransformation


def add_candidate(ledger: ExperienceLedger, experience_id: str) -> None:
    transformation = QueryTransformation(
        experience_id=experience_id,
        source_query_id=f"source-{experience_id}",
        source_query="Who wrote the novel?",
        transformed_query="novel title author",
        atomic_units=("relation keyword", "named entity"),
        environment=EnvironmentFingerprint(
            corpus_id="corpus",
            corpus_version="2026-08",
            retriever_id="bm25",
            retriever_version="1",
            rewriter_id="prompt",
            rewriter_version="2",
            index_id="main",
            index_version="7",
            analyzer_id="english",
            analyzer_version="3",
        ),
        provenance="paired evaluation",
        transformation_type="relation_rewrite",
        application_note="apply only to authorship questions",
        applicability_signature=("authorship", "book"),
        source_dataset_id="toy",
        source_split="train",
        source_group_id=f"source-group-{experience_id}",
    )
    source = SourceEvidence(
        direct_score=0,
        reuse_score=1,
        dataset_id="toy",
        split="train",
        source_group_id=f"source-group-{experience_id}",
        evidence_role=EvidenceRole.DEVELOPMENT,
        metric_protocol_id="answer-em-v1",
        applier_id="literal-rewrite",
        applier_version="1",
    )
    ledger.add_candidate(transformation, source)


def observation(target_id: str, direct: float, reuse: float) -> TransferObservation:
    return TransferObservation(
        target_query_id=target_id,
        direct_score=direct,
        reuse_score=reuse,
        good_threshold=0.5,
        dataset_id="toy",
        split="validation",
        target_group_id=f"group-{target_id}",
        evidence_role=EvidenceRole.CALIBRATION,
        metric_protocol_id="answer-em-v1",
        applier_id="literal-rewrite",
        applier_version="1",
    )


def active_ledger() -> ExperienceLedger:
    policy = LifecyclePolicy(
        promotion_min_observations=3,
        promotion_min_direct_good_observations=2,
        promotion_min_benefits=1,
        promotion_min_mean_gain=0.05,
        promotion_max_wilson_breakage_upper_bound=0.70,
        quarantine_at_harm_count=1,
    )
    ledger = ExperienceLedger(policy=policy)
    add_candidate(ledger, "exp-1")
    ledger.record_transfer("exp-1", observation("benefit", 0.0, 1.0))
    ledger.record_transfer("exp-1", observation("safe-1", 0.8, 0.9))
    ledger.record_transfer("exp-1", observation("safe-2", 0.7, 0.8))
    assert ledger.get("exp-1").state is ExperienceState.ACTIVE
    return ledger


def parsed_snapshot(ledger: ExperienceLedger | None = None) -> dict[str, object]:
    return json.loads(dumps_snapshot(ledger or active_ledger(), memory_snapshot_id="memory-7"))


def test_roundtrip_preserves_complete_record_policy_and_identity() -> None:
    original = active_ledger()
    encoded = dumps_snapshot(original, memory_snapshot_id="memory-7")

    loaded = loads_snapshot(encoded)

    assert loaded.schema_version == 1
    assert loaded.memory_snapshot_id == "memory-7"
    assert loaded.ledger.policy == original.policy
    assert loaded.ledger.records == original.records
    assert loaded.ledger.get("exp-1").state is ExperienceState.ACTIVE
    assert dumps_snapshot(loaded.ledger, memory_snapshot_id=loaded.memory_snapshot_id) == encoded


def test_canonical_json_is_deterministic_across_insertion_order() -> None:
    first = ExperienceLedger()
    add_candidate(first, "b")
    add_candidate(first, "a")
    second = ExperienceLedger()
    add_candidate(second, "a")
    add_candidate(second, "b")

    first_json = dumps_snapshot(first, memory_snapshot_id="same")
    second_json = dumps_snapshot(second, memory_snapshot_id="same")

    assert first_json == second_json
    identifiers = [
        item["transformation"]["experience_id"] for item in json.loads(first_json)["records"]
    ]
    assert identifiers == [
        "a",
        "b",
    ]


def test_atomic_file_save_and_load(tmp_path) -> None:
    target = tmp_path / "nested" / "memory.json"

    returned = save_snapshot(target, active_ledger(), memory_snapshot_id="file-memory")
    loaded = load_snapshot(target)

    assert returned == target
    assert loaded.memory_snapshot_id == "file-memory"
    assert loaded.ledger.get("exp-1").state is ExperienceState.ACTIVE
    assert not list(target.parent.glob("*.tmp"))


@pytest.mark.parametrize("where", ["root", "environment", "observation"])
def test_unknown_fields_fail_closed(where: str) -> None:
    raw = parsed_snapshot()
    if where == "root":
        raw["future_field"] = True
    elif where == "environment":
        raw["records"][0]["transformation"]["environment"]["future_field"] = True
    else:
        raw["records"][0]["transfer_observations"][0]["future_field"] = True

    with pytest.raises(SnapshotValidationError, match="fields differ"):
        loads_snapshot(json.dumps(raw))


def test_unknown_schema_version_fails_closed() -> None:
    raw = parsed_snapshot()
    raw["schema_version"] = 2

    with pytest.raises(SnapshotValidationError, match="unsupported schema_version"):
        loads_snapshot(json.dumps(raw))


def test_duplicate_json_key_is_rejected() -> None:
    duplicated = '{"schema_version":1,"schema_version":1}'

    with pytest.raises(SnapshotValidationError, match="duplicate JSON key"):
        loads_snapshot(duplicated)


def test_test_role_cannot_be_smuggled_in_during_load() -> None:
    raw = parsed_snapshot()
    raw["records"][0]["transfer_observations"][0]["evidence_role"] = "test"

    with pytest.raises(SnapshotValidationError, match="test transfer evidence"):
        loads_snapshot(json.dumps(raw))


def test_derived_outcome_tampering_is_rejected() -> None:
    raw = parsed_snapshot()
    raw["records"][0]["transfer_observations"][0]["outcome"] = "direct_good_reuse_bad"

    with pytest.raises(SnapshotValidationError, match="inconsistent derived observation"):
        loads_snapshot(json.dumps(raw))


def test_lifecycle_state_is_preserved_but_unsafe_state_is_rejected() -> None:
    candidate = ExperienceLedger()
    add_candidate(candidate, "exp-1")
    candidate.record_transfer("exp-1", observation("benefit", 0.0, 1.0))
    encoded = dumps_snapshot(candidate, memory_snapshot_id="candidate-memory")

    loaded = loads_snapshot(encoded)
    assert loaded.ledger.get("exp-1").state is ExperienceState.CANDIDATE

    raw = json.loads(encoded)
    raw["records"][0]["state"] = "active"
    with pytest.raises(SnapshotValidationError, match="contradicts its ordered evidence"):
        loads_snapshot(json.dumps(raw))


def test_snapshot_contains_all_environment_and_evidence_fields() -> None:
    raw = parsed_snapshot()
    record = raw["records"][0]

    assert set(record["transformation"]["environment"]) == {
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
    assert "passes_paired_write_gate" in record["source_evidence"]
    assert "outcome" in record["transfer_observations"][0]
    assert set(raw["lifecycle_policy"]) == {
        "promotion_min_observations",
        "promotion_min_direct_good_observations",
        "promotion_min_benefits",
        "promotion_min_mean_gain",
        "promotion_max_wilson_breakage_upper_bound",
        "quarantine_at_harm_count",
    }
