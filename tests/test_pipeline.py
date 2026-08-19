from __future__ import annotations

from dataclasses import fields, replace

from growrag.experience import (
    EvidenceRole,
    ExperienceLedger,
    ExperienceRecord,
    SourceEvidence,
    TransferObservation,
)
from growrag.models import EnvironmentFingerprint, QueryTransformation
from growrag.pipeline import TrustedReuseLayer, TrustedReuseResult
from growrag.rewrite import PrecomputedPlanApplier
from growrag.selection import DecisionReason, DeploymentAction

ENVIRONMENT = EnvironmentFingerprint(
    corpus_id="toy-wiki",
    corpus_version="2026-08",
    retriever_id="bm25",
    retriever_version="1",
    rewriter_id="precomputed-plan",
    rewriter_version="v1",
    index_id="toy-index",
    index_version="sha256:demo",
    analyzer_id="unicode-words",
    analyzer_version="1",
)
TARGET_ID = "target-1"
TARGET_QUERY = "alpha beta"


def add_active_experience(
    ledger: ExperienceLedger,
    experience_id: str,
    *,
    source_query: str,
    signatures: tuple[str, ...] = ("intent",),
    environment: EnvironmentFingerprint = ENVIRONMENT,
) -> ExperienceRecord:
    transformation = QueryTransformation(
        experience_id=experience_id,
        source_query_id=f"source-{experience_id}",
        source_query=source_query,
        transformed_query=f"source plan for {experience_id}",
        atomic_units=("toy query operation",),
        environment=environment,
        provenance="development paired evaluation",
        transformation_type="toy_operation",
        applicability_signature=signatures,
        source_dataset_id="toy",
        source_split="development",
        source_group_id=f"source-group-{experience_id}",
    )
    added = ledger.add_candidate(
        transformation,
        SourceEvidence(
            direct_score=0.0,
            reuse_score=1.0,
            dataset_id="toy",
            split="development",
            source_group_id=f"source-group-{experience_id}",
            evidence_role=EvidenceRole.DEVELOPMENT,
            metric_protocol_id="toy-f1-v1",
            applier_id="precomputed-plan",
            applier_version="v1",
        ),
    )
    assert added is not None
    for suffix, direct_score, reuse_score in (
        ("benefit", 0.0, 1.0),
        ("safe-1", 0.8, 0.9),
        ("safe-2", 0.7, 0.8),
    ):
        record = ledger.record_transfer(
            experience_id,
            TransferObservation.from_scores(
                target_query_id=f"{experience_id}-{suffix}",
                direct_score=direct_score,
                reuse_score=reuse_score,
                good_threshold=0.5,
                dataset_id="toy",
                split="calibration",
                target_group_id=f"group-{experience_id}-{suffix}",
                evidence_role=EvidenceRole.CALIBRATION,
                metric_protocol_id="toy-f1-v1",
                applier_id="precomputed-plan",
                applier_version="v1",
            ),
        )
    return record


def layer(
    plans: dict[tuple[str, str], str],
    *,
    max_candidates: int = 5,
) -> TrustedReuseLayer:
    return TrustedReuseLayer(
        applier=PrecomputedPlanApplier(plans=plans),
        max_candidates=max_candidates,
    )


def decide(
    reuse_layer: TrustedReuseLayer,
    records: ExperienceLedger | tuple[ExperienceRecord, ...],
    *,
    signatures: frozenset[str] = frozenset({"intent"}),
):
    return reuse_layer.decide(
        records=records,
        target_query_id=TARGET_ID,
        target_query=TARGET_QUERY,
        current_environment=ENVIRONMENT,
        current_signatures=signatures,
    )


def test_pipeline_runs_complete_query_side_reuse_from_ledger() -> None:
    ledger = ExperienceLedger()
    add_active_experience(ledger, "exp-reuse", source_query="alpha beta source")

    result = decide(
        layer({(TARGET_ID, "exp-reuse"): "rewritten alpha beta"}),
        ledger,
    )

    assert result.decision.executed_action is DeploymentAction.REUSE
    assert result.decision.executed_experience_id == "exp-reuse"
    assert result.decision.executed_query == "rewritten alpha beta"
    assert result.recall_scorer_id == "source-query-jaccard-v1"
    assert result.max_candidates == 5
    assert result.eligible_experience_ids == ("exp-reuse",)
    assert result.candidate_experience_ids == ("exp-reuse",)
    forbidden = {"gold", "answer", "retrieval_results", "paired_outcome"}
    assert {field.name for field in fields(TrustedReuseResult)}.isdisjoint(forbidden)


def test_inactive_or_environment_mismatched_records_never_enter_recall() -> None:
    ledger = ExperienceLedger()
    mismatched = replace(ENVIRONMENT, index_version="sha256:different")
    record = add_active_experience(
        ledger,
        "wrong-environment",
        source_query="alpha beta",
        environment=mismatched,
    )

    result = decide(layer({}), (record,))

    assert result.decision.executed_action is DeploymentAction.DIRECT
    assert result.decision.reason is DecisionReason.NO_CANDIDATES
    assert result.eligible_experience_ids == ()
    assert result.candidate_experience_ids == ()


def test_missing_precomputed_plan_becomes_auditable_direct_fallback() -> None:
    ledger = ExperienceLedger()
    add_active_experience(ledger, "missing-plan", source_query="alpha beta")

    result = decide(layer({}), ledger)

    assert result.candidate_experience_ids == ("missing-plan",)
    assert result.decision.intended_action is DeploymentAction.REUSE
    assert result.decision.executed_action is DeploymentAction.DIRECT
    assert result.decision.reason is DecisionReason.INVALID_QUERY_PLAN_FALLBACK
    assert result.decision.traces[0].query_plan_valid is False


def test_top_m_recall_is_fixed_and_deterministic() -> None:
    ledger = ExperienceLedger()
    exact = add_active_experience(ledger, "exact", source_query="alpha beta")
    tie_b = add_active_experience(ledger, "tie-b", source_query="alpha")
    far = add_active_experience(ledger, "far", source_query="unrelated words")
    tie_a = add_active_experience(ledger, "tie-a", source_query="alpha")
    plans = {
        (TARGET_ID, record.experience_id): f"plan {record.experience_id}"
        for record in (tie_b, far, tie_a, exact)
    }

    result = decide(layer(plans, max_candidates=2), (tie_b, far, tie_a, exact))

    assert result.max_candidates == 2
    assert result.candidate_experience_ids == ("exact", "tie-a")
    assert result.decision.executed_experience_id == "exact"


def test_gate_can_choose_more_applicable_candidate_over_nearer_candidate() -> None:
    ledger = ExperienceLedger()
    nearer_lower_fit = add_active_experience(
        ledger,
        "nearer-lower-fit",
        source_query="alpha beta",
        signatures=("intent", "extra_condition"),
    )
    farther_higher_fit = add_active_experience(
        ledger,
        "farther-higher-fit",
        source_query="alpha",
        signatures=("intent",),
    )
    plans = {
        (TARGET_ID, nearer_lower_fit.experience_id): "nearer candidate plan",
        (TARGET_ID, farther_higher_fit.experience_id): "farther candidate plan",
    }

    result = decide(layer(plans, max_candidates=2), ledger)

    assert result.candidate_experience_ids == (
        "nearer-lower-fit",
        "farther-higher-fit",
    )
    assert result.decision.executed_experience_id == "farther-higher-fit"
    traces = {trace.experience_id: trace for trace in result.decision.traces}
    assert traces["nearer-lower-fit"].applicability_score == 0.5
    assert traces["farther-higher-fit"].applicability_score == 1.0
