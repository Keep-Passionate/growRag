"""Deterministic, dependency-free demonstration of the trusted reuse gate."""

from __future__ import annotations

from growrag.experience import (
    EvidenceRole,
    ExperienceLedger,
    SourceEvidence,
    TransferObservation,
)
from growrag.models import EnvironmentFingerprint, QueryTransformation
from growrag.rewrite import PrecomputedPlanApplier
from growrag.selection import (
    GateCandidate,
    GateDecision,
    SignatureApplicabilityScorer,
    TrustedReuseGate,
)


def demo_environment() -> EnvironmentFingerprint:
    """Return the explicit environment fingerprint used by the demo."""

    return EnvironmentFingerprint(
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


def _transfer(target_id: str, direct: float, reuse: float) -> TransferObservation:
    return TransferObservation.from_scores(
        target_query_id=target_id,
        direct_score=direct,
        reuse_score=reuse,
        good_threshold=0.5,
        dataset_id="toy",
        split="calibration",
        target_group_id=f"group-{target_id}",
        evidence_role=EvidenceRole.CALIBRATION,
        metric_protocol_id="answer-f1-v1",
        applier_id="precomputed-plan",
        applier_version="v1",
    )


def build_demo_experience() -> QueryTransformation:
    """Return the source transformation used by demo and file fixtures."""

    return QueryTransformation(
        experience_id="comparison-decomposition-1",
        source_query_id="source-1",
        source_query="Who was born first, Alice or Bob?",
        transformed_query="Alice birth date; Bob birth date",
        atomic_units=("split comparison into two entity-attribute queries",),
        environment=demo_environment(),
        provenance="toy paired source evaluation",
        transformation_type="comparison_decomposition",
        applicability_signature=("comparison", "two_entity"),
        source_dataset_id="toy",
        source_split="development",
        source_group_id="source-family",
        diagnosed_failure=(
            "The comparison query did not expose each entity's birth-date attribute."
        ),
        gap_categories=("attribute", "relation"),
        contraindication_signature=("single_entity",),
    )


def build_demo_ledger() -> ExperienceLedger:
    """Return a ledger whose demo experience has independent ACTIVE evidence."""

    experience = build_demo_experience()
    ledger = ExperienceLedger()
    ledger.add_candidate(
        experience,
        SourceEvidence(
            direct_score=0.0,
            reuse_score=1.0,
            dataset_id="toy",
            split="development",
            source_group_id="source-family",
            evidence_role=EvidenceRole.DEVELOPMENT,
            metric_protocol_id="answer-f1-v1",
            applier_id="precomputed-plan",
            applier_version="v1",
        ),
    )
    ledger.record_transfer(experience.experience_id, _transfer("benefit-1", 0.0, 1.0))
    ledger.record_transfer(experience.experience_id, _transfer("safe-1", 0.8, 0.9))
    record = ledger.record_transfer(
        experience.experience_id,
        _transfer("safe-2", 0.7, 0.8),
    )
    assert record.experience_id == experience.experience_id
    return ledger


def build_demo_decision() -> GateDecision:
    """Build one ACTIVE experience and make a complete pre-retrieval decision."""

    environment = demo_environment()
    experience = build_demo_experience()
    ledger = build_demo_ledger()
    record = ledger.get(experience.experience_id)

    target_id = "target-1"
    target_query = "Who was born first, Carol or David?"
    prepared = PrecomputedPlanApplier(
        plans={(target_id, experience.experience_id): "Carol birth date; David birth date"},
        applier_id="precomputed-plan",
        application_version="v1",
    ).prepare(
        target_query_id=target_id,
        target_query=target_query,
        experience=experience,
    )
    fit = SignatureApplicabilityScorer().score(
        target_query,
        experience,
        current_signatures=frozenset({"comparison", "two_entity"}),
    )
    return TrustedReuseGate().decide(
        target_query_id=target_id,
        target_query=target_query,
        current_environment=environment,
        candidates=(
            GateCandidate(
                record=record,
                recall_rank=1,
                prepared=prepared,
                applicability=fit,
            ),
        ),
    )
