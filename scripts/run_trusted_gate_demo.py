"""Run the smallest complete DIRECT/REUSE decision without a retriever or LLM."""

from __future__ import annotations

import json
from dataclasses import asdict

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
    SignatureApplicabilityScorer,
    TrustedReuseGate,
)


def environment() -> EnvironmentFingerprint:
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


def transfer(target_id: str, direct: float, reuse: float) -> TransferObservation:
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


def main() -> int:
    experience = QueryTransformation(
        experience_id="comparison-decomposition-1",
        source_query_id="source-1",
        source_query="Who was born first, Alice or Bob?",
        transformed_query="Alice birth date; Bob birth date",
        atomic_units=("split comparison into two entity-attribute queries",),
        environment=environment(),
        provenance="toy paired source evaluation",
        transformation_type="comparison_decomposition",
        applicability_signature=("comparison", "two_entity"),
        source_dataset_id="toy",
        source_split="development",
        source_group_id="source-family",
    )
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
    ledger.record_transfer(experience.experience_id, transfer("benefit-1", 0.0, 1.0))
    ledger.record_transfer(experience.experience_id, transfer("safe-1", 0.8, 0.9))
    record = ledger.record_transfer(
        experience.experience_id,
        transfer("safe-2", 0.7, 0.8),
    )

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
    decision = TrustedReuseGate().decide(
        target_query_id=target_id,
        target_query=target_query,
        current_environment=environment(),
        candidates=(
            GateCandidate(
                record=record,
                recall_rank=1,
                prepared=prepared,
                applicability=fit,
            ),
        ),
    )
    print(json.dumps(asdict(decision), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
