from __future__ import annotations

from dataclasses import fields, replace
from inspect import signature

from growrag.experience import (
    EvidenceRole,
    ExperienceRecord,
    ExperienceState,
    SourceEvidence,
    TransferObservation,
)
from growrag.models import EnvironmentFingerprint, PreparedReuseCandidate, QueryTransformation
from growrag.selection import (
    ApplicabilityEstimate,
    CandidateGateTrace,
    DecisionReason,
    DeploymentAction,
    GateCandidate,
    GatePolicy,
    QueryBudget,
    TrustedReuseGate,
)

ENVIRONMENT = EnvironmentFingerprint(
    corpus_id="toy-wiki",
    corpus_version="2026-08",
    retriever_id="bm25",
    retriever_version="1",
    rewriter_id="source-writer",
    rewriter_version="1",
    index_id="toy-index",
    index_version="sha256:abc",
    analyzer_id="unicode-words",
    analyzer_version="1",
)


def observation(
    target_id: str,
    *,
    direct_score: float = 0.0,
    reuse_score: float = 1.0,
) -> TransferObservation:
    return TransferObservation.from_scores(
        target_query_id=target_id,
        direct_score=direct_score,
        reuse_score=reuse_score,
        good_threshold=0.5,
        dataset_id="toy",
        split="support",
        target_group_id=f"group-{target_id}",
        evidence_role=EvidenceRole.CALIBRATION,
        metric_protocol_id="toy-f1-v1",
        applier_id="deterministic-template",
        applier_version="1",
    )


def active_record(
    experience_id: str,
    *,
    observation_count: int = 3,
    environment: EnvironmentFingerprint = ENVIRONMENT,
    provenance: str = "support-split paired evaluation",
    state: ExperienceState = ExperienceState.ACTIVE,
) -> ExperienceRecord:
    transformation = QueryTransformation(
        experience_id=experience_id,
        source_query_id=f"source-{experience_id}",
        source_query="Which author wrote the source book?",
        transformed_query="source book author",
        atomic_units=("replace question phrase with author relation",),
        environment=environment,
        provenance=provenance,
        transformation_type="relation_expansion",
        applicability_signature=("author_question",),
    )
    return ExperienceRecord(
        transformation=transformation,
        state=state,
        source_evidence=SourceEvidence(
            direct_score=0.0,
            reuse_score=1.0,
            dataset_id="toy",
            split="development",
            source_group_id=f"source-group-{experience_id}",
            evidence_role=EvidenceRole.DEVELOPMENT,
            metric_protocol_id="toy-f1-v1",
            applier_id="deterministic-template",
            applier_version="1",
        ),
        transfer_observations=tuple(
            observation(
                f"support-{experience_id}-{index}",
                direct_score=0.0 if index == 0 else 0.8,
                reuse_score=1.0 if index == 0 else 0.9,
            )
            for index in range(observation_count)
        ),
    )


def prepared(
    experience_id: str,
    *,
    transformed_query: str = "target book author",
    requested_top_k: int = 5,
    context_token_budget: int = 512,
) -> PreparedReuseCandidate:
    return PreparedReuseCandidate(
        target_query_id="target-1",
        target_query="Who wrote the target book?",
        experience_id=experience_id,
        transformed_query=transformed_query,
        generated_by="deterministic-template",
        application_version="1",
        retrieval_query_count=1,
        requested_top_k=requested_top_k,
        context_token_budget=context_token_budget,
    )


def applicability(score: float) -> ApplicabilityEstimate:
    return ApplicabilityEstimate(
        score=score,
        matched_signatures=("author_question",),
        required_signatures=("author_question",),
        scorer_id="signature-coverage-v1",
    )


def candidate(
    experience_id: str = "exp-1",
    *,
    rank: int = 1,
    score: float = 1.0,
    record: ExperienceRecord | None = None,
    query_plan: PreparedReuseCandidate | None = None,
) -> GateCandidate:
    return GateCandidate(
        record=record or active_record(experience_id),
        recall_rank=rank,
        prepared=query_plan if query_plan is not None else prepared(experience_id),
        applicability=applicability(score),
    )


def decide(*candidates: GateCandidate, gate: TrustedReuseGate | None = None):
    return (gate or TrustedReuseGate()).decide(
        target_query_id="target-1",
        target_query="Who wrote the target book?",
        current_environment=ENVIRONMENT,
        candidates=tuple(candidates),
    )


def test_deployable_gate_only_exposes_direct_and_reuse() -> None:
    assert set(DeploymentAction) == {DeploymentAction.DIRECT, DeploymentAction.REUSE}


def test_valid_candidate_is_selected_for_reuse() -> None:
    decision = decide(candidate())

    assert decision.intended_action is DeploymentAction.REUSE
    assert decision.executed_action is DeploymentAction.REUSE
    assert decision.action is DeploymentAction.REUSE
    assert decision.intended_experience_id == "exp-1"
    assert decision.executed_experience_id == "exp-1"
    assert decision.executed_query == "target book author"
    assert decision.reason is DecisionReason.REUSE_SELECTED
    assert decision.traces[0].passed


def test_no_candidate_cleanly_bypasses_to_direct() -> None:
    decision = decide()

    assert decision.intended_action is DeploymentAction.DIRECT
    assert decision.executed_action is DeploymentAction.DIRECT
    assert decision.executed_query == "Who wrote the target book?"
    assert decision.reason is DecisionReason.NO_CANDIDATES


def test_gate_checks_active_state_before_later_failures() -> None:
    mismatched_environment = replace(ENVIRONMENT, index_version="sha256:different")
    inactive = active_record(
        "inactive",
        state=ExperienceState.CANDIDATE,
        environment=mismatched_environment,
        provenance="unspecified",
    )

    decision = decide(candidate("inactive", record=inactive))

    assert decision.action is DeploymentAction.DIRECT
    trace = decision.traces[0]
    assert trace.first_failure == "experience_not_active"
    assert trace.rejection_reasons[:3] == (
        "experience_not_active",
        "environment_mismatch",
        "missing_provenance",
    )


def test_unknown_or_mismatched_environment_cannot_reuse() -> None:
    unknown_environment = replace(ENVIRONMENT, index_id="unspecified")
    record = active_record("unknown-env", environment=unknown_environment)

    decision = decide(candidate("unknown-env", record=record))

    assert decision.action is DeploymentAction.DIRECT
    assert decision.traces[0].first_failure == "environment_mismatch"


def test_insufficient_reliability_cannot_be_rescued_by_high_applicability() -> None:
    weak_record = active_record("weak", observation_count=1)

    decision = decide(candidate("weak", record=weak_record, score=1.0))

    assert decision.action is DeploymentAction.DIRECT
    assert "insufficient_reliability_observations" in decision.traces[0].rejection_reasons


def test_low_applicability_keeps_the_original_query() -> None:
    decision = decide(candidate(score=0.49))

    assert decision.intended_action is DeploymentAction.DIRECT
    assert decision.executed_action is DeploymentAction.DIRECT
    assert decision.traces[0].first_failure == "applicability_below_threshold"


def test_invalid_applied_query_records_reuse_intent_but_safely_falls_back() -> None:
    unchanged = prepared("exp-1", transformed_query="Who wrote the target book?")

    decision = decide(candidate(query_plan=unchanged))

    assert decision.intended_action is DeploymentAction.REUSE
    assert decision.executed_action is DeploymentAction.DIRECT
    assert decision.intended_experience_id == "exp-1"
    assert decision.executed_experience_id is None
    assert decision.executed_query == "Who wrote the target book?"
    assert decision.reason is DecisionReason.INVALID_QUERY_PLAN_FALLBACK
    assert "unchanged_transformed_query" in decision.traces[0].rejection_reasons


def test_query_plan_over_budget_safely_falls_back_to_direct() -> None:
    too_expensive = prepared("exp-1", requested_top_k=11)
    gate = TrustedReuseGate(budget=QueryBudget(max_top_k=10))

    decision = decide(candidate(query_plan=too_expensive), gate=gate)

    assert decision.intended_action is DeploymentAction.REUSE
    assert decision.executed_action is DeploymentAction.DIRECT
    assert "top_k_budget_invalid" in decision.traces[0].rejection_reasons


def test_applied_query_must_use_the_applier_bound_to_reliability() -> None:
    wrong_applier = replace(
        prepared("exp-1"),
        generated_by="different-template",
        application_version="2",
    )

    decision = decide(candidate(query_plan=wrong_applier))

    assert decision.intended_action is DeploymentAction.REUSE
    assert decision.executed_action is DeploymentAction.DIRECT
    assert decision.reason is DecisionReason.INVALID_QUERY_PLAN_FALLBACK
    assert "application_applier_mismatch" in decision.traces[0].rejection_reasons
    assert "application_version_mismatch" in decision.traces[0].rejection_reasons


def test_candidate_choice_uses_frozen_lexicographic_order() -> None:
    high_fit_higher_risk = candidate(
        "higher-risk",
        rank=1,
        score=0.9,
        record=active_record("higher-risk", observation_count=3),
    )
    lower_fit = candidate(
        "lower-fit",
        rank=2,
        score=0.8,
        record=active_record("lower-fit", observation_count=11),
    )
    high_fit_lower_risk = candidate(
        "lower-risk",
        rank=3,
        score=0.9,
        record=active_record("lower-risk", observation_count=11),
    )

    decision = decide(lower_fit, high_fit_higher_risk, high_fit_lower_risk)

    assert decision.executed_experience_id == "lower-risk"
    traces = {trace.experience_id: trace for trace in decision.traces}
    assert traces["lower-risk"].risk_upper_bound < traces["higher-risk"].risk_upper_bound


def test_gate_contract_and_trace_have_no_outcome_or_gold_inputs() -> None:
    parameter_names = set(signature(TrustedReuseGate.decide).parameters)
    forbidden_parameters = {
        "gold",
        "answer",
        "retrieval_results",
        "direct_score",
        "reuse_score",
        "paired_outcome",
    }
    trace_field_names = {field.name for field in fields(CandidateGateTrace)}

    assert parameter_names.isdisjoint(forbidden_parameters)
    assert trace_field_names.isdisjoint(forbidden_parameters)


def test_policy_thresholds_are_frozen_inputs_not_target_outcomes() -> None:
    strict_gate = TrustedReuseGate(
        policy=GatePolicy(
            minimum_observations=4,
            minimum_benefits=1,
            maximum_risk_upper_bound=0.70,
            minimum_applicability=0.50,
        )
    )

    decision = decide(candidate(), gate=strict_gate)

    assert decision.action is DeploymentAction.DIRECT
    assert decision.traces[0].first_failure == "insufficient_reliability_observations"
