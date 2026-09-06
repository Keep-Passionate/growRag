from dataclasses import FrozenInstanceError, replace

import pytest

from growrag.episodes import (
    DecisionRecord,
    EpisodeAction,
    EpisodeArchive,
    EpisodeCost,
    EpisodeResult,
    EpisodeTurn,
    EpisodeTurnRef,
    EvidenceRef,
    FrozenQueryEpisode,
    GapCategory,
    GapItem,
    NextAction,
    ProgressState,
    QueryAction,
    RetrievalState,
    RunContext,
    Sufficiency,
    TerminalReason,
    VerificationEvent,
    VerificationSource,
    VerificationTarget,
    VerificationVerdict,
)
from growrag.experience import (
    ActivationStage,
    CardActivation,
    CardActivationPolicy,
    CardLifecycle,
    CardProvenance,
    CardServing,
    CardValidation,
    ExperienceCard,
    ExperienceCardRegistry,
    RepairSpecification,
    VerificationTier,
)


def _validation(**overrides: object) -> CardValidation:
    values: dict[str, object] = {
        "verification_tier": VerificationTier.GOLD,
        "verification_event_ids": ("verify-1", "verify-2"),
        "matched_trials": 2,
        "benefit_count": 2,
        "neutral_count": 0,
        "harm_count": 0,
        "direct_correct_trials": 2,
        "mean_gain": 0.4,
        "last_validated_at": "2026-08-22T12:00:00+08:00",
    }
    values.update(overrides)
    return CardValidation(**values)  # type: ignore[arg-type]


def _card(**overrides: object) -> ExperienceCard:
    refs = (
        EpisodeTurnRef("ep-treatment-1", "turn-2"),
        EpisodeTurnRef("ep-treatment-2", "turn-2"),
    )
    values: dict[str, object] = {
        "card_id": "card-bridge",
        "version": "v1",
        "lifecycle_state": CardLifecycle.ACTIVE,
        "created_at": "2026-08-22T12:00:00+08:00",
        "activation": CardActivation(
            stage=ActivationStage.POST_RETRIEVAL,
            query_pattern="two-hop person relation",
            gap_pattern="missing bridge entity",
            preconditions=("first-hop entity grounded",),
            contraindications=("entity remains ambiguous",),
            required_retriever_capabilities=("lexical_search",),
        ),
        "repair": RepairSpecification(
            operator_type="bridge_entity",
            slot_template="Search {known_entity} for relation {missing_relation}",
            evidence_contract="New evidence must connect the known and missing entities.",
        ),
        "provenance": CardProvenance(
            source_episode_turn_refs=refs,
            canonical_example_refs=(refs[0],),
            independent_episode_count=2,
            independent_document_count=2,
        ),
        "validation": _validation(),
        "serving": CardServing(expected_cost=1.0),
        "activation_policy": CardActivationPolicy(),
    }
    values.update(overrides)
    return ExperienceCard(**values)  # type: ignore[arg-type]


def test_active_card_keeps_recomputable_paired_statistics() -> None:
    card = _card()
    assert card.versioned_id == "card-bridge@v1"
    assert card.validation.conditional_harm_rate == 0.0
    assert card.activation_policy_failures() == ()
    with pytest.raises(FrozenInstanceError):
        card.version = "v2"  # type: ignore[misc]


def test_revised_card_is_new_unvalidated_candidate_and_preserves_parent() -> None:
    parent = _card()
    revised_scope = replace(parent.activation, preconditions=("current entity unambiguous",))
    child = parent.revised_candidate(
        version="v2",
        created_at="2026-09-06T12:00:00+08:00",
        activation=revised_scope,
        repair=replace(parent.repair, slot_template="Shorter rule"),
        expected_cost=0.5,
    )
    assert child.lifecycle_state is CardLifecycle.CANDIDATE
    assert child.validation.matched_trials == 0
    assert child.validation.last_validated_at is None
    assert child.validation.conditional_harm_rate is None
    assert child.parent_versioned_ids == (parent.versioned_id,)
    assert child.activation == revised_scope
    assert child.serving.use_count == 0
    assert parent.validation.matched_trials == 2
    assert parent.lifecycle_state is CardLifecycle.ACTIVE
    with pytest.raises(ValueError, match="ACTIVE"):
        replace(child, lifecycle_state=CardLifecycle.ACTIVE)


def test_revision_cannot_silently_replace_same_card_version() -> None:
    parent = _card()
    with pytest.raises(ValueError, match="new version"):
        parent.revised_candidate(
            version=parent.version,
            created_at=parent.created_at,
            activation=parent.activation,
            repair=parent.repair,
            expected_cost=1.0,
        )


def test_observed_validation_cannot_claim_unknown_validation_time() -> None:
    with pytest.raises(ValueError, match="require last_validated_at"):
        _validation(last_validated_at=None)


def test_card_validation_counts_and_event_refs_must_form_a_complete_partition() -> None:
    with pytest.raises(ValueError, match="must equal matched_trials"):
        _validation(matched_trials=3)
    with pytest.raises(ValueError, match="account for every matched trial"):
        _validation(
            verification_event_ids=("verify-1",),
            matched_trials=2,
        )


def test_active_card_requires_strong_cross_document_low_harm_evidence() -> None:
    proxy = _validation(verification_tier=VerificationTier.PROXY)
    with pytest.raises(ValueError, match="verification tier"):
        _card(validation=proxy)

    one_source = CardProvenance(
        source_episode_turn_refs=(EpisodeTurnRef("ep-treatment-1", "turn-2"),),
        canonical_example_refs=(EpisodeTurnRef("ep-treatment-1", "turn-2"),),
        independent_episode_count=1,
        independent_document_count=1,
    )
    with pytest.raises(ValueError, match="too few independent"):
        _card(provenance=one_source)

    harmful = _validation(
        benefit_count=1,
        neutral_count=0,
        harm_count=1,
        direct_correct_trials=2,
        mean_gain=0.1,
    )
    with pytest.raises(ValueError, match="conditional harm"):
        _card(validation=harmful)


def test_policy_threshold_is_explicit_and_can_be_relaxed_for_ablation() -> None:
    validation = _validation(
        benefit_count=1,
        neutral_count=0,
        harm_count=1,
        direct_correct_trials=2,
        mean_gain=0.1,
    )
    relaxed = CardActivationPolicy(
        min_benefit_count=1,
        max_conditional_harm_rate=0.5,
    )
    card = _card(validation=validation, activation_policy=relaxed)
    assert card.validation.conditional_harm_rate == 0.5


def test_candidate_can_exist_offline_but_needs_source_episode() -> None:
    validation = _validation(
        verification_event_ids=(),
        matched_trials=0,
        benefit_count=0,
        neutral_count=0,
        harm_count=0,
        direct_correct_trials=0,
        mean_gain=0.0,
    )
    candidate = _card(lifecycle_state=CardLifecycle.CANDIDATE, validation=validation)
    assert candidate.lifecycle_state is CardLifecycle.CANDIDATE

    with pytest.raises(ValueError, match="at least one source"):
        CardProvenance(
            source_episode_turn_refs=(),
            canonical_example_refs=(),
            independent_episode_count=1,
            independent_document_count=1,
        )


def test_canonical_examples_must_be_source_refs() -> None:
    with pytest.raises(ValueError, match="subset"):
        CardProvenance(
            source_episode_turn_refs=(EpisodeTurnRef("ep-1", "turn-2"),),
            canonical_example_refs=(EpisodeTurnRef("ep-2", "turn-2"),),
            independent_episode_count=1,
            independent_document_count=1,
        )


def _context() -> RunContext:
    return RunContext(
        run_context_id="ctx-1",
        created_at="2026-08-22T09:00:00+08:00",
        corpus_id="hotpotqa",
        corpus_version="v1",
        chunk_index_policy_id="paragraph-v1",
        retriever_family="bm25",
        retriever_id="bm25-v1",
        generator_id="reader-v1",
        state_prompt_version="state-v1",
        repair_prompt_version="repair-v1",
        judge_id="gold-v1",
        budget_policy_id="two-calls-v1",
    )


def _evidence(doc_id: str) -> EvidenceRef:
    return EvidenceRef(doc_id, "unit-1", 1, 1.0, f"sha256:{doc_id}")


def _gap(index: int) -> GapItem:
    return GapItem(
        f"gap-{index}",
        GapCategory.BRIDGE_ENTITY,
        "entity",
        "relation",
        "missing bridge relation",
    )


def _pair(index: int) -> tuple[FrozenQueryEpisode, FrozenQueryEpisode, VerificationEvent]:
    gap = _gap(index)
    base_evidence = _evidence(f"base-doc-{index}")
    repair_evidence = _evidence(f"repair-doc-{index}")
    base_turn = EpisodeTurn(
        f"base-turn-{index}",
        EpisodeAction.BASE,
        QueryAction("identity", (f"question {index}",)),
        (base_evidence,),
        RetrievalState(Sufficiency.INSUFFICIENT, (gap,), ProgressState(1, (), 0.0)),
        DecisionRecord(NextAction.ANSWER, "forced-baseline"),
        EpisodeCost(1, 10, 10, 1.0),
    )
    direct = FrozenQueryEpisode(
        episode_id=f"ep-direct-{index}",
        run_context_id="ctx-1",
        comparison_group_id=f"pair-{index}",
        original_query=f"question {index}",
        opened_at="2026-08-22T10:00:00+08:00",
        frozen_at="2026-08-22T10:00:01+08:00",
        turns=(base_turn,),
        result=EpisodeResult(
            TerminalReason.BASELINE_COMPLETE,
            final_answer_ref=f"answer-direct-{index}",
            final_answer_hash=f"sha256:direct-{index}",
            final_evidence_refs=(base_evidence,),
        ),
    )

    treatment_base = replace(
        base_turn,
        decision=DecisionRecord(NextAction.FRESH_REPAIR, "repair-gap"),
    )
    repair_turn = EpisodeTurn(
        "turn-2",
        EpisodeAction.FRESH_REPAIR,
        QueryAction(
            "bridge_entity",
            (f"repair question {index}",),
            target_gap_ids=(gap.gap_id,),
        ),
        (repair_evidence,),
        RetrievalState(Sufficiency.SUFFICIENT, (), ProgressState(1, (gap.gap_id,), 0.0)),
        DecisionRecord(NextAction.ANSWER, "sufficient"),
        EpisodeCost(1, 10, 10, 1.0),
    )
    treatment = FrozenQueryEpisode(
        episode_id=f"ep-treatment-{index}",
        run_context_id="ctx-1",
        comparison_group_id=f"pair-{index}",
        original_query=f"question {index}",
        opened_at="2026-08-22T10:00:00+08:00",
        frozen_at="2026-08-22T10:00:02+08:00",
        turns=(treatment_base, repair_turn),
        result=EpisodeResult(
            TerminalReason.SUFFICIENT,
            final_answer_ref=f"answer-treatment-{index}",
            final_answer_hash=f"sha256:treatment-{index}",
            final_evidence_refs=(repair_evidence,),
        ),
    )
    event = VerificationEvent(
        verification_id=f"verify-{index}",
        episode_id=treatment.episode_id,
        target=VerificationTarget.PAIRED_BENEFIT,
        source=VerificationSource.GOLD,
        metric_id="answer-f1-v1",
        score=0.4,
        verdict=VerificationVerdict.PASS,
        direct_baseline_episode_ref=direct.episode_id,
        baseline_score=0.6,
        treatment_score=1.0,
        quality_threshold=0.5,
        created_at="2026-08-22T11:00:00+08:00",
    )
    return direct, treatment, event


def _archive_with_two_beneficial_pairs() -> EpisodeArchive:
    archive = EpisodeArchive()
    archive.register_context(_context())
    for index in (1, 2):
        direct, treatment, event = _pair(index)
        archive.append_episode(direct)
        archive.append_episode(treatment)
        archive.append_verification(event)
    return archive


def test_registry_proves_card_counts_sources_and_document_independence() -> None:
    archive = _archive_with_two_beneficial_pairs()
    registry = ExperienceCardRegistry(archive)
    card = _card()
    registry.register(card)
    assert registry.cards == (card,)

    with pytest.raises(ValueError, match="already exists"):
        registry.register(card)


def test_registry_rejects_hand_written_statistics_and_unverified_sources() -> None:
    archive = _archive_with_two_beneficial_pairs()
    registry = ExperienceCardRegistry(archive)
    wrong_counts = _validation(
        benefit_count=1,
        neutral_count=1,
        harm_count=0,
        mean_gain=0.4,
    )
    candidate = _card(
        lifecycle_state=CardLifecycle.CANDIDATE,
        validation=wrong_counts,
    )
    with pytest.raises(ValueError, match="counts do not match"):
        registry.register(candidate)

    unknown_ref = EpisodeTurnRef("ep-unknown", "turn-2")
    bad_provenance = CardProvenance(
        source_episode_turn_refs=(
            unknown_ref,
            EpisodeTurnRef("ep-treatment-2", "turn-2"),
        ),
        canonical_example_refs=(unknown_ref,),
        independent_episode_count=2,
        independent_document_count=2,
    )
    with pytest.raises(KeyError, match="ep-unknown"):
        registry.register(_card(provenance=bad_provenance))
