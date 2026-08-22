from dataclasses import FrozenInstanceError, replace

import pytest

from growrag.episodes import (
    DecisionRecord,
    DocumentSession,
    EpisodeAction,
    EpisodeArchive,
    EpisodeCost,
    EpisodeResult,
    EpisodeTurn,
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


def _context(context_id: str = "ctx-1") -> RunContext:
    return RunContext(
        run_context_id=context_id,
        created_at="2026-08-22T09:00:00+08:00",
        corpus_id="hotpotqa",
        corpus_version="v1",
        chunk_index_policy_id="wiki-paragraph-v1",
        retriever_family="bm25",
        retriever_id="pyserini-bm25-v1",
        generator_id="frozen-reader-v1",
        state_prompt_version="state-v1",
        repair_prompt_version="repair-v1",
        judge_id="gold-answer-v1",
        budget_policy_id="two-retrieval-calls-v1",
        capabilities=("lexical_search",),
    )


def _evidence(doc_id: str = "doc-1", unit_id: str = "paragraph-2") -> EvidenceRef:
    return EvidenceRef(
        doc_id=doc_id,
        unit_id=unit_id,
        rank=1,
        retrieval_score=4.2,
        content_hash=f"sha256:{doc_id}:{unit_id}",
    )


def _gap() -> GapItem:
    return GapItem(
        gap_id="gap-bridge",
        category=GapCategory.BRIDGE_ENTITY,
        target="film",
        slot="director",
        description="The bridge entity connecting the film to its director is missing.",
    )


def _turn(
    turn_id: str,
    action: EpisodeAction,
    *,
    sufficiency: Sufficiency,
    next_action: NextAction,
    gaps: tuple[GapItem, ...] = (),
    evidence: tuple[EvidenceRef, ...] = (),
) -> EpisodeTurn:
    card_version = "card-bridge@v1" if action is EpisodeAction.REUSE_REPAIR else None
    target_gaps = ("gap-bridge",) if action is not EpisodeAction.BASE else ()
    executed_query = (
        "Who directed the film?" if action is EpisodeAction.BASE else "film bridge entity director"
    )
    return EpisodeTurn(
        turn_id=turn_id,
        action=action,
        query_action=QueryAction(
            operator="identity" if action is EpisodeAction.BASE else "bridge_entity",
            executed_queries=(executed_query,),
            target_gap_ids=target_gaps,
            experience_card_version=card_version,
        ),
        evidence_refs=evidence,
        retrieval_state=RetrievalState(
            sufficiency=sufficiency,
            gaps=gaps,
            progress=ProgressState(
                new_evidence_count=len(evidence),
                gaps_closed=("gap-bridge",) if sufficiency is Sufficiency.SUFFICIENT else (),
                duplicate_ratio=0.0,
            ),
        ),
        decision=DecisionRecord(next_action=next_action, reason_code="test-decision"),
        cost=EpisodeCost(1, 100, 20, 12.5),
    )


def _episode(
    episode_id: str,
    *,
    repair_action: EpisodeAction | None = None,
    forced_direct_baseline: bool = False,
    session_id: str | None = None,
    context_id: str = "ctx-1",
    comparison_group_id: str = "pair-1",
    original_query: str = "Who directed the film?",
) -> FrozenQueryEpisode:
    if repair_action is not None:
        base = _turn(
            "turn-1",
            EpisodeAction.BASE,
            sufficiency=Sufficiency.INSUFFICIENT,
            next_action=NextAction(repair_action.value),
            gaps=(_gap(),),
            evidence=(_evidence("doc-base"),),
        )
        repair = _turn(
            "turn-2",
            repair_action,
            sufficiency=Sufficiency.SUFFICIENT,
            next_action=NextAction.ANSWER,
            evidence=(_evidence("doc-repair"),),
        )
        turns = (base, repair)
        terminal_reason = TerminalReason.SUFFICIENT
        final_evidence = repair.evidence_refs
    else:
        sufficiency = Sufficiency.INSUFFICIENT if forced_direct_baseline else Sufficiency.SUFFICIENT
        base = _turn(
            "turn-1",
            EpisodeAction.BASE,
            sufficiency=sufficiency,
            next_action=NextAction.ANSWER,
            gaps=(_gap(),) if forced_direct_baseline else (),
            evidence=(_evidence("doc-base"),),
        )
        turns = (base,)
        terminal_reason = (
            TerminalReason.BASELINE_COMPLETE
            if forced_direct_baseline
            else TerminalReason.SUFFICIENT
        )
        final_evidence = base.evidence_refs

    return FrozenQueryEpisode(
        episode_id=episode_id,
        run_context_id=context_id,
        comparison_group_id=comparison_group_id,
        original_query=original_query,
        opened_at="2026-08-22T10:00:00+08:00",
        frozen_at="2026-08-22T10:00:01+08:00",
        turns=turns,
        result=EpisodeResult(
            terminal_reason=terminal_reason,
            final_answer_ref=f"answers/{episode_id}",
            final_answer_hash=f"sha256:{episode_id}",
            final_evidence_refs=final_evidence,
        ),
        document_session_id=session_id,
    )


def _paired_event(
    verification_id: str = "verify-1",
    *,
    episode_id: str = "ep-reuse",
    baseline_id: str = "ep-direct",
    baseline_score: float = 0.0,
    treatment_score: float = 1.0,
) -> VerificationEvent:
    return VerificationEvent(
        verification_id=verification_id,
        episode_id=episode_id,
        target=VerificationTarget.PAIRED_BENEFIT,
        source=VerificationSource.GOLD,
        metric_id="answer-em-v1",
        score=treatment_score - baseline_score,
        baseline_score=baseline_score,
        treatment_score=treatment_score,
        quality_threshold=0.5,
        verdict=VerificationVerdict.PASS,
        direct_baseline_episode_ref=baseline_id,
        created_at="2026-08-22T10:05:00+08:00",
    )


def test_archive_is_append_only_and_supports_delayed_paired_verification() -> None:
    archive = EpisodeArchive()
    archive.register_context(_context())
    direct = _episode("ep-direct", forced_direct_baseline=True)
    reuse = _episode("ep-reuse", repair_action=EpisodeAction.REUSE_REPAIR)
    archive.append_episode(direct)
    archive.append_episode(reuse)
    event = _paired_event()
    archive.append_verification(event)

    assert archive.episodes == (direct, reuse)
    assert archive.verifications_for("ep-reuse") == (event,)
    with pytest.raises(ValueError, match="episode already exists"):
        archive.append_episode(reuse)
    with pytest.raises(ValueError, match="verification event already exists"):
        archive.append_verification(event)


def test_episode_is_frozen_but_failure_can_still_be_archived() -> None:
    episode = _episode("ep-failure")
    with pytest.raises(FrozenInstanceError):
        episode.original_query = "mutated"  # type: ignore[misc]

    archive = EpisodeArchive()
    archive.register_context(_context())
    archive.append_episode(episode)
    archive.append_verification(
        VerificationEvent(
            verification_id="verify-fail",
            episode_id="ep-failure",
            target=VerificationTarget.ANSWER,
            source=VerificationSource.GOLD,
            metric_id="answer-em-v1",
            score=0.0,
            verdict=VerificationVerdict.FAIL,
            created_at="2026-08-22T10:05:00+08:00",
        )
    )
    assert archive.verifications_for("ep-failure")[0].verdict is VerificationVerdict.FAIL


def test_paired_verification_requires_scores_and_registered_direct_baseline() -> None:
    with pytest.raises(ValueError, match="requires a DIRECT baseline"):
        VerificationEvent(
            verification_id="verify-1",
            episode_id="ep-reuse",
            target=VerificationTarget.PAIRED_BENEFIT,
            source=VerificationSource.GOLD,
            metric_id="answer-em-v1",
            score=1.0,
            baseline_score=0.0,
            treatment_score=1.0,
            quality_threshold=0.5,
            verdict=VerificationVerdict.PASS,
            created_at="2026-08-22T10:05:00+08:00",
        )


def test_paired_verification_rejects_mismatched_query_context_and_self_reference() -> None:
    archive = EpisodeArchive()
    archive.register_context(_context())
    archive.register_context(_context("ctx-2"))
    direct = _episode("ep-direct", forced_direct_baseline=True)
    archive.append_episode(direct)

    wrong_query = _episode(
        "ep-wrong-query",
        repair_action=EpisodeAction.FRESH_REPAIR,
        original_query="A different question?",
    )
    archive.append_episode(wrong_query)
    with pytest.raises(ValueError, match="same original query"):
        archive.append_verification(
            _paired_event(episode_id="ep-wrong-query", baseline_id="ep-direct")
        )

    wrong_context = _episode(
        "ep-wrong-context",
        repair_action=EpisodeAction.FRESH_REPAIR,
        context_id="ctx-2",
    )
    archive.append_episode(wrong_context)
    with pytest.raises(ValueError, match="same frozen run context"):
        archive.append_verification(
            _paired_event(episode_id="ep-wrong-context", baseline_id="ep-direct")
        )

    with pytest.raises(ValueError, match="cannot compare an episode with itself"):
        archive.append_verification(_paired_event(episode_id="ep-direct", baseline_id="ep-direct"))


def test_episode_enforces_base_first_and_controller_transition() -> None:
    valid = _episode("ep-valid", repair_action=EpisodeAction.FRESH_REPAIR)
    with pytest.raises(ValueError, match="must start with a BASE"):
        replace(valid, turns=(valid.turns[1],))

    mismatched_base = replace(
        valid.turns[0],
        decision=DecisionRecord(NextAction.REUSE_REPAIR, "wrong-next-action"),
    )
    with pytest.raises(ValueError, match="must match the next executed action"):
        replace(valid, turns=(mismatched_base, valid.turns[1]))


def test_episode_restricts_one_query_per_turn_and_final_evidence_lineage() -> None:
    with pytest.raises(ValueError, match="exactly one query"):
        QueryAction(operator="multi-query", executed_queries=("q1", "q2"))

    episode = _episode("ep-valid", repair_action=EpisodeAction.FRESH_REPAIR)
    bad_result = replace(
        episode.result,
        final_evidence_refs=(_evidence("never-retrieved"),),
    )
    with pytest.raises(ValueError, match="subset of evidence retrieved"):
        replace(episode, result=bad_result)


def test_sufficient_state_cannot_keep_unresolved_gaps() -> None:
    with pytest.raises(ValueError, match="must not retain unresolved gaps"):
        RetrievalState(
            sufficiency=Sufficiency.SUFFICIENT,
            gaps=(_gap(),),
            progress=ProgressState(0, (), 1.0),
        )


def test_reuse_turn_requires_versioned_card_and_base_cannot_claim_one() -> None:
    reuse = _turn(
        "turn-2",
        EpisodeAction.REUSE_REPAIR,
        sufficiency=Sufficiency.SUFFICIENT,
        next_action=NextAction.ANSWER,
        evidence=(_evidence(),),
    )
    with pytest.raises(ValueError, match="must identify an experience card"):
        replace(reuse, query_action=replace(reuse.query_action, experience_card_version=None))

    base = _turn(
        "turn-1",
        EpisodeAction.BASE,
        sufficiency=Sufficiency.SUFFICIENT,
        next_action=NextAction.ANSWER,
        evidence=(_evidence(),),
    )
    with pytest.raises(ValueError, match="only REUSE_REPAIR"):
        replace(base, query_action=replace(base.query_action, experience_card_version="card@v1"))


def test_document_session_only_groups_matching_frozen_episodes() -> None:
    archive = EpisodeArchive()
    archive.register_context(_context())
    episode = _episode("ep-1", session_id="session-1")
    archive.append_episode(episode)
    session = DocumentSession(
        session_id="session-1",
        scope_ref="paper-1",
        scope_version="sha256:paper",
        episode_ids=("ep-1",),
        opened_at="2026-08-22T09:00:00+08:00",
        frozen_at="2026-08-22T11:00:00+08:00",
    )
    archive.append_session(session)
    assert archive.sessions == (session,)

    with pytest.raises(ValueError, match="does not match episode grouping"):
        archive.append_session(
            DocumentSession(
                session_id="session-2",
                scope_ref="paper-1",
                scope_version="sha256:paper",
                episode_ids=("ep-1",),
                opened_at="2026-08-22T09:00:00+08:00",
                frozen_at="2026-08-22T11:00:00+08:00",
            )
        )
