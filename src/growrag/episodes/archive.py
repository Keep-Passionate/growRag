"""Small append-only in-memory archive for immutable episode records."""

from __future__ import annotations

from growrag.episodes.models import (
    DocumentSession,
    FrozenQueryEpisode,
    RunContext,
    VerificationEvent,
    VerificationTarget,
)


class EpisodeArchive:
    """Register immutable contexts, episodes, sessions, and verification events.

    The archive deliberately exposes no update or delete operation.  Persistent
    snapshot I/O will be added after the schema and experiment protocol settle.
    """

    def __init__(self) -> None:
        self._contexts: dict[str, RunContext] = {}
        self._episodes: dict[str, FrozenQueryEpisode] = {}
        self._sessions: dict[str, DocumentSession] = {}
        self._verification_events: dict[str, VerificationEvent] = {}

    @property
    def contexts(self) -> tuple[RunContext, ...]:
        return tuple(self._contexts[key] for key in sorted(self._contexts))

    @property
    def episodes(self) -> tuple[FrozenQueryEpisode, ...]:
        return tuple(self._episodes[key] for key in sorted(self._episodes))

    @property
    def sessions(self) -> tuple[DocumentSession, ...]:
        return tuple(self._sessions[key] for key in sorted(self._sessions))

    @property
    def verification_events(self) -> tuple[VerificationEvent, ...]:
        return tuple(self._verification_events[key] for key in sorted(self._verification_events))

    def register_context(self, context: RunContext) -> None:
        if context.run_context_id in self._contexts:
            raise ValueError(f"run context already exists: {context.run_context_id}")
        self._contexts[context.run_context_id] = context

    def append_episode(self, episode: FrozenQueryEpisode) -> None:
        if episode.episode_id in self._episodes:
            raise ValueError(f"episode already exists: {episode.episode_id}")
        if episode.run_context_id not in self._contexts:
            raise ValueError(f"unknown run context: {episode.run_context_id}")
        self._episodes[episode.episode_id] = episode

    def append_verification(self, event: VerificationEvent) -> None:
        if event.verification_id in self._verification_events:
            raise ValueError(f"verification event already exists: {event.verification_id}")
        if event.episode_id not in self._episodes:
            raise ValueError(f"unknown episode: {event.episode_id}")
        baseline_ref = event.direct_baseline_episode_ref
        if baseline_ref is not None and baseline_ref not in self._episodes:
            raise ValueError(f"unknown DIRECT baseline episode: {baseline_ref}")
        if event.target is VerificationTarget.PAIRED_BENEFIT:
            self._validate_paired_verification(event)
        self._verification_events[event.verification_id] = event

    def append_session(self, session: DocumentSession) -> None:
        if session.session_id in self._sessions:
            raise ValueError(f"document session already exists: {session.session_id}")
        missing = sorted(set(session.episode_ids).difference(self._episodes))
        if missing:
            raise ValueError(f"document session references unknown episodes: {missing}")
        mismatched = sorted(
            episode_id
            for episode_id in session.episode_ids
            if self._episodes[episode_id].document_session_id != session.session_id
        )
        if mismatched:
            raise ValueError(f"document session does not match episode grouping for: {mismatched}")
        self._sessions[session.session_id] = session

    def verifications_for(self, episode_id: str) -> tuple[VerificationEvent, ...]:
        if episode_id not in self._episodes:
            raise KeyError(episode_id)
        return tuple(event for event in self.verification_events if event.episode_id == episode_id)

    def get_episode(self, episode_id: str) -> FrozenQueryEpisode:
        try:
            return self._episodes[episode_id]
        except KeyError:
            raise KeyError(episode_id) from None

    def get_verification(self, verification_id: str) -> VerificationEvent:
        try:
            return self._verification_events[verification_id]
        except KeyError:
            raise KeyError(verification_id) from None

    def _validate_paired_verification(self, event: VerificationEvent) -> None:
        baseline_ref = event.direct_baseline_episode_ref
        assert baseline_ref is not None
        if baseline_ref == event.episode_id:
            raise ValueError("paired verification cannot compare an episode with itself")

        treatment = self._episodes[event.episode_id]
        baseline = self._episodes[baseline_ref]
        if not baseline.is_direct_baseline:
            raise ValueError("paired verification baseline must be a one-turn BASE episode")
        if not treatment.contains_repair:
            raise ValueError("paired verification treatment must contain a repair turn")
        if treatment.original_query != baseline.original_query:
            raise ValueError("paired episodes must use the same original query")
        if treatment.run_context_id != baseline.run_context_id:
            raise ValueError("paired episodes must use the same frozen run context")
        if treatment.comparison_group_id != baseline.comparison_group_id:
            raise ValueError("paired episodes must share a comparison group")
