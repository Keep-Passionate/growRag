"""Append-only registry that proves a card is backed by archived outcomes."""

from __future__ import annotations

from math import isclose

from growrag.episodes import (
    EpisodeAction,
    EpisodeArchive,
    VerificationEvent,
    VerificationSource,
    VerificationTarget,
    VerificationVerdict,
)
from growrag.experience.cards import ExperienceCard, VerificationTier


class ExperienceCardRegistry:
    """Validate provenance and sufficient statistics before a card can serve.

    ``ExperienceCard`` is a serializable record.  This registry is the trusted
    boundary: a card is not considered available to the controller merely
    because somebody managed to instantiate the dataclass.
    """

    def __init__(self, episode_archive: EpisodeArchive) -> None:
        self._episode_archive = episode_archive
        self._cards: dict[str, ExperienceCard] = {}

    @property
    def cards(self) -> tuple[ExperienceCard, ...]:
        return tuple(self._cards[key] for key in sorted(self._cards))

    def get(self, versioned_id: str) -> ExperienceCard:
        try:
            return self._cards[versioned_id]
        except KeyError:
            raise KeyError(versioned_id) from None

    def register(self, card: ExperienceCard) -> None:
        if card.versioned_id in self._cards:
            raise ValueError(f"experience card version already exists: {card.versioned_id}")
        missing_parents = sorted(set(card.parent_versioned_ids).difference(self._cards))
        if missing_parents:
            raise ValueError(f"card references unknown parent versions: {missing_parents}")

        source_episode_ids, source_document_ids = self._validate_source_turns(card)
        events = self._validate_statistics(card)
        beneficial_episode_ids = {
            event.episode_id for event in events if event.verdict is VerificationVerdict.PASS
        }
        if not source_episode_ids.issubset(beneficial_episode_ids):
            raise ValueError("every source episode must have verified paired benefit")
        if len(source_episode_ids) != card.provenance.independent_episode_count:
            raise ValueError("independent_episode_count does not match source provenance")
        if len(source_document_ids) != card.provenance.independent_document_count:
            raise ValueError("independent_document_count does not match source evidence")

        self._cards[card.versioned_id] = card

    def _validate_source_turns(self, card: ExperienceCard) -> tuple[set[str], set[str]]:
        episode_ids: set[str] = set()
        document_ids: set[str] = set()
        for ref in card.provenance.source_episode_turn_refs:
            episode = self._episode_archive.get_episode(ref.episode_id)
            turn = next((item for item in episode.turns if item.turn_id == ref.turn_id), None)
            if turn is None:
                raise ValueError(f"unknown source turn: {ref.key}")
            if turn.action is EpisodeAction.BASE:
                raise ValueError(f"card provenance must point to a repair turn: {ref.key}")
            episode_ids.add(episode.episode_id)
            document_ids.update(item.doc_id for item in turn.evidence_refs)
        if not document_ids:
            raise ValueError("card provenance must include retrieved evidence documents")
        return episode_ids, document_ids

    def _validate_statistics(self, card: ExperienceCard) -> tuple[VerificationEvent, ...]:
        try:
            events = tuple(
                self._episode_archive.get_verification(event_id)
                for event_id in card.validation.verification_event_ids
            )
        except KeyError as error:
            raise ValueError(
                f"card references unknown verification event: {error.args[0]}"
            ) from None
        if not events:
            raise ValueError("candidate cards require at least one paired verification event")
        if any(event.target is not VerificationTarget.PAIRED_BENEFIT for event in events):
            raise ValueError("card validation may only cite paired benefit events")
        if len({event.episode_id for event in events}) != len(events):
            raise ValueError("each matched trial must come from an independent treatment episode")
        if not _sources_support_tier(
            tuple(event.source for event in events),
            card.validation.verification_tier,
        ):
            raise ValueError("verification event sources do not support the declared tier")

        benefit_count = sum(event.verdict is VerificationVerdict.PASS for event in events)
        neutral_count = sum(event.verdict is VerificationVerdict.UNCERTAIN for event in events)
        harm_count = sum(event.verdict is VerificationVerdict.FAIL for event in events)
        direct_correct_trials = sum(event.baseline_is_correct is True for event in events)
        mean_gain = sum(event.paired_gain or 0.0 for event in events) / len(events)
        observed = (
            len(events),
            benefit_count,
            neutral_count,
            harm_count,
            direct_correct_trials,
        )
        declared = (
            card.validation.matched_trials,
            card.validation.benefit_count,
            card.validation.neutral_count,
            card.validation.harm_count,
            card.validation.direct_correct_trials,
        )
        if observed != declared:
            raise ValueError("card validation counts do not match archived paired events")
        if not isclose(mean_gain, card.validation.mean_gain, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("card mean_gain does not match archived paired events")
        return events


def _sources_support_tier(
    sources: tuple[VerificationSource, ...],
    tier: VerificationTier,
) -> bool:
    if tier is VerificationTier.GOLD:
        allowed = {VerificationSource.GOLD}
    elif tier is VerificationTier.HUMAN:
        allowed = {VerificationSource.GOLD, VerificationSource.HUMAN}
    elif tier is VerificationTier.MULTI_JUDGE:
        allowed = {
            VerificationSource.GOLD,
            VerificationSource.HUMAN,
            VerificationSource.JUDGE,
        }
    else:
        allowed = set(VerificationSource)
    return all(source in allowed for source in sources)
