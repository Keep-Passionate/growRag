"""Compile a PRESELECTED card into a small diagnostic view, not a trust gate.

The old registry/ACTIVE policy is not weakened. A candidate may be inspected in
an explicitly configured experiment, but this view is not an online endorsement.
Natural-language applicability is passed through, not automatically verified.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from growrag.episodes import EpisodeAction, EpisodeArchive
from growrag.experience.cards import ActivationStage, CardLifecycle, ExperienceCard
from growrag.experiments.data_protocol import normalize_question
from growrag.experiments.protocol import Evidence, MemoryView, RuntimeQuestion
from growrag.query_operators import ExpansionBody, ParaphraseBody, RewriteForm


def _question_hash(text: str) -> str:
    return hashlib.sha256(normalize_question(text).encode("utf-8")).hexdigest()


def _unique_fields(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate typed card JSON field")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class CardMemoryView(MemoryView):
    """Runtime metadata retains ALL source questions; the body contains no scores."""

    source_query_ids: tuple[str, ...]
    source_question_hashes: tuple[str, ...]
    form: RewriteForm
    intent: str
    stage: ActivationStage
    requires_current_evidence: bool

    def __post_init__(self) -> None:
        MemoryView.__post_init__(self)
        for name in ("source_query_ids", "source_question_hashes"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or not values
                or any(not isinstance(v, str) or not v.strip() for v in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"{name} must contain unique nonempty strings")
        if self.source_query_id not in self.source_query_ids:
            raise ValueError("canonical source must belong to complete source IDs")
        if len(self.source_query_ids) != len(self.source_question_hashes):
            raise ValueError("source IDs and question hashes must have matching counts")
        if self.form not in (RewriteForm.PARAPHRASE, RewriteForm.EXPAND):
            raise ValueError("only paraphrase/expand card views are supported")
        if not isinstance(self.form, RewriteForm) or not isinstance(self.stage, ActivationStage):
            raise TypeError("view form and stage must be typed enums")
        if not isinstance(self.intent, str) or not self.intent.strip():
            raise ValueError("view intent is required")
        if type(self.requires_current_evidence) is not bool:
            raise TypeError("requires_current_evidence must be bool")
        if self.view_type != "typed_card_diagnostic_v1":
            raise ValueError("typed card view must identify its diagnostic protocol")
        if any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in self.source_question_hashes
        ):
            raise ValueError("source question hashes must be SHA-256 fingerprints")
        self._check_payload()

    def _check_payload(self) -> None:
        """Prevent dataclass replacement from silently changing the action metadata."""
        try:
            value = json.loads(self.text, object_pairs_hook=_unique_fields)
            if not isinstance(value, dict) or set(value) != {
                "schema",
                "form",
                "intent",
                "conditions",
                "body",
                "success_criterion",
            }:
                raise ValueError("unexpected view payload")
            if (
                value["schema"] != "growrag-query-card-view-v1"
                or value["form"] != self.form.value
                or value["intent"] != self.intent
                or value["conditions"]["stage"] != self.stage.value
            ):
                raise ValueError("view metadata does not match its payload")
            conditions = value["conditions"]
            if set(conditions) != {
                "stage",
                "query_pattern",
                "gap_pattern",
                "preconditions",
                "contraindications",
            }:
                raise ValueError("unexpected condition fields")
            for label in (value["success_criterion"], conditions["query_pattern"]):
                if not isinstance(label, str) or not label.strip():
                    raise ValueError("missing card description")
            if conditions["gap_pattern"] is not None and (
                not isinstance(conditions["gap_pattern"], str)
                or not conditions["gap_pattern"].strip()
            ):
                raise ValueError("gap must be text or null")
            for field in ("preconditions", "contraindications"):
                entries = conditions[field]
                if not isinstance(entries, list) or any(
                    not isinstance(entry, str) or not entry.strip() for entry in entries
                ):
                    raise ValueError("condition entries must be text lists")
            body_data = value["body"]
            if self.form is RewriteForm.PARAPHRASE:
                if not isinstance(body_data["preserve"], list):
                    raise ValueError("preserve must be a list in JSON")
                body = ParaphraseBody(**{**body_data, "preserve": tuple(body_data["preserve"])})
                requires_evidence = False
            else:
                if not isinstance(body_data["term_roles"], list):
                    raise ValueError("term_roles must be a list in JSON")
                body = ExpansionBody(**{**body_data, "term_roles": tuple(body_data["term_roles"])})
                requires_evidence = body.requires_current_evidence
            if requires_evidence != self.requires_current_evidence:
                raise ValueError("view evidence requirement does not match its body")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid or inconsistent typed card payload") from error

    def is_source(self, question: RuntimeQuestion) -> bool:
        return (
            question.question_id in self.source_query_ids
            or _question_hash(question.text) in self.source_question_hashes
        )

    def check_stage(self, *, after_retrieval: bool, evidence: tuple[Evidence, ...]) -> None:
        current = (
            ActivationStage.POST_RETRIEVAL if after_retrieval else ActivationStage.PRE_RETRIEVAL
        )
        if self.stage is not current:
            raise ValueError("card stage does not match the current retrieval stage")
        if self.requires_current_evidence and not evidence:
            raise ValueError("this action requires current evidence")


def card_to_view(
    card: ExperienceCard,
    *,
    archive: EpisodeArchive,
    source_questions: Mapping[str, RuntimeQuestion],
    available_capabilities: tuple[str, ...] = (),
) -> CardMemoryView:
    """Explicit diagnostic conversion; keys bind every source episode to its query.

    No gold, answer, source query text, validation score or document body is sent
    to the generator. Arbitrary procedural text still needs semantic auditing.
    """
    body = card.repair.action_body
    if body is None or card.schema_version != "experience_card.v1":
        raise ValueError("legacy cards require an explicit typed, revalidated revision")
    if card.lifecycle_state not in (CardLifecycle.CANDIDATE, CardLifecycle.ACTIVE):
        raise ValueError("quarantined or retired cards cannot enter this experiment")
    if not isinstance(available_capabilities, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in available_capabilities
    ):
        raise ValueError("available_capabilities must be an explicit tuple of names")
    missing = set(card.activation.required_retriever_capabilities) - set(available_capabilities)
    if missing:
        raise ValueError("required retriever capabilities are unavailable")
    refs = card.provenance.source_episode_turn_refs
    episode_ids = {ref.episode_id for ref in refs}
    if set(source_questions) != episode_ids:
        raise ValueError("bind exactly ALL source episodes to their runtime questions")
    if len(episode_ids) != card.provenance.independent_episode_count:
        raise ValueError("independent source episode count mismatch")
    questions = tuple(source_questions[key] for key in sorted(episode_ids))
    if not all(isinstance(question, RuntimeQuestion) for question in questions):
        raise TypeError("source bindings must contain gold-free RuntimeQuestion values")
    ids = tuple(question.question_id for question in questions)
    hashes = tuple(_question_hash(question.text) for question in questions)
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("independent episodes must not duplicate a source question")
    for ref in refs:
        episode = archive.get_episode(ref.episode_id)
        if _question_hash(episode.original_query) != _question_hash(
            source_questions[ref.episode_id].text
        ):
            raise ValueError("source question does not match the archived episode")
        turn = next((turn for turn in episode.turns if turn.turn_id == ref.turn_id), None)
        if turn is None or turn.action is EpisodeAction.BASE:
            raise ValueError("source must resolve to an archived repair turn")
    canonical = (card.provenance.canonical_example_refs or refs)[0]
    payload = {
        "schema": "growrag-query-card-view-v1",
        "form": body.form.value,
        "intent": card.repair.intent,
        "conditions": {
            "stage": card.activation.stage.value,
            "query_pattern": card.activation.query_pattern,
            "gap_pattern": card.activation.gap_pattern,
            "preconditions": card.activation.preconditions,
            "contraindications": card.activation.contraindications,
        },
        "body": asdict(body),
        "success_criterion": card.repair.evidence_contract,
    }
    return CardMemoryView(
        memory_id=card.versioned_id,
        source_query_id=source_questions[canonical.episode_id].question_id,
        source_step_id=canonical.key,
        view_type="typed_card_diagnostic_v1",
        text=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        source_query_ids=ids,
        source_question_hashes=hashes,
        form=body.form,
        intent=card.repair.intent,
        stage=card.activation.stage,
        requires_current_evidence=(
            isinstance(body, ExpansionBody) and body.requires_current_evidence
        ),
    )
