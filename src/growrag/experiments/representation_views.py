"""Read-only selection views over one frozen PRE paraphrase source per candidate.

This module does not extract, admit, trust, execute, or update experience. M2/M3/M5
are explicitly supplied text, not simulated model calls. Source observations,
inferred conditions, and unverified counterexamples must remain distinguishable
in that text; shape checks cannot prove its semantic correctness or remove a
leaked fact. Provenance is a caller declaration, not proof of an API invocation.

Audit dataclasses retain source records and canonical actions. Only the explicit
``selector_payload`` whitelist belongs in a selector request: never serialize the
audit dataclasses into that request. Lengths are characters, NOT tokenizer tokens.
The production CardMemoryView contract and source admission policy are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from growrag.experience.cards import ActivationStage
from growrag.experience.pre_sources import PreSourceRecord
from growrag.experience.query_views import CardMemoryView
from growrag.experiments.data_protocol import normalize_question
from growrag.experiments.protocol import Action, RuntimeQuestion
from growrag.query_operators import RewriteForm


class RepresentationKind(StrEnum):
    M1 = "M1"
    M2 = "M2"
    M3 = "M3"
    M5 = "M5"


_SUPPLIED_KINDS = frozenset(RepresentationKind) - {RepresentationKind.M1}


def _nonempty(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


@dataclass(frozen=True, slots=True)
class ViewProvenance:
    """Audit-only declarations; pre_generated requires a named model and prompt."""

    method: str
    model_id: str | None = None
    prompt_version: str | None = None
    reference: str | None = None

    def __post_init__(self) -> None:
        if self.method not in ("manual", "pre_generated", "mechanical"):
            raise ValueError("unsupported construction method")
        for name in ("model_id", "prompt_version", "reference"):
            value = getattr(self, name)
            if value is not None:
                _nonempty(value, name)
        if self.method == "pre_generated" and (
            self.model_id is None or self.prompt_version is None
        ):
            raise ValueError("pre_generated text needs model_id and prompt_version")
        if self.method == "mechanical" and self.model_id is not None:
            raise ValueError("mechanical projection does not invoke a model")


@dataclass(frozen=True, slots=True)
class RepresentationView:
    kind: RepresentationKind
    text: str
    provenance: ViewProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RepresentationKind):
            raise TypeError("kind must be RepresentationKind")
        _nonempty(self.text, "view text")
        if not isinstance(self.provenance, ViewProvenance):
            raise TypeError("provenance must be ViewProvenance")
        if (self.kind is RepresentationKind.M1) != (self.provenance.method == "mechanical"):
            raise ValueError("only M1 is a mechanical projection")

    @property
    def char_count(self) -> int:
        return len(self.text)


def _pair_text(source: PreSourceRecord) -> str:
    return json.dumps(
        {
            "source_question": source.question.text,
            "rewritten_query": source.fresh.state.rounds[0].search_query,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class RepresentationBundle:
    """Complete immutable four-view snapshot; source records are audit-only.

    The builder checks identity and actual changed PRE execution, not gold
    admission. The caller must first apply the existing source admission policy.
    A source success does not verify applicability or transfer to a target.
    """

    source: PreSourceRecord
    source_fingerprint: str
    canonical_action: CardMemoryView
    views: tuple[RepresentationView, ...]
    max_chars: int = 1200

    def __post_init__(self) -> None:
        if not isinstance(self.source, PreSourceRecord):
            raise TypeError("source must be PreSourceRecord")
        if self.source_fingerprint != self.source.fingerprint:
            raise ValueError("source observation fingerprint mismatch")
        if not isinstance(self.canonical_action, CardMemoryView):
            raise TypeError("canonical_action must be CardMemoryView")
        if type(self.max_chars) is not int or self.max_chars < 1:
            raise ValueError("max_chars must be a positive integer character limit")
        source, action = self.source, self.canonical_action
        for result in (source.base, source.fresh):
            if len(result.state.rounds) != 1 or any(
                event.status == "error" for event in result.events
            ):
                raise ValueError("source must contain completed independent PRE arms")
        row = source.fresh.state.rounds[0]
        if row.decision.action is not Action.FRESH or normalize_question(
            row.search_query
        ) == normalize_question(source.question.text):
            raise ValueError("source must contain an actual changed FRESH query")
        question_hash = hashlib.sha256(
            normalize_question(source.question.text).encode("utf-8")
        ).hexdigest()
        if (
            action.source_query_id != source.question.question_id
            or action.source_query_ids != (source.question.question_id,)
            or action.source_question_hashes != (question_hash,)
            or action.source_step_id != f"pre-source:{source.source_id}#FRESH"
        ):
            raise ValueError("canonical action must bind exactly this single source ID and hash")
        if (
            action.stage is not ActivationStage.PRE_RETRIEVAL
            or action.form is not RewriteForm.PARAPHRASE
            or action.requires_current_evidence
            or action.intent != source.intent
            or json.loads(action.text)["conditions"]["gap_pattern"] is not None
        ):
            raise ValueError("canonical action must be same-intent PRE paraphrase without a gap")
        if not isinstance(self.views, tuple) or any(
            not isinstance(view, RepresentationView) for view in self.views
        ):
            raise TypeError("views must be a tuple of RepresentationView values")
        if len(self.views) != len(RepresentationKind) or {v.kind for v in self.views} != set(
            RepresentationKind
        ):
            raise ValueError("exactly one of every M1/M2/M3/M5 view is required")
        if any(view.char_count > self.max_chars for view in self.views):
            raise ValueError("view exceeds max_chars; do not silently truncate or drop one group")
        if self.view(RepresentationKind.M1).text != _pair_text(source):
            raise ValueError("M1 must be the unchanged mechanical source query-pair projection")

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def source_question(self) -> RuntimeQuestion:
        return self.source.question

    @property
    def rewritten_query(self) -> str:
        return self.source.fresh.state.rounds[0].search_query

    def view(self, kind: RepresentationKind) -> RepresentationView:
        if not isinstance(kind, RepresentationKind):
            raise TypeError("kind must be RepresentationKind")
        return next(view for view in self.views if view.kind is kind)


def build_representation_bundle(
    source: PreSourceRecord,
    canonical_action: CardMemoryView,
    *,
    expected_source_fingerprint: str,
    texts: Mapping[RepresentationKind, str],
    provenance: Mapping[RepresentationKind, ViewProvenance],
    max_chars: int = 1200,
) -> RepresentationBundle:
    """Project M1 and accept explicitly prepared M2/M3/M5; performs no model call."""
    if not isinstance(source, PreSourceRecord):
        raise TypeError("source must be PreSourceRecord")
    for label, values in (("texts", texts), ("provenance", provenance)):
        if not isinstance(values, Mapping) or any(
            not isinstance(key, RepresentationKind) for key in values
        ):
            raise TypeError(f"{label} must be keyed by RepresentationKind")
        if set(values) != _SUPPLIED_KINDS:
            raise ValueError(f"{label} must contain exactly M2/M3/M5")
    if len(source.fresh.state.rounds) != 1:
        raise ValueError("source must contain a completed FRESH arm")
    views = (
        RepresentationView(
            RepresentationKind.M1,
            _pair_text(source),
            ViewProvenance("mechanical", prompt_version="mechanical-qpair-v1"),
        ),
        *(
            RepresentationView(kind, texts[kind], provenance[kind])
            for kind in RepresentationKind
            if kind is not RepresentationKind.M1
        ),
    )
    return RepresentationBundle(
        source, expected_source_fingerprint, canonical_action, views, max_chars
    )


def _validate_pool(bundles: tuple[RepresentationBundle, ...]) -> None:
    if not isinstance(bundles, tuple) or any(
        not isinstance(bundle, RepresentationBundle) for bundle in bundles
    ):
        raise TypeError("bundles must be a tuple of RepresentationBundle values")
    identities = (
        [bundle.source_id for bundle in bundles],
        [bundle.source_question.question_id for bundle in bundles],
        [normalize_question(bundle.source_question.text) for bundle in bundles],
        [bundle.canonical_action.memory_id for bundle in bundles],
    )
    if any(len(values) != len(set(values)) for values in identities):
        raise ValueError(
            "candidate pool must have unique sources, source questions, and memory IDs"
        )


@dataclass(frozen=True, slots=True)
class RepresentationCandidate:
    candidate_id: str
    bundle: RepresentationBundle
    score: float

    def __post_init__(self) -> None:
        if self.candidate_id not in ("c1", "c2", "c3"):
            raise ValueError("candidate_id must be a neutral c1/c2/c3 slot")
        if not isinstance(self.bundle, RepresentationBundle):
            raise TypeError("bundle must be RepresentationBundle")
        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
            or not 0 < self.score <= 1
        ):
            raise ValueError("candidate score must be a positive finite Jaccard score")

    @property
    def canonical_action(self) -> CardMemoryView:
        return self.bundle.canonical_action


@dataclass(frozen=True, slots=True)
class CandidateSet:
    question: RuntimeQuestion
    candidates: tuple[RepresentationCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.question, RuntimeQuestion):
            raise TypeError("question must be gold-free RuntimeQuestion")
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(candidate, RepresentationCandidate) for candidate in self.candidates
        ):
            raise TypeError("candidates must be a tuple of RepresentationCandidate values")
        if len(self.candidates) > 3 or tuple(c.candidate_id for c in self.candidates) != tuple(
            f"c{index + 1}" for index in range(len(self.candidates))
        ):
            raise ValueError("at most three consecutive neutral slots are required")
        _validate_pool(tuple(candidate.bundle for candidate in self.candidates))
        if any(c.canonical_action.is_source(self.question) for c in self.candidates):
            raise ValueError(
                "a source question or normalized duplicate cannot be its own candidate"
            )

    def selector_payload(self, kind: RepresentationKind) -> dict:
        """Only target question text and neutral IDs/view text; no audit metadata."""
        if not isinstance(kind, RepresentationKind):
            raise TypeError("kind must be RepresentationKind")
        return {
            "original_question": self.question.text,
            "candidates": [
                {"candidate_id": candidate.candidate_id, "text": candidate.bundle.view(kind).text}
                for candidate in self.candidates
            ],
        }

    def resolve(self, candidate_id: str) -> CardMemoryView | None:
        if not isinstance(candidate_id, str):
            raise TypeError("selection must be an exact candidate ID or FRESH")
        if candidate_id == "FRESH":
            return None
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate.canonical_action
        raise ValueError("selection is not an available candidate ID or FRESH")


def retrieve_candidates(
    question: RuntimeQuestion,
    bundles: tuple[RepresentationBundle, ...],
    *,
    limit: int = 3,
) -> CandidateSet:
    """One deterministic source-query Jaccard shortlist, NOT an applicability gate.

    Excludes self/normalized duplicates, keeps only positive matches, and breaks
    ties by source ID. The pool is independent of all four representation texts.
    """
    if not isinstance(question, RuntimeQuestion):
        raise TypeError("question must be gold-free RuntimeQuestion")
    if type(limit) is not int or not 1 <= limit <= 3:
        raise ValueError("limit must be an integer from 1 to 3")
    _validate_pool(bundles)
    wanted = set(normalize_question(question.text).split())
    ranked = []
    for bundle in bundles:
        if bundle.canonical_action.is_source(question):
            continue
        terms = set(normalize_question(bundle.source_question.text).split())
        union = wanted | terms
        score = len(wanted & terms) / len(union) if union else 0.0
        if score > 0:
            ranked.append((score, bundle.source_id, bundle))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return CandidateSet(
        question,
        tuple(
            RepresentationCandidate(f"c{index + 1}", bundle, score)
            for index, (score, _, bundle) in enumerate(ranked[:limit])
        ),
    )
