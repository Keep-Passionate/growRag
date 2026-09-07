"""Conservative query-pattern candidates, not a learned or trusted reuse gate.

The first prototype recognizes explicit two-name age-comparison syntax only.
Name-like spans and ``who`` are heuristics, not verified entity/person recognition.
Scope is an explicit developer declaration, never inferred from card prose.
Changing ANY view field invalidates the declaration until explicitly re-reviewed.
No retrieval, model call, lifecycle promotion, probability or reliability update.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass

from growrag.experiments.data_protocol import normalize_question
from growrag.experiments.protocol import RuntimeQuestion
from growrag.query_operators import RewriteForm

from .cards import ActivationStage
from .query_views import CardMemoryView

SCOPE_VERSION = "growrag-declared-pattern-scope-v1"
MATCHER_VERSION = "growrag-two-name-age-pattern-v1"
_NEGATION = re.compile(r"\b(?:not|no|never|neither|nor|without)\b|n['’]t\b", re.I)
_OTHER_ATTRIBUTE = re.compile(
    r"\b(?:tall(?:er|est)?|short(?:er|est)?|height|heav(?:y|ier)|weight|rich(?:er|est)?|"
    r"wealth|faster|speed|profession|occupation|nationality|citizenship|died|death)\b",
    re.I,
)
_NONPERSON = re.compile(
    r"\b(?:book|novel|film|movie|portrait|painting|song|album|city|cities|country|"
    r"countries|building|bridge|company|university|software|version|release)\b",
    re.I,
)
_AMBIGUOUS = frozenset(
    "he she they it this that these those someone anyone everyone somebody anybody nobody "
    "person people persons man woman men women both either one other first second former "
    "latter author athlete brother sister his her their the".split()
)
_PARTICLES = frozenset("de del da di du van von der den la le".split())
_PATTERNS = (
    (
        "between_who",
        re.compile(
            r"between\s+(?:(?P<people>two (?:people|persons))\s+)?(?P<a>.+?)\s+and\s+"
            r"(?P<b>.+?),\s*who\s+is\s+(?P<comparison>older|younger)\??",
            re.I,
        ),
    ),
    (
        "who_or",
        re.compile(
            r"who\s+is\s+(?P<comparison>older|younger),\s*(?P<a>.+?)\s+or\s+(?P<b>.+?)\??", re.I
        ),
    ),
    (
        "born_than",
        re.compile(
            r"was\s+(?P<a>.+?)\s+born\s+(?P<comparison>earlier|later)\s+than\s+(?P<b>.+?)\??", re.I
        ),
    ),
)


def view_fingerprint(view: CardMemoryView) -> str:
    """Bind ID/version, exact prose, all source IDs/hashes, stage/form and metadata."""
    if not isinstance(view, CardMemoryView):
        raise TypeError("a typed CardMemoryView is required")
    encoded = json.dumps(asdict(view), ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ReviewedScope:
    """Legacy-friendly name; this prototype records provisional developer review only.

    A sidecar is not evidence that the user/human reviewed it or that its prose
    is semantically correct. Its note must explain the explicit scope decision.
    """

    view_fingerprint: str
    task_kind: str = "pairwise_comparison"
    attribute: str = "age"
    reviewer_note: str = ""
    review_origin: str = "developer_provisional"
    scope_version: str = SCOPE_VERSION
    matcher_version: str = MATCHER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.view_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.view_fingerprint
        ):
            raise ValueError("scope must bind a complete view SHA-256 fingerprint")
        for name in ("task_kind", "attribute", "reviewer_note"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be explicitly supplied")
        if self.review_origin != "developer_provisional" or self.scope_version != SCOPE_VERSION:
            raise ValueError("this prototype supports only its provisional developer scope")
        if self.matcher_version != MATCHER_VERSION:
            raise ValueError("matcher version changed; explicit scope review is required")


@dataclass(frozen=True, slots=True)
class QueryPattern:
    status: str
    task_kind: str | None = None
    attribute: str | None = None
    operands: tuple[str, ...] = ()
    signals: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


def _name_like(span: str) -> bool:
    words = span.split()
    if not 1 <= len(words) <= 6 or len(span) > 80 or not any(w[0].isupper() for w in words):
        return False
    for word in words:
        if word.casefold() in _AMBIGUOUS or word.casefold() in {
            "and",
            "or",
            "versus",
            "vs",
            "plus",
        }:
            return False
        letters = word.replace("-", "").replace("'", "").replace("’", "").replace(".", "")
        if not letters.isalpha() or (not word[0].isupper() and word.casefold() not in _PARTICLES):
            return False
    return True


def detect_query_pattern(question: RuntimeQuestion) -> QueryPattern:
    """UNKNOWN is intentional; unsupported wording must not be guessed into a task."""
    if not isinstance(question, RuntimeQuestion):
        raise TypeError("pattern detection requires a gold-free RuntimeQuestion")
    text = " ".join(question.text.split())
    for rejected, reason in (
        (len(text) > 500, "query_too_long"),
        (bool(_NEGATION.search(text)), "negation_or_exclusion"),
        (bool(_OTHER_ATTRIBUTE.search(text)), "other_or_mixed_attribute"),
        (bool(_NONPERSON.search(text)), "nonperson_or_ambiguous_object_type"),
    ):
        if rejected:
            return QueryPattern("UNKNOWN", reasons=(reason,))
    for label, pattern in _PATTERNS:
        match = pattern.fullmatch(text)
        if match is None:
            continue
        operands = tuple(match.group(name).strip() for name in ("a", "b"))
        if not all(_name_like(span) for span in operands):
            return QueryPattern("UNKNOWN", reasons=("not_exactly_two_unambiguous_name_spans",))
        if normalize_question(operands[0]) == normalize_question(operands[1]):
            return QueryPattern("UNKNOWN", reasons=("same_operand_twice",))
        return QueryPattern(
            "MATCH",
            "pairwise_comparison",
            "age",
            operands,
            (
                f"syntax:{label}",
                f"comparison:{match.group('comparison').lower()}",
                "two_distinct_name_like_spans",
                "entity_type_unverified",
                "heuristic_only",
            ),
        )
    return QueryPattern("UNKNOWN", reasons=("unsupported_or_ambiguous_syntax",))


@dataclass(frozen=True, slots=True)
class CandidateAudit:
    memory_id: str
    view_fingerprint: str
    scope_match: bool
    selected: bool
    signals: tuple[str, ...]
    reasons: tuple[str, ...]
    scope_status: str = "developer_declared_unvalidated"


@dataclass(frozen=True, slots=True)
class MatchResult:
    route: str
    pattern: QueryPattern
    candidates: tuple[CardMemoryView, ...]
    audits: tuple[CandidateAudit, ...]
    matcher_version: str = MATCHER_VERSION
    notice: str = (
        "Heuristic experimental candidate scope matching only, not trusted REUSE. "
        "No entity verification, semantic proof, probability or lifecycle change; "
        "free-text card preconditions remain unverified."
    )


def match_reviewed_candidates(
    question: RuntimeQuestion,
    candidates: tuple[tuple[CardMemoryView, ReviewedScope | None], ...],
    *,
    top_k: int = 3,
) -> MatchResult:
    """Return deterministic candidates or BASE; never create scope automatically."""
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(candidates, tuple) or any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not isinstance(pair[0], CardMemoryView)
        or (pair[1] is not None and not isinstance(pair[1], ReviewedScope))
        for pair in candidates
    ):
        raise TypeError(
            "candidates must be immutable (CardMemoryView, explicit scope or None) pairs"
        )
    pattern = detect_query_pattern(question)
    identifiers = Counter(view.memory_id for view, _ in candidates)
    ordered = sorted(candidates, key=lambda pair: (pair[0].memory_id, view_fingerprint(pair[0])))
    selected, audits = [], []
    for view, scope in ordered:
        fingerprint = view_fingerprint(view)
        reasons = []
        for failed, reason in (
            (identifiers[view.memory_id] > 1, "duplicate_memory_id"),
            (view.is_source(question), "same_source_question"),
            (view.stage is not ActivationStage.PRE_RETRIEVAL, "pre_retrieval_only"),
            (view.form is not RewriteForm.PARAPHRASE, "paraphrase_only"),
            (view.requires_current_evidence, "current_evidence_unavailable"),
            (pattern.status != "MATCH", "query_pattern_unknown"),
            (scope is None, "missing_explicit_scope"),
        ):
            if failed:
                reasons.append(reason)
        if scope is not None:
            if scope.view_fingerprint != fingerprint:
                reasons.append("view_changed_requires_scope_review")
            if (scope.task_kind, scope.attribute) != ("pairwise_comparison", "age"):
                reasons.append("unsupported_declared_scope")
        matched = not reasons
        chosen = matched and len(selected) < top_k
        if chosen:
            selected.append(view)
        if matched and not chosen:
            reasons.append("deterministic_top_k_limit")
        audits.append(
            CandidateAudit(
                view.memory_id,
                fingerprint,
                matched,
                chosen,
                ("explicit_scope_bound", "task_pattern_match", "no_entity_overlap_score")
                if matched
                else (),
                tuple(reasons),
                "developer_declared_unvalidated" if scope else "missing",
            )
        )
    return MatchResult("CANDIDATE" if selected else "BASE", pattern, tuple(selected), tuple(audits))
