"""Versioned, immutable records for one-query retrieval episodes.

These records store observable decisions, executed queries, evidence pointers,
and outcomes.  They intentionally do not store hidden chain-of-thought or copy
document bodies into procedural memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class EpisodeAction(StrEnum):
    """Retrieval action that was actually executed in one turn."""

    BASE = "base"
    REUSE_REPAIR = "reuse_repair"
    FRESH_REPAIR = "fresh_repair"


class NextAction(StrEnum):
    """Controller decision after observing a retrieval state."""

    ANSWER = "answer"
    REUSE_REPAIR = "reuse_repair"
    FRESH_REPAIR = "fresh_repair"
    STOP_ABSTAIN = "stop_abstain"


class Sufficiency(StrEnum):
    """Whether current evidence supports answering the original query."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


class GapCategory(StrEnum):
    """Small, auditable vocabulary for post-retrieval evidence gaps."""

    BRIDGE_ENTITY = "bridge_entity"
    ATTRIBUTE = "attribute"
    RELATION = "relation"
    EVIDENCE_SPAN = "evidence_span"
    CONTRADICTION = "contradiction"
    OTHER = "other"


class TerminalReason(StrEnum):
    """Why a completed episode stopped."""

    SUFFICIENT = "sufficient"
    BASELINE_COMPLETE = "baseline_complete"
    NO_PROGRESS = "no_progress"
    BUDGET = "budget"
    RISK_REJECT = "risk_reject"
    ERROR = "error"


class VerificationTarget(StrEnum):
    """What an appended verification event evaluates."""

    ANSWER = "answer"
    EVIDENCE = "evidence"
    PAIRED_BENEFIT = "paired_benefit"


class VerificationSource(StrEnum):
    """Strength and origin of delayed outcome feedback."""

    GOLD = "gold"
    HUMAN = "human"
    JUDGE = "judge"
    PROXY = "proxy"


class VerificationVerdict(StrEnum):
    """Discrete conclusion of a verification event."""

    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable pointer target for a frozen RAG environment.

    Episodes carry only ``run_context_id``.  This avoids repeating the full
    corpus/retriever/prompt configuration in every trajectory while keeping it
    replayable and auditable.
    """

    run_context_id: str
    created_at: str
    corpus_id: str
    corpus_version: str
    chunk_index_policy_id: str
    retriever_family: str
    retriever_id: str
    generator_id: str
    state_prompt_version: str
    repair_prompt_version: str
    judge_id: str
    budget_policy_id: str
    reranker_id: str | None = None
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text_fields(
            run_context_id=self.run_context_id,
            created_at=self.created_at,
            corpus_id=self.corpus_id,
            corpus_version=self.corpus_version,
            chunk_index_policy_id=self.chunk_index_policy_id,
            retriever_family=self.retriever_family,
            retriever_id=self.retriever_id,
            generator_id=self.generator_id,
            state_prompt_version=self.state_prompt_version,
            repair_prompt_version=self.repair_prompt_version,
            judge_id=self.judge_id,
            budget_policy_id=self.budget_policy_id,
        )
        if self.reranker_id is not None and not self.reranker_id.strip():
            raise ValueError("reranker_id must be non-empty when provided")
        object.__setattr__(
            self,
            "capabilities",
            _normalize_unique_text(self.capabilities, field_name="capabilities"),
        )


@dataclass(frozen=True, slots=True)
class GapItem:
    """One explicit missing evidence item in the current query state."""

    gap_id: str
    category: GapCategory
    target: str
    slot: str
    description: str

    def __post_init__(self) -> None:
        _require_text_fields(
            gap_id=self.gap_id,
            target=self.target,
            slot=self.slot,
            description=self.description,
        )
        object.__setattr__(self, "category", GapCategory(self.category))


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Pointer to retrieved evidence; document content is not duplicated."""

    doc_id: str
    unit_id: str
    rank: int
    retrieval_score: float
    content_hash: str

    def __post_init__(self) -> None:
        _require_text_fields(
            doc_id=self.doc_id,
            unit_id=self.unit_id,
            content_hash=self.content_hash,
        )
        if self.rank < 1:
            raise ValueError("rank must be at least 1")
        if not isfinite(self.retrieval_score):
            raise ValueError("retrieval_score must be finite")


@dataclass(frozen=True, slots=True)
class ProgressState:
    """Observable components used by no-progress stopping rules."""

    new_evidence_count: int
    gaps_closed: tuple[str, ...]
    duplicate_ratio: float

    def __post_init__(self) -> None:
        if self.new_evidence_count < 0:
            raise ValueError("new_evidence_count must be non-negative")
        if not isfinite(self.duplicate_ratio) or not 0.0 <= self.duplicate_ratio <= 1.0:
            raise ValueError("duplicate_ratio must be finite and in [0, 1]")
        object.__setattr__(
            self,
            "gaps_closed",
            _normalize_unique_text(self.gaps_closed, field_name="gaps_closed"),
        )


@dataclass(frozen=True, slots=True)
class QueryAction:
    """Concrete query plan that was executed, not merely proposed."""

    operator: str
    executed_queries: tuple[str, ...]
    target_gap_ids: tuple[str, ...] = ()
    experience_card_version: str | None = None

    def __post_init__(self) -> None:
        _require_text_fields(operator=self.operator)
        object.__setattr__(
            self,
            "executed_queries",
            _normalize_unique_text(
                self.executed_queries,
                field_name="executed_queries",
            ),
        )
        if not self.executed_queries:
            raise ValueError("executed_queries must contain at least one query")
        if len(self.executed_queries) != 1:
            raise ValueError("one episode turn must execute exactly one query")
        object.__setattr__(
            self,
            "target_gap_ids",
            _normalize_unique_text(self.target_gap_ids, field_name="target_gap_ids"),
        )
        if self.experience_card_version is not None and not self.experience_card_version.strip():
            raise ValueError("experience_card_version must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class RetrievalState:
    """Structured evidence state after a retrieval turn."""

    sufficiency: Sufficiency
    gaps: tuple[GapItem, ...]
    progress: ProgressState

    def __post_init__(self) -> None:
        object.__setattr__(self, "sufficiency", Sufficiency(self.sufficiency))
        gap_ids = [gap.gap_id for gap in self.gaps]
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("gap_id values must be unique within one retrieval state")
        if self.sufficiency is Sufficiency.SUFFICIENT and self.gaps:
            raise ValueError("sufficient states must not retain unresolved gaps")


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Controller output recorded after observing a state."""

    next_action: NextAction
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "next_action", NextAction(self.next_action))
        _require_text_fields(reason_code=self.reason_code)


@dataclass(frozen=True, slots=True)
class EpisodeCost:
    """Measured cost for one turn."""

    retrieval_calls: int
    input_tokens: int
    output_tokens: int
    latency_ms: float

    def __post_init__(self) -> None:
        for name, value in (
            ("retrieval_calls", self.retrieval_calls),
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("latency_ms must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class EpisodeTurn:
    """One executed query, observed evidence state, and next decision."""

    turn_id: str
    action: EpisodeAction
    query_action: QueryAction
    evidence_refs: tuple[EvidenceRef, ...]
    retrieval_state: RetrievalState
    decision: DecisionRecord
    cost: EpisodeCost

    def __post_init__(self) -> None:
        _require_text_fields(turn_id=self.turn_id)
        object.__setattr__(self, "action", EpisodeAction(self.action))
        if (
            self.action is EpisodeAction.REUSE_REPAIR
            and self.query_action.experience_card_version is None
        ):
            raise ValueError("REUSE_REPAIR turns must identify an experience card version")
        if (
            self.action is not EpisodeAction.REUSE_REPAIR
            and self.query_action.experience_card_version is not None
        ):
            raise ValueError("only REUSE_REPAIR turns may identify an experience card version")
        evidence_keys = [(item.doc_id, item.unit_id) for item in self.evidence_refs]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("evidence refs must be unique within one turn")


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Terminal, frozen outcome without embedding historical facts in a card."""

    terminal_reason: TerminalReason
    final_answer_ref: str | None = None
    final_answer_hash: str | None = None
    final_evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "terminal_reason", TerminalReason(self.terminal_reason))
        for name, value in (
            ("final_answer_ref", self.final_answer_ref),
            ("final_answer_hash", self.final_answer_hash),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{name} must be non-empty when provided")
        evidence_keys = [(item.doc_id, item.unit_id) for item in self.final_evidence_refs]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("final evidence refs must be unique")


@dataclass(frozen=True, slots=True)
class EpisodeTurnRef:
    """Typed, auditable pointer from a card to one concrete repair turn."""

    episode_id: str
    turn_id: str

    def __post_init__(self) -> None:
        _require_text_fields(episode_id=self.episode_id, turn_id=self.turn_id)

    @property
    def key(self) -> str:
        return f"{self.episode_id}#{self.turn_id}"


@dataclass(frozen=True, slots=True)
class FrozenQueryEpisode:
    """Append-only unit of one completed question-solving process."""

    episode_id: str
    run_context_id: str
    comparison_group_id: str
    original_query: str
    opened_at: str
    frozen_at: str
    turns: tuple[EpisodeTurn, ...]
    result: EpisodeResult
    document_session_id: str | None = None
    schema_version: str = "query_episode.v0"

    def __post_init__(self) -> None:
        _require_text_fields(
            episode_id=self.episode_id,
            run_context_id=self.run_context_id,
            comparison_group_id=self.comparison_group_id,
            original_query=self.original_query,
            opened_at=self.opened_at,
            frozen_at=self.frozen_at,
            schema_version=self.schema_version,
        )
        if self.document_session_id is not None and not self.document_session_id.strip():
            raise ValueError("document_session_id must be non-empty when provided")
        if not self.turns:
            raise ValueError("a frozen episode must contain at least one turn")
        turn_ids = [turn.turn_id for turn in self.turns]
        if len(turn_ids) != len(set(turn_ids)):
            raise ValueError("turn_id values must be unique within an episode")
        if self.turns[0].action is not EpisodeAction.BASE:
            raise ValueError("BASE-first episodes must start with a BASE turn")
        if any(turn.action is EpisodeAction.BASE for turn in self.turns[1:]):
            raise ValueError("BASE may only appear as the first episode turn")

        for index, turn in enumerate(self.turns):
            is_last = index == len(self.turns) - 1
            next_action = turn.decision.next_action

            if turn.action is EpisodeAction.BASE and turn.query_action.target_gap_ids:
                raise ValueError("BASE turns must not target a prior evidence gap")
            if index > 0:
                prior_gap_ids = {gap.gap_id for gap in self.turns[index - 1].retrieval_state.gaps}
                target_gap_ids = set(turn.query_action.target_gap_ids)
                if not target_gap_ids:
                    raise ValueError("repair turns must target at least one prior evidence gap")
                if not target_gap_ids.issubset(prior_gap_ids):
                    raise ValueError("repair turns may only target gaps from the prior state")

            if turn.retrieval_state.sufficiency is Sufficiency.SUFFICIENT:
                if next_action is not NextAction.ANSWER:
                    raise ValueError("sufficient evidence must transition to ANSWER")
            elif next_action is NextAction.ANSWER:
                is_forced_baseline = (
                    len(self.turns) == 1
                    and turn.action is EpisodeAction.BASE
                    and self.result.terminal_reason is TerminalReason.BASELINE_COMPLETE
                )
                if not is_forced_baseline:
                    raise ValueError("insufficient or unknown evidence cannot transition to ANSWER")

            if next_action in {NextAction.ANSWER, NextAction.STOP_ABSTAIN}:
                if not is_last:
                    raise ValueError("terminal controller actions must occur on the final turn")
            elif is_last:
                raise ValueError("the final turn must ANSWER or STOP_ABSTAIN")
            else:
                expected_action = EpisodeAction(next_action.value)
                if self.turns[index + 1].action is not expected_action:
                    raise ValueError("controller decision must match the next executed action")

        final_turn = self.turns[-1]
        if self.result.terminal_reason is TerminalReason.SUFFICIENT:
            if (
                final_turn.retrieval_state.sufficiency is not Sufficiency.SUFFICIENT
                or final_turn.decision.next_action is not NextAction.ANSWER
            ):
                raise ValueError("SUFFICIENT termination requires sufficient final evidence")
        elif self.result.terminal_reason is TerminalReason.BASELINE_COMPLETE:
            if not self.is_direct_baseline:
                raise ValueError("BASELINE_COMPLETE requires a one-turn BASE episode")
        elif final_turn.decision.next_action is not NextAction.STOP_ABSTAIN:
            raise ValueError("non-answer terminal reasons require STOP_ABSTAIN")

        retrieved_keys = {
            (item.doc_id, item.unit_id) for turn in self.turns for item in turn.evidence_refs
        }
        final_keys = {(item.doc_id, item.unit_id) for item in self.result.final_evidence_refs}
        if not final_keys.issubset(retrieved_keys):
            raise ValueError(
                "final evidence must be a subset of evidence retrieved in this episode"
            )

    @property
    def is_direct_baseline(self) -> bool:
        """Whether this is the untouched, one-turn BASE comparison arm."""

        return len(self.turns) == 1 and self.turns[0].action is EpisodeAction.BASE

    @property
    def contains_repair(self) -> bool:
        return any(turn.action is not EpisodeAction.BASE for turn in self.turns)


@dataclass(frozen=True, slots=True)
class VerificationEvent:
    """Delayed feedback appended without mutating the frozen episode."""

    verification_id: str
    episode_id: str
    target: VerificationTarget
    source: VerificationSource
    metric_id: str
    verdict: VerificationVerdict
    created_at: str
    score: float | None = None
    direct_baseline_episode_ref: str | None = None
    baseline_score: float | None = None
    treatment_score: float | None = None
    quality_threshold: float | None = None

    def __post_init__(self) -> None:
        _require_text_fields(
            verification_id=self.verification_id,
            episode_id=self.episode_id,
            metric_id=self.metric_id,
            created_at=self.created_at,
        )
        object.__setattr__(self, "target", VerificationTarget(self.target))
        object.__setattr__(self, "source", VerificationSource(self.source))
        object.__setattr__(self, "verdict", VerificationVerdict(self.verdict))
        if self.score is not None and not isfinite(self.score):
            raise ValueError("score must be finite when provided")
        for name, value in (
            ("baseline_score", self.baseline_score),
            ("treatment_score", self.treatment_score),
            ("quality_threshold", self.quality_threshold),
        ):
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite when provided")
        if self.direct_baseline_episode_ref is not None and not (
            self.direct_baseline_episode_ref.strip()
        ):
            raise ValueError("direct_baseline_episode_ref must be non-empty when provided")
        if (
            self.target is VerificationTarget.PAIRED_BENEFIT
            and self.direct_baseline_episode_ref is None
        ):
            raise ValueError("paired benefit verification requires a DIRECT baseline episode")
        if self.target is VerificationTarget.PAIRED_BENEFIT:
            if None in (self.baseline_score, self.treatment_score, self.quality_threshold):
                raise ValueError("paired benefit verification requires both scores and a threshold")
            if self.score is None or not _float_close(self.score, self.paired_gain):
                raise ValueError("paired benefit score must equal treatment_score - baseline_score")
            expected = self._expected_paired_verdict()
            if self.verdict is not expected:
                raise ValueError("paired benefit verdict is inconsistent with the paired scores")
        elif any(
            value is not None
            for value in (self.baseline_score, self.treatment_score, self.quality_threshold)
        ):
            raise ValueError("paired score fields are only valid for PAIRED_BENEFIT events")

    @property
    def paired_gain(self) -> float | None:
        if self.baseline_score is None or self.treatment_score is None:
            return None
        return self.treatment_score - self.baseline_score

    @property
    def baseline_is_correct(self) -> bool | None:
        if self.baseline_score is None or self.quality_threshold is None:
            return None
        return self.baseline_score >= self.quality_threshold

    def _expected_paired_verdict(self) -> VerificationVerdict:
        assert self.baseline_score is not None
        assert self.treatment_score is not None
        assert self.quality_threshold is not None
        if (
            self.baseline_score >= self.quality_threshold
            and self.treatment_score < self.quality_threshold
        ):
            return VerificationVerdict.FAIL
        if (
            self.treatment_score >= self.quality_threshold
            and self.treatment_score > self.baseline_score
        ):
            return VerificationVerdict.PASS
        return VerificationVerdict.UNCERTAIN


@dataclass(frozen=True, slots=True)
class DocumentSession:
    """Optional frozen grouping of independent single-query episodes."""

    session_id: str
    scope_ref: str
    scope_version: str
    episode_ids: tuple[str, ...]
    opened_at: str
    frozen_at: str
    schema_version: str = "document_session.v0"

    def __post_init__(self) -> None:
        _require_text_fields(
            session_id=self.session_id,
            scope_ref=self.scope_ref,
            scope_version=self.scope_version,
            opened_at=self.opened_at,
            frozen_at=self.frozen_at,
            schema_version=self.schema_version,
        )
        object.__setattr__(
            self,
            "episode_ids",
            _normalize_unique_text(self.episode_ids, field_name="episode_ids"),
        )
        if not self.episode_ids:
            raise ValueError("a frozen document session must contain at least one episode")


def _require_text_fields(**values: str) -> None:
    for name, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")


def _normalize_unique_text(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name} must be a tuple of strings")
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-empty string")
        cleaned = " ".join(value.strip().split())
        if cleaned in normalized:
            raise ValueError(f"duplicate value in {field_name}: {cleaned}")
        normalized.append(cleaned)
    return tuple(normalized)


def _float_close(left: float, right: float | None, *, tolerance: float = 1e-9) -> bool:
    return right is not None and abs(left - right) <= tolerance
