"""Independent empty-prefix PRE source observations and diagnostic candidate cards.

BASE and FRESH are sibling executions. They are never coerced into a fictional
BASE-first episode. Source gold is read only by the offline admission functions;
the runtime view compiler accepts no gold and confers no serving trust.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from growrag.experience.cards import (
    ActivationStage,
    CardActivation,
    CardLifecycle,
    CardServing,
    CardValidation,
    ExperienceCard,
    PreSourceMetrics,
    PreSourceProvenance,
    PreSourceRef,
    RepairSpecification,
    VerificationTier,
)
from growrag.experience.query_views import CardMemoryView, _question_hash
from growrag.experiments.data_protocol import normalize_question
from growrag.experiments.hotpot import LayeredFeedback, evaluate_layered_feedback
from growrag.experiments.protocol import Action, ExecutionKind, GoldRecord, RuntimeQuestion
from growrag.query_operators import ParaphraseBody, RewriteForm

if TYPE_CHECKING:
    from growrag.outer_loop import LoopResult

SOURCE_SCHEMA_VERSION = "pre_source_pair.v1"


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


@dataclass(frozen=True, slots=True)
class PreSourceRecord:
    source_id: str
    question: RuntimeQuestion
    base: LoopResult
    fresh: LoopResult
    form: RewriteForm
    intent: str
    rag_fingerprint: str
    generator_fingerprint: str
    execution_kind: ExecutionKind
    schema_version: str = SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Local import avoids outer_loop -> query_views -> experience cycles.
        from growrag.outer_loop import LoopResult

        for name in ("source_id", "intent", "rag_fingerprint", "generator_fingerprint"):
            _text(getattr(self, name), name)
        if self.schema_version != SOURCE_SCHEMA_VERSION:
            raise ValueError("unsupported PRE source schema")
        if not isinstance(self.question, RuntimeQuestion):
            raise TypeError("source question must be gold-free RuntimeQuestion")
        if self.form is not RewriteForm.PARAPHRASE:
            raise ValueError("this source protocol supports paraphrase only")
        if not isinstance(self.execution_kind, ExecutionKind):
            raise TypeError("source execution provenance must be explicit")
        for action, result in ((Action.BASE, self.base), (Action.FRESH, self.fresh)):
            if not isinstance(result, LoopResult) or result.state.question != self.question:
                raise ValueError("source arms must belong to the same original question")
            if len(result.state.rounds) > 1:
                raise ValueError("PRE source arms start independently and allow one RAG call")
            operations = tuple(event.operation for event in result.events)
            expected = ("rag",) if action is Action.BASE else ("rewrite", "rag")
            if not operations or operations != expected[: len(operations)]:
                raise ValueError("source must have the PRE empty-prefix call sequence")
            if any(
                event.execution_kind is not self.execution_kind
                for event in (*result.events, *result.component_events)
            ):
                raise ValueError("source execution provenance is mixed")
            allowed = {"local_compute", "composed_rag"}
            allowed.add("mock" if self.execution_kind is ExecutionKind.MOCK else "live_api")
            if any(
                event.status != "error" and event.transport_source not in allowed
                for event in (*result.events, *result.component_events)
            ):
                raise ValueError("source contains unverified transport provenance")
            errors = [index for index, event in enumerate(result.events) if event.status == "error"]
            if errors and errors != [len(result.events) - 1]:
                raise ValueError("failed source calls cannot be followed by more calls")
            if result.state.rounds:
                if operations != expected or errors:
                    raise ValueError("completed source arm must retain its successful calls")
                if result.stop_reason not in ("unassessed", "evidence_unavailable"):
                    raise ValueError("source has a non-PRE or assessed stopping state")
                row = result.state.rounds[0]
                if row.decision.memory is not None:
                    raise ValueError("source generation must not reuse memory")
                if row.feedback.sufficient is not None or row.feedback.useful_gain is not None:
                    raise ValueError("source is an unassessed PRE observation, not a repair loop")
                if result.state.observed_evidence != (row.reply.evidence or ()):
                    raise ValueError("source observation must equal its own reader evidence")
                if action is Action.BASE:
                    if (
                        row.decision.action is not Action.BASE
                        or row.search_query != self.question.text
                    ):
                        raise ValueError("source BASE must use the original query")
                elif row.decision.action is Action.FRESH:
                    if row.decision.form is not self.form or row.decision.intent != self.intent:
                        raise ValueError("source FRESH must follow the declared form and intent")
                elif row.decision.action is not Action.BASE or normalize_question(
                    row.search_query
                ) != normalize_question(self.question.text):
                    raise ValueError(
                        "source FRESH may only become BASE when its query is unchanged"
                    )
            elif result.state.observed_evidence:
                raise ValueError("failed PRE source cannot have unrecorded prefix evidence")

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {**asdict(self), "source_fingerprint": self.fingerprint}


@dataclass(frozen=True, slots=True)
class PreSourceAdmission:
    source_id: str
    source_fingerprint: str
    eligible: bool
    reasons: tuple[str, ...]
    metrics: PreSourceMetrics | None
    base_feedback: LayeredFeedback | None
    fresh_feedback: LayeredFeedback | None


def evaluate_pre_source(source: PreSourceRecord, gold: GoldRecord) -> PreSourceAdmission:
    """Source-only offline feedback; it never becomes a target matched trial."""
    if not isinstance(source, PreSourceRecord):
        raise TypeError("a typed PRE source record is required")
    if not isinstance(gold, GoldRecord) or gold.question_id != source.question.question_id:
        raise ValueError("gold must belong to this source question")
    reasons: list[str] = []
    feedback: list[LayeredFeedback | None] = []
    for label, result in (("BASE", source.base), ("FRESH", source.fresh)):
        if not result.state.rounds or any(event.status == "error" for event in result.events):
            reasons.append(f"{label}_not_completed")
            feedback.append(None)
        elif result.state.rounds[0].reply.evidence is None:
            reasons.append(f"{label}_evidence_unavailable")
            feedback.append(None)
        else:
            reply = result.state.rounds[0].reply
            feedback.append(
                evaluate_layered_feedback(source.question, (), reply.evidence, reply.answer, gold)
            )
    base, fresh = feedback
    metrics = None
    if not gold.supporting_facts:
        reasons.append("annotated_support_unavailable")
    if base is not None and fresh is not None and gold.supporting_facts:
        changed = normalize_question(
            source.fresh.state.rounds[0].search_query
        ) != normalize_question(source.question.text)
        metrics = PreSourceMetrics(
            base.answer_em,
            fresh.answer_em,
            base.answer_f1,
            fresh.answer_f1,
            base.retrieved_gold_support_recall,
            fresh.retrieved_gold_support_recall,
            changed,
        )
        if not changed:
            reasons.append("query_unchanged")
        if fresh.answer_em != 1.0:
            reasons.append("FRESH_answer_not_correct")
        if fresh.answer_em < base.answer_em or fresh.answer_f1 < base.answer_f1:
            reasons.append("answer_regressed")
        if metrics.fresh_support_recall < metrics.base_support_recall:
            reasons.append("support_regressed")
        if not (
            fresh.answer_em > base.answer_em
            or fresh.answer_f1 > base.answer_f1
            or metrics.fresh_support_recall > metrics.base_support_recall
        ):
            reasons.append("no_answer_or_support_increment")
    return PreSourceAdmission(
        source.source_id,
        source.fingerprint,
        not reasons and metrics is not None,
        tuple(reasons),
        metrics,
        base,
        fresh,
    )


def make_pre_source_candidate(
    source: PreSourceRecord,
    *,
    gold: GoldRecord,
    card_id: str,
    version: str,
    created_at: str,
    query_pattern: str,
    preconditions: tuple[str, ...],
    contraindications: tuple[str, ...],
    body: ParaphraseBody,
    evidence_contract: str,
    extraction_prompt_version: str,
    expected_cost: float = 0.0,
) -> ExperienceCard:
    """Use caller-extracted procedural text, after recomputing source admission.

    No extraction model is called here. Text must come from the caller's frozen,
    audited extraction procedure; shape validation is not semantic fact removal.
    """
    admission = evaluate_pre_source(source, gold)
    if not admission.eligible:
        raise ValueError(f"source pair is not eligible: {', '.join(admission.reasons)}")
    if not isinstance(body, ParaphraseBody):
        raise TypeError("this source protocol requires a ParaphraseBody")
    return ExperienceCard(
        card_id=card_id,
        version=version,
        lifecycle_state=CardLifecycle.CANDIDATE,
        created_at=created_at,
        activation=CardActivation(
            ActivationStage.PRE_RETRIEVAL,
            query_pattern,
            None,
            preconditions,
            contraindications,
        ),
        repair=RepairSpecification(
            body.form.value,
            None,
            evidence_contract,
            body,
            source.intent,
        ),
        provenance=PreSourceProvenance(
            (PreSourceRef(source.source_id, source.fingerprint, admission.metrics),),
            extraction_prompt_version,
            len({item.title for item in source.fresh.state.rounds[0].reply.evidence}),
        ),
        validation=CardValidation(VerificationTier.PROXY, (), 0, 0, 0, 0, 0, 0.0, None),
        serving=CardServing(expected_cost),
        schema_version="experience_card.v1",
    )


def pre_source_card_to_view(
    card: ExperienceCard,
    *,
    sources: Mapping[str, PreSourceRecord],
    available_capabilities: tuple[str, ...] = (),
) -> CardMemoryView:
    """Bind ALL actual PRE sources without reading target or source gold labels."""
    if not isinstance(card, ExperienceCard) or not isinstance(card.provenance, PreSourceProvenance):
        raise TypeError("a card with explicit PRE source provenance is required")
    if card.lifecycle_state is not CardLifecycle.CANDIDATE:
        raise ValueError("PRE views are diagnostic candidates, not ACTIVE serving memory")
    if not isinstance(available_capabilities, tuple) or any(
        not isinstance(value, str) or not value.strip() for value in available_capabilities
    ):
        raise ValueError("available_capabilities must be explicit names")
    if set(card.activation.required_retriever_capabilities) - set(available_capabilities):
        raise ValueError("required retriever capabilities are unavailable")
    refs = card.provenance.source_refs
    if set(sources) != {ref.source_id for ref in refs}:
        raise ValueError("bind exactly ALL referenced PRE sources")
    records = []
    for ref in refs:
        record = sources[ref.source_id]
        if not isinstance(record, PreSourceRecord) or (
            record.source_id != ref.source_id or record.fingerprint != ref.source_fingerprint
        ):
            raise ValueError("PRE source identity or observation fingerprint changed")
        if record.form is not card.repair.action_body.form or record.intent != card.repair.intent:
            raise ValueError("source and card form/intent must agree")
        if (
            not record.base.state.rounds
            or not record.fresh.state.rounds
            or any(
                event.status == "error"
                for result in (record.base, record.fresh)
                for event in result.events
            )
        ):
            raise ValueError("card must reference completed source arms")
        if record.fresh.state.rounds[0].decision.action is not Action.FRESH or normalize_question(
            record.fresh.state.rounds[0].search_query
        ) == normalize_question(record.question.text):
            raise ValueError("card must reference an actual changed FRESH query")
        records.append(record)
    if len({record.execution_kind for record in records}) != 1:
        raise ValueError("cannot combine mock and real source observations")
    ids = tuple(record.question.question_id for record in records)
    hashes = tuple(_question_hash(record.question.text) for record in records)
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError("independent PRE sources must have distinct questions")
    documents = {
        item.title
        for record in records
        for item in record.fresh.state.rounds[0].reply.evidence or ()
    }
    if len(documents) != card.provenance.independent_document_count:
        raise ValueError("PRE source document count does not match observations")
    payload = {
        "schema": "growrag-query-card-view-v1",
        "form": card.repair.action_body.form.value,
        "intent": card.repair.intent,
        "conditions": {
            "stage": card.activation.stage.value,
            "query_pattern": card.activation.query_pattern,
            "gap_pattern": card.activation.gap_pattern,
            "preconditions": card.activation.preconditions,
            "contraindications": card.activation.contraindications,
        },
        "body": asdict(card.repair.action_body),
        "success_criterion": card.repair.evidence_contract,
    }
    return CardMemoryView(
        card.versioned_id,
        ids[0],
        f"pre-source:{refs[0].source_id}#FRESH",
        "typed_card_diagnostic_v1",
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        ids,
        hashes,
        card.repair.action_body.form,
        card.repair.intent,
        ActivationStage.PRE_RETRIEVAL,
        False,
    )
