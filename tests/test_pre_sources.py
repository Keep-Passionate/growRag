"""Source-only PRE pairing, admission and unified-card bridge; all calls are mock."""

import json
from dataclasses import asdict, replace

import pytest

from growrag.episodes import EpisodeArchive
from growrag.experience.cards import (
    ActivationStage,
    CardLifecycle,
    CardValidation,
    PreSourceMetrics,
    PreSourceProvenance,
    PreSourceRef,
    VerificationTier,
)
from growrag.experience.pre_sources import (
    PreSourceRecord,
    evaluate_pre_source,
    make_pre_source_candidate,
    pre_source_card_to_view,
)
from growrag.experience.query_views import card_to_view
from growrag.experience.registry import ExperienceCardRegistry
from growrag.experiments.protocol import (
    Action,
    Answer,
    BackendCallError,
    CallResult,
    Evidence,
    ExecutionKind,
    GoldRecord,
    MemoryView,
    RuntimeQuestion,
    Usage,
)
from growrag.outer_loop import Feedback, RagReply, run_outer_loop
from growrag.query_actions import RewriteDecision
from growrag.query_operators import ParaphraseBody, RewriteForm

QUESTION = RuntimeQuestion("source-1", "When was Northbridge University founded?", "synthetic")
SUPPORT = (
    Evidence("e1", "Northbridge", 0, "A fictional university."),
    Evidence("e2", "Northbridge", 1, "It was founded in 1900."),
)
GOLD = GoldRecord(QUESTION.question_id, ("1900",), (("Northbridge", 0), ("Northbridge", 1)))
INTENT = "terminology alignment"
BODY = ParaphraseBody("Use an equivalent founding noun phrase.", ("entity", "time", "direction"))


class Backend:
    execution_kind = ExecutionKind.MOCK

    def __init__(self, answer, evidence):
        self.answer, self.evidence = answer, evidence

    def run(self, request):
        return CallResult(
            RagReply(Answer(self.answer), self.evidence),
            usage=Usage(10, 2, 0),
            transport_source="mock",
        )


class Generator:
    execution_kind = ExecutionKind.MOCK

    def __init__(self, query):
        self.query = query

    def generate(self, question, decision, **kwargs):
        assert kwargs == {"evidence": (), "previous_queries": ()}
        return CallResult(self.query, usage=Usage(5, 2, 0), transport_source="mock")


def source(
    *,
    base_answer="wrong",
    fresh_answer="1900",
    base_evidence=SUPPORT[:1],
    fresh_evidence=SUPPORT,
    query="Northbridge University establishment date",
):
    base = run_outer_loop(QUESTION, Backend(base_answer, base_evidence), max_rag_calls=1)
    fresh = run_outer_loop(
        QUESTION,
        Backend(fresh_answer, fresh_evidence),
        generator=Generator(query),
        policy=lambda state: RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, INTENT),
        max_rag_calls=1,
    )
    return PreSourceRecord(
        "pre-source-1",
        QUESTION,
        base,
        fresh,
        RewriteForm.PARAPHRASE,
        INTENT,
        "fixed-rag",
        "fixed-generator",
        ExecutionKind.MOCK,
    )


def candidate(record=None, **kwargs):
    values = dict(
        gold=GOLD,
        card_id="pre-card",
        version="v1",
        created_at="2026-09-06",
        query_pattern="founding questions",
        preconditions=("A named entity is present.",),
        contraindications=("Do not change historical time scope.",),
        body=BODY,
        evidence_contract="Retain constraints and find supporting information.",
        extraction_prompt_version="synthetic-extraction-v1",
    )
    values.update(kwargs)
    return make_pre_source_candidate(record or source(), **values)


def test_independent_source_record_is_not_a_fictional_base_first_episode():
    record = source()
    assert record.schema_version == "pre_source_pair.v1"
    assert len(record.base.state.rounds) == len(record.fresh.state.rounds) == 1
    assert record.fresh.state.rounds[0].decision.action is Action.FRESH
    assert [event.operation for event in record.fresh.events] == ["rewrite", "rag"]
    assert "turns" not in record.to_dict()
    assert record.fingerprint == replace(record).fingerprint
    assert record.fingerprint != replace(record, rag_fingerprint="changed-rag").fingerprint
    assert json.loads(json.dumps(record.to_dict()))["execution_kind"] == "mock"


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"base_answer": "1900"},
        {"base_evidence": SUPPORT},
    ],
)
def test_answer_or_support_increment_can_admit_when_neither_layer_regresses(changes):
    admission = evaluate_pre_source(source(**changes), GOLD)
    assert admission.eligible and admission.reasons == ()
    assert admission.metrics.eligible
    assert admission.fresh_feedback.answer_em == 1
    assert asdict(admission)["source_id"] == "pre-source-1"


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"fresh_answer": "wrong"}, "FRESH_answer_not_correct"),
        ({"base_evidence": SUPPORT, "fresh_evidence": SUPPORT[:1]}, "support_regressed"),
        ({"base_answer": "1900", "base_evidence": SUPPORT}, "no_answer_or_support_increment"),
        ({"query": QUESTION.text.upper()}, "query_unchanged"),
        ({"fresh_evidence": None}, "FRESH_evidence_unavailable"),
        ({"base_evidence": None}, "BASE_evidence_unavailable"),
    ],
)
def test_noneligible_sources_are_preserved_but_not_converted_to_cards(changes, reason):
    record = source(**changes)
    admission = evaluate_pre_source(record, GOLD)
    assert not admission.eligible and reason in admission.reasons
    with pytest.raises(ValueError, match="not eligible"):
        candidate(record)


def test_missing_or_wrong_source_gold_is_not_silently_treated_as_success():
    record = source()
    admission = evaluate_pre_source(record, replace(GOLD, supporting_facts=()))
    assert not admission.eligible and admission.metrics is None
    assert "annotated_support_unavailable" in admission.reasons
    with pytest.raises(ValueError, match="source question"):
        evaluate_pre_source(record, replace(GOLD, question_id="target-question"))


def test_failed_source_has_auditable_cost_but_is_not_an_admitted_candidate():
    class Broken(Backend):
        def run(self, request):
            raise BackendCallError("synthetic failure", usage=Usage(2, 0, 0))

    record = replace(source(), base=run_outer_loop(QUESTION, Broken("", ()), max_rag_calls=1))
    admission = evaluate_pre_source(record, GOLD)
    assert not admission.eligible and "BASE_not_completed" in admission.reasons
    assert record.base.events[0].usage.input_tokens == 2


@pytest.mark.parametrize(
    "corruption",
    [
        "question",
        "multiple_rounds",
        "prefix",
        "judge",
        "reuse",
        "kind",
        "transport",
        "observed",
        "intent",
    ],
)
def test_source_record_rejects_incompatible_or_ambiguous_origin(corruption):
    record = source()
    result = record.fresh
    row = result.state.rounds[0]
    if corruption == "question":
        result = replace(
            result, state=replace(result.state, question=RuntimeQuestion("other", "Other?"))
        )
    elif corruption == "multiple_rounds":
        result = replace(result, state=replace(result.state, rounds=(row, row)))
    elif corruption == "prefix":
        result = replace(
            result, events=(replace(result.events[0], operation="initial_retrieve"), *result.events)
        )
    elif corruption == "judge":
        result = replace(
            result, state=replace(result.state, rounds=(replace(row, feedback=Feedback(True)),))
        )
    elif corruption == "reuse":
        decision = RewriteDecision(
            Action.REUSE, RewriteForm.PARAPHRASE, INTENT, MemoryView("m", "other", "s", "v", "rule")
        )
        result = replace(
            result, state=replace(result.state, rounds=(replace(row, decision=decision),))
        )
    elif corruption == "kind":
        result = replace(
            result,
            events=(replace(result.events[0], execution_kind=ExecutionKind.REAL), result.events[1]),
        )
    elif corruption == "transport":
        result = replace(
            result,
            events=(replace(result.events[0], transport_source="live_api"), result.events[1]),
        )
    elif corruption == "observed":
        result = replace(result, state=replace(result.state, observed_evidence=()))
    elif corruption == "intent":
        result = replace(
            result,
            state=replace(
                result.state,
                rounds=(replace(row, decision=replace(row.decision, intent="different purpose")),),
            ),
        )
    with pytest.raises(ValueError):
        replace(record, fresh=result)


def test_source_admission_is_not_counted_as_independent_transfer_validation():
    card = candidate()
    assert isinstance(card.provenance, PreSourceProvenance)
    assert card.provenance.source_type == "pre_independent_pair.v1"
    assert card.provenance.source_refs[0].admission.fresh_answer_em == 1
    assert card.validation == CardValidation(VerificationTier.PROXY, (), 0, 0, 0, 0, 0, 0.0, None)
    assert card.lifecycle_state is CardLifecycle.CANDIDATE
    assert card.schema_version == "experience_card.v1"
    assert not hasattr(card.provenance, "source_episode_turn_refs")
    with pytest.raises(ValueError, match="CANDIDATE"):
        replace(card, lifecycle_state=CardLifecycle.ACTIVE)
    with pytest.raises(ValueError, match="not matched"):
        replace(
            card,
            validation=CardValidation(
                VerificationTier.GOLD, ("fake",), 1, 1, 0, 0, 1, 1.0, "today"
            ),
        )


def test_existing_registry_and_episode_view_do_not_accept_the_new_source_variant():
    card = candidate()
    archive = EpisodeArchive()
    with pytest.raises(ValueError, match="episode provenance"):
        ExperienceCardRegistry(archive).register(card)
    with pytest.raises(ValueError, match="pre_source_card_to_view"):
        card_to_view(card, archive=archive, source_questions={})


def test_compiler_binds_sources_and_whitelists_runtime_data_without_gold(monkeypatch):
    record = source()
    card = candidate(record)

    def forbidden(*args, **kwargs):
        raise AssertionError("compiler must not call offline gold evaluator")

    monkeypatch.setattr("growrag.experience.pre_sources.evaluate_pre_source", forbidden)
    view = pre_source_card_to_view(card, sources={record.source_id: record})
    payload = json.loads(view.text)
    assert payload["body"] == {"rewrite_rule": BODY.rewrite_rule, "preserve": list(BODY.preserve)}
    assert payload["conditions"]["stage"] == "pre_retrieval"
    assert payload["conditions"]["gap_pattern"] is None
    assert view.source_step_id == "pre-source:pre-source-1#FRESH"
    assert view.is_source(QUESTION)
    assert view.is_source(RuntimeQuestion("renamed-source", QUESTION.text.upper()))
    assert not view.is_source(RuntimeQuestion("target", "Who founded Cedar Library?"))
    assert "1900" not in view.text and QUESTION.text not in view.text
    assert "admission" not in view.text and "validation" not in view.text
    assert "observed_evidence" not in view.text and "rag_fingerprint" not in view.text


@pytest.mark.parametrize("corruption", ["missing", "extra", "identity", "fingerprint", "documents"])
def test_compiler_rejects_missing_changed_or_miscounted_source_records(corruption):
    record = source()
    card = candidate(record)
    sources = {record.source_id: record}
    if corruption == "missing":
        sources = {}
    elif corruption == "extra":
        sources["extra"] = record
    elif corruption == "identity":
        sources[record.source_id] = replace(record, source_id="other")
    elif corruption == "fingerprint":
        sources[record.source_id] = replace(record, rag_fingerprint="changed")
    else:
        card = replace(card, provenance=replace(card.provenance, independent_document_count=99))
    with pytest.raises(ValueError):
        pre_source_card_to_view(card, sources=sources)


def test_cannot_relabel_pre_candidate_as_post_or_fake_admission_success():
    card = candidate()
    with pytest.raises(ValueError, match="PRE v1"):
        replace(card, activation=replace(card.activation, stage=ActivationStage.POST_RETRIEVAL))
    bad_metrics = replace(card.provenance.source_refs[0].admission, query_changed=False)
    with pytest.raises(ValueError, match="eligible"):
        replace(card.provenance.source_refs[0], admission=bad_metrics)
    with pytest.raises(ValueError, match="finite"):
        PreSourceMetrics(0, 1, 0, 1, float("nan"), 1, True)
    with pytest.raises(ValueError, match="SHA-256"):
        PreSourceRef("id", "invalid", card.provenance.source_refs[0].admission)


def test_required_capability_and_source_card_intent_are_not_assumed():
    record = source()
    card = candidate(record)
    needs = replace(
        card, activation=replace(card.activation, required_retriever_capabilities=("x",))
    )
    with pytest.raises(ValueError, match="capabilities"):
        pre_source_card_to_view(needs, sources={record.source_id: record})
    pre_source_card_to_view(
        needs, sources={record.source_id: record}, available_capabilities=("x",)
    )
    altered = replace(card, repair=replace(card.repair, intent="other purpose"))
    with pytest.raises(ValueError, match="form/intent"):
        pre_source_card_to_view(altered, sources={record.source_id: record})
