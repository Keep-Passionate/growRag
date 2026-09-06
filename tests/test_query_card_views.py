"""Typed-card contract tests with mock transport only, not model-quality tests."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_experience_cards import _archive_with_two_beneficial_pairs, _card, _context, _pair

from growrag.episodes import EpisodeArchive, EpisodeTurnRef
from growrag.experience import (
    ActivationStage,
    CardActivation,
    CardLifecycle,
    CardProvenance,
    ExperienceCardRegistry,
    RepairSpecification,
)
from growrag.experience.query_views import card_to_view
from growrag.experiments.api_client import ChatResponse
from growrag.experiments.llm_adapters import APIRewriter
from growrag.experiments.protocol import (
    Action,
    Answer,
    CallResult,
    DecisionState,
    Evidence,
    ExecutionKind,
    MemoryView,
    RuntimeQuestion,
    Usage,
)
from growrag.outer_loop import Feedback, RagReply, run_outer_loop
from growrag.query_actions import (
    CARD_ACTION_PROMPT_VERSION,
    APISingleQueryGenerator,
    RewriteDecision,
    diagnostic_reuse_decision,
)
from growrag.query_operators import ExpansionBody, ParaphraseBody, RewriteForm

TARGET = RuntimeQuestion("target", "When was Northbridge University established?")
CURRENT = Evidence("current-1", "Current document", 0, "A fictional current observation.")
PARAPHRASE = ParaphraseBody(
    "Replace colloquial wording with equivalent search vocabulary.",
    ("current entities", "time constraints", "relation direction"),
)
EXPAND = ExpansionBody(
    ("grounded alias", "equivalent relation term"),
    "Bind any factual alias only to the current question or current evidence.",
)


class MockClient:
    transport_source = "mock"
    config = SimpleNamespace(base_url="https://test.invalid/v1", model="TEST-NOT-LIVE")

    def __init__(self):
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        return ChatResponse(
            '{"query":"Northbridge University founding year"}',
            "TEST-NOT-LIVE",
            "TEST-NOT-LIVE",
            "mock-card-request",
            None,
            1,
            1,
            0,
            Path("not-written-by-test"),
            "mock",
        )


class MockRag:
    execution_kind = ExecutionKind.MOCK

    def __init__(self):
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return CallResult(
            RagReply(Answer("synthetic answer", (CURRENT.evidence_id,)), (CURRENT,)),
            usage=Usage(0, 0, 0),
            transport_source="mock",
        )


def source_questions():
    return {
        "ep-treatment-1": RuntimeQuestion("source-1", "question 1"),
        "ep-treatment-2": RuntimeQuestion("source-2", "question 2"),
    }


def typed_card(body=PARAPHRASE, *, stage=ActivationStage.PRE_RETRIEVAL, **overrides):
    return _card(
        activation=CardActivation(stage, "vocabulary alignment", gap_pattern=None),
        repair=RepairSpecification(
            body.form.value,
            None,
            "Preserve the original information need while retrieving useful current support.",
            body,
            "align search vocabulary",
        ),
        schema_version="experience_card.v1",
        **overrides,
    )


def compile_view(card=None, *, bindings=None, archive=None, capabilities=()):
    return card_to_view(
        card or typed_card(),
        archive=archive or _archive_with_two_beneficial_pairs(),
        source_questions=source_questions() if bindings is None else bindings,
        available_capabilities=capabilities,
    )


@pytest.mark.parametrize("body", [PARAPHRASE, EXPAND])
def test_typed_semantic_card_reaches_api_adapter_and_outer_loop_without_gap(body):
    view = compile_view(typed_card(body))
    decision = diagnostic_reuse_decision(view)
    client, backend = MockClient(), MockRag()
    result = run_outer_loop(
        TARGET,
        backend,
        generator=APISingleQueryGenerator(client),
        policy=lambda state: decision,
    )
    assert result.stop_reason == "unassessed"
    assert [event.operation for event in result.events] == ["rewrite", "rag"]
    assert len(client.requests) == len(backend.requests) == 1
    assert all(event.usage.api_requests == 0 for event in result.events)
    assert backend.requests[0].question == TARGET
    assert backend.requests[0].search_query == "Northbridge University founding year"
    messages, metadata = client.requests[0]
    payload = json.loads(messages[1]["content"])
    assert payload["current_evidence"] == payload["previous_queries"] == []
    assert "gap" not in payload
    assert payload["form"] == body.form.value
    assert metadata["prompt_version"] == CARD_ACTION_PROMPT_VERSION
    card_payload = json.loads(payload["optional_historical_procedure"]["text"])
    assert card_payload["conditions"]["gap_pattern"] is None
    assert card_payload["body"] == json.loads(json.dumps(asdict(body)))
    assert "not pre-verified" in messages[0]["content"]


def test_paraphrase_and_expansion_keep_distinct_body_schemas():
    paraphrase = json.loads(compile_view(typed_card(PARAPHRASE)).text)
    expansion = json.loads(compile_view(typed_card(EXPAND)).text)
    assert set(paraphrase["body"]) == {"rewrite_rule", "preserve"}
    assert set(expansion["body"]) == {
        "term_roles",
        "grounding_rule",
        "requires_current_evidence",
    }
    assert paraphrase["form"] != expansion["form"]


@pytest.mark.parametrize("body,wrong_kind", [(PARAPHRASE, "expand"), (EXPAND, "paraphrase")])
def test_operator_kind_cannot_disagree_with_typed_body(body, wrong_kind):
    with pytest.raises(ValueError, match="agree"):
        RepairSpecification(wrong_kind, None, "current support", body, "intent")


@pytest.mark.parametrize("wrong_field", ["form", "intent"])
def test_decision_cannot_override_the_selected_card_kind_or_intent(wrong_field):
    view = compile_view()
    form = RewriteForm.EXPAND if wrong_field == "form" else view.form
    intent = "different purpose" if wrong_field == "intent" else view.intent
    with pytest.raises(ValueError, match="must match"):
        RewriteDecision(Action.REUSE, form, intent, view)


@pytest.mark.parametrize("field", ["form", "intent", "stage", "requires_current_evidence"])
def test_view_metadata_cannot_silently_disagree_with_its_json_body(field):
    view = compile_view()
    mismatches = {
        "form": RewriteForm.EXPAND,
        "intent": "changed intent",
        "stage": ActivationStage.POST_RETRIEVAL,
        "requires_current_evidence": True,
    }
    with pytest.raises(ValueError, match="inconsistent typed card payload"):
        replace(view, **{field: mismatches[field]})


def test_legacy_gap_rewriter_rejects_typed_view_before_call():
    client = MockClient()
    state = DecisionState(TARGET, (CURRENT,), "synthetic gap", "test-context")
    with pytest.raises(ValueError, match="typed"):
        APIRewriter(client).rewrite(state, memory=compile_view())
    assert not client.requests


@pytest.mark.parametrize("change", ["missing_second", "extra", "swapped_text"])
def test_all_sources_must_be_exactly_bound_to_archived_question_text(change):
    bindings = source_questions()
    if change == "missing_second":
        bindings.pop("ep-treatment-2")
    elif change == "extra":
        bindings["unrelated-episode"] = RuntimeQuestion("other", "unrelated question")
    else:
        bindings["ep-treatment-2"] = RuntimeQuestion("source-2", "different source text")
    with pytest.raises(ValueError, match="ALL source|does not match"):
        compile_view(bindings=bindings)


@pytest.mark.parametrize(
    "target",
    [
        RuntimeQuestion("source-1", "A different spelling cannot bypass the source ID."),
        RuntimeQuestion("source-2", "A different spelling cannot bypass the second source ID."),
        RuntimeQuestion("changed-id", " QUESTION 2!!! "),
        RuntimeQuestion("also-changed-id", "ｑｕｅｓｔｉｏｎ １"),
        RuntimeQuestion("whitespace-id", "  QuEsTiOn    2  "),
    ],
)
def test_every_source_and_normalized_same_question_are_rejected_before_calls(target):
    decision = diagnostic_reuse_decision(compile_view())
    client, backend = MockClient(), MockRag()
    generator = APISingleQueryGenerator(client)
    with pytest.raises(ValueError, match="own experience"):
        generator.generate(target, decision)
    result = run_outer_loop(target, backend, generator=generator, policy=lambda state: decision)
    assert result.stop_reason == "same_question_memory_rejected"
    assert result.events == ()
    assert client.requests == backend.requests == []


def test_second_canonical_source_does_not_discard_first_source_isolation():
    card = typed_card()
    card = replace(
        card,
        provenance=replace(
            card.provenance, canonical_example_refs=(card.provenance.source_episode_turn_refs[1],)
        ),
    )
    view = compile_view(card)
    assert view.source_query_id == "source-2"
    assert view.source_step_id == card.provenance.canonical_example_refs[0].key
    assert set(view.source_query_ids) == {"source-1", "source-2"}
    assert view.is_source(RuntimeQuestion("new-id", "Question 1?"))


def test_missing_canonical_example_falls_back_to_first_declared_source():
    card = typed_card()
    card = replace(card, provenance=replace(card.provenance, canonical_example_refs=()))
    view = compile_view(card)
    assert view.source_query_id == "source-1"
    assert view.source_step_id == card.provenance.source_episode_turn_refs[0].key
    assert set(view.source_query_ids) == {"source-1", "source-2"}


def test_two_episodes_with_the_same_question_are_not_independent_sources():
    archive = EpisodeArchive()
    archive.register_context(_context())
    for index in (1, 2):
        _, treatment, _ = _pair(index)
        archive.append_episode(replace(treatment, original_query="same source question"))
    bindings = {
        f"ep-treatment-{index}": RuntimeQuestion(f"source-{index}", "same source question")
        for index in (1, 2)
    }
    with pytest.raises(ValueError, match="duplicate a source question"):
        compile_view(bindings=bindings, archive=archive)


@pytest.mark.parametrize("turn_id", ["missing-turn", "base-turn-1"])
def test_provenance_must_resolve_to_a_repair_turn(turn_id):
    card = typed_card()
    refs = (EpisodeTurnRef("ep-treatment-1", turn_id), card.provenance.source_episode_turn_refs[1])
    card = replace(card, provenance=CardProvenance(refs, (refs[0],), 2, 2))
    with pytest.raises(ValueError, match="archived repair turn"):
        compile_view(card)


def test_independent_episode_count_is_checked_against_complete_sources():
    card = typed_card(lifecycle_state=CardLifecycle.CANDIDATE)
    card = replace(card, provenance=replace(card.provenance, independent_episode_count=3))
    with pytest.raises(ValueError, match="count mismatch"):
        compile_view(card)


def test_archived_answers_gold_validation_and_scores_do_not_enter_view_or_prompt():
    card = typed_card()
    view = compile_view(card)
    serialized = json.dumps(asdict(view), ensure_ascii=False)
    for excluded in (
        "question 1",
        "question 2",
        "answer-treatment",
        "sha256:treatment",
        "verification_tier",
        "verification_event_ids",
        "matched_trials",
        "mean_gain",
        "conditional_harm_rate",
        "gold",
        "0.4",
    ):
        assert excluded not in serialized
    client = MockClient()
    APISingleQueryGenerator(client).generate(TARGET, diagnostic_reuse_decision(view))
    payload = json.loads(client.requests[0][0][1]["content"])
    assert payload["optional_historical_procedure"] == json.loads(serialized)
    assert set(json.loads(view.text)) == {
        "schema",
        "form",
        "intent",
        "conditions",
        "body",
        "success_criterion",
    }


@pytest.mark.parametrize("lifecycle", [CardLifecycle.RETIRED, CardLifecycle.QUARANTINE])
def test_retired_and_quarantined_cards_cannot_compile(lifecycle):
    with pytest.raises(ValueError, match="retired"):
        compile_view(typed_card(lifecycle_state=lifecycle))


def test_capabilities_must_be_available_before_view_compilation():
    card = typed_card()
    card = replace(
        card,
        activation=replace(card.activation, required_retriever_capabilities=("lexical_search",)),
    )
    with pytest.raises(ValueError, match="capabilities"):
        compile_view(card)
    assert compile_view(card, capabilities=("lexical_search",)).memory_id == card.versioned_id


@pytest.mark.parametrize(
    "stage,previous,evidence",
    [
        (ActivationStage.POST_RETRIEVAL, (), (CURRENT,)),
        (ActivationStage.PRE_RETRIEVAL, (TARGET.text,), (CURRENT,)),
    ],
)
def test_api_adapter_rejects_wrong_stage_without_a_model_call(stage, previous, evidence):
    decision = diagnostic_reuse_decision(compile_view(typed_card(stage=stage)))
    client = MockClient()
    with pytest.raises(ValueError, match="stage"):
        APISingleQueryGenerator(client).generate(
            TARGET, decision, evidence=evidence, previous_queries=previous
        )
    assert not client.requests


def test_expansion_requires_current_evidence_when_its_body_says_so():
    body = replace(EXPAND, requires_current_evidence=True)
    view = compile_view(typed_card(body, stage=ActivationStage.POST_RETRIEVAL))
    decision, client = diagnostic_reuse_decision(view), MockClient()
    with pytest.raises(ValueError, match="requires current evidence"):
        APISingleQueryGenerator(client).generate(
            TARGET, decision, evidence=(), previous_queries=(TARGET.text,)
        )
    assert not client.requests
    APISingleQueryGenerator(client).generate(
        TARGET, decision, evidence=(CURRENT,), previous_queries=(TARGET.text,)
    )
    assert len(client.requests) == 1


def test_outer_loop_rejects_post_retrieval_card_before_first_rag():
    decision = diagnostic_reuse_decision(
        compile_view(typed_card(EXPAND, stage=ActivationStage.POST_RETRIEVAL))
    )
    client, backend = MockClient(), MockRag()
    result = run_outer_loop(
        TARGET,
        backend,
        generator=APISingleQueryGenerator(client),
        policy=lambda state: decision,
    )
    assert result.stop_reason == "card_stage_or_evidence_mismatch"
    assert client.requests == backend.requests == []


def test_post_retrieval_card_sees_current_evidence_after_base_round():
    body = replace(EXPAND, requires_current_evidence=True)
    view = compile_view(typed_card(body, stage=ActivationStage.POST_RETRIEVAL))
    client, backend = MockClient(), MockRag()
    result = run_outer_loop(
        TARGET,
        backend,
        generator=APISingleQueryGenerator(client),
        policy=lambda state: diagnostic_reuse_decision(view) if state.rounds else RewriteDecision(),
        assessor=lambda state, reply: Feedback(False, origin="synthetic_test"),
    )
    assert len(backend.requests) == 2
    assert [record.decision.action for record in result.state.rounds] == [Action.BASE, Action.REUSE]
    payload = json.loads(client.requests[0][0][1]["content"])
    assert payload["current_evidence"] == [asdict(CURRENT)]
    assert payload["previous_queries"] == [TARGET.text]


def test_outer_loop_rejects_pre_retrieval_card_after_base_without_generating():
    decision = diagnostic_reuse_decision(compile_view())
    client, backend = MockClient(), MockRag()
    result = run_outer_loop(
        TARGET,
        backend,
        generator=APISingleQueryGenerator(client),
        policy=lambda state: decision if state.rounds else RewriteDecision(),
        assessor=lambda state, reply: Feedback(False, origin="synthetic_test"),
    )
    assert result.stop_reason == "card_stage_or_evidence_mismatch"
    assert len(backend.requests) == 1
    assert client.requests == []
    assert [event.operation for event in result.events] == ["rag", "assess"]


def test_typed_revision_is_diagnostic_candidate_not_active_or_registry_serving():
    parent = _card()
    typed = typed_card()
    child = parent.revised_candidate(
        version="v2",
        created_at="2026-09-06T12:00:00+08:00",
        activation=typed.activation,
        repair=typed.repair,
        expected_cost=0.5,
    )
    assert child.schema_version == "experience_card.v1"
    assert child.lifecycle_state is CardLifecycle.CANDIDATE
    assert child.validation.matched_trials == 0
    assert child.validation.verification_event_ids == ()
    assert child.parent_versioned_ids == (parent.versioned_id,)
    assert diagnostic_reuse_decision(compile_view(child)).memory.memory_id == child.versioned_id
    with pytest.raises(ValueError, match="ACTIVE"):
        replace(child, lifecycle_state=CardLifecycle.ACTIVE)
    registry = ExperienceCardRegistry(_archive_with_two_beneficial_pairs())
    registry.register(parent)
    with pytest.raises(ValueError, match="verification event"):
        registry.register(child)
    assert parent.lifecycle_state is CardLifecycle.ACTIVE
    assert parent.validation.matched_trials == 2


def test_legacy_cards_need_an_explicit_typed_revision_not_only_a_version_label():
    for card in (_card(), replace(_card(), schema_version="experience_card.v1")):
        with pytest.raises(ValueError, match="explicit typed"):
            compile_view(card)
    with pytest.raises(ValueError, match="typed action bodies require"):
        replace(typed_card(), schema_version="experience_card.v0")
    with pytest.raises(TypeError, match="typed card view"):
        diagnostic_reuse_decision(MemoryView("m", "s", "turn", "legacy", "old procedure"))


def test_typed_body_cannot_duplicate_legacy_slot_template():
    with pytest.raises(ValueError, match="replaces slot_template"):
        RepairSpecification("paraphrase", "legacy text", "support", PARAPHRASE, "intent")


@pytest.mark.parametrize("key", ["intent", "rewrite_rule", "query_pattern"])
def test_duplicate_json_fields_are_rejected_at_all_nesting_levels(key):
    view = compile_view()
    token = json.dumps(key) + ":"
    ambiguous = view.text.replace(token, token + ' "conflicting value", ' + token, 1)
    assert ambiguous != view.text
    with pytest.raises(ValueError, match="inconsistent typed card payload"):
        replace(view, text=ambiguous)
