"""Synthetic read-only representation isolation tests; no network/model calls."""

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from growrag.experience.cards import ActivationStage
from growrag.experience.pre_sources import (
    PreSourceRecord,
    make_pre_source_candidate,
    pre_source_card_to_view,
)
from growrag.experiments.protocol import (
    Action,
    Answer,
    CallResult,
    Evidence,
    ExecutionKind,
    GoldRecord,
    RuntimeQuestion,
)
from growrag.experiments.representation_views import (
    CandidateSet,
    RepresentationCandidate,
    RepresentationKind,
    RepresentationView,
    ViewProvenance,
    build_representation_bundle,
    retrieve_candidates,
)
from growrag.outer_loop import RagReply, run_outer_loop
from growrag.query_actions import RewriteDecision
from growrag.query_operators import ParaphraseBody, RewriteForm


class _Backend:
    execution_kind = ExecutionKind.MOCK

    def run(self, request):
        return CallResult(
            RagReply(
                Answer(
                    "SOURCE_ANSWER_SENTINEL" if "established" in request.search_query else "wrong"
                ),
                (Evidence("e1", "SOURCE_DOCUMENT_SENTINEL", 0, "SOURCE_EVIDENCE_SENTINEL"),),
            ),
            transport_source="mock",
        )


class _Generator:
    execution_kind = ExecutionKind.MOCK

    def __init__(self, query):
        self.query = query

    def generate(self, question, decision, **kwargs):
        assert kwargs == {"evidence": (), "previous_queries": ()}
        return CallResult(self.query, transport_source="mock")


def _bundle(label="alpha", text=None, *, supplied=None):
    question = RuntimeQuestion(
        f"question-{label}", text or f"When was {label} University founded?", "synthetic"
    )
    intent = "SOURCE_INTENT_SENTINEL"
    base = run_outer_loop(question, _Backend(), max_rag_calls=1)
    fresh = run_outer_loop(
        question,
        _Backend(),
        generator=_Generator(f"{label} University established date"),
        policy=lambda _: RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, intent),
        max_rag_calls=1,
    )
    source = PreSourceRecord(
        f"source-{label}",
        question,
        base,
        fresh,
        RewriteForm.PARAPHRASE,
        intent,
        "rag-v1",
        "generator-v1",
        ExecutionKind.MOCK,
    )
    card = make_pre_source_candidate(
        source,
        gold=GoldRecord(
            question.question_id, ("SOURCE_ANSWER_SENTINEL",), (("SOURCE_DOCUMENT_SENTINEL", 0),)
        ),
        card_id=f"memory-{label}",
        version="v1",
        created_at="2026-09-11",
        query_pattern="SOURCE_PATTERN_SENTINEL",
        preconditions=("SOURCE_CONDITION_SENTINEL",),
        contraindications=(),
        body=ParaphraseBody("SOURCE_ACTION_SENTINEL", ("entity",)),
        evidence_contract="SOURCE_CRITERION_SENTINEL",
        extraction_prompt_version="synthetic-v1",
    )
    canonical = pre_source_card_to_view(card, sources={source.source_id: source})
    texts = supplied or {
        RepresentationKind.M2: "A brief procedural description.",
        RepresentationKind.M3: "Align founding wording while retaining the current entity.",
        RepresentationKind.M5: "Observed: founding request. Unknown: other relationships.",
    }
    provenance = {kind: ViewProvenance("manual", reference="synthetic fixture") for kind in texts}
    return build_representation_bundle(
        source,
        canonical,
        expected_source_fingerprint=source.fingerprint,
        texts=texts,
        provenance=provenance,
    )


def test_projection_is_actual_source_pair_and_audit_is_separate():
    bundle = _bundle()
    assert json.loads(bundle.view(RepresentationKind.M1).text) == {
        "source_question": bundle.source.question.text,
        "rewritten_query": bundle.source.fresh.state.rounds[0].search_query,
    }
    assert bundle.source_id == bundle.source.source_id
    assert bundle.source_question == bundle.source.question
    assert bundle.rewritten_query == bundle.source.fresh.state.rounds[0].search_query
    assert bundle.view(RepresentationKind.M1).provenance.method == "mechanical"
    audit = json.dumps(asdict(bundle))
    assert "SOURCE_ANSWER_SENTINEL" in audit
    assert "SOURCE_ACTION_SENTINEL" in audit
    assert "synthetic fixture" in audit
    assert "char_count" not in audit  # Calculated property, not a token claim.
    for view in bundle.views:
        assert view.char_count == len(view.text)
    with pytest.raises(FrozenInstanceError):
        bundle.max_chars = 9999


def test_all_views_share_slots_and_exact_canonical_execution_identity():
    bundles = (_bundle("beta"), _bundle("alpha"), _bundle("gamma"), _bundle("delta"))
    question = RuntimeQuestion("target-id", "When was another University founded?")
    candidates = retrieve_candidates(question, bundles)
    assert [c.bundle.source_id for c in candidates.candidates] == [
        "source-alpha",
        "source-beta",
        "source-delta",
    ]
    for kind in RepresentationKind:
        payload = candidates.selector_payload(kind)
        assert set(payload) == {"original_question", "candidates"}
        assert payload["original_question"] == question.text
        assert [c["candidate_id"] for c in payload["candidates"]] == ["c1", "c2", "c3"]
        for row in payload["candidates"]:
            assert set(row) == {"candidate_id", "text"}
        raw = json.dumps(payload)
        for secret in (
            "SOURCE_ANSWER_SENTINEL",
            "SOURCE_DOCUMENT_SENTINEL",
            "SOURCE_EVIDENCE_SENTINEL",
            "SOURCE_CONDITION_SENTINEL",
            "SOURCE_ACTION_SENTINEL",
            "SOURCE_INTENT_SENTINEL",
            "SOURCE_PATTERN_SENTINEL",
            "SOURCE_CRITERION_SENTINEL",
            "synthetic fixture",
            "source-alpha",
            "memory-alpha",
            "target-id",
        ):
            assert secret not in raw
        assert candidates.resolve("c1") is candidates.candidates[0].canonical_action
    assert candidates.resolve("FRESH") is None
    assert retrieve_candidates(question, tuple(reversed(bundles))) == candidates


def test_changing_view_text_does_not_change_candidate_pool_or_scores():
    bundle = _bundle()
    changed = replace(
        bundle,
        views=tuple(
            replace(view, text="irrelevant lexical words")
            if view.kind is RepresentationKind.M5
            else view
            for view in bundle.views
        ),
    )
    question = RuntimeQuestion("target", "University founded")
    first = retrieve_candidates(question, (bundle,)).candidates[0]
    second = retrieve_candidates(question, (changed,)).candidates[0]
    assert first.score == second.score
    assert first.candidate_id == second.candidate_id
    assert first.canonical_action is second.canonical_action


def test_source_ids_and_normalized_duplicate_questions_are_excluded():
    bundle = _bundle()
    assert not retrieve_candidates(bundle.source_question, (bundle,)).candidates
    duplicate = RuntimeQuestion("different-id", " WHEN WAS ALPHA university founded!!! ")
    assert not retrieve_candidates(duplicate, (bundle,)).candidates
    same_id = RuntimeQuestion(bundle.source_question.question_id, "Another University founded?")
    assert not retrieve_candidates(same_id, (bundle,)).candidates
    with pytest.raises(ValueError, match="own candidate"):
        CandidateSet(duplicate, (RepresentationCandidate("c1", bundle, 0.5),))


def test_empty_and_zero_match_pools_fall_back_without_hallucinating_a_candidate():
    question = RuntimeQuestion("target", "quantum photons")
    for pool in ((), (_bundle(),)):
        candidates = retrieve_candidates(question, pool)
        assert candidates.selector_payload(RepresentationKind.M2) == {
            "original_question": question.text,
            "candidates": [],
        }
        assert candidates.resolve("FRESH") is None
        with pytest.raises(ValueError, match="available candidate"):
            candidates.resolve("c1")


@pytest.mark.parametrize("selection", ["unknown", "c0", "c4", " c1", "fresh", "memory-alpha:v1"])
def test_unknown_or_nonexact_selector_ids_are_rejected(selection):
    candidates = retrieve_candidates(RuntimeQuestion("target", "University"), (_bundle(),))
    with pytest.raises(ValueError):
        candidates.resolve(selection)
    with pytest.raises(TypeError):
        candidates.resolve({"candidate_id": "c1"})


@pytest.mark.parametrize("corruption", ["source_hash", "source_id", "question_hash", "step_id"])
def test_bundle_refuses_mismatched_source_identity_and_hash(corruption):
    bundle = _bundle()
    if corruption == "source_hash":
        changes = {"source_fingerprint": "0" * 64}
    elif corruption == "source_id":
        changes = {
            "canonical_action": replace(
                bundle.canonical_action, source_query_id="other", source_query_ids=("other",)
            )
        }
    elif corruption == "question_hash":
        changes = {
            "canonical_action": replace(
                bundle.canonical_action,
                source_question_hashes=(hashlib.sha256(b"other").hexdigest(),),
            )
        }
    else:
        changes = {"canonical_action": replace(bundle.canonical_action, source_step_id="other")}
    with pytest.raises(ValueError, match="fingerprint|single source"):
        replace(bundle, **changes)


def test_post_and_multiple_source_canonical_actions_are_rejected():
    bundle = _bundle()
    payload = json.loads(bundle.canonical_action.text)
    payload["conditions"]["stage"] = "post_retrieval"
    post = replace(
        bundle.canonical_action, text=json.dumps(payload), stage=ActivationStage.POST_RETRIEVAL
    )
    with pytest.raises(ValueError, match="PRE paraphrase"):
        replace(bundle, canonical_action=post)
    multi = replace(
        bundle.canonical_action,
        source_query_ids=(*bundle.canonical_action.source_query_ids, "other"),
        source_question_hashes=(*bundle.canonical_action.source_question_hashes, "0" * 64),
    )
    with pytest.raises(ValueError, match="single source"):
        replace(bundle, canonical_action=multi)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "changed_m1", "empty", "too_long"])
def test_complete_nonempty_bounded_views_are_required(corruption):
    bundle = _bundle()
    views = bundle.views
    with pytest.raises(ValueError):
        if corruption == "missing":
            views = views[:-1]
        elif corruption == "duplicate":
            views = (*views[:-1], views[1])
        elif corruption == "changed_m1":
            views = (replace(views[0], text="invented historical query"), *views[1:])
        elif corruption == "empty":
            views = (views[0], replace(views[1], text=" "), *views[2:])
        else:
            views = (views[0], replace(views[1], text="x" * 1201), *views[2:])
        replace(bundle, views=views)


def test_bad_runtime_types_and_mutable_inputs_do_not_enter_selector():
    bundle = _bundle()
    with pytest.raises(TypeError):
        retrieve_candidates(GoldRecord("target", ("answer",)), (bundle,))
    with pytest.raises(TypeError):
        retrieve_candidates(RuntimeQuestion("target", "University"), [bundle])
    with pytest.raises(TypeError):
        retrieve_candidates(RuntimeQuestion("target", "University"), ({"gold": "answer"},))
    with pytest.raises(TypeError):
        replace(bundle, views=list(bundle.views))
    with pytest.raises(TypeError):
        replace(bundle, canonical_action={"body": "untyped"})
    with pytest.raises(TypeError):
        RepresentationView("M2", "text", ViewProvenance("manual"))
    candidates = retrieve_candidates(RuntimeQuestion("target", "University"), (bundle,))
    with pytest.raises(TypeError):
        candidates.selector_payload("M2")


@pytest.mark.parametrize("limit", [0, 4, True, 1.5, "3"])
def test_invalid_candidate_limits_are_rejected(limit):
    with pytest.raises(ValueError):
        retrieve_candidates(RuntimeQuestion("target", "University"), (), limit=limit)


def test_duplicate_pool_sources_or_normalized_questions_are_not_silently_deduplicated():
    first = _bundle()
    with pytest.raises(ValueError, match="unique"):
        retrieve_candidates(RuntimeQuestion("target", "University"), (first, first))
    other = _bundle("second", text=first.source_question.text.upper())
    with pytest.raises(ValueError, match="unique"):
        retrieve_candidates(RuntimeQuestion("target", "University"), (first, other))


def test_builder_rejects_wrong_keys_extra_gold_or_unattributed_generated_views():
    bundle = _bundle()
    kwargs = {
        "expected_source_fingerprint": bundle.source_fingerprint,
        "texts": {
            view.kind: view.text for view in bundle.views if view.kind is not RepresentationKind.M1
        },
        "provenance": {
            view.kind: view.provenance
            for view in bundle.views
            if view.kind is not RepresentationKind.M1
        },
    }
    for changes in (
        {"texts": {"M2": "untyped"}},
        {"texts": {**kwargs["texts"], "gold": "answer"}},
        {"texts": {RepresentationKind.M2: "missing"}},
        {"provenance": {}},
        {"max_chars": True},
    ):
        with pytest.raises((TypeError, ValueError)):
            build_representation_bundle(
                bundle.source, bundle.canonical_action, **(kwargs | changes)
            )
    with pytest.raises(ValueError, match="model_id and prompt_version"):
        ViewProvenance("pre_generated")
    with pytest.raises(ValueError):
        ViewProvenance("live_api")
    supplied = ViewProvenance("pre_generated", "declared-model", "declared-prompt", "artifact.json")
    assert supplied.method == "pre_generated"  # Does not claim that this test called a model.


def test_chars_not_tokens_and_payload_mutation_cannot_mutate_frozen_views():
    bundle = _bundle(
        supplied={
            kind: "条件是推断，不是已验证反例。"
            for kind in (RepresentationKind.M2, RepresentationKind.M3, RepresentationKind.M5)
        }
    )
    assert bundle.view(RepresentationKind.M2).char_count == len("条件是推断，不是已验证反例。")
    candidates = retrieve_candidates(RuntimeQuestion("target", "University"), (bundle,))
    payload = candidates.selector_payload(RepresentationKind.M2)
    payload["candidates"][0]["text"] = "injected result"
    assert (
        candidates.selector_payload(RepresentationKind.M2)["candidates"][0]["text"]
        != "injected result"
    )
