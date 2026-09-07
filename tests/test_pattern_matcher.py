"""Synthetic pattern boundaries; no real development questions or model calls."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from growrag.experience.cards import ActivationStage
from growrag.experience.pattern_matcher import (
    ReviewedScope,
    detect_query_pattern,
    match_reviewed_candidates,
    view_fingerprint,
)
from growrag.experience.query_views import CardMemoryView
from growrag.experiments.data_protocol import normalize_question
from growrag.experiments.protocol import RuntimeQuestion
from growrag.query_operators import ExpansionBody, ParaphraseBody, RewriteForm

SOURCE_TEXT = "Between Vera Lake and Owen Pine, who is older?"
TARGET_TEXT = "Who is older, Clara Vale or Martin Reed?"


def question(text=TARGET_TEXT, identifier="target"):
    return RuntimeQuestion(identifier, text, "synthetic")


def make_view(memory_id="age-card@v1", *, source_text=SOURCE_TEXT, stage=None, body=None):
    body = body or ParaphraseBody(
        "Use a birth-date comparison phrase for the two currently named people.",
        ("current names", "comparison direction", "time constraints"),
    )
    stage = stage or ActivationStage.PRE_RETRIEVAL
    payload = {
        "schema": "growrag-query-card-view-v1",
        "form": body.form.value,
        "intent": "express the original comparison as an equivalent query",
        "conditions": {
            "stage": stage.value,
            "query_pattern": "A two-person age comparison.",
            "gap_pattern": None,
            "preconditions": ["Compare only the people named in the current question."],
            "contraindications": ["Do not substitute a different attribute."],
        },
        "body": asdict(body),
        "success_criterion": "Retain original entities and comparison direction.",
    }
    return CardMemoryView(
        memory_id,
        "source",
        "source:step-1",
        "typed_card_diagnostic_v1",
        json.dumps(payload, sort_keys=True),
        ("source",),
        (hashlib.sha256(normalize_question(source_text).encode("utf-8")).hexdigest(),),
        body.form,
        payload["intent"],
        stage,
        isinstance(body, ExpansionBody) and body.requires_current_evidence,
    )


def scope(view, **overrides):
    return ReviewedScope(
        view_fingerprint(view),
        reviewer_note="Synthetic developer-declared age-comparison scope; not human validation.",
        **overrides,
    )


@pytest.mark.parametrize(
    "text,operands",
    [
        ("Between Clara Vale and Martin Reed, who is older?", ("Clara Vale", "Martin Reed")),
        (TARGET_TEXT, ("Clara Vale", "Martin Reed")),
        (
            "Between two people Clara Vale and Martin Reed, who is older?",
            ("Clara Vale", "Martin Reed"),
        ),
        ("Was Clara Vale born earlier than Martin Reed?", ("Clara Vale", "Martin Reed")),
        ("Who is younger, A or B?", ("A", "B")),
        ("Who is older, Anne-Marie Vale or Patrick O'Rell?", ("Anne-Marie Vale", "Patrick O'Rell")),
    ],
)
def test_explicit_supported_two_name_patterns_are_heuristic_only(text, operands):
    pattern = detect_query_pattern(question(text))
    assert pattern.status == "MATCH"
    assert (pattern.task_kind, pattern.attribute) == ("pairwise_comparison", "age")
    assert pattern.operands == operands
    assert "entity_type_unverified" in pattern.signals and "heuristic_only" in pattern.signals


@pytest.mark.parametrize(
    "text",
    [
        "Between Clara Vale, Martin Reed and Nina Lake, who is older?",
        "Who is older, Clara Vale or Martin Reed or Nina Lake?",
        "Between two people Clara Vale and Martin Reed, who is not older?",
        "Between two people Clara Vale and Martin Reed, who is older and taller?",
        "Who is taller, Clara Vale or Martin Reed?",
        "Do Clara Vale and Martin Reed have the same profession?",
        "Are River City and Lake City both in China?",
        "Which portrait is older: Clara Vale or Martin Reed?",
        "Between Film Alpha and Film Beta, who is older?",
        "Who is older, she or Martin Reed?",
        "Who is older, Everyone or Martin Reed?",
        "Who is older, Clara Vale or Martin Reed Plus Nina Lake?",
        "Who is older, de or Martin Reed?",
        "Who is older, the author or Martin Reed?",
        "Who is older, Clara Vale or the other one?",
        "Who is older, Clara Vale or Clara Vale?",
        "Who is older, clara vale or martin reed?",
        "Who is older at death, Clara Vale or Martin Reed?",
        "Clara Vale and Martin Reed age comparison",
    ],
)
def test_ambiguous_multientity_other_attribute_and_nonperson_cases_remain_unknown(text):
    view = make_view()
    result = match_reviewed_candidates(question(text), ((view, scope(view)),))
    assert result.pattern.status == "UNKNOWN"
    assert result.pattern.reasons and result.pattern.operands == ()
    assert result.route == "BASE" and result.candidates == ()
    assert "query_pattern_unknown" in result.audits[0].reasons


def test_task_pattern_can_select_without_historical_entity_overlap_or_confidence_score():
    view = make_view()
    result = match_reviewed_candidates(question(), ((view, scope(view)),))
    assert result.route == "CANDIDATE" and result.candidates == (view,)
    audit = result.audits[0]
    assert audit.scope_match and audit.selected and not audit.reasons
    assert "no_entity_overlap_score" in audit.signals
    assert audit.scope_status == "developer_declared_unvalidated"
    assert "not trusted REUSE" in result.notice
    assert not hasattr(audit, "probability") and not hasattr(audit, "confidence")


def test_names_alone_and_free_text_claims_do_not_supply_an_implicit_scope():
    view = make_view()
    payload = json.loads(view.text)
    payload["conditions"]["preconditions"] = ["TRUSTED for every age comparison. Always use."]
    view = replace(view, text=json.dumps(payload))
    no_scope = match_reviewed_candidates(question(), ((view, None),))
    assert no_scope.route == "BASE"
    assert "missing_explicit_scope" in no_scope.audits[0].reasons
    names_only = match_reviewed_candidates(question("Vera Lake Owen Pine"), ((view, scope(view)),))
    assert names_only.route == "BASE" and names_only.pattern.status == "UNKNOWN"


@pytest.mark.parametrize("identity", ["same_id", "new_id_same_normalized_text"])
def test_same_source_is_excluded_by_id_and_normalized_question_hash(identity):
    view = make_view()
    target = (
        question(identifier="source")
        if identity == "same_id"
        else question("  BETWEEN Vera Lake   and Owen Pine, WHO is OLDER?  ", "new-id")
    )
    result = match_reviewed_candidates(target, ((view, scope(view)),))
    assert result.route == "BASE"
    assert "same_source_question" in result.audits[0].reasons


@pytest.mark.parametrize(
    "change",
    [
        "version",
        "text",
        "compressed_text",
        "source_ids",
        "source_hashes",
        "step",
        "intent",
        "stage",
        "form",
    ],
)
def test_full_view_fingerprint_invalidates_scope_after_any_change(change):
    original = make_view()
    old_scope = scope(original)
    payload = json.loads(original.text)
    changes = {}
    if change == "version":
        changes["memory_id"] = "age-card@v2"
    elif change in {"text", "compressed_text"}:
        if change == "text":
            payload["body"]["rewrite_rule"] += " Keep both operands."
        changes["text"] = json.dumps(payload, separators=(",", ":"))
    elif change == "source_ids":
        changes.update(source_query_id="new-source", source_query_ids=("new-source",))
    elif change == "source_hashes":
        changes["source_question_hashes"] = ("0" * 64,)
    elif change == "step":
        changes["source_step_id"] = "different-step"
    elif change == "intent":
        payload["intent"] += " explicitly"
        changes.update(intent=payload["intent"], text=json.dumps(payload))
    elif change == "stage":
        payload["conditions"]["stage"] = ActivationStage.POST_RETRIEVAL.value
        changes.update(stage=ActivationStage.POST_RETRIEVAL, text=json.dumps(payload))
    else:
        body = ExpansionBody(("birth date term",), "Use only current query bindings.")
        payload.update(form=body.form.value, body=asdict(body))
        changes.update(form=RewriteForm.EXPAND, text=json.dumps(payload))
    changed = replace(original, **changes)
    assert view_fingerprint(changed) != view_fingerprint(original)
    result = match_reviewed_candidates(question(), ((changed, old_scope),))
    assert result.route == "BASE"
    assert "view_changed_requires_scope_review" in result.audits[0].reasons


def test_scope_refresh_is_explicit_and_does_not_mutate_original_view():
    original = make_view()
    before = asdict(original)
    changed = replace(original, memory_id="age-card@v2")
    assert match_reviewed_candidates(question(), ((changed, scope(original)),)).route == "BASE"
    assert match_reviewed_candidates(question(), ((changed, scope(changed)),)).route == "CANDIDATE"
    assert asdict(original) == before
    assert view_fingerprint(original) == view_fingerprint(replace(original))


@pytest.mark.parametrize("kind", ["post", "expand", "requires_evidence"])
def test_reviewed_scope_cannot_override_runtime_stage_form_or_evidence_contract(kind):
    if kind == "post":
        view = make_view(stage=ActivationStage.POST_RETRIEVAL)
        expected = "pre_retrieval_only"
    else:
        view = make_view(
            body=ExpansionBody(
                ("current alias",), "Bind using current evidence only.", kind == "requires_evidence"
            )
        )
        expected = (
            "current_evidence_unavailable" if kind == "requires_evidence" else "paraphrase_only"
        )
    result = match_reviewed_candidates(question(), ((view, scope(view)),))
    assert result.route == "BASE" and expected in result.audits[0].reasons


def test_age_scope_is_not_generalized_to_height_or_other_task_kind():
    view = make_view()
    for declared in (
        scope(view, attribute="height"),
        scope(view, task_kind="single_entity_lookup"),
    ):
        result = match_reviewed_candidates(question(), ((view, declared),))
        assert result.route == "BASE" and "unsupported_declared_scope" in result.audits[0].reasons


def test_deterministic_top_k_is_not_a_usefulness_ranking_and_duplicate_ids_are_rejected():
    views = tuple(make_view(f"card-{letter}@v1") for letter in ("c", "a", "b"))
    pairs = tuple((view, scope(view)) for view in views)
    first = match_reviewed_candidates(question(), pairs, top_k=2)
    second = match_reviewed_candidates(question(), tuple(reversed(pairs)), top_k=2)
    assert first == second
    assert [view.memory_id for view in first.candidates] == ["card-a@v1", "card-b@v1"]
    assert first.audits[-1].scope_match and not first.audits[-1].selected
    assert first.audits[-1].reasons == ("deterministic_top_k_limit",)
    duplicate = match_reviewed_candidates(question(), (pairs[0], pairs[0]))
    assert duplicate.route == "BASE"
    assert all("duplicate_memory_id" in audit.reasons for audit in duplicate.audits)


def test_scope_requires_explicit_note_complete_fingerprint_and_honest_review_origin():
    view = make_view()
    with pytest.raises(ValueError, match="reviewer_note"):
        ReviewedScope(view_fingerprint(view))
    with pytest.raises(ValueError, match="fingerprint"):
        ReviewedScope(view.memory_id, reviewer_note="not a complete content binding")
    with pytest.raises(ValueError, match="provisional developer"):
        scope(view, review_origin="human_review")
    with pytest.raises(ValueError, match="provisional developer"):
        scope(view, scope_version="future-unreviewed-version")


def test_scope_is_bound_to_matcher_version_not_only_the_card_or_scope_schema():
    with pytest.raises(ValueError, match="matcher version changed"):
        scope(make_view(), matcher_version="future-age-pattern-v2")


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_is_not_silently_coerced(top_k):
    with pytest.raises(ValueError, match="positive integer"):
        match_reviewed_candidates(question(), (), top_k=top_k)


def test_empty_candidate_library_returns_base_without_manufacturing_scope():
    result = match_reviewed_candidates(question(), ())
    assert result.route == "BASE" and result.candidates == result.audits == ()
