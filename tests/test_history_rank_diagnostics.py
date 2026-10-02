"""Synthetic immutable cards only; no API, datasets, labels, or file writes."""

import math
from dataclasses import replace

import pytest

from growrag.experiments import history_rank_diagnostics as ranker
from growrag.experiments.history_runtime import shortlist_cards
from growrag.experiments.protocol import RuntimeQuestion
from growrag.history_library import (
    FrozenHistoryLibrary,
    HistoryCard,
    HistoryCondition,
    HistoryExample,
    HistoryRecord,
)
from growrag.macro_operators import FieldEquals, GapField, OperatorSpec, QueryStep


def _rule(identity="RULE_A", text="birthplace"):
    return HistoryCard(identity, "1", text, text, text)


def _template(identity="TEMPLATE_A", query="{entity} birthplace"):
    return HistoryCard(
        identity,
        "1",
        "Template op_opaque",
        "Generic source template.",
        "This display wrapper is not the executable rule.",
        operator_spec=OperatorSpec(
            "OP_OPAQUE",
            "1",
            ("lookup",),
            (GapField("entity"),),
            (QueryStep("STEP_OPAQUE", query),),
        ),
    )


def _library(*cards, candidate_ids=()):
    return FrozenHistoryLibrary(
        "synthetic-ranking",
        (),
        tuple(
            HistoryRecord(
                card,
                "reference",
                "synthetic/provenance",
                "0" * 64,
                status="candidate" if card.card_id in candidate_ids else "published",
            )
            for card in cards
        ),
    )


@pytest.fixture
def library():
    return _library(
        _rule("RULE_LOCATION", "Locate a birthplace"),
        _rule("RULE_COMPARE", "Compare two directors"),
        _template("TEMPLATE_LOCATION"),
        _template("TEMPLATE_CAST", "{entity} cast actor"),
        _rule("UNPUBLISHED", "Locate a birthplace"),
        candidate_ids=("UNPUBLISHED",),
    )


@pytest.mark.parametrize(
    "question", ["Where was the actor born?", "Compare two directors", "operator_spec", "", "词汇"]
)
def test_legacy_exact_top_three_scores_and_order(library, question):
    original = (
        shortlist_cards(RuntimeQuestion("synthetic", question or " "), library)
        if question
        else None
    )
    # RuntimeQuestion refuses empty text; the old lexical function itself permits it.
    if original is None:
        from types import SimpleNamespace

        original = shortlist_cards(SimpleNamespace(text=""), library)
    got = ranker.rank_cards(question, library, "legacy_json_jaccard", limit=3)
    assert got == tuple(
        {"card_id": row["card_id"], "score": row["lexical_score"]} for row in original
    )


def test_template_action_text_uses_actual_spec_not_wrapper_or_ids():
    card = _template()
    assert ranker.action_text(card) == "entity\n{entity} birthplace"
    text = ranker.action_text(card)
    for excluded in (
        card.name,
        card.description,
        card.transformation_rule,
        "OP_OPAQUE",
        "STEP_OPAQUE",
    ):
        assert excluded not in text


def test_rule_action_text_retains_original_name_description_rule_verbatim():
    card = HistoryCard("RULE", "1", "original name", "original description", "original rule")
    assert ranker.action_text(card) == "original name\noriginal description\noriginal rule"


def test_template_when_constraints_and_required_binding_names_are_not_invented():
    card = replace(
        _template(),
        operator_spec=OperatorSpec(
            "BOUND",
            "7",
            ("lookup",),
            (GapField("year", "integer"),),
            (
                QueryStep(
                    "step", "{person} {year} birthplace", (FieldEquals("year", 1900),), ("person",)
                ),
            ),
        ),
    )
    assert ranker.action_text(card) == "year\n{person} {year} birthplace\nyear 1900\nperson"
    assert "requires_bindings" not in ranker.action_text(card)


@pytest.mark.parametrize("policy", ranker.POLICIES[1:])
def test_template_display_alias_and_all_identity_renames_do_not_change_action_scores(policy):
    card = _template()
    changed = replace(
        card,
        card_id="DIFFERENT_CARD",
        version="99",
        name="birthplace birthplace",
        description="different alias text",
        transformation_rule="invented wrapper noise",
        operator_spec=replace(
            card.operator_spec,
            operator_id="DIFFERENT_SPEC",
            version="different",
            steps=(replace(card.operator_spec.steps[0], step_id="DIFFERENT_STEP"),),
        ),
    )
    assert ranker.action_text(changed) == ranker.action_text(card)
    assert (
        ranker.rank_cards("birthplace", _library(card), policy)[0]["score"]
        == (ranker.rank_cards("birthplace", _library(changed), policy)[0]["score"])
    )


@pytest.mark.parametrize("policy", ranker.POLICIES)
def test_examples_conditions_and_provenance_poison_do_not_create_relevance(policy):
    card = replace(
        _template(),
        examples=(HistoryExample("FORBIDDENSECRET", "FORBIDDENSECRET answer"),),
        conditions=(HistoryCondition("FORBIDDENSECRET", "FORBIDDENSECRET"),),
    )
    record = HistoryRecord(card, "reference", "FORBIDDENSECRET", "a" * 64, status="published")
    lib = FrozenHistoryLibrary("FORBIDDENSECRET", (), (record,))
    assert ranker.rank_cards("FORBIDDENSECRET", lib, policy)[0]["score"] == 0.0


@pytest.mark.parametrize("policy", ranker.POLICIES[1:])
def test_action_policies_do_not_serialize_json_field_names(policy):
    lib = _library(_template(), _rule(text="location"))
    query = "operator_spec gap_schema transformation_rule action_kind"
    assert all(row["score"] == 0 for row in ranker.rank_cards(query, lib, policy))
    assert any(row["score"] > 0 for row in ranker.rank_cards(query, lib, "legacy_json_jaccard"))


@pytest.mark.parametrize("policy", ranker.POLICIES[1:])
def test_serializer_key_noise_cannot_affect_action_rankings(library, policy, monkeypatch):
    expected = ranker.rank_cards("birthplace", library, policy)

    def forbidden(*args, **kwargs):
        pytest.fail("action ranking must not use the serialized card view")

    monkeypatch.setattr(ranker, "card_view", forbidden)
    assert ranker.rank_cards("birthplace", library, policy) == expected


@pytest.mark.parametrize("policy", ranker.POLICIES)
def test_returns_all_published_candidates_and_top_three_is_a_prefix(library, policy):
    all_rows = ranker.rank_cards("birthplace", library, policy)
    assert len(all_rows) == 4
    assert "UNPUBLISHED" not in {row["card_id"] for row in all_rows}
    assert all(set(row) == {"card_id", "score"} for row in all_rows)
    assert ranker.rank_cards("birthplace", library, policy, limit=3) == all_rows[:3]
    assert all(type(row["score"]) is float and math.isfinite(row["score"]) for row in all_rows)


@pytest.mark.parametrize("policy", ranker.POLICIES)
def test_default_limit_covers_the_fixed_44_card_pool(policy):
    lib = _library(*(_rule(f"CARD_{index:02d}", "shared text") for index in reversed(range(44))))
    rows = ranker.rank_cards("shared", lib, policy)
    assert len(rows) == 44
    assert [row["card_id"] for row in rows] == [f"CARD_{index:02d}" for index in range(44)]
    assert ranker.rank_cards("shared", lib, policy, limit=3) == rows[:3]


@pytest.mark.parametrize("policy", ranker.POLICIES)
@pytest.mark.parametrize("query", ["", "   !!!", "NONOVERLAPPINGTOKEN"])
def test_empty_or_no_overlap_query_keeps_all_zero_score_cards_deterministically(policy, query):
    lib = _library(_rule("ZZZ", "alpha"), _rule("AAA", "beta"))
    assert ranker.rank_cards(query, lib, policy) == (
        {"card_id": "AAA", "score": 0.0},
        {"card_id": "ZZZ", "score": 0.0},
    )


@pytest.mark.parametrize("policy", ranker.POLICIES)
def test_empty_library_is_valid_and_no_fake_candidate_is_added(policy):
    assert ranker.rank_cards("anything", _library(), policy) == ()


def test_bm25_fixed_parameters_and_hand_computed_score():
    # Each rule repeats its actual text in name/description/rule; tf(alpha)=6, dl=6.
    lib = _library(_rule("ALPHA", "alpha alpha"), _rule("BETA", "beta"))
    got = ranker.rank_cards("alpha", lib, "action_text_bm25")
    expected = math.log1p(1.5 / 1.5) * (6 * 2.2 / (6 + 1.2 * (0.25 + 0.75 * 6 / 4.5)))
    assert ranker.BM25_K1 == 1.2 and ranker.BM25_B == 0.75
    assert got[0] == {"card_id": "ALPHA", "score": pytest.approx(expected)}
    assert got[1] == {"card_id": "BETA", "score": 0.0}
    assert ranker.rank_cards("alpha alpha alpha", lib, "action_text_bm25") == got


def test_bm25_explicit_stopwords_casefold_and_empty_token_corpus():
    lib = _library(_rule("ALPHA", "alpha"), _rule("THE", "the and of"))
    assert {"the", "and", "of"} <= ranker.ENGLISH_STOPWORDS
    assert isinstance(ranker.ENGLISH_STOPWORDS, frozenset)
    assert not {"no", "not", "without"} & ranker.ENGLISH_STOPWORDS
    assert ranker.rank_cards("THE ALPHA and OF", lib, "action_text_bm25") == ranker.rank_cards(
        "alpha", lib, "action_text_bm25"
    )
    stop_only = _library(_rule("ZZZ", "the"), _rule("AAA", "and"))
    assert ranker.rank_cards("the", stop_only, "action_text_bm25") == (
        {"card_id": "AAA", "score": 0.0},
        {"card_id": "ZZZ", "score": 0.0},
    )


@pytest.mark.parametrize("policy", ranker.POLICIES)
def test_nonmutation_defensive_results_and_repeatability(library, policy):
    before, fingerprint = library.to_json(), library.fingerprint
    result = ranker.rank_cards("birthplace", library, policy)
    expected = tuple(dict(row) for row in result)
    result[0]["score"] = -999
    assert ranker.rank_cards("birthplace", library, policy) == expected
    assert library.to_json() == before and library.fingerprint == fingerprint


@pytest.mark.parametrize("limit", [None, True, False, 0, -1, 45, 3.0, "3"])
def test_strict_limit_configuration(library, limit):
    with pytest.raises(ValueError, match="limit"):
        ranker.rank_cards("query", library, "action_text_jaccard", limit=limit)


@pytest.mark.parametrize("policy", [None, 1, True, "bm25", "ACTION_TEXT_BM25", ""])
def test_only_exact_registered_policy_names(library, policy):
    with pytest.raises(ValueError, match="policy"):
        ranker.rank_cards("query", library, policy)


@pytest.mark.parametrize("question", [None, 1, True, {"text": "query"}])
def test_question_requires_text_not_runtime_or_label_objects(library, question):
    with pytest.raises(TypeError, match="question_text"):
        ranker.rank_cards(question, library, "action_text_jaccard")


def test_action_and_library_types_are_explicit():
    with pytest.raises(TypeError, match="HistoryCard"):
        ranker.action_text({"query": "not a card"})
    with pytest.raises(TypeError, match="FrozenHistoryLibrary"):
        ranker.rank_cards("query", (), "action_text_jaccard")
