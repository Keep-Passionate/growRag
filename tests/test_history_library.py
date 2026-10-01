"""Synthetic-only history foundation contracts; no data files, model, or network."""

import hashlib
import json
import re
from dataclasses import FrozenInstanceError, replace

import pytest

from growrag.history_library import (
    FrozenHistoryLibrary,
    HistoryCard,
    HistoryCondition,
    HistoryExample,
    HistoryRecord,
    card_view,
    condition_observations,
    import_operator_bank,
    import_reformer_patterns,
    resolve_card,
    resolve_operator,
)
from growrag.macro_operators import (
    GapField,
    GoalContract,
    OperatorRegistry,
    OperatorSpec,
    QueryStep,
    RuntimeState,
)
from growrag.operator_bank import FrozenOperatorBank, OperatorRecord

SOURCE_SHA = "a" * 64
PROTOCOL = "synthetic-history-v1"


@pytest.fixture
def spec():
    return OperatorSpec(
        "SYNTHETIC_LOCATION",
        "1",
        ("lookup",),
        (GapField("entity"),),
        (QueryStep("locate", "{entity} location"),),
    )


@pytest.fixture
def card(spec):
    return HistoryCard(
        card_id="CARD_LOCATION",
        version="1",
        name="Locate an entity",
        description="Fill the declared entity field in the recorded location query.",
        transformation_rule="Render the executable location template without adding facts.",
        examples=(HistoryExample("Where is Cedar Hall?", "Cedar Hall location"),),
        conditions=(
            HistoryCondition("entity_known", "The current entity has been identified."),
            HistoryCondition("location_needed", "Location information is still needed."),
            HistoryCondition("scope_known", "The current scope has been checked."),
        ),
        operator_spec=spec,
    )


@pytest.fixture
def learned_record(card):
    return HistoryRecord(
        card=card,
        source_kind="learned",
        source_ref="synthetic/source-1/episode/1",
        source_sha256=SOURCE_SHA,
        source_qids=("source-1",),
        status="published",
    )


@pytest.fixture
def library(learned_record):
    return FrozenHistoryLibrary(PROTOCOL, ("source-1",), (learned_record,))


def _pattern(**changes):
    return {
        "pattern_name": "Synthetic subject clarification",
        "description": "Preserve the subject in a retrieval query.",
        "transformation_rule": "Repeat the explicitly named subject; do not add an answer.",
        "examples": [["Where is Cedar Hall?", "Cedar Hall location"]],
        **changes,
    }


def _import_patterns(patterns, *, expected_sha256=None):
    raw = json.dumps(patterns, ensure_ascii=False).encode("utf-8")
    return import_reformer_patterns(
        raw,
        expected_sha256=expected_sha256 or hashlib.sha256(raw).hexdigest(),
        source_ref="synthetic/reference/patterns.json",
        protocol_id=PROTOCOL,
    )


def _operator_bank(spec):
    candidate = replace(spec, operator_id="SYNTHETIC_CANDIDATE")
    records = (
        OperatorRecord(
            spec,
            ("source-1",),
            "synthetic-operator-v1",
            "synthetic/source-1/op",
            "validated",
            "synthetic/source-1/op#publication",
        ),
        OperatorRecord(
            candidate,
            ("source-2",),
            "synthetic-operator-v1",
            "synthetic/source-2/op",
        ),
    )
    return FrozenOperatorBank("synthetic-operator-v1", ("source-1", "source-2"), records)


def test_frozen_round_trip_preserves_records_spec_and_fingerprint(library):
    encoded = library.to_json()
    restored = FrozenHistoryLibrary.from_json(encoded)
    assert restored == library
    assert restored.fingerprint == library.fingerprint
    assert restored.to_json() == encoded
    assert restored.published_cards == (library.records[0].card,)
    with pytest.raises(FrozenInstanceError):
        restored.protocol_id = "mutated"


def test_candidate_is_retained_but_not_published(learned_record):
    candidate = replace(learned_record, status="candidate")
    bank = FrozenHistoryLibrary(PROTOCOL, ("source-1",), (candidate,))
    assert bank.records == (candidate,)
    assert bank.published_cards == ()


def test_reference_library_does_not_require_source_question_ids(card):
    record = HistoryRecord(card, "reference", "synthetic/reference", SOURCE_SHA, status="published")
    bank = FrozenHistoryLibrary(PROTOCOL, (), (record,))
    assert bank.published_cards == (card,)
    assert record.source_qids == ()


@pytest.mark.parametrize(
    ("kind", "ids"),
    [("reference", ("source-1",)), ("learned", ()), ("evaluation", ("source-1",))],
)
def test_source_kind_contract_is_not_inferred_or_coerced(card, kind, ids):
    with pytest.raises((TypeError, ValueError)):
        HistoryRecord(card, kind, "synthetic/source", SOURCE_SHA, ids)


def test_learned_record_outside_allowed_sources_is_rejected(learned_record):
    with pytest.raises((TypeError, ValueError)):
        FrozenHistoryLibrary(PROTOCOL, ("another-source",), (learned_record,))


def test_duplicate_source_ids_are_not_independent_evidence(learned_record):
    with pytest.raises((TypeError, ValueError)):
        replace(learned_record, source_qids=("source-1", "source-1"))


@pytest.mark.parametrize("status", ["validated", "trusted", "", True])
def test_status_does_not_silently_promote_or_alias(learned_record, status):
    with pytest.raises((TypeError, ValueError)):
        replace(learned_record, status=status)


@pytest.mark.parametrize("sha", ["", "not-a-sha", "a" * 63, "g" * 64])
def test_record_requires_well_formed_source_sha(learned_record, sha):
    with pytest.raises((TypeError, ValueError)):
        replace(learned_record, source_sha256=sha)


def test_duplicate_card_identity_is_rejected(learned_record):
    with pytest.raises((TypeError, ValueError)):
        FrozenHistoryLibrary(PROTOCOL, ("source-1",), (learned_record, learned_record))


def test_view_changes_affect_library_fingerprint_not_original_spec(learned_record, library):
    revised_card = replace(learned_record.card, description="A changed display description.")
    revised = FrozenHistoryLibrary(
        PROTOCOL, ("source-1",), (replace(learned_record, card=revised_card),)
    )
    assert revised.fingerprint != library.fingerprint
    assert revised_card.operator_spec == learned_record.card.operator_spec


def test_three_views_change_only_optional_display_fields(card):
    rule = card_view(card, representation="rule")
    examples = card_view(card, representation="examples")
    conditions = card_view(card, representation="conditions")
    assert "examples" not in rule and "conditions" not in rule
    assert "examples" in examples and "conditions" not in examples
    assert "examples" in conditions and "conditions" in conditions
    assert {k: v for k, v in examples.items() if k != "examples"} == rule
    assert {
        k: v
        for k, v in conditions.items()
        if k not in {"examples", "conditions", "applicability_status"}
    } == rule
    assert conditions["applicability_status"] == "unknown"
    assert card_view(card) == conditions


def test_no_recorded_conditions_do_not_mean_unconditional_applicability(card):
    view = card_view(replace(card, conditions=()))
    assert view["conditions"] == []
    assert view["applicability_status"] == "unknown"


def test_serving_view_does_not_contain_provenance_status_or_quality_scores(card):
    rendered = json.dumps(card_view(card))
    for field in (
        "source_qids",
        "source_ref",
        "source_sha256",
        "publication_status",
        "gold_answer",
        "answer_em",
        "answer_f1",
    ):
        assert f'"{field}"' not in rendered


def test_unknown_representation_is_rejected(card):
    with pytest.raises((TypeError, ValueError)):
        card_view(card, representation="invented")


def test_condition_observations_are_explicit_and_in_declaration_order(card):
    observed = {"entity_known": True, "location_needed": False}
    before = observed.copy()
    assert condition_observations(card, observed) == [
        {"condition_id": "entity_known", "status": "supported"},
        {"condition_id": "location_needed", "status": "contradicted"},
        {"condition_id": "scope_known", "status": "unknown"},
    ]
    assert observed == before
    assert all(row["status"] == "unknown" for row in condition_observations(card, {}))
    assert condition_observations(card, {"entity_known": None})[0]["status"] == "unknown"


@pytest.mark.parametrize("value", [0, 1, "true", "false", "", [], {}])
def test_condition_truth_values_are_never_coerced(card, value):
    with pytest.raises((TypeError, ValueError)):
        condition_observations(card, {"entity_known": value})


def test_unknown_condition_id_cannot_become_a_new_condition(card):
    with pytest.raises((TypeError, ValueError)):
        condition_observations(card, {"invented_condition": True})


def test_no_conditions_means_no_claims_not_implicit_support(card):
    unconditioned = replace(card, conditions=())
    assert condition_observations(unconditioned, {}) == []
    assert unconditioned.conditions == ()


def test_resolve_published_offered_card_returns_original_spec(library, card, spec):
    assert resolve_card(library, (card.card_id,), card.card_id) == card
    resolved = resolve_operator(library, (card.card_id,), card.card_id)
    assert resolved == spec
    registry = OperatorRegistry()
    registry.register(resolved)
    plan = registry.plan(
        resolved.operator_id,
        resolved.version,
        goal=GoalContract("Where is Cedar Hall?", "lookup"),
        gap={"entity": "Cedar Hall"},
        state=RuntimeState(remaining_retrievals=1),
    )
    assert [request.query for request in plan.requests] == ["Cedar Hall location"]


@pytest.mark.parametrize("offered", [(), ("another-card",)])
def test_selection_must_have_been_offered(library, card, offered):
    with pytest.raises((TypeError, ValueError, KeyError)):
        resolve_card(library, offered, card.card_id)


def test_offering_candidate_does_not_authorize_execution(learned_record):
    record = replace(learned_record, status="candidate")
    bank = FrozenHistoryLibrary(PROTOCOL, ("source-1",), (record,))
    with pytest.raises((TypeError, ValueError, KeyError)):
        resolve_operator(bank, (record.card.card_id,), record.card.card_id)


def test_selection_cannot_invent_an_absent_published_card(library):
    with pytest.raises((TypeError, ValueError, KeyError)):
        resolve_card(library, ("invented-card",), "invented-card")


def test_rule_only_card_cannot_become_an_executable_operator(learned_record):
    card = replace(learned_record.card, operator_spec=None)
    bank = FrozenHistoryLibrary(PROTOCOL, ("source-1",), (replace(learned_record, card=card),))
    assert resolve_card(bank, (card.card_id,), card.card_id) == card
    with pytest.raises((TypeError, ValueError)):
        resolve_operator(bank, (card.card_id,), card.card_id)


def test_reformer_import_preserves_reference_text_without_invented_conditions():
    pattern = _pattern(pattern_name="Synthetic 查询 clarification")
    library = _import_patterns([pattern])
    assert library.protocol_id == PROTOCOL
    assert len(library.records) == len(library.published_cards) == 1
    record = library.records[0]
    card = record.card
    assert re.fullmatch(r"RF_[0-9a-f]{16}", card.card_id)
    assert card.name == pattern["pattern_name"]
    assert card.description == pattern["description"]
    assert card.transformation_rule == pattern["transformation_rule"]
    assert card.examples == tuple(HistoryExample(*pair) for pair in pattern["examples"])
    assert card.conditions == () and card.operator_spec is None
    assert record.source_kind == "reference" and record.source_qids == ()
    assert record.status == "published"
    assert _import_patterns([pattern]).fingerprint == library.fingerprint


def test_reformer_import_requires_exact_bytes_sha():
    with pytest.raises((TypeError, ValueError)):
        _import_patterns([_pattern()], expected_sha256="b" * 64)


def test_reformer_import_does_not_require_a_fixed_ten_patterns():
    first = _pattern()
    second = _pattern(pattern_name="Another reference", transformation_rule="Preserve location.")
    library = _import_patterns([first, second])
    assert len(library.published_cards) == 2
    assert len({card.card_id for card in library.published_cards}) == 2


@pytest.mark.parametrize(
    "field", ["pattern_name", "description", "transformation_rule", "examples"]
)
def test_reformer_import_rejects_missing_original_fields(field):
    pattern = _pattern()
    del pattern[field]
    with pytest.raises((TypeError, ValueError)):
        _import_patterns([pattern])


def test_reformer_import_rejects_extra_fields_instead_of_dropping_them():
    with pytest.raises((TypeError, ValueError)):
        _import_patterns([_pattern(gold_answer="Synthetic forbidden label")])


@pytest.mark.parametrize(
    "examples",
    ["not an array", ["q", "rewrite"], [["q"]], [["q", "rewrite", "extra"]], [["q", 1]]],
)
def test_reformer_examples_must_be_two_dimensional_string_pairs(examples):
    with pytest.raises((TypeError, ValueError)):
        _import_patterns([_pattern(examples=examples)])


def test_operator_import_preserves_all_specs_sources_and_publication_status(spec):
    source_bank = _operator_bank(spec)
    imported = import_operator_bank(
        source_bank,
        source_ref="synthetic/frozen-bank.json",
        source_sha256=SOURCE_SHA,
        protocol_id=PROTOCOL,
    )
    assert len(imported.records) == 2
    assert len(imported.published_cards) == 1
    assert imported.allowed_source_ids == source_bank.allowed_source_ids
    by_spec = {record.card.operator_spec.operator_id: record for record in imported.records}
    for original in source_bank.records:
        record = by_spec[original.spec.operator_id]
        assert record.card.operator_spec == original.spec
        assert record.source_qids == original.source_qids
        assert record.source_kind == "learned"
        assert record.status == ("published" if original.status == "validated" else "candidate")
        assert record.card.conditions == ()
        assert record.card.examples == ()
    assert imported.published_cards[0].operator_spec == spec
    assert FrozenHistoryLibrary.from_json(imported.to_json()) == imported


def test_imported_operator_display_changes_cannot_change_execution(spec):
    imported = import_operator_bank(
        _operator_bank(spec), "synthetic/bank.json", SOURCE_SHA, protocol_id=PROTOCOL
    )
    published = imported.published_cards[0]
    changed = replace(published, name="Different display name", description="Only a display edit.")
    assert changed.operator_spec == spec
    assert published.operator_spec == spec
    assert card_view(changed, representation="rule") != card_view(published, representation="rule")


def test_import_and_projection_are_pure_functions(spec):
    source_bank = _operator_bank(spec)
    before = source_bank.to_json()
    imported = import_operator_bank(source_bank, "synthetic/bank.json", SOURCE_SHA)
    snapshot = imported.to_json()
    for card in imported.published_cards:
        for representation in ("rule", "examples", "conditions"):
            card_view(card, representation=representation)
        condition_observations(card, {})
    assert source_bank.to_json() == before
    assert imported.to_json() == snapshot
