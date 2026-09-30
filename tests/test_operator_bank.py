"""Synthetic-only operator serialization and frozen source-boundary tests."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from growrag.macro_operators import (
    FieldEquals,
    GapField,
    GoalContract,
    OperatorRegistry,
    OperatorSpec,
    QueryStep,
    RuntimeState,
)
from growrag.operator_bank import (
    FrozenOperatorBank,
    MemoryBuilder,
    OperatorRecord,
    operator_from_dict,
    operator_from_json,
    operator_to_dict,
    operator_to_json,
    record_from_dict,
    record_to_dict,
)


@pytest.fixture
def spec():
    return OperatorSpec(
        "CURRENT_GAP",
        "1",
        ("lookup",),
        (GapField("term"), GapField("missing", "bool"), GapField("hop", "integer")),
        (
            QueryStep(
                "search",
                "{original_question} {term}",
                (FieldEquals("missing", True), FieldEquals("hop", 1)),
            ),
        ),
    )


@pytest.fixture
def record(spec):
    return OperatorRecord(spec, ("source-1",), "protocol-a", "audit/source-1/op-1")


def test_operator_round_trip_and_execution_are_identical(spec):
    encoded = operator_to_dict(spec)
    assert type(encoded["steps"]) is list
    assert type(encoded["steps"][0]["when"]) is list
    assert operator_from_dict(encoded) == spec
    decoded = operator_from_json(operator_to_json(spec))
    assert decoded == spec
    registry = OperatorRegistry()
    registry.register(decoded)
    plan = registry.plan(
        "CURRENT_GAP",
        "1",
        goal=GoalContract("Which place?", "lookup"),
        gap={"term": "location", "missing": True, "hop": 1},
        state=RuntimeState(),
    )
    assert plan.requests[0].query == "Which place? location"


@pytest.mark.parametrize("path", [(), ("gap_schema", 0), ("steps", 0), ("steps", 0, "when", 0)])
def test_unknown_fields_at_every_object_depth_are_rejected(spec, path):
    payload = operator_to_dict(spec)
    target = payload
    for part in path:
        target = target[part]
    target["gold_answer"] = "synthetic forbidden label"
    with pytest.raises(ValueError, match="exactly"):
        operator_from_dict(payload)


def test_missing_explicit_fields_are_rejected(spec):
    payload = operator_to_dict(spec)
    del payload["gap_schema"][0]["required"]
    with pytest.raises(ValueError, match="exactly"):
        operator_from_dict(payload)


@pytest.mark.parametrize("value", ["true", 1, None])
def test_codec_does_not_coerce_required_boolean(spec, value):
    payload = operator_to_dict(spec)
    payload["gap_schema"][0]["required"] = value
    with pytest.raises(TypeError, match="bool"):
        operator_from_dict(payload)


def test_condition_value_keeps_integer_distinct_from_bool(spec):
    payload = operator_to_dict(spec)
    payload["steps"][0]["when"][1]["value"] = True
    with pytest.raises(TypeError, match="integer"):
        operator_from_dict(payload)


@pytest.mark.parametrize("field", ["supported_intents", "gap_schema", "steps"])
def test_python_tuple_or_other_array_substitute_is_rejected(spec, field):
    payload = operator_to_dict(spec)
    payload[field] = tuple(payload[field])
    with pytest.raises(ValueError, match="JSON array"):
        operator_from_dict(payload)


@pytest.mark.parametrize("template", ["{term.__class__}", "{term[0]}", "{term!r}", "{term:>10}"])
def test_codec_never_adds_python_template_execution(spec, template):
    payload = operator_to_dict(spec)
    payload["steps"][0]["template"] = template
    with pytest.raises(ValueError, match="plain placeholders"):
        operator_from_dict(payload)


@pytest.mark.parametrize("text", ['{"operator_id":"A","operator_id":"B"}', '{"x":{"y":1,"y":2}}'])
def test_duplicate_json_keys_are_rejected_before_parsing(text):
    with pytest.raises(ValueError, match="duplicate JSON key"):
        operator_from_json(text)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_constants_are_rejected(value):
    with pytest.raises(ValueError, match="nonstandard JSON"):
        operator_from_json('{"x":' + value + "}")


@pytest.mark.parametrize("name", ["gold_answer", "answer", "supporting_facts", "Final_Answer"])
def test_declared_label_inputs_cannot_enter_runtime_schema(name):
    spec = OperatorSpec(
        "BAD_INPUT",
        "1",
        ("lookup",),
        (GapField(name),),
        (QueryStep("search", "{original_question}"),),
    )
    with pytest.raises(ValueError, match="label fields"):
        operator_to_dict(spec)


def test_bound_label_input_is_also_rejected():
    spec = OperatorSpec(
        "BAD_INPUT",
        "1",
        ("lookup",),
        (),
        (QueryStep("search", "{gold_answer}", requires_bindings=("gold_answer",)),),
    )
    with pytest.raises(ValueError, match="label fields"):
        operator_to_json(spec)


def test_record_round_trip_is_data_only(record):
    payload = record_to_dict(record)
    assert record_from_dict(payload) == record
    assert set(payload) == {
        "spec",
        "source_qids",
        "protocol_id",
        "source_trace_ref",
        "status",
        "validation_ref",
    }
    payload["final_answer"] = "not allowed"
    with pytest.raises(ValueError, match="exactly"):
        record_from_dict(payload)


@pytest.mark.parametrize("ids", [(), ("source-1", "source-1"), ["source-1"]])
def test_source_ids_must_be_nonempty_unique_immutable(record, ids):
    with pytest.raises(ValueError):
        replace(record, source_qids=ids)


def test_freeze_preserves_candidates_but_does_not_publish_them(record):
    builder = MemoryBuilder(("source-1",), "protocol-a")
    builder.add(record)
    bank = builder.freeze()
    assert bank.records == (record,)
    assert bank.published_specs == ()
    assert len(bank.fingerprint) == 64
    with pytest.raises(RuntimeError, match="frozen"):
        builder.add(record)
    with pytest.raises(FrozenInstanceError):
        bank.protocol_id = "new"
    with pytest.raises(FrozenInstanceError):
        bank.records[0].status = "validated"


def test_explicit_validation_is_required_for_publication(record):
    with pytest.raises(ValueError, match="validation_ref"):
        replace(record, status="validated")
    with pytest.raises(ValueError, match="candidate"):
        replace(record, validation_ref="audit/validation")
    validated = replace(record, status="validated", validation_ref="audit/source-only-validation")
    builder = MemoryBuilder(("source-1",), "protocol-a")
    builder.add(validated)
    bank = builder.freeze()
    assert bank.published_specs == (record.spec,)
    assert bank.published_specs[0] is record.spec


@pytest.mark.parametrize("qid", ["calibration-1", "test-1", "unknown-1"])
def test_non_source_records_rejected_before_they_can_affect_bank(record, qid):
    builder = MemoryBuilder(("source-1",), "protocol-a")
    with pytest.raises(ValueError, match="non-source"):
        builder.add(replace(record, source_qids=("source-1", qid)))
    assert builder.freeze().records == ()


def test_protocol_mismatch_rejected(record):
    builder = MemoryBuilder(("source-1",), "protocol-b")
    with pytest.raises(ValueError, match="protocol"):
        builder.add(record)


def test_same_id_version_cannot_overwrite_even_if_content_matches(record):
    builder = MemoryBuilder(("source-1",), "protocol-a")
    builder.add(record)
    with pytest.raises(ValueError, match="overwritten"):
        builder.add(record)
    newer = replace(record, spec=replace(record.spec, version="2"))
    builder.add(newer)
    assert len(builder.freeze().records) == 2


def test_frozen_constructor_enforces_same_source_and_duplicate_guards(record):
    with pytest.raises(ValueError, match="non-source"):
        FrozenOperatorBank("protocol-a", ("other-source",), (record,))
    with pytest.raises(ValueError, match="duplicate"):
        FrozenOperatorBank("protocol-a", ("source-1",), (record, record))
    with pytest.raises(TypeError, match="immutable"):
        FrozenOperatorBank("protocol-a", ("source-1",), [record])


def test_source_size_prefixes_are_independent(record):
    small = MemoryBuilder(("source-1",), "protocol-a")
    large = MemoryBuilder(("source-1", "source-2"), "protocol-a")
    small.add(record)
    large.add(record)
    extra = replace(record, spec=replace(record.spec, version="2"), source_qids=("source-2",))
    with pytest.raises(ValueError, match="non-source"):
        small.add(extra)
    large.add(extra)
    small_bank, large_bank = small.freeze(), large.freeze()
    assert len(small_bank.records) == 1
    assert len(large_bank.records) == 2
    assert small_bank.fingerprint != large_bank.fingerprint
    assert small_bank.records[0] == large_bank.records[0]


def test_fingerprint_is_order_independent_but_covers_source_pool(record):
    second = replace(record, spec=replace(record.spec, version="2"), source_qids=("source-2",))
    one = FrozenOperatorBank("protocol-a", ("source-1", "source-2"), (record, second))
    two = FrozenOperatorBank("protocol-a", ("source-2", "source-1"), (second, record))
    assert one.fingerprint == two.fingerprint
    assert (
        one.fingerprint
        != replace(one, allowed_source_ids=("source-1", "source-2", "source-3")).fingerprint
    )
    assert one.fingerprint != replace(one, records=(record,)).fingerprint


def test_equal_fingerprints_also_have_equal_publication_order(record):
    first = replace(record, status="validated", validation_ref="audit/validation-1")
    second = replace(
        first, spec=replace(record.spec, version="2"), source_qids=("source-2", "source-1")
    )
    one = FrozenOperatorBank("protocol-a", ("source-1", "source-2"), (first, second))
    two = FrozenOperatorBank("protocol-a", ("source-2", "source-1"), (second, first))
    assert one.fingerprint == two.fingerprint
    assert one == two
    assert one.published_specs == two.published_specs
    assert FrozenOperatorBank.from_json(two.to_json()) == two


def test_bank_json_round_trip_and_detached_payload(record):
    builder = MemoryBuilder(("source-1",), "protocol-a")
    builder.add(record)
    bank = builder.freeze()
    assert FrozenOperatorBank.from_json(bank.to_json()) == bank
    bank.verify_fingerprint(bank.fingerprint)
    payload = bank.to_dict()
    payload["records"][0]["spec"]["version"] = "tampered"
    assert bank.records[0].spec.version == "1"
    with pytest.raises(ValueError, match="fingerprint"):
        FrozenOperatorBank.from_dict(payload)


def test_candidate_cannot_be_published_by_changing_json_without_detection(record):
    bank = FrozenOperatorBank("protocol-a", ("source-1",), (record,))
    payload = bank.to_dict()
    payload["records"][0]["status"] = "validated"
    payload["records"][0]["validation_ref"] = "unverified-declaration"
    with pytest.raises(ValueError, match="fingerprint"):
        FrozenOperatorBank.from_dict(payload)


@pytest.mark.parametrize("version", [True, "1", 2])
def test_bank_schema_version_is_exact_integer(record, version):
    payload = FrozenOperatorBank("protocol-a", ("source-1",), (record,)).to_dict()
    payload["schema_version"] = version
    with pytest.raises(ValueError, match="schema version"):
        FrozenOperatorBank.from_json(json.dumps(payload))


def test_snapshot_rejects_unknown_gold_field_even_with_valid_hash(record):
    payload = FrozenOperatorBank("protocol-a", ("source-1",), (record,)).to_dict()
    payload["gold"] = {"test-1": "answer"}
    with pytest.raises(ValueError, match="exactly"):
        FrozenOperatorBank.from_dict(payload)
