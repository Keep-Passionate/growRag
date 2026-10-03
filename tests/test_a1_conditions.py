"""Mechanical A1 tests only: hand-authored wire is not a semantic model result."""

import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from growrag.experiments import a1_conditions as a1

FIXTURE = Path(__file__).resolve().parents[1] / "experiments/fixtures/a1_conditions_v2.json"


@pytest.fixture
def rows():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"]


@pytest.fixture
def payload(rows):
    return deepcopy(next(row["input"] for row in rows if row["row_id"] == "14A"))


def wire(value):
    return json.dumps(value, ensure_ascii=False)


def ref(source, text, field="text"):
    return {"source_id": source, "field": field, "quote": text}


def unknown_response(prepared):
    reasons = {
        "T": "insufficient_type_evidence",
        "R": "ambiguous_subgoal",
        "P1": "missing_role_evidence",
    }
    return {
        "checks": [
            {"kind": kind, "slot": slot, "status": "unknown", "reason": reasons[kind], "refs": []}
            for kind, slot in prepared.semantic_checks
        ]
    }


def example_response(payload):
    q = ref("Q", payload["original_question"])
    e = ref("E1", payload["visible_evidence"][0]["text"])
    action = payload["proposal"]["compiled_queries"][0]
    a = ref(action["source_id"], action["query"], "query")
    return {
        "checks": [
            {
                "kind": "T",
                "slot": "gap.entity",
                "status": "supported",
                "reason": "expected_type_supported",
                "refs": [e],
            },
            {
                "kind": "R",
                "slot": "action.lookup",
                "status": "supported",
                "reason": "direct_goal_relation",
                "refs": [q, a],
            },
            {
                "kind": "P1",
                "slot": "gap.entity",
                "status": "supported",
                "reason": "target_role_supported",
                "refs": [e],
            },
        ]
    }


def body(payload, text, *, title="Current synthetic evidence", start=0, original=None):
    payload["visible_evidence"] = [
        {
            "source_id": "E1",
            "title": title,
            "text": text,
            "window": {
                "text_start": start,
                "text_end": start + len(text),
                "original_text_chars": original if original is not None else start + len(text),
            },
        }
    ]


def local_p0(prepared, slot="gap.entity"):
    return next(row for row in prepared.local_checks if row["kind"] == "P0" and row["slot"] == slot)


def test_prepared_canonical_hashes_and_detached_views(payload):
    snapshot = {"card_id": "synthetic", "source_sha": "local-only", "nested": [1, {"a": 2}]}
    prepared = a1.prepare_payload(payload, snapshot)
    assert prepared.input_sha256 == hashlib.sha256(a1.canonical_bytes(payload)).hexdigest()
    assert prepared.snapshot["input"] == payload
    assert prepared.snapshot["execution_snapshot"] == snapshot
    before = prepared.to_dict()
    payload["proposal"]["gap"]["entity"] = "MODIFIED"
    snapshot["nested"][1]["a"] = 99
    prepared.payload["original_question"] = "MODIFIED"
    prepared.snapshot["execution_snapshot"]["nested"].clear()
    assert prepared.to_dict() == before
    with pytest.raises(FrozenInstanceError):
        prepared.input_sha256 = "0" * 64
    assert a1.prepare_payload(wire(before["payload"])).input_sha256 == prepared.input_sha256


def test_prompt_single_request_no_fixture_labels_or_local_identity(payload):
    prepared = a1.prepare_payload(payload, {"expected": "LOCAL_ONLY", "secret_label": "HIDDEN"})
    messages = a1.observer_messages(prepared)
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": a1.OBSERVER_PROMPT}
    assert json.loads(messages[1]["content"]) == payload
    assert prepared.input_sha256 not in wire(messages)
    assert prepared.snapshot_sha256 not in wire(messages)
    for value in (
        "LOCAL_ONLY",
        "HIDDEN",
        "nominal_vector",
        "pair_id",
        "row_id",
        "Glass Harbor",
        "Nora Venn",
    ):
        assert value not in a1.OBSERVER_PROMPT
    assert "source_id, field, quote" in a1.OBSERVER_PROMPT
    assert "No closed-world assumption" in a1.OBSERVER_PROMPT
    assert "intermediate subgoals" in a1.OBSERVER_PROMPT
    assert a1.PROMPT_SHA256 == hashlib.sha256(a1.OBSERVER_PROMPT.encode()).hexdigest()
    messages[1]["content"] = "changed"
    assert a1.observer_messages(prepared)[1]["content"] != "changed"


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(answer="FORBIDDEN"),
        lambda p: p.update(schema_version="wrong"),
        lambda p: p.update(remaining_retrievals=True),
        lambda p: p.update(remaining_retrievals=2),
        lambda p: p.update(remaining_retrievals=0),
        lambda p: p.update(previous_queries="query"),
        lambda p: p.update(previous_queries=[3]),
        lambda p: p["visible_evidence"][0].update(source_id="e1"),
        lambda p: p["visible_evidence"][0].update(source_id="E01"),
        lambda p: p["visible_evidence"].append(deepcopy(p["visible_evidence"][0])),
        lambda p: p["visible_evidence"][0].update(hidden_tail="FORBIDDEN"),
        lambda p: p["visible_evidence"][0]["window"].update(text_start=True),
        lambda p: p["visible_evidence"][0]["window"].update(text_end=999),
        lambda p: p["proposal"].update(spec_version=1),
        lambda p: p["proposal"].update(goal="FORBIDDEN"),
        lambda p: p["proposal"]["gap"].update(entity=2.5),
        lambda p: p["proposal"]["gap"].update(entity=True),
        lambda p: p["proposal"]["gap"].update(entity=None),
        lambda p: p["proposal"]["gap"].update(entity="  "),
        lambda p: p["proposal"]["gap"].update(original_question="bad"),
        lambda p: p["proposal"].update(compiled_queries=[]),
        lambda p: p["proposal"]["compiled_queries"].append(
            deepcopy(p["proposal"]["compiled_queries"][0])
        ),
        lambda p: p["proposal"]["compiled_queries"][0].update(step_id="x.y"),
        lambda p: p["proposal"]["compiled_queries"][0].update(source_id="E1"),
        lambda p: p["slot_contracts"].pop(),
        lambda p: p["slot_contracts"].append(deepcopy(p["slot_contracts"][0])),
        lambda p: p["slot_contracts"][0].update(slot="gap.unregistered"),
        lambda p: p["slot_contracts"][0].update(kind="made_up"),
        lambda p: p["slot_contracts"][0].update(required_type=False),
        lambda p: p["slot_contracts"][0].update(requires_current_source=0),
        lambda p: p["slot_contracts"][1].update(requires_current_source=None),
        lambda p: p["slot_contracts"][1].update(required_type="relation"),
        lambda p: p["required_checks"].pop(),
        lambda p: p["required_checks"].append(deepcopy(p["required_checks"][0])),
        lambda p: p["required_checks"][0].update(slot="gap.missing"),
        lambda p: p["required_checks"][0].update(kind="X"),
    ],
)
def test_payload_strict_shapes_types_coverage_and_budget(payload, change):
    change(payload)
    with pytest.raises(a1.ConditionContractError):
        a1.prepare_payload(payload)


@pytest.mark.parametrize(
    "raw",
    [
        '{"checks":[],"checks":[]}',
        '{"checks":NaN}',
        '{"checks":Infinity}',
        '{"checks":-Infinity}',
        '```json\n{"checks":[]}\n```',
        "[]",
        "{}",
    ],
)
def test_raw_response_strict_json_preserves_failures(payload, raw):
    with pytest.raises(a1.ConditionContractError) as caught:
        a1.resolve_response(raw, a1.prepare_payload(payload))
    assert caught.value.raw_text == raw


def test_duplicate_input_keys_non_json_and_invalid_unicode_rejected(payload):
    with pytest.raises(a1.ConditionContractError, match="duplicate JSON key"):
        a1.prepare_payload('{"schema_version":"x","schema_version":"y"}')
    for value in ((1, 2), {1: "not-a-string-key"}, {"a": float("nan")}):
        with pytest.raises(a1.ConditionContractError):
            a1.prepare_payload(payload, value)
    payload["original_question"] = "\ud800"
    with pytest.raises(a1.ConditionContractError):
        a1.prepare_payload(payload)


def test_fixed_null_type_and_role_are_local_not_model_checks(payload):
    payload["slot_contracts"][0].update(required_type=None, target_role=None)
    prepared = a1.prepare_payload(payload)
    assert prepared.semantic_checks == (("R", "action.lookup"),)
    report = a1.resolve_response(wire(unknown_response(prepared)), prepared)
    checks = {(row["kind"], row["slot"]): row for row in report.checks}
    assert checks["T", "gap.entity"]["reason"] == "missing_type_requirement"
    assert checks["P1", "gap.entity"]["reason"] == "ambiguous_role"
    assert checks["T", "gap.entity"]["producer"] == "local"
    response = unknown_response(prepared)
    response["checks"].append(example_response(payload)["checks"][0])
    with pytest.raises(a1.ConditionContractError, match="local-only"):
        a1.resolve_response(wire(response), prepared)


def test_p0_false_is_not_applicable_not_unknown_and_t_p1_remain(payload):
    payload["slot_contracts"][0]["requires_current_source"] = False
    payload["required_checks"] = [row for row in payload["required_checks"] if row["kind"] != "P0"]
    prepared = a1.prepare_payload(payload)
    report = a1.resolve_response(wire(example_response(payload)), prepared)
    assert {row["kind"] for row in report.checks} == {"T", "R", "P1"}
    omitted = next(
        row for row in report.coverage if row["kind"] == "P0" and row["slot"] == "gap.entity"
    )
    assert omitted == {
        "kind": "P0",
        "slot": "gap.entity",
        "producer": "local",
        "applicability": "not_applicable",
        "reason": "source_not_required",
    }
    assert "status" not in omitted
    decision = a1.decide(report, ("P0",), strict_unknown=True)
    assert decision.decision == "allow" and decision.considered_checks == 0
    assert decision.not_applicable_checks > 0
    assert a1.decide(report, strict_unknown=True).decision == "allow"
    payload["required_checks"].append({"kind": "P0", "slot": "gap.entity"})
    with pytest.raises(a1.ConditionContractError, match="coverage"):
        a1.prepare_payload(payload)


def test_p0_null_unknown_retains_strict_policy(payload):
    payload["slot_contracts"][0]["requires_current_source"] = None
    prepared = a1.prepare_payload(payload)
    assert local_p0(prepared)["status"] == "unknown"
    assert local_p0(prepared)["reason"] == "source_requirement_unknown"
    report = a1.resolve_response(wire(example_response(payload)), prepared)
    assert a1.decide(report).decision == "allow_uncertain"
    assert a1.decide(report, strict_unknown=True).decision == "reject"
    assert a1.decide(report, ("T",)).decision == "allow"
    assert a1.decide(report, ()).to_dict()["decision"] == "allow"


@pytest.mark.parametrize(
    "kind,reason", [("free_query", "uncovered_free_query"), ("unknown", "unclassified_slot")]
)
def test_unclassified_query_is_explicit_local_unknown_not_empty_success(payload, kind, reason):
    payload["proposal"]["gap"] = {"query_text": "current search"}
    payload["proposal"]["template"] = "{query_text}"
    payload["slot_contracts"] = [
        {"slot": "gap.query_text", "kind": kind, "requires_current_source": None}
    ]
    payload["required_checks"] = [{"kind": k, "slot": "gap.query_text"} for k in ("T", "P0", "P1")]
    payload["required_checks"].append({"kind": "R", "slot": "action.lookup"})
    prepared = a1.prepare_payload(payload)
    assert prepared.semantic_checks == (("R", "action.lookup"),)
    assert len(prepared.local_checks) == 3
    assert all(
        row["reason"] == reason and row["status"] == "unknown" for row in prepared.local_checks
    )
    report = a1.resolve_response(wire(unknown_response(prepared)), prepared)
    assert a1.decide(report, ("T",)).decision == "allow_uncertain"
    assert a1.decide(report, ("T",), strict_unknown=True).decision == "reject"
    payload["slot_contracts"][0]["requires_current_source"] = False
    with pytest.raises(a1.ConditionContractError):
        a1.prepare_payload(payload)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Joanne is a person.", "contradicted"),
        ("Ann is a person.", "supported"),
        ("ANN is a person.", "supported"),
        ("XAnnY is mentioned.", "contradicted"),
    ],
)
def test_p0_word_boundaries_and_query_not_provenance(payload, text, expected):
    payload["proposal"]["gap"]["entity"] = "Ann"
    payload["proposal"]["compiled_queries"][0]["query"] = "Ann birth year"
    body(payload, text)
    prepared = a1.prepare_payload(payload)
    check = local_p0(prepared)
    assert check["status"] == expected
    assert check["search"]["input_sha256"] == prepared.input_sha256
    assert all(not row["source_id"].startswith("A") for row in check["search"]["fields"])
    if expected == "contradicted":
        assert check["refs"] == []


def test_p0_unicode_expansion_whitespace_maps_back_to_original_visible_field(payload):
    text = "😀 Straße\t \nAnn remains visible."
    payload["proposal"]["gap"]["entity"] = "STRASSE ann"
    body(payload, text, start=100, original=500)
    check = local_p0(a1.prepare_payload(payload))
    assert check["status"] == "supported"
    item = check["refs"][0]
    assert item["quote"] == "Straße\t \nAnn"
    assert item["start"] == 2
    assert text[item["start"] : item["end"]] == item["quote"]
    assert a1._normalize_with_spans(text)[0] == " ".join(text.casefold().split())


def test_p0_title_question_and_multiple_occurrences_are_local_allowed(payload):
    body(payload, "No role facts here.", title="Nora Venn; Nora Venn")
    check = local_p0(a1.prepare_payload(payload))
    assert len(check["refs"]) == 2
    assert all(row["field"] == "title" for row in check["refs"])
    body(payload, "No facts here.", title="")
    payload["original_question"] = "When was Nora Venn born?"
    check = local_p0(a1.prepare_payload(payload))
    assert check["refs"][0]["source_id"] == "Q"


def test_integer_literal_slot_is_not_cast_and_each_slot_has_checks(payload):
    payload["proposal"]["gap"]["year"] = 1990
    body(payload, "The work was published in 1990. Nora Venn is a person.")
    payload["slot_contracts"].append(
        {
            "slot": "gap.year",
            "kind": "literal_constraint",
            "required_type": None,
            "requires_current_source": True,
            "target_role": None,
        }
    )
    payload["required_checks"].extend(
        {"kind": kind, "slot": "gap.year"} for kind in ("T", "P0", "P1")
    )
    prepared = a1.prepare_payload(payload)
    assert type(prepared.payload["proposal"]["gap"]["year"]) is int
    assert local_p0(prepared, "gap.year")["refs"][0]["quote"] == "1990"
    assert ("P0", "gap.entity") in prepared.required_checks
    assert ("P0", "gap.year") in prepared.required_checks
    assert ("T", "gap.year") not in prepared.semantic_checks


def test_binding_slots_and_exact_citation_aliases(payload):
    value = payload["proposal"]["gap"].pop("entity")
    payload["proposal"]["bindings"] = [{"name": "entity", "value": value, "evidence_ids": ["E1"]}]
    payload["slot_contracts"][0]["slot"] = "binding.entity"
    for row in payload["required_checks"]:
        if row["slot"] == "gap.entity":
            row["slot"] = "binding.entity"
    prepared = a1.prepare_payload(payload)
    assert local_p0(prepared, "binding.entity")["status"] == "supported"
    payload["proposal"]["bindings"][0]["evidence_ids"] = ["Q"]
    with pytest.raises(a1.ConditionContractError, match="binding evidence"):
        a1.prepare_payload(payload)


def test_success_report_preserves_raw_and_is_detached(payload):
    prepared = a1.prepare_payload(payload)
    response = example_response(payload)
    raw = " \n" + json.dumps(response, ensure_ascii=False, indent=2) + "\t\n"
    report = a1.resolve_response(raw, prepared)
    assert report.raw_text == raw
    assert report.to_dict()["response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert report.to_dict()["input_sha256"] == prepared.input_sha256
    assert report.to_dict()["snapshot_sha256"] == prepared.snapshot_sha256
    assert [row["kind"] for row in report.checks] == ["T", "R", "P0", "P1"]
    report.checks[0]["status"] = "MODIFIED"
    report.to_dict()["checks"].clear()
    assert len(report.checks) == 4
    assert a1.decide(report).decision == "allow"
    assert a1.decide(report).decisive_checks == ()


@pytest.mark.parametrize("quote", ["", " \t", "nora venn", "absent quote", "Nora Venn"])
def test_quote_empty_mismatch_or_ambiguous_is_not_repaired(payload, quote):
    prepared = a1.prepare_payload(payload)
    response = example_response(payload)
    response["checks"][0]["refs"][0]["quote"] = quote
    raw = wire(response)
    with pytest.raises(a1.ConditionContractError) as caught:
        a1.resolve_response(raw, prepared)
    assert caught.value.raw_text == raw


def test_overlapping_quote_occurrences_fail(payload):
    body(payload, "aaa")
    response = example_response(payload)
    response["checks"][0]["refs"][0]["quote"] = "aa"
    with pytest.raises(a1.ConditionContractError, match="overlapping"):
        a1.resolve_response(wire(response), a1.prepare_payload(payload))


def test_unicode_codepoint_exact_quote_and_untrimmed_whitespace(payload):
    text = '😀 café\n"Alpha"\\path.  End'
    body(payload, text)
    response = example_response(payload)
    quote = 'café\n"Alpha"\\path.  End'
    response["checks"][0]["refs"][0]["quote"] = quote
    report = a1.resolve_response(wire(response), a1.prepare_payload(payload))
    found = report.checks[0]["refs"][0]
    assert found["start"] == 2 and found["end"] == 2 + len(quote)
    assert text[found["start"] : found["end"]] == quote
    for bad in (quote.replace("é", "e\u0301"), quote.replace("\n", " "), quote.replace("  ", " ")):
        response["checks"][0]["refs"][0]["quote"] = bad
        with pytest.raises(a1.ConditionContractError):
            a1.resolve_response(wire(response), a1.prepare_payload(payload))


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(input_sha256="0" * 64),
        lambda r: r["checks"].pop(),
        lambda r: r["checks"].append(deepcopy(r["checks"][0])),
        lambda r: r["checks"][0].update(kind="P0"),
        lambda r: r["checks"][0].update(status=True),
        lambda r: r["checks"][0].update(reason="current_value_occurs"),
        lambda r: r["checks"][0].update(status="unknown", reason="missing_type_requirement"),
        lambda r: r["checks"][0]["refs"][0].update(start=0),
        lambda r: r["checks"][0]["refs"][0].update(source_id="e1"),
        lambda r: r["checks"][0]["refs"][0].update(source_id="E01"),
        lambda r: r["checks"][0]["refs"][0].update(source_id="E9"),
        lambda r: r["checks"][0]["refs"][0].update(field="body"),
        lambda r: r["checks"][0]["refs"].append(deepcopy(r["checks"][0]["refs"][0])),
        lambda r: r["checks"][0].update(refs=[]),
        lambda r: r["checks"][1].update(refs=r["checks"][1]["refs"][:1]),
        lambda r: r["checks"][1].update(refs=r["checks"][1]["refs"][1:]),
        lambda r: r["checks"][1].update(reason="permitted_intermediate_subgoal"),
    ],
)
def test_response_fields_sources_status_reasons_and_required_refs(payload, change):
    response = example_response(payload)
    change(response)
    with pytest.raises(a1.ConditionContractError):
        a1.resolve_response(wire(response), a1.prepare_payload(payload))


def test_action_query_cannot_be_type_or_role_provenance(payload):
    response = example_response(payload)
    action_ref = response["checks"][1]["refs"][1]
    for index in (0, 2):
        changed = deepcopy(response)
        changed["checks"][index]["refs"] = [action_ref]
        with pytest.raises(a1.ConditionContractError, match="not T/P1 provenance"):
            a1.resolve_response(wire(changed), a1.prepare_payload(payload))


def test_quote_uniqueness_is_per_field_and_not_hidden_tail(payload):
    text = "Visible only."
    body(payload, text, title=text, original=200)
    prepared = a1.prepare_payload(payload, {"full_document": text + " Hidden person sentence."})
    response = example_response(payload)
    a1.resolve_response(wire(response), prepared)  # Same quote in title is not body ambiguity.
    response["checks"][0]["refs"][0]["quote"] = "Hidden person sentence."
    with pytest.raises(a1.ConditionContractError, match="exactly once"):
        a1.resolve_response(wire(response), prepared)


def test_conflict_requires_two_distinct_current_passages(payload):
    response = example_response(payload)
    response["checks"][2].update(status="unknown", reason="conflicting_evidence")
    with pytest.raises(a1.ConditionContractError, match="two distinct"):
        a1.resolve_response(wire(response), a1.prepare_payload(payload))
    copy = deepcopy(payload["visible_evidence"][0])
    copy["source_id"] = "E2"
    payload["visible_evidence"].append(copy)
    response["checks"][2]["refs"].append(ref("E2", copy["text"]))
    with pytest.raises(a1.ConditionContractError, match="two distinct"):
        a1.resolve_response(wire(response), a1.prepare_payload(payload))
    copy["text"] = "The target role is explicitly denied."
    copy["window"].update(text_end=len(copy["text"]), original_text_chars=len(copy["text"]))
    response["checks"][2]["refs"][1]["quote"] = copy["text"]
    report = a1.resolve_response(wire(response), a1.prepare_payload(payload))
    assert a1.decide(report).decision == "allow_uncertain"
    assert a1.decide(report, strict_unknown=True).decision == "reject"


@pytest.mark.parametrize(
    "kind,status,reason",
    [
        (kind, status, reason)
        for kind, states in a1._REASONS.items()
        for status, reasons in states.items()
        for reason in reasons
    ],
)
def test_each_registered_model_reason_accepts_only_its_registered_shape(
    payload, kind, status, reason
):
    text = "A second, incompatible current statement."
    payload["visible_evidence"].append(
        {
            "source_id": "E2",
            "title": "Other current source",
            "text": text,
            "window": {"text_start": 0, "text_end": len(text), "original_text_chars": len(text)},
        }
    )
    response = example_response(payload)
    check = next(row for row in response["checks"] if row["kind"] == kind)
    check.update(status=status, reason=reason)
    if reason == "conflicting_evidence":
        check["refs"] = [ref("E1", payload["visible_evidence"][0]["text"]), ref("E2", text)]
    elif reason == "permitted_intermediate_subgoal":
        check["refs"].append(ref("E1", payload["visible_evidence"][0]["text"]))
    elif status == "unknown":
        check["refs"] = []
    report = a1.resolve_response(wire(response), a1.prepare_payload(payload))
    assert next(row for row in report.checks if row["kind"] == kind)["reason"] == reason
    check["reason"] = "made_up_reason"
    with pytest.raises(a1.ConditionContractError, match="reason/status"):
        a1.resolve_response(wire(response), a1.prepare_payload(payload))


def test_prepared_tamper_and_report_tamper_fail_before_reuse(payload):
    prepared = a1.prepare_payload(payload)
    with pytest.raises(a1.ConditionContractError, match="SHA"):
        a1.observer_messages(replace(prepared, input_sha256="0" * 64))
    changed = prepared.snapshot
    changed["input"]["original_question"] = "Another question"
    changed_bytes = a1.canonical_bytes(changed)
    changed_prepared = replace(
        prepared,
        snapshot_bytes=changed_bytes,
        snapshot_sha256=hashlib.sha256(changed_bytes).hexdigest(),
    )
    raw = wire(example_response(payload))
    with pytest.raises(a1.ConditionContractError, match="binding") as caught:
        a1.resolve_response(raw, changed_prepared)
    assert caught.value.raw_text == raw
    report = a1.resolve_response(raw, prepared)
    with pytest.raises(a1.ConditionContractError, match="SHA"):
        a1.decide(replace(report, report_sha256="0" * 64))


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["checks"].clear(),
        lambda r: r["checks"].pop(),
        lambda r: r["coverage"].clear(),
        lambda r: r["required_checks"].clear(),
        lambda r: r["checks"][0].update(status="contradicted", reason="expected_type_conflict"),
        lambda r: r["checks"][0]["refs"][0].update(start=999),
        lambda r: r.update(raw_text='{"checks":[]}'),
        lambda r: r.update(input_sha256="0" * 64),
    ],
)
def test_rehashed_report_tampering_fails_source_rederivation(payload, change):
    prepared = a1.prepare_payload(payload)
    report = a1.resolve_response(wire(example_response(payload)), prepared)
    changed = json.loads(report.report_bytes)
    change(changed)
    data = a1.canonical_bytes(changed)
    forged = replace(report, report_bytes=data, report_sha256=hashlib.sha256(data).hexdigest())
    for operation in (forged.to_dict, lambda: a1.decide(forged), lambda: a1.decide(forged, ())):
        with pytest.raises(a1.ConditionContractError, match="binding"):
            operation()


def test_fabricated_empty_report_with_correct_self_hash_is_not_validated(payload):
    prepared = a1.prepare_payload(payload)
    data = a1.canonical_bytes({"schema_version": a1.REPORT_VERSION, "checks": [], "coverage": []})
    forged = a1.ObservationReport(
        data, hashlib.sha256(data).hexdigest(), prepared, wire(example_response(payload))
    )
    with pytest.raises(a1.ConditionContractError, match="binding"):
        a1.decide(forged)
    with pytest.raises(a1.ConditionContractError, match="missing semantic checks"):
        a1.decide(replace(forged, response_text='{"checks":[]}'))


def test_report_frozen_source_bindings_and_changed_prepared_input(payload):
    report = a1.resolve_response(wire(example_response(payload)), a1.prepare_payload(payload))
    with pytest.raises(FrozenInstanceError):
        report.response_text = '{"checks":[]}'
    payload["original_question"] += " Extra instruction."
    with pytest.raises(a1.ConditionContractError, match="binding"):
        a1.decide(replace(report, prepared=a1.prepare_payload(payload)))


def test_dimension_replay_without_model_and_rejection_precedence(payload):
    response = example_response(payload)
    response["checks"][0].update(status="unknown", reason="insufficient_type_evidence", refs=[])
    response["checks"][2].update(status="contradicted", reason="target_role_negated")
    report = a1.resolve_response(wire(response), a1.prepare_payload(payload))
    assert a1.decide(report).decision == "reject"
    assert a1.decide(report).decisive_checks == (("P1", "gap.entity"),)
    assert a1.decide(report, ("T",)).decision == "allow_uncertain"
    assert a1.decide(report, ("T",), strict_unknown=True).decision == "reject"
    assert a1.decide(report, ("R", "P0")).decision == "allow"
    assert a1.decide(report, ()).decision == "allow"
    for bad in ("T", ["T", "T"], ["bad"], [1]):
        with pytest.raises(a1.ConditionContractError):
            a1.decide(report, bad)
    with pytest.raises(a1.ConditionContractError):
        a1.decide(report, strict_unknown=1)


def test_all_32_nominal_fixture_rows_mechanically_resolve_not_model_predictions(rows):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for row in rows:
        payload, expected = row["input"], row["expected"]
        prepared = a1.prepare_payload(payload)
        assert (
            prepared.input_sha256 == fixture["integrity"]["input_sha256_by_row_id"][row["row_id"]]
        )
        response = example_response(payload)
        by_kind = {check["kind"]: check for check in response["checks"]}
        for nominal in expected["checks"]:
            kind, status = nominal["kind"], nominal["status"]
            if kind == "P0":
                continue
            check = by_kind[kind]
            check["status"] = status
            if kind == "T":
                check["reason"] = {
                    "supported": "expected_type_supported",
                    "contradicted": "expected_type_conflict",
                    "unknown": "insufficient_type_evidence",
                }[status]
            elif kind == "R":
                check["reason"] = (
                    "relation_conflict" if status == "contradicted" else "direct_goal_relation"
                )
            else:
                check["reason"] = {
                    "supported": "target_role_supported",
                    "contradicted": "exclusive_role_conflict",
                    "unknown": "missing_role_evidence",
                }[status]
            if status == "unknown":
                check["refs"] = []
        if row["row_id"] == "14B":
            by_kind["P1"].update(
                reason="conflicting_evidence",
                refs=[ref(e["source_id"], e["text"]) for e in payload["visible_evidence"]],
            )
        report = a1.resolve_response(wire(response), prepared)
        assert [
            {key: check[key] for key in ("kind", "slot", "status")} for check in report.checks
        ] == expected["checks"]
        assert a1.decide(report).decision == expected["contradiction_only"]
        assert a1.decide(report, strict_unknown=True).decision == expected["strict_unknown_reject"]
