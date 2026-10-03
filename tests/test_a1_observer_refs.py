"""Catalog-v3 mechanics only; authored responses are not model semantic results."""

import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from growrag.experiments import a1_conditions as conditions
from growrag.experiments import a1_observer_refs as refs


def wire(value):
    return json.dumps(value, ensure_ascii=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def payload():
    text = "Mira Quell is a person and the sole cartographer of Estuary Ledger."
    return {
        "schema_version": conditions.INPUT_VERSION,
        "original_question": "When was the cartographer of Estuary Ledger born?",
        "visible_evidence": [
            {
                "source_id": "E1",
                "title": "Estuary Ledger",
                "text": text,
                "window": {
                    "text_start": 0,
                    "text_end": len(text),
                    "original_text_chars": len(text),
                },
            }
        ],
        "previous_queries": ["Estuary Ledger cartographer"],
        "remaining_retrievals": 1,
        "proposal": {
            "spec_id": "synthetic_catalog_contract",
            "spec_version": "1",
            "template": "{entity} birth year",
            "gap": {"entity": "Mira Quell"},
            "bindings": [],
            "compiled_queries": [
                {"step_id": "lookup", "source_id": "A1", "query": "Mira Quell birth year"}
            ],
        },
        "slot_contracts": [
            {
                "slot": "gap.entity",
                "kind": "entity",
                "required_type": "person",
                "requires_current_source": True,
                "target_role": "cartographer of Estuary Ledger",
            }
        ],
        "required_checks": [
            {"kind": kind, "slot": "action.lookup" if kind == "R" else "gap.entity"}
            for kind in ("T", "R", "P0", "P1")
        ],
    }


def ref_id(prepared, source, field="text"):
    return next(
        row["ref_id"]
        for row in refs.catalog(prepared)
        if (row["source_id"], row["field"]) == (source, field)
    )


def response(prepared):
    reasons = {
        "T": "expected_type_supported",
        "R": "direct_goal_relation",
        "P1": "target_role_supported",
    }
    return {
        "checks": [
            {
                "kind": kind,
                "slot": slot,
                "status": "supported",
                "reason": reasons[kind],
                "ref_ids": [ref_id(prepared, "Q"), ref_id(prepared, "A1", "query")]
                if kind == "R"
                else [ref_id(prepared, "E1")],
            }
            for kind, slot in prepared.semantic_checks
        ]
    }


def set_text(payload, text):
    evidence = payload["visible_evidence"][0]
    evidence["text"] = text
    evidence["window"] = {
        "text_start": 7,
        "text_end": 7 + len(text),
        "original_text_chars": 7 + len(text) + 100,
    }


def test_catalog_is_complete_deterministic_and_detached(payload):
    prepared = conditions.prepare_payload(payload, {"hidden": "Never cite this tail."})
    entries = refs.catalog(prepared)
    assert [(row["ref_id"], row["source_id"], row["field"]) for row in entries] == [
        ("r1", "Q", "text"),
        ("r2", "E1", "title"),
        ("r3", "E1", "text"),
        ("r4", "A1", "query"),
    ]
    assert [row["quote"] for row in entries] == [
        payload["original_question"],
        payload["visible_evidence"][0]["title"],
        payload["visible_evidence"][0]["text"],
        payload["proposal"]["compiled_queries"][0]["query"],
    ]
    assert refs.catalog(prepared) == entries
    entries[0]["quote"] = "Changed"
    payload["original_question"] = "Changed"
    assert refs.catalog(prepared)[0]["quote"] != "Changed"
    assert "Never cite this tail" not in wire(refs.request(prepared))


def test_blank_fields_skipped_without_stripping_real_quotes_or_sorting_aliases(payload):
    payload["visible_evidence"][0]["source_id"] = "E9"
    payload["visible_evidence"][0]["title"] = " \t\n"
    set_text(payload, "  😀 repeated repeated \n")
    second = deepcopy(payload["visible_evidence"][0])
    second.update(source_id="E2", title="Visible title", text="")
    second["window"] = {"text_start": 0, "text_end": 0, "original_text_chars": 0}
    payload["visible_evidence"].append(second)
    entries = refs.catalog(conditions.prepare_payload(payload))
    assert [(row["ref_id"], row["source_id"], row["field"]) for row in entries] == [
        ("r1", "Q", "text"),
        ("r2", "E9", "text"),
        ("r3", "E2", "title"),
        ("r4", "A1", "query"),
    ]
    assert entries[1]["quote"] == "  😀 repeated repeated \n"


def test_request_has_original_input_metadata_only_catalog_and_exact_model_checks(payload):
    prepared = conditions.prepare_payload(payload, {"expected": "local sentinel"})
    visible = refs.request(prepared)
    assert set(visible) == {"input", "ref_catalog", "model_checks"}
    assert visible["input"] == payload
    assert len(visible["input"]) == 8
    assert all(set(row) == {"ref_id", "source_id", "field"} for row in visible["ref_catalog"])
    assert [(row["kind"], row["slot"]) for row in visible["model_checks"]] == list(
        prepared.semantic_checks
    )
    relational = next(row for row in visible["model_checks"] if row["kind"] == "R")
    rules = relational["reference_rules"]
    assert rules["supported_or_contradicted"]["must_include"] == ["r1", "r4"]
    assert rules["permitted_intermediate_subgoal_additionally"]["at_least_one_of"] == ["r2", "r3"]
    for row in visible["model_checks"]:
        if row["kind"] != "R":
            assert "r4" not in row["allowed_ref_ids"]
    sent = refs.messages(prepared)
    assert sent == [
        {"role": "system", "content": refs.OBSERVER_PROMPT},
        {"role": "user", "content": conditions.canonical_bytes(visible).decode("utf-8")},
    ]
    for excluded in (prepared.input_sha256, prepared.snapshot_sha256, "local sentinel"):
        assert excluded not in wire(sent)
    assert sent[1]["content"].count(payload["visible_evidence"][0]["text"]) == 1
    assert "Mira Quell" not in refs.OBSERVER_PROMPT
    assert "Nora Venn" not in refs.OBSERVER_PROMPT


def test_semantic_instructions_reused_verbatim_but_quote_wire_not_requested():
    semantic = (
        "T: assess" + conditions.OBSERVER_PROMPT.split("T: assess", 1)[1].split("\nstatus is", 1)[0]
    )
    assert semantic in refs.OBSERVER_PROMPT
    assert "Each ref has exactly source_id, field, quote" not in refs.OBSERVER_PROMPT
    assert "never fills in references" in refs.OBSERVER_PROMPT
    assert refs.WIRE_VERSION == "growrag-a1-observer-catalog-v3"
    assert refs.PROMPT_SHA256 == digest(refs.OBSERVER_PROMPT.encode())


def test_resolution_preserves_original_wire_and_exact_conversion_audit(payload):
    prepared = conditions.prepare_payload(payload)
    value = response(prepared)
    raw = " \n" + wire(value) + "\n "
    report, audit = refs.resolve(raw, prepared)
    content = audit.to_dict()
    assert content["raw_text"] == raw
    assert content["response_sha256"] == digest(raw.encode())
    assert content["catalog"] == refs.catalog(prepared)
    assert content["catalog_sha256"] == digest(conditions.canonical_bytes(content["catalog"]))
    assert content["request_sha256"] == digest(conditions.canonical_bytes(refs.request(prepared)))
    assert content["messages_sha256"] == digest(conditions.canonical_bytes(refs.messages(prepared)))
    assert content["prompt_sha256"] == refs.PROMPT_SHA256
    assert content["input_sha256"] == prepared.input_sha256
    assert content["snapshot_sha256"] == prepared.snapshot_sha256
    assert content["outcome"] == "valid_response" and content["error"] is None
    converted = json.loads(content["converted_raw_text"])
    lookup = {entry["ref_id"]: entry for entry in content["catalog"]}
    for source, derived in zip(value["checks"], converted["checks"], strict=True):
        assert {key: source[key] for key in ("kind", "slot", "status", "reason")} == {
            key: derived[key] for key in ("kind", "slot", "status", "reason")
        }
        assert derived["refs"] == [
            {key: lookup[rid][key] for key in ("source_id", "field", "quote")}
            for rid in source["ref_ids"]
        ]
    assert report.raw_text == content["converted_raw_text"] != raw
    assert content["converted_response_sha256"] == digest(report.raw_text.encode())
    assert content["derived_report_sha256"] == report.report_sha256
    assert report.to_dict()["wire_version"] == content["derived_report_wire_version"]
    assert content["derived_report_wire_version"] == conditions.WIRE_VERSION
    audit.verify_report(report)
    assert conditions.decide(report).decision == "allow"
    assert refs.resolve(raw, prepared)[1].audit_bytes == audit.audit_bytes


def test_whole_fields_resolve_repetition_unicode_and_whitespace_at_original_offsets(payload):
    text = '  Mira Quell 😀 repeated repeated café\n"X"\\path.  End  '
    set_text(payload, text)
    prepared = conditions.prepare_payload(payload)
    report, _ = refs.resolve(wire(response(prepared)), prepared)
    target = report.checks[0]["refs"][0]
    assert target == {
        "source_id": "E1",
        "field": "text",
        "quote": text,
        "start": 0,
        "end": len(text),
    }
    # Offsets refer to the visible field, not the hidden/full-document start of 7.
    assert target["quote"] == prepared.payload["visible_evidence"][0]["text"]


@pytest.mark.parametrize(
    "bad", ["r0", "r01", "R1", " r1", "r1 ", "r999", "[r1]", "r1.text", 1, True, None, {}, []]
)
def test_reference_spelling_unknown_and_type_fail_without_repair(payload, bad):
    prepared = conditions.prepare_payload(payload)
    value = response(prepared)
    value["checks"][0]["ref_ids"] = [bad]
    raw = wire(value)
    with pytest.raises(refs.CatalogContractError, match="non-exact") as caught:
        refs.resolve(raw, prepared)
    assert caught.value.raw_text == raw
    audit = caught.value.audit.to_dict()
    assert audit["raw_text"] == raw
    assert audit["outcome"] == "invalid_response"
    assert audit["converted_raw_text"] is None
    assert audit["derived_report_sha256"] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(input_sha256="0" * 64),
        lambda v: v.update(ref_catalog=[]),
        lambda v: v["checks"][0].update(refs=[]),
        lambda v: v["checks"][0].update(quote="quoted"),
        lambda v: v["checks"][0].update(ref_ids="r3"),
        lambda v: v["checks"][0].update(ref_ids=["r3", "r3"]),
        lambda v: v["checks"][0].update(kind="P0"),
        lambda v: v["checks"][0].update(slot="gap.missing"),
        lambda v: v["checks"][0].update(status="true"),
        lambda v: v["checks"][0].update(reason="made_up"),
        lambda v: v["checks"].pop(),
        lambda v: v["checks"].append(deepcopy(v["checks"][0])),
    ],
)
def test_wire_exact_shape_and_original_semantic_contract_remain_strict(payload, change):
    prepared = conditions.prepare_payload(payload)
    value = response(prepared)
    change(value)
    raw = wire(value)
    with pytest.raises(refs.CatalogContractError) as caught:
        refs.resolve(raw, prepared)
    assert caught.value.raw_text == raw
    content = caught.value.audit.to_dict()
    assert content["outcome"] == "invalid_response" and content["error"]
    assert content["derived_report_sha256"] is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"checks":[],"checks":[]}',
        '{"checks":[{"ref_ids":[],"ref_ids":[]}]}',
        '{"checks":NaN}',
        '{"checks":Infinity}',
        '```json\n{"checks":[]}\n```',
        'explanation {"checks":[]}',
        "[]",
        '{"checks":[]} trailing',
        "",
    ],
)
def test_json_duplicate_nonfinite_fences_and_trailing_text_not_repaired(payload, raw):
    with pytest.raises(refs.CatalogContractError) as caught:
        refs.resolve(raw, conditions.prepare_payload(payload))
    assert caught.value.raw_text == raw
    assert caught.value.audit.to_dict()["converted_raw_text"] is None


@pytest.mark.parametrize("missing", ["question", "action", "evidence"])
def test_r_required_sources_are_never_added_automatically(payload, missing):
    prepared = conditions.prepare_payload(payload)
    value = response(prepared)
    check = next(row for row in value["checks"] if row["kind"] == "R")
    if missing == "question":
        check["ref_ids"] = ["r4"]
    elif missing == "action":
        check["ref_ids"] = ["r1"]
    else:
        check["reason"] = "permitted_intermediate_subgoal"
    raw = wire(value)
    with pytest.raises(refs.CatalogContractError, match="R needs|needs current E") as caught:
        refs.resolve(raw, prepared)
    converted = json.loads(caught.value.audit.to_dict()["converted_raw_text"])
    assert len(converted["checks"][1]["refs"]) == len(check["ref_ids"])
    assert converted["checks"][1]["reason"] == check["reason"]


@pytest.mark.parametrize("kind", ["T", "P1"])
def test_action_cannot_be_current_type_or_role_provenance(payload, kind):
    prepared = conditions.prepare_payload(payload)
    value = response(prepared)
    next(row for row in value["checks"] if row["kind"] == kind)["ref_ids"] = ["r4"]
    with pytest.raises(refs.CatalogContractError, match="not T/P1 provenance"):
        refs.resolve(wire(value), prepared)


def test_unknown_no_refs_and_intermediate_r_with_all_sources(payload):
    prepared = conditions.prepare_payload(payload)
    value = response(prepared)
    value["checks"][0].update(status="unknown", reason="insufficient_type_evidence", ref_ids=[])
    value["checks"][1].update(reason="permitted_intermediate_subgoal", ref_ids=["r1", "r4", "r3"])
    report, _ = refs.resolve(wire(value), prepared)
    assert conditions.decide(report).decision == "allow_uncertain"
    assert conditions.decide(report, strict_unknown=True).decision == "reject"
    assert conditions.decide(report, ("R",)).decision == "allow"


def test_local_unknown_and_p0_not_applicable_remain_local(payload):
    slot = payload["slot_contracts"][0]
    slot.update(required_type=None, target_role=None, requires_current_source=False)
    payload["required_checks"] = [row for row in payload["required_checks"] if row["kind"] != "P0"]
    prepared = conditions.prepare_payload(payload)
    assert [(row["kind"], row["slot"]) for row in refs.request(prepared)["model_checks"]] == [
        ("R", "action.lookup")
    ]
    value = response(prepared)
    report, _ = refs.resolve(wire(value), prepared)
    assert [
        (row["kind"], row["reason"]) for row in report.checks if row["producer"] == "local"
    ] == [("T", "missing_type_requirement"), ("P1", "ambiguous_role")]
    assert conditions.decide(report, ("P0",)).not_applicable_checks == 1
    value["checks"].append(
        {
            "kind": "T",
            "slot": "gap.entity",
            "status": "supported",
            "reason": "expected_type_supported",
            "ref_ids": ["r3"],
        }
    )
    with pytest.raises(refs.CatalogContractError, match="local-only"):
        refs.resolve(wire(value), prepared)


def test_conflicting_evidence_needs_two_distinct_current_field_texts(payload):
    text = payload["visible_evidence"][0]["text"]
    payload["visible_evidence"][0]["title"] = text
    prepared = conditions.prepare_payload(payload)
    value = response(prepared)
    check = value["checks"][2]
    check.update(status="unknown", reason="conflicting_evidence", ref_ids=["r2", "r3"])
    # Equal text in different fields remains separate catalog candidates, but not a conflict.
    assert refs.catalog(prepared)[1]["quote"] == refs.catalog(prepared)[2]["quote"]
    with pytest.raises(refs.CatalogContractError, match="two distinct"):
        refs.resolve(wire(value), prepared)
    payload["visible_evidence"][0]["title"] = "Mira Quell is not the cartographer."
    report, _ = refs.resolve(wire(value), conditions.prepare_payload(payload))
    assert conditions.decide(report).decision == "allow_uncertain"
    check["ref_ids"] = ["r3"]
    with pytest.raises(refs.CatalogContractError, match="two distinct"):
        refs.resolve(wire(value), conditions.prepare_payload(payload))


def test_whole_field_cannot_express_two_distinct_intrafield_conflict_quotes(payload):
    set_text(payload, "Mira Quell is the cartographer. Mira Quell is not the cartographer.")
    prepared = conditions.prepare_payload(payload)
    value = response(prepared)
    value["checks"][2].update(status="unknown", reason="conflicting_evidence", ref_ids=["r3", "r3"])
    with pytest.raises(refs.CatalogContractError, match="duplicate catalog"):
        refs.resolve(wire(value), prepared)
    # The inherited v2 wire can represent these two nonidentical spans in the same field.
    converted = refs._convert(wire(response(prepared)), refs.catalog(prepared))
    quoted = json.loads(converted)
    quoted["checks"][2].update(
        status="unknown",
        reason="conflicting_evidence",
        refs=[
            {"source_id": "E1", "field": "text", "quote": sentence}
            for sentence in (
                "Mira Quell is the cartographer.",
                "Mira Quell is not the cartographer.",
            )
        ],
    )
    assert (
        conditions.decide(conditions.resolve_response(wire(quoted), prepared)).decision
        == "allow_uncertain"
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda audit: audit["catalog"][0].update(quote="Changed"),
        lambda audit: audit["request"]["model_checks"].clear(),
        lambda audit: audit.update(raw_text="Changed"),
        lambda audit: audit.update(converted_raw_text='{"checks":[]}'),
        lambda audit: audit.update(derived_report_sha256="0" * 64),
        lambda audit: audit.update(outcome="invalid_response", error="fabricated"),
    ],
)
def test_rehashed_audit_tamper_rejected_by_full_rederivation(payload, change):
    prepared = conditions.prepare_payload(payload)
    _, audit = refs.resolve(wire(response(prepared)), prepared)
    value = json.loads(audit.audit_bytes)
    change(value)
    data = conditions.canonical_bytes(value)
    with pytest.raises(conditions.ConditionContractError, match="binding"):
        replace(audit, audit_bytes=data, audit_sha256=digest(data)).to_dict()


def test_audit_immutable_detached_and_cross_report_binding(payload):
    prepared = conditions.prepare_payload(payload)
    value = response(prepared)
    report, audit = refs.resolve(wire(value), prepared)
    with pytest.raises(FrozenInstanceError):
        audit.response_text = "Changed"
    audit.to_dict()["catalog"].clear()
    assert audit.to_dict()["catalog"]
    with pytest.raises(conditions.ConditionContractError, match="SHA"):
        replace(audit, audit_sha256="0" * 64).to_dict()
    value["checks"][0].update(status="contradicted", reason="expected_type_conflict")
    other, _ = refs.resolve(wire(value), prepared)
    with pytest.raises(conditions.ConditionContractError, match="does not belong"):
        audit.verify_report(other)
    audit.verify_report(report)


def test_failure_audit_keeps_conversion_and_rejects_report_binding(payload):
    prepared = conditions.prepare_payload(payload)
    raw = '{"checks":[]}'
    with pytest.raises(refs.CatalogContractError, match="missing semantic") as caught:
        refs.resolve(raw, prepared)
    audit = caught.value.audit
    assert audit.to_dict()["converted_raw_text"] == raw
    report, _ = refs.resolve(wire(response(prepared)), prepared)
    with pytest.raises(conditions.ConditionContractError, match="does not belong"):
        audit.verify_report(report)


def test_invalid_prepared_or_non_text_response_never_runs_conversion(payload):
    prepared = conditions.prepare_payload(payload)
    with pytest.raises(refs.CatalogContractError, match="SHA") as caught:
        refs.resolve("raw retained", replace(prepared, snapshot_sha256="0" * 64))
    assert caught.value.raw_text == "raw retained"
    assert caught.value.audit is None
    for raw in (None, {}, b'{"checks":[]}', "\ud800"):
        with pytest.raises(refs.CatalogContractError) as caught:
            refs.resolve(raw, prepared)
        assert caught.value.raw_text == raw


@pytest.mark.parametrize("kind", ["free_query", "unknown"])
def test_unclassified_slots_remain_local_unknown_not_hidden_model_work(payload, kind):
    payload["slot_contracts"][0] = {
        "slot": "gap.entity",
        "kind": kind,
        "requires_current_source": None,
    }
    prepared = conditions.prepare_payload(payload)
    assert [row["kind"] for row in refs.request(prepared)["model_checks"]] == ["R"]
    report, _ = refs.resolve(wire(response(prepared)), prepared)
    assert conditions.decide(report).decision == "allow_uncertain"
    assert len([row for row in report.checks if row["producer"] == "local"]) == 3


def test_all_32_frozen_fixture_shapes_mechanically_fit_catalog_not_model_predictions():
    path = Path(__file__).resolve().parents[1] / "experiments/fixtures/a1_conditions_v2.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 32
    for row in rows:
        prepared = conditions.prepare_payload(row["input"])
        value = response(prepared)
        nominal = {
            (check["kind"], check["slot"]): check["status"] for check in row["expected"]["checks"]
        }
        for check in value["checks"]:
            kind = check["kind"]
            status = nominal[kind, check["slot"]]
            check["status"] = status
            if status == "contradicted":
                check["reason"] = {
                    "T": "expected_type_conflict",
                    "R": "relation_conflict",
                    "P1": "exclusive_role_conflict",
                }[kind]
            elif status == "unknown":
                check.update(
                    reason={"T": "insufficient_type_evidence", "P1": "missing_role_evidence"}[kind],
                    ref_ids=[],
                )
        if row["row_id"] == "14B":
            check = next(check for check in value["checks"] if check["kind"] == "P1")
            check.update(
                reason="conflicting_evidence",
                ref_ids=[ref_id(prepared, "E1"), ref_id(prepared, "E2")],
            )
        report, audit = refs.resolve(wire(value), prepared)
        assert [
            {key: check[key] for key in ("kind", "slot", "status")} for check in report.checks
        ] == row["expected"]["checks"]
        assert conditions.decide(report).decision == row["expected"]["contradiction_only"]
        audit.verify_report(report)
