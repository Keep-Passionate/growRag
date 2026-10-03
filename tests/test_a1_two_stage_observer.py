"""Offline mechanics only; authored responses are not model semantic results."""

import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from growrag.experiments import a1_conditions as conditions
from growrag.experiments import a1_observer_refs as refs
from growrag.experiments import a1_two_stage_observer as observer


def wire(value):
    return json.dumps(value, ensure_ascii=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def prepared():
    text = "Iona Pell is a person and the author of Copper Estuary."
    payload = {
        "schema_version": conditions.INPUT_VERSION,
        "original_question": "In what year was the author of Copper Estuary born?",
        "visible_evidence": [
            {
                "source_id": "E1",
                "title": "Copper Estuary",
                "text": text,
                "window": {
                    "text_start": 7,
                    "text_end": 7 + len(text),
                    "original_text_chars": 107 + len(text),
                },
            }
        ],
        "previous_queries": ["Copper Estuary author"],
        "remaining_retrievals": 1,
        "proposal": {
            "spec_id": "two_stage_contract",
            "spec_version": "1",
            "template": "{entity} birth year",
            "gap": {"entity": "Iona Pell"},
            "bindings": [],
            "compiled_queries": [
                {"step_id": "lookup", "source_id": "A1", "query": "Iona Pell birth year"}
            ],
        },
        "slot_contracts": [
            {
                "slot": "gap.entity",
                "kind": "entity",
                "required_type": "person",
                "requires_current_source": True,
                "target_role": "author of Copper Estuary",
            }
        ],
        "required_checks": [
            {"kind": kind, "slot": "action.lookup" if kind == "R" else "gap.entity"}
            for kind in ("T", "R", "P0", "P1")
        ],
    }
    return conditions.prepare_payload(payload, {"local_only": "hidden audit tail"})


def location_value(prepared, ids=None):
    return {
        "locations": [
            {"kind": kind, "slot": slot, "candidate_ref_ids": ["r3"] if ids is None else list(ids)}
            for kind, slot in prepared.semantic_checks
        ]
    }


def judgment_value(prepared):
    reasons = {
        "T": "expected_type_supported",
        "R": "permitted_intermediate_subgoal",
        "P1": "target_role_supported",
    }
    return {
        "checks": [
            {
                "kind": kind,
                "slot": slot,
                "status": "supported",
                "reason": reasons[kind],
                "ref_ids": ["r1", "r4", "r3"] if kind == "R" else ["r3"],
            }
            for kind, slot in prepared.semantic_checks
        ]
    }


def location(prepared, ids=None):
    return observer.resolve_location(wire(location_value(prepared, ids)), prepared)


def test_prompts_versions_full_window_and_detached_requests(prepared):
    hints = location(prepared)
    locate_sent = observer.locator_messages(prepared)
    judge_sent = observer.judge_messages(prepared, hints)
    assert observer.LOCATOR_PROMPT_VERSION == "growrag-a1-two-stage-locator-v1"
    assert observer.JUDGE_PROMPT_VERSION == "growrag-a1-two-stage-judge-v1"
    assert observer.LOCATOR_PROMPT_SHA256 == digest(observer.LOCATOR_PROMPT.encode())
    assert observer.JUDGE_PROMPT_SHA256 == digest(observer.JUDGE_PROMPT.encode())
    assert refs.OBSERVER_PROMPT in observer.JUDGE_PROMPT
    assert locate_sent[0]["content"] == observer.LOCATOR_PROMPT
    assert judge_sent[0]["content"] == observer.JUDGE_PROMPT
    locate_request, judge_request = (
        json.loads(locate_sent[1]["content"]),
        json.loads(judge_sent[1]["content"]),
    )
    assert locate_request["input"] == judge_request["input"] == prepared.payload
    assert [row["ref_id"] for row in locate_request["ref_catalog"]] == ["r1", "r2", "r3"]
    assert judge_request == {**refs.request(prepared), "candidate_locations": list(hints.locations)}
    for request in (locate_request, judge_request):
        assert all("quote" not in row for row in request["ref_catalog"])
        assert [(row["kind"], row["slot"]) for row in request["model_checks"]] == list(
            prepared.semantic_checks
        )
    for excluded in (prepared.input_sha256, prepared.snapshot_sha256, "hidden audit tail"):
        assert excluded not in wire(locate_sent) + wire(judge_sent)
    judge_sent[1]["content"] = "tampered"
    assert observer.judge_messages(prepared, hints)[1]["content"] != "tampered"


def test_location_raw_coverage_and_repeated_views_do_not_change_input(prepared):
    before = prepared.to_dict()
    raw = " \n" + wire(location_value(prepared)) + " \n"
    hints = observer.resolve_location(raw, prepared)
    view = hints.to_dict()
    assert hints.raw_text == raw
    assert view["response_sha256"] == digest(raw.encode())
    assert view["location_sha256"] == digest(hints.location_bytes)
    assert view["messages_sha256"] == digest(
        conditions.canonical_bytes(observer.locator_messages(prepared))
    )
    for coverage in view["coverage"]:
        assert coverage["candidate_ref_count"] == 1
        assert coverage["candidate_text_chars"] == len(
            prepared.payload["visible_evidence"][0]["text"]
        )
        assert coverage["visible_ref_count"] == 3
    view["locations"][0]["candidate_ref_ids"].clear()
    view["coverage"].clear()
    for _ in range(3):
        hints.locations[0]["candidate_ref_ids"].clear()
        assert hints.to_dict()["locations"][0]["candidate_ref_ids"] == ["r3"]
        assert prepared.to_dict() == before
    with pytest.raises(FrozenInstanceError):
        hints.response_text = "tampered"


@pytest.mark.parametrize("ids", [[], ["r1", "r2", "r3"]])
def test_empty_and_all_candidates_legal_without_business_verdict(prepared, ids):
    hints = location(prepared, ids)
    assert all(row["candidate_ref_ids"] == ids for row in hints.locations)
    assert all("status" not in row for row in hints.locations)
    report, audit = observer.resolve_judgment(wire(judgment_value(prepared)), prepared, hints)
    assert conditions.decide(report).decision == "allow"
    audit.verify_report(report)


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda v: v.update(answer="not allowed"), "invalid_response_fields"),
        (lambda v: v.update(locations={}), "checks_not_array"),
        (lambda v: v["locations"][0].update(status="supported"), "invalid_check_fields"),
        (lambda v: v["locations"][0].update(reason="found"), "invalid_check_fields"),
        (lambda v: v["locations"][0].update(query="new query"), "invalid_check_fields"),
        (lambda v: v["locations"][0].update(kind="P0"), "unexpected_check"),
        (lambda v: v["locations"][0].update(kind=[]), "invalid_check_identity"),
        (lambda v: v["locations"][0].update(slot="gap.stale"), "unexpected_check"),
        (lambda v: v["locations"].pop(), "missing_checks"),
        (lambda v: v["locations"].append(deepcopy(v["locations"][0])), "duplicate_check"),
        (lambda v: v["locations"][0].update(candidate_ref_ids="r3"), "invalid_refs_array"),
        (lambda v: v["locations"][0].update(candidate_ref_ids=["r3", "r3"]), "duplicate_ref_id"),
        (lambda v: v["locations"][0].update(candidate_ref_ids=["r4"]), "disallowed_ref_source"),
    ],
)
def test_locator_contract_failures_keep_raw_and_unattempted_judge(prepared, change, code):
    value = location_value(prepared)
    change(value)
    raw = wire(value)
    with pytest.raises(observer.TwoStageContractError) as caught:
        observer.resolve_location(raw, prepared)
    exc = caught.value
    assert (exc.stage, exc.code, exc.observation_status) == ("locate", code, "model_output_error")
    assert exc.raw_text == raw
    content = exc.audit.to_dict()
    assert content["locator"]["raw_text"] == raw and content["judge"] is None
    assert content["location_report"] is None
    assert content["code"] == code and content["observation_status"] == "model_output_error"
    assert content["locator"]["messages"] == observer.locator_messages(prepared)


@pytest.mark.parametrize("bad", ["r0", "r01", "R3", " r3", "r999", "[r3]", 1, True, None, {}, []])
@pytest.mark.parametrize("stage", ["locate", "judge"])
def test_nonexact_stale_unknown_reference_ids_not_repaired(prepared, bad, stage):
    if stage == "locate":
        value = location_value(prepared)
        value["locations"][0]["candidate_ref_ids"] = [bad]
    else:
        value = judgment_value(prepared)
        value["checks"][0]["ref_ids"] = [bad]
    with pytest.raises(observer.TwoStageContractError) as caught:
        if stage == "locate":
            observer.resolve_location(wire(value), prepared)
        else:
            observer.resolve_judgment(wire(value), prepared, location(prepared))
    assert caught.value.code == "unknown_ref_id"
    assert caught.value.stage == stage
    assert caught.value.audit.to_dict()["observation_status"] == "model_output_error"


@pytest.mark.parametrize("stage", ["locate", "judge"])
@pytest.mark.parametrize(
    "raw",
    [
        "",
        "[]",
        "```json\n{}\n```",
        "explanation {}",
        '{"checks":NaN}',
        '{"locations":[],"locations":[]}',
        '{"checks":[],"checks":[]}',
        "{} trailing",
    ],
)
def test_strict_json_no_fences_duplicate_keys_or_nonfinite_repair(prepared, stage, raw):
    with pytest.raises(observer.TwoStageContractError) as caught:
        if stage == "locate":
            observer.resolve_location(raw, prepared)
        else:
            observer.resolve_judgment(raw, prepared, location(prepared))
    assert caught.value.raw_text == raw
    assert caught.value.observation_status == "model_output_error"
    assert (
        caught.value.audit.to_dict()[stage == "judge" and "judge" or "locator"]["raw_text"] == raw
    )


def test_audit_binds_actual_two_stage_messages_conversion_and_report(prepared):
    hints = location(prepared, ["r2"])
    raw = " \n" + wire(judgment_value(prepared)) + "\n"
    report, audit = observer.resolve_judgment(raw, prepared, hints)
    content = audit.to_dict()
    assert content["outcome"] == "valid_response" and content["observation_status"] == "valid"
    assert content["locator"]["messages"] == observer.locator_messages(prepared)
    assert content["judge"]["messages"] == observer.judge_messages(prepared, hints)
    assert content["judge"]["messages_sha256"] == digest(
        conditions.canonical_bytes(observer.judge_messages(prepared, hints))
    )
    assert content["judge"]["messages_sha256"] != digest(
        conditions.canonical_bytes(refs.messages(prepared))
    )
    assert content["judge"]["request_sha256"] == digest(
        conditions.canonical_bytes(observer.judge_request(prepared, hints))
    )
    assert content["locator"]["request_sha256"] == digest(
        conditions.canonical_bytes(observer.locator_request(prepared))
    )
    assert content["judge"]["prompt_sha256"] == observer.JUDGE_PROMPT_SHA256
    assert content["judge"]["parser_prompt_sha256"] == refs.PROMPT_SHA256
    assert content["judge"]["derived_report_prompt_sha256"] == conditions.PROMPT_SHA256
    assert content["judge"]["raw_text"] == raw
    assert content["judge"]["response_sha256"] == digest(raw.encode())
    assert content["judge"]["converted_raw_text"] == report.raw_text != raw
    assert content["judge"]["derived_report_sha256"] == report.report_sha256
    assert content["location_report"] == hints.to_dict()
    assert content["judge_added_ref_ids"] == [
        {"kind": "T", "slot": "gap.entity", "ref_ids": ["r3"]},
        {"kind": "R", "slot": "action.lookup", "ref_ids": ["r1", "r3"]},
        {"kind": "P1", "slot": "gap.entity", "ref_ids": ["r3"]},
    ]
    audit.verify_report(report)
    assert observer.resolve_judgment(raw, prepared, hints)[1].audit_bytes == audit.audit_bytes


@pytest.mark.parametrize(
    "kind,change,code,status",
    [
        ("R", {"ref_ids": ["r4", "r3"]}, "missing_required_sources", "unverified"),
        ("R", {"ref_ids": ["r1", "r3"]}, "missing_required_sources", "unverified"),
        ("R", {"ref_ids": ["r1", "r4"]}, "missing_intermediate_evidence", "unverified"),
        ("T", {"ref_ids": []}, "missing_current_evidence", "unverified"),
        ("P1", {"ref_ids": []}, "missing_current_evidence", "unverified"),
        ("T", {"ref_ids": ["r4"]}, "disallowed_ref_source", "model_output_error"),
        ("P1", {"ref_ids": ["r4"]}, "disallowed_ref_source", "model_output_error"),
        (
            "P1",
            {"status": "unknown", "reason": "conflicting_evidence", "ref_ids": ["r3"]},
            "incomplete_conflict_evidence",
            "unverified",
        ),
    ],
)
def test_no_auto_added_hints_or_provenance_and_stable_failure_categories(
    prepared, kind, change, code, status
):
    hints = location(prepared, ["r1", "r2", "r3"])
    value = judgment_value(prepared)
    next(row for row in value["checks"] if row["kind"] == kind).update(change)
    with pytest.raises(observer.TwoStageContractError) as caught:
        observer.resolve_judgment(wire(value), prepared, hints)
    exc = caught.value
    assert (exc.stage, exc.code, exc.observation_status) == ("judge", code, status)
    content = exc.audit.to_dict()
    assert content["locator"]["outcome"] == "valid_response"
    assert content["judge"]["outcome"] == "invalid_response"
    converted = json.loads(content["judge"]["converted_raw_text"])
    actual = next(row for row in converted["checks"] if row["kind"] == kind)
    assert len(actual["refs"]) == len(change["ref_ids"])
    assert content["judge"]["derived_report_sha256"] is None


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda v: v.update(judge_added_refs=[]), "invalid_response_fields"),
        (lambda v: v.update(checks=None), "checks_not_array"),
        (lambda v: v["checks"][0].update(candidate_ref_ids=[]), "invalid_check_fields"),
        (lambda v: v["checks"][0].update(kind="P0"), "unexpected_check"),
        (lambda v: v["checks"][0].update(slot="gap.stale"), "unexpected_check"),
        (lambda v: v["checks"][0].update(status=[]), "invalid_status_reason"),
        (lambda v: v["checks"][0].update(reason="guessed"), "invalid_status_reason"),
        (lambda v: v["checks"][0].update(ref_ids=["r3", "r3"]), "duplicate_ref_id"),
        (lambda v: v["checks"][0].update(ref_ids="r3"), "invalid_refs_array"),
        (lambda v: v["checks"].pop(), "missing_checks"),
        (lambda v: v["checks"].append(deepcopy(v["checks"][0])), "duplicate_check"),
    ],
)
def test_judge_fields_enums_and_coverage_preserve_catalog_contract(prepared, change, code):
    value = judgment_value(prepared)
    change(value)
    with pytest.raises(observer.TwoStageContractError) as caught:
        observer.resolve_judgment(wire(value), prepared, location(prepared))
    assert caught.value.code == code
    assert caught.value.audit.to_dict()["code"] == code


def test_unknown_empty_refs_not_a_contract_failure_and_local_null_type_kept(prepared):
    payload = prepared.payload
    payload["slot_contracts"][0]["required_type"] = None
    reduced = conditions.prepare_payload(payload)
    hints = location(reduced, [])
    assert [row["kind"] for row in hints.locations] == ["R", "P1"]
    value = judgment_value(reduced)
    value["checks"][1].update(status="unknown", reason="missing_role_evidence", ref_ids=[])
    report, _ = observer.resolve_judgment(wire(value), reduced, hints)
    assert conditions.decide(report).decision == "allow_uncertain"
    assert (
        next(row for row in report.checks if row["kind"] == "T")["reason"]
        == "missing_type_requirement"
    )
    assert next(row for row in report.checks if row["kind"] == "P0")["producer"] == "local"
    value["checks"].append(
        {
            "kind": "T",
            "slot": "gap.entity",
            "status": "supported",
            "reason": "expected_type_supported",
            "ref_ids": ["r3"],
        }
    )
    with pytest.raises(observer.TwoStageContractError) as caught:
        observer.resolve_judgment(wire(value), reduced, hints)
    assert caught.value.code == "unexpected_check"


def test_equal_field_texts_not_counted_as_two_conflict_passages(prepared):
    payload = prepared.payload
    payload["visible_evidence"][0]["title"] = payload["visible_evidence"][0]["text"]
    duplicate = conditions.prepare_payload(payload)
    value = judgment_value(duplicate)
    value["checks"][2].update(status="unknown", reason="conflicting_evidence", ref_ids=["r2", "r3"])
    with pytest.raises(observer.TwoStageContractError) as caught:
        observer.resolve_judgment(wire(value), duplicate, location(duplicate))
    assert caught.value.code == "incomplete_conflict_evidence"
    payload["visible_evidence"][0]["title"] = "Iona Pell is not the author."
    differing = conditions.prepare_payload(payload)
    report, _ = observer.resolve_judgment(wire(value), differing, location(differing))
    assert conditions.decide(report).decision == "allow_uncertain"


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v["locations"][0].update(candidate_ref_ids=["r1"]),
        lambda v: v["coverage"][0].update(candidate_text_chars=0),
        lambda v: v.update(raw_text="tampered"),
        lambda v: v.update(messages_sha256="0" * 64),
    ],
)
def test_rehashed_location_forgery_rejected_on_every_use(prepared, change):
    hints = location(prepared)
    view = json.loads(hints.location_bytes)
    change(view)
    data = conditions.canonical_bytes(view)
    forged = replace(hints, location_bytes=data, location_sha256=digest(data))
    for call in (
        forged.to_dict,
        lambda: observer.judge_messages(prepared, forged),
        lambda: observer.resolve_judgment(wire(judgment_value(prepared)), prepared, forged),
    ):
        with pytest.raises(conditions.ConditionContractError, match="binding"):
            call()


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v["catalog"][0].update(quote="changed"),
        lambda v: v["locator"]["request"]["model_checks"].clear(),
        lambda v: v["judge"]["request"]["candidate_locations"].clear(),
        lambda v: v["judge"]["messages"][0].update(content="changed"),
        lambda v: v["locator"].update(raw_text="changed"),
        lambda v: v["judge"].update(raw_text="changed"),
        lambda v: v["judge"].update(derived_report_sha256="0" * 64),
        lambda v: v["location_report"]["locations"][0].update(candidate_ref_ids=["r1"]),
        lambda v: v["judge_added_ref_ids"].clear(),
        lambda v: v.update(snapshot_sha256="0" * 64),
    ],
)
def test_rehashed_audit_cross_object_forgery_rejected(prepared, change):
    _, audit = observer.resolve_judgment(
        wire(judgment_value(prepared)), prepared, location(prepared, [])
    )
    view = json.loads(audit.audit_bytes)
    change(view)
    data = conditions.canonical_bytes(view)
    with pytest.raises(conditions.ConditionContractError, match="binding"):
        replace(audit, audit_bytes=data, audit_sha256=digest(data)).to_dict()


def test_raw_snapshot_cross_bindings_and_report_forgery(prepared):
    hints = location(prepared)
    raw = wire(judgment_value(prepared))
    report, audit = observer.resolve_judgment(raw, prepared, hints)
    different_snapshot = conditions.prepare_payload(prepared.payload, {"different": True})
    with pytest.raises(conditions.ConditionContractError, match="snapshot binding"):
        observer.judge_messages(different_snapshot, hints)
    with pytest.raises(conditions.ConditionContractError, match="binding"):
        replace(audit, locator_response_text=hints.raw_text + " ").to_dict()
    with pytest.raises(conditions.ConditionContractError, match="binding"):
        replace(audit, judgment_response_text=raw + " ").to_dict()
    altered = json.loads(report.report_bytes)
    altered["checks"][0].update(status="contradicted", reason="expected_type_conflict")
    data = conditions.canonical_bytes(altered)
    with pytest.raises(conditions.ConditionContractError, match="binding"):
        audit.verify_report(replace(report, report_bytes=data, report_sha256=digest(data)))
    value = judgment_value(prepared)
    value["checks"][0].update(status="contradicted", reason="expected_type_conflict")
    other, _ = observer.resolve_judgment(wire(value), prepared, hints)
    with pytest.raises(conditions.ConditionContractError, match="does not belong"):
        audit.verify_report(other)
    audit.to_dict()["judge"]["messages"].clear()
    assert audit.to_dict()["judge"]["messages"]
    with pytest.raises(FrozenInstanceError):
        audit.judgment_response_text = "changed"


@pytest.mark.parametrize("stage", ["locate", "judge"])
@pytest.mark.parametrize("raw", [None, {}, b"{}", "\ud800"])
def test_nontext_or_invalid_unicode_response_has_no_serializable_audit(prepared, raw, stage):
    with pytest.raises(observer.TwoStageContractError) as caught:
        if stage == "locate":
            observer.resolve_location(raw, prepared)
        else:
            observer.resolve_judgment(raw, prepared, location(prepared))
    assert caught.value.code == "invalid_response_text"
    assert caught.value.raw_text == raw and caught.value.audit is None


def test_prepared_tampering_is_integrity_error_not_safe_model_failure(prepared):
    broken = replace(prepared, snapshot_sha256="0" * 64)
    with pytest.raises(conditions.ConditionContractError, match="SHA") as caught:
        observer.resolve_location("{}", broken)
    assert not isinstance(caught.value, observer.TwoStageContractError)
    hints = location(prepared)
    with pytest.raises(conditions.ConditionContractError, match="SHA"):
        observer.resolve_judgment("{}", broken, hints)


def test_failure_audit_replay_and_no_report_binding(prepared):
    hints = location(prepared)
    with pytest.raises(observer.TwoStageContractError) as caught:
        observer.resolve_judgment('{"checks":[]}', prepared, hints)
    failure = caught.value.audit
    with pytest.raises(observer.TwoStageContractError) as repeated:
        observer.resolve_judgment('{"checks":[]}', prepared, hints)
    assert repeated.value.audit.to_dict() == failure.to_dict()
    report, _ = observer.resolve_judgment(wire(judgment_value(prepared)), prepared, hints)
    with pytest.raises(conditions.ConditionContractError, match="does not belong"):
        failure.verify_report(report)


def test_unverified_judgment_still_records_its_actual_added_qe_choices(prepared):
    hints = location(prepared, [])
    value = judgment_value(prepared)
    value["checks"][1]["ref_ids"] = ["r1", "r4"]
    with pytest.raises(observer.TwoStageContractError) as caught:
        observer.resolve_judgment(wire(value), prepared, hints)
    assert caught.value.observation_status == "unverified"
    assert caught.value.audit.to_dict()["judge_added_ref_ids"] == [
        {"kind": "T", "slot": "gap.entity", "ref_ids": ["r3"]},
        {"kind": "R", "slot": "action.lookup", "ref_ids": ["r1"]},
        {"kind": "P1", "slot": "gap.entity", "ref_ids": ["r3"]},
    ]


def test_unparseable_raw_tamper_of_validated_objects_is_an_integrity_failure(prepared):
    hints = location(prepared)
    _, audit = observer.resolve_judgment(wire(judgment_value(prepared)), prepared, hints)
    for corrupt in (
        replace(hints, response_text="not JSON"),
        replace(hints, response_text=None),
        replace(audit, judgment_response_text="\ud800"),
    ):
        with pytest.raises(conditions.ConditionContractError, match="binding") as caught:
            corrupt.to_dict()
        assert not isinstance(caught.value, observer.TwoStageContractError)
    with pytest.raises(observer.TwoStageContractError) as failed:
        observer.resolve_location("not JSON", prepared)
    with pytest.raises(conditions.ConditionContractError, match="failed location"):
        replace(failed.value.audit, judgment_response_text="{}").to_dict()


def test_all_exposed_development_inputs_fit_two_stage_shape_without_claiming_model_accuracy():
    path = Path(__file__).resolve().parents[1] / "experiments/fixtures/a1_conditions_v2.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 32
    for row in rows:
        prepared = conditions.prepare_payload(row["input"])
        current = [
            entry["ref_id"]
            for entry in refs.catalog(prepared)
            if not entry["source_id"].startswith("A")
        ]
        hints = location(prepared, current)
        reason = {
            "T": "insufficient_type_evidence",
            "R": "ambiguous_subgoal",
            "P1": "missing_role_evidence",
        }
        value = {
            "checks": [
                {
                    "kind": kind,
                    "slot": slot,
                    "status": "unknown",
                    "reason": reason[kind],
                    "ref_ids": [],
                }
                for kind, slot in prepared.semantic_checks
            ]
        }
        report, audit = observer.resolve_judgment(wire(value), prepared, hints)
        audit.verify_report(report)
        assert len(report.checks) == len(prepared.required_checks)
