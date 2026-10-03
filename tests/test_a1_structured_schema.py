"""Pure shape/local-contract tests; no provider requests or semantic scores."""

import hashlib
import json
from copy import deepcopy

import pytest

from growrag.experiments import a1_conditions as conditions
from growrag.experiments import a1_observer_refs as refs
from growrag.experiments import a1_structured_schema as schema
from growrag.experiments import a1_two_stage_observer as observer
from growrag.experiments import output_schemas as registered
from growrag.experiments.operator_schemas import validate_wire_shape

OLD_FINGERPRINTS = {
    "growrag-evidence-requirements-v2": (
        "16bfe9d21a2985d5e7aa98e031c6c1bd3c46c5c46b1f6cb2496c852c357c1445"
    ),
    "growrag-gap-query-v2": "07c68ccf775ada60c615f137487894712f8e0276b457d17b7860ad89c2a3d7e7",
    "growrag-operator-fresh-v3": "acbcf32c136c93e9788f295fe47b22a5ca4a5065f6ddb2d034427fe8636d8d4a",
    "growrag-operator-fresh-v4": "3fee44d0963318cd989096002909b95a4c0aa2e28d52a89281a4a7b6323a6703",
    "growrag-operator-memory-v3": (
        "14fbded2d84b9c50e2ec60e7787f175021938679847e001135b51fa01bc9549e"
    ),
    "growrag-operator-memory-v4": (
        "ec9a23b1c62977fafa53361875f28c7d1982f3db47897147762e60cbb27ffa06"
    ),
    "growrag-operator-reader-v2": (
        "065dfb0c463294c020480289fdb9eec54570f317e5732294015f290ddde4246f"
    ),
    "growrag-operator-static-v3": (
        "38d631e1841696dcd1d0323af183982956a70293de4ce9480062580ccd88e0cf"
    ),
    "growrag-operator-static-v4": (
        "ac4cf98de4285f85480739e8c0157d73342dd9a665c52fbc87c33825bb70b6dd"
    ),
    "growrag-short-supported-answer-v2": (
        "529e35493017e57aa27eb7eca68c66b489368ad9c6575fbc1e605dd4f24bab10"
    ),
    "growrag-three-way-route-v2": (
        "3b912bcf9dac60cd698824d555fdb1599a730fefabc3eb23f21637f2a0fcd6f0"
    ),
    "s2g-author-5d842a6-extract-api-v1": (
        "789794b95c6242decf9a9c2900892d7bbc5ced6d9494912614f7aeedec0d1816"
    ),
    "s2g-author-5d842a6-judge-api-v1": (
        "60128708fd26dc4e073c3235e54672c41740e2db01954af142a20a3bb011aa85"
    ),
}
VERSIONS = (observer.LOCATOR_PROMPT_VERSION, observer.JUDGE_PROMPT_VERSION)


def shape(version):
    return schema.response_format_for(version)["json_schema"]["schema"]


def validate(version, value):
    validate_wire_shape(value, shape(version))


def closed(node):
    assert set(node) <= {"type", "properties", "required", "additionalProperties", "items", "enum"}
    if node["type"] == "object":
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])
        for child in node["properties"].values():
            closed(child)
    elif node["type"] == "array":
        closed(node["items"])
    else:
        assert node["type"] == "string"


@pytest.fixture
def prepared():
    text = "Rowan Vale is a novel."
    # Deliberately fictional data. The schema must not know that novel != person.
    payload = {
        "schema_version": conditions.INPUT_VERSION,
        "original_question": "In what year was Rowan Vale born?",
        "visible_evidence": [
            {
                "source_id": "E1",
                "title": "Rowan Vale",
                "text": text,
                "window": {
                    "text_start": 0,
                    "text_end": len(text),
                    "original_text_chars": len(text),
                },
            }
        ],
        "previous_queries": ["In what year was Rowan Vale born?"],
        "remaining_retrievals": 1,
        "proposal": {
            "spec_id": "fictional_contract",
            "spec_version": "1",
            "template": "{entity} birth year",
            "gap": {"entity": "Rowan Vale"},
            "bindings": [],
            "compiled_queries": [
                {"step_id": "lookup", "source_id": "A1", "query": "Rowan Vale birth year"}
            ],
        },
        "slot_contracts": [
            {
                "slot": "gap.entity",
                "kind": "entity",
                "required_type": "person",
                "requires_current_source": True,
                "target_role": "current question target",
            }
        ],
        "required_checks": [
            {"kind": kind, "slot": "action.lookup" if kind == "R" else "gap.entity"}
            for kind in ("T", "R", "P0", "P1")
        ],
    }
    return conditions.prepare_payload(payload)


def locations(prepared):
    return {
        "locations": [
            {"kind": kind, "slot": slot, "candidate_ref_ids": []}
            for kind, slot in prepared.semantic_checks
        ]
    }


def checks(prepared):
    reasons = {
        "T": "insufficient_type_evidence",
        "R": "ambiguous_subgoal",
        "P1": "missing_role_evidence",
    }
    return {
        "checks": [
            {
                "kind": kind,
                "slot": slot,
                "status": "unknown",
                "reason": reasons[kind],
                "ref_ids": [],
            }
            for kind, slot in prepared.semantic_checks
        ]
    }


def resolve(value, prepared):
    location = observer.resolve_location(json.dumps(locations(prepared)), prepared)
    return observer.resolve_judgment(json.dumps(value), prepared, location)


def test_exact_versions_registered_with_strict_true_and_v5():
    assert registered.REGISTRY_VERSION == "growrag-output-schemas-v5"
    assert set(schema.two_stage_schema_registry()) == set(VERSIONS)
    assert set(registered.registry_manifest()["schemas"]) == set(OLD_FINGERPRINTS) | set(VERSIONS)
    for version in VERSIONS:
        wire = schema.response_format_for(version)
        assert wire["type"] == "json_schema" and wire["json_schema"]["strict"] is True
        assert registered.response_format_for(version) == wire
        assert registered.schema_fingerprint(version) == schema.schema_fingerprint(version)
        closed(shape(version))


def test_all_thirteen_old_full_wire_fingerprints_unchanged():
    assert {
        version: registered.schema_fingerprint(version) for version in OLD_FINGERPRINTS
    } == OLD_FINGERPRINTS


def test_prompt_text_and_semantic_versions_unchanged():
    assert observer.LOCATOR_PROMPT_VERSION == "growrag-a1-two-stage-locator-v1"
    assert observer.JUDGE_PROMPT_VERSION == "growrag-a1-two-stage-judge-v1"
    assert (
        observer.LOCATOR_PROMPT_SHA256
        == "fa33af4514921922c79f3e76185005ae258f1b43b8c6807d7733c21491350697"
    )
    assert (
        observer.JUDGE_PROMPT_SHA256
        == "8ee926e7ee52cd8126ab5153dda8fe12fefc43d2a2948dade0e0708b97631334"
    )
    assert conditions.DIMENSIONS == ("T", "R", "P0", "P1")


@pytest.mark.parametrize("version", VERSIONS)
def test_no_expected_question_values_or_dynamic_catalog_in_schema(version):
    serialized = json.dumps(shape(version))
    assert all(
        text not in serialized
        for text in ('"expected":', "nominal_vector", "Rowan", '"r1"', "gap.entity")
    )
    props = shape(version)["properties"]["locations" if version == VERSIONS[0] else "checks"][
        "items"
    ]["properties"]
    assert props["kind"]["enum"] == ["T", "R", "P1"]
    assert props["slot"] == {"type": "string"}


def test_judge_status_and_reason_union_only_uses_existing_semantic_enums():
    props = shape(VERSIONS[1])["properties"]["checks"]["items"]["properties"]
    assert props["status"]["enum"] == ["supported", "contradicted", "unknown"]
    reasons = {
        reason
        for values in conditions._REASONS.values()
        for codes in values.values()
        for reason in codes
    }
    assert props["reason"]["enum"] == sorted(reasons)


@pytest.mark.parametrize("version", VERSIONS)
def test_nested_copies_and_manifest_cannot_mutate_fingerprints(version):
    before = deepcopy(conditions._REASONS)
    digest = schema.schema_fingerprint(version)
    wire = schema.response_format_for(version)
    serialized = json.dumps(
        wire, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    assert hashlib.sha256(serialized.encode()).hexdigest() == digest
    wire["json_schema"]["strict"] = False
    wire["json_schema"]["schema"]["additionalProperties"] = True
    registry = schema.two_stage_schema_registry()
    registry[version][1]["properties"].clear()
    manifest = schema.schema_manifest()
    manifest["schemas"][version]["response_format"]["json_schema"]["schema"]["required"].clear()
    assert schema.schema_fingerprint(version) == digest
    assert registered.schema_fingerprint(version) == digest
    assert conditions._REASONS == before


@pytest.mark.parametrize(
    "version",
    [
        None,
        3,
        "",
        "locator",
        "growrag-a1-two-stage-locator-v2",
        conditions.WIRE_VERSION,
        refs.WIRE_VERSION,
    ],
)
def test_unregistered_version_has_no_schema_alias_or_fallback(version):
    with pytest.raises(ValueError):
        schema.response_format_for(version)
    with pytest.raises(ValueError):
        schema.schema_fingerprint(version)
    with pytest.raises(ValueError):
        registered.response_format_for(version)


def test_valid_shapes_do_not_change_local_unknown_resolution(prepared):
    locator, judge = locations(prepared), checks(prepared)
    validate(VERSIONS[0], locator)
    validate(VERSIONS[1], judge)
    report, audit = resolve(judge, prepared)
    audit.verify_report(report)
    assert {row["kind"] for row in report.to_dict()["checks"]} == {"T", "R", "P0", "P1"}
    assert conditions.decide(report).decision == "allow_uncertain"


@pytest.mark.parametrize("which", ["locations", "checks"])
def test_wrong_outer_key_is_rejected_before_local_resolution(which):
    version = VERSIONS[0] if which == "checks" else VERSIONS[1]
    with pytest.raises(ValueError):
        validate(version, {which: []})


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(
    "case", ["P0", "extra_top", "extra_item", "non_array", "integer_slot", "integer_ref"]
)
def test_schema_protects_shape_kinds_and_string_types(prepared, version, case):
    value = locations(prepared) if version == VERSIONS[0] else checks(prepared)
    root = "locations" if version == VERSIONS[0] else "checks"
    ref_field = "candidate_ref_ids" if version == VERSIONS[0] else "ref_ids"
    if case == "P0":
        value[root][0]["kind"] = "P0"
    elif case == "extra_top":
        value["answer"] = "fictional"
    elif case == "extra_item":
        value[root][0]["confidence"] = 1
    elif case == "non_array":
        value[root] = {}
    elif case == "integer_slot":
        value[root][0]["slot"] = 7
    else:
        value[root][0][ref_field] = [7]
    with pytest.raises(ValueError):
        validate(version, value)


def test_invalid_kind_status_reason_combination_stays_locally_checked(prepared):
    value = checks(prepared)
    value["checks"][0]["reason"] = "relation_conflict"
    validate(VERSIONS[1], value)  # It is in the union, but not T/unknown.
    with pytest.raises(observer.TwoStageContractError) as error:
        resolve(value, prepared)
    assert error.value.code == "invalid_status_reason"


@pytest.mark.parametrize("case", ["missing", "duplicate", "wrong_slot", "invented_ref"])
def test_exact_check_set_and_ref_allowlist_still_local(prepared, case):
    value = locations(prepared)
    if case == "missing":
        value["locations"].pop()
    elif case == "duplicate":
        value["locations"].append(deepcopy(value["locations"][0]))
    elif case == "wrong_slot":
        value["locations"][0]["slot"] = "gap.not_current"
    else:
        value["locations"][0]["candidate_ref_ids"] = ["r999"]
    validate(VERSIONS[0], value)
    with pytest.raises(observer.TwoStageContractError):
        observer.resolve_location(json.dumps(value), prepared)


def test_supported_without_current_evidence_is_not_repaired_by_schema(prepared):
    value = checks(prepared)
    value["checks"][0].update(status="supported", reason="expected_type_supported", ref_ids=[])
    validate(VERSIONS[1], value)
    with pytest.raises(observer.TwoStageContractError) as error:
        resolve(value, prepared)
    assert error.value.code == "missing_current_evidence"
    assert error.value.observation_status == "unverified"


def test_valid_schema_and_existing_reference_do_not_prove_semantic_truth(prepared):
    value = checks(prepared)
    evidence_ref = next(
        item["ref_id"]
        for item in refs.catalog(prepared)
        if item["source_id"] == "E1" and item["field"] == "text"
    )
    # Authored WRONG claim: current text says novel, not person. Both mechanical
    # validators accept existing references; neither adjudicates entailment.
    value["checks"][0].update(
        status="supported", reason="expected_type_supported", ref_ids=[evidence_ref]
    )
    validate(VERSIONS[1], value)
    report, audit = resolve(value, prepared)
    audit.verify_report(report)
    assert (
        next(item for item in report.to_dict()["checks"] if item["kind"] == "T")["status"]
        == "supported"
    )
