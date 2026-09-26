"""Offline shape contracts, not provider support or evidence-quality validation."""

import hashlib
import json

import pytest

from growrag.controller import (
    ASSESS_PROMPT_VERSION,
    GAP_QUERY_PROMPT_VERSION,
    ROUTE_PROMPT_VERSION,
)
from growrag.experiments.output_schemas import (
    registry_manifest,
    response_format_for,
    schema_fingerprint,
)
from growrag.experiments.run_bounded_system import READER_VERSION
from growrag.experiments.s2g_author_api import PROMPT_VERSIONS as S2G_AUTHOR_VERSIONS


def test_registry_tracks_exact_current_prompts_without_alias_fallback():
    manifest = registry_manifest()
    assert set(manifest["schemas"]) == {
        ASSESS_PROMPT_VERSION,
        GAP_QUERY_PROMPT_VERSION,
        ROUTE_PROMPT_VERSION,
        READER_VERSION,
        S2G_AUTHOR_VERSIONS["judge"],
        S2G_AUTHOR_VERSIONS["extract"],
    }
    for version, item in manifest["schemas"].items():
        assert item["sha256"] == schema_fingerprint(version)
        assert item["response_format"] == response_format_for(version)


@pytest.mark.parametrize("version", ["", None, 3, "growrag-evidence-requirements-v1", "v999"])
def test_unreviewed_prompt_has_no_default_schema(version):
    with pytest.raises(ValueError, match="no reviewed schema"):
        response_format_for(version)
    with pytest.raises(ValueError, match="no reviewed schema"):
        schema_fingerprint(version)


def check_closed_objects(node):
    kind = node.get("type")
    if kind == "object":
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])
        assert len(node["required"]) == len(set(node["required"]))
        for child in node["properties"].values():
            check_closed_objects(child)
    elif kind == "array":
        check_closed_objects(node["items"])
    elif isinstance(kind, list):
        assert kind in (["string", "null"], ["boolean", "null"])
    else:
        assert kind in {"string", "boolean", "integer"}
    # No unverified provider-specific length/conditional schema keywords.
    assert set(node) <= {"type", "properties", "required", "additionalProperties", "items", "enum"}


def test_every_nested_object_is_closed_and_all_fields_are_required():
    for item in registry_manifest()["schemas"].values():
        response_format = item["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        check_closed_objects(response_format["json_schema"]["schema"])


def test_assessment_uses_boolean_or_null_gain_not_array_or_text():
    props = response_format_for(ASSESS_PROMPT_VERSION)["json_schema"]["schema"]["properties"]
    assert props["sufficient"] == {"type": "boolean"}
    assert props["useful_gain"] == {"type": ["boolean", "null"]}
    requirement = props["requirements"]["items"]["properties"]
    assert requirement["status"]["enum"] == ["supported", "missing", "conflicted", "unknown"]
    assert requirement["evidence_ids"] == {"type": "array", "items": {"type": "string"}}
    assert props["reason"] == {"type": "string"}


def test_route_action_and_applicability_status_have_different_enums():
    props = response_format_for(ROUTE_PROMPT_VERSION)["json_schema"]["schema"]["properties"]
    assert props["action"]["enum"] == ["BASE", "FRESH", "REUSE"]
    assert props["memory_id"] == {"type": ["string", "null"]}
    check = props["condition_checks"]["items"]["properties"]
    assert check["status"]["enum"] == ["supported", "conflicted", "unknown"]
    assert check["critical"] == {"type": "boolean"}


def test_short_answer_and_single_query_contracts():
    answer = response_format_for(READER_VERSION)["json_schema"]["schema"]
    query = response_format_for(GAP_QUERY_PROMPT_VERSION)["json_schema"]["schema"]
    assert answer["required"] == ["answer", "cited_evidence_ids"]
    assert query["required"] == ["query"]
    assert answer["properties"]["cited_evidence_ids"]["items"]["type"] == "string"


def test_returned_copies_cannot_mutate_registry_and_hash_covers_full_wire_format():
    version = ASSESS_PROMPT_VERSION
    original = response_format_for(version)
    digest = schema_fingerprint(version)
    serialized = json.dumps(original, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(serialized.encode("utf-8")).hexdigest() == digest
    original["json_schema"]["schema"]["properties"]["sufficient"]["type"] = "string"
    manifest = registry_manifest()
    manifest["schemas"][version]["response_format"]["json_schema"]["strict"] = False
    assert schema_fingerprint(version) == digest
    assert response_format_for(version)["json_schema"]["strict"] is True


def test_s2g_author_judge_schema_matches_existing_prompt_without_new_policy():
    schema = response_format_for(S2G_AUTHOR_VERSIONS["judge"])["json_schema"]["schema"]
    assert schema["required"] == ["sufficient", "gap_items"]
    assert schema["properties"]["sufficient"] == {"type": "boolean"}
    item = schema["properties"]["gap_items"]["items"]
    assert item["required"] == ["category", "target", "slot", "description"]
    assert item["properties"]["category"] == {
        "type": "string",
        "enum": ["bridge_entity", "attribute", "relation", "evidence_span", "other"],
    }
    for field in ("target", "slot", "description"):
        assert item["properties"][field] == {"type": "string"}
    check_closed_objects(schema)


def test_s2g_author_extractor_uses_integer_ids_without_extra_length_constraints():
    schema = response_format_for(S2G_AUTHOR_VERSIONS["extract"])["json_schema"]["schema"]
    assert schema["required"] == ["evidence_global_ids"]
    assert schema["properties"] == {
        "evidence_global_ids": {"type": "array", "items": {"type": "integer"}}
    }
    # 作者提示写最多 K 句，但迁移中不擅自给原解析器加 maxItems、minimum 或去重策略。
    check_closed_objects(schema)


def test_s2g_author_text_answer_intentionally_has_no_json_schema():
    with pytest.raises(ValueError, match="no reviewed schema"):
        response_format_for(S2G_AUTHOR_VERSIONS["answer"])


def test_original_schema_versions_and_wire_fingerprints_remain_unchanged():
    original = {
        "growrag-evidence-requirements-v2": (
            "16bfe9d21a2985d5e7aa98e031c6c1bd3c46c5c46b1f6cb2496c852c357c1445"
        ),
        "growrag-gap-query-v2": (
            "07c68ccf775ada60c615f137487894712f8e0276b457d17b7860ad89c2a3d7e7"
        ),
        "growrag-short-supported-answer-v2": (
            "529e35493017e57aa27eb7eca68c66b489368ad9c6575fbc1e605dd4f24bab10"
        ),
        "growrag-three-way-route-v2": (
            "3b912bcf9dac60cd698824d555fdb1599a730fefabc3eb23f21637f2a0fcd6f0"
        ),
    }
    assert {version: schema_fingerprint(version) for version in original} == original
