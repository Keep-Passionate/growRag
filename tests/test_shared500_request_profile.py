"""Synthetic wire requests only; no API/corpus/credential access."""

import importlib.util
import json
from pathlib import Path

import pytest

from growrag.experiments.output_schemas import response_format_for

SPEC = importlib.util.spec_from_file_location(
    "profile_audit",
    Path(__file__).resolve().parents[1] / "scripts" / "audits" / "request_profile_audit.py",
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def event(stage):
    version = f"s2g-author-5d842a6-{stage}-api-v1"
    request = {
        "model": AUDIT.MODEL,
        "messages": [
            {"role": "system", "content": "SENTINEL_NEVER_OUTPUT"},
            {"role": "user", "content": "SENTINEL_NEVER_OUTPUT"},
        ],
        "max_tokens": AUDIT.CAPS[stage],
        "stream": False,
        "enable_thinking": False,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    value = {
        "prompt_version": version,
        "request": request,
        "retry_count": 0,
        "execution_kind": "api_transport",
        "transport_source": "live_api",
        "network_attempted": True,
        "api_requests": 1,
        "response": {"model": AUDIT.MODEL},
    }
    if stage in AUDIT.SCHEMAS:
        request["response_format"] = response_format_for(version)
        value["output_schema_sha256"] = AUDIT.SCHEMAS[stage]
        value["output_schema_registry_version"] = "growrag-output-schemas-v2"
    value["request_sha256"] = AUDIT.sha(
        json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
    )
    return value


@pytest.mark.parametrize("stage", ["judge", "extract", "answer"])
def test_expected_stage_profiles_and_no_message_disclosure(stage):
    profile, issues = AUDIT.project_request(event(stage))
    assert issues == []
    assert profile["max_tokens"] == AUDIT.CAPS[stage]
    assert "SENTINEL_NEVER_OUTPUT" not in json.dumps(profile)
    assert "messages" not in profile


def test_temperature_and_cap_drift_and_body_hash_are_detected():
    value = event("answer")
    value["request"]["temperature"] = 0.6
    value["request"]["max_tokens"] = 256
    _, issues = AUDIT.project_request(value)
    assert "parameter_mismatch:temperature" in issues
    assert "parameter_mismatch:max_tokens" in issues
    assert "request_body_hash_mismatch" in issues


def test_schema_strictness_change_is_detected_without_schema_text_output():
    value = event("judge")
    value["request"]["response_format"]["json_schema"]["strict"] = False
    profile, issues = AUDIT.project_request(value)
    assert "strict_schema_wire_fingerprint_mismatch" in issues
    assert "schema" not in profile


def test_unknown_returned_model_on_transport_failure_is_not_fabricated():
    value = event("judge")
    value.pop("response")
    _, issues = AUDIT.project_request(value)
    assert issues == []


def test_retry_or_boolean_temperature_is_rejected():
    value = event("answer")
    value["retry_count"] = 1
    value["request"]["temperature"] = False
    _, issues = AUDIT.project_request(value)
    assert "retry_count_not_zero" in issues
    assert "numeric_parameter_type:temperature" in issues


def test_claim_other_series_and_unsealed_batch_are_not_audited(tmp_path):
    (tmp_path / (AUDIT.PREFIX + "0000_0025.claim.json")).touch()
    (tmp_path / (AUDIT.PREFIX + "0000_0025")).mkdir()
    (tmp_path / "2026-09-27_s2g_shared32_v3_0000_0032").mkdir()
    result = AUDIT.audit(tmp_path)
    assert result["audited_raw_request_count"] == 0
    assert result["unsealed_skipped"] == [AUDIT.PREFIX + "0000_0025"]
