"""Synthetic transport/accounting regression only: no API, keys or natural gold."""

import hashlib
import json
from copy import deepcopy

import pytest

from growrag.experiments import a0_v2_feedback as audit
from growrag.experiments import a1_two_stage_budget as budget
from growrag.experiments import output_schemas, prior_budget

NEW_PREFIX = "2026-10-03_a1schema_"
NEW_PROTOCOL = "growrag-a1-schema-probe-v1"
MODEL = budget.catalog.old.PILOT_MODEL


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def simple_root(runs, *, prefix=NEW_PREFIX, protocol=NEW_PROTOCOL, model=MODEL):
    root = runs / f"{prefix}offline"
    save(root / "launch_plan.json", {"protocol": protocol, "model": model})
    save(root / "final_budget.json", {"api_requests": 1, "calls": [{}], "reserved_cny": 0.01})
    return root


def test_optional_registry_preserves_default_and_metadata(tmp_path, monkeypatch):
    captured = []
    result = {"prior_reserved_cny": 53.6, "prior_total_actual_cny": None}
    old_before = deepcopy(budget.catalog.old.REVIEWED_OTHER_SERIES)

    def reconcile(path, *, reviewed_extra_ledgers):
        captured.append(reviewed_extra_ledgers)
        return result

    monkeypatch.setattr(budget, "reconcile_history", reconcile)
    assert budget.reviewed_history(tmp_path) is result
    assert budget.reviewed_history(tmp_path, reviewed_other_series={}) is result
    assert captured[0] == captured[1]
    root = simple_root(tmp_path)
    registration = {NEW_PREFIX: frozenset({NEW_PROTOCOL})}
    assert budget.reviewed_history(tmp_path, reviewed_other_series=registration) is result
    assert captured[2] == (*captured[0], f"{root.name}/final_budget.json")
    assert registration == {NEW_PREFIX: frozenset({NEW_PROTOCOL})}
    assert budget.catalog.old.REVIEWED_OTHER_SERIES == old_before


@pytest.mark.parametrize(
    "registrations",
    [
        [],
        {"../bad_": {NEW_PROTOCOL}},
        {"wild*_": {NEW_PROTOCOL}},
        {"bad\\path_": {NEW_PROTOCOL}},
        {NEW_PREFIX: NEW_PROTOCOL},
        {NEW_PREFIX: set()},
        {NEW_PREFIX: {"bad protocol"}},
        {NEW_PREFIX: {1}},
        {NEW_PREFIX: [NEW_PROTOCOL, NEW_PROTOCOL]},
        {budget.PREFIX: {NEW_PROTOCOL}},
        {budget.PREFIX + "child_": {NEW_PROTOCOL}},
        {"2026-10-03_": {NEW_PROTOCOL}},
        {NEW_PREFIX: {budget.PROTOCOL}},
        {NEW_PREFIX: {NEW_PROTOCOL}, NEW_PREFIX + "nested_": {"growrag-other-v1"}},
        {NEW_PREFIX: {NEW_PROTOCOL}, "2026-10-03_distinct_": {NEW_PROTOCOL}},
    ],
)
def test_invalid_or_overlapping_registrations_fail_before_reconciliation(
    tmp_path, monkeypatch, registrations
):
    monkeypatch.setattr(budget, "reconcile_history", lambda *a, **k: pytest.fail("unsafe scan"))
    with pytest.raises(ValueError):
        budget.reviewed_history(tmp_path, reviewed_other_series=registrations)


@pytest.mark.parametrize(
    "protocol,model",
    [("unregistered", MODEL), (NEW_PROTOCOL, "different-model")],
)
def test_new_root_still_requires_registered_protocol_and_fixed_model(
    tmp_path, monkeypatch, protocol, model
):
    simple_root(tmp_path, protocol=protocol, model=model)
    monkeypatch.setattr(budget, "reconcile_history", lambda *a, **k: pytest.fail("unsafe scan"))
    with pytest.raises(ValueError, match="unregistered"):
        budget.reviewed_history(tmp_path, reviewed_other_series={NEW_PREFIX: {NEW_PROTOCOL}})


def real_ledger(path, trace, *, unknown=False):
    filename = hashlib.sha256(trace.encode()).hexdigest() + ".json"
    raw_path = path.parent / "api_audit" / filename
    row = {
        "trace_id": trace,
        "audit_path": str(raw_path),
        "reserved_cny": 0.01,
        "estimated_actual_cny": None if unknown else 0.000001,
        "status": "failed" if unknown else "completed",
        "input_tokens": None if unknown else 1,
        "output_tokens": None if unknown else 1,
        "api_requests": 1,
    }
    save(raw_path, {**row, "transport_source": "live_api"})
    save(
        path,
        {
            "calls": [row],
            "api_requests": 1,
            "reserved_cny": 0.01,
            "limits": {"input_per_million_cny": 0.2, "output_per_million_cny": 0.8},
        },
    )


def test_new_series_real_reconciliation_keeps_old_and_unknown_reservations(tmp_path):
    legacy = (*prior_budget.ROOT_LEDGERS, *budget.catalog.old.HISTORICAL_ROOTS)
    for i, relative in enumerate(legacy):
        real_ledger(tmp_path / relative, f"offline-old-{i}", unknown=i == 0)
    root = simple_root(tmp_path)
    real_ledger(root / "final_budget.json", "offline-schema-new", unknown=True)
    old_bytes = (tmp_path / legacy[0]).read_bytes()
    result = budget.reviewed_history(tmp_path, reviewed_other_series={NEW_PREFIX: {NEW_PROTOCOL}})
    assert result["prior_reserved_cny"] == pytest.approx((len(legacy) + 1) * 0.01)
    assert result["prior_total_actual_cny"] is None
    assert (tmp_path / legacy[0]).read_bytes() == old_bytes


def synthetic_http(directory, *, strict=False, failed=False):
    trace, prompt = "offline-probe/locate", "growrag-gap-query-v2"
    fmt = output_schemas.response_format_for(prompt) if strict else {"type": "json_object"}
    request = {
        "model": MODEL,
        "max_tokens": 2048,
        "stream": False,
        "enable_thinking": False,
        "temperature": 0,
        "top_p": 1,
        "response_format": fmt,
        "messages": [{"role": "user", "content": "Offline synthetic JSON only."}],
    }
    filename = hashlib.sha256(trace.encode()).hexdigest() + ".json"
    path = directory / "api_audit" / filename
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    call = {
        "trace_id": trace,
        "prompt_version": prompt,
        "status": "failed" if failed else "completed",
        "api_requests": 1,
        "input_tokens": None if failed else 10,
        "output_tokens": None if failed else 5,
        "reserved_cny": 0.01,
        "estimated_actual_cny": None if failed else 0.000006,
        "returned_model": MODEL,
        "audit_path": str(path),
    }
    event = {
        **{key: call[key] for key in ("trace_id", "prompt_version", "status", "api_requests")},
        "transport_source": "live_api",  # Synthetic archive of this wire contract, not a real call.
        "network_attempted": True,
        "retry_count": 0,
        "input_tokens": call["input_tokens"],
        "output_tokens": call["output_tokens"],
        "request": request,
        "request_sha256": hashlib.sha256(
            json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
        ).hexdigest(),
        "http_status": 400 if failed else 200,
        "finish_reason": "stop",
        "response_redacted": False,
        "response": {
            "model": MODEL,
            "usage": usage,
            "choices": [{"finish_reason": "stop", "message": {"content": '{"query":"offline"}'}}],
        },
    }
    if strict:
        event.update(
            output_schema_registry_version=output_schemas.REGISTRY_VERSION,
            output_schema_sha256=output_schemas.schema_fingerprint(prompt),
        )
    if failed:
        event.pop("response")
        event["error_type"] = "HTTPError"
    save(path, event)
    return call, event, path, fmt


def run_http_audit(directory, call, **kwargs):
    return audit.audit_http(
        call,
        directory,
        {"model": MODEL, "max_output_tokens": 2048},
        lambda path: path,
        set(),
        "offline-probe/",
        **kwargs,
    )


def test_http_default_json_object_contract_unchanged(tmp_path):
    call, event, path, _ = synthetic_http(tmp_path)
    before = path.read_bytes()
    _, request, content = run_http_audit(tmp_path, call)
    assert request == event["request"] and content == '{"query":"offline"}'
    assert path.read_bytes() == before


@pytest.mark.parametrize("failed", [False, True])
def test_explicit_strict_http_keeps_raw_request_and_unknown_failure_usage(tmp_path, failed):
    call, event, path, fmt = synthetic_http(tmp_path, strict=True, failed=failed)
    before, original_fmt = path.read_bytes(), deepcopy(fmt)
    _, request, content = run_http_audit(tmp_path, call, expected_response_format=fmt)
    assert request == event["request"] and fmt == original_fmt
    assert path.read_bytes() == before
    assert (content is None) is failed
    if failed:
        assert call["estimated_actual_cny"] is None


def test_strict_wire_cannot_pass_legacy_json_object_auditor(tmp_path):
    call, _, _, _ = synthetic_http(tmp_path, strict=True)
    with pytest.raises(ValueError, match="configuration drift"):
        run_http_audit(tmp_path, call)


@pytest.mark.parametrize(
    "change",
    [
        lambda event: event.update(output_schema_registry_version="unregistered"),
        lambda event: event.update(output_schema_sha256="0" * 64),
        lambda event: event.pop("output_schema_sha256"),
        lambda event: event["request"].update(response_format={"type": "json_object"}),
        lambda event: event.update(request_sha256="0" * 64),
    ],
)
def test_strict_schema_metadata_wire_or_hash_drift_rejected(tmp_path, change):
    call, event, path, fmt = synthetic_http(tmp_path, strict=True)
    change(event)
    save(path, event)
    with pytest.raises(ValueError):
        run_http_audit(tmp_path, call, expected_response_format=fmt)


@pytest.mark.parametrize("value", [{"type": "json_object"}, {}, "json_schema"])
def test_explicit_format_must_be_strict_registered_schema(tmp_path, value):
    call, _, _, _ = synthetic_http(tmp_path, strict=True)
    with pytest.raises(ValueError, match="strict JSON schema"):
        run_http_audit(tmp_path, call, expected_response_format=value)


def test_strict_schema_must_belong_to_exact_prompt_version(tmp_path):
    call, _, _, _ = synthetic_http(tmp_path, strict=True)
    wrong = output_schemas.response_format_for("growrag-short-supported-answer-v2")
    with pytest.raises(ValueError, match="registered prompt"):
        run_http_audit(tmp_path, call, expected_response_format=wrong)
