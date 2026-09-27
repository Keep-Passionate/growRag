"""Read-only shared500 request-parameter audit. Never prints messages or credentials.

Only sealed 2026-09-27_s2g_shared500_v1 batches are included. The output is a
safe parameter projection, not proof of provider-side sampling determinism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PREFIX = "2026-09-27_s2g_shared500_v1_"
MANIFEST_SHA = "2fb117c03b64be3ab12398a297c82ce3315600c14e33f5dcd56c5c6c5c59e064"
MODEL = "qwen3.7-flash-2026-07-15"
PROTOCOL = "growrag-s2g-shared-qwen-capacity-v3"
GENERATION = "qwen_capacity_v5_judge768_extract128_answer1024"
CAPS = {"judge": 768, "extract": 128, "answer": 1024}
ARMS = {"BASE1_AUTHOR_READER", "S2G_AUTHOR_API4"}
SCHEMAS = {
    "judge": "60128708fd26dc4e073c3235e54672c41740e2db01954af142a20a3bb011aa85",
    "extract": "789794b95c6242decf9a9c2900892d7bbc5ced6d9494912614f7aeedec0d1816",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sealed_exit(path):
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 16384))
        lines = handle.read().splitlines()
    if not lines:
        return None
    value = json.loads(lines[-1])
    return value if value.get("kind") == "exit" else None


def project_request(event):
    """Return ONLY allowlisted parameter values and validation labels."""
    issues = []
    version = event.get("prompt_version")
    versions = {f"s2g-author-5d842a6-{stage}-api-v1": stage for stage in CAPS}
    stage = versions.get(version, "unknown")
    if stage == "unknown":
        issues.append("unknown_prompt_version")
    request = event.get("request") or {}
    expected_keys = {
        "model",
        "messages",
        "max_tokens",
        "stream",
        "enable_thinking",
        "temperature",
        "top_p",
    }
    if stage in SCHEMAS:
        expected_keys.add("response_format")
    if set(request) != expected_keys:
        issues.append("unexpected_request_keys")
    parameters = {
        key: request.get(key)
        for key in ("model", "temperature", "top_p", "enable_thinking", "stream", "max_tokens")
    }
    expected = {
        "model": MODEL,
        "temperature": 0,
        "top_p": 1,
        "enable_thinking": False,
        "stream": False,
        "max_tokens": CAPS.get(stage),
    }
    for key, value in parameters.items():
        if value != expected[key]:
            issues.append("parameter_mismatch:" + key)
    if type(parameters["enable_thinking"]) is not bool or type(parameters["stream"]) is not bool:
        issues.append("boolean_parameter_type")
    if type(parameters["max_tokens"]) is not int:
        issues.append("max_tokens_not_integer")
    for key in ("temperature", "top_p"):
        if not isinstance(parameters[key], (int, float)) or isinstance(parameters[key], bool):
            issues.append("numeric_parameter_type:" + key)
    roles = [message.get("role") for message in request.get("messages", [])]
    if roles != ["system", "user"]:
        issues.append("message_role_layout")
    response_format = request.get("response_format")
    schema_sha = sha(canonical(response_format).encode()) if response_format is not None else None
    if stage in SCHEMAS:
        if schema_sha != SCHEMAS[stage]:
            issues.append("strict_schema_wire_fingerprint_mismatch")
        if event.get("output_schema_sha256") != schema_sha:
            issues.append("logged_schema_sha_mismatch")
        if event.get("output_schema_registry_version") != "growrag-output-schemas-v2":
            issues.append("schema_registry_version_mismatch")
    elif response_format is not None or event.get("output_schema_sha256") is not None:
        issues.append("plain_answer_has_schema")
    # Same encoding as the actual API client; message contents are never emitted.
    request_sha = sha(json.dumps(request, ensure_ascii=False, allow_nan=False).encode())
    if request_sha != event.get("request_sha256"):
        issues.append("request_body_hash_mismatch")
    if event.get("retry_count") != 0 or type(event.get("retry_count")) is not int:
        issues.append("retry_count_not_zero")
    if (
        event.get("execution_kind") != "api_transport"
        or event.get("transport_source") != "live_api"
        or event.get("network_attempted") is not True
        or type(event.get("api_requests")) is not int
        or event.get("api_requests") != 1
    ):
        issues.append("not_exactly_one_logged_live_attempt")
    returned = (event.get("response") or {}).get("model")
    if returned is not None and returned != MODEL:
        issues.append("returned_model_mismatch")
    profile = {
        "stage": stage,
        **parameters,
        "message_roles": roles,
        "response_format_type": (response_format or {}).get("type", "absent"),
        "strict_schema": (response_format or {}).get("json_schema", {}).get("strict"),
        "response_format_sha256": schema_sha,
    }
    return profile, issues


def audit(runs_root):
    profiles, stage_counts, returns, retry_counts = Counter(), Counter(), Counter(), Counter()
    errors, skipped, batches, calls = [], [], [], []
    traces, request_ids, response_ids, question_ids = set(), set(), set(), set()
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = sha(raw)
        return json.loads(raw)

    for directory in sorted(Path(runs_root).glob(PREFIX + "*")):
        if not directory.is_dir():
            continue
        terminal = sealed_exit(directory / "events.jsonl")
        if terminal is None or not all(
            (directory / name).is_file() for name in ("launch_plan.json", "final_budget.json")
        ):
            skipped.append(directory.name)
            continue
        launch = read(directory / "launch_plan.json")
        if (
            launch.get("series") != "500_v1"
            or launch.get("manifest_sha256") != MANIFEST_SHA
            or launch.get("run_id") != directory.name
        ):
            raise ValueError("matching directory is outside the exact frozen series")
        for key, expected in (
            ("model", MODEL),
            ("protocol", PROTOCOL),
            ("generation_profile", GENERATION),
            ("backend_output_caps", CAPS),
        ):
            if launch.get(key) != expected:
                errors.append({"location": directory.name, "issue": "launch_mismatch:" + key})
        ledger = read(directory / "final_budget.json")["calls"]
        expected_traces = {item["trace_id"] for item in ledger}
        if len(expected_traces) != len(ledger):
            raise ValueError("duplicate final ledger trace")
        batch_traces = set()
        for path in sorted((directory / "api_audit").glob("*.json")):
            event = read(path)
            trace = event.get("trace_id")
            if not isinstance(trace, str) or trace in traces:
                raise ValueError("missing or duplicate raw request trace")
            traces.add(trace)
            batch_traces.add(trace)
            parts = trace.split("/")
            if (
                len(parts) != 5
                or parts[0] != directory.name
                or parts[1] not in launch["question_ids"]
                or parts[2] not in ARMS
                or parts[3] != "s2g-author"
                or not re.fullmatch(r"\d+-(judge|extract|answer)", parts[4])
            ):
                raise ValueError("request trace outside launch or wrong format")
            question_ids.add(parts[1])
            profile, problems = project_request(event)
            if not parts[4].endswith("-" + profile["stage"]):
                problems.append("trace_stage_mismatch")
            if parts[2] == "BASE1_AUTHOR_READER" and profile["stage"] != "answer":
                problems.append("base_arm_called_nonanswer_role")
            profile["declared_generation_profile"] = launch.get("generation_profile")
            profile["arm"] = parts[2]
            profiles[canonical(profile)] += 1
            stage_counts[profile["stage"]] += 1
            retry_counts[str(event.get("retry_count"))] += 1
            returned = (event.get("response") or {}).get("model")
            returns[returned or "unavailable"] += 1
            for value, seen, label in (
                (event.get("request_id"), request_ids, "request_id"),
                ((event.get("response") or {}).get("id"), response_ids, "response_id"),
            ):
                if value is not None:
                    if value in seen:
                        problems.append("duplicate_" + label)
                    seen.add(value)
            errors.extend({"location": trace, "issue": issue} for issue in problems)
            calls.append(
                {
                    "trace_id": trace,
                    "request_sha256_verified": "request_body_hash_mismatch" not in problems,
                    "profile_sha256": sha(canonical(profile).encode()),
                    "audit_file": str(path),
                    "audit_file_sha256": hashes[str(path)],
                    "validation_issues": problems,
                }
            )
        if batch_traces != expected_traces:
            errors.append(
                {"location": directory.name, "issue": "raw_audits_ledger_trace_set_mismatch"}
            )
        if sealed_exit(directory / "events.jsonl") != terminal:
            raise ValueError("batch seal changed during audit")
        batches.append(
            {
                "run_id": directory.name,
                "audited_calls": len(batch_traces),
                "ledger_calls": len(ledger),
            }
        )
    for path, expected in hashes.items():
        if sha(Path(path).read_bytes()) != expected:
            raise ValueError("sealed input changed during audit")
    return {
        "schema_version": "growrag-shared500-request-profile-audit-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "audit_status": "passed" if not errors and calls else "issues_or_no_calls",
        "series": "500_v1",
        "manifest_sha256": MANIFEST_SHA,
        "planned_question_count": 500,
        "sealed_batch_count": len(batches),
        "unique_started_questions_with_audited_requests": len(question_ids),
        "audited_raw_request_count": len(calls),
        "unique_trace_count": len(traces),
        "unique_available_request_ids": len(request_ids),
        "unique_available_response_ids": len(response_ids),
        "stage_counts": dict(stage_counts),
        "returned_model_counts": dict(returns),
        "retry_count_distribution": dict(retry_counts),
        "parameter_profiles": [
            {"count": count, **json.loads(key)} for key, count in sorted(profiles.items())
        ],
        "issues": errors,
        "sealed_batches": batches,
        "unsealed_skipped": skipped,
        "calls": calls,
        "script_sha256": sha(Path(__file__).read_bytes()),
        "notices": [
            "Only raw requests from sealed exact shared500_v1 batches are audited; "
            "other experiments are excluded.",
            "Failed requests remain in the request denominator; "
            "returned model may be unavailable after transport failure.",
            "No messages, response contents, headers, endpoints, or credentials are emitted.",
            "Unique traces/IDs and retry_count=0 support no logged retry; "
            "these local logs are not provider attestation.",
            "Identical decoding parameters do not guarantee deterministic provider outputs.",
            "Question count is unique IDs with audited requests, "
            "not the complete scored-pair denominator.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    repository_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--runs-root", type=Path, default=repository_root / "runs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        output = args.output.resolve()
        if output.exists() or not output.is_relative_to(repository_root / "runs"):
            raise ValueError("output must be a NEW file inside the repository runs directory")
    result = audit(args.runs_root)
    if args.output is not None:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"calls", "sealed_batches"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
