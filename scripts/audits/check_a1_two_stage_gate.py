"""Read-only A1 natural-data gate; no prepare, score writer, retrieval or API.

外部固定三份 SHA 后，从封存 HTTP 重建合成观察及原门槛。完成批次不等于
通过门禁；gate_closed 的退出码非零，而且绝不代替调用者准备自然题。
放在 src 以外，避免改变已冻结实验的运行时源码集合。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

from growrag.experiments import a1_two_stage_feedback as feedback
from growrag.experiments import a1_two_stage_study as study

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FEEDBACK_FILES = {"SUMMARY.json", "per_row.json", "audit.json"}
THRESHOLDS = {
    "completed": 32,
    "main_matches": 29,
    "A_retained": 15,
    "conflict_B_rejected": 12,
    "unknown_B_preserved": 3,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pin(value, label):
    _require(type(value) is str and _SHA.fullmatch(value), f"external {label} SHA256 required")


def _file(project, relative):
    """Reject traversal and redirected inputs before opening their contents."""
    _require(type(relative) is str and "\\" not in relative, "invalid audit input path")
    path = PurePosixPath(relative)
    _require(
        not path.is_absolute() and ".." not in path.parts and str(path) == relative,
        "audit input path must be canonical project-relative",
    )
    candidate = project / path
    resolved = candidate.resolve(strict=True)
    _require(
        resolved.is_relative_to(project)
        and resolved == candidate.absolute()
        and resolved.is_file(),
        "redirected or escaping audit input",
    )
    return resolved


def _json(path):
    return json.loads(path.read_bytes())


def check_gate(project, *, freeze_sha, terminal_sha, feedback_sha):
    """Validate existing sealed artifacts without writing or obtaining new labels."""
    for value, label in (
        (freeze_sha, "freeze"),
        (terminal_sha, "terminal"),
        (feedback_sha, "feedback_frozen"),
    ):
        _pin(value, label)
    project = Path(project).resolve(strict=True)
    freeze_path = _file(project, study.FREEZE)
    _require(study._sha(freeze_path) == freeze_sha, "external freeze SHA mismatch")
    frozen = study.load_freeze(project, freeze_sha)
    target = project / study.FEEDBACK
    seal_path = _file(project, f"{study.FEEDBACK}/feedback_frozen.json")
    _require(study._sha(seal_path) == feedback_sha, "external feedback seal SHA mismatch")
    seals = _json(seal_path)
    _require(type(seals) is dict and set(seals) == _FEEDBACK_FILES, "feedback file set differs")
    _require(
        {path.name for path in target.glob("*.json")} == _FEEDBACK_FILES | {seal_path.name},
        "unsealed feedback JSON exists",
    )
    for name, digest in seals.items():
        _pin(digest, "feedback member")
        _require(
            study._sha(_file(project, f"{study.FEEDBACK}/{name}")) == digest,
            "feedback member SHA mismatch",
        )
    summary = _json(target / "SUMMARY.json")
    details = _json(target / "per_row.json")
    audit = _json(target / "audit.json")
    declared = audit.get("input_sha256")
    _require(
        type(declared) is dict and declared and audit.get("offline_replay") is True,
        "feedback does not declare a sealed offline audit",
    )
    # The synthetic auditor has no reason to open natural-data files or secrets.
    root_prefix = study.OUTPUT + "/"
    claim = f"runs/{study.RUN_ID}.claim.json"
    for relative, digest in declared.items():
        _require(
            type(relative) is str and (relative.startswith(root_prefix) or relative == claim),
            "unregistered audit input",
        )
        _pin(digest, "audit input")
        _require(study._sha(_file(project, relative)) == digest, "audit input SHA mismatch")
    tracked = {}

    def track(path):
        relative = Path(path).absolute().relative_to(project).as_posix()
        safe = _file(project, relative)
        _require(relative in declared, "replayed input was not sealed in audit")
        _require(study._sha(safe) == declared[relative], "replayed input SHA mismatch")
        tracked[relative] = declared[relative]
        return safe

    checked, directory, terminal, ledger, plan = feedback._metadata(
        project, freeze_sha, terminal_sha, track
    )
    _require(checked == frozen, "freeze changed during gate audit")
    captured = []
    seen = set()
    for call in ledger["calls"]:
        captured.append(feedback.audit_http(call, directory, plan, track, seen, study.RUN_ID + "/"))
    _require(
        {p.name for p in (directory / "api_audit").glob("*.json")}
        == {hashlib.sha256(c[0]["trace_id"].encode()).hexdigest() + ".json" for c in captured},
        "unattributed raw HTTP artifact",
    )
    calls_by_trace = {item[0]["trace_id"]: item for item in captured}
    rows = study.load_fixture(project)  # Existing synthetic development annotations only.
    _require(
        len(rows) == 32
        and [r["row_id"] for r in rows]
        == [r["row_id"] for r in terminal["rows"]]
        == frozen["row_ids"],
        "synthetic row identity or coverage differs",
    )
    reports, failures, used, refused = {}, {}, [], None
    stopped, attempted = False, 0
    for index, (row, entry) in enumerate(zip(rows, terminal["rows"], strict=True)):
        status = entry.get("status")
        _require(status in {"completed", "failed", "not_attempted"}, "invalid row terminal")
        if status == "not_attempted":
            _require(set(entry) == {"row_id", "status"}, "unattempted row has result")
            stopped = True
            continue
        _require(not stopped and entry.get("path") == f"row_{index:02d}.json", "row order differs")
        path = track(directory / entry["path"])
        _require(study._sha(path) == entry.get("sha256"), "row terminal seal differs")
        record = _json(path)
        _require(
            record.get("row_id") == row["row_id"]
            and record.get("sequence") == index
            and record.get("input_sha256") == study.input_sha(row["input"])
            and record.get("status") == status,
            "row input identity differs",
        )
        report, row_used, row_refused = feedback._replay_row(
            record, study.prepare_payload(row["input"]), index, calls_by_trace, ledger
        )
        used.extend(row_used)
        if row_refused is not None:
            _require(refused is None, "multiple unsent refusals")
            refused = row_refused
        if report is not None:
            reports[row["row_id"]] = report
        else:
            failures[row["row_id"]] = {
                key: record[key]
                for key in ("failure_type", "failure_code", "observation_status", "fatal")
            }
        attempted += 1
        stopped = record.get("fatal") is True
    _require(
        used == [item[0]["trace_id"] for item in captured], "stage call order/coverage differs"
    )
    _require(
        (terminal["status"] == "completed") == (attempted == 32 and not stopped),
        "batch completion differs from row terminals",
    )
    refusal = feedback.audit_journal(directory, ledger, captured, track, refused_stage=refused)
    _require(tracked == declared, "audit contains unused or missing input declarations")
    gate, rebuilt = feedback.gate_summary(rows, reports)
    agreement = {kind: {"matched": 0, "observed": 0} for kind in ("T", "R", "P0", "P1")}
    for detail in rebuilt:
        if detail["status"] != "scored":
            if detail["row_id"] in failures:
                detail["observation_failure"] = failures[detail["row_id"]]
            continue
        expected = {(c["kind"], c["slot"]): c["status"] for c in detail["expected"]["checks"]}
        for condition in detail["observed"]["checks"]:
            agreement[condition["kind"]]["observed"] += 1
            agreement[condition["kind"]]["matched"] += (
                condition["status"] == expected[condition["kind"], condition["slot"]]
            )
    passed = (
        gate["completed"] == 32
        and gate["main_matches"] >= 29
        and gate["A_retained"] >= 15
        and gate["conflict_B_rejected"] >= 12
        and gate["unknown_B_preserved"] == 3
    )
    _require(gate["passed"] is passed, "computed gate disagrees with fixed thresholds")
    _require(
        details == rebuilt and summary.get("gate") == gate, "sealed feedback differs from replay"
    )
    amounts = feedback.totals(ledger["calls"])
    _require(
        summary.get("protocol") == study.PROTOCOL
        and summary.get("freeze_sha256") == freeze_sha
        and summary.get("terminal_sha256") == terminal_sha
        and summary.get("batch_status") == terminal["status"]
        and summary.get("attempted_rows") == attempted
        and summary.get("observation_failures") == failures
        and summary.get("dimension_nominal_agreement") == agreement
        and summary.get("totals") == amounts
        and summary.get("api_calls") == 0
        and summary.get("gold_loaded") is False
        and summary.get("memory_updated") is False
        and audit.get("stages") == used
        and audit.get("http_requests") == len(captured),
        "summary, HTTP, usage or observation audit differs",
    )
    unknown = sum(
        any(
            type(call.get(key)) not in (int, float) or not math.isfinite(call[key]) or call[key] < 0
            for key in ("input_tokens", "output_tokens", "estimated_actual_cny")
        )
        for call in ledger["calls"]
    )
    safe_usage = not unknown and not ledger.get("block_reason") and not refusal
    allowed = passed and terminal["status"] == "completed" and safe_usage
    reasons = []
    for key, threshold in THRESHOLDS.items():
        okay = (
            gate[key] == threshold
            if key in {"completed", "unknown_B_preserved"}
            else gate[key] >= threshold
        )
        if not okay:
            reasons.append(f"{key}_threshold_not_met")
    if terminal["status"] != "completed":
        reasons.append("batch_not_completed")
    if not safe_usage:
        reasons.append("new_usage_or_transport_not_complete")
    # Final byte checks close the read-only verification against mid-audit changes.
    _require(study._sha(seal_path) == feedback_sha, "feedback seal changed during audit")
    _require(study._sha(freeze_path) == freeze_sha, "freeze changed during audit")
    for relative, digest in declared.items():
        _require(study._sha(_file(project, relative)) == digest, "input changed during gate audit")
    return {
        "status": "gate_open" if allowed else "gate_closed",
        "natural_prepare_allowed": allowed,
        "reasons": reasons,
        "gate": gate,
        "fixed_thresholds": THRESHOLDS,
        "http_requests": len(captured),
        "stage_counts": {
            stage: sum(trace.endswith("/" + stage) for trace in used)
            for stage in ("locate", "judge")
        },
        "new_unknown_usage_requests": unknown,
        "api_calls": 0,
        "gold_loaded": False,
        "memory_updated": False,
        "files_written": 0,
        "preparation_performed": False,
        "notice": (
            "Read-only authorization gate, not a natural-data preparation command "
            "or RAG accuracy result."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-terminal-sha256", required=True)
    parser.add_argument("--expected-feedback-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = check_gate(
            args.project,
            freeze_sha=args.expected_freeze_sha256,
            terminal_sha=args.expected_terminal_sha256,
            feedback_sha=args.expected_feedback_sha256,
        )
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(
            json.dumps(
                {
                    "status": "integrity_error",
                    "natural_prepare_allowed": False,
                    "error_type": type(error).__name__,
                    "api_calls": 0,
                    "files_written": 0,
                },
                indent=2,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["natural_prepare_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
