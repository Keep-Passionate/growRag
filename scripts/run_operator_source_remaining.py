"""Serial source-only launcher; failures never authorize replay of attempted questions.

默认只展示命令。网络模式逐个等待原 runner 真正退出，不用锁文件推断进程状态。
只有显式允许、原始传输审计一致、完整续行证明通过，才能运行未启动后缀。
本文件不读 gold/问题正文，不改方法，不持有 API 客户端，也不另算预算。
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from growrag.experiments.operator_profiles import ACTION_LIST, action_list_execution_signature
from growrag.experiments.operator_resume import build_certificate
from growrag.experiments.pre_pilot import write_json
from growrag.experiments.representation_runner import fingerprint
from growrag.experiments.shared_continuation import _read, _safe_file

ARMS = ["base", "fresh", "static"]
TRANSPORT_ERRORS = {"TimeoutError", "URLError"}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--expected-manifest-sha256", required=True)
    result.add_argument("--expected-execution-sha256", required=True)
    result.add_argument("--start", type=int, required=True)
    result.add_argument("--end", type=int, default=500)
    result.add_argument("--batch-size", type=int, default=25)
    result.add_argument("--project-cap-cny", type=float, default=50)
    result.add_argument("--api-config", type=Path, default=Path("qwenAPI.md"))
    result.add_argument("--allow-network", action="store_true")
    result.add_argument("--continue-untouched-transport", action="store_true")
    return result


def source_ids(args, project):
    """Read only the frozen manifest's role IDs; runtime/gold files stay unopened."""
    if not (0 <= args.start < args.end <= 500 and 1 <= args.batch_size <= 25):
        raise ValueError("require 0 <= start < end <= 500 and batch size 1..25")
    if not math.isfinite(args.project_cap_cny) or not 0 < args.project_cap_cny <= 200:
        raise ValueError("project cap must be positive, finite and at most 200 CNY")
    for digest in (args.expected_manifest_sha256, args.expected_execution_sha256):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("explicit manifest and method SHA256 values required")
    path = (project / args.manifest).resolve(strict=True)
    if not path.is_relative_to(project):
        raise ValueError("manifest must stay inside project")
    manifest, digest = _read(path)
    if digest != args.expected_manifest_sha256:
        raise ValueError("manifest changed")
    if action_list_execution_signature(project)["sha256"] != args.expected_execution_sha256:
        raise ValueError("frozen action-list-v3 method changed")
    ids = manifest["roles"]["source"]
    if (
        type(ids) is not list
        or len(ids) != 500
        or any(type(qid) is not str or not re.fullmatch(r"[0-9a-f]{24}", qid) for qid in ids)
        or len(set(ids)) != 500
        or set(ids) & set(manifest["roles"]["calibration"])
        or set(ids) & set(manifest["roles"]["evaluation"])
    ):
        raise ValueError("invalid or overlapping frozen source IDs")
    return ids


def run_name(start, end):
    return f"{ACTION_LIST.prefix}source_base_fresh_static_{start:04d}_{end:04d}"


def command(args, start, end, certificate=None):
    result = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "growrag.experiments.run_operator_study",
        "--profile",
        ACTION_LIST.name,
        "--manifest",
        str(args.manifest),
        "--expected-manifest-sha256",
        args.expected_manifest_sha256,
        "--expected-execution-sha256",
        args.expected_execution_sha256,
        "--phase",
        "source",
        "--start",
        str(start),
        "--count",
        str(end - start),
        "--arms",
        *ARMS,
        "--budget-cny",
        "2",
        "--api-config",
        str(args.api_config),
        "--project-cap-cny",
        str(args.project_cap_cny),
    ]
    if certificate is not None:
        result.extend(["--resume-certificate", str(certificate)])
    if args.allow_network:
        result.append("--allow-network")
    return result


def completed_batch(directory, expected_ids):
    """Child exit 0 AND exact clean completed seal are required before advancing."""
    seal, _ = _read(_safe_file(directory, "predictions_frozen.json"))
    reports, _ = _read(_safe_file(directory, "predictions.json"))
    if (
        seal.get("status") != "completed"
        or seal.get("cleanup_errors") != []
        or seal.get("phase") != "source"
        or seal.get("gold_loaded") is not False
        or seal.get("question_ids") != expected_ids
        or seal.get("sha256") != fingerprint(reports)
        or [row.get("question_id") for row in reports] != expected_ids
        or any(
            list(row.get("arms", {})) != ARMS
            or any(arm.get("status") != "completed" for arm in row["arms"].values())
            for row in reports
        )
    ):
        raise ValueError("successful child lacks complete clean predictions")


def transport_progress(directory):
    """Accept only one failed network attempt, matched to the failed arm and audit."""
    budget, _ = _read(_safe_file(directory, "final_budget.json"))
    reports, _ = _read(_safe_file(directory, "predictions.json"))
    failures = [a for row in reports for a in row["arms"].values() if a["status"] == "failed"]
    calls = budget["calls"]
    failed_calls = [call for call in calls if call.get("status") != "completed"]
    if (
        budget.get("block_reason") != "transport_failure"
        or len(failures) != 1
        or failures[0].get("error_type") != "APIRequestError"
        or len(failed_calls) != 1
        or failed_calls[0] != calls[-1]
        or failures[0].get("calls", [])[-1:] != failed_calls
    ):
        raise ValueError("not an isolated audited transport failure; stop")
    call = failed_calls[0]
    audit_path = Path(call["audit_path"]).resolve(strict=True)
    if audit_path.parent != (directory / "api_audit").resolve(strict=True):
        raise ValueError("API audit path escapes run")
    audit, _ = _read(audit_path)
    if (
        audit.get("error_type") not in TRANSPORT_ERRORS
        or audit.get("status") != "failed"
        or audit.get("trace_id") != call["trace_id"]
        or audit.get("network_attempted") is not True
        or audit.get("retry_count") != 0
        or audit.get("api_requests") != 1
        or audit.get("transport_source") != "live_api"
    ):
        raise ValueError("API audit does not prove the same transport failure")
    return sum(
        list(row["arms"]) == ARMS
        and all(arm["status"] == "completed" for arm in row["arms"].values())
        for row in reports
    )


def audited_suffix(args, runs, name, expected_ids, *, certifier=build_certificate):
    certificate = certifier(runs, name, profile=ACTION_LIST.name)
    proof = certificate["proof"]
    started, remaining = proof["started_question_ids"], proof["question_ids"]
    if (
        not started
        or not remaining
        or started + remaining != expected_ids
        or proof["phase"] != "source"
        or proof["parent_run_id"] != name
        or proof["arms"] != ARMS
        or proof["manifest_sha256"] != args.expected_manifest_sha256
        or proof["source_execution_sha256"] != args.expected_execution_sha256
    ):
        raise ValueError("certificate does not prove the exact strictly advancing source suffix")
    completed = transport_progress(runs / name)
    return certificate, len(started), completed


def collect(args, project, ids, *, execute=subprocess.run, certifier=build_certificate):
    """No detached jobs or shell commands: execute returns only when child has exited."""
    runs = project / "runs"
    environment = {**os.environ, "PYTHONPATH": str(project / "src"), "PYTHONUTF8": "1"}
    start, zero_progress = args.start, 0
    while start < args.end:
        end, certificate_path = min(start + args.batch_size, args.end), None
        while start < end:
            argv = command(args, start, end, certificate_path)
            print(subprocess.list2cmdline(argv), flush=True)
            if not args.allow_network:
                start = end
                continue
            name = run_name(start, end)
            # Existing claims need manual audited entry. Never infer restart from files or locks.
            if (runs / name).exists() or (runs / f"{name}.claim.json").exists():
                raise ValueError("target already exists; no replay or implicit resume")
            child = execute(argv, cwd=project, env=environment, check=False)
            if type(child.returncode) is not int:
                raise ValueError("child exit status unavailable; do not restart")
            print(f"Child exited: {name}; code={child.returncode}", flush=True)
            if child.returncode == 0:
                completed_batch(runs / name, ids[start:end])
                start, zero_progress = end, 0
                continue
            if not args.continue_untouched_transport:
                return child.returncode
            certificate, used, completed = audited_suffix(
                args, runs, name, ids[start:end], certifier=certifier
            )
            zero_progress = 0 if completed else zero_progress + 1
            if zero_progress >= 3:
                print("Stopped: three consecutive transport failures without a full question.")
                return 1
            certificate_path = runs / f"{name}_untouched_certificate.json"
            write_json(certificate_path, certificate)  # Exclusive; never overwrite prior proof.
            start += used  # Resume the COMPLETE untouched suffix, never a failed question.
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    try:
        ids = source_ids(args, project)
        return collect(args, project, ids)
    except (ValueError, KeyError, TypeError, OSError) as error:
        # Do not print arbitrary exception contents, credential files or request payloads.
        print(f"Source launcher stopped ({type(error).__name__}); inspect preserved audit files.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
