"""Serial source-only launcher; failures never authorize replay of attempted questions.

默认只展示命令。网络模式逐个等待原 runner 真正退出，不用锁文件推断进程状态。
只有显式允许、原始传输审计一致、完整续行证明通过，才能运行未启动后缀。
本文件不读 gold/问题正文，不改方法，不持有 API 客户端，也不另算预算。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from growrag.experiments.operator_profiles import ACTION_LIST, action_list_execution_signature
from growrag.experiments.operator_resume import (
    _evaluation_binding,
    _method_signature,
    build_certificate,
    verify_certificate,
)
from growrag.experiments.pre_pilot import write_json
from growrag.experiments.representation_runner import fingerprint
from growrag.experiments.shared_continuation import _read, _request_evidence, _safe_file

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
    result.add_argument("--resume-certificate", type=Path)
    result.add_argument("--stop-file", type=Path)
    result.add_argument("--allow-network", action="store_true")
    result.add_argument("--continue-untouched-transport", action="store_true")
    result.add_argument("--continue-untouched-recorded-model-errors", action="store_true")
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


def recorded_model_progress(directory):
    """Recorded post-response rejection is NOT success and never authorizes a retry."""
    budget, _ = _read(_safe_file(directory, "final_budget.json"))
    reports, _ = _read(_safe_file(directory, "predictions.json"))
    failures = [a for row in reports for a in row["arms"].values() if a["status"] == "failed"]
    calls = budget["calls"]
    if (
        budget.get("block_reason") is not None
        or len(failures) != 1
        or not calls
        or failures[0].get("error_type") not in {"ValueError", "TypeError"}
        or any(c.get("status") != "completed" or c.get("validation_status") for c in calls)
        or failures[0].get("calls", [])[-1:] != calls[-1:]
    ):
        raise ValueError("not an isolated recorded model-output rejection")
    failed, call = failures[0], calls[-1]
    audit_path = Path(call["audit_path"]).resolve(strict=True)
    if audit_path.parent != (directory / "api_audit").resolve(strict=True):
        raise ValueError("model response audit escapes run")
    audit, _ = _read(audit_path)
    trace = f"{directory.name}/{failed['question_id']}/{failed['arm']}"
    kind = "reader_record" if call["trace_id"] == f"{trace}/reader" else "planner_record"
    if (
        kind == "planner_record"
        and not re.fullmatch(re.escape(trace) + r"/plan/[1-9][0-9]*", call["trace_id"])
        or audit.get("trace_id") != call["trace_id"]
        or audit.get("status") != "completed"
        or call.get("api_requests") != 1
        or audit.get("http_status") != 200
        or audit.get("network_attempted") is not True
        or audit.get("retry_count") != 0
        or audit.get("api_requests") != 1
        or audit.get("transport_source") != "live_api"
        or audit.get("response_redacted") is not False
        or audit.get("finish_reason") != "stop"
        or audit.get("error_type") is not None
    ):
        raise ValueError("missing matching completed HTTP200 live model response")
    events = [
        json.loads(line)
        for line in _safe_file(directory, "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_indices = [
        i
        for i, event in enumerate(events)
        if event.get("kind") == kind
        and event.get("stage") == "raw_wire"
        and event.get("question_id") == failed["question_id"]
        and event.get("arm") == failed["arm"]
    ]
    if not raw_indices:
        raise ValueError("model output has no raw planner/reader record")
    index = raw_indices[-1]
    record, tail = events[index], events[index + 1 :]
    choices = audit["response"]["choices"]
    if (
        len(choices) != 1
        or record.get("raw_content") != choices[0]["message"]["content"]
        or record.get("prompt_version") != audit.get("prompt_version")
        or record.get("prompt_version") != call.get("prompt_version")
        or record.get("payload") != json.loads(audit["request"]["messages"][1]["content"])
        or len(tail) not in {2, 3}
        or len(tail) == 3
        and (
            kind != "planner_record"
            or tail[0].get("kind") != kind
            or tail[0].get("stage") != "normalized_parser_input"
        )
        or tail[-2].get("kind") != "failure"
        or tail[-2].get("error_type") != failed["error_type"]
        or tail[-1].get("kind") != "exit"
        or tail[-1].get("status") != "failed"
        or any(
            e.get("question_id") != failed["question_id"] or e.get("arm") != failed["arm"]
            for e in tail
        )
    ):
        raise ValueError("raw model output/ordered failure records differ from API response")
    return sum(
        list(row["arms"]) == ARMS and all(a["status"] == "completed" for a in row["arms"].values())
        for row in reports
    )


def continuation_progress(directory, args):
    budget, _ = _read(_safe_file(directory, "final_budget.json"))
    if args.continue_untouched_transport and budget.get("block_reason") == "transport_failure":
        return transport_progress(directory)
    if args.continue_untouched_recorded_model_errors and budget.get("block_reason") is None:
        return recorded_model_progress(directory)
    raise ValueError("failure class is not explicitly enabled for untouched continuation")


def audited_suffix(
    args, runs, name, expected_ids, *, certifier=build_certificate, classify=transport_progress
):
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
    completed = classify(runs / name)
    return certificate, len(started), completed


def initial_resume(args, project, ids):
    """An explicit fully reaudited suffix may shorten the FIRST batch, never later ones."""
    if args.resume_certificate is None:
        return None, None
    path = (project / args.resume_certificate).resolve(strict=True)
    if not path.is_relative_to(project):
        raise ValueError("initial certificate must stay inside project")
    wanted = _read(path)[0]["proof"]["question_ids"]
    if (
        type(wanted) is not list
        or not 1 <= len(wanted) <= args.batch_size
        or args.start + len(wanted) > args.end
        or wanted != ids[args.start : args.start + len(wanted)]
    ):
        raise ValueError("initial certificate must cover its complete contiguous source suffix")
    signature = action_list_execution_signature(project)
    if signature["sha256"] != args.expected_execution_sha256:
        raise ValueError("initial certificate method mismatch")
    verify_certificate(
        project / "runs",
        path,
        phase="source",
        question_ids=wanted,
        arms=ARMS,
        manifest_sha256=args.expected_manifest_sha256,
        model=signature["configuration"]["model"],
        signature=signature,
        profile=ACTION_LIST.name,
    )
    return path, args.start + len(wanted)


def terminal_interval(args, project, name, expected_ids, *, phase="source", arms=ARMS):
    """Fully audit a closed interval; terminal_failed is never prediction success.

    No empty resume certificate is fabricated when the LAST question failed.
    The caller must already have waited for its own child and classified any failure.
    """
    runs, directory = project / "runs", (project / "runs" / name).resolve(strict=True)
    if directory.parent != runs.resolve(strict=True):
        raise ValueError("terminal interval escapes runs")

    def read(filename):
        return _read(_safe_file(directory, filename))[0]

    launch, reports, seal, budget = (
        read(n)
        for n in (
            "launch_plan.json",
            "predictions.json",
            "predictions_frozen.json",
            "final_budget.json",
        )
    )
    claim = _read(_safe_file(runs, f"{name}.claim.json"))[0]
    if (
        launch.get("run_id") != name
        or launch.get("protocol") != ACTION_LIST.protocol
        or launch.get("profile") != ACTION_LIST.name
        or launch.get("phase") != phase
        or launch.get("arms") != arms
        or launch.get("question_ids") != expected_ids
        or launch.get("gold_loaded") is not False
        or launch.get("memory_updates") is not False
        or launch.get("manifest_sha256") != args.expected_manifest_sha256
        or _method_signature(launch.get("execution_signature"), ACTION_LIST)
        != args.expected_execution_sha256
        or launch.get("model") != launch["execution_signature"]["configuration"]["model"]
        or claim != {**launch, "plan_sha256": fingerprint(launch)}
        or [row.get("question_id") for row in reports] != expected_ids
        or seal.get("question_ids") != expected_ids
        or seal.get("sha256") != fingerprint(reports)
        or seal.get("phase") != phase
        or seal.get("gold_loaded") is not False
        or seal.get("status") not in {"completed", "failed"}
        or seal.get("cleanup_errors") != []
    ):
        raise ValueError("interval is not completely and cleanly terminal under the frozen method")
    owned, records, failed_count, complete = [], set(), 0, 0
    for number, row in enumerate(reports):
        qid, outcomes = row["question_id"], row["arms"]
        checkpoint = f"checkpoint_{number:04d}.json"
        if (
            not outcomes
            or list(outcomes) != arms[: len(outcomes)]
            or read(checkpoint) != row
            or row.get("feedback") is not None
            or row.get("memory_updated", False)
        ):
            raise ValueError("terminal checkpoint or arm sequence differs")
        records.add(checkpoint)
        failures = []
        for arm, outcome in outcomes.items():
            filename = f"{qid}_{arm}.json"
            records.add(filename)
            if (
                read(filename) != outcome
                or outcome.get("question_id") != qid
                or outcome.get("arm") != arm
                or outcome.get("status") not in {"completed", "failed"}
                or outcome.get("feedback") is not None
                or outcome.get("memory_updated") is not False
                or type(outcome.get("calls")) is not list
                or any(
                    not c.get("trace_id", "").startswith(f"{name}/{qid}/{arm}/")
                    for c in outcome["calls"]
                )
            ):
                raise ValueError("terminal arm report or call ownership differs")
            owned.extend(outcome["calls"])
            if outcome["status"] == "failed":
                failures.append(arm)
        if (
            len(failures) > 1
            or failures
            and (number != len(reports) - 1 or failures != [list(outcomes)[-1]])
            or not failures
            and list(outcomes) != arms
        ):
            raise ValueError("only the final question may have one terminal failing arm")
        failed_count += len(failures)
        complete += int(not failures)
    actual = {
        p.name
        for p in directory.iterdir()
        if re.fullmatch(r"(?:[0-9a-f]{24}_.+|checkpoint_.+)\.json", p.name)
    }
    if (
        records != actual
        or budget.get("calls") != owned
        or len({c["trace_id"] for c in owned}) != len(owned)
        or (seal["status"] == "failed") != (failed_count == 1)
    ):
        raise ValueError("terminal files, final ledger or failure count differ")
    events = [
        json.loads(line)
        for line in _safe_file(directory, "events.jsonl").read_bytes().splitlines()
        if line.strip()
    ]
    if (
        not events
        or events[-1].get("kind") != "exit"
        or events[-1].get("status") != seal["status"]
        or events[-1].get("requests") != budget.get("api_requests")
        or any(e.get("kind") == "exit" for e in events[:-1])
    ):
        raise ValueError("terminal exit and ledger differ")
    _request_evidence(directory, budget, set(), reports, events[-4:])
    for filename in ("live.log", "process.json", "source_snapshot.json", "cumulative_budget.json"):
        _safe_file(directory, filename)
    if phase == "evaluation":
        binding = _evaluation_binding(runs, launch, read, ACTION_LIST)
        if (
            binding["freeze_path"] != str(args.evaluation_freeze.resolve(strict=True))
            or binding["freeze_sha256"] != args.expected_freeze_sha256
        ):
            raise ValueError("terminal evaluation freeze differs")
    if launch.get("resume_parent_run_id") is not None:
        certificate = read("resume_certificate.json")
        if (
            fingerprint(certificate) != launch.get("resume_certificate_sha256")
            or certificate["proof"]["parent_run_id"] != launch["resume_parent_run_id"]
        ):
            raise ValueError("terminal ancestry differs")
        verify_certificate(
            runs,
            directory / "resume_certificate.json",
            phase=phase,
            question_ids=expected_ids,
            arms=arms,
            manifest_sha256=args.expected_manifest_sha256,
            model=launch["model"],
            signature=launch["execution_signature"],
            profile=ACTION_LIST.name,
            **(
                {
                    "evaluation_freeze": args.evaluation_freeze,
                    "expected_freeze_sha256": args.expected_freeze_sha256,
                }
                if phase == "evaluation"
                else {}
            ),
        )
    return {
        "status": "terminal_failed" if failed_count else "completed",
        "completed_questions": complete,
    }


def stop_requested(args, project):
    if args.stop_file is None:
        return False
    path = (project / args.stop_file).resolve()
    if not path.is_relative_to(project):
        raise ValueError("stop file must stay inside project")
    return path.exists()  # Do not read contents or terminate an in-flight child.


def collect(args, project, ids, *, execute=subprocess.run, certifier=build_certificate):
    """No detached jobs or shell commands: execute returns only when child has exited."""
    runs = project / "runs"
    environment = {**os.environ, "PYTHONPATH": str(project / "src"), "PYTHONUTF8": "1"}
    start, zero_progress = args.start, 0
    initial_path, initial_end = initial_resume(args, project, ids)
    while start < args.end:
        end = initial_end if initial_end is not None else min(start + args.batch_size, args.end)
        certificate_path = initial_path
        initial_path, initial_end = None, None
        while start < end:
            if stop_requested(args, project):
                print("Stopped at audited boundary; no next child launched.", flush=True)
                return 2
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
            if not (
                args.continue_untouched_transport or args.continue_untouched_recorded_model_errors
            ):
                return child.returncode
            if (
                _read(_safe_file(runs / name, "predictions_frozen.json"))[0].get("question_ids")
                == ids[start:end]
            ):
                continuation_progress(runs / name, args)
                terminal = terminal_interval(args, project, name, ids[start:end])
                zero_progress = 0 if terminal["completed_questions"] else zero_progress + 1
                print(
                    f"Interval status: {terminal['status']}; failed question will not be replayed."
                )
                if zero_progress >= 3:
                    return 1
                start = end
                break
            certificate, used, completed = audited_suffix(
                args,
                runs,
                name,
                ids[start:end],
                certifier=certifier,
                classify=lambda directory: continuation_progress(directory, args),
            )
            zero_progress = 0 if completed else zero_progress + 1
            if zero_progress >= 3:
                print("Stopped: three consecutive recorded failures without a full question.")
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
