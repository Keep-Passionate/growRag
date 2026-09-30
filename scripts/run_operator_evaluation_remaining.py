"""Serial frozen v3 evaluation. Only untouched questions continue; no gold or API client.

Seven arms share the published freeze. A failed LAST question is terminal, not a
success: the complete interval is audited before advancing to a new batch.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import run_operator_source_remaining as shared

from growrag.experiments.operator_resume import build_certificate, verify_certificate
from growrag.experiments.run_operator_study import _evaluation_gate

ARMS = ["base", "fresh", "static", "memory50", "memory100", "memory250", "memory500"]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "banks", "evaluation-freeze"):
        result.add_argument(f"--{option}", type=Path, required=True)
    for option in ("manifest", "execution", "freeze"):
        result.add_argument(f"--expected-{option}-sha256", required=True)
    result.add_argument("--start", type=int, default=0)
    result.add_argument("--end", type=int, default=500)
    result.add_argument("--batch-size", type=int, default=25)
    result.add_argument("--project-cap-cny", type=float, default=200)
    result.add_argument("--api-config", type=Path, default=Path("qwenAPI.md"))
    result.add_argument("--resume-certificate", type=Path)
    result.add_argument("--stop-file", type=Path)
    for option in (
        "allow-network",
        "continue-untouched-transport",
        "continue-untouched-recorded-model-errors",
    ):
        result.add_argument(f"--{option}", action="store_true")
    return result


def load_context(args, project):
    if not (0 <= args.start < args.end <= 500 and 1 <= args.batch_size <= 25):
        raise ValueError("invalid fixed500 range or batch size")
    if not math.isfinite(args.project_cap_cny) or not 0 < args.project_cap_cny <= 200:
        raise ValueError("project cap must be finite, positive and at most 200 CNY")
    for key in ("manifest", "execution", "freeze"):
        if not re.fullmatch(r"[0-9a-f]{64}", getattr(args, f"expected_{key}_sha256")):
            raise ValueError("explicit frozen SHA256 values required")
    options = SimpleNamespace(**vars(args))
    options.profile, options.phase, options.arms = shared.ACTION_LIST.name, "evaluation", ARMS
    options.count, options.resume_certificate = min(args.batch_size, args.end - args.start), None
    return _evaluation_gate(options, project)  # Existing full freeze audit, no question text/gold.


def run_name(start, end):
    return f"{shared.ACTION_LIST.prefix}evaluation_{'_'.join(ARMS)}_{start:04d}_{end:04d}"


def command(args, start, end, certificate=None):
    argv = shared.command(args, start, end, certificate)
    argv[argv.index("--phase") + 1] = "evaluation"
    argv[argv.index("--arms") + 1 : argv.index("--budget-cny")] = ARMS
    argv[argv.index("--budget-cny") + 1] = "5"
    return argv + [
        "--banks",
        str(args.banks),
        "--evaluation-freeze",
        str(args.evaluation_freeze),
        "--expected-freeze-sha256",
        args.expected_freeze_sha256,
    ]


def initial_resume(args, project, freeze):
    if args.resume_certificate is None:
        return None, None
    path = (project / args.resume_certificate).resolve(strict=True)
    if not path.is_relative_to(project):
        raise ValueError("initial certificate escapes project")
    wanted = shared._read(path)[0]["proof"]["question_ids"]
    ids = freeze["evaluation_ids"]
    if (
        type(wanted) is not list
        or not 1 <= len(wanted) <= args.batch_size
        or args.start + len(wanted) > args.end
        or wanted != ids[args.start : args.start + len(wanted)]
    ):
        raise ValueError("initial certificate must be the complete untouched evaluation suffix")
    verify_certificate(
        project / "runs",
        path,
        phase="evaluation",
        question_ids=wanted,
        arms=ARMS,
        manifest_sha256=args.expected_manifest_sha256,
        model=freeze["execution_signature"]["configuration"]["model"],
        signature=freeze["execution_signature"],
        profile=shared.ACTION_LIST.name,
        evaluation_freeze=args.evaluation_freeze,
        expected_freeze_sha256=args.expected_freeze_sha256,
    )
    return path, args.start + len(wanted)


def collect(args, project, freeze, *, execute=subprocess.run, certifier=build_certificate):
    ids, runs = freeze["evaluation_ids"], project / "runs"
    environment = {**os.environ, "PYTHONPATH": str(project / "src"), "PYTHONUTF8": "1"}
    start, zero_progress = args.start, 0
    initial_path, initial_end = initial_resume(args, project, freeze)
    while start < args.end:
        end = initial_end if initial_end is not None else min(start + args.batch_size, args.end)
        certificate_path, initial_path, initial_end = initial_path, None, None
        while start < end:
            if shared.stop_requested(args, project):
                print("Stopped at audited boundary; no next child launched.", flush=True)
                return 2
            argv = command(args, start, end, certificate_path)
            print(subprocess.list2cmdline(argv), flush=True)
            if not args.allow_network:
                start = end
                continue
            name, directory = run_name(start, end), runs / run_name(start, end)
            if directory.exists() or (runs / f"{name}.claim.json").exists():
                raise ValueError("target exists; no replay or implicit resume")
            child = execute(argv, cwd=project, env=environment, check=False)
            if type(child.returncode) is not int:
                raise ValueError("child has no verified exit status")
            print(f"Child exited: {name}; code={child.returncode}", flush=True)
            if child.returncode != 0:
                if not (
                    args.continue_untouched_transport
                    or args.continue_untouched_recorded_model_errors
                ):
                    return child.returncode
                shared.continuation_progress(
                    directory, args
                )  # Classify only; ignore three-arm count.
            reports = shared._read(shared._safe_file(directory, "predictions.json"))[0]
            started = [row["question_id"] for row in reports]
            complete = sum(
                list(row["arms"]) == ARMS
                and all(a["status"] == "completed" for a in row["arms"].values())
                for row in reports
            )
            if child.returncode == 0 or started == ids[start:end]:
                terminal = shared.terminal_interval(
                    args, project, name, ids[start:end], phase="evaluation", arms=ARMS
                )
                if (terminal["status"] == "completed") != (child.returncode == 0):
                    raise ValueError("child exit disagrees with terminal interval")
                zero_progress = 0 if complete else zero_progress + 1
                print(
                    f"Interval status: {terminal['status']}; failures remain failures.", flush=True
                )
                if zero_progress >= 3:
                    return 1
                start = end
                break
            certificate = certifier(runs, name, profile=shared.ACTION_LIST.name)
            proof = certificate["proof"]
            binding = proof.get("evaluation_binding", {})
            if (
                not started
                or proof["started_question_ids"] != started
                or not proof["question_ids"]
                or started + proof["question_ids"] != ids[start:end]
                or proof["phase"] != "evaluation"
                or proof["parent_run_id"] != name
                or proof["arms"] != ARMS
                or proof["manifest_sha256"] != args.expected_manifest_sha256
                or proof["source_execution_sha256"] != args.expected_execution_sha256
                or binding.get("freeze_sha256") != args.expected_freeze_sha256
                or binding.get("freeze_path") != str(args.evaluation_freeze.resolve(strict=True))
            ):
                raise ValueError("certificate is not this exact frozen evaluation suffix")
            zero_progress = 0 if complete else zero_progress + 1
            if zero_progress >= 3:
                print("Stopped: three failures without a complete seven-arm question.")
                return 1
            certificate_path = runs / f"{name}_untouched_certificate.json"
            shared.write_json(certificate_path, certificate)
            start += len(started)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    try:
        return collect(args, project, load_context(args, project))
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Evaluation launcher stopped ({type(error).__name__}); preserved audits unchanged.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
