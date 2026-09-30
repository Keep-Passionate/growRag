"""Thin v3 coordinator: existing audited CLIs do the research; no new model logic.

默认仅列出阶段，不花钱、不读标签、不关机。真实模式串行等待每个子进程退出；
标签门由原评分器执行，失败题不重试。只有显式开关才同步白名单汇总或非强制关机。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from growrag.experiments.operator_profiles import action_list_execution_signature
from growrag.experiments.run_shared_s2g import reviewed_history

PROFILE = "action-list-v3"
FEEDBACK = "runs/operator_v3_source_feedback"
BANKS = "runs/operator_v3_source_banks"
FREEZE = "runs/operator_v3_evaluation_freeze.json"
EVALUATION = "runs/operator_v3_evaluation_feedback"
PUBLIC = "docs/experiments/2026-10-01_v3流水线执行状态.md"
STAGES = ("source500", "source_score", "banks", "freeze", "evaluation500", "evaluation_score")
WINDOWS = os.name == "nt"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--expected-manifest-sha256", required=True)
    result.add_argument("--expected-execution-sha256", required=True)
    result.add_argument("--source-start", type=int, required=True)
    result.add_argument("--source-resume-certificate", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--project-cap-cny", type=float, default=200)
    result.add_argument("--allow-network", action="store_true")
    result.add_argument("--sync-github", action="store_true")
    result.add_argument("--shutdown-on-complete", action="store_true")
    result.add_argument("--shutdown-on-quota-stop", action="store_true")
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("task path escapes workspace")
    return path


def record(path, value, *, append=False):
    """Fsync generated audit artifacts before proceeding or requesting shutdown."""
    with path.open("a" if append else "x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def budget_snapshot(root):
    history = reviewed_history(root / "runs")
    return {
        key: history[key]
        for key in (
            "prior_api_requests",
            "prior_reserved_cny",
            "prior_known_estimated_cny",
            "prior_unknown_cost_requests",
        )
    }


def execute(root, output, stage, argv):
    """No shell expansion and no detached children: return only after real exit."""
    event = {"stage": stage, "utc": datetime.now(UTC).isoformat(), "argv": argv}
    record(output / "events.jsonl", {**event, "kind": "start"}, append=True)
    print(f"[pipeline] {stage}", flush=True)
    environment = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONUTF8": "1"}
    with (output / f"{stage}.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            code = process.wait()
        except BaseException:
            # Preserve the child's own output stream. A boundary stop is NOT a quota stop.
            marker = output / "stop_requested.flag"
            try:
                if not marker.exists():
                    record(marker, {"reason": "coordinator_interrupted"})
            finally:
                while process.poll() is None:
                    try:
                        process.wait()
                    except BaseException:
                        continue
            raise
        log.flush()
        os.fsync(log.fileno())
    record(output / "events.jsonl", {**event, "kind": "exit", "returncode": code}, append=True)
    return code


def stage_command(args, root, stage, stop_file):
    py = [sys.executable, "-X", "utf8"]

    def module(name):
        return [*py, "-m", "growrag.experiments." + name]

    common = ["--profile", PROFILE]
    manifest = ["--manifest", str(args.manifest)]
    signed = [
        "--expected-manifest-sha256",
        args.expected_manifest_sha256,
        "--expected-execution-sha256",
        args.expected_execution_sha256,
    ]
    supervisor = [
        *manifest,
        *signed,
        "--end",
        "500",
        "--batch-size",
        "25",
        "--project-cap-cny",
        str(args.project_cap_cny),
        "--api-config",
        "qwenAPI.md",
        "--allow-network",
        "--continue-untouched-transport",
        "--continue-untouched-recorded-model-errors",
        "--stop-file",
        str(stop_file),
    ]
    if stage == "source500":
        command = [
            *py,
            "scripts/run_operator_source_remaining.py",
            *supervisor,
            "--start",
            str(args.source_start),
        ]
        if args.source_resume_certificate is not None:
            command += ["--resume-certificate", str(args.source_resume_certificate)]
        return command
    if stage == "source_score":
        return [
            *module("score_operator_sources"),
            *common,
            *manifest,
            "--runs",
            "runs",
            "--output-dir",
            FEEDBACK,
            "--score-new",
        ]
    if stage == "banks":
        return [
            *module("build_operator_banks"),
            *common,
            *manifest,
            "--feedback-dir",
            FEEDBACK,
            "--output-dir",
            BANKS,
            "--score-audit-sha256",
            sha(root / FEEDBACK / "audit.json"),
            "--build-new",
        ]
    if stage == "freeze":
        return [
            *module("operator_evaluation_freeze"),
            *common,
            *manifest,
            *signed,
            "--certificate",
            FREEZE,
            "--runs",
            "runs",
            "--feedback-dir",
            FEEDBACK,
            "--banks-dir",
            BANKS,
            "--expected-scoring-audit-sha256",
            sha(root / FEEDBACK / "audit.json"),
            "--expected-bank-bundle-sha256",
            sha(root / BANKS / "bank_bundle.json"),
            "--issue-new",
        ]
    if stage == "evaluation500":
        return [
            *py,
            "scripts/run_operator_evaluation_remaining.py",
            *supervisor,
            "--start",
            "0",
            "--banks",
            BANKS,
            "--evaluation-freeze",
            FREEZE,
            "--expected-freeze-sha256",
            sha(root / FREEZE),
        ]
    if stage == "evaluation_score":
        return [
            *module("score_operator_evaluation"),
            *common,
            "--certificate",
            FREEZE,
            "--expected-certificate-sha256",
            sha(root / FREEZE),
            "--runs",
            "runs",
            "--output-dir",
            EVALUATION,
            "--score-new",
        ]
    raise ValueError("unknown pipeline stage")


def public_report(root, result):
    """Publish allowlisted aggregates, never raw prompts, answers or memory text."""
    lines = [
        "# v3 流水线执行状态",
        "",
        f"更新时间（UTC）：{result['utc']}",
        f"状态：{result['status']}；最后阶段：{result['stage']}",
        f"已成功退出的阶段：{', '.join(result['completed_stages']) or '无'}",
        "",
        "500来源与500评价是两个独立集合；终态覆盖不等于500题全成功。",
        "当前FRESH是同能力动态算子对照，不冒称原版S2G。共享封闭语料，不报官方排行榜成绩。",
        f"方法SHA：`{result['method_sha256']}`",
        "",
    ]
    budget = result.get("budget", {})
    for key, value in budget.items():
        lines.append(f"- {key}：{value}")
    lines += [
        "",
        "费用是按冻结单价估算，不是供应商账单；未知请求保留预留。",
        f"本地日志：`{result['output_dir']}`；完整轨迹仅保存在本地runs，不进入GitHub。",
        "",
    ]
    if result["status"] == "completed":
        summary = json.loads((root / EVALUATION / "summary.json").read_text(encoding="utf-8"))
        lines += [
            "## 独立评价汇总",
            "",
            "| 分支 | 完成题 | EM有效题数 | EM均值 | 实际复用题 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for arm, row in summary["arms"].items():
            em = row["own_scorable"]["answer_em"]
            lines.append(
                f"| {arm} | {row['status_counts']['completed']} | {em['n']} | "
                f"{em['mean']} | {row['execution']['executed_reuse_questions']} |"
            )
        lines += [
            "",
            "各臂有效集合可能不同，不能直接将此表作为配对胜负。",
            "",
            "| 记忆−FRESH | 有效配对数 | EM净增量 | 修复 | 损害 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, pair in summary["paired"].items():
            if name.startswith("memory") and name.endswith("_minus_fresh"):
                em = pair["metrics"]["answer_em"]
                lines.append(
                    f"| {name} | {em['n']} | {em['mean']} | {em['repairs']} | {em['harms']} |"
                )
        lines += [
            "",
            "原始summary及inference保留共同分母、bootstrap区间、McNemar/Holm与失败成本。",
            "pilot结果不证明动作因果贡献、整体创新性或跨语料泛化。",
            "",
        ]
    return "\n".join(lines)


def sync_public(root, text):
    target = root / PUBLIC
    # Do not include already staged user changes in an automatic task commit.
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    )
    if staged.stdout.strip():
        raise ValueError("unrelated staged changes prevent automatic task commit")
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    if branch != "codex/dynamic-operators-v1":
        raise ValueError("unexpected branch; report saved but not committed")
    for command in (
        ["git", "add", "--", PUBLIC],
        [
            "git",
            "commit",
            "--only",
            "-m",
            "Record real v3 pipeline outcome and cost aggregates",
            "--",
            PUBLIC,
        ],
        ["git", "push", "origin", branch],
    ):
        subprocess.run(command, cwd=root, check=True, timeout=120)


def project_budget_stopped(root, args, old_runs):
    """A smaller per-batch cap is not automatically exhaustion of the PROJECT cap."""
    for directory in (root / "runs").glob("2026-09-30_operator_v3_*"):
        if directory.name in old_runs or not directory.is_dir():
            continue
        plan_path, budget_path = directory / "launch_plan.json", directory / "final_budget.json"
        if not plan_path.is_file() or not budget_path.is_file():
            continue
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        if plan.get("manifest_sha256") != args.expected_manifest_sha256:
            continue
        remaining = args.project_cap_cny - plan["prior_reserved_cny"]
        if budget.get("block_reason") == "estimated_budget_limit" and math.isclose(
            plan["subcap_cny"], remaining, abs_tol=1e-10
        ):
            return True
    return False


def request_shutdown(output, result):
    if not WINDOWS:
        raise ValueError("Windows-only non-forced shutdown")
    record(output / "shutdown_requested.json", {"status": result["status"], "forced": False})
    # /t > 0 implicitly forces applications closed on Windows; never use it.
    subprocess.run(["shutdown.exe", "/s", "/t", "0"], check=True)


def run(args, root):
    if not args.allow_network:
        print("Dry run only: " + " -> ".join(STAGES))
        print("No API, gold, output directory, Git mutation or shutdown.")
        return 0
    if not 0 <= args.source_start < 500 or not 0 < args.project_cap_cny <= 200:
        raise ValueError("invalid authorized source range or budget")
    if sha(inside(root, args.manifest)) != args.expected_manifest_sha256:
        raise ValueError("manifest changed")
    if action_list_execution_signature(root)["sha256"] != args.expected_execution_sha256:
        raise ValueError("method changed")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    )
    if dirty.stdout.strip():
        raise ValueError("commit code before starting paid pipeline")
    # Never overwrite partial prior outputs or silently reuse a stale certificate.
    if any((root / path).exists() for path in (FEEDBACK, BANKS, FREEZE, EVALUATION)):
        raise ValueError("pipeline products already exist; require audited manual continuation")
    output = inside(root, args.output_dir)
    if not output.is_relative_to(root / "runs"):
        raise ValueError("pipeline artifacts must stay inside runs")
    output.mkdir(parents=True, exist_ok=False)
    stop_file, quota_file = output / "stop_requested.flag", output / "quota_stop.flag"
    old_runs = {path.name for path in (root / "runs").iterdir() if path.is_dir()}
    result = {
        "status": "running",
        "stage": "preflight",
        "completed_stages": [],
        "output_dir": str(output),
        "method_sha256": args.expected_execution_sha256,
    }
    try:
        for stage in STAGES:
            result["stage"] = stage
            if stop_file.exists():
                result["status"] = "quota_stop" if quota_file.exists() else "technical_stop"
                break
            command = stage_command(args, root, stage, stop_file)
            code = execute(root, output, stage, command)
            if code != 0:
                result["status"] = (
                    "quota_stop" if code == 2 and quota_file.exists() else "technical_stop"
                )
                result["child_returncode"] = code
                break
            result["completed_stages"].append(stage)
        else:
            result["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="technical_stop", error_type=type(error).__name__)
    try:
        result["budget"] = budget_snapshot(root)
        if result["status"] != "completed" and (
            result["budget"]["prior_reserved_cny"] >= args.project_cap_cny
            or project_budget_stopped(root, args, old_runs)
        ):
            result["status"] = "quota_stop"
    except Exception as error:
        result["budget_audit_error"] = type(error).__name__
    result["utc"] = datetime.now(UTC).isoformat()
    record(output / "pipeline_result.json", result)
    text = public_report(root, result)
    with (output / "SUMMARY.md").open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if args.sync_github:
        try:
            sync_public(root, text)
            record(output / "sync_result.json", {"pushed": True})
        except Exception as error:
            record(
                output / "sync_result.json", {"pushed": False, "error_type": type(error).__name__}
            )
    shut = (result["status"] == "completed" and args.shutdown_on_complete) or (
        result["status"] == "quota_stop" and args.shutdown_on_quota_stop
    )
    if shut:
        request_shutdown(output, result)
    return 0 if result["status"] == "completed" else 1


def main(argv=None):
    args = parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        return run(args, root)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Pipeline stopped ({type(error).__name__}); saved artifacts remain unchanged.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
