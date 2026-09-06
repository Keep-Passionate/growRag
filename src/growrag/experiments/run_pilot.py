"""Explicit paid entry point for the frozen 8-source/8-target development pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import BudgetedChatClient, PriceLimits
from .data_protocol import DATASET_ID, SPLIT_VERSION, build_manifest
from .hotpot import load_hotpot

PILOT_MODEL = "qwen3.7-flash-2026-07-15"


def load_selected_examples(manifest_path: Path):
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "growrag-hotpot-manifest-v1"
        or manifest.get("dataset") != DATASET_ID
        or manifest.get("official_split") != "train"
        or manifest.get("configuration") != "distractor"
        or manifest.get("split_version") != SPLIT_VERSION
    ):
        raise ValueError("expected frozen Hotpot train development manifest")
    data_path = (manifest_path.parent / manifest["data_file"]).resolve()
    if data_path.parent != manifest_path.parent:
        raise ValueError("manifest data file must be in the same directory")
    raw = data_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["data_sha256"]:
        raise ValueError("dataset checksum changed; do not silently rerun on different data")
    sources = manifest["selected"]["memory_seed"]
    targets = manifest["selected"]["calibration_dev"]
    if len(sources) != 8 or len(targets) != 8:
        raise ValueError("this first paid pilot is capped and frozen at 8+8 questions")
    expected = build_manifest(json.loads(raw), seed=manifest["seed"])
    for key in ("roles", "selected", "role_counts", "record_count", "role_weights"):
        if manifest[key] != expected[key]:
            raise ValueError("manifest roles or selection do not match the frozen split algorithm")
    all_examples = load_hotpot(data_path, dataset="hotpotqa-distractor-train-preview-200")
    by_id = {example.question.question_id: example for example in all_examples}
    return (
        tuple(by_id[qid] for qid in sources),
        tuple(by_id[qid] for qid in targets),
        manifest,
    )


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
        return {"commit": commit, "worktree_dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "worktree_dirty": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--api-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    if not args.allow_network:
        parser.error("requires explicit --allow-network for an authorized small API experiment")
    # Complete all local validation before creating a client or exposing the key.
    sources, targets, manifest = load_selected_examples(args.manifest)
    settings = read_local_bailian_settings(args.api_config)
    hostname = urlsplit(settings.base_url).hostname or ""
    if hostname != "dashscope.aliyuncs.com" and not hostname.endswith(
        ".cn-beijing.maas.aliyuncs.com"
    ):
        raise ValueError("this price profile is Beijing-only; no API request was sent")
    args.output.mkdir(parents=True, exist_ok=False)
    git_state = _git_state()
    limits = PriceLimits()
    plan = {
        "schema_version": "growrag-paid-development-pilot-v1",
        "git": git_state,
        "manifest": manifest,
        "model": PILOT_MODEL,
        "enable_thinking": False,
        "max_api_calls": 96,
        "max_output_tokens": 768,
        "price_limits": asdict(limits),
        "initial_top_k": 4,
        "repair_top_k": 4,
        "notice": "train-only convenience development sample, not an official benchmark result",
    }
    (args.output / "launch_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    variable = "GROWRAG_PILOT_API_KEY"
    previous = os.environ.get(variable)
    os.environ[variable] = settings.api_key
    client = BudgetedChatClient(
        LiveChatClient(
            ChatConfig(
                settings.base_url,
                PILOT_MODEL,
                variable,
                max_calls=96,
                max_output_tokens=768,
                timeout_seconds=45,
                output_limit_parameter="max_tokens",
                enable_thinking=False,
            ),
            args.output / "api_audit",
            allow_network=True,
        ),
        limits,
    )
    exit_code, failure_type = 0, None
    try:
        from .pilot_engine import run_pilot

        run_pilot(sources, targets, client, args.output, initial_top_k=4, repair_top_k=4)
        if client.block_reason:
            exit_code, failure_type = 1, "BudgetOrAPIStopped"
    except Exception as error:
        # No raw exception text: provider errors can contain private request data.
        exit_code, failure_type = 1, type(error).__name__
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
        budget_report = client.report()
        budget_report.update(process_exit_code=exit_code, failure_type=failure_type)
        (args.output / "budget_report.json").write_text(
            json.dumps(budget_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "api_requests": client.attempts,
                    "estimated_cny": budget_report["estimated_actual_cny"],
                    "blocked": client.block_reason,
                    "failure_type": failure_type,
                    "output": str(args.output.resolve()),
                }
            ),
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
