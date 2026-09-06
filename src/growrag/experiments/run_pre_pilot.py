"""Explicit 32-source/16-target PRE pilot; one shared 5 CNY estimated budget."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import BudgetedChatClient, PriceLimits
from .pre_manifest import load_pre_examples
from .pre_pilot import EXTRACTION_VERSION, PRE_INTENT, run_pre_batch, write_json
from .pre_report import render_batch, summarize_batch
from .run_pilot import PILOT_MODEL, _git_state

MAX_CALLS = 256
MAX_OUTPUT_TOKENS = 768
BUDGET_CNY = 5.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--source-count", type=int, default=32)
    parser.add_argument("--target-count", type=int, default=16)
    args = parser.parse_args(argv)
    sources, targets, manifest = load_pre_examples(
        args.manifest, args.source_count, args.target_count
    )
    limits = PriceLimits(budget_cny=BUDGET_CNY, max_elapsed_seconds=1800.0)
    git_state = _git_state()
    plan = {
        "schema_version": "growrag-pre-launch-v1",
        "git": git_state,
        "manifest": manifest,
        "model": PILOT_MODEL,
        "temperature": 0.0,
        "enable_thinking": False,
        "max_api_calls": MAX_CALLS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "limits": asdict(limits),
        "top_k": 4,
        "intent": PRE_INTENT,
        "card_extraction_prompt": EXTRACTION_VERSION,
        "pricing_verified_date": "2026-09-06",
        "pricing_source": "https://help.aliyun.com/zh/model-studio/model-pricing",
        "model_source": "https://help.aliyun.com/zh/model-studio/text-generation-model/",
        "data_scope": "selected public Hotpot train questions, candidate text "
        "and generated source procedures only",
        "notice": "5 CNY applies to the WHOLE batch. No retries, automatic resume, "
        "model switch or purchased subscription.",
    }
    if args.output.exists():
        raise FileExistsError("output already exists; no implicit resume or duplicate charges")
    if not args.allow_network:
        # Dry plan neither reads the secret file nor creates an output directory.
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.api_config is None:
        parser.error("--api-config is required for explicitly authorized live execution")
    if git_state["commit"] is None or git_state["worktree_dirty"]:
        raise ValueError("commit reviewed code and plans before live execution")
    try:
        settings = read_local_bailian_settings(args.api_config)
    except (OSError, ValueError):
        print("Local API settings invalid or unavailable; no request sent. No secrets displayed.")
        return 2
    host = urlsplit(settings.base_url).hostname or ""
    if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
        raise ValueError("the verified price profile permits only the original Beijing endpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "launch_plan.json", plan)
    variable = "GROWRAG_PRE_PILOT_API_KEY"
    previous = os.environ.get(variable)
    client = BudgetedChatClient(
        LiveChatClient(
            ChatConfig(
                settings.base_url,
                PILOT_MODEL,
                variable,
                max_calls=MAX_CALLS,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                timeout_seconds=45,
                output_limit_parameter="max_tokens",
                enable_thinking=False,
                temperature=0.0,
            ),
            args.output / "api_audit",
            allow_network=True,
        ),
        limits,
    )
    os.environ[variable] = settings.api_key
    try:
        result = run_pre_batch(
            sources, targets, client, args.output, manifest=manifest, allow_real=True
        )
        write_json(args.output / "aggregate.json", summarize_batch(result))
        with (args.output / "report.md").open("x", encoding="utf-8") as handle:
            handle.write(render_batch(result))
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "candidate_count": result["candidate_count"],
                    "api_requests": client.attempts,
                    "estimated_cost_cny": client.report()["estimated_actual_cny"],
                    "budget_block_reason": client.block_reason,
                },
                ensure_ascii=False,
            )
        )
        return 0 if not client.block_reason else 1
    except Exception as error:
        # Preserve type, not arbitrary exception bodies/keys/server responses.
        write_json(
            args.output / "interrupted.json",
            {
                "status": "interrupted",
                "error_type": type(error).__name__,
                "notice": "Do not automatically rerun or resend unknown calls.",
            },
        )
        print(
            "Batch interrupted. Prior branch files and API audits are retained; no automatic retry."
        )
        return 1
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


if __name__ == "__main__":
    raise SystemExit(main())
