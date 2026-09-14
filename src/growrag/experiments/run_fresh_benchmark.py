"""Explicit small paid FRESH comparison on NEW selector_train questions only.

No parameter training, official dev/test, historical memory, retries or resume.
The previous batch ledger is carried conservatively with an exclusive claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import BudgetedChatClient, PriceLimits
from .data_protocol import normalize_question, role_for_question
from .fresh_baselines import BASELINE_SPECS
from .fresh_benchmark import VARIANTS, run_fresh_benchmark
from .hotpot import parse_hotpot_example
from .pre_pilot import write_json
from .run_pilot import PILOT_MODEL, _git_state
from .run_representation_manifest import DATASET_LABEL, EXPECTED_TRAIN_ROWS

KEY_VARIABLE = "GROWRAG_FRESH_BENCHMARK_API_KEY"


def select_records(records: list[dict], manifest: dict, *, count: int = 32) -> list[dict]:
    """Select by question text/ID/type only; never examine an outcome or gold."""
    if type(count) is not int or count < 2 or count > 32 or count % 2:
        raise ValueError("this diagnostic permits a positive even count up to 32")
    ids = [r["_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate IDs in source data")
    counts = Counter(normalize_question(r["question"]) for r in records)
    excluded = set(manifest["exclusions"]["excluded_union_question_ids"])
    excluded.update(manifest["source_expansion_order"])
    excluded.update(manifest["selected"]["target"])
    pools = {kind: [] for kind in ("bridge", "comparison")}
    for record in records:
        if (
            record["_id"] in excluded
            or counts[normalize_question(record["question"])] != 1
            or role_for_question(record["question"]) != "selector_train"
        ):
            continue
        if record.get("type") in pools:
            pools[record["type"]].append(record)
    for pool in pools.values():
        pool.sort(
            key=lambda r: hashlib.sha256(f"fresh-benchmark-v1:42:{r['_id']}".encode()).hexdigest()
        )
        if len(pool) < count // 2:
            raise ValueError("not enough unique questions in both strata")
    return [
        row
        for pair in zip(
            pools["bridge"][: count // 2], pools["comparison"][: count // 2], strict=True
        )
        for row in pair
    ]


def load_benchmark(manifest_path: Path, *, count: int = 32):
    raw_manifest = Path(manifest_path).read_bytes()
    manifest = json.loads(raw_manifest)
    if (
        manifest.get("schema_version") != "growrag-representation-manifest-v1"
        or manifest.get("official_split") != "train"
        or manifest.get("synthetic_data") is not False
        or manifest.get("input_record_count") != EXPECTED_TRAIN_ROWS
        or manifest.get("source_bytes_verified_by_builder") is not True
    ):
        raise ValueError("expected previously verified complete train manifest")
    raw = Path(manifest["input_path"]).read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    if source_hash != manifest["source_sha256"]:
        raise ValueError("full train bytes changed")
    records = json.loads(raw)
    if len(records) != EXPECTED_TRAIN_ROWS:
        raise ValueError("full train record count changed")
    selected = select_records(records, manifest, count=count)
    # Only selected training examples are parsed into GoldRecord objects.
    examples = tuple(parse_hotpot_example(r, dataset=DATASET_LABEL) for r in selected)
    plan = {
        "schema_version": "growrag-fresh-benchmark-manifest-v1",
        "prior_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "source_sha256": source_hash,
        "official_split": "train",
        "role": "selector_train",
        "selection_seed": 42,
        "question_ids": [e.question.question_id for e in examples],
        "question_type_counts": dict(Counter(e.question_type for e in examples)),
        "reserved_representation_targets_used": False,
        "official_dev_test_used": False,
        "memory_source_questions_used": False,
        "semantic_near_duplicates_checked": False,
        "notice": "Outcome-independent balanced training diagnostic; not a blind test. "
        "Original 8 debug/24 check representation targets remain reserved.",
    }
    return examples, plan


def carryover_budget(path: Path) -> dict:
    """Fail closed on incomplete/unknown prior billing; never reset 5 CNY."""
    raw = Path(path).read_bytes()
    prior = json.loads(raw)
    if (
        prior.get("block_reason") is not None
        or not prior.get("calls")
        or prior.get("limits", {}).get("budget_cny") != 5.0
    ):
        raise ValueError("expected the completed authorized 5 CNY source ledger")
    prices = prior["limits"]
    if prices.get("input_per_million_cny") != 0.2 or prices.get("output_per_million_cny") != 0.8:
        raise ValueError("carryover price profile differs")
    for call in prior["calls"]:
        if (
            call.get("status") != "completed"
            or call.get("returned_model") != PILOT_MODEL
            or call.get("validation_status") is not None
        ):
            raise ValueError("prior calls incomplete or model drifted")
        for field in ("input_tokens", "output_tokens", "reserved_cny", "estimated_actual_cny"):
            value = call.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("unknown prior accounting")
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid prior accounting")
    for field in ("input_tokens", "output_tokens", "reserved_cny", "estimated_actual_cny"):
        if not math.isclose(
            sum(c[field] for c in prior["calls"]), prior.get(field, -1), rel_tol=1e-9, abs_tol=1e-10
        ):
            raise ValueError("prior ledger totals inconsistent")
    if prior.get("api_requests") != len(prior["calls"]):
        raise ValueError("prior request count inconsistent")
    reserved = prior["reserved_cny"]
    if not prior["estimated_actual_cny"] <= reserved < 5:
        raise ValueError("no conservative budget remains")
    return {
        "parent_path": str(Path(path).resolve()),
        "parent_sha256": hashlib.sha256(raw).hexdigest(),
        "prior_estimated_actual_cny": prior["estimated_actual_cny"],
        "prior_reserved_cny": reserved,
        "authorized_total_cny": 5.0,
        "new_run_subcap_cny": min(1.0, 5.0 - reserved),
    }


def claim_continuation(parent_path: Path, output: Path, carry: dict) -> Path:
    """One child only. A crash retains claim; no silent re-launch against old funds."""
    if hashlib.sha256(Path(parent_path).read_bytes()).hexdigest() != carry["parent_sha256"]:
        raise ValueError("parent budget changed before claim")
    claim = Path(parent_path).parent.with_suffix(".budget-continuation.json")
    with claim.open("x", encoding="utf-8") as handle:
        json.dump(
            {**carry, "child_output": str(Path(output).resolve()), "no_automatic_resume": True},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return claim


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prior-budget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("new output directory required; no retry/resume")
    carry = carryover_budget(args.prior_budget)
    examples, data_plan = load_benchmark(args.manifest)
    limits = PriceLimits(budget_cny=carry["new_run_subcap_cny"], max_elapsed_seconds=1800)
    plan = {
        "data": data_plan,
        "carryover": carry,
        "git": _git_state(),
        "variants": {k: asdict(v) for k, v in BASELINE_SPECS.items()},
        "all_arms": VARIANTS,
        "limits": asdict(limits),
        "model": PILOT_MODEL,
        "temperature": 0,
        "thinking": False,
        "max_output_tokens": 768,
        "max_api_calls": 224,
        "top_k": 4,
        "pricing_checked_date": "2026-09-14",
        "pricing_source": "https://help.aliyun.com/zh/model-studio/model-pricing",
        "source_scope": "Only public train questions and their candidate evidence; "
        "gold is offline-only. No personal Obsidian notes are uploaded.",
    }
    if not args.allow_network:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.api_config is None:
        parser.error("--api-config required for authorized live execution")
    if not plan["git"]["commit"] or plan["git"]["worktree_dirty"]:
        raise ValueError("review and commit code/plan before live execution")
    try:
        settings = read_local_bailian_settings(args.api_config)
    except (OSError, ValueError):
        print("Local API configuration invalid; no request sent or secrets shown.")
        return 2
    host = urlsplit(settings.base_url).hostname or ""
    if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
        raise ValueError("declared price permits only Beijing endpoint")
    claim_continuation(args.prior_budget, args.output, carry)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "launch_plan.json", plan)
    previous = os.environ.get(KEY_VARIABLE)
    os.environ[KEY_VARIABLE] = settings.api_key
    client = None
    try:
        client = BudgetedChatClient(
            LiveChatClient(
                ChatConfig(
                    settings.base_url,
                    PILOT_MODEL,
                    KEY_VARIABLE,
                    max_calls=224,
                    max_output_tokens=768,
                    timeout_seconds=45,
                    output_limit_parameter="max_tokens",
                    enable_thinking=False,
                    temperature=0,
                ),
                args.output / "api_audit",
                allow_network=True,
            ),
            limits,
        )
        report = run_fresh_benchmark(examples, client, args.output, allow_real=True)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except Exception as error:
        write_json(
            args.output / "interrupted.json",
            {
                "error_type": type(error).__name__,
                "notice": "No automatic retry; inspect audits, do not reopen 5 CNY.",
            },
        )
        print("FRESH comparison interrupted; audit retained, no automatic retry.")
        return 1
    finally:
        if previous is None:
            os.environ.pop(KEY_VARIABLE, None)
        else:
            os.environ[KEY_VARIABLE] = previous
        if client is not None:
            report = client.report()
            write_json(args.output / "final_budget.json", report)
            actual = report["estimated_actual_cny"]
            write_json(
                args.output / "cumulative_budget.json",
                {
                    **carry,
                    "child_estimated_actual_cny": actual,
                    "cumulative_estimated_actual_cny": (
                        carry["prior_estimated_actual_cny"] + actual if actual is not None else None
                    ),
                    "cumulative_reserved_cny": carry["prior_reserved_cny"] + report["reserved_cny"],
                    "billing_notice": "Declared-price estimate, not provider invoice.",
                },
            )


if __name__ == "__main__":
    raise SystemExit(main())
