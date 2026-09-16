"""Freeze a prompt ablation plan without a key, network client or paid execution.

中文：这里只定名单和对照，不读取API配置、不续跑失败实验、不训练模型。
BASE用于观察改坏；两个关键词组只比较是否显式提醒保留原问题约束。
该名单属于训练内部的开发题，不能在调prompt后再宣称它是独立测试。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .fresh_benchmark import validate_variants
from .pre_pilot import write_json
from .prompt_registry import prompt_identity
from .run_fresh_benchmark import load_benchmark
from .run_pilot import PILOT_MODEL, _git_state

PROMPT_ARMS = ("BASE", "RRR_MINIMAL", "RRR_KEYWORDS")


def build_plan(data: dict, *, variants: tuple[str, ...] = PROMPT_ARMS) -> dict:
    """Accept only a train diagnostic manifest; no labels or observed scores."""
    variants = validate_variants(variants)
    if variants != PROMPT_ARMS:
        raise ValueError("this frozen prompt contrast requires exactly PROMPT_ARMS")
    if (
        data.get("official_split") != "train"
        or data.get("role") != "selector_train"
        or data.get("official_dev_test_used") is not False
        or data.get("reserved_representation_targets_used") is not False
    ):
        raise ValueError("prompt development must not consume final/representation test data")
    ids = data.get("question_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(q, str) or not q for q in ids):
        raise ValueError("nonempty question IDs required")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question IDs")
    # Rebuild a small allowlist: never copy arbitrary fields or gold into a plan.
    selected_data = {
        key: data.get(key)
        for key in (
            "official_split",
            "role",
            "selection_seed",
            "source_sha256",
            "prior_manifest_sha256",
            "question_ids",
            "question_type_counts",
            "reserved_representation_targets_used",
            "official_dev_test_used",
        )
    }
    identity = {name: prompt_identity(name) for name in variants if name != "BASE"}
    payload = {
        "schema_version": "growrag-prompt-comparison-plan-v1",
        "status": "planned_not_executed",
        "data": selected_data,
        "variants": variants,
        "prompt_identity": identity,
        "reader_identity": prompt_identity("READER"),
        "controls": {
            "model": PILOT_MODEL,
            "temperature": 0,
            "enable_thinking": False,
            "top_k": 4,
            "index": "per-question candidate context; title-sentence BM25",
            "max_rag_calls_per_arm": 1,
            "memory_enabled": False,
            "reader_original_question": True,
            "max_output_tokens": 768,
        },
        "primary_contrast": {
            "left": "RRR_MINIMAL",
            "right": "RRR_KEYWORDS",
            "factor": "explicit intent/entity/relation/constraint preservation instructions",
            "claim": "prompt wording ablation, not a complete paper-method reproduction",
        },
        "secondary_contrast_not_in_this_plan": {
            "left": "RRR_KEYWORDS",
            "right": "RRR_QUERY_ANCHOR",
            "factor": "prepend original query programmatically; identical system prompt",
            "notice": "composition ablation, not a different-prompt result",
        },
        "evaluation": {
            "measures": [
                "answer_em",
                "answer_f1",
                "support_recall",
                "rescues_vs_base",
                "harms_vs_base",
                "format_failures",
                "tokens",
                "cost",
                "wall_time",
            ],
            "pairing": "same-question complete-case cohort; report all missing/failed arms",
            "selection_rule": "choose on internal development only; freeze before held-out check",
            "no_gold_runtime": True,
            "gold_support_not_sufficiency_proof": True,
            "no_significance_claim_from_small_sample": True,
        },
        "planned_max_api_calls": len(ids) * sum(1 if n == "BASE" else 2 for n in variants),
        "actual_api_calls": 0,
        "model_training_performed": False,
        "run_authorized_by_this_file": False,
        "budget_gate": "Prior 400/unknown usage retained; "
        "this planner cannot resume or reset funds.",
        "notice": "Previously exposed developer questions are not an independent final test.",
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {**payload, "plan_sha256": hashlib.sha256(encoded).hexdigest()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("use a new plan directory; preserve old specifications")
    _, data = load_benchmark(args.manifest, count=args.count)
    plan = build_plan(data)
    plan["git"] = _git_state()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    print(
        json.dumps(
            {
                "plan": str(args.output / "plan.json"),
                "questions": args.count,
                "actual_api_calls": 0,
                "status": "planned_not_executed",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
