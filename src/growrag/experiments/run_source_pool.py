"""Opt-in 64-source diagnostic with one 5 CNY budget and no target calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from .api_client import ChatConfig, LiveChatClient
from .api_preflight import read_local_bailian_settings
from .budget import BudgetedChatClient, PriceLimits
from .hotpot import HotpotExample, parse_hotpot_example
from .pre_pilot import EXTRACTION_VERSION, PRE_INTENT, write_json
from .run_pilot import PILOT_MODEL, _git_state
from .run_representation_manifest import DATASET_LABEL, EXPECTED_TRAIN_ROWS
from .source_pool_pilot import run_source_pool, validate_source_pool_inputs

SOURCE_COUNT = 64
MAX_CALLS = 256
MAX_OUTPUT_TOKENS = 768
BUDGET_CNY = 5.0
KEY_VARIABLE = "GROWRAG_SOURCE_POOL_API_KEY"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_source_pool_examples(
    manifest_path: Path, data_path: Path, *, allow_synthetic: bool = False
) -> tuple[tuple[HotpotExample, ...], dict, dict]:
    """Verify both files; instantiate gold records for the 64 source IDs only.

    The selected file also contains reserved examples. Their raw JSON is read
    to verify file identity/order, but their answers/support are never inspected
    or parsed into runtime/feedback records. No credentials or network is used.
    The checksum verifies consistency, not an official cryptographic signature.
    """
    manifest_path, data_path = Path(manifest_path).resolve(), Path(data_path).resolve()
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest, object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    synthetic = manifest.get("synthetic_data")
    if synthetic is not False and not (allow_synthetic is True and synthetic is True):
        raise ValueError("production source runs reject synthetic or unmarked data")
    if not synthetic and (
        manifest.get("expected_row_count") != EXPECTED_TRAIN_ROWS
        or manifest.get("input_record_count") != EXPECTED_TRAIN_ROWS
        or manifest.get("input_dataset_label") != DATASET_LABEL
        or manifest.get("data_version") != "HotpotQA v1.1 train"
    ):
        raise ValueError("production requires the complete HotpotQA v1.1 train declaration")
    if manifest.get("source_bytes_verified_by_builder") is not True:
        raise ValueError("source bytes were not verified by the manifest builder")
    filename = manifest.get("selected_records_file")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("selected_records_file must be a local filename")
    if data_path != (manifest_path.parent / filename).resolve():
        raise ValueError("--data must identify the manifest's selected records file")
    original_path = manifest.get("input_path")
    if not isinstance(original_path, str) or not Path(original_path).is_absolute():
        raise ValueError("manifest must identify the original full train JSON path")
    original_path = Path(original_path).resolve()
    if original_path.suffix.casefold() != ".json" or original_path == data_path:
        raise ValueError("original train must be a distinct JSON file")
    original_hash = _file_sha256(original_path)
    if original_hash != manifest.get("source_sha256"):
        raise ValueError("original train SHA-256 differs from the frozen manifest")
    raw_selected = data_path.read_bytes()
    selected_hash = hashlib.sha256(raw_selected).hexdigest()
    if selected_hash != manifest.get("selected_records_sha256"):
        raise ValueError("selected records SHA-256 differs from the frozen manifest")
    records = json.loads(raw_selected, object_pairs_hook=_unique_object)
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("selected data must be an array of record objects")
    if len(records) != manifest.get("selected_records_count"):
        raise ValueError("selected record count differs from manifest")
    selected = manifest.get("selected", {})
    source_ids, target_ids = selected.get("source"), selected.get("target")
    expansion = manifest.get("source_expansion_order")
    if not all(isinstance(value, list) for value in (source_ids, target_ids, expansion)):
        raise ValueError("manifest selection lists are missing")
    if len(source_ids) != SOURCE_COUNT:
        raise ValueError("this authorization is for exactly 64 sources, no automatic expansion")
    ids = [row.get("_id") for row in records]
    if any(not isinstance(qid, str) or not qid for qid in ids) or len(set(ids)) != len(ids):
        raise ValueError("selected records require unique nonempty IDs")
    if ids != expansion + target_ids:
        raise ValueError("selected file IDs/order differ from the frozen source and target lists")
    by_id = dict(zip(ids, records, strict=True))
    sources = tuple(
        parse_hotpot_example(by_id[qid], dataset=manifest["input_dataset_label"])
        for qid in source_ids
    )
    validate_source_pool_inputs(sources, manifest)
    verification = {
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "source_sha256": original_hash,
        "selected_records_sha256": selected_hash,
        "original_bytes_reverified": True,
        "selected_bytes_reverified": True,
        "source_ids_verified": True,
        "source_count": len(sources),
        "target_gold_parsed": False,
        "official_authenticity_independently_verified": False,
    }
    return sources, manifest, verification


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    sources, manifest, verified = load_source_pool_examples(args.manifest, args.data)
    if args.output.exists():
        raise FileExistsError("output already exists; no implicit resume or duplicate charges")
    limits = PriceLimits(
        budget_cny=BUDGET_CNY,
        input_per_million_cny=0.2,
        output_per_million_cny=0.8,
        max_prompt_bytes=24000,
        max_elapsed_seconds=1800,
    )
    git_state = _git_state()
    plan = {
        "schema_version": "growrag-source-pool-launch-v1",
        "git": git_state,
        "manifest": manifest,
        "data_verification": verified,
        "authorized_batch_budget_cny": BUDGET_CNY,
        "source_count": SOURCE_COUNT,
        "target_count": 0,
        "automatic_expansion": False,
        "model": PILOT_MODEL,
        "temperature": 0.0,
        "enable_thinking": False,
        "max_api_calls": MAX_CALLS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "limits": asdict(limits),
        "top_k": 4,
        "intent": PRE_INTENT,
        "card_extraction_prompt": EXTRACTION_VERSION,
        "pricing_verified_date": "2026-09-11",
        "pricing_source": "https://help.aliyun.com/zh/model-studio/model-pricing",
        "model_source": "https://help.aliyun.com/zh/model-studio/text-generation-model/",
        "data_scope": "Only 64 selected public train questions, retrieved candidate text, "
        "and generated procedures; source gold is offline only.",
        "notice": "One 5 CNY declared-price estimated ledger for the entire batch. "
        "No retries, resume, target calls, automatic source expansion or model switching.",
    }
    if not args.allow_network:
        # No credentials, new files, transport objects or API calls on dry run.
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.api_config is None:
        parser.error("--api-config is required for explicitly authorized live execution")
    if git_state["commit"] is None or git_state["worktree_dirty"]:
        raise ValueError("commit reviewed code and plans before live execution")
    try:
        settings = read_local_bailian_settings(args.api_config)
    except (OSError, ValueError):
        print("Local API settings unavailable or invalid; no request sent or secrets displayed.")
        return 2
    host = urlsplit(settings.base_url).hostname or ""
    if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
        raise ValueError("verified price profile permits only the Beijing endpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "launch_plan.json", plan)
    previous = os.environ.get(KEY_VARIABLE)
    client = None
    os.environ[KEY_VARIABLE] = settings.api_key
    try:
        client = BudgetedChatClient(
            LiveChatClient(
                ChatConfig(
                    settings.base_url,
                    PILOT_MODEL,
                    KEY_VARIABLE,
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
        result = run_source_pool(sources, client, args.output, manifest=manifest, allow_real=True)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "status": result.summary["status"],
                    "processed_source_count": result.summary["processed_source_count"],
                    "candidate_count": result.summary["candidate_count"],
                    "api_requests": client.attempts,
                    "estimated_cost_cny": client.report()["estimated_actual_cny"],
                    "budget_block_reason": client.block_reason,
                    "manual_pattern_review_required": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result.summary["status"] == "completed" and not client.block_reason else 1
    except Exception as error:
        write_json(
            args.output / "interrupted.json",
            {
                "status": "interrupted",
                "error_type": type(error).__name__,
                "notice": "No automatic retry or resume; inspect retained source and API audits.",
            },
        )
        print("Source pool interrupted; retained records are audit-only, no automatic retry.")
        return 1
    finally:
        if previous is None:
            os.environ.pop(KEY_VARIABLE, None)
        else:
            os.environ[KEY_VARIABLE] = previous
        if client is not None and not (args.output / "final_budget.json").exists():
            write_json(args.output / "final_budget.json", client.report())


if __name__ == "__main__":
    raise SystemExit(main())
