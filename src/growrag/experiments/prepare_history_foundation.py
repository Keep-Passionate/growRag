"""离线建立ReFormeR参考规则＋旧train来源模板的历史基底，不发API。

只读作者数据、来源库、ID manifest和审计bundle，不打开题目正文/gold/评价轨迹。
所有输出写新目录，旧库、原始作者模式、旧冻结文件一律不覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from growrag.history_library import (
    FrozenHistoryLibrary,
    _load,
    combine_libraries,
    import_operator_bank,
    import_reformer_patterns,
)
from growrag.operator_bank import FrozenOperatorBank

from .pre_pilot import write_json
from .reformer_api import PINNED_HASHES, UPSTREAM_COMMIT


def _read(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def prepare_foundation(
    *,
    reformer_snapshot: Path,
    bank_path: Path,
    manifest_path: Path,
    bundle_path: Path,
    output: Path,
    expected_manifest_sha256: str,
    expected_bundle_sha256: str,
) -> dict:
    """只有已与官方train角色和既有建库bundle核对的来源卡才能迁移。"""
    pattern_path = reformer_snapshot / "reformer/patterns/extracted_patterns.json"
    pattern_raw, pattern_sha = _read(pattern_path)
    bank_raw, bank_sha = _read(bank_path)
    manifest_raw, manifest_sha = _read(manifest_path)
    bundle_raw, bundle_sha = _read(bundle_path)
    if manifest_sha != expected_manifest_sha256 or bundle_sha != expected_bundle_sha256:
        raise ValueError("source manifest/bundle differs from the externally pinned SHA256")
    manifest, bundle = _load(manifest_raw), _load(bundle_raw)
    bank = FrozenOperatorBank.from_json(bank_raw.decode("utf-8"))
    if manifest["official_splits"]["source"] != "train":
        raise ValueError("only the official train source role may build history")
    if (
        bundle["schema_version"] != "growrag-operator-bank-bundle-v1"
        or bundle["phase"] != "source"
        or bundle["manifest_sha256"] != manifest_sha
    ):
        raise ValueError("source bundle does not bind this manifest")
    source = set(manifest["roles"]["source"])
    protected = set(manifest["roles"]["calibration"]) | set(manifest["roles"]["evaluation"])
    if not set(bank.allowed_source_ids) <= source or set(bank.allowed_source_ids) & protected:
        raise ValueError("bank source scope overlaps protected questions or escapes train")
    expected_prefix = manifest["nested_source_ids"][str(len(bank.allowed_source_ids))]
    if set(expected_prefix) != set(bank.allowed_source_ids):
        raise ValueError("bank is not the declared source prefix")
    declaration = bundle["banks"][str(len(bank.allowed_source_ids))]
    if (
        declaration["bank_fingerprint"] != bank.fingerprint
        or declaration["sha256"] != bank_sha
        or bank.protocol_id != bundle["protocol"]
    ):
        raise ValueError("source bank fingerprint/protocol differs from audited bundle")
    reference = import_reformer_patterns(
        pattern_raw,
        PINNED_HASHES["reformer/patterns/extracted_patterns.json"],
        f"ReFormeR:{UPSTREAM_COMMIT}:reformer/patterns/extracted_patterns.json",
    )
    learned = import_operator_bank(bank, str(bank_path.resolve()), bank_sha)
    combined = combine_libraries(reference, learned)
    products = {"reference_rules": reference, "source_templates": learned, "combined": combined}
    summary = {
        "schema_version": "growrag-history-foundation-report-v1",
        "api_calls": 0,
        "new_gold_loaded": False,
        "evaluation_questions_or_predictions_loaded": False,
        "model_training": False,
        "new_operators_induced": False,
        "runtime_connected_to_live_api": False,
        "old_bank_unchanged": True,
        "legacy_bank_fingerprint": bank.fingerprint,
        "source_question_count": len(bank.allowed_source_ids),
        "inputs": {
            "reformer_patterns": {"path": str(pattern_path.resolve()), "sha256": pattern_sha},
            "operator_bank": {"path": str(bank_path.resolve()), "sha256": bank_sha},
            "manifest": {"path": str(manifest_path.resolve()), "sha256": manifest_sha},
            "source_bundle": {"path": str(bundle_path.resolve()), "sha256": bundle_sha},
        },
        "libraries": {
            name: {
                "records": len(library.records),
                "published": len(library.published_cards),
                "rewrite_rules": sum(
                    c.action_kind == "rewrite_rule" for c in library.published_cards
                ),
                "templates": sum(c.action_kind == "template" for c in library.published_cards),
                "explicit_condition_cards": sum(
                    bool(c.conditions) for c in library.published_cards
                ),
                "fingerprint": library.fingerprint,
            }
            for name, library in products.items()
        },
        "notices": [
            "Published is experimental eligibility, not learned reliability or transfer proof.",
            "Imported reference examples retain upstream wording; "
            "they are not certified successes.",
            "Source-only deterministic descriptions add no inferred applicability conditions.",
            "Combining libraries is an engineering foundation, not evidence of research novelty.",
            "The prior dev500 has been analyzed: it is diagnostic, not a new untouched test set.",
        ],
    }
    # 所有检查先完成；存在的目录整批拒绝，绝不覆盖旧结果。
    output.mkdir(parents=True, exist_ok=False)
    for name, library in products.items():
        encoded = library.to_json()
        if FrozenHistoryLibrary.from_json(encoded) != library:
            raise ValueError("history round trip changed the library")
        write_json(output / f"{name}.json", json.loads(encoded))
    write_json(output / "SUMMARY.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reformer-snapshot", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_foundation(
        reformer_snapshot=args.reformer_snapshot,
        bank_path=args.bank,
        manifest_path=args.manifest,
        bundle_path=args.source_bundle,
        output=args.output,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_bundle_sha256=args.expected_source_bundle_sha256,
    )
    print(
        json.dumps(
            {"api_calls": 0, "libraries": result["libraries"], "output": str(args.output)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
