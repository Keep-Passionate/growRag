"""Build four independent frozen source-prefix banks from audited scalar feedback.

中文：只消费已评分来源，不打开新标签、不调用模型。每个前缀单独提炼，不能先
用500题聚合后截断成50题库。空库合法且应回退FRESH，不能补抽题目凑卡片。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from growrag.operator_bank import FrozenOperatorBank

from .fresh_dev_manifest import _sha
from .operator_data_plan import SOURCE_SIZES
from .operator_profiles import (
    ACTION_LIST,
    LEGACY,
    study_profile,
    validate_profile_execution_signature,
)
from .operator_source_bank import METRICS, build_source_bank
from .operator_source_bank import SOURCE_PROTOCOL as SOURCE_PROTOCOL
from .pre_pilot import write_json
from .representation_runner import fingerprint
from .run_operator_study import load_inputs
from .score_operator_sources import SCHEMA as FEEDBACK_SCHEMA
from .score_operator_sources import collect_frozen_sources

SCHEMA = "growrag-operator-bank-bundle-v1"
_ARMS = {"base", "fresh", "static"}
_SHA = re.compile(r"[0-9a-f]{64}")


def _read(path: Path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key in frozen input")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"nonstandard JSON constant: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject
    )


def _inside(root: Path, path: Path) -> Path:
    path = (root / path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("frozen input must remain inside project")
    return path


def _validate_feedback(feedback: dict, source_ids: list[str]) -> None:
    if type(feedback) is not dict or set(feedback) != set(source_ids):
        raise ValueError("feedback must cover exactly the frozen source IDs")
    for arms in feedback.values():
        if type(arms) is not dict or set(arms) != _ARMS:
            raise ValueError("feedback must contain exactly BASE/FRESH/STATIC")
        for values in arms.values():
            if type(values) is not dict or set(values) != set(METRICS):
                raise ValueError("feedback must contain exactly three scalar metrics")
            for key, value in values.items():
                if value is not None and (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                    or (key == "answer_em" and value not in (0, 1))
                ):
                    raise ValueError("invalid scalar source feedback")


def _verify_profile_predictions(root, manifest, manifest_sha, scoring, reports, selected):
    """Reconstruct the v3 source handoff from original label-free terminal files."""
    inputs = scoring["prediction_inputs"]
    paths = [_inside(root, Path(entry["path"])) for entry in inputs]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate frozen prediction audit input")
    launches = [path for path in paths if path.name == "launch_plan.json"]
    runs_roots = {path.parent.parent for path in launches}
    if len(runs_roots) != 1:
        raise ValueError("source scoring must retain launches from one runs root")
    recovered, model, audited = collect_frozen_sources(
        manifest, manifest_sha, runs_roots.pop(), profile=selected.name
    )
    expected = {str(path): entry["sha256"] for path, entry in zip(paths, inputs, strict=True)}
    actual = {entry["path"]: entry["sha256"] for entry in audited}
    signatures = {
        validate_profile_execution_signature(_read(path).get("execution_signature"), selected)
        for path in launches
    }
    if (
        recovered != reports
        or model != scoring["model"]
        or expected != actual
        or scoring.get("profile") != selected.name
        or signatures != {scoring.get("execution_signature_sha256")}
    ):
        raise ValueError("source scoring differs from its frozen profile/method/predictions")


def build_operator_banks(
    root: Path,
    manifest_path: Path,
    feedback_dir: Path,
    output_dir: Path,
    *,
    expected_audit_sha256: str,
    write: bool = False,
    profile=LEGACY.name,
) -> dict:
    """Verify and plan by default. write=True publishes only a new frozen bundle.

    expected_audit_sha256 is an explicit handoff fingerprint, not an authenticity
    proof. The caller must obtain it from the completed offline scoring operation.
    No evaluation authorization certificate is issued by this builder.
    """
    selected = study_profile(profile)
    root = Path(root).resolve(strict=True)
    manifest_path = _inside(root, manifest_path)
    feedback_dir = (root / feedback_dir).resolve(strict=True)
    output_dir = (root / output_dir).resolve()
    if (
        not feedback_dir.is_relative_to(root)
        or not feedback_dir.is_dir()
        or not output_dir.is_relative_to(root)
        or output_dir in {root, feedback_dir}
    ):
        raise ValueError("feedback/output must remain inside project, never replace roots")
    if write and output_dir.exists():
        raise FileExistsError("bank output exists; never overwrite frozen or partial banks")
    if not isinstance(expected_audit_sha256, str) or not _SHA.fullmatch(expected_audit_sha256):
        raise ValueError("explicit scoring audit SHA256 is required")
    manifest, _ = load_inputs(manifest_path, "source")
    manifest_sha = _sha(manifest_path)
    source_ids = manifest["roles"]["source"]
    audit_path = _inside(root, feedback_dir / "audit.json")
    if _sha(audit_path) != expected_audit_sha256:
        raise ValueError("scoring audit SHA mismatch")
    scoring = _read(audit_path)
    if (
        scoring.get("schema_version") != FEEDBACK_SCHEMA
        or scoring.get("phase") != "source"
        or scoring.get("protocol") != selected.protocol
        or scoring.get("manifest_sha256") != manifest_sha
        or scoring.get("source_ids") != source_ids
        or scoring.get("source_count") != len(source_ids)
        or scoring.get("gold_loaded") is not True
        or scoring.get("calibration_evaluation_gold_loaded") is not False
        or scoring.get("raw_predictions_modified") is not False
        or not isinstance(scoring.get("model"), str)
        or not scoring["model"].strip()
    ):
        raise ValueError("scoring audit is not the complete approved source phase")
    input_files = [
        {"path": str(manifest_path), "sha256": manifest_sha},
        {"path": str(audit_path), "sha256": expected_audit_sha256},
    ]
    loaded = {}
    for name, content_hash_key in (
        ("feedback.json", "feedback_sha256"),
        ("source_reports.json", "source_reports_sha256"),
    ):
        path = _inside(root, feedback_dir / name)
        if path.parent != feedback_dir or _sha(path) != scoring["artifacts"][name]["sha256"]:
            raise ValueError("scored artifact SHA mismatch")
        loaded[name] = _read(path)
        if fingerprint(loaded[name]) != scoring.get(content_hash_key):
            raise ValueError("scored artifact content fingerprint mismatch")
        input_files.append({"path": str(path), "sha256": _sha(path)})
    # Prediction inputs are label-free files. Gold source paths are deliberately
    # NOT opened again here; their scoring provenance stays inside audit.json.
    prediction_inputs = scoring.get("prediction_inputs")
    if type(prediction_inputs) is not list or not prediction_inputs:
        raise ValueError("scoring audit must retain frozen prediction inputs")
    for entry in prediction_inputs:
        path = _inside(root, Path(entry["path"]))
        if path.name not in {"launch_plan.json", "predictions.json", "predictions_frozen.json"}:
            raise ValueError("unexpected file in prediction-only audit inputs")
        if _sha(path) != entry["sha256"]:
            raise ValueError("raw prediction changed since source scoring")
        input_files.append({"path": str(path), "sha256": entry["sha256"]})
    reports, feedback = loaded["source_reports.json"], loaded["feedback.json"]
    if (
        type(reports) is not list
        or any(type(item) is not dict for item in reports)
        or [item.get("question_id") for item in reports] != source_ids
    ):
        raise ValueError("source report IDs/order must exactly match source500")
    if any(
        item.get("phase") != "source" or item.get("protocol") != selected.protocol
        for item in reports
    ):
        raise ValueError("calibration/evaluation/foreign reports cannot build memory")
    _validate_feedback(feedback, source_ids)
    if selected != LEGACY:
        _verify_profile_predictions(root, manifest, manifest_sha, scoring, reports, selected)

    products, summaries = {}, {}
    for size in SOURCE_SIZES:
        prefix = tuple(manifest["nested_source_ids"][str(size)])
        if prefix != tuple(source_ids[:size]) or len(prefix) != size:
            raise ValueError("nested source prefix mismatch")
        # Pass only this prefix, even though the extractor also enforces the scope.
        bank, audit = build_source_bank(
            prefix,
            reports[:size],
            {qid: feedback[qid] for qid in prefix},
            protocol_id=selected.protocol,
            retrieval_budget=3,
            **({"profile": selected.name} if selected != LEGACY else {}),
        )
        if (
            set(bank.allowed_source_ids) != set(prefix)
            or bank.protocol_id != selected.protocol
            or any(not set(record.source_qids) <= set(prefix) for record in bank.records)
        ):
            raise ValueError("built bank escaped its source prefix")
        bank.verify_fingerprint(FrozenOperatorBank.from_json(bank.to_json()).fingerprint)
        published = len(bank.published_specs)
        summaries[str(size)] = {
            "source_question_count": size,
            "retained_record_count": len(bank.records),
            "candidate_only_count": len(bank.records) - published,
            "published_count": published,
            "bank_fingerprint": bank.fingerprint,
            "empty_published_bank": published == 0,
            "empty_bank_policy": "fresh_fallback_without_resampling",
            "source_status_counts": dict(
                Counter(item["status"] for item in audit["questions"].values())
            ),
            "trusted": False,
            "cross_question_transfer_verified": False,
        }
        products[str(size)] = (bank, audit)
    bundle = {
        "schema_version": SCHEMA,
        "protocol": selected.protocol,
        "phase": "source",
        "model": scoring["model"],
        "manifest_sha256": manifest_sha,
        "source_sizes": list(SOURCE_SIZES),
        "scoring_audit_sha256": expected_audit_sha256,
        "input_files": input_files,
        "banks": summaries,
        "api_calls": 0,
        "new_gold_loaded": False,
        "evaluation_authorized": False,
        "calibration_evaluation_used_for_memory": False,
        "construction": "independent_prefix_extraction_before_aggregation",
        "publication_notice": "Published means eligible for exploratory frozen evaluation, "
        "not cross-question validation or trust. Candidate retention uses whole-episode "
        "association; it does not establish each operator's causal gain.",
        "empty_bank_notice": "Zero published operators is a valid result: the memory arm "
        "must fall back to FRESH, preserve the same source/test IDs, and report zero publication.",
    }
    if selected != LEGACY:
        bundle.update(
            profile=selected.name,
            execution_signature_sha256=scoring["execution_signature_sha256"],
        )
    if not write:
        return bundle
    for entry in input_files:
        if _sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("frozen input changed during bank construction")
    output_dir.mkdir(parents=True, exist_ok=False)
    for size, (bank, audit) in products.items():
        path = output_dir / f"bank_{size}.json"
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(bank.to_json() + "\n")
        audit_path = output_dir / f"bank_{size}_audit.json"
        write_json(audit_path, audit)
        summaries[size].update(
            file=path.name,
            sha256=_sha(path),
            audit_file=audit_path.name,
            audit_sha256=_sha(audit_path),
        )
    bundle_path = output_dir / "bank_bundle.json"
    write_json(bundle_path, bundle)
    with (output_dir / "bank_bundle.sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{_sha(bundle_path)}  bank_bundle.json\n")
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feedback-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-audit-sha256", required=True)
    parser.add_argument("--build-new", action="store_true")
    parser.add_argument("--profile", choices=[LEGACY.name, ACTION_LIST.name], default=LEGACY.name)
    args = parser.parse_args(argv)
    result = build_operator_banks(
        args.root,
        args.manifest,
        args.feedback_dir,
        args.output_dir,
        expected_audit_sha256=args.score_audit_sha256,
        write=args.build_new,
        **({"profile": args.profile} if args.profile != LEGACY.name else {}),
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "schema_version",
                    "source_sizes",
                    "banks",
                    "api_calls",
                    "new_gold_loaded",
                    "evaluation_authorized",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
