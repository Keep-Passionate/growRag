"""Offline freeze certificate: checks provenance, never runs or unlocks evaluation.

中文：SHA证书仅检测意外更改，不是可信第三方签名。必须先有完整来源预测、独立
评分与四库；本模块不读dev标签，不解码评测题文，不调用API，不修改运行器硬锁。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from growrag.operator_bank import FrozenOperatorBank

from .build_operator_banks import SCHEMA as BANK_SCHEMA
from .build_operator_banks import _read, build_operator_banks
from .fresh_dev_manifest import _sha
from .operator_data_plan import SOURCE_SIZES
from .operator_execution_signature import execution_signature, validate_execution_signature
from .pre_pilot import write_json
from .representation_runner import fingerprint
from .run_operator_study import ARMS, PREFIX, PROTOCOL, artifact, load_inputs
from .score_operator_sources import collect_frozen_sources

SCHEMA = "growrag-operator-evaluation-freeze-v1"
_SHA = re.compile(r"[0-9a-f]{64}")


def _path(root: Path, value: str | Path, *, directory: bool = False) -> Path:
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root) or (not path.is_dir() if directory else not path.is_file()):
        raise ValueError("freeze input escapes the project or has the wrong type")
    return path


def _expected(value: str) -> None:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("explicit expected SHA256 required")


def _no_evaluation_claims(runs: Path) -> None:
    # Claims count as preparation already underway, even before the first API call.
    paths = [*runs.glob(f"{PREFIX}*/launch_plan.json"), *runs.glob(f"{PREFIX}*.claim.json")]
    for path in paths:
        data = _read(_path(runs, path))
        if data.get("protocol") == PROTOCOL and data.get("phase") == "evaluation":
            raise ValueError("evaluation already claimed; cannot create a before-evaluation freeze")


def _freeze_body(
    root: Path,
    manifest_path: Path,
    runs_root: Path,
    feedback_dir: Path,
    banks_dir: Path,
    expected: dict,
    *,
    require_unstarted_evaluation: bool,
) -> dict:
    root = root.resolve(strict=True)
    manifest_path = _path(root, manifest_path)
    runs_root, feedback_dir, banks_dir = (
        _path(root, path, directory=True) for path in (runs_root, feedback_dir, banks_dir)
    )
    for digest in expected.values():
        _expected(digest)
    if require_unstarted_evaluation:
        _no_evaluation_claims(runs_root)
    snapshots = {}

    def pin(path: Path, digest: str | None = None):
        path = _path(root, path)
        actual = _sha(path)
        if digest is not None and actual != digest:
            raise ValueError("freeze input SHA mismatch")
        snapshots[path.relative_to(root).as_posix()] = actual
        return actual

    pin(manifest_path, expected["manifest"])
    manifest, _ = load_inputs(manifest_path, "source")  # Never decodes evaluation runtime.
    if len(manifest["roles"]["source"]) != 500 or len(manifest["roles"]["evaluation"]) != 500:
        raise ValueError("source500 and independent evaluation500 required")
    for role in ("source", "calibration", "evaluation"):
        name = f"{role}_runtime_questions.jsonl"
        pin(artifact(manifest_path.parent, manifest, name), manifest["artifacts"][name]["sha256"])
    pin(
        artifact(manifest_path.parent, manifest, "corpus.jsonl"),
        manifest["artifacts"]["corpus.jsonl"]["sha256"],
    )
    signature = execution_signature(root)
    if validate_execution_signature(signature) != expected["execution"]:
        raise ValueError("current method differs from the expected frozen execution signature")
    for name, digest in signature["files"].items():
        pin(root / name, digest)
    # Runner orchestration is intentionally outside the nine method files, but its
    # empty-memory fallback and evaluation dispatch must not change after freeze.
    wrapper_path = _path(root, root / "src/growrag/experiments/run_operator_study.py")
    wrapper_sha = pin(wrapper_path)

    reports, model, prediction_inputs = collect_frozen_sources(
        manifest, expected["manifest"], runs_root
    )
    if model != signature["configuration"]["model"]:
        raise ValueError("source model differs from frozen execution model")
    for entry in prediction_inputs:
        path = _path(root, Path(entry["path"]))
        pin(path, entry["sha256"])
        if path.name == "launch_plan.json":
            launch = _read(path)
            if (
                validate_execution_signature(launch.get("execution_signature"))
                != expected["execution"]
            ):
                raise ValueError("source launch used a different method signature")
    scoring_path = _path(root, feedback_dir / "audit.json")
    pin(scoring_path, expected["scoring_audit"])
    scoring = _read(scoring_path)
    if fingerprint(reports) != scoring.get("source_reports_sha256"):
        raise ValueError("scored reports differ from current frozen source collection")

    def input_map(entries):
        result = {}
        for entry in entries:
            path = _path(root, Path(entry["path"]))
            name = path.relative_to(root).as_posix()
            if name in result:
                raise ValueError("duplicate source provenance input")
            result[name] = entry["sha256"]
        return result

    if input_map(scoring.get("prediction_inputs", [])) != input_map(prediction_inputs):
        raise ValueError("source scoring provenance omitted or added prediction inputs")
    # Verify train label provenance by byte hashes only; no parquet/answer decoding.
    declared = {item["path"].replace("\\", "/"): item["sha256"] for item in manifest["input_files"]}
    train_dir = "data/hotpotqa/official_train_v1_1/"
    train_provenance = _path(root, root / f"{train_dir}mirror_provenance.json")
    if f"{train_dir}mirror_provenance.json" not in declared:
        raise ValueError("train provenance missing from frozen data manifest")
    pin(train_provenance, declared.get(f"{train_dir}mirror_provenance.json"))
    provenance = _read(train_provenance)
    if provenance.get("official_split") != "train":
        raise ValueError("source scoring provenance is not official train")
    expected_shards = {}
    for shard in provenance.get("shards", []):
        path = _path(root, train_provenance.parent / shard["file"])
        name = path.relative_to(root).as_posix()
        if not name.startswith(train_dir) or declared.get(name) != shard["sha256"]:
            raise ValueError("source shard provenance differs from the data manifest")
        pin(path, shard["sha256"])
        expected_shards[name] = shard["sha256"]
    if not expected_shards or input_map(scoring.get("gold_inputs", [])) != expected_shards:
        raise ValueError("scoring label provenance does not cover exactly the pinned train shards")

    # Rebuild in memory, independently per prefix, to verify published banks.
    rebuilt = build_operator_banks(
        root,
        manifest_path,
        feedback_dir,
        banks_dir,
        expected_audit_sha256=expected["scoring_audit"],
        write=False,
    )
    bundle_path = _path(root, banks_dir / "bank_bundle.json")
    pin(bundle_path, expected["bank_bundle"])
    sidecar = _path(root, banks_dir / "bank_bundle.sha256")
    if (
        sidecar.read_text(encoding="ascii").strip()
        != f"{expected['bank_bundle']}  bank_bundle.json"
    ):
        raise ValueError("bank bundle SHA sidecar mismatch")
    pin(sidecar)
    bundle = _read(bundle_path)
    for key in (
        "schema_version",
        "protocol",
        "phase",
        "model",
        "manifest_sha256",
        "source_sizes",
        "scoring_audit_sha256",
        "input_files",
        "construction",
    ):
        if bundle.get(key) != rebuilt.get(key):
            raise ValueError("bank bundle differs from independently rebuilt source provenance")
    if bundle.get("schema_version") != BANK_SCHEMA or set(bundle.get("banks", {})) != {
        str(size) for size in SOURCE_SIZES
    }:
        raise ValueError("all four frozen banks are required")
    banks = {}
    for size in SOURCE_SIZES:
        item, computed = bundle["banks"][str(size)], rebuilt["banks"][str(size)]
        if any(item.get(key) != value for key, value in computed.items()):
            raise ValueError("bank summary differs from independent prefix extraction")
        if (
            item.get("file") != f"bank_{size}.json"
            or item.get("audit_file") != f"bank_{size}_audit.json"
        ):
            raise ValueError("bank filename differs from fixed scale")
        path, audit_path = (
            _path(root, banks_dir / item["file"]),
            _path(root, banks_dir / item["audit_file"]),
        )
        pin(path, item["sha256"])
        pin(audit_path, item["audit_sha256"])
        bank = FrozenOperatorBank.from_json(path.read_text(encoding="utf-8"))
        if (
            bank.fingerprint != computed["bank_fingerprint"]
            or bank.protocol_id != PROTOCOL
            or set(bank.allowed_source_ids) != set(manifest["nested_source_ids"][str(size)])
        ):
            raise ValueError("bank content or source scope mismatch")
        bank_audit = _read(audit_path)
        if (
            bank_audit.get("protocol_id") != PROTOCOL
            or bank_audit.get("source_prefix_ids") != manifest["nested_source_ids"][str(size)]
            or set(bank_audit.get("questions", {})) != set(bank.allowed_source_ids)
        ):
            raise ValueError("bank audit source scope mismatch")
        banks[str(size)] = {
            "fingerprint": bank.fingerprint,
            "file_sha256": item["sha256"],
            "published_count": len(bank.published_specs),
            "empty_bank_fallback": "fresh" if not bank.published_specs else None,
        }
    for entry in rebuilt["input_files"]:
        pin(_path(root, Path(entry["path"])), entry["sha256"])
    # Ensure none of the inspected files changed while the certificate was assembled.
    for name, digest in snapshots.items():
        if _sha(_path(root, root / name)) != digest:
            raise ValueError("freeze input changed during audit")
    configuration = signature["configuration"]
    return {
        "schema_version": SCHEMA,
        "protocol": PROTOCOL,
        "paths": {
            "manifest": manifest_path.relative_to(root).as_posix(),
            "runs": runs_root.relative_to(root).as_posix(),
            "feedback": feedback_dir.relative_to(root).as_posix(),
            "banks": banks_dir.relative_to(root).as_posix(),
        },
        "expected": expected,
        "execution_signature": signature,
        "runner_sha256": wrapper_sha,
        "memory_fallback_policy": "empty-visible-library-is-fresh-v1",
        "evaluation_ids": manifest["roles"]["evaluation"],
        "evaluation_order_sha256": fingerprint(manifest["roles"]["evaluation"]),
        "source_count": 500,
        "source_terminal_counts": dict(
            Counter(
                "all_arms_completed"
                if len(item["arms"]) == 3
                and all(row["status"] == "completed" for row in item["arms"].values())
                else "failure_preserved"
                for item in reports
            )
        ),
        "banks": banks,
        "input_files": snapshots,
        "controls": {
            arm: {
                "retrieval_budget": 1 if arm == "base" else 3,
                "top_k": configuration["top_k"],
                "max_decisions": configuration["max_decisions"],
                "memory_updates": False,
            }
            for arm in ARMS
        },
        "calibration_policy": "syntax_and_cost_only_no_gold_no_hyperparameter_selection",
        "evaluation_feedback_policy": "score_only_after_all_predictions_frozen_never_update",
        "new_gold_decoded": False,
        "evaluation_runtime_decoded": False,
        "api_calls": 0,
        "runner_unlock_performed": False,
        "limitations": "Hashes detect accidental changes, not adversarial re-signing. "
        "The evaluation order is bound to manifest IDs and the runtime file SHA, without "
        "decoding its text. Source provenance hashes do not prove dataset authenticity. "
        "Calibration no-tuning is a declared protocol, not proof of all human decisions. "
        "A certificate is necessary but not sufficient for a future runner authorization.",
    }


def create_evaluation_freeze(
    root: Path,
    manifest_path: Path,
    runs_root: Path,
    feedback_dir: Path,
    banks_dir: Path,
    certificate_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_scoring_audit_sha256: str,
    expected_bank_bundle_sha256: str,
    expected_execution_sha256: str,
    write: bool = False,
) -> dict:
    """Create a new certificate only before any evaluation launch/claim exists."""
    root = Path(root).resolve(strict=True)
    certificate_path = (root / certificate_path).resolve()
    if not certificate_path.is_relative_to(root) or certificate_path.suffix != ".json":
        raise ValueError("certificate must be a new project-local JSON file")
    if write and (certificate_path.exists() or certificate_path.with_suffix(".sha256").exists()):
        raise FileExistsError("freeze certificate already exists; never overwrite")
    expected = {
        "manifest": expected_manifest_sha256,
        "scoring_audit": expected_scoring_audit_sha256,
        "bank_bundle": expected_bank_bundle_sha256,
        "execution": expected_execution_sha256,
    }
    body = _freeze_body(
        root,
        manifest_path,
        runs_root,
        feedback_dir,
        banks_dir,
        expected,
        require_unstarted_evaluation=True,
    )
    certificate = {
        "freeze": body,
        "issued_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_unstarted_at_issue": True,
    }
    certificate["fingerprint"] = fingerprint(certificate)
    if write:
        certificate_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(certificate_path, certificate)
        with certificate_path.with_suffix(".sha256").open("x", encoding="ascii") as handle:
            handle.write(f"{_sha(certificate_path)}  {certificate_path.name}\n")
    return certificate


def validate_evaluation_freeze(
    root: Path, certificate_path: Path, *, expected_certificate_sha256: str
) -> dict:
    """Re-audit every dependency; does not authorize API calls or load runtime text."""
    root = Path(root).resolve(strict=True)
    path = _path(root, certificate_path)
    _expected(expected_certificate_sha256)
    if _sha(path) != expected_certificate_sha256:
        raise ValueError("evaluation certificate SHA mismatch")
    if (
        path.with_suffix(".sha256").read_text(encoding="ascii").strip()
        != f"{expected_certificate_sha256}  {path.name}"
    ):
        raise ValueError("evaluation certificate sidecar mismatch")
    certificate = _read(path)
    if (
        set(certificate)
        != {"freeze", "issued_at_utc", "evaluation_unstarted_at_issue", "fingerprint"}
        or certificate.get("evaluation_unstarted_at_issue") is not True
        or certificate["fingerprint"]
        != fingerprint({key: value for key, value in certificate.items() if key != "fingerprint"})
    ):
        raise ValueError("invalid evaluation certificate payload")
    body = certificate["freeze"]
    paths = body["paths"]
    refreshed = _freeze_body(
        root,
        Path(paths["manifest"]),
        Path(paths["runs"]),
        Path(paths["feedback"]),
        Path(paths["banks"]),
        body["expected"],
        require_unstarted_evaluation=False,
    )
    if refreshed != body:
        raise ValueError("evaluation freeze dependencies changed")
    return body


def main(argv=None):
    """Thin offline CLI: default is a full prerequisite audit without writing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--certificate", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--issue-new", action="store_true")
    mode.add_argument("--validate", action="store_true")
    parser.add_argument("--expected-certificate-sha256")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--feedback-dir", type=Path)
    parser.add_argument("--banks-dir", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-scoring-audit-sha256")
    parser.add_argument("--expected-bank-bundle-sha256")
    parser.add_argument("--expected-execution-sha256")
    args = parser.parse_args(argv)
    if args.validate:
        if args.expected_certificate_sha256 is None:
            parser.error("--validate requires --expected-certificate-sha256")
        body = validate_evaluation_freeze(
            args.root,
            args.certificate,
            expected_certificate_sha256=args.expected_certificate_sha256,
        )
        action, written = "validated", False
    else:
        required = (
            "manifest",
            "feedback_dir",
            "banks_dir",
            "expected_manifest_sha256",
            "expected_scoring_audit_sha256",
            "expected_bank_bundle_sha256",
            "expected_execution_sha256",
        )
        missing = [
            "--" + name.replace("_", "-") for name in required if getattr(args, name) is None
        ]
        if missing:
            parser.error("preflight/issuance requires " + ", ".join(missing))
        result = create_evaluation_freeze(
            args.root,
            args.manifest,
            args.runs,
            args.feedback_dir,
            args.banks_dir,
            args.certificate,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_scoring_audit_sha256=args.expected_scoring_audit_sha256,
            expected_bank_bundle_sha256=args.expected_bank_bundle_sha256,
            expected_execution_sha256=args.expected_execution_sha256,
            write=args.issue_new,
        )
        body = result["freeze"]
        action, written = ("issued", True) if args.issue_new else ("preflight_only", False)
    summary = {
        "mode": action,
        "certificate_written": written,
        "freeze_fingerprint": fingerprint(body),
        "source_count": body["source_count"],
        "evaluation_count": len(body["evaluation_ids"]),
        "arm_count": len(body["controls"]),
        "runner_unlock_performed": False,
        "api_calls": 0,
    }
    if written or args.validate:
        summary["certificate_sha256"] = _sha((args.root / args.certificate).resolve(strict=True))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
