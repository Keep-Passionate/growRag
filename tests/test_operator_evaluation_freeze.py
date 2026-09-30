"""Offline certificate tests using synthetic complete source provenance only."""

import json
from pathlib import Path

import pytest

from growrag.experiments import build_operator_banks as builder
from growrag.experiments import operator_evaluation_freeze as module
from growrag.experiments.operator_execution_signature import METHOD_FILES, execution_signature
from growrag.experiments.score_operator_sources import SCHEMA as FEEDBACK_SCHEMA


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _qid(number):
    return f"{number:024x}"


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    # These are inert text fixtures hashed by execution_signature, never imported.
    for name in (
        *METHOD_FILES,
        "src/growrag/experiments/run_operator_study.py",
        *module.ANALYSIS_FILES,
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic frozen code\n", encoding="utf-8")
    signature = execution_signature(tmp_path)
    data = tmp_path / "data/operator"
    data.mkdir(parents=True)
    roles = {
        "source": [_qid(i) for i in range(500)],
        "calibration": [_qid(i) for i in range(500, 600)],
        "evaluation": [_qid(i) for i in range(1000, 1500)],
    }
    manifest = {
        "roles": roles,
        "nested_source_ids": {str(size): roles["source"][:size] for size in (50, 100, 250, 500)},
        "artifacts": {},
        "input_files": [],
    }
    for name, count in (
        ("source_runtime_questions.jsonl", 500),
        ("calibration_runtime_questions.jsonl", 100),
        ("evaluation_runtime_questions.jsonl", 500),
        ("corpus.jsonl", 1),
    ):
        path = data / name
        path.write_bytes(b"opaque synthetic bytes; never decode evaluation question text\n")
        manifest["artifacts"][name] = {
            "sha256": module._sha(path),
            "rows": count,
            "contains_gold": False,
            "runtime_safe": True,
        }
    train = tmp_path / "data/hotpotqa/official_train_v1_1"
    train.mkdir(parents=True)
    shard = train / "train.parquet"
    shard.write_bytes(b"opaque provenance bytes, not real labels")
    provenance = train / "mirror_provenance.json"
    _write(
        provenance,
        {"official_split": "train", "shards": [{"file": shard.name, "sha256": module._sha(shard)}]},
    )
    manifest["input_files"] = [
        {"path": p.relative_to(tmp_path).as_posix(), "sha256": module._sha(p)}
        for p in (provenance, shard)
    ]
    manifest_path = data / "manifest.json"
    _write(manifest_path, manifest)
    monkeypatch.setattr(module, "load_inputs", lambda path, phase: (manifest, ()))
    monkeypatch.setattr(builder, "load_inputs", lambda path, phase: (manifest, ()))
    runs = tmp_path / "runs"
    launch_dir = runs / f"{module.PREFIX}source_fixture"
    reports = [
        {
            "question_id": qid,
            "arms": {
                arm: {"status": "failed", "feedback": None, "memory_updated": False}
                for arm in ("base", "fresh", "static")
            },
        }
        for qid in roles["source"]
    ]
    launch = {
        "protocol": module.PROTOCOL,
        "phase": "source",
        "model": signature["configuration"]["model"],
        "manifest_sha256": module._sha(manifest_path),
        "question_ids": roles["source"],
        "arms": ["base", "fresh", "static"],
        "gold_loaded": False,
        "memory_updates": False,
        "execution_signature": signature,
    }
    _write(launch_dir / "launch_plan.json", launch)
    _write(launch_dir / "predictions.json", reports)
    _write(
        launch_dir / "predictions_frozen.json",
        {
            "sha256": module.fingerprint(reports),
            "phase": "source",
            "question_ids": roles["source"],
            "gold_loaded": False,
            "status": "failed",
            "cleanup_errors": [],
        },
    )
    feedback_dir = tmp_path / "scored"
    scored_reports = [
        {**report, "phase": "source", "protocol": module.PROTOCOL} for report in reports
    ]
    feedback = {
        qid: {
            arm: dict.fromkeys(("answer_em", "answer_f1", "raw_support_recall"))
            for arm in ("base", "fresh", "static")
        }
        for qid in roles["source"]
    }
    _write(feedback_dir / "feedback.json", feedback)
    _write(feedback_dir / "source_reports.json", scored_reports)
    audit = {
        "schema_version": FEEDBACK_SCHEMA,
        "protocol": module.PROTOCOL,
        "phase": "source",
        "model": launch["model"],
        "manifest_sha256": module._sha(manifest_path),
        "source_ids": roles["source"],
        "source_count": 500,
        "gold_loaded": True,
        "calibration_evaluation_gold_loaded": False,
        "raw_predictions_modified": False,
        "prediction_inputs": [
            {"path": str(launch_dir / name), "sha256": module._sha(launch_dir / name)}
            for name in ("launch_plan.json", "predictions.json", "predictions_frozen.json")
        ],
        "gold_inputs": [{"path": str(shard), "sha256": module._sha(shard)}],
        "feedback_sha256": module.fingerprint(feedback),
        "source_reports_sha256": module.fingerprint(scored_reports),
        "artifacts": {
            name: {"sha256": module._sha(feedback_dir / name)}
            for name in ("feedback.json", "source_reports.json")
        },
    }
    _write(feedback_dir / "audit.json", audit)
    banks = tmp_path / "banks"
    builder.build_operator_banks(
        tmp_path,
        manifest_path,
        feedback_dir,
        banks,
        expected_audit_sha256=module._sha(feedback_dir / "audit.json"),
        write=True,
    )
    return {
        "root": tmp_path,
        "manifest_path": manifest_path,
        "runs_root": runs,
        "feedback_dir": feedback_dir,
        "banks_dir": banks,
        "certificate_path": tmp_path / "certificates/evaluation.json",
        "expected_manifest_sha256": module._sha(manifest_path),
        "expected_scoring_audit_sha256": module._sha(feedback_dir / "audit.json"),
        "expected_bank_bundle_sha256": module._sha(banks / "bank_bundle.json"),
        "expected_execution_sha256": signature["sha256"],
    }


def _validate(args):
    return module.validate_evaluation_freeze(
        args["root"],
        args["certificate_path"],
        expected_certificate_sha256=module._sha(args["certificate_path"]),
    )


def test_dry_freeze_checks_full_source_and_binds_500_order_seven_arms_without_writing(frozen):
    result = module.create_evaluation_freeze(**frozen)
    body = result["freeze"]
    assert not frozen["certificate_path"].exists()
    assert len(body["evaluation_ids"]) == 500
    assert body["source_terminal_counts"] == {"failure_preserved": 500}
    assert set(body["controls"]) == set(module.ARMS)
    assert body["controls"]["base"]["retrieval_budget"] == 1
    assert all(
        value["retrieval_budget"] == 3 for arm, value in body["controls"].items() if arm != "base"
    )
    assert all(value["memory_updates"] is False for value in body["controls"].values())
    assert all(item["empty_bank_fallback"] == "fresh" for item in body["banks"].values())
    assert body["runner_unlock_performed"] is False
    assert body["memory_fallback_policy"] == "empty-visible-library-is-fresh-v1"
    assert body["evaluation_runtime_decoded"] is False and body["new_gold_decoded"] is False
    assert body["analysis"] == module.analysis_signature(frozen["root"])


def test_issue_once_then_revalidate_every_dependency(frozen):
    certificate = module.create_evaluation_freeze(**frozen, write=True)
    assert _validate(frozen) == certificate["freeze"]
    with pytest.raises(FileExistsError):
        module.create_evaluation_freeze(**frozen, write=True)


def test_no_eval_text_or_raw_labels_are_decoded(frozen, monkeypatch):
    original = Path.read_text

    def guarded(path, *args, **kwargs):
        assert path.name not in {"evaluation_runtime_questions.jsonl", "train.parquet"}
        assert "official_dev" not in path.as_posix()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    module.create_evaluation_freeze(**frozen, write=True)
    _validate(frozen)


@pytest.mark.parametrize(
    "name", ["corpus.jsonl", "source_runtime_questions.jsonl", "evaluation_runtime_questions.jsonl"]
)
def test_changed_runtime_or_corpus_invalidates_certificate(frozen, name):
    module.create_evaluation_freeze(**frozen, write=True)
    path = frozen["manifest_path"].parent / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact path/hash"):
        _validate(frozen)


def test_changed_runner_is_detected_even_when_nine_method_signature_is_unchanged(frozen):
    module.create_evaluation_freeze(**frozen, write=True)
    path = frozen["root"] / "src/growrag/experiments/run_operator_study.py"
    path.write_text("changed runner")
    assert execution_signature(frozen["root"])["sha256"] == frozen["expected_execution_sha256"]
    with pytest.raises(ValueError, match="dependencies changed"):
        _validate(frozen)


def test_changed_method_is_detected(frozen):
    module.create_evaluation_freeze(**frozen, write=True)
    (frozen["root"] / METHOD_FILES[0]).write_text("changed method")
    with pytest.raises(ValueError, match="method differs"):
        _validate(frozen)


@pytest.mark.parametrize("name", module.ANALYSIS_FILES)
def test_analysis_code_is_frozen_before_any_evaluation(frozen, name):
    module.create_evaluation_freeze(**frozen, write=True)
    (frozen["root"] / name).write_text("changed analysis", encoding="utf-8")
    assert execution_signature(frozen["root"])["sha256"] == frozen["expected_execution_sha256"]
    with pytest.raises(ValueError, match="dependencies changed"):
        _validate(frozen)


def test_analysis_library_version_change_invalidates_freeze(frozen, monkeypatch):
    module.create_evaluation_freeze(**frozen, write=True)
    monkeypatch.setattr(module, "version", lambda name: "different-version")
    with pytest.raises(ValueError, match="dependencies changed"):
        _validate(frozen)


def test_missing_source_terminal_report_prevents_issuing_certificate(frozen):
    folder = next(frozen["runs_root"].glob(f"{module.PREFIX}*/launch_plan.json")).parent
    reports = json.loads((folder / "predictions.json").read_text())[:-1]
    _write(folder / "predictions.json", reports)
    seal = json.loads((folder / "predictions_frozen.json").read_text())
    seal.update(
        sha256=module.fingerprint(reports), question_ids=[item["question_id"] for item in reports]
    )
    _write(folder / "predictions_frozen.json", seal)
    with pytest.raises(ValueError, match="not all source"):
        module.create_evaluation_freeze(**frozen, write=True)
    assert not frozen["certificate_path"].exists()


def test_existing_eval_claim_prevents_retroactive_issuance_but_not_existing_certificate_validation(
    frozen,
):
    module.create_evaluation_freeze(**frozen, write=True)
    claim = frozen["runs_root"] / f"{module.PREFIX}evaluation.claim.json"
    _write(claim, {"protocol": module.PROTOCOL, "phase": "evaluation"})
    assert _validate(frozen)["source_count"] == 500
    other = {**frozen, "certificate_path": frozen["root"] / "second.json"}
    with pytest.raises(ValueError, match="already claimed"):
        module.create_evaluation_freeze(**other)


def test_bank_file_and_bundle_tampering_are_rejected(frozen):
    module.create_evaluation_freeze(**frozen, write=True)
    path = frozen["banks_dir"] / "bank_50.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA mismatch"):
        _validate(frozen)


def test_train_provenance_bytes_are_pinned_without_parsing_gold(frozen):
    module.create_evaluation_freeze(**frozen, write=True)
    shard = frozen["root"] / "data/hotpotqa/official_train_v1_1/train.parquet"
    shard.write_bytes(b"changed opaque shard")
    with pytest.raises(ValueError, match="SHA mismatch"):
        _validate(frozen)


def test_certificate_sha_and_sidecar_are_both_checked(frozen):
    module.create_evaluation_freeze(**frozen, write=True)
    with pytest.raises(ValueError, match="certificate SHA mismatch"):
        module.validate_evaluation_freeze(
            frozen["root"], frozen["certificate_path"], expected_certificate_sha256="f" * 64
        )
    frozen["certificate_path"].with_suffix(".sha256").write_text("bad")
    with pytest.raises(ValueError, match="sidecar"):
        _validate(frozen)


def test_missing_bank_audit_refuses_certificate(frozen):
    path = frozen["banks_dir"] / "bank_50_audit.json"
    # Rename is test-local fixture manipulation, not an operation on real results.
    path.rename(path.with_suffix(".missing"))
    with pytest.raises(FileNotFoundError):
        module.create_evaluation_freeze(**frozen)


def _cli_args(frozen):
    return [
        "--root",
        str(frozen["root"]),
        "--certificate",
        str(frozen["certificate_path"]),
        "--manifest",
        str(frozen["manifest_path"]),
        "--runs",
        str(frozen["runs_root"]),
        "--feedback-dir",
        str(frozen["feedback_dir"]),
        "--banks-dir",
        str(frozen["banks_dir"]),
        "--expected-manifest-sha256",
        frozen["expected_manifest_sha256"],
        "--expected-scoring-audit-sha256",
        frozen["expected_scoring_audit_sha256"],
        "--expected-bank-bundle-sha256",
        frozen["expected_bank_bundle_sha256"],
        "--expected-execution-sha256",
        frozen["expected_execution_sha256"],
    ]


def test_cli_default_preflight_outputs_only_short_summary_and_no_certificate(frozen, capsys):
    module.main(_cli_args(frozen))
    text = capsys.readouterr().out
    result = json.loads(text)
    assert result["mode"] == "preflight_only" and result["certificate_written"] is False
    assert result["source_count"] == result["evaluation_count"] == 500
    assert result["arm_count"] == 7
    assert len(result["freeze_fingerprint"]) == 64
    assert "evaluation_ids" not in text and _qid(1000) not in text
    assert not frozen["certificate_path"].exists()


def test_cli_explicit_issue_and_validate_modes(frozen, capsys):
    module.main([*_cli_args(frozen), "--issue-new"])
    issued = json.loads(capsys.readouterr().out)
    assert issued["mode"] == "issued" and issued["certificate_written"] is True
    assert issued["certificate_sha256"] == module._sha(frozen["certificate_path"])
    module.main(
        [
            "--root",
            str(frozen["root"]),
            "--certificate",
            str(frozen["certificate_path"]),
            "--validate",
            "--expected-certificate-sha256",
            issued["certificate_sha256"],
        ]
    )
    checked = json.loads(capsys.readouterr().out)
    assert checked["mode"] == "validated" and checked["certificate_written"] is False
    assert checked["freeze_fingerprint"] == issued["freeze_fingerprint"]
    assert checked["runner_unlock_performed"] is False and checked["api_calls"] == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--certificate", "example.json"],
        ["--certificate", "example.json", "--validate"],
        ["--certificate", "example.json", "--validate", "--issue-new"],
    ],
)
def test_cli_missing_or_conflicting_modes_do_not_touch_any_inputs(args, monkeypatch):
    monkeypatch.setattr(
        module, "create_evaluation_freeze", lambda *a, **kw: pytest.fail("unexpected audit")
    )
    monkeypatch.setattr(
        module, "validate_evaluation_freeze", lambda *a, **kw: pytest.fail("unexpected audit")
    )
    with pytest.raises(SystemExit) as error:
        module.main(args)
    assert error.value.code == 2
