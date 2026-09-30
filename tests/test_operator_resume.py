"""Synthetic closed-run continuation proofs, never live API or gold data."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from growrag.experiments import operator_resume as resume
from growrag.experiments.operator_execution_signature import (
    METHOD_FILES,
    execution_configuration,
)
from growrag.experiments.operator_execution_signature import (
    SCHEMA as SIGNATURE_SCHEMA,
)
from growrag.experiments.representation_runner import fingerprint


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def signature(marker="a"):
    body = {
        "schema": SIGNATURE_SCHEMA,
        "configuration": execution_configuration(),
        "files": {name: marker * 64 for name in METHOD_FILES},
    }
    return {**body, "sha256": fingerprint(body)}


def closed_run(
    root, *, name=None, phase="calibration", ids=None, sig=None, parent=None, evaluation=None
):
    name = name or f"{resume.PREFIX}{phase}_base_0000_0004"
    ids = ids or [f"{n:024x}" for n in range(4)]
    directory = root / name
    directory.mkdir(parents=True)
    for folder in ("request_journal", "api_audit"):
        (directory / folder).mkdir()
    launch = {
        "protocol": resume.PROTOCOL,
        "run_id": name,
        "phase": phase,
        "question_ids": ids,
        "arms": ["base"],
        "manifest_sha256": "c" * 64,
        "model": "fixture-model",
        "gold_loaded": False,
        "memory_updates": False,
    }
    if sig:
        launch["execution_signature"] = sig
    if evaluation is not None:
        launch.update(
            arms=list(resume._EVALUATION_ARMS),
            model=evaluation.body["execution_signature"]["configuration"]["model"],
            evaluation_freeze_path=str(evaluation.path),
            evaluation_freeze_sha256=evaluation.sha,
            evaluation_order_sha256=evaluation.body["evaluation_order_sha256"],
            bank_sha256={
                f"memory{n}": row["fingerprint"] for n, row in evaluation.body["banks"].items()
            },
            bank_file_sha256={
                f"memory{n}": row["file_sha256"] for n, row in evaluation.body["banks"].items()
            },
        )
        dump(directory / "evaluation_freeze_verified.json", evaluation.body)
    if parent:
        launch["resume_parent_run_id"] = parent["proof"]["parent_run_id"]
        launch["resume_certificate_sha256"] = fingerprint(parent)
        dump(directory / "resume_certificate.json", parent)
    dump(directory / "launch_plan.json", launch)
    dump(root / f"{name}.claim.json", {**launch, "plan_sha256": fingerprint(launch)})
    trace = f"{name}/{ids[0]}/base/reader"
    call = {"trace_id": trace, "api_requests": 1, "status": "completed"}
    budget = {"calls": [call], "api_requests": 1, "block_reason": None}
    outcome = {
        "question_id": ids[0],
        "arm": "base",
        "status": "failed",
        "error_type": "ValueError",
        "feedback": None,
        "memory_updated": False,
        "calls": [call],
        "episode": None,
    }
    reports = [{"question_id": ids[0], "arms": {"base": outcome}}]
    dump(directory / f"{ids[0]}_base.json", outcome)
    dump(directory / "checkpoint_0000.json", reports[0])
    dump(directory / "predictions.json", reports)
    dump(
        directory / "predictions_frozen.json",
        {
            "sha256": fingerprint(reports),
            "phase": phase,
            "question_ids": [ids[0]],
            "status": "failed",
            "cleanup_errors": [],
            "gold_loaded": False,
        },
    )
    dump(directory / "final_budget.json", budget)
    dump(directory / "request_journal/0000_intent.json", {"trace_id": trace})
    dump(directory / "request_journal/0000_after.json", budget)
    dump(directory / "api_audit/request.json", {"trace_id": trace})
    events = [
        {"kind": "launch"},
        {"kind": "failure", "question_id": ids[0]},
        {"kind": "exit", "status": "failed", "requests": 1},
    ]
    (directory / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
    )
    for filename in ("process.json", "source_snapshot.json", "cumulative_budget.json"):
        dump(directory / filename, {})
    (directory / "live.log").write_text("closed fixture\n", encoding="utf-8")
    return directory, ids


def verify(root, path, ids, *, phase="calibration", method=None):
    return resume.verify_certificate(
        root,
        path,
        phase=phase,
        question_ids=ids,
        arms=["base"],
        manifest_sha256="c" * 64,
        model="fixture-model",
        signature=method,
    )


def test_releases_only_entirely_unstarted_questions(tmp_path):
    directory, ids = closed_run(tmp_path)
    cert = resume.build_certificate(tmp_path, directory.name)
    assert cert["proof"]["question_ids"] == ids[1:]
    assert cert["proof"]["started_question_ids"] == ids[:1]
    assert cert["proof"]["request_evidence"]["raw_audit_count"] == 1
    path = tmp_path / "proof.json"
    dump(path, cert)
    assert verify(tmp_path, path, ids[2:]) == cert
    with pytest.raises(ValueError, match="untouched"):
        verify(tmp_path, path, ids[:1])


@pytest.mark.parametrize(
    "artifact",
    [
        "predictions_frozen.json",
        "checkpoint_0000.json",
        "final_budget.json",
        "request_journal/0000_intent.json",
        "request_journal/0000_after.json",
        "api_audit/request.json",
        "events.jsonl",
        "source_snapshot.json",
    ],
)
def test_missing_evidence_cannot_be_waived(tmp_path, artifact):
    directory, _ = closed_run(tmp_path)
    (directory / artifact).unlink()
    with pytest.raises((ValueError, OSError)):
        resume.build_certificate(tmp_path, directory.name)


@pytest.mark.parametrize("where", ["event", "stray_report", "intent", "audit"])
def test_any_target_footprint_prevents_release(tmp_path, where):
    directory, ids = closed_run(tmp_path)
    if where == "event":
        path = directory / "events.jsonl"
        old = path.read_text().splitlines()
        old.insert(1, json.dumps({"kind": "retrieval", "question_id": ids[1]}))
        path.write_text("\n".join(old), encoding="utf-8")
    elif where == "stray_report":
        dump(directory / f"{ids[1]}_base.json", {})
    elif where == "intent":
        dump(
            directory / "request_journal/0001_intent.json",
            {"trace_id": f"{directory.name}/{ids[1]}/base/reader"},
        )
    else:
        dump(
            directory / "api_audit/extra.json",
            {"trace_id": f"{directory.name}/{ids[1]}/base/reader"},
        )
    with pytest.raises(ValueError):
        resume.build_certificate(tmp_path, directory.name)


def test_old_claim_cannot_be_rewritten(tmp_path):
    directory, _ = closed_run(tmp_path)
    path = tmp_path / f"{directory.name}.claim.json"
    claim = read(path)
    claim["question_ids"] = claim["question_ids"][1:]
    dump(path, claim)
    with pytest.raises(ValueError, match="claim identity"):
        resume.build_certificate(tmp_path, directory.name)


def test_changed_parent_after_certificate_creation_is_detected(tmp_path):
    directory, ids = closed_run(tmp_path)
    cert = resume.build_certificate(tmp_path, directory.name)
    path = tmp_path / "proof.json"
    dump(path, cert)
    (directory / "live.log").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        verify(tmp_path, path, ids[1:])


def test_source_requires_frozen_same_execution_signature(tmp_path):
    directory, ids = closed_run(tmp_path, phase="source", sig=signature())
    cert = resume.build_certificate(tmp_path, directory.name)
    path = tmp_path / "proof.json"
    dump(path, cert)
    assert verify(tmp_path, path, ids[1:], phase="source", method=signature()) == cert
    with pytest.raises(ValueError, match="approved method"):
        verify(tmp_path, path, ids[1:], phase="source", method=signature("b"))


def test_legacy_source_without_signature_is_not_resumable(tmp_path):
    directory, _ = closed_run(tmp_path, phase="source")
    with pytest.raises(ValueError, match="signature"):
        resume.build_certificate(tmp_path, directory.name)


def test_legacy_calibration_can_continue_with_changed_method(tmp_path):
    directory, ids = closed_run(tmp_path)
    cert = resume.build_certificate(tmp_path, directory.name)
    path = tmp_path / "proof.json"
    dump(path, cert)
    assert verify(tmp_path, path, ids[1:], method=signature("b")) == cert


def test_recursive_continuation_preserves_all_ancestor_claims(tmp_path):
    first, ids = closed_run(tmp_path)
    cert = resume.build_certificate(tmp_path, first.name)
    second, _ = closed_run(
        tmp_path, name=f"{resume.PREFIX}calibration_base_0001_0004", ids=ids[1:], parent=cert
    )
    continued = resume.build_certificate(tmp_path, second.name)
    assert continued["proof"]["question_ids"] == ids[2:]
    assert continued["proof"]["ancestor_run_ids"] == [second.name, first.name]
    assert (tmp_path / f"{first.name}.claim.json").is_file()
    assert (tmp_path / f"{second.name}.claim.json").is_file()
    path = tmp_path / "second-proof.json"
    dump(path, continued)
    with pytest.raises(ValueError, match="untouched"):
        verify(tmp_path, path, [ids[1]])


def test_certificate_forgery_recomputed_checksum_still_fails_audit(tmp_path):
    directory, ids = closed_run(tmp_path)
    cert = deepcopy(resume.build_certificate(tmp_path, directory.name))
    cert["proof"]["question_ids"] = ids
    cert["sha256"] = fingerprint({key: cert[key] for key in ("schema", "protocol", "proof")})
    path = tmp_path / "proof.json"
    dump(path, cert)
    with pytest.raises(ValueError, match="evidence changed"):
        verify(tmp_path, path, ids)


@pytest.mark.parametrize("status", ["completed", "running"])
def test_parent_must_be_closed_failed_not_running(tmp_path, status):
    directory, _ = closed_run(tmp_path)
    path = directory / "predictions_frozen.json"
    seal = read(path)
    seal["status"] = status
    dump(path, seal)
    with pytest.raises(ValueError, match="failed prediction seal"):
        resume.build_certificate(tmp_path, directory.name)


def test_unclosed_journal_cannot_be_ignored(tmp_path):
    directory, _ = closed_run(tmp_path)
    dump(directory / "request_journal/0001_intent.json", {"trace_id": "unknown"})
    with pytest.raises(ValueError, match="journal"):
        resume.build_certificate(tmp_path, directory.name)


def test_cli_certificate_is_exclusive_and_offline(tmp_path, capsys):
    directory, ids = closed_run(tmp_path)
    path = tmp_path / "proof.json"
    command = [
        "--runs-root",
        str(tmp_path),
        "--parent-run-id",
        directory.name,
        "--output",
        str(path),
    ]
    assert resume.main(command) == 0
    assert "3 untouched questions; no API" in capsys.readouterr().out
    assert read(path)["proof"]["question_ids"] == ids[1:]
    with pytest.raises(FileExistsError):
        resume.main(command)


@pytest.fixture
def eval_context(tmp_path, monkeypatch):
    from growrag.experiments import operator_evaluation_freeze as freeze

    runs = tmp_path / "runs"
    runs.mkdir()
    runner = tmp_path / "runner-fixture.py"
    runner.write_text("synthetic immutable runner\n", encoding="utf-8")
    ids = [f"{n:024x}" for n in range(500)]
    body = {
        "protocol": resume.PROTOCOL,
        "paths": {"runs": "runs"},
        "expected": {"manifest": "c" * 64, "execution": signature()["sha256"]},
        "execution_signature": signature(),
        "evaluation_ids": ids,
        "evaluation_order_sha256": fingerprint(ids),
        "runner_sha256": freeze._sha(runner),
        "banks": {
            str(n): {"fingerprint": "d" * 64, "file_sha256": "e" * 64} for n in (50, 100, 250, 500)
        },
    }
    path = tmp_path / "evaluation-freeze.json"
    dump(path, {"freeze": body})

    def validate(root, certificate_path, *, expected_certificate_sha256):
        assert root == tmp_path
        if (
            certificate_path != path
            or freeze._sha(path) != expected_certificate_sha256
            or freeze._sha(runner) != body["runner_sha256"]
        ):
            raise ValueError("synthetic evaluation freeze dependencies changed")
        return deepcopy(body)

    monkeypatch.setattr(freeze, "validate_evaluation_freeze", validate)
    return SimpleNamespace(
        root=tmp_path, runs=runs, path=path, sha=freeze._sha(path), body=body, runner=runner
    )


def eval_run(context, *, start=0, end=4, parent=None):
    return closed_run(
        context.runs,
        name=f"{resume.PREFIX}evaluation_all_{start:04d}_{end:04d}",
        phase="evaluation",
        ids=context.body["evaluation_ids"][start:end],
        sig=signature(),
        parent=parent,
        evaluation=context,
    )


def verify_eval(context, path, ids, **changes):
    arguments = {
        "phase": "evaluation",
        "question_ids": ids,
        "arms": resume._EVALUATION_ARMS,
        "manifest_sha256": "c" * 64,
        "model": signature()["configuration"]["model"],
        "signature": signature(),
        "evaluation_freeze": context.path,
        "expected_freeze_sha256": context.sha,
    }
    arguments.update(changes)
    return resume.verify_certificate(context.runs, path, **arguments)


def test_evaluation_has_separate_schema_and_requires_whole_untouched_suffix(eval_context):
    directory, ids = eval_run(eval_context)
    cert = resume.build_certificate(eval_context.runs, directory.name)
    assert cert["schema"] == resume.EVALUATION_SCHEMA
    binding = cert["proof"]["evaluation_binding"]
    assert binding["freeze_sha256"] == eval_context.sha
    assert binding["runner_sha256"] == eval_context.body["runner_sha256"]
    path = eval_context.root / "resume.json"
    dump(path, cert)
    assert verify_eval(eval_context, path, ids[1:]) == cert
    for chosen in (ids[:1], ids[2:], ids[1:][::-1], ids[1:2]):
        with pytest.raises(ValueError, match="untouched"):
            verify_eval(eval_context, path, chosen)


@pytest.mark.parametrize("field", ["freeze", "path", "method", "arms"])
def test_evaluation_cannot_resume_under_other_freeze_or_method(eval_context, field):
    directory, ids = eval_run(eval_context)
    path = eval_context.root / "resume.json"
    dump(path, resume.build_certificate(eval_context.runs, directory.name))
    changes = {
        "freeze": {"expected_freeze_sha256": "f" * 64},
        "path": {"evaluation_freeze": eval_context.runner},
        "method": {"signature": signature("b")},
        "arms": {"arms": ["base"]},
    }[field]
    with pytest.raises(ValueError):
        verify_eval(eval_context, path, ids[1:], **changes)


@pytest.mark.parametrize("field", ["freeze", "copy", "runner", "order", "banks", "arms"])
def test_evaluation_original_freeze_and_bindings_are_reaudited(eval_context, field):
    directory, _ = eval_run(eval_context)
    if field == "freeze":
        dump(eval_context.path, {"changed": True})
    elif field == "copy":
        dump(directory / "evaluation_freeze_verified.json", {"changed": True})
    elif field == "runner":
        eval_context.runner.write_text("changed", encoding="utf-8")
    else:
        launch = read(directory / "launch_plan.json")
        if field == "order":
            launch["evaluation_order_sha256"] = "f" * 64
        elif field == "banks":
            launch["bank_file_sha256"]["memory50"] = "f" * 64
        else:
            launch["arms"] = ["base"]
        dump(directory / "launch_plan.json", launch)
        dump(
            eval_context.runs / f"{directory.name}.claim.json",
            {**launch, "plan_sha256": fingerprint(launch)},
        )
    with pytest.raises(ValueError, match="evaluation"):
        resume.build_certificate(eval_context.runs, directory.name)


def test_evaluation_ancestry_supports_worst_case_25_question_batch(eval_context):
    certificate = None
    for start in range(24):
        directory, ids = eval_run(eval_context, start=start, end=25, parent=certificate)
        certificate = resume.build_certificate(eval_context.runs, directory.name)
        assert certificate["proof"]["question_ids"] == ids[1:]
    assert len(certificate["proof"]["ancestor_run_ids"]) == 24
    assert certificate["proof"]["question_ids"] == eval_context.body["evaluation_ids"][24:25]
    assert len(list(eval_context.runs.glob("*.claim.json"))) == 24


def test_source_calibration_proof_shape_stays_legacy_and_depth_16(tmp_path):
    certificate = None
    ids = [f"{n:024x}" for n in range(18)]
    for start in range(17):
        directory, _ = closed_run(
            tmp_path,
            name=f"{resume.PREFIX}calibration_chain_{start}",
            ids=ids[start:],
            parent=certificate,
        )
        if start == 16:
            with pytest.raises(ValueError, match="ancestry"):
                resume.build_certificate(tmp_path, directory.name)
        else:
            certificate = resume.build_certificate(tmp_path, directory.name)
            assert certificate["schema"] == resume.SCHEMA
            assert "evaluation_binding" not in certificate["proof"]


def test_source_certificate_never_authorizes_evaluation(eval_context):
    directory, ids = closed_run(eval_context.runs, phase="source", sig=signature())
    path = eval_context.root / "source-resume.json"
    dump(path, resume.build_certificate(eval_context.runs, directory.name))
    with pytest.raises(ValueError, match="approved method/role"):
        verify_eval(eval_context, path, ids[1:])


def test_evaluation_schema_cannot_be_downgraded_to_legacy(eval_context):
    directory, ids = eval_run(eval_context)
    certificate = resume.build_certificate(eval_context.runs, directory.name)
    certificate["schema"] = resume.SCHEMA
    certificate["sha256"] = fingerprint(
        {k: certificate[k] for k in ("schema", "protocol", "proof")}
    )
    path = eval_context.root / "downgraded.json"
    dump(path, certificate)
    with pytest.raises(ValueError, match="schema"):
        verify_eval(eval_context, path, ids[1:])
