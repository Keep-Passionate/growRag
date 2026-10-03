"""Only generated temporary synthetic archives; no API or real-run writes.

复用已有单测的离线HTTP假客户端建造临时封存，不把这些测试结果称为模型成绩。
真实项目输入只读取已开发过的合成fixture，绝不调用自然题准备器或标签接口。
"""

from __future__ import annotations

import importlib.util
import json
import runpy
from pathlib import Path

import pytest

from growrag.experiments import a1_two_stage_feedback as feedback
from growrag.experiments import a1_two_stage_study as study

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "a1_gate_audit", PROJECT / "scripts/audits/check_a1_two_stage_gate.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
ARCHIVES = runpy.run_path(str(PROJECT / "tests/test_a1_two_stage_feedback.py"))


def nominal_wires(rows):
    """Synthetic labels-as-output fixture, not genuine entailment or API judgments."""
    responses = {}
    reasons = {
        "T": {
            "supported": "expected_type_supported",
            "contradicted": "expected_type_conflict",
            "unknown": "insufficient_type_evidence",
        },
        "R": {
            "supported": "direct_goal_relation",
            "contradicted": "relation_conflict",
            "unknown": "ambiguous_subgoal",
        },
        "P1": {
            "supported": "question_target_supported",
            "contradicted": "target_role_negated",
            "unknown": "missing_role_evidence",
        },
    }
    for index, row in enumerate(rows):
        prepared = study.prepare_payload(row["input"])
        catalog = study.previous.observer.catalog(prepared)
        question_ref = next(ref["ref_id"] for ref in catalog if ref["source_id"] == "Q")
        action_ref = next(ref["ref_id"] for ref in catalog if ref["source_id"] == "A1")
        expected = {
            (item["kind"], item["slot"]): item["status"] for item in row["expected"]["checks"]
        }
        checks = []
        for kind, slot in prepared.semantic_checks:
            status = expected[kind, slot]
            refs = [] if status == "unknown" else [question_ref]
            if kind == "R" and status != "unknown":
                refs.append(action_ref)
            checks.append(
                {
                    "kind": kind,
                    "slot": slot,
                    "status": status,
                    "reason": reasons[kind][status],
                    "ref_ids": refs,
                }
            )
        responses[index, "judge"] = json.dumps({"checks": checks})
    return responses


@pytest.fixture
def archive(tmp_path, monkeypatch):
    def build(*, passing=False, bad=None, fatal=None):
        freeze_path = tmp_path / study.FREEZE
        freeze_path.parent.mkdir(parents=True, exist_ok=True)
        study.write_json(freeze_path, {"test_only": True, "synthetic_not_model_result": True})
        freeze_sha = study._sha(freeze_path)
        builder = ARCHIVES["archive"]
        monkeypatch.setitem(builder.__globals__, "FREEZE_SHA", freeze_sha)
        wire = nominal_wires(study.load_fixture(PROJECT)) if passing else {}
        wire.update(bad or {})
        terminal_sha = builder(tmp_path, monkeypatch, bad=wire, fatal=fatal)
        # This scoring call creates ONLY generated temporary test artifacts.
        # The gate being tested does not call score and never writes output.
        feedback.score(tmp_path, freeze_sha, terminal_sha)
        return {
            "project": tmp_path,
            "freeze_sha": freeze_sha,
            "terminal_sha": terminal_sha,
            "feedback_sha": study._sha(tmp_path / study.FEEDBACK / "feedback_frozen.json"),
        }

    return build


def reseal(project):
    directory = project / study.FEEDBACK
    path = directory / "feedback_frozen.json"
    path.unlink()
    study.write_json(path, {name: study._sha(directory / name) for name in gate._FEEDBACK_FILES})
    return study._sha(path)


def edit(path, action):
    value = json.loads(path.read_bytes())
    action(value)
    path.unlink()
    study.write_json(path, value)


def snapshot(project):
    return {
        str(path.relative_to(project)): path.read_bytes()
        for path in project.rglob("*")
        if path.is_file()
    }


def test_closed_gate_replays_and_writes_nothing(archive):
    options = archive()
    before = snapshot(options["project"])
    result = gate.check_gate(**options)
    assert result["status"] == "gate_closed"
    assert result["natural_prepare_allowed"] is False
    assert result["gate"]["completed"] == 32
    assert result["http_requests"] == 64
    assert result["stage_counts"] == {"locate": 32, "judge": 32}
    assert result["api_calls"] == result["files_written"] == 0
    assert result["preparation_performed"] is False
    assert snapshot(options["project"]) == before


def test_all_local_locator_failures_are_closed_not_unknown(archive):
    options = archive(bad={(index, "locate"): '{"checks":[]}' for index in range(32)})
    result = gate.check_gate(**options)
    assert result["status"] == "gate_closed"
    assert result["gate"]["completed"] == 0
    assert result["http_requests"] == 32
    assert result["stage_counts"] == {"locate": 32, "judge": 0}


def test_generated_nominal_pass_opens_readonly_without_preparing(archive, monkeypatch):
    options = archive(passing=True)
    monkeypatch.setattr(feedback, "score", lambda *args: pytest.fail("gate must not score/write"))
    result = gate.check_gate(**options)
    assert result["status"] == "gate_open"
    assert result["natural_prepare_allowed"] is True
    assert result["reasons"] == []
    assert result["gate"] == {
        "completed": 32,
        "main_matches": 32,
        "A_retained": 16,
        "conflict_B_rejected": 13,
        "unknown_B_preserved": 3,
        "passed": True,
    }
    assert result["files_written"] == 0 and result["preparation_performed"] is False


@pytest.mark.parametrize("field", ["freeze_sha", "terminal_sha", "feedback_sha"])
@pytest.mark.parametrize("value", [None, "", "A" * 64, "a" * 63])
def test_external_pin_format_is_checked_before_any_read(tmp_path, monkeypatch, field, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid external pin must stop before any file read")

    monkeypatch.setattr(gate, "_file", forbidden)
    options = {
        "project": tmp_path,
        "freeze_sha": "a" * 64,
        "terminal_sha": "b" * 64,
        "feedback_sha": "c" * 64,
    }
    options[field] = value
    with pytest.raises(ValueError):
        gate.check_gate(**options)


@pytest.mark.parametrize("field", ["freeze_sha", "terminal_sha", "feedback_sha"])
def test_external_pin_must_match_actual_sealed_bytes(archive, field):
    options = archive()
    options[field] = "0" * 64
    with pytest.raises(ValueError):
        gate.check_gate(**options)


def test_resealed_passed_boolean_is_not_authority(archive):
    options = archive()
    edit(
        options["project"] / study.FEEDBACK / "SUMMARY.json",
        lambda value: value["gate"].update(passed=True),
    )
    options["feedback_sha"] = reseal(options["project"])
    with pytest.raises(ValueError, match="feedback differs from replay"):
        gate.check_gate(**options)


def test_resealed_threshold_counts_cannot_forge_pass(archive):
    options = archive()
    edit(
        options["project"] / study.FEEDBACK / "SUMMARY.json",
        lambda value: value["gate"].update(
            main_matches=32,
            A_retained=16,
            conflict_B_rejected=13,
            unknown_B_preserved=3,
            passed=True,
        ),
    )
    options["feedback_sha"] = reseal(options["project"])
    with pytest.raises(ValueError, match="feedback differs from replay"):
        gate.check_gate(**options)


@pytest.mark.parametrize("name", ["SUMMARY.json", "audit.json", "per_row.json"])
def test_changed_feedback_without_reseal_is_rejected(archive, name):
    options = archive()
    path = options["project"] / study.FEEDBACK / name
    path.write_bytes(path.read_bytes() + b" ")  # Generated temporary corruption only.
    with pytest.raises(ValueError, match="member SHA"):
        gate.check_gate(**options)


def test_extra_feedback_json_is_not_ignored(archive):
    options = archive()
    study.write_json(options["project"] / study.FEEDBACK / "unsealed.json", {})
    with pytest.raises(ValueError, match="unsealed"):
        gate.check_gate(**options)


@pytest.mark.parametrize(
    "relative", ["../escape.json", "qwenAPI.md", "data/new_gold.json", "C:/escape.json"]
)
def test_resealed_audit_cannot_open_unregistered_paths(archive, relative):
    options = archive()
    edit(
        options["project"] / study.FEEDBACK / "audit.json",
        lambda value: value["input_sha256"].update({relative: "0" * 64}),
    )
    options["feedback_sha"] = reseal(options["project"])
    with pytest.raises(ValueError, match="unregistered audit input"):
        gate.check_gate(**options)


def test_missing_declared_replay_input_is_rejected(archive):
    options = archive()
    edit(
        options["project"] / study.FEEDBACK / "audit.json",
        lambda value: value["input_sha256"].pop(f"{study.OUTPUT}/final_budget.json"),
    )
    options["feedback_sha"] = reseal(options["project"])
    with pytest.raises(ValueError, match="not sealed in audit"):
        gate.check_gate(**options)


def test_new_unknown_usage_remains_closed(archive):
    options = archive(fatal="unknown_usage")
    result = gate.check_gate(**options)
    assert result["status"] == "gate_closed"
    assert result["new_unknown_usage_requests"] == 1
    assert "new_usage_or_transport_not_complete" in result["reasons"]


@pytest.mark.parametrize("passing,code,status", [(False, 1, "gate_closed"), (True, 0, "gate_open")])
def test_cli_exit_matches_gate_not_batch_completion(archive, capsys, passing, code, status):
    options = archive(passing=passing)
    result = gate.main(
        [
            "--project",
            str(options["project"]),
            "--expected-freeze-sha256",
            options["freeze_sha"],
            "--expected-terminal-sha256",
            options["terminal_sha"],
            "--expected-feedback-sha256",
            options["feedback_sha"],
        ]
    )
    assert result == code
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_cli_integrity_failure_is_nonzero_and_not_authorization(archive, capsys):
    options = archive()
    code = gate.main(
        [
            "--project",
            str(options["project"]),
            "--expected-freeze-sha256",
            options["freeze_sha"],
            "--expected-terminal-sha256",
            "0" * 64,
            "--expected-feedback-sha256",
            options["feedback_sha"],
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["natural_prepare_allowed"] is False
