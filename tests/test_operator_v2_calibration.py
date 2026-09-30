"""Synthetic v2 dispatch and fail-closed scope; never API, gold or real dataset."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_run_operator_study import args, dump, read
from test_run_operator_study import sandbox as sandbox  # Reuse only synthetic runner fixtures.

from growrag.experiments import operator_profiles as profiles
from growrag.experiments import run_operator_study as study
from growrag.macro_operators import GoalContract
from growrag.operator_loop import ActionProposal


@pytest.fixture
def structured(sandbox, monkeypatch):
    signature = {
        "schema": profiles.STRUCTURED_SCHEMA,
        "configuration": profiles.structured_execution_configuration(),
        "files": {name: "b" * 64 for name in profiles.STRUCTURED_METHOD_FILES},
    }
    signature["sha256"] = study.fingerprint(signature)
    monkeypatch.setattr(study, "structured_execution_signature", lambda root: signature)
    calls = []

    class Planner:
        def __init__(self, client, *, mode, specs, trace_prefix, on_record):
            self.client, self.trace = client, trace_prefix
            calls.append(("planner_construct", mode))

        def __call__(self, state):
            self.client.complete(
                [{"role": "user", "content": "synthetic structured planner"}],
                trace_id=f"{self.trace}/plan/{state.decision_number}",
                prompt_version="growrag-evidence-requirements-v2",
            )
            calls.append(("planner_call", state.question.question_id))
            return ActionProposal(GoalContract(state.question.text, "lookup"), None, origin="stop")

    def reader(client, question, evidence, *, trace_id, on_record=None):
        client.complete(
            [{"role": "user", "content": "synthetic structured reader"}],
            trace_id=trace_id,
            prompt_version="growrag-short-supported-answer-v2",
        )
        calls.append(("reader_call", question.question_id))
        assert callable(on_record)
        on_record({"wire_output": {"synthetic": True}})
        return {"answer": "", "supported": False, "evidence_ids": []}

    monkeypatch.setitem(
        sys.modules,
        "growrag.experiments.operator_model_v2",
        SimpleNamespace(ModelOperatorPlannerV2=Planner, answer_episode_v2=reader),
    )
    return SimpleNamespace(signature=signature, calls=calls)


def command(sandbox, structured, *, live=True):
    return args(sandbox, count=1, live=live) + [
        "--profile",
        "structured-v2",
        "--expected-execution-sha256",
        structured.signature["sha256"],
    ]


def output(sandbox):
    return next(
        path
        for path in (sandbox.root / "runs").glob(f"{profiles.STRUCTURED.prefix}*")
        if path.is_dir()
    )


@pytest.mark.parametrize(
    "options",
    [
        ["--phase", "source"],
        ["--phase", "evaluation"],
        ["--banks", "missing"],
        ["--resume-certificate", "missing"],
        ["--evaluation-freeze", "missing"],
        ["--expected-freeze-sha256", "a" * 64],
        ["--arms", "memory50"],
    ],
)
@pytest.mark.parametrize("live", [False, True])
def test_v2_closed_phases_rejected_before_any_input_path(tmp_path, monkeypatch, options, live):
    monkeypatch.chdir(tmp_path)
    cmd = [
        "--manifest",
        "NONEXISTENT/manifest.json",
        "--profile",
        "structured-v2",
        "--phase",
        "calibration",
    ] + options
    if live:
        cmd.append("--allow-network")
    with pytest.raises(ValueError, match="calibration-only"):
        study.main(cmd)
    assert not (tmp_path / "runs").exists()


def test_v2_live_requires_method_pin_before_manifest_access(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="expected execution"):
        study.main(
            [
                "--manifest",
                "missing",
                "--phase",
                "calibration",
                "--profile",
                "structured-v2",
                "--allow-network",
            ]
        )


def test_v2_dry_run_does_not_load_adapter_keys_history_or_network(sandbox, structured, capsys):
    assert study.main(command(sandbox, structured, live=False)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["protocol"] == profiles.STRUCTURED.protocol
    assert result["profile"] == profiles.STRUCTURED.name
    assert result["run_id"].startswith(profiles.STRUCTURED.prefix)
    assert structured.calls == sandbox.clients == sandbox.indexes == []
    assert sandbox.history_calls == 0
    assert not (sandbox.root / "runs").exists()


def test_v2_calibration_dispatch_records_structured_config_and_inherited_budget(
    sandbox, structured
):
    sandbox.prior = 199.99
    assert study.main(command(sandbox, structured) + ["--project-cap-cny", "200"]) == 0
    root = output(sandbox)
    plan, ledger = read(root / "launch_plan.json"), read(root / "final_budget.json")
    assert plan["protocol"] == profiles.STRUCTURED.protocol
    assert plan["profile_scope"] == "calibration_only_not_source_or_evaluation"
    assert plan["execution_signature"] == structured.signature
    assert plan["json_schema_mode"] is True and plan["json_object_mode"] is False
    assert plan["prior_reserved_cny"] == 199.99
    assert plan["project_cap_cny"] == 200 and plan["historical_authorized_total_cny"] == 50
    assert ledger["limits"]["budget_cny"] == pytest.approx(0.01)
    assert ledger["api_requests"] == 5
    assert sandbox.clients[0].config.json_schema_mode is True
    assert sandbox.clients[0].config.json_object_mode is False
    assert len([x for x in structured.calls if x[0] == "planner_call"]) == 2
    assert len([x for x in structured.calls if x[0] == "reader_call"]) == 3
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert len([event for event in events if event["kind"] == "reader_record"]) == 3
    assert read(root / "predictions_frozen.json")["status"] == "completed"
    assert not (root.parent / ".operator-study.lock").exists()


def test_v2_wrong_method_pin_stops_before_new_claim_or_calls(sandbox, structured):
    with pytest.raises(ValueError, match="execution signature mismatch"):
        study.main(command(sandbox, structured) + ["--expected-execution-sha256", "f" * 64])
    assert sandbox.clients == sandbox.indexes == []
    assert not list((sandbox.root / "runs").glob("*.claim.json"))


@pytest.mark.parametrize("kind", ["claim", "launch"])
@pytest.mark.parametrize("phase", ["source", "calibration", "evaluation"])
def test_v2_cannot_escape_legacy_role_or_arm_claims(sandbox, structured, kind, phase):
    runs = sandbox.root / "runs"
    runs.mkdir()
    old = {
        "protocol": profiles.LEGACY.protocol,
        "phase": phase,
        "question_ids": sandbox.manifest["roles"]["calibration"][:1],
        "arms": ["base"],
    }
    if kind == "claim":
        path = runs / f"{profiles.LEGACY.prefix}old.claim.json"
    else:
        folder = runs / f"{profiles.LEGACY.prefix}old"
        folder.mkdir()
        path = folder / "launch_plan.json"
    dump(path, old)
    with pytest.raises(ValueError, match="different phase|no automatic replay"):
        study.main(command(sandbox, structured))
    assert read(path) == old
    assert sandbox.clients == []


def test_legacy_cannot_replay_a_structured_calibration_question(sandbox, structured):
    assert study.main(command(sandbox, structured)) == 0
    with pytest.raises(ValueError, match="no automatic replay"):
        study.main(args(sandbox, count=1))
    assert len(sandbox.clients) == 1


def test_v2_failure_is_frozen_without_retry_or_implicit_resume(sandbox, structured):
    sandbox.failure_at = 4
    assert study.main(command(sandbox, structured)) == 1
    root = output(sandbox)
    report = read(root / "predictions.json")[0]["arms"]
    assert report["base"]["status"] == report["fresh"]["status"] == "completed"
    assert report["static"]["status"] == "failed"
    assert read(root / "final_budget.json")["api_requests"] == 4
    assert read(root / "predictions_frozen.json")["status"] == "failed"
    with pytest.raises(ValueError, match="no automatic replay"):
        study.main(command(sandbox, structured))
    assert len(sandbox.clients) == 1


def test_v1_and_v2_use_the_same_nonstealable_serial_lock(sandbox, structured):
    runs = sandbox.root / "runs"
    runs.mkdir()
    (runs / ".operator-study.lock").write_text("legacy work is still running", encoding="utf-8")
    with pytest.raises(FileExistsError):
        study.main(command(sandbox, structured))
    assert sandbox.clients == []
    assert (runs / ".operator-study.lock").read_text() == "legacy work is still running"


@pytest.mark.parametrize("malformed_static", [False, True])
def test_real_v2_adapter_schema_and_runner_integrate_without_network(
    sandbox, monkeypatch, malformed_static
):
    from growrag.experiments.operator_schemas import PLANNER_VERSIONS, READER_VERSION

    signature = {
        "schema": profiles.STRUCTURED_SCHEMA,
        "configuration": profiles.structured_execution_configuration(),
        "files": {name: "b" * 64 for name in profiles.STRUCTURED_METHOD_FILES},
    }
    signature["sha256"] = study.fingerprint(signature)
    monkeypatch.setattr(study, "structured_execution_signature", lambda root: signature)
    fake_live = study.LiveChatClient

    class StructuredFakeLive(fake_live):
        def complete(self, messages, *, trace_id, prompt_version):
            response = super().complete(messages, trace_id=trace_id, prompt_version=prompt_version)
            if prompt_version in PLANNER_VERSIONS.values():
                value = {
                    "decision": "stop",
                    "reason": "no_useful_query",
                    "intent": "lookup",
                    "constraints": [],
                    "selected_operator": None,
                    "operator": None,
                    "gap_entries": [],
                    "bindings": [],
                }
                if malformed_static and prompt_version == PLANNER_VERSIONS["static"]:
                    value["operator"] = {}  # Never normalize an illegal STOP into success.
                return replace(response, content=json.dumps(value))
            assert prompt_version == READER_VERSION
            return response

    monkeypatch.setattr(study, "LiveChatClient", StructuredFakeLive)
    cmd = args(sandbox, count=1) + [
        "--profile",
        "structured-v2",
        "--expected-execution-sha256",
        signature["sha256"],
    ]
    assert study.main(cmd) == int(malformed_static)
    root = output(sandbox)
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    reader_raw = [
        event
        for event in events
        if event["kind"] == "reader_record" and event.get("stage") == "raw_wire"
    ]
    planner_raw = [
        event
        for event in events
        if event["kind"] == "planner_record" and event.get("stage") == "raw_wire"
    ]
    assert len(reader_raw) == (2 if malformed_static else 3)
    assert len(planner_raw) == 2
    assert all(event["prompt_version"] == READER_VERSION for event in reader_raw)
    assert read(root / "final_budget.json")["api_requests"] == (4 if malformed_static else 5)
    assert read(root / "predictions_frozen.json")["status"] == (
        "failed" if malformed_static else "completed"
    )
