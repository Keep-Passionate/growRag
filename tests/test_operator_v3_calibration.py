"""Action-list profile integration with synthetic transport, never API or gold."""

import json
import sys
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_operator_profiles import budget_root, synthetic_project
from test_operator_v2_calibration import structured as structured
from test_run_operator_study import args, dump, read
from test_run_operator_study import sandbox as sandbox

from growrag.experiments import operator_profiles as profiles
from growrag.experiments import run_operator_study as study
from growrag.experiments import run_shared_s2g as history_runner
from growrag.experiments.operator_execution_signature import (
    execution_signature,
    validate_execution_signature,
)


def signature_body():
    body = {
        "schema": profiles.ACTION_LIST_SCHEMA,
        "configuration": profiles.action_list_execution_configuration(),
        "files": {name: "c" * 64 for name in profiles.ACTION_LIST_METHOD_FILES},
    }
    return {**body, "sha256": study.fingerprint(body)}


@pytest.fixture
def action_list(sandbox, structured, monkeypatch):
    signature = signature_body()
    monkeypatch.setattr(study, "action_list_execution_signature", lambda root: signature)
    monkeypatch.setitem(
        sys.modules,
        "growrag.experiments.operator_model_v3",
        SimpleNamespace(
            ModelOperatorPlannerV3=sys.modules[
                "growrag.experiments.operator_model_v2"
            ].ModelOperatorPlannerV2
        ),
    )
    return SimpleNamespace(signature=signature, calls=structured.calls)


def command(sandbox, signature, *, live=True):
    return args(sandbox, count=1, live=live) + [
        "--profile",
        "action-list-v3",
        "--expected-execution-sha256",
        signature["sha256"],
    ]


def output(sandbox):
    return next(
        path
        for path in (sandbox.root / "runs").glob(f"{profiles.ACTION_LIST.prefix}*")
        if path.is_dir()
    )


def action_project(tmp_path):
    root = synthetic_project(tmp_path)
    for name in set(profiles.ACTION_LIST_METHOD_FILES) - set(profiles.STRUCTURED_METHOD_FILES):
        (root / name).write_text("# synthetic action-list method source\n", encoding="utf-8")
    return root


def test_action_list_signature_keeps_both_historical_validation_contracts(tmp_path):
    root = action_project(tmp_path)
    v1 = execution_signature(root)
    v2 = profiles.structured_execution_signature(root)
    v3 = profiles.action_list_execution_signature(root)
    assert len(v3["files"]) == 15 and len(v2["files"]) == 13
    assert profiles.validate_action_list_execution_signature(v3) == v3["sha256"]
    (root / "src/growrag/experiments/operator_profiles.py").write_text("# new profile registration")
    assert execution_signature(root) == v1
    assert validate_execution_signature(v1) == v1["sha256"]
    assert profiles.validate_structured_execution_signature(v2) == v2["sha256"]
    assert profiles.structured_execution_signature(root)["sha256"] != v2["sha256"]
    assert profiles.validate_action_list_execution_signature(v3) == v3["sha256"]
    for validator, value in (
        (validate_execution_signature, v3),
        (profiles.validate_structured_execution_signature, v3),
        (profiles.validate_action_list_execution_signature, v2),
    ):
        with pytest.raises(ValueError):
            validator(value)


@pytest.mark.parametrize("name", profiles.ACTION_LIST_METHOD_FILES)
def test_each_action_list_method_dependency_is_pinned(tmp_path, name):
    root = action_project(tmp_path)
    before = profiles.action_list_execution_signature(root)
    (root / name).write_text("# changed pinned behavior\n", encoding="utf-8")
    assert profiles.action_list_execution_signature(root)["sha256"] != before["sha256"]


@pytest.mark.parametrize("alter", ["field", "schema", "configuration", "file", "digest", "sha"])
def test_action_list_signature_rejects_rehashed_unknown_contracts(alter):
    value = signature_body()
    if alter == "field":
        value["undeclared"] = True
    elif alter == "schema":
        value["schema"] = profiles.STRUCTURED_SCHEMA
    elif alter == "configuration":
        value["configuration"]["json_schema_mode"] = False
    elif alter == "file":
        value["files"].pop(profiles.ACTION_LIST_METHOD_FILES[-1])
    elif alter == "digest":
        value["files"][profiles.ACTION_LIST_METHOD_FILES[-1]] = "g" * 64
    value["sha256"] = (
        "0" * 64
        if alter == "sha"
        else study.fingerprint({key: value[key] for key in ("schema", "configuration", "files")})
    )
    with pytest.raises(ValueError):
        profiles.validate_action_list_execution_signature(value)


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
def test_action_list_formal_paths_stay_locked_before_input_read(
    tmp_path, monkeypatch, options, live
):
    monkeypatch.chdir(tmp_path)
    cmd = ["--manifest", "missing", "--profile", "action-list-v3", "--phase", "calibration"]
    if live:
        cmd.append("--allow-network")
    with pytest.raises(ValueError, match="calibration-only"):
        study.main(cmd + options)
    assert not (tmp_path / "runs").exists()


def test_action_list_dry_run_never_constructs_transport_or_loads_history(
    sandbox, action_list, capsys
):
    assert study.main(command(sandbox, action_list.signature, live=False)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["profile"] == profiles.ACTION_LIST.name
    assert result["protocol"] == profiles.ACTION_LIST.protocol
    assert result["run_id"].startswith(profiles.ACTION_LIST.prefix)
    assert sandbox.clients == sandbox.indexes == action_list.calls == []
    assert sandbox.history_calls == 0


def test_action_list_dispatch_keeps_inherited_budget_and_reader_audit(sandbox, action_list):
    sandbox.prior = 199.99
    assert study.main(command(sandbox, action_list.signature) + ["--project-cap-cny", "200"]) == 0
    root = output(sandbox)
    plan, budget = read(root / "launch_plan.json"), read(root / "final_budget.json")
    assert plan["protocol"] == profiles.ACTION_LIST.protocol
    assert plan["execution_signature"] == action_list.signature
    assert plan["profile_scope"] == "calibration_only_not_source_or_evaluation"
    assert plan["prior_reserved_cny"] == 199.99
    assert plan["project_cap_cny"] == 200 and plan["historical_authorized_total_cny"] == 50
    assert budget["limits"]["budget_cny"] == pytest.approx(0.01)
    assert budget["api_requests"] == 5
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "reader_record"]) == 3
    assert read(root / "predictions_frozen.json")["status"] == "completed"


def test_action_list_budget_exhaustion_seals_zero_network_attempts(sandbox, action_list):
    assert study.main(command(sandbox, action_list.signature) + ["--budget-cny", "0.0000001"]) == 1
    root = output(sandbox)
    budget = read(root / "final_budget.json")
    assert budget["api_requests"] == 0 and budget["block_reason"] == "estimated_budget_limit"
    assert sandbox.clients[0].attempts == 0
    assert read(root / "predictions_frozen.json")["status"] == "failed"
    assert (root.parent / f"{root.name}.claim.json").exists()


@pytest.mark.parametrize("registered", profiles.PROFILES)
@pytest.mark.parametrize("phase", ["source", "calibration", "evaluation"])
@pytest.mark.parametrize("kind", ["claim", "launch"])
def test_action_list_cannot_escape_any_version_role_or_arm_claim(
    sandbox, action_list, registered, phase, kind
):
    runs = sandbox.root / "runs"
    runs.mkdir()
    old = {
        "protocol": registered.protocol,
        "phase": phase,
        "arms": ["base"],
        "question_ids": sandbox.manifest["roles"]["calibration"][:1],
    }
    if kind == "claim":
        path = runs / f"{registered.prefix}old.claim.json"
    else:
        directory = runs / f"{registered.prefix}old"
        directory.mkdir()
        path = directory / "launch_plan.json"
    dump(path, old)
    with pytest.raises(ValueError, match="different phase|no automatic replay|offline audit"):
        study.main(command(sandbox, action_list.signature))
    assert read(path) == old and sandbox.clients == []


def test_legacy_cannot_replay_action_list_question(sandbox, action_list):
    assert study.main(command(sandbox, action_list.signature)) == 0
    with pytest.raises(ValueError, match="no automatic replay"):
        study.main(args(sandbox, count=1))
    assert len(sandbox.clients) == 1


def test_action_list_wrong_pin_never_creates_claim(sandbox, action_list):
    with pytest.raises(ValueError, match="signature mismatch"):
        study.main(
            command(sandbox, action_list.signature) + ["--expected-execution-sha256", "f" * 64]
        )
    assert sandbox.clients == []
    assert not list((sandbox.root / "runs").glob("*.claim.json"))


def test_pending_action_list_requests_block_project_history(tmp_path):
    budget_root(tmp_path, profiles.ACTION_LIST, pending=True)
    with pytest.raises(ValueError, match="unfinished"):
        history_runner.reviewed_history(tmp_path)


@pytest.mark.parametrize("scenario", ["stop", "create", "select", "too_many", "stale_stop_fields"])
def test_real_action_list_adapter_schema_and_runner_are_integrated_offline(
    sandbox, monkeypatch, scenario
):
    from growrag.experiments.operator_model import _CREATE_EXAMPLE, _SELECT_EXAMPLE
    from growrag.experiments.operator_model_v3 import encode_wire_plan
    from growrag.experiments.operator_schemas_v3 import PLANNER_VERSIONS, READER_VERSION

    signature = signature_body()
    monkeypatch.setattr(study, "action_list_execution_signature", lambda root: signature)
    fake_live = study.LiveChatClient

    class ActionListFakeLive(fake_live):
        def complete(self, messages, *, trace_id, prompt_version):
            response = super().complete(messages, trace_id=trace_id, prompt_version=prompt_version)
            if prompt_version == READER_VERSION:
                return response
            assert prompt_version in PLANNER_VERSIONS.values()
            payload = json.loads(messages[1]["content"])
            value = {
                "reason": "no_useful_query",
                "intent": "lookup",
                "constraints": [],
                "actions": [],
            }
            mode = payload["mode"]
            first = len(payload["previous_queries"]) == 1
            if first and (
                mode == "fresh"
                and scenario in {"create", "too_many"}
                or mode == "static"
                and scenario == "select"
            ):
                canonical = deepcopy(_CREATE_EXAMPLE if mode == "fresh" else _SELECT_EXAMPLE)
                canonical["reason"] = "missing_evidence"
                value = encode_wire_plan(canonical, mode=mode)
                if scenario == "too_many":
                    value["actions"] *= 2
            if first and mode == "fresh" and scenario == "stale_stop_fields":
                value["bindings"] = []
            return replace(response, content=json.dumps(value))

    monkeypatch.setattr(study, "LiveChatClient", ActionListFakeLive)
    fails = scenario in {"too_many", "stale_stop_fields"}
    assert study.main(command(sandbox, signature)) == int(fails)
    root = output(sandbox)
    budget = read(root / "final_budget.json")
    # The fixture index returns identical evidence: one real extra retrieval
    # stops as no_new_evidence, without paying for a second planner call.
    assert budget["api_requests"] == (2 if fails else 5)
    assert read(root / "predictions_frozen.json")["status"] == ("failed" if fails else "completed")
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    raw = [e for e in events if e["kind"] == "planner_record" and e.get("stage") == "raw_wire"]
    assert raw and all(e["prompt_version"] in PLANNER_VERSIONS.values() for e in raw)
    if scenario in {"create", "select"}:
        chosen = "fresh" if scenario == "create" else "static"
        report = read(root / "predictions.json")[0]["arms"][chosen]
        assert len(report["episode"]["searches"]) == 2
        assert report["episode"]["stop_reason"] == "no_new_evidence"
        assert report["episode"]["proposals"][0]["origin"] == chosen
