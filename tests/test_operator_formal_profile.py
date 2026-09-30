"""Formal v3 runner integration with synthetic transport, never provider calls."""

import json
from dataclasses import replace

import pytest
from test_operator_profiles import options
from test_operator_v2_calibration import structured as structured
from test_operator_v3_calibration import action_list as action_list
from test_operator_v3_calibration import action_project, output, signature_body
from test_run_operator_study import args, dump, evaluation_args, read
from test_run_operator_study import evaluation as evaluation
from test_run_operator_study import sandbox as sandbox

from growrag.experiments import operator_evaluation_freeze as freeze
from growrag.experiments import operator_profiles as profiles
from growrag.experiments import run_operator_study as study
from growrag.operator_bank import FrozenOperatorBank


def test_formal_dispatch_preserves_separate_historical_signature_contracts(tmp_path):
    project = action_project(tmp_path)
    for profile in (profiles.LEGACY, profiles.ACTION_LIST):
        assert profiles.profile_from_protocol(profile.protocol) == profile
        assert profiles.study_profile(profile) == profile
        signature = profiles.profile_execution_signature(project, profile.name)
        assert (
            profiles.validate_profile_execution_signature(signature, profile.name)
            == signature["sha256"]
        )
        other = profiles.ACTION_LIST if profile == profiles.LEGACY else profiles.LEGACY
        with pytest.raises(ValueError):
            profiles.validate_profile_execution_signature(signature, other.name)
    with pytest.raises(ValueError, match="calibration-only"):
        profiles.profile_execution_signature(project, profiles.STRUCTURED.name)
    with pytest.raises(ValueError, match="unknown operator protocol"):
        profiles.profile_from_protocol("action-list-v3")
    with pytest.raises(ValueError, match="unregistered"):
        profiles.study_profile(replace(profiles.ACTION_LIST, prefix="unreviewed"))


@pytest.mark.parametrize("phase", ["source", "calibration", "evaluation"])
def test_v3_formal_scope_still_requires_explicit_method_before_network(phase):
    selected = options(phase=phase, resume_certificate="requires-separate-validation")
    profiles.validate_profile_options(profiles.ACTION_LIST, selected)
    selected.allow_network = True
    with pytest.raises(ValueError, match="explicit expected execution"):
        profiles.validate_profile_options(profiles.ACTION_LIST, selected)
    selected.expected_execution_sha256 = "c" * 64
    profiles.validate_profile_options(profiles.ACTION_LIST, selected)


def test_formal_source_uses_v3_and_keeps_feedback_sealed(sandbox, action_list):
    command = args(sandbox, phase="source", count=2) + [
        "--profile",
        profiles.ACTION_LIST.name,
        "--expected-execution-sha256",
        action_list.signature["sha256"],
    ]
    assert study.main(command) == 0
    run = output(sandbox)
    plan = read(run / "launch_plan.json")
    reports = read(run / "predictions.json")
    assert plan["question_ids"] == sandbox.manifest["roles"]["source"][:2]
    assert plan["protocol"] == profiles.ACTION_LIST.protocol
    assert plan["execution_signature"] == action_list.signature
    assert read(run / "final_budget.json")["api_requests"] == 10
    assert all(
        arm["status"] == "completed" and arm["feedback"] is None and not arm["memory_updated"]
        for report in reports
        for arm in report["arms"].values()
    )
    assert read(run / "predictions_frozen.json")["gold_loaded"] is False
    with pytest.raises(ValueError, match="no automatic replay"):
        study.main(command)


def test_certified_v3_evaluation_uses_all_arms_and_exact_fresh_empty_bank_fallback(
    sandbox, evaluation, monkeypatch
):
    """This tests runner wiring; actual certificate provenance is tested separately."""
    from growrag.experiments.operator_schemas_v3 import PLANNER_VERSIONS, READER_VERSION

    signature = signature_body()
    monkeypatch.setattr(study, "action_list_execution_signature", lambda root: signature)
    monkeypatch.setattr(study, "profile_execution_signature", lambda root, profile: signature)
    evaluation.body.update(protocol=profiles.ACTION_LIST.protocol, execution_signature=signature)
    evaluation.body["expected"]["execution"] = signature["sha256"]
    for size in study.SOURCE_SIZES:
        bank = FrozenOperatorBank(
            profiles.ACTION_LIST.protocol,
            tuple(sandbox.manifest["nested_source_ids"][str(size)]),
            (),
        )
        path = evaluation.banks / f"bank_{size}.json"
        path.write_text(bank.to_json(), encoding="utf-8")
        evaluation.body["banks"][str(size)] = {
            "file_sha256": study._sha(path),
            "fingerprint": bank.fingerprint,
        }
    validate = freeze.validate_evaluation_freeze

    def v3_validate(root, path, *, expected_certificate_sha256, profile):
        assert profile == profiles.ACTION_LIST.name
        return validate(root, path, expected_certificate_sha256=expected_certificate_sha256)

    monkeypatch.setattr(freeze, "validate_evaluation_freeze", v3_validate)
    fake_live = study.LiveChatClient

    class ActionListFakeLive(fake_live):
        def complete(self, messages, *, trace_id, prompt_version):
            response = super().complete(messages, trace_id=trace_id, prompt_version=prompt_version)
            if prompt_version == READER_VERSION:
                return response
            assert prompt_version in PLANNER_VERSIONS.values()
            return replace(
                response,
                content=json.dumps(
                    {
                        "reason": "no_useful_query",
                        "intent": "lookup",
                        "constraints": [],
                        "actions": [],
                    }
                ),
            )

    monkeypatch.setattr(study, "LiveChatClient", ActionListFakeLive)
    command = evaluation_args(sandbox, evaluation) + [
        "--profile",
        profiles.ACTION_LIST.name,
        "--expected-execution-sha256",
        signature["sha256"],
    ]
    assert study.main(command) == 0
    assert evaluation.validations == [False, True, True]
    run = output(sandbox)
    report = read(run / "predictions.json")[0]
    assert list(report["arms"]) == list(study.ARMS)
    messages = sandbox.clients[0].messages
    assert len(messages) == 13
    # BASE reader, FRESH planner/reader, STATIC planner/reader, then four memories.
    for index, arm in enumerate(study.ARMS[3:]):
        assert messages[5 + 2 * index] == messages[1]
        assert report["arms"][arm]["memory_context"]["effective_planner_mode"] == "fresh"
    assert all(not value["memory_updated"] for value in report["arms"].values())


def test_old_source_claim_is_not_released_by_v3_protocol_name(sandbox, action_list):
    runs = sandbox.root / "runs"
    runs.mkdir()
    dump(
        runs / f"{profiles.LEGACY.prefix}source-old.claim.json",
        {
            "protocol": profiles.LEGACY.protocol,
            "phase": "source",
            "arms": ["base"],
            "question_ids": sandbox.manifest["roles"]["source"][:1],
        },
    )
    command = args(sandbox, phase="source", count=1) + [
        "--profile",
        profiles.ACTION_LIST.name,
        "--expected-execution-sha256",
        action_list.signature["sha256"],
    ]
    with pytest.raises(ValueError, match="no automatic replay"):
        study.main(command)
    assert sandbox.clients == []
