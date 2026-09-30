"""Profile identity and historical signatures; synthetic files only, no API/data."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from growrag.experiments import operator_profiles as profiles
from growrag.experiments import run_shared_s2g as history_runner
from growrag.experiments.operator_execution_signature import (
    METHOD_FILES,
    execution_signature,
    validate_execution_signature,
)
from growrag.experiments.representation_runner import fingerprint


def synthetic_project(tmp_path):
    for name in profiles.STRUCTURED_METHOD_FILES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# synthetic pinned source\n", encoding="utf-8")
    return tmp_path


def test_profiles_are_explicit_distinct_and_legacy_is_default():
    assert profiles.get_profile() == profiles.LEGACY
    assert profiles.get_profile("structured-v2") == profiles.STRUCTURED
    assert profiles.get_profile("action-list-v3") == profiles.ACTION_LIST
    assert profiles.LEGACY.protocol != profiles.STRUCTURED.protocol
    assert profiles.LEGACY.prefix != profiles.STRUCTURED.prefix
    assert len({p.protocol for p in profiles.PROFILES}) == 3
    assert len({p.prefix for p in profiles.PROFILES}) == 3


@pytest.mark.parametrize("name", [None, 1, "v2", "structured-v3", ""])
def test_unknown_profile_is_not_inferred(name):
    with pytest.raises(ValueError, match="unknown operator profile"):
        profiles.get_profile(name)


def test_new_signature_keeps_legacy_identity_and_validation_unchanged(tmp_path):
    root = synthetic_project(tmp_path)
    legacy = execution_signature(root)
    structured = profiles.structured_execution_signature(root)
    assert set(METHOD_FILES) < set(structured["files"])
    assert structured["configuration"]["json_schema_mode"] is True
    assert structured["configuration"]["json_object_mode"] is False
    assert legacy["configuration"]["json_object_mode"] is True
    assert "json_schema_mode" not in legacy["configuration"]
    assert profiles.validate_structured_execution_signature(structured) == structured["sha256"]
    for name in set(profiles.STRUCTURED_METHOD_FILES) - set(METHOD_FILES):
        (root / name).write_text("# changed structured-only implementation\n", encoding="utf-8")
    assert execution_signature(root) == legacy
    assert validate_execution_signature(legacy) == legacy["sha256"]
    assert profiles.structured_execution_signature(root)["sha256"] != structured["sha256"]
    assert profiles.validate_structured_execution_signature(structured) == structured["sha256"]
    with pytest.raises(ValueError):
        validate_execution_signature(structured)
    with pytest.raises(ValueError):
        profiles.validate_structured_execution_signature(legacy)


@pytest.mark.parametrize("name", profiles.STRUCTURED_METHOD_FILES)
def test_every_structured_dependency_changes_the_method_identity(tmp_path, name):
    root = synthetic_project(tmp_path)
    before = profiles.structured_execution_signature(root)
    (root / name).write_text("# one behavior dependency changed\n", encoding="utf-8")
    assert profiles.structured_execution_signature(root)["sha256"] != before["sha256"]


@pytest.mark.parametrize("alter", ["field", "schema", "configuration", "file", "digest", "sha"])
def test_structured_signature_rejects_forgery_even_with_rehashed_body(tmp_path, alter):
    value = deepcopy(profiles.structured_execution_signature(synthetic_project(tmp_path)))
    if alter == "field":
        value["undeclared"] = True
    elif alter == "schema":
        value["schema"] = "legacy"
    elif alter == "configuration":
        value["configuration"]["json_object_mode"] = True
    elif alter == "file":
        value["files"].pop(profiles.STRUCTURED_METHOD_FILES[-1])
    elif alter == "digest":
        value["files"][profiles.STRUCTURED_METHOD_FILES[-1]] = "g" * 64
    if alter != "sha":
        value["sha256"] = fingerprint({k: value[k] for k in ("schema", "configuration", "files")})
    else:
        value["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        profiles.validate_structured_execution_signature(value)


def test_structured_configuration_returns_a_copy():
    changed = profiles.structured_execution_configuration()
    changed["max_decisions"] = 99
    assert profiles.structured_execution_configuration()["max_decisions"] == 2


def options(**changed):
    return SimpleNamespace(
        **{
            "phase": "calibration",
            "arms": ["base", "fresh", "static"],
            "banks": None,
            "resume_certificate": None,
            "evaluation_freeze": None,
            "expected_freeze_sha256": None,
            "allow_network": False,
            "expected_execution_sha256": None,
            **changed,
        }
    )


@pytest.mark.parametrize(
    "changed",
    [
        {"phase": "source"},
        {"phase": "evaluation"},
        {"arms": ["memory50"]},
        {"banks": "missing"},
        {"resume_certificate": "missing"},
        {"evaluation_freeze": "missing"},
        {"expected_freeze_sha256": "0" * 64},
    ],
)
def test_structured_scope_is_not_formal_source_or_evaluation(changed):
    with pytest.raises(ValueError, match="calibration-only"):
        profiles.validate_profile_options(profiles.STRUCTURED, options(**changed))


def budget_root(tmp_path, profile, *, pending=False, protocol=None):
    root = tmp_path / f"{profile.prefix}calibration_base_0008_0009"
    root.mkdir()
    (root / "launch_plan.json").write_text(
        json.dumps({"protocol": protocol or profile.protocol, "model": history_runner.PILOT_MODEL})
    )
    if pending:
        journal = root / "request_journal"
        journal.mkdir()
        (journal / "0000_intent.json").write_text("{}")
    else:
        (root / "final_budget.json").write_text(json.dumps({"api_requests": 1, "calls": [1]}))
    return root


def test_project_history_includes_all_profile_ledgers_without_reset(tmp_path, monkeypatch):
    roots = [budget_root(tmp_path, profile) for profile in profiles.PROFILES]
    captured = {}

    def reconcile(runs, *, reviewed_extra_ledgers):
        captured["roots"] = reviewed_extra_ledgers
        return {"prior_reserved_cny": 12.5, "authorized_total_cny": 50.0}

    monkeypatch.setattr(history_runner, "reconcile_history", reconcile)
    result = history_runner.reviewed_history(tmp_path)
    assert result == {"prior_reserved_cny": 12.5, "authorized_total_cny": 50.0}
    for root in roots:
        assert captured["roots"].count(f"{root.name}/final_budget.json") == 1


def test_pending_structured_cost_blocks_shared_history(tmp_path):
    budget_root(tmp_path, profiles.STRUCTURED, pending=True)
    with pytest.raises(ValueError, match="unfinished"):
        history_runner.reviewed_history(tmp_path)


def test_structured_prefix_cannot_hide_legacy_protocol(tmp_path):
    budget_root(tmp_path, profiles.STRUCTURED, protocol=profiles.LEGACY.protocol)
    with pytest.raises(ValueError, match="unreviewed"):
        history_runner.reviewed_history(tmp_path)
