"""No network or gold: method changes must not hide behind a new runner commit."""

from copy import deepcopy

import pytest

from growrag.experiments.operator_execution_signature import (
    METHOD_FILES,
    execution_configuration,
    execution_signature,
    validate_execution_signature,
)


def _project(tmp_path):
    for name in METHOD_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic method source\n", encoding="utf-8")
    return tmp_path


def test_signature_stable_but_method_change_detected(tmp_path):
    project = _project(tmp_path)
    before = execution_signature(project)
    assert validate_execution_signature(before) == before["sha256"]
    assert before == execution_signature(project)
    (project / METHOD_FILES[0]).write_text("# changed grammar\n", encoding="utf-8")
    assert execution_signature(project)["sha256"] != before["sha256"]
    assert validate_execution_signature(before) == before["sha256"]


def test_runner_only_change_does_not_redefine_method(tmp_path):
    project = _project(tmp_path)
    before = execution_signature(project)
    (project / "src/growrag/experiments/run_operator_study.py").write_text(
        "# resume plumbing\n", encoding="utf-8"
    )
    assert execution_signature(project) == before


@pytest.mark.parametrize("key", ["schema", "configuration", "files", "sha256"])
def test_incomplete_signature_rejected(tmp_path, key):
    value = execution_signature(_project(tmp_path))
    del value[key]
    with pytest.raises(ValueError):
        validate_execution_signature(value)


def test_forged_field_hash_and_configuration_rejected(tmp_path):
    value = execution_signature(_project(tmp_path))
    bad = deepcopy(value)
    bad["files"][METHOD_FILES[0]] = "g" * 64
    with pytest.raises(ValueError):
        validate_execution_signature(bad)
    value["configuration"]["retrieval_budget"] = 7
    with pytest.raises(ValueError):
        validate_execution_signature(value)


def test_configuration_returns_independent_copy():
    first = execution_configuration()
    first["retrieval_budget"] = 100
    assert execution_configuration()["retrieval_budget"] == 3
