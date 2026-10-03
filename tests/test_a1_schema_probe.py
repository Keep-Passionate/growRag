"""One-call runner checks using temporary files and a fake HTTP opener only."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import a1_schema_probe as probe
from growrag.experiments import api_client


class OfflineResponse:
    status = 200
    headers = {"x-request-id": "00000000-1111-2222-3333-444444444444"}

    def __init__(self, content, *, model=probe.PILOT_MODEL, usage=None):
        self.raw = json.dumps(
            {
                "id": "test-only-response",
                "model": model,
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": usage
                if usage is not None
                else {"prompt_tokens": 10, "completion_tokens": 20},
            }
        ).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.raw if size < 0 else self.raw[:size]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    monkeypatch.chdir(probe.CODE_PROJECT)
    monkeypatch.setattr(
        probe, "_git_state", lambda: {"commit": "test-only", "worktree_dirty": False}
    )
    history = {
        "prior_reserved_cny": 53.7234394,
        "prior_known_estimated_cny": 7.7970522,
        "prior_unknown_cost_requests": 20,
        "prior_total_actual_cny": None,
    }
    seen = []

    def reconciliation(runs, **kwargs):
        assert Path(runs) == tmp_path / "runs"
        assert kwargs == {"reviewed_other_series": {probe.PREFIX: frozenset({probe.PROTOCOL})}}
        return history

    monkeypatch.setattr(probe, "reviewed_history", reconciliation)
    monkeypatch.setattr(
        probe,
        "read_local_bailian_settings",
        lambda path: SimpleNamespace(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="offline-test-secret",
        ),
    )

    def opener(content, *, model=probe.PILOT_MODEL, usage=None):
        def open_request(request, timeout):
            seen.append(json.loads(request.data))
            return OfflineResponse(content, model=model, usage=usage)

        monkeypatch.setattr(
            api_client.urllib.request,
            "build_opener",
            lambda *args: SimpleNamespace(open=open_request),
        )

    freeze = probe.freeze(tmp_path)
    return tmp_path, freeze["freeze_sha256"], opener, seen, history


def wire():
    _, prepared, _ = probe.probe_input()
    return json.dumps(
        {
            "locations": [
                {"kind": kind, "slot": slot, "candidate_ref_ids": []}
                for kind, slot in prepared.semantic_checks
            ]
        }
    )


def test_dry_run_never_reads_secret_or_connects(setup, monkeypatch):
    project, sha, _, seen, _ = setup
    monkeypatch.setattr(
        probe, "read_local_bailian_settings", lambda path: pytest.fail("no secret on dry run")
    )
    assert probe.run(project, sha)["api_calls"] == 0
    assert not seen and not (project / probe.OUTPUT).exists()


def test_probe_uses_strict_one_call_carries_history_and_audits_readonly(setup):
    project, sha, opener, seen, history = setup
    opener(wire())
    result = probe.run(project, sha, allow_network=True)
    assert result["status"] == "completed" and len(seen) == 1
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["response_format"]["json_schema"]["strict"] is True
    assert probe._read(project / probe.OUTPUT / "prior_budget.json") == history
    before = {str(p): p.read_bytes() for p in project.rglob("*") if p.is_file()}
    audit = probe.audit(project, sha, result["terminal_sha256"])
    assert audit["status"] == "format_compatibility_verified"
    assert audit["api_calls"] == audit["files_written"] == 0
    assert not audit["natural_prepare_allowed"]
    assert audit["totals"]["api_requests"] == 1
    assert before == {str(p): p.read_bytes() for p in project.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="claimed"):
        probe.run(project, sha, allow_network=True)
    assert len(seen) == 1


def test_format_failure_is_preserved_without_retry_and_audited(setup):
    project, sha, opener, seen, _ = setup
    opener('{"locations":{}}')
    result = probe.run(project, sha, allow_network=True)
    assert result["status"] == "model_output_error" and len(seen) == 1
    audit = probe.audit(project, sha, result["terminal_sha256"])
    assert audit["status"] == "probe_failed_audited" and not audit["format_valid"]


def test_model_drift_stops_and_preserves_cost(setup):
    project, sha, opener, seen, _ = setup
    opener(wire(), model="wrong-model")
    result = probe.run(project, sha, allow_network=True)
    assert result["status"] == "stopped" and len(seen) == 1
    assert probe.audit(project, sha, result["terminal_sha256"])["status"] == "probe_failed_audited"


def test_modified_schema_freeze_is_rejected_before_network(setup, monkeypatch):
    project, sha, opener, seen, _ = setup
    opener(wire())
    monkeypatch.setattr(probe, "response_format_for", lambda version: {"type": "json_object"})
    with pytest.raises(ValueError, match="changed"):
        probe.run(project, sha, allow_network=True)
    assert not seen


def test_tampered_terminal_cannot_be_resealed_implicitly(setup):
    project, sha, opener, _, _ = setup
    opener(wire())
    result = probe.run(project, sha, allow_network=True)
    path = project / probe.OUTPUT / "TERMINAL.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA"):
        probe.audit(project, sha, result["terminal_sha256"])


def test_frozen_code_and_artifact_roots_are_distinct(setup):
    project, sha, _, _, _ = setup
    frozen = probe.load_freeze(project, sha)
    assert frozen["code_project"] == str(probe.CODE_PROJECT)
    assert frozen["artifact_project"] == str(project)
    assert frozen["code_project"] != frozen["artifact_project"]
    assert frozen["messages_sha256"] == probe.fingerprint(probe.probe_input()[2])
    assert frozen["configuration"]["max_calls"] == 1
    assert frozen["configuration"]["project_cap_cny"] is None
    assert len(hashlib.sha256(json.dumps(frozen).encode()).hexdigest()) == 64


@pytest.mark.parametrize(
    "artifact,field,value",
    [
        ("probe.json", "row_id", "unrelated-row"),
        ("probe.json", "status", "stopped"),
        ("launch_plan.json", "prior_reserved_cny", 0.0),
        ("launch_plan.json", "code_project", "wrong-code-root"),
    ],
)
def test_resealed_wrong_metadata_is_not_a_valid_probe(setup, artifact, field, value):
    project, sha, opener, _, _ = setup
    opener(wire())
    probe.run(project, sha, allow_network=True)
    directory = project / probe.OUTPUT
    artifact_path = directory / artifact
    changed = probe._read(artifact_path)
    changed[field] = value
    # Deliberate tampering is confined to this test's temporary fake archive.
    artifact_path.write_text(json.dumps(changed), encoding="utf-8")
    terminal_path = directory / "TERMINAL.json"
    terminal = probe._read(terminal_path)
    seal_key = "record_sha256" if artifact == "probe.json" else "plan_sha256"
    terminal[seal_key] = probe._sha(artifact_path)
    if artifact == "launch_plan.json":
        claim = project / "runs" / (probe.RUN_ID + ".claim.json")
        claim.write_text(
            json.dumps({**changed, "plan_sha256": probe.fingerprint(changed)}), encoding="utf-8"
        )
        terminal["claim_sha256"] = probe._sha(claim)
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        probe.audit(project, sha, probe._sha(terminal_path))
