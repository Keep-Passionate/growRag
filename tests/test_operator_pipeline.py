"""Coordinator safety checks use fake subprocesses only; never API or shutdown."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_operator_pipeline.py"
SPEC = importlib.util.spec_from_file_location("operator_pipeline", PATH)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def args(*extra):
    return pipeline.parser().parse_args(
        [
            "--manifest",
            "data/manifest.json",
            "--expected-manifest-sha256",
            "a" * 64,
            "--expected-execution-sha256",
            "b" * 64,
            "--source-start",
            "81",
            "--output-dir",
            "runs/pipeline",
            *extra,
        ]
    )


def test_default_has_no_side_effects(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("dry run attempted a side effect")

    monkeypatch.setattr(pipeline, "sha", forbidden)
    monkeypatch.setattr(pipeline.subprocess, "run", forbidden)
    assert pipeline.run(args("--shutdown-on-complete", "--sync-github"), tmp_path) == 0
    assert list(tmp_path.iterdir()) == []


def test_commands_use_existing_gates_and_supervisors(tmp_path):
    config = args("--source-resume-certificate", "runs/resume.json")
    source = pipeline.stage_command(config, tmp_path, "source500", tmp_path / "stop.flag")
    assert source[source.index("--start") + 1] == "81"
    assert source[source.index("--end") + 1] == "500"
    assert Path(source[source.index("--resume-certificate") + 1]) == Path("runs/resume.json")
    score = pipeline.stage_command(config, tmp_path, "source_score", tmp_path / "stop.flag")
    assert "growrag.experiments.score_operator_sources" in score
    assert "--score-new" in score
    freeze = tmp_path / pipeline.FREEZE
    freeze.parent.mkdir()
    freeze.write_text("{}", encoding="utf-8")
    evaluation = pipeline.stage_command(config, tmp_path, "evaluation500", tmp_path / "stop.flag")
    assert "scripts/run_operator_evaluation_remaining.py" in evaluation
    assert evaluation[evaluation.index("--expected-freeze-sha256") + 1] == pipeline.sha(freeze)


def test_paths_cannot_escape_workspace(tmp_path):
    with pytest.raises(ValueError):
        pipeline.inside(tmp_path, "../outside")


@pytest.mark.parametrize(
    "project_remaining, batch_cap, expected", [(0.02, 0.02, True), (10, 2, False)]
)
def test_budget_stop_distinguishes_project_from_batch(
    tmp_path, project_remaining, batch_cap, expected
):
    directory = tmp_path / "runs/2026-09-30_operator_v3_source_test"
    directory.mkdir(parents=True)
    (directory / "launch_plan.json").write_text(
        json.dumps(
            {
                "manifest_sha256": "a" * 64,
                "prior_reserved_cny": 200 - project_remaining,
                "subcap_cny": batch_cap,
            }
        ),
        encoding="utf-8",
    )
    (directory / "final_budget.json").write_text(
        json.dumps(
            {
                "block_reason": "estimated_budget_limit",
            }
        ),
        encoding="utf-8",
    )
    assert pipeline.project_budget_stopped(tmp_path, args(), set()) is expected
    assert not pipeline.project_budget_stopped(tmp_path, args(), {directory.name})


def test_interrupted_wait_requests_boundary_stop_and_reaps_child(tmp_path, monkeypatch):
    class Child:
        waited = 0

        def wait(self):
            self.waited += 1
            if self.waited == 1:
                raise KeyboardInterrupt
            return 0

        def poll(self):
            return None if self.waited < 2 else 0

    child = Child()
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **k: child)
    with pytest.raises(KeyboardInterrupt):
        pipeline.execute(tmp_path, tmp_path, "fake", ["fake"])
    assert child.waited == 2
    assert (tmp_path / "stop_requested.flag").exists()
    assert not (tmp_path / "quota_stop.flag").exists()


@pytest.mark.parametrize("outcome", ["completed", "technical_stop", "quota_stop"])
def test_finish_only_shuts_down_for_authorized_conditions(tmp_path, monkeypatch, outcome):
    (tmp_path / "runs").mkdir()
    monkeypatch.setattr(pipeline, "sha", lambda p: "a" * 64)
    monkeypatch.setattr(pipeline, "action_list_execution_signature", lambda p: {"sha256": "b" * 64})
    calls = []

    def subprocess_run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", subprocess_run)
    monkeypatch.setattr(pipeline, "stage_command", lambda *a: ["fake"])
    monkeypatch.setattr(pipeline, "budget_snapshot", lambda p: {"prior_reserved_cny": 10})
    monkeypatch.setattr(pipeline, "public_report", lambda *a: "safe aggregate report\n")
    monkeypatch.setattr(pipeline, "sync_public", lambda *a: calls.append(["sync_public"]))
    # Replace the shutdown boundary itself, not os.name (which would affect pathlib).
    monkeypatch.setattr(
        pipeline, "request_shutdown", lambda output, result: calls.append(["shutdown"])
    )

    def execute(root, output, stage, argv):
        if outcome == "quota_stop":
            (output / "quota_stop.flag").write_text("quota", encoding="utf-8")
            (output / "stop_requested.flag").write_text("stop", encoding="utf-8")
            return 2
        return 1 if outcome == "technical_stop" else 0

    monkeypatch.setattr(pipeline, "execute", execute)
    config = args(
        "--allow-network", "--sync-github", "--shutdown-on-complete", "--shutdown-on-quota-stop"
    )
    assert pipeline.run(config, tmp_path) == (0 if outcome == "completed" else 1)
    result = json.loads(
        (tmp_path / "runs/pipeline/pipeline_result.json").read_text(encoding="utf-8")
    )
    assert result["status"] == outcome
    assert ["sync_public"] in calls
    assert (["shutdown"] in calls) is (outcome != "technical_stop")
    if outcome != "technical_stop":
        assert calls.index(["sync_public"]) < calls.index(["shutdown"])


def test_shutdown_is_non_forced(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "WINDOWS", True)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda argv, **kw: calls.append(argv))
    pipeline.request_shutdown(tmp_path, {"status": "completed"})
    assert calls == [["shutdown.exe", "/s", "/t", "0"]]
    assert json.loads((tmp_path / "shutdown_requested.json").read_text()) == {
        "status": "completed",
        "forced": False,
    }
