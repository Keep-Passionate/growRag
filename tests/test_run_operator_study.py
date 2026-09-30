"""Synthetic serial-runner tests. No provider, dataset or sealed evaluation reads."""

import json
import os
from types import SimpleNamespace

import pytest

from growrag.experiments import run_operator_study as study
from growrag.experiments.api_client import APIRequestError, ChatResponse
from growrag.experiments.operator_execution_signature import (
    METHOD_FILES,
)
from growrag.experiments.operator_execution_signature import (
    SCHEMA as SIGNATURE_SCHEMA,
)
from growrag.experiments.operator_model import PLANNER_VERSION
from growrag.experiments.operator_resume import build_certificate
from growrag.operator_bank import FrozenOperatorBank, OperatorRecord


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def seal(path, manifest):
    dump(path, manifest)
    path.with_suffix(".sha256").write_text(f"{study._sha(path)}  manifest.json\n", encoding="ascii")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "data"
    bundle.mkdir()
    source = [f"{i:024x}" for i in range(500)]
    calibration = [f"{i:024x}" for i in range(500, 600)]
    evaluation = [f"{i:024x}" for i in range(600, 1100)]
    roles = {"source": source, "calibration": calibration, "evaluation": evaluation}
    artifacts = {}
    for role in ("source", "calibration"):
        path = bundle / f"{role}_runtime_questions.jsonl"
        rows = [
            {"question_id": qid, "text": f"Synthetic question {qid}", "dataset": f"fixture-{role}"}
            for qid in roles[role]
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        artifacts[path.name] = {
            "sha256": study._sha(path),
            "rows": len(rows),
            "contains_gold": False,
            "runtime_safe": True,
        }
    corpus = bundle / "corpus.jsonl"
    corpus.write_text(
        '{"doc_id":"fixture","title":"Fixture","sentences":["Text"]}\n', encoding="utf-8"
    )
    artifacts[corpus.name] = {
        "sha256": study._sha(corpus),
        "rows": 1,
        "contains_gold": False,
        "runtime_safe": True,
    }
    manifest = {
        "schema_version": study.SCHEMA,
        "official_splits": {"source": "train", "calibration": "train", "evaluation": "dev"},
        "roles": roles,
        "counts": dict(study.COUNTS),
        "source_sizes": list(study.SOURCE_SIZES),
        "nested_source_ids": {str(n): source[:n] for n in study.SOURCE_SIZES},
        "official_test_used": False,
        "evaluation_memory_updates_allowed": False,
        "calibration_memory_updates_allowed": False,
        "gold_projected": False,
        "artifacts": artifacts,
    }
    path = bundle / "manifest.json"
    seal(path, manifest)
    state = SimpleNamespace(
        root=tmp_path,
        path=path,
        manifest=manifest,
        clients=[],
        indexes=[],
        history_calls=0,
        prior=4.0,
        on_complete=None,
        failure_at=None,
        malformed_at=None,
        close_fail=False,
    )

    class FakeLive:
        def __init__(self, config, audit, *, allow_network):
            self.config, self.audit = config, audit
            audit.mkdir(parents=True)
            self.attempts = 0
            self.messages = []
            self.transport_source = "synthetic_test_double_not_api"
            state.clients.append(self)
            assert allow_network is True
            assert os.environ[study.KEY_VARIABLE] == "synthetic-test-secret"

        def complete(self, messages, *, trace_id, prompt_version):
            self.attempts += 1
            assert self.attempts <= self.config.max_calls
            self.messages.append(messages)
            audit = self.audit / f"{self.attempts}.json"
            dump(audit, {"trace_id": trace_id})
            if state.failure_at == self.attempts:
                raise APIRequestError(
                    "synthetic failure",
                    api_requests=1,
                    input_tokens=10,
                    output_tokens=2,
                    audit_path=audit,
                )
            if state.on_complete:
                state.on_complete(self)
            value = (
                {
                    "decision": "stop",
                    "reason": "fixture",
                    "intent": "lookup",
                    "constraints": [],
                    "selected_operator": None,
                    "operator": None,
                    "gap": {},
                    "bindings": [],
                }
                if prompt_version == PLANNER_VERSION
                else {"answer": "", "supported": False, "evidence_ids": []}
            )
            content = "{}" if state.malformed_at == self.attempts else json.dumps(value)
            return ChatResponse(
                content,
                self.config.model,
                self.config.model,
                "response",
                "request",
                10,
                2,
                0.01,
                audit,
                self.transport_source,
            )

    class FakeIndex:
        def __init__(self, *args, **kwargs):
            self.calls, self.closed = [], False
            state.indexes.append(self)

        def __call__(self, query, top_k):
            self.calls.append((query, top_k))
            return (SimpleNamespace(doc_id="doc", title="Fixture", text="Current evidence."),)

        def close(self):
            self.closed = True
            if state.close_fail:
                raise OSError("synthetic close failure")

    def history(runs):
        state.history_calls += 1
        assert (runs / ".operator-study.lock").exists()
        return {"prior_reserved_cny": state.prior}

    monkeypatch.setattr(study, "LiveChatClient", FakeLive)
    monkeypatch.setattr(study, "SharedBM25Index", FakeIndex)
    monkeypatch.setattr(study, "reviewed_history", history)
    monkeypatch.setattr(study, "_git_state", lambda: {"commit": "f" * 40, "worktree_dirty": False})
    monkeypatch.setattr(study, "source_snapshot", lambda root: {"sha256": "a" * 64, "files": {}})
    method = {
        "schema": SIGNATURE_SCHEMA,
        "configuration": study.execution_configuration(),
        "files": {name: "a" * 64 for name in METHOD_FILES},
    }
    monkeypatch.setattr(
        study, "execution_signature", lambda root: {**method, "sha256": study.fingerprint(method)}
    )
    monkeypatch.setattr(
        study,
        "read_local_bailian_settings",
        lambda path: SimpleNamespace(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="synthetic-test-secret",
        ),
    )
    return state


def args(state, *, phase="calibration", count=2, arms=("base", "fresh", "static"), live=True):
    result = [
        "--manifest",
        str(state.path),
        "--phase",
        phase,
        "--count",
        str(count),
        "--arms",
        *arms,
    ]
    if live:
        result.extend(
            [
                "--allow-network",
                "--api-config",
                "synthetic-config.md",
                "--expected-manifest-sha256",
                study._sha(state.path),
            ]
        )
    return result


def run_dir(state):
    return next(path for path in (state.root / "runs").glob(f"{study.PREFIX}*") if path.is_dir())


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_bank(state, *, protocol=study.PROTOCOL, published=True, source_ids=None):
    directory = state.root / "banks"
    directory.mkdir()
    ids = tuple(source_ids or state.manifest["nested_source_ids"]["50"])
    record = OperatorRecord(
        study.seed_specs()[0],
        (ids[0],),
        protocol,
        "source-audit",
        "validated" if published else "candidate",
        "validation-audit" if published else None,
    )
    bank = FrozenOperatorBank(protocol, ids, (record,))
    path = directory / "bank_50.json"
    path.write_text(bank.to_json(), encoding="utf-8")
    return directory, path


def test_dry_run_never_reads_keys_history_index_or_network(sandbox):
    assert study.main(args(sandbox, live=False)) == 0
    assert sandbox.clients == sandbox.indexes == []
    assert sandbox.history_calls == 0
    assert not (sandbox.root / "runs").exists()


def test_two_question_main_has_immutable_checkpoints_and_budget(sandbox, monkeypatch, capsys):
    monkeypatch.setenv(study.KEY_VARIABLE, "previous-private-value")
    assert study.main(args(sandbox)) == 0
    output = run_dir(sandbox)
    assert len(list(output.glob("checkpoint_*.json"))) == 2
    reports = read(output / "predictions.json")
    assert len(reports) == 2
    assert all(
        a["feedback"] is None and not a["memory_updated"]
        for r in reports
        for a in r["arms"].values()
    )
    plan = read(output / "launch_plan.json")
    assert plan["max_calls"] == 14  # 2 * (BASE1 + FRESH3 + STATIC3)
    assert sandbox.clients[0].attempts == 10  # Fixture planners immediately stop.
    assert sandbox.clients[0].config.max_output_tokens == 2048
    assert sandbox.clients[0].config.json_object_mode is True
    assert len(list((output / "request_journal").glob("*_intent.json"))) == 10
    assert read(output / "final_budget.json")["api_requests"] == 10
    assert read(output / "cumulative_budget.json")["cumulative_reserved_cny"] < 50
    assert read(output / "predictions_frozen.json")["status"] == "completed"
    assert sandbox.indexes[0].closed
    assert os.environ[study.KEY_VARIABLE] == "previous-private-value"
    assert not (sandbox.root / "runs" / ".operator-study.lock").exists()
    assert "synthetic-test-secret" not in capsys.readouterr().out
    assert not (sandbox.path.parent / "evaluation_runtime_questions.jsonl").exists()
    assert not any(
        "gold" in str(message).lower() for c in sandbox.clients for message in c.messages
    )


@pytest.mark.parametrize("live", [False, True])
def test_evaluation_gate_precedes_all_file_access(tmp_path, monkeypatch, live):
    monkeypatch.chdir(tmp_path)
    command = ["--manifest", "NONEXISTENT/manifest.json", "--phase", "evaluation"]
    if live:
        command.append("--allow-network")
    with pytest.raises(ValueError, match="evaluation locked"):
        study.main(command)
    with pytest.raises(ValueError, match="evaluation locked"):
        study.load_inputs(tmp_path / "nonexistent", "evaluation")


@pytest.mark.parametrize(
    "field",
    [
        "gold_projected",
        "official_test_used",
        "evaluation_memory_updates_allowed",
        "calibration_memory_updates_allowed",
    ],
)
def test_manifest_permission_flags_fail_closed(sandbox, field):
    sandbox.manifest[field] = True
    seal(sandbox.path, sandbox.manifest)
    with pytest.raises(ValueError, match="manifest required"):
        study.main(args(sandbox))
    assert sandbox.clients == []


@pytest.mark.parametrize("alter", ["role_overlap", "nested", "count", "path_id", "official_split"])
def test_manifest_role_and_source_prefix_validation(sandbox, alter):
    manifest = sandbox.manifest
    if alter == "role_overlap":
        manifest["roles"]["evaluation"][0] = manifest["roles"]["source"][0]
    elif alter == "nested":
        manifest["nested_source_ids"]["50"] = manifest["nested_source_ids"]["50"][::-1]
    elif alter == "count":
        manifest["counts"]["calibration"] = 99
    elif alter == "path_id":
        manifest["roles"]["calibration"][0] = "../../escape"
    else:
        manifest["official_splits"]["evaluation"] = "train"
    seal(sandbox.path, manifest)
    with pytest.raises(ValueError):
        study.main(args(sandbox))
    assert sandbox.clients == []


def test_runtime_gold_field_rejected_even_with_updated_file_hash(sandbox):
    path = sandbox.path.parent / "calibration_runtime_questions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["answer"] = "forbidden label"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    sandbox.manifest["artifacts"][path.name]["sha256"] = study._sha(path)
    seal(sandbox.path, sandbox.manifest)
    with pytest.raises(ValueError, match="unallowlisted"):
        study.main(args(sandbox))
    assert sandbox.clients == []


def test_manifest_sidecar_and_artifact_hash_are_checked(sandbox):
    sandbox.path.with_suffix(".sha256").write_text("bad sidecar", encoding="ascii")
    with pytest.raises(ValueError, match="sidecar"):
        study.load_inputs(sandbox.path, "source")
    seal(sandbox.path, sandbox.manifest)
    (sandbox.path.parent / "source_runtime_questions.jsonl").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        study.load_inputs(sandbox.path, "source")


def test_artifact_cannot_escape_or_mark_labels_runtime_safe(sandbox):
    with pytest.raises(ValueError, match="local"):
        study.artifact(sandbox.path.parent, sandbox.manifest, "../gold.jsonl")
    sandbox.manifest["artifacts"]["source_runtime_questions.jsonl"]["contains_gold"] = True
    with pytest.raises(ValueError, match="metadata"):
        study.artifact(sandbox.path.parent, sandbox.manifest, "source_runtime_questions.jsonl")


@pytest.mark.parametrize(
    "options",
    [
        ["--count", "0"],
        ["--count", "26"],
        ["--start", "-1"],
        ["--budget-cny", "nan"],
        ["--budget-cny", "inf"],
        ["--budget-cny", "-1"],
        ["--budget-cny", "11"],
        ["--arms", "base", "base"],
    ],
)
def test_invalid_launch_limits_fail_before_network(sandbox, options):
    with pytest.raises(ValueError):
        study.main(args(sandbox) + options)
    assert sandbox.clients == []


@pytest.mark.parametrize("prior", [50.0, 51.0, -1.0, float("nan"), True])
def test_cumulative_budget_cannot_reset_or_be_unknown(sandbox, prior):
    sandbox.prior = prior
    with pytest.raises(ValueError, match="budget"):
        study.main(args(sandbox))
    assert sandbox.clients == []


def test_remaining_project_cap_bounds_requested_subcap(sandbox):
    sandbox.prior = 49.99
    assert study.main(args(sandbox, count=1, arms=("base",)) + ["--budget-cny", "10"]) == 0
    plan = read(run_dir(sandbox) / "launch_plan.json")
    assert plan["subcap_cny"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    "git", [{"commit": "x", "worktree_dirty": True}, {"commit": None, "worktree_dirty": False}]
)
def test_clean_commit_required(sandbox, monkeypatch, git):
    monkeypatch.setattr(study, "_git_state", lambda: git)
    with pytest.raises(ValueError, match="clean committed"):
        study.main(args(sandbox))
    assert sandbox.clients == []


def test_live_needs_explicit_config_and_reviewed_endpoint(sandbox, monkeypatch):
    command = args(sandbox, live=False) + [
        "--allow-network",
        "--expected-manifest-sha256",
        study._sha(sandbox.path),
    ]
    with pytest.raises(ValueError, match="specified config"):
        study.main(command)
    monkeypatch.setattr(
        study,
        "read_local_bailian_settings",
        lambda p: SimpleNamespace(base_url="https://evil.example/v1", api_key="secret"),
    )
    with pytest.raises(ValueError, match="Beijing endpoint"):
        study.main(args(sandbox))
    assert sandbox.clients == []


@pytest.mark.parametrize("kind", ["claim", "launch", "malformed"])
def test_pending_or_old_launch_prevents_automatic_replay(sandbox, kind):
    runs = sandbox.root / "runs"
    runs.mkdir()
    old = {
        "protocol": study.PROTOCOL,
        "phase": "calibration",
        "arms": ["base"],
        "question_ids": sandbox.manifest["roles"]["calibration"][:1],
    }
    if kind == "launch":
        directory = runs / f"{study.PREFIX}old"
        directory.mkdir()
        path = directory / "launch_plan.json"
    else:
        path = runs / f"{study.PREFIX}old.claim.json"
    dump(path, {} if kind == "malformed" else old)
    with pytest.raises(ValueError, match="replay|audit"):
        study.main(args(sandbox))
    assert sandbox.clients == []


def test_successful_run_cannot_replay_under_different_arm_order(sandbox):
    assert study.main(args(sandbox, count=1)) == 0
    with pytest.raises(ValueError, match="replay"):
        study.main(args(sandbox, count=1, arms=("static", "fresh", "base")))
    assert len(sandbox.clients) == 1


def test_existing_serial_lock_is_never_stolen(sandbox):
    runs = sandbox.root / "runs"
    runs.mkdir()
    path = runs / ".operator-study.lock"
    path.write_text("another process", encoding="utf-8")
    with pytest.raises(FileExistsError):
        study.main(args(sandbox))
    assert path.read_text() == "another process"
    assert sandbox.clients == []


@pytest.mark.parametrize("failure", ["transport", "parse"])
def test_failure_stops_without_retry_preserves_partial_outputs_and_spend(sandbox, failure):
    if failure == "transport":
        sandbox.failure_at = 2
    else:
        sandbox.malformed_at = 2
    assert study.main(args(sandbox)) == 1
    output = run_dir(sandbox)
    reports = read(output / "predictions.json")
    assert len(reports) == 1
    assert reports[0]["arms"]["base"]["status"] == "completed"
    assert reports[0]["arms"]["fresh"]["status"] == "failed"
    assert "static" not in reports[0]["arms"]
    assert sandbox.clients[0].attempts == 2
    assert read(output / "final_budget.json")["api_requests"] == 2
    assert read(output / "predictions_frozen.json")["status"] == "failed"
    assert sandbox.indexes[0].closed
    assert study.KEY_VARIABLE not in os.environ
    failure_file = output / f"{reports[0]['question_id']}_fresh.json"
    assert read(failure_file)["status"] == "failed"


def test_reader_failure_keeps_completed_retrieval_episode(sandbox):
    sandbox.malformed_at = 1
    assert study.main(args(sandbox, count=1, arms=("base",))) == 1
    report = read(run_dir(sandbox) / "predictions.json")[0]["arms"]["base"]
    assert report["episode"]["stop_reason"] == "retrieval_budget"
    assert len(report["episode"]["searches"]) == 1
    assert report["reader"] is None


def test_index_cleanup_failure_changes_return_code(sandbox):
    sandbox.close_fail = True
    assert study.main(args(sandbox, count=1, arms=("base",))) == 1
    assert read(run_dir(sandbox) / "predictions_frozen.json")["cleanup_errors"] == ["OSError"]


def test_frozen_bank_change_during_run_fails_and_stops_next_arm(sandbox):
    directory, path = make_bank(sandbox)
    sandbox.on_complete = lambda client: path.write_text("changed", encoding="utf-8")
    assert (
        study.main(args(sandbox, count=1, arms=("memory50", "fresh")) + ["--banks", str(directory)])
        == 1
    )
    report = read(run_dir(sandbox) / "predictions.json")[0]
    assert report["arms"]["memory50"]["status"] == "failed"
    assert "fresh" not in report["arms"]
    assert read(run_dir(sandbox) / "predictions_frozen.json")["status"] == "failed"


def test_frozen_bank_only_publishes_specs_and_is_unchanged(sandbox):
    directory, path = make_bank(sandbox)
    digest = study._sha(path)
    assert study.main(args(sandbox, count=1, arms=("memory50",)) + ["--banks", str(directory)]) == 0
    assert study._sha(path) == digest
    prompts = str(sandbox.clients[0].messages)
    assert "source-audit" not in prompts
    assert "validation-audit" not in prompts
    assert "allowed_source_ids" not in prompts


@pytest.mark.parametrize("case", ["missing", "protocol", "scope"])
def test_memory_bank_validation(sandbox, case):
    directory = None
    if case != "missing":
        options = (
            {"protocol": "other"}
            if case == "protocol"
            else {"source_ids": tuple(sandbox.manifest["roles"]["calibration"][:50])}
        )
        directory, _ = make_bank(sandbox, **options)
    with pytest.raises(ValueError):
        study.load_banks(directory, ("memory50",), sandbox.manifest)


def test_empty_published_bank_is_valid_and_falls_back_without_replacement(sandbox):
    directory, path = make_bank(sandbox, published=False)
    digest = study._sha(path)
    assert study.main(args(sandbox, count=1, arms=("memory50",)) + ["--banks", str(directory)]) == 0
    assert study._sha(path) == digest
    prompt = json.loads(sandbox.clients[0].messages[0][1]["content"])
    assert prompt["mode"] == "memory" and prompt["candidate_specs"] == []


def test_source_memory_arm_rejected_before_bank_or_manifest_read(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="own future source"):
        study.main(["--manifest", "missing", "--phase", "source", "--arms", "memory50"])


def test_source_is_prediction_only_and_does_not_write_bank(sandbox):
    assert study.main(args(sandbox, phase="source", count=1, arms=("fresh",))) == 0
    assert not (sandbox.root / "banks").exists()
    report = read(run_dir(sandbox) / "predictions.json")[0]["arms"]["fresh"]
    assert report["feedback"] is None and report["memory_updated"] is False


def test_manifest_outside_project_is_not_accepted(sandbox, monkeypatch, tmp_path_factory):
    other = tmp_path_factory.mktemp("different-project")
    monkeypatch.chdir(other)
    with pytest.raises(ValueError, match="inside the project"):
        study.main(args(sandbox))


def test_reservation_limit_stops_before_any_paid_request(sandbox):
    assert study.main(args(sandbox, count=1, arms=("base",)) + ["--budget-cny", "0.0000001"]) == 1
    assert sandbox.clients[0].attempts == 0
    budget = read(run_dir(sandbox) / "final_budget.json")
    assert budget["block_reason"] == "estimated_budget_limit"
    assert budget["api_requests"] == 0
    assert read(run_dir(sandbox) / "predictions.json")[0]["arms"]["base"]["status"] == "failed"


def test_elapsed_limit_stops_before_any_paid_request(sandbox, monkeypatch):
    original = study.DurableBudgetClient

    class ExpiredClient(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.started_at -= 1900

    monkeypatch.setattr(study, "DurableBudgetClient", ExpiredClient)
    assert study.main(args(sandbox, count=1, arms=("base",))) == 1
    assert sandbox.clients[0].attempts == 0
    assert read(run_dir(sandbox) / "final_budget.json")["block_reason"] == "elapsed_time_limit"


def test_frozen_input_change_during_run_affects_return_status(sandbox):
    sandbox.on_complete = lambda client: sandbox.path.write_text("changed", encoding="utf-8")
    assert study.main(args(sandbox, count=1, arms=("base",))) == 1
    assert read(run_dir(sandbox) / "predictions_frozen.json")["status"] == "failed"


def test_source_change_before_launch_never_calls_provider(sandbox, monkeypatch):
    states = iter([{"sha256": "a" * 64}, {"sha256": "b" * 64}])
    monkeypatch.setattr(study, "source_snapshot", lambda project: next(states))
    with pytest.raises(ValueError, match="source changed"):
        study.main(args(sandbox))
    assert sandbox.clients == []
    assert not (sandbox.root / "runs" / ".operator-study.lock").exists()


def test_final_artifact_failure_still_restores_secret_environment(sandbox, monkeypatch):
    monkeypatch.setenv(study.KEY_VARIABLE, "earlier-secret")
    write = study.write_json

    def fail_final(path, value):
        if path.name == "predictions_frozen.json":
            raise OSError("synthetic full disk")
        return write(path, value)

    monkeypatch.setattr(study, "write_json", fail_final)
    with pytest.raises(OSError):
        study.main(args(sandbox, count=1, arms=("base",)))
    assert os.environ[study.KEY_VARIABLE] == "earlier-secret"
    assert sandbox.indexes[0].closed
    output = run_dir(sandbox)
    assert (output / "predictions.json").exists()
    assert list((output / "request_journal").glob("*_after.json"))
    assert list((sandbox.root / "runs").glob("*.claim.json"))


def test_live_requires_expected_manifest_fingerprint(sandbox):
    with pytest.raises(ValueError, match="explicit expected manifest"):
        study.main(args(sandbox, live=False) + ["--allow-network", "--api-config", "config"])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        study.main(args(sandbox) + ["--expected-manifest-sha256", "f" * 64])
    assert sandbox.clients == []


def test_certificate_continues_only_unstarted_qids_and_cannot_be_reused(sandbox):
    sandbox.failure_at = 7  # Question 1 finished; question 2 starts then fails.
    assert study.main(args(sandbox, count=3)) == 1
    parent = run_dir(sandbox)
    certificate = build_certificate(sandbox.root / "runs", parent.name)
    expected = sandbox.manifest["roles"]["calibration"][2:3]
    assert certificate["proof"]["question_ids"] == expected
    path = sandbox.root / "resume.json"
    dump(path, certificate)
    sandbox.failure_at = None
    command = args(sandbox, count=1) + ["--start", "2", "--resume-certificate", str(path)]
    assert study.main(command) == 0
    children = [
        p for p in (sandbox.root / "runs").glob(f"{study.PREFIX}*") if p.is_dir() and p != parent
    ]
    assert len(children) == 1
    launch = read(children[0] / "launch_plan.json")
    assert launch["resume_parent_run_id"] == parent.name
    assert launch["question_ids"] == expected
    assert (children[0] / "resume_certificate.json").is_file()
    assert (sandbox.root / "runs" / f"{parent.name}.claim.json").is_file()
    with pytest.raises(ValueError, match="replay"):
        study.main(command)
    with pytest.raises(ValueError, match="untouched"):
        study.main(args(sandbox, count=1) + ["--start", "1", "--resume-certificate", str(path)])
