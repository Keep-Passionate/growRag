"""候选开发协议的合成保护测试；不读数据集、不运行网络。"""

import json
import os
from types import SimpleNamespace

import pytest

from growrag.experiments import history_candidate_study as study
from growrag.experiments.pre_pilot import write_json as exclusive_write_json
from growrag.experiments.protocol import RuntimeQuestion


def read(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    exclusive_write_json(path, value)


def snapshot():
    files = {"src/synthetic.py": {"sha256": "synthetic", "text": "synthetic"}}
    return {"files": files, "sha256": study.fingerprint(files)}


def seal_metadata(root, plan):
    write_json(root / "launch_plan.json", plan)
    write_json(
        root.with_name(root.name + ".claim.json"), {**plan, "plan_sha256": study.fingerprint(plan)}
    )
    write_json(root / "source_snapshot.json", snapshot())


def probe_fixture(runs, monkeypatch):
    monkeypatch.setattr(study, "_probe_library", lambda runs: object())
    monkeypatch.setattr(study, "probe_request", lambda case, library: (None, (), None, None))
    monkeypatch.setattr(
        study,
        "compile_probe",
        lambda case, library, context, evidence, content: {
            "requests": [{"query": content}],
        },
    )
    root = runs / study.PROBE_RUN_ID
    seal_metadata(
        root,
        {
            "protocol": study.PROBE_PROTOCOL,
            "run_id": study.PROBE_RUN_ID,
            "model": study.PILOT_MODEL,
            "library_sha256": study.LIBRARY_SHA256,
            "source_sha256": snapshot()["sha256"],
            "cases": list(study.probe_cases()),
            "cases_sha256": study.fingerprint(study.probe_cases()),
        },
    )
    hashes, calls = {}, []
    for i, case in enumerate(study.probe_cases()):
        target = root / f"{case['case_id']}.json"
        write_json(
            target,
            {
                "status": "completed",
                "case": case,
                "raw_content": "synthetic",
                "compiled": {"requests": [{"query": "synthetic"}]},
            },
        )
        hashes[target.name] = study._sha(target)
        trace = f"{study.PROBE_RUN_ID}/{case['case_id']}/fill"
        audit = root / "api_audit" / f"{i}.json"
        call = {
            "trace_id": trace,
            "status": "completed",
            "prompt_version": study.FILL_VERSION,
            "returned_model": study.PILOT_MODEL,
            "input_tokens": 10,
            "output_tokens": 5,
            "api_requests": 1,
            "audit_path": str(audit),
        }
        calls.append(call)
        write_json(
            audit,
            {
                **call,
                "transport_source": "live_api",
                "http_status": 200,
                "request": {"model": study.PILOT_MODEL},
                "response": {
                    "model": study.PILOT_MODEL,
                    "choices": [{"message": {"content": "synthetic"}}],
                },
            },
        )
    write_json(root / "final_budget.json", {"api_requests": 5, "calls": calls})
    write_json(
        root / "SUMMARY.json",
        {
            "protocol": study.PROBE_PROTOCOL,
            "status": "completed",
            "planned_cases": 5,
            "completed_cases": 5,
            "api_requests": 5,
            "source_sha256": snapshot()["sha256"],
            "library_sha256": study.LIBRARY_SHA256,
            "prediction_sha256": hashes,
        },
    )
    return root


def prior_fixture(runs, start, count, prior_hashes=None):
    ids = [f"q{i}" for i in range(start, start + count)]
    root = runs / study.run_id(start, count)
    seal_metadata(
        root,
        {
            **study.CONFIGURATION,
            "protocol": study.PROTOCOL,
            "run_id": root.name,
            "phase": "candidate_development",
            "question_ids": ids,
            "methods": list(study.METHODS),
            "start": start,
            "count": count,
            "bundle_sha256": study.BUNDLE_SHA256,
            "library_sha256": study.LIBRARY_SHA256,
            "source_sha256": snapshot()["sha256"],
            "probe_summary_sha256": "probe-sha",
            "prior_batch_summary_sha256": prior_hashes or {},
        },
    )
    hashes, terminal = {}, []
    for qid in ids:
        row = {"question_id": qid, "methods": {}}
        for method in study.METHODS:
            path = root / f"{qid}_{method}.json"
            write_json(
                path,
                {
                    "status": "completed",
                    "question_id": qid,
                    "method": method,
                    "arm": study.underlying_arm(method),
                    "gold_loaded": False,
                    "memory_updated": False,
                },
            )
            hashes[path.name] = study._sha(path)
            row["methods"][method] = {"status": "completed", "path": path.name}
        terminal.append(row)
    write_json(root / "final_budget.json", {"calls": []})
    write_json(
        root / "SUMMARY.json",
        {
            "protocol": study.PROTOCOL,
            "status": "completed",
            "planned_questions": count,
            "completed_questions": count,
            "source_sha256": snapshot()["sha256"],
            "prediction_sha256": hashes,
            "terminal": terminal,
        },
    )
    return root


@pytest.mark.parametrize("start,count", [(0, 50), (5, 5), (20, 25), (True, 5), (0, False), (-1, 5)])
def test_unregistered_batch_rejected_before_read(tmp_path, start, count):
    with pytest.raises(ValueError, match="preregistered"):
        study.prepare(tmp_path, start=start, count=count)


@pytest.mark.parametrize("sha", [None, "", "x" * 64, "a" * 63])
def test_bundle_external_pin_required_before_read(tmp_path, monkeypatch, sha):
    monkeypatch.setattr(study, "BUNDLE_SHA256", sha)
    monkeypatch.setattr(study, "load_bundle", lambda *a: pytest.fail("must not read"))
    with pytest.raises(ValueError, match="external bundle"):
        study.prepare(tmp_path, start=0, count=5)


def test_configuration_and_underlying_identity_are_fixed():
    assert study.BATCHES == ((0, 5), (5, 20), (25, 25))
    assert study.run_id(25, 25).endswith("v1_0025_0050")
    assert tuple(study.CONFIGURATION["candidate_methods"]) == study.METHODS[2:]
    assert study.CONFIGURATION["candidate_methods"]["history_body8"] == {
        "policy": "action_text_bm25",
        "limit": 8,
    }
    assert study.CONFIGURATION["max_output_tokens"] == 2048
    assert study.CONFIGURATION["retrieval_budget"] == 3
    assert study.CONFIGURATION["runtime_memory_updates"] is False
    assert study.underlying_arm("history_body8") == "history"
    with pytest.raises(ValueError, match="unregistered"):
        study.underlying_arm("static_rules")


def test_claims_protect_pending_and_failed_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "check_history_claims", lambda *a: None)
    write_json(
        tmp_path / f"{study.PREFIX}old.claim.json",
        {
            "protocol": study.PROTOCOL,
            "question_ids": ["q0"],
            "methods": list(study.METHODS),
        },
    )
    with pytest.raises(ValueError, match="previously claimed"):
        study.check_claims(tmp_path, ["q0"])
    study.check_claims(tmp_path, ["q5"])


def test_probe_only_claim_does_not_claim_real_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "check_history_claims", lambda *a: None)
    write_json(
        tmp_path / f"{study.PROBE_RUN_ID}.claim.json",
        {
            "protocol": study.PROBE_PROTOCOL,
            "run_id": study.PROBE_RUN_ID,
        },
    )
    study.check_claims(tmp_path, ["q0"])


def test_orphan_launch_also_blocks_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "check_history_claims", lambda *a: None)
    write_json(
        tmp_path / f"{study.PREFIX}old" / "launch_plan.json",
        {
            "protocol": study.PROTOCOL,
            "question_ids": ["q0"],
            "methods": list(study.METHODS),
        },
    )
    with pytest.raises(ValueError, match="previously claimed"):
        study.check_claims(tmp_path, ["q0"])


def test_successful_probe_gate_checks_real_accounting(tmp_path, monkeypatch):
    root = probe_fixture(tmp_path, monkeypatch)
    assert study.verify_probe(tmp_path, snapshot()["sha256"]) == study._sha(root / "SUMMARY.json")


@pytest.mark.parametrize("damage", ["raw_http", "saved_compilation"])
def test_probe_content_and_compiler_are_bound_not_just_status(tmp_path, monkeypatch, damage):
    root = probe_fixture(tmp_path, monkeypatch)
    path = root / f"{study.probe_cases()[0]['case_id']}.json"
    record = read(path)
    if damage == "raw_http":
        record["raw_content"] = "fabricated content"
    else:
        record["compiled"]["requests"][0]["query"] = "fabricated query"
    path.write_text(json.dumps(record))
    summary = read(root / "SUMMARY.json")
    summary["prediction_sha256"][path.name] = study._sha(path)
    (root / "SUMMARY.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="actual HTTP|frozen consumer"):
        study.verify_probe(tmp_path, snapshot()["sha256"])


@pytest.mark.parametrize(
    "damage",
    ["source", "hash", "missing", "ledger_usage", "ledger_version", "http", "model", "compile"],
)
def test_probe_damaged_evidence_blocks(tmp_path, monkeypatch, damage):
    root = probe_fixture(tmp_path, monkeypatch)
    path = root / "SUMMARY.json"
    if damage == "source":
        with pytest.raises(ValueError, match="snapshot"):
            study.verify_probe(tmp_path, "other-source")
        return
    if damage in {"hash", "missing", "compile"}:
        path = root / f"{study.probe_cases()[0]['case_id']}.json"
        if damage == "missing":
            path.unlink()
        else:
            value = read(path)
            value["raw_content"] = "changed"
            if damage == "compile":
                value["compiled"] = {"requests": []}
            path.write_text(json.dumps(value))
            if damage == "compile":
                seal = read(root / "SUMMARY.json")
                seal["prediction_sha256"][path.name] = study._sha(path)
                (root / "SUMMARY.json").write_text(json.dumps(seal))
    elif damage.startswith("ledger"):
        path = root / "final_budget.json"
        value = read(path)
        value["calls"][0]["input_tokens" if damage == "ledger_usage" else "prompt_version"] = None
        path.write_text(json.dumps(value))
    else:
        path = root / "api_audit/0.json"
        value = read(path)
        if damage == "http":
            value["http_status"] = 400
        else:
            value["response"]["model"] = "other-model"
        path.write_text(json.dumps(value))
    with pytest.raises((ValueError, FileNotFoundError)):
        study.verify_probe(tmp_path, snapshot()["sha256"])


def test_prior_batch_gate_requires_all_previous_sealed_files(tmp_path):
    manifest = {"question_ids": [f"q{i}" for i in range(50)]}
    first = prior_fixture(tmp_path, 0, 5)
    hashes = {first.name: study._sha(first / "SUMMARY.json")}
    assert (
        study.verify_prior_batches(tmp_path, manifest, 5, snapshot()["sha256"], "probe-sha")
        == hashes
    )
    second = prior_fixture(tmp_path, 5, 20, hashes)
    hashes[second.name] = study._sha(second / "SUMMARY.json")
    assert (
        study.verify_prior_batches(tmp_path, manifest, 25, snapshot()["sha256"], "probe-sha")
        == hashes
    )
    (first / "q0_history_body8.json").unlink()
    with pytest.raises(FileNotFoundError):
        study.verify_prior_batches(tmp_path, manifest, 25, snapshot()["sha256"], "probe-sha")


@pytest.mark.parametrize("damage", ["failed", "config", "terminal", "method", "hashes", "probe"])
def test_prior_batch_no_handwritten_completed_shortcut(tmp_path, damage):
    root = prior_fixture(tmp_path, 0, 5)
    if damage == "config":
        path = root / "launch_plan.json"
        value = read(path)
        value["max_output_tokens"] = 4096
        path.write_text(json.dumps(value))
    elif damage == "method":
        path = root / "q0_history_body8.json"
        value = read(path)
        value["method"] = "history_body3"
        path.write_text(json.dumps(value))
    else:
        path = root / "SUMMARY.json"
        value = read(path)
        if damage == "failed":
            value["status"] = "failed"
        elif damage == "terminal":
            value["terminal"].pop()
        elif damage == "hashes":
            value["prediction_sha256"].pop("q0_base.json")
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        study.verify_prior_batches(
            tmp_path,
            {"question_ids": [f"q{i}" for i in range(50)]},
            5,
            snapshot()["sha256"],
            "other-probe" if damage == "probe" else "probe-sha",
        )


@pytest.mark.parametrize("method", study.METHODS)
@pytest.mark.parametrize("fail", [False, True])
def test_method_wrapper_preserves_underlying_report_and_failure(
    tmp_path, monkeypatch, method, fail
):
    called = []
    factory = object()
    monkeypatch.setattr(study, "planner_factory", lambda name: factory)

    def execute(question, arm, index, client, **kwargs):
        called.append((arm, kwargs))
        write_json(
            kwargs["target"],
            {"arm": arm, "status": "failed" if fail else "completed", "calls": ["paid-synthetic"]},
        )
        if fail:
            raise ValueError("local validation")

    monkeypatch.setattr(study, "execute_arm", execute)
    target = tmp_path / "prediction.json"
    args = (RuntimeQuestion("q", "synthetic"), method, None, None)
    kwargs = {
        "library": "library",
        "trace": f"run/q/{method}",
        "log": lambda e: None,
        "target": target,
    }
    if fail:
        with pytest.raises(ValueError, match="local validation"):
            study.execute_method(*args, **kwargs)
    else:
        study.execute_method(*args, **kwargs)
    final, raw = read(target), read(tmp_path / "raw_execution" / target.name)
    assert final == {**raw, "method": method}
    assert final["arm"] == study.underlying_arm(method)
    assert final["calls"] == ["paid-synthetic"]
    assert called[0][1]["trace"].endswith(method)
    if method.startswith("history"):
        assert called[0][1]["planner_class"] is factory


@pytest.mark.parametrize("raw", [False, True])
def test_method_wrapper_replay_fails_before_any_execution(tmp_path, monkeypatch, raw):
    target = tmp_path / "prediction.json"
    write_json(tmp_path / "raw_execution" / target.name if raw else target, {"sealed": True})
    monkeypatch.setattr(study, "execute_arm", lambda *a, **kw: pytest.fail("no replay"))
    with pytest.raises(FileExistsError, match="never replay"):
        study.execute_method(
            RuntimeQuestion("q", "synthetic"),
            "base",
            None,
            None,
            library=None,
            trace="trace",
            log=lambda _: None,
            target=target,
        )


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    questions = tuple(RuntimeQuestion(f"q{i}", "synthetic question") for i in range(50))
    manifest = {
        "question_ids": [q.question_id for q in questions],
        "corpus_ref": {"index_path": "data/index.sqlite3", "sha256": "corpus", "rows": 10},
        "library": {"path": "library.json"},
    }
    monkeypatch.setattr(
        study,
        "prepare",
        lambda project, start, count: (
            manifest,
            questions[start : start + count],
            SimpleNamespace(fingerprint="library"),
            tmp_path / "corpus.jsonl",
        ),
    )
    monkeypatch.setattr(study, "check_claims", lambda *a: None)
    monkeypatch.setattr(study, "source_snapshot", lambda *a: snapshot())
    monkeypatch.setattr(
        study, "_git_state", lambda: {"commit": "synthetic", "worktree_dirty": False}
    )
    monkeypatch.setattr(study, "verify_probe", lambda *a: "probe-sha")
    monkeypatch.setattr(study, "verify_prior_batches", lambda *a: {})
    monkeypatch.setattr(study, "_runtime_unchanged", lambda *a: None)
    monkeypatch.setattr(
        study, "reviewed_history", lambda *a: {"calls": [], "prior_reserved_cny": 10.0}
    )
    monkeypatch.setattr(
        study,
        "read_local_bailian_settings",
        lambda *a: SimpleNamespace(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key="synthetic-key"
        ),
    )
    events, configs, executions, clients = [], [], [], []

    class Log:
        arm = "setup"
        question_id = ""

        def __init__(self, output):
            pass

        def __call__(self, event):
            events.append({"arm": self.arm, "question_id": self.question_id, **event})

    class Client:
        block_reason = None

        def __init__(self, delegate, limits, journal):
            self.calls, self.limits = [], limits
            clients.append(self)

        def report(self):
            return {
                "api_requests": len(self.calls),
                "calls": self.calls,
                "reserved_cny": 0.1,
                "estimated_actual_cny": 0.01,
                "block_reason": self.block_reason,
            }

    def live(config, *args, **kwargs):
        configs.append(config)
        return object()

    def execute(question, method, index, client, **kwargs):
        executions.append((question.question_id, method, kwargs["trace"], kwargs["log"].arm))
        client.calls.append({"trace_id": kwargs["trace"], "status": "completed"})
        write_json(
            kwargs["target"],
            {"status": "completed", "method": method, "arm": study.underlying_arm(method)},
        )

    monkeypatch.setattr(study, "OperatorLog", Log)
    monkeypatch.setattr(study, "SharedBM25Index", lambda *a: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(study, "LiveChatClient", live)
    monkeypatch.setattr(study, "DurableBudgetClient", Client)
    monkeypatch.setattr(study, "HistoryContractClient", lambda client, **kw: client)
    monkeypatch.setattr(study, "execute_method", execute)
    return SimpleNamespace(
        root=tmp_path,
        events=events,
        configs=configs,
        executions=executions,
        clients=clients,
        args=SimpleNamespace(start=0, count=5, allow_network=True),
    )


def test_dry_run_has_no_keys_budget_gate_writes_or_calls(sandbox, monkeypatch):
    sandbox.args.allow_network = False
    for name in ("read_local_bailian_settings", "reviewed_history", "verify_probe", "serial_lock"):
        monkeypatch.setattr(study, name, lambda *a: pytest.fail("not during dry run"))
    assert study.run(sandbox.args) == 0
    assert not (sandbox.root / "runs").exists()
    assert not sandbox.configs and not sandbox.executions


def test_runner_five_methods_same_reader_config_and_durable_budget(sandbox, monkeypatch):
    monkeypatch.setenv(study.KEY_VARIABLE, "previous-value")
    assert study.run(sandbox.args) == 0
    root = sandbox.root / "runs" / study.run_id(0, 5)
    summary, launch = read(root / "SUMMARY.json"), read(root / "launch_plan.json")
    assert summary["completed_questions"] == 5 and len(summary["prediction_sha256"]) == 25
    assert set(summary["terminal"][0]["methods"]) == set(study.METHODS)
    assert launch["phase"] == "candidate_development" and launch["methods"] == list(study.METHODS)
    assert launch["subcap_cny"] == 5 and launch["probe_summary_sha256"] == "probe-sha"
    assert launch["prior_batch_summary_sha256"] == {}
    assert sandbox.configs[0].max_calls == 95 and sandbox.configs[0].max_output_tokens == 2048
    assert sandbox.configs[0].json_object_mode and not sandbox.configs[0].json_schema_mode
    assert sandbox.clients[0].limits.max_elapsed_seconds == 1800
    assert sandbox.clients[0].limits.input_per_million_cny == 0.2
    assert len(sandbox.executions) == 25
    assert all(
        method == logged and trace.endswith(method)
        for _, method, trace, logged in sandbox.executions
    )
    assert os.environ[study.KEY_VARIABLE] == "previous-value"
    assert (root / "final_budget.json").is_file()
    assert root.with_name(root.name + ".claim.json").is_file()
    assert not (root.parent / ".operator-study.lock").exists()


@pytest.mark.parametrize("kind", ["output", "claim"])
def test_existing_output_or_claim_never_overwritten(sandbox, kind):
    root = sandbox.root / "runs" / study.run_id(0, 5)
    if kind == "output":
        root.mkdir(parents=True)
    else:
        write_json(root.with_name(root.name + ".claim.json"), {"untouched": True})
    with pytest.raises(FileExistsError):
        study.run(sandbox.args)
    assert not sandbox.executions


@pytest.mark.parametrize(
    "gate", ["verify_probe", "verify_prior_batches", "reviewed_history", "_runtime_unchanged"]
)
def test_failed_gate_does_not_claim_or_call(sandbox, monkeypatch, gate):
    def fail(*args):
        raise ValueError("gate blocked")

    monkeypatch.setattr(study, gate, fail)
    with pytest.raises(ValueError, match="gate blocked"):
        study.run(sandbox.args)
    assert not sandbox.configs and not sandbox.executions
    assert not list((sandbox.root / "runs").glob("*.claim.json"))


@pytest.mark.parametrize(
    "prior,series,expected",
    [(199.0, 0, 1.0), (10.0, 4.5, 0.5), (200.0, 0, None), (10.0, 5.0, None)],
)
def test_project_and_series_caps_both_apply(sandbox, monkeypatch, prior, series, expected):
    monkeypatch.setattr(
        study,
        "reviewed_history",
        lambda *a: {
            "prior_reserved_cny": prior,
            "calls": [{"trace_id": study.PREFIX + "fillcheck", "reserved_cny": series}],
        },
    )
    if expected is None:
        with pytest.raises(ValueError, match="exhausted"):
            study.run(sandbox.args)
        assert not sandbox.configs
    else:
        assert study.run(sandbox.args) == 0
        assert sandbox.clients[0].limits.budget_cny == expected


def test_failed_method_is_preserved_rest_not_attempted_and_no_retry(sandbox, monkeypatch):
    original = study.execute_method

    def fail(question, method, index, client, **kwargs):
        if method == "history_body3":
            client.calls.append({"trace_id": kwargs["trace"], "status": "completed"})
            write_json(kwargs["target"], {"status": "failed", "method": method, "arm": "history"})
            raise ValueError("synthetic local validation after paid completion")
        return original(question, method, index, client, **kwargs)

    monkeypatch.setattr(study, "execute_method", fail)
    monkeypatch.delenv(study.KEY_VARIABLE, raising=False)
    assert study.run(sandbox.args) == 1
    root = sandbox.root / "runs" / study.run_id(0, 5)
    summary, budget = read(root / "SUMMARY.json"), read(root / "final_budget.json")
    assert summary["status"] == "failed" and summary["completed_questions"] == 0
    assert summary["terminal"][0]["methods"]["history_body3"]["status"] == "failed"
    assert summary["terminal"][0]["methods"]["history_body8"]["status"] == "not_attempted"
    assert all(
        row["status"] == "not_attempted" for row in summary["terminal"][1]["methods"].values()
    )
    assert budget["api_requests"] == 4 and budget["calls"][-1]["status"] == "completed"
    assert study.KEY_VARIABLE not in os.environ
    assert len(summary["prediction_sha256"]) == 4


def test_close_failure_still_seals_budget_and_restores_key(sandbox, monkeypatch):
    def close():
        raise OSError("synthetic index close failure")

    monkeypatch.setattr(study, "SharedBM25Index", lambda *a: SimpleNamespace(close=close))
    monkeypatch.setenv(study.KEY_VARIABLE, "previous")
    assert study.run(sandbox.args) == 1
    root = sandbox.root / "runs" / study.run_id(0, 5)
    assert read(root / "SUMMARY.json")["failure_type"] == "OSError"
    assert read(root / "final_budget.json")["api_requests"] == 25
    assert os.environ[study.KEY_VARIABLE] == "previous"


def test_beijing_price_not_silently_used_for_another_region(sandbox, monkeypatch):
    monkeypatch.setattr(
        study,
        "read_local_bailian_settings",
        lambda *a: SimpleNamespace(
            base_url="https://dashscope-intl.aliyuncs.com/v1", api_key="synthetic"
        ),
    )
    with pytest.raises(ValueError, match="Beijing"):
        study.run(sandbox.args)
    assert not sandbox.configs


def test_source_change_after_prediction_marks_batch_failed(sandbox, monkeypatch):
    snapshots = iter((snapshot(), snapshot(), {"sha256": "changed"}))
    monkeypatch.setattr(study, "source_snapshot", lambda *a: next(snapshots))
    assert study.run(sandbox.args) == 1
    root = sandbox.root / "runs" / study.run_id(0, 5)
    assert read(root / "SUMMARY.json")["status"] == "failed"
    assert read(root / "final_budget.json")["api_requests"] == 25


def test_main_defaults_to_preflight(monkeypatch):
    seen = []
    monkeypatch.setattr(study, "run", lambda args: seen.append(args) or 0)
    assert study.main(["--start", "0", "--count", "5"]) == 0
    assert seen[0].allow_network is False
