"""Synthetic paired-runner tests; fabricated answers are never experimental data."""

import hashlib
import inspect
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import run_reformer_control as runner
from growrag.experiments import run_reformer_pilot as baseline
from growrag.experiments import shared_s2g_corpus as corpus
from growrag.experiments.protocol import RuntimeQuestion

GOLD_MARKER = "SYNTHETIC-SECRET-GOLD"


def call(trace, cost=0.001):
    return {
        "trace_id": trace,
        "api_requests": 1,
        "input_tokens": 30 if cost is not None else None,
        "output_tokens": 10 if cost is not None else None,
        "estimated_actual_cny": cost,
        "reserved_cny": 0.01,
    }


@dataclass
class FakeClient:
    fail_trace: str | None = None
    block_reason: str | None = None
    attempts: int = 0

    def __post_init__(self):
        self.calls, self.payloads, self.selections = [], [], []


class FakeProgress:
    arm = None

    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append({"arm": self.arm, **event})


def question(qid="q0"):
    return RuntimeQuestion(qid, f"When was synthetic Northbridge {qid} founded?", "synthetic")


@pytest.fixture
def fake_backend(monkeypatch):
    class FakeAdapter:
        provenance = {"synthetic": True}

        def __init__(self, reformer, reader, client, index, *, event_callback=None):
            self.client, self.events = client, []

        def request(self, trace, payload):
            assert GOLD_MARKER not in json.dumps(payload)
            self.client.payloads.append(deepcopy(payload))
            self.client.calls.append(call(trace))
            self.client.attempts += 1
            if self.client.fail_trace and self.client.fail_trace in trace:
                self.events.append({"kind": "synthetic_failure"})
                raise ValueError("synthetic failure")

        def select(self, query, trace):
            self.request(trace + "/select", query)
            result = {
                "question": query,
                "question_id": trace,
                "pattern_id": 3,
                "selected_pattern": {"name": "synthetic", "examples": ["example"]},
                "initial_selector_documents": [{"doc_id": "initial"}],
                "events": [],
                "retrieval_rounds": 1,
            }
            result["selection_sha256"] = runner.fingerprint(result)
            return result

        def run_selected(self, query, trace, selection, *, include_examples):
            seal = selection.pop("selection_sha256")
            assert runner.fingerprint(selection) == seal
            assert selection["question_id"].endswith("/SELECTION")
            self.client.selections.append(deepcopy(selection))
            # Deliberate mutation tests that the sibling arm gets an isolated copy.
            selection["selected_pattern"]["examples"].append("mutated only locally")
            for stage in ("rewrite", "answer"):
                self.request(
                    trace + "/" + stage, {"query": query, "include_examples": include_examples}
                )
            return {
                "question_id": trace,
                "answer": "1901",
                "retrieval_rounds": 1,
                "logical_retrieval_rounds": 2,
                "retrieved_documents": [{"doc_id": "with" if include_examples else "without"}],
                "pattern_id": 3,
                "include_examples": include_examples,
                "rewritten_query": "synthetic rewrite",
                "events": [],
            }

    monkeypatch.setattr(runner, "ReFormeRControlAPI", FakeAdapter)
    return SimpleNamespace(index="synthetic-index", metadata={"synthetic": True})


def install_gold_spies(monkeypatch, output, *, expected_pairs, fail_score=None):
    gold_reads = []

    def load_gold(manifest, *, completed_question_ids, expected_manifest_sha256):
        assert expected_manifest_sha256 == "synthetic-manifest-sha"
        seal = json.loads((output / "predictions_frozen.json").read_bytes())
        assert seal["completed_question_ids"] == expected_pairs
        predictions = []
        for p in sorted((output / "questions").glob("*/prediction_report.json")):
            prediction = json.loads(p.read_bytes())
            predictions.append(prediction)
            assert GOLD_MARKER not in json.dumps(prediction)
            for row in prediction["arms"].values():
                assert row["feedback"] is None
        assert runner.fingerprint(predictions) == seal["reports_sha256_before_scoring"]
        qid = completed_question_ids[0]
        gold_reads.append(qid)
        return {qid: SimpleNamespace(question_id=qid, answers=(GOLD_MARKER,))}

    def score(result, gold, index, *, retained_mode):
        assert result["question_id"] == gold.question_id
        assert retained_mode == "raw"
        context = result["retrieved_documents"][0]["doc_id"]
        if context == fail_score:
            raise ValueError("synthetic score failure")
        return {
            "answer_em": float(context == "with"),
            "answer_f1": 1.0,
            "raw_support_recall": 0.5,
            "raw_support_hits": 1,
            "gold_support_count": 2,
            "coverage_notice": "not entailment",
        }

    monkeypatch.setattr(corpus, "load_gold_after_execution", load_gold)
    monkeypatch.setattr(corpus, "score_result", score)
    return gold_reads


def execute(tmp_path, backend, client, n=2, start=0):
    return runner.execute_batch(
        [question(f"q{i}") for i in range(start, start + n)],
        backend,
        client,
        Path("r"),
        Path("s"),
        Path("manifest"),
        "synthetic-manifest-sha",
        tmp_path,
        FakeProgress(),
        run_id="synthetic",
        start=start,
    )


def test_prediction_interface_excludes_gold_and_uses_five_unique_calls(tmp_path, fake_backend):
    params = inspect.signature(runner.execute_question).parameters
    assert not {"gold", "question_types", "baseline_results", "feedback"} & set(params)
    client = FakeClient()
    report = runner.execute_question(
        question(),
        fake_backend.index,
        client,
        Path("r"),
        Path("s"),
        tmp_path / "q",
        FakeProgress(),
        run_id="synthetic",
        offset=0,
    )
    assert report["selection"]["result"]["question_id"] == "synthetic/q0/SELECTION"
    assert report["arm_order"] == ["WITH_EXAMPLES", "WITHOUT_EXAMPLES"]
    owned = report["selection"]["calls"] + sum(
        (report["arms"][a]["calls"] for a in report["arm_order"]), []
    )
    assert owned == client.calls and len(owned) == client.attempts == 5
    assert len({c["trace_id"] for c in owned}) == 5
    assert client.selections[0] == client.selections[1]
    assert report["selection"]["result"]["selected_pattern"]["examples"] == ["example"]
    for arm in runner.ARMS:
        row = report["arms"][arm]
        assert row["result"]["question_id"] == "q0"
        assert row["result"]["execution_trace_id"] == "synthetic/q0/" + arm
        assert row["feedback"] is None
        assert (tmp_path / "q" / f"{arm}_execution.json").is_file()


def test_batch_has_alternating_order_and_gold_after_all_predictions(
    tmp_path, monkeypatch, fake_backend
):
    reads = install_gold_spies(monkeypatch, tmp_path, expected_pairs=["q0", "q1"])
    client = FakeClient()
    summary = execute(tmp_path, fake_backend, client)
    reports = json.loads((tmp_path / "reports.json").read_bytes())
    assert reports[0]["arm_order"] == list(runner.ARMS)
    assert reports[1]["arm_order"] == list(reversed(runner.ARMS))
    assert client.calls[6]["trace_id"].startswith("synthetic/q1/WITHOUT_EXAMPLES/")
    assert reads == ["q0", "q1"]
    assert summary["completed_pairs"] == summary["scored_pairs"] == 2
    assert summary["arms"]["WITH_EXAMPLES"]["em"] == 1
    assert summary["arms"]["WITHOUT_EXAMPLES"]["em"] == 0
    assert summary["all_actual_calls_including_interrupted"]["api_requests"] == 10
    assert all("answer_em" not in r["selection"]["initial_feedback"] for r in reports)
    assert all(r["selection"]["feedback"] is None for r in reports)
    assert GOLD_MARKER not in json.dumps(client.payloads)


@pytest.mark.parametrize(
    "stage,requests,completed_arm",
    [
        ("SELECTION", 1, None),
        ("WITH_EXAMPLES/rewrite", 2, None),
        ("WITH_EXAMPLES/answer", 3, None),
        ("WITHOUT_EXAMPLES/rewrite", 4, "WITH_EXAMPLES"),
        ("WITHOUT_EXAMPLES/answer", 5, "WITH_EXAMPLES"),
    ],
)
def test_any_component_failure_stops_paid_execution_preserves_partial(
    tmp_path,
    monkeypatch,
    fake_backend,
    stage,
    requests,
    completed_arm,
):
    reads = install_gold_spies(monkeypatch, tmp_path, expected_pairs=[])
    client = FakeClient(fail_trace=f"q0/{stage}")
    summary = execute(tmp_path, fake_backend, client)
    assert summary["started"] == 1 and summary["completed_pairs"] == 0
    assert client.attempts == requests
    assert not (tmp_path / "questions" / "0001").exists()
    assert summary["stop_reason"] == "reformer_control_component_failure"
    assert reads == (["q0"] if completed_arm else [])
    report = json.loads((tmp_path / "reports.json").read_bytes())[0]
    if completed_arm:
        assert report["scoring_status"] == "partial_completed"
        assert report["arms"][completed_arm]["feedback"]["answer_em"] == 1
    assert summary["all_actual_calls_including_interrupted"]["api_requests"] == requests


@pytest.mark.parametrize("failed_context", ["with", "without", "initial"])
def test_scoring_failure_does_not_commit_partial_labels(
    tmp_path, monkeypatch, fake_backend, failed_context
):
    install_gold_spies(
        monkeypatch, tmp_path, expected_pairs=["q0", "q1"], fail_score=failed_context
    )
    client = FakeClient()
    summary = execute(tmp_path, fake_backend, client)
    reports = json.loads((tmp_path / "reports.json").read_bytes())
    assert summary["completed_pairs"] == 2 and summary["scored_pairs"] == 0
    assert summary["stop_reason"] == "offline_scoring_failure"
    assert all(r["scoring_status"] == "failed" for r in reports)
    assert all(r["arms"][a]["feedback"] is None for r in reports for a in runner.ARMS)
    assert client.attempts == 10


def test_unknown_usage_is_not_zero_cost():
    summary = runner.summarize(
        [],
        [call("unknown", None)],
        run_id="synthetic",
        planned=1,
        stop_reason="unknown_usage",
        started=1,
    )
    assert summary["all_actual_calls_including_interrupted"]["estimated_actual_cny"] is None
    assert summary["started_without_report"] == 1


@pytest.mark.parametrize("start,count", [(0, 1), (0, 25), (75, 25), (99, 1)])
def test_bounds_and_separate_output_namespace(start, count):
    assert runner.batch_identity(start, count) == f"{runner.PREFIX}{start:04d}_{start + count:04d}"
    assert runner.batch_identity(start, count) != baseline.batch_identity(start, count)


@pytest.mark.parametrize(
    "start,count", [(-1, 1), (True, 1), (0, True), (0, 0), (0, 26), (100, 1), (90, 11), (1.0, 2)]
)
def test_bad_bounds_rejected(start, count):
    with pytest.raises(ValueError):
        runner.batch_identity(start, count)


@pytest.mark.parametrize("artifact", ["claim", "launch"])
def test_prior_claim_blocks_replay_and_protocol_changes(tmp_path, artifact):
    name = runner.batch_identity(0, 5)
    path = tmp_path / (f"{name}.claim.json" if artifact == "claim" else name + "/launch_plan.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_json(
        path, {"protocol": runner.PROTOCOL, "manifest_sha256": "frozen", "question_ids": ["q0"]}
    )
    with pytest.raises(ValueError, match="already claimed"):
        runner.check_no_replay(tmp_path, "frozen", ["q0"])
    with pytest.raises(ValueError, match="change protocol"):
        runner.check_no_replay(tmp_path, "other", ["q1"])
    runner.check_no_replay(tmp_path, "frozen", ["q1"])


def test_bad_claim_is_not_ignored(tmp_path):
    runner.write_json(
        tmp_path / f"{runner.batch_identity(0, 5)}.claim.json",
        {"protocol": runner.PROTOCOL, "manifest_sha256": "frozen", "question_ids": []},
    )
    with pytest.raises(ValueError, match="offline audit"):
        runner.check_no_replay(tmp_path, "frozen", ["q0"])


def put_budget(tmp_path, start, reservation):
    directory = tmp_path / runner.batch_identity(start, 1)
    directory.mkdir()
    runner.write_json(
        directory / "launch_plan.json", {"protocol": runner.PROTOCOL, "model": runner.PILOT_MODEL}
    )
    (directory / "final_budget.json").write_text(json.dumps({"reserved_cny": reservation}))


def test_series_cap_accumulates(tmp_path):
    put_budget(tmp_path, 0, 0.5)
    put_budget(tmp_path, 1, 0.25)
    assert runner.series_reserved(tmp_path) == 0.75


@pytest.mark.parametrize("reservation", [None, True, -0.1, float("nan"), float("inf")])
def test_series_unknown_or_bad_reservations_block(tmp_path, reservation):
    put_budget(tmp_path, 0, reservation)
    with pytest.raises(ValueError, match="reservation"):
        runner.series_reserved(tmp_path)


def test_unfinished_claim_blocks_other_batches(tmp_path):
    runner.write_json(tmp_path / f"{runner.batch_identity(0, 5)}.claim.json", {})
    with pytest.raises(ValueError, match="unfinished"):
        runner.series_reserved(tmp_path)


def prepare_cli(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    runner.write_json(
        manifest,
        {
            "official_split": "train",
            "role": "development",
            "question_ids": [f"q{i}" for i in range(500)],
        },
    )
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "FROZEN_MANIFEST_SHA", sha)
    monkeypatch.setattr(baseline, "FROZEN_MANIFEST_SHA", sha)
    monkeypatch.setattr(runner, "source_snapshot", lambda _: {"sha256": "synthetic-source"})
    monkeypatch.setattr(
        runner, "_git_state", lambda: {"commit": "synthetic", "worktree_dirty": True}
    )
    monkeypatch.setattr(runner, "reviewed_history", lambda _: {"prior_reserved_cny": 5.0})
    monkeypatch.setattr(
        runner, "read_local_bailian_settings", lambda _: pytest.fail("no key access")
    )
    monkeypatch.setattr(runner, "LiveChatClient", lambda *a, **k: pytest.fail("no network"))
    return [
        "--manifest",
        str(manifest),
        "--reformer-snapshot",
        "r",
        "--reader-snapshot",
        "s",
        "--index-path",
        "i.sqlite",
        "--runs-root",
        str(tmp_path / "runs"),
        "--start",
        "0",
        "--count",
        "5",
    ]


def test_dry_run_has_no_secret_network_gold_or_claim(tmp_path, monkeypatch, fake_backend, capsys):
    args = prepare_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(corpus, "load_gold_after_execution", lambda *a, **k: pytest.fail("no gold"))
    assert runner.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["max_calls"] == 25 and plan["subcap_cny"] == 3
    assert (
        plan["shared_selection"] and plan["deployable_cost_policy"] == "full_selection_plus_one_arm"
    )
    assert plan["question_ids"] == [f"q{i}" for i in range(5)]
    assert plan["retrieval_rounds_per_deployable_path"] == 2
    assert not (tmp_path / "runs").exists()


def test_live_requires_clean_commit_before_secret(tmp_path, monkeypatch, fake_backend):
    args = prepare_cli(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="clean committed checkout"):
        runner.main(args + ["--allow-network", "--api-config", "secret.md"])
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("project_used,series_used", [(50, 0), (5, 3)])
def test_project_and_series_caps_before_model_access(
    tmp_path, monkeypatch, fake_backend, project_used, series_used
):
    args = prepare_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "reviewed_history", lambda _: {"prior_reserved_cny": project_used})
    monkeypatch.setattr(runner, "series_reserved", lambda _: series_used)
    with pytest.raises(ValueError, match="budget exhausted"):
        runner.main(args)


def test_remaining_series_subcap(tmp_path, monkeypatch, fake_backend, capsys):
    args = prepare_cli(tmp_path, monkeypatch)
    monkeypatch.setattr(runner, "series_reserved", lambda _: 2.75)
    assert runner.main(args) == 0
    assert json.loads(capsys.readouterr().out)["subcap_cny"] == 0.25
