"""Synthetic runner checks; none of these outputs are research measurements."""

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import run_reformer_pilot as runner
from growrag.experiments import shared_s2g_corpus as corpus
from growrag.experiments.protocol import RuntimeQuestion

GOLD_MARKER = "SYNTHETIC-SECRET-GOLD"


def call(trace, *, cost=0.001):
    return {
        "trace_id": trace, "api_requests": 1,
        "input_tokens": 30 if cost is not None else None,
        "output_tokens": 10 if cost is not None else None,
        "estimated_actual_cny": cost, "reserved_cny": 0.01,
    }


@dataclass
class FakeClient:
    fail_question: str | None = None
    block_reason: str | None = None
    attempts: int = 0

    def __post_init__(self):
        self.calls, self.payloads, self.traces = [], [], []


class FakeProgress:
    def __init__(self):
        self.arm, self.events = None, []

    def __call__(self, event):
        self.events.append({"arm": self.arm, **event})


def question(qid="q0"):
    return RuntimeQuestion(qid, f"When was Northbridge {qid} founded?", "synthetic")


@pytest.fixture
def fake_backend(monkeypatch):
    class FakeAdapter:
        provenance = {"synthetic": True}

        def __init__(self, reformer, reader, client, index, *, event_callback=None):
            self.client, self.events = client, []

        def run(self, query, trace):
            assert GOLD_MARKER not in query
            self.client.payloads.append(query)
            self.client.traces.append(trace)
            for stage in ("select", "rewrite", "answer"):
                self.client.calls.append(call(f"{trace}/{stage}"))
                self.client.attempts += 1
                if self.client.fail_question and f"/{self.client.fail_question}/" in trace:
                    self.events.append({"kind": "synthetic_failure"})
                    raise ValueError("synthetic failure, not a real model response")
            return {
                "question_id": trace, "answer": "1901", "retrieval_rounds": 2,
                "retrieved_documents": [{"doc_id": "final"}],
                "initial_selector_documents": [{"doc_id": "initial"}],
                "selected_pattern": "synthetic pattern", "rewritten_query": "foundation date",
                "retrieval_query": query + " foundation date", "events": [],
            }

    monkeypatch.setattr(runner, "ReFormeRAPI", FakeAdapter)
    return SimpleNamespace(index="synthetic-index-not-used", metadata={"synthetic": True})


def install_gold_spies(monkeypatch, output, order, *, expected_completed, fail_score=None):
    def load_gold(manifest, *, completed_question_ids, expected_manifest_sha256):
        assert expected_manifest_sha256 == "synthetic-manifest-sha"
        qid = completed_question_ids[0]
        seal = json.loads((output / "predictions_frozen.json").read_bytes())
        assert seal["completed_question_ids"] == expected_completed
        # Every prediction in the batch, not merely this one, precedes the first gold read.
        for completed_id in expected_completed:
            path = output / "questions" / f"{int(completed_id[1:]):04d}"
            raw = json.loads((path / f"{runner.ARM}_execution.json").read_bytes())
            assert raw["status"] == "completed" and raw["feedback"] is None
            assert raw["result"]["question_id"] == completed_id
        order.append((qid, "gold"))
        return {qid: SimpleNamespace(question_id=qid, answers=(GOLD_MARKER,))}

    def score(result, gold, index, *, retained_mode):
        assert result["question_id"] == gold.question_id and retained_mode == "raw"
        context = result["retrieved_documents"][0]["doc_id"]
        order.append((gold.question_id, context))
        if fail_score == context:
            raise ValueError("synthetic scoring failure")
        return {
            "answer_em": 1.0, "answer_f1": 1.0, "raw_support_recall": 0.5,
            "raw_support_hits": 1, "gold_support_count": 2,
            "coverage_notice": "coverage is not entailment",
        }

    monkeypatch.setattr(corpus, "load_gold_after_execution", load_gold)
    monkeypatch.setattr(corpus, "score_result", score)


def execute(tmp_path, backend, client, *, n=2):
    return runner.execute_batch(
        [question(f"q{i}") for i in range(n)], backend, client,
        Path("synthetic-reformer"), Path("synthetic-reader"),
        Path("synthetic-manifest"), "synthetic-manifest-sha", tmp_path, FakeProgress(),
        run_id="synthetic-run", start=0,
    )


def test_prediction_interface_cannot_accept_gold_or_cached_baselines(tmp_path, fake_backend):
    params = inspect.signature(runner.execute_question).parameters
    assert not {"gold", "question_types", "baseline_results", "feedback"} & set(params)
    client = FakeClient()
    row = runner.execute_question(
        question(), fake_backend.index, client, Path("r"), Path("s"),
        tmp_path / "q", FakeProgress(), run_id="synthetic-run",
    )
    assert row["status"] == "completed" and row["feedback"] is None
    assert row["result"]["question_id"] == "q0"
    assert row["result"]["execution_trace_id"] == f"synthetic-run/q0/{runner.ARM}"
    assert len(row["calls"]) == client.attempts == 3
    assert row["calls"] == client.calls
    assert GOLD_MARKER not in json.dumps(client.payloads)


def test_all_predictions_sealed_before_gold_and_initial_only_has_coverage(
    tmp_path, monkeypatch, fake_backend,
):
    order, client = [], FakeClient()
    install_gold_spies(monkeypatch, tmp_path, order, expected_completed=["q0", "q1"])
    summary = execute(tmp_path, fake_backend, client)
    assert order == [(q, step) for q in ("q0", "q1") for step in ("gold", "final", "initial")]
    assert summary["completed"] == summary["scored"] == summary["started"] == 2
    assert summary["all_actual_calls_including_interrupted"]["api_requests"] == 6
    assert summary["completed_retrieval_rounds"] == 4
    assert summary["em"] == 1 and summary["benchmark_reproduction"] is False
    reports = json.loads((tmp_path / "reports.json").read_bytes())
    assert all("answer_em" not in r["outcome"]["initial_feedback"] for r in reports)
    assert all(r["outcome"]["initial_feedback"]["raw_support_recall"] == 0.5 for r in reports)
    owned = [c for r in reports for c in r["outcome"]["calls"]]
    assert owned == client.calls
    assert GOLD_MARKER not in json.dumps(client.payloads)


def test_first_api_failure_never_reads_gold_or_executes_remaining(
    tmp_path, monkeypatch, fake_backend,
):
    monkeypatch.setattr(corpus, "load_gold_after_execution", lambda *a, **k: pytest.fail("no gold"))
    client = FakeClient(fail_question="q0")
    summary = execute(tmp_path, fake_backend, client)
    assert summary["started"] == 1 and summary["planned"] == 2
    assert summary["scored"] == summary["completed"] == 0
    assert summary["em"] is None and summary["f1"] is None
    assert summary["all_actual_calls_including_interrupted"]["api_requests"] == 1
    assert not (tmp_path / "questions" / "0001").exists()


def test_later_failure_still_scores_already_frozen_completed_prediction(
    tmp_path, monkeypatch, fake_backend,
):
    order, client = [], FakeClient(fail_question="q1")
    install_gold_spies(monkeypatch, tmp_path, order, expected_completed=["q0"])
    summary = execute(tmp_path, fake_backend, client, n=3)
    assert summary["started"] == 2 and summary["completed"] == summary["scored"] == 1
    assert client.attempts == 4 and summary["stop_reason"] == "reformer_component_failure"
    assert not (tmp_path / "questions" / "0002").exists()


@pytest.mark.parametrize("failure", ["final", "initial"])
def test_score_failure_never_commits_partial_feedback(tmp_path, monkeypatch, fake_backend, failure):
    order, client = [], FakeClient()
    install_gold_spies(
        monkeypatch, tmp_path, order, expected_completed=["q0", "q1"], fail_score=failure,
    )
    summary = execute(tmp_path, fake_backend, client)
    assert summary["completed"] == 2 and summary["scored"] == 0
    assert summary["stop_reason"] == "offline_scoring_failure"
    reports = json.loads((tmp_path / "reports.json").read_bytes())
    assert all(r["scoring_status"] == "failed" for r in reports)
    assert all(r["outcome"]["feedback"] is None for r in reports)
    assert client.attempts == 6


def test_unknown_usage_stays_unknown_not_free():
    summary = runner.summarize(
        [], [call("unknown", cost=None)], run_id="synthetic", planned=1,
        stop_reason="unknown_usage", started=1,
    )
    totals = summary["all_actual_calls_including_interrupted"]
    assert totals["api_requests"] == 1 and totals["estimated_actual_cny"] is None
    assert summary["started_without_report"] == 1


@pytest.mark.parametrize("start,count", [(0, 1), (0, 25), (75, 25), (99, 1)])
def test_valid_fixed_cohort_bounds(start, count):
    assert runner.batch_identity(start, count).endswith(f"{start:04d}_{start + count:04d}")


@pytest.mark.parametrize("start,count", [
    (-1, 1), (True, 1), (0, True), (0, 0), (0, 26), (100, 1), (90, 11), (1.0, 2),
])
def test_invalid_fixed_cohort_bounds(start, count):
    with pytest.raises(ValueError):
        runner.batch_identity(start, count)


def write_manifest(tmp_path, monkeypatch, *, split="train", role="development", n=500):
    path = tmp_path / "manifest.json"
    runner.write_json(path, {
        "official_split": split, "role": role,
        "question_ids": [f"q{i}" for i in range(n)],
    })
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "FROZEN_MANIFEST_SHA", sha)
    return path, sha


def test_cohort_is_manifest_prefix_not_score_selected(tmp_path, monkeypatch):
    path, sha = write_manifest(tmp_path, monkeypatch)
    _, ids = runner.frozen_ids(path, sha, 25, 25)
    assert ids == [f"q{i}" for i in range(25, 50)]
    with pytest.raises(ValueError, match="reviewed"):
        runner.frozen_ids(path, "unreviewed", 0, 5)


@pytest.mark.parametrize("fields", [
    {"split": "dev"}, {"split": "test"}, {"role": "training"}, {"n": 100},
])
def test_wrong_source_or_size_rejected(tmp_path, monkeypatch, fields):
    path, sha = write_manifest(tmp_path, monkeypatch, **fields)
    with pytest.raises(ValueError):
        runner.frozen_ids(path, sha, 0, 5)


@pytest.mark.parametrize("artifact", ["launch", "claim"])
def test_any_existing_claim_blocks_replay_even_if_no_output(tmp_path, artifact):
    name = runner.batch_identity(0, 5)
    path = tmp_path / (f"{name}.claim.json" if artifact == "claim" else name + "/launch_plan.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.write_json(path, {
        "protocol": runner.PROTOCOL, "manifest_sha256": "frozen", "question_ids": ["q0", "q1"],
    })
    with pytest.raises(ValueError, match="already claimed"):
        runner.check_no_replay(tmp_path, "frozen", ["q1"])
    with pytest.raises(ValueError, match="change protocol or dataset"):
        runner.check_no_replay(tmp_path, "other", ["q2"])
    runner.check_no_replay(tmp_path, "frozen", ["q2"])


def test_unknown_claim_cannot_be_silently_ignored(tmp_path):
    runner.write_json(tmp_path / f"{runner.batch_identity(0, 5)}.claim.json", {
        "protocol": runner.PROTOCOL, "manifest_sha256": "frozen", "question_ids": [],
    })
    with pytest.raises(ValueError, match="offline audit"):
        runner.check_no_replay(tmp_path, "frozen", ["q0"])


def cli_args(tmp_path, manifest):
    return [
        "--manifest", str(manifest), "--reformer-snapshot", "synthetic-reformer",
        "--reader-snapshot", "synthetic-reader", "--index-path", "synthetic.sqlite",
        "--runs-root", str(tmp_path / "runs"), "--start", "0", "--count", "5",
    ]


def prepare_plan(monkeypatch):
    monkeypatch.setattr(runner, "source_snapshot", lambda _: {"sha256": "synthetic-source"})
    monkeypatch.setattr(
        runner, "_git_state", lambda: {"commit": "synthetic", "worktree_dirty": True}
    )
    monkeypatch.setattr(runner, "reviewed_history", lambda _: {"prior_reserved_cny": 5.0})
    monkeypatch.setattr(runner, "read_local_bailian_settings", lambda _: pytest.fail("no secret"))
    monkeypatch.setattr(runner, "LiveChatClient", lambda *a, **k: pytest.fail("no network"))


def test_default_plan_reads_neither_secret_nor_gold_and_creates_no_claim(
    tmp_path, monkeypatch, fake_backend, capsys,
):
    manifest, _ = write_manifest(tmp_path, monkeypatch)
    prepare_plan(monkeypatch)
    monkeypatch.setattr(corpus, "load_gold_after_execution", lambda *a, **k: pytest.fail("no gold"))
    assert runner.main(cli_args(tmp_path, manifest)) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["max_calls"] == 15 and plan["subcap_cny"] == 3.0
    assert plan["model"] == runner.PILOT_MODEL and plan["no_training_no_memory_updates"]
    assert plan["question_ids"] == [f"q{i}" for i in range(5)]
    assert not (tmp_path / "runs").exists()


def test_live_request_requires_clean_git_before_loading_key(tmp_path, monkeypatch, fake_backend):
    manifest, _ = write_manifest(tmp_path, monkeypatch)
    prepare_plan(monkeypatch)
    with pytest.raises(ValueError, match="clean committed checkout"):
        runner.main(cli_args(tmp_path, manifest) + ["--allow-network", "--api-config", "secret.md"])
    assert not (tmp_path / "runs").exists()


def test_project_cap_is_enforced_before_any_model_access(tmp_path, monkeypatch, fake_backend):
    manifest, _ = write_manifest(tmp_path, monkeypatch)
    prepare_plan(monkeypatch)
    monkeypatch.setattr(runner, "reviewed_history", lambda _: {"prior_reserved_cny": 50.0})
    with pytest.raises(ValueError, match="budget exhausted"):
        runner.main(cli_args(tmp_path, manifest))


def put_prior_budget(runs, *, start, reserved):
    directory = runs / runner.batch_identity(start, 1)
    directory.mkdir(parents=True)
    runner.write_json(directory / "launch_plan.json", {
        "protocol": runner.PROTOCOL, "model": runner.PILOT_MODEL,
        "manifest_sha256": "frozen", "question_ids": [f"q{start}"],
    })
    # Deliberately allow invalid/nonfinite JSON numbers to test corrupted ledgers.
    (directory / "final_budget.json").write_text(
        json.dumps({"reserved_cny": reserved}), encoding="utf-8"
    )
    return directory


def test_series_subcap_accumulates_all_previous_batches(tmp_path):
    put_prior_budget(tmp_path, start=0, reserved=0.4)
    put_prior_budget(tmp_path, start=1, reserved=0.6)
    assert runner.series_reserved(tmp_path) == 1.0


@pytest.mark.parametrize("reserved", [None, True, -0.1, float("nan"), float("inf")])
def test_invalid_or_unknown_series_cost_is_not_treated_as_free(tmp_path, reserved):
    put_prior_budget(tmp_path, start=0, reserved=reserved)
    with pytest.raises(ValueError, match="reservation"):
        runner.series_reserved(tmp_path)


def test_incomplete_claim_blocks_other_nonoverlapping_batches(tmp_path):
    runner.write_json(tmp_path / f"{runner.batch_identity(0, 5)}.claim.json", {
        "protocol": runner.PROTOCOL, "manifest_sha256": "frozen", "question_ids": ["q0"],
    })
    with pytest.raises(ValueError, match="unfinished ReFormeR claim"):
        runner.series_reserved(tmp_path)


def test_series_cap_independent_of_larger_project_cap(tmp_path, monkeypatch, fake_backend):
    manifest, _ = write_manifest(tmp_path, monkeypatch)
    prepare_plan(monkeypatch)
    monkeypatch.setattr(runner, "series_reserved", lambda _: 3.0)
    with pytest.raises(ValueError, match="series budget exhausted"):
        runner.main(cli_args(tmp_path, manifest))


def test_plan_reserves_only_remaining_series_budget(tmp_path, monkeypatch, fake_backend, capsys):
    manifest, _ = write_manifest(tmp_path, monkeypatch)
    prepare_plan(monkeypatch)
    monkeypatch.setattr(runner, "series_reserved", lambda _: 2.75)
    assert runner.main(cli_args(tmp_path, manifest)) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["subcap_cny"] == 0.25 and plan["series_prior_reserved_cny"] == 2.75

