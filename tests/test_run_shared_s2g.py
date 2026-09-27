"""Offline runner contracts; synthetic adapters never load author weights or call APIs.

中文：只测试批次隔离、事后评分、费用和错误状态；合成输出不是科研结果。
"""

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import run_s2g_author_pilot as legacy
from growrag.experiments import run_shared_s2g as runner
from growrag.experiments import shared_s2g_corpus as corpus
from growrag.experiments.api_client import ChatConfig, ChatResponse
from growrag.experiments.budget import PriceLimits, request_input_bytes
from growrag.experiments.protocol import RuntimeQuestion
from growrag.experiments.s2g_author_api import PROMPT_VERSIONS, AuthorDocument

GOLD_MARKER = "SYNTHETIC-SECRET-GOLD"


def call(*, actual=0.002):
    return {
        "api_requests": 1,
        "input_tokens": 10 if actual is not None else None,
        "output_tokens": 4 if actual is not None else None,
        "estimated_actual_cny": actual,
        "reserved_cny": 0.01,
    }


@dataclass
class FakeClient:
    fail_arm: str | None = None
    block_reason: str | None = None
    attempts: int = 0

    def __post_init__(self):
        self.calls, self.payloads, self.traces = [], [], []


class FakeProgress:
    def __init__(self):
        self.arm, self.events = None, []

    def __call__(self, event):
        self.events.append(dict(event, arm=self.arm))


@pytest.fixture
def fake_backend(monkeypatch):
    """Mirror the real adapter's trace-as-result-ID behavior without its snapshot."""
    documents = (AuthorDocument("synthetic-doc", "Northbridge", "Founded in 1901."),)

    def index(question, top_k):
        assert GOLD_MARKER not in question
        assert top_k == 6
        return documents

    class FakeAdapter:
        def __init__(self, upstream, client, index, **kwargs):
            self.client, self.events, self.progress = client, [], kwargs["event_callback"]
            assert kwargs["max_turns"] == 4 and kwargs["top_docs"] == 6
            assert kwargs["gap_profile"] == "paper_k1" and kwargs["remove_repeat_docs"]
            self.scope = {"concat_raw_retrieved_docs": lambda titles, texts: "\n".join(texts)}

        def _execute(self, question, trace, context):
            assert GOLD_MARKER not in question + context
            self.client.payloads.append((question, context))
            self.client.traces.append(trace)
            self.client.attempts += 1
            self.client.calls.append(call())
            if self.client.fail_arm == self.progress.arm:
                raise RuntimeError("synthetic API failure")
            return {
                "question_id": trace,
                "question": question,
                "answer": "1901",
                "retrieval_rounds": 2,
                "events": [],
            }

        def answer_once(self, question, context, trace):
            return self._execute(question, trace, context)

        def run(self, question, trace):
            return self._execute(question, trace, "synthetic retained evidence")

    monkeypatch.setattr(runner, "S2GAuthorAPI", FakeAdapter)
    return SimpleNamespace(index=index)


def question(qid="q0"):
    return RuntimeQuestion(qid, "When was Northbridge founded?", "synthetic")


def install_gold_spies(monkeypatch, output, order, *, fail_gold=False, fail_score=False):
    def load_gold(manifest, *, completed_question_ids, expected_manifest_sha256):
        assert expected_manifest_sha256 == "synthetic-manifest-sha"
        qid = completed_question_ids[0]
        offset = int(qid[1:])
        for arm in runner.ARMS:
            path = output / "questions" / f"{offset:04d}" / f"{arm}_execution.json"
            row = json.loads(path.read_bytes())
            assert row["status"] == "completed"
            assert row["feedback"] is None
        order.append((qid, "gold"))
        if fail_gold:
            raise ValueError("synthetic gold integrity failure")
        return {qid: SimpleNamespace(question_id=qid, answers=(GOLD_MARKER,))}

    def score(result, gold, index, *, retained_mode):
        # Real scorer checks this; adapter result IDs must not remain trace IDs.
        assert result["question_id"] == gold.question_id
        order.append((gold.question_id, retained_mode))
        if fail_score and retained_mode == "sources":
            raise ValueError("synthetic second-arm scoring failure")
        return {"answer_em": 1.0, "answer_f1": 1.0}

    monkeypatch.setattr(corpus, "load_gold_after_execution", load_gold)
    monkeypatch.setattr(corpus, "score_result", score)


def run_batch(tmp_path, fake_backend, client, *, n=2):
    questions = [question(f"q{i}") for i in range(n)]
    return runner.execute_batch(
        questions,
        fake_backend,
        client,
        Path("synthetic-upstream-does-not-exist"),
        Path("synthetic-manifest"),
        "synthetic-manifest-sha",
        tmp_path,
        FakeProgress(),
        run_id="synthetic-run",
        start=0,
        question_types={q.question_id: "bridge" for q in questions},
    )


def test_pair_has_no_gold_or_question_type_interface(tmp_path, fake_backend):
    assert "gold" not in inspect.signature(runner.execute_pair).parameters
    assert "question_types" not in inspect.signature(runner.execute_pair).parameters
    client = FakeClient()
    result = runner.execute_pair(
        question(),
        fake_backend.index,
        client,
        Path("unused"),
        tmp_path / "q",
        FakeProgress(),
        run_id="unique-run",
    )
    assert client.attempts == 2
    assert len(set(client.traces)) == 2
    assert all(t.startswith("unique-run/q0/") for t in client.traces)
    assert all(r["status"] == "completed" and r["feedback"] is None for r in result.values())
    assert all(r["result"]["question_id"] == "q0" for r in result.values())
    assert GOLD_MARKER not in json.dumps(client.payloads)


def test_pair_failure_records_cost_and_skips_remaining_arm(tmp_path, fake_backend):
    client = FakeClient(fail_arm=runner.ARMS[0])
    rows = runner.execute_pair(
        question(),
        fake_backend.index,
        client,
        Path("unused"),
        tmp_path / "q",
        FakeProgress(),
        run_id="failed-run",
    )
    assert rows[runner.ARMS[0]]["status"] == "failed"
    assert rows[runner.ARMS[1]]["status"] == "not_executed"
    assert all(row["feedback"] is None for row in rows.values())
    assert len(rows[runner.ARMS[0]]["calls"]) == client.attempts == 1
    assert client.block_reason


def test_two_question_batch_scores_only_after_durable_pair(tmp_path, monkeypatch, fake_backend):
    order, client = [], FakeClient()
    install_gold_spies(monkeypatch, tmp_path, order)
    summary = run_batch(tmp_path, fake_backend, client)
    assert order == [(qid, stage) for qid in ("q0", "q1") for stage in ("gold", "raw", "sources")]
    assert summary["started"] == summary["complete_pairs"] == summary["scored_pairs"] == 2
    assert summary["all_actual_calls_including_interrupted"]["api_requests"] == 4
    assert summary["stop_reason"] is None
    assert json.loads((tmp_path / "summary.json").read_bytes()) == summary
    assert len(json.loads((tmp_path / "reports.json").read_bytes())) == 2
    assert GOLD_MARKER not in json.dumps(client.payloads)


def test_batch_first_api_failure_never_reads_gold_or_continues(tmp_path, monkeypatch, fake_backend):
    monkeypatch.setattr(corpus, "load_gold_after_execution", lambda *a, **k: pytest.fail("no gold"))
    monkeypatch.setattr(corpus, "score_result", lambda *a, **k: pytest.fail("no scoring"))
    client = FakeClient(fail_arm=runner.ARMS[0])
    summary = run_batch(tmp_path, fake_backend, client)
    assert summary["planned"] == 2 and summary["started"] == 1
    assert summary["complete_pairs"] == summary["scored_pairs"] == 0
    assert summary["all_actual_calls_including_interrupted"]["api_requests"] == 1
    assert all(a["em"] is None and a["n"] == 0 for a in summary["arms"].values())
    assert not (tmp_path / "questions" / "0001").exists()


@pytest.mark.parametrize("failure", ["gold", "score"])
def test_offline_failure_preserves_pair_but_never_partial_scores(
    tmp_path, monkeypatch, fake_backend, failure
):
    order, client = [], FakeClient()
    install_gold_spies(
        monkeypatch, tmp_path, order, fail_gold=failure == "gold", fail_score=failure == "score"
    )
    summary = run_batch(tmp_path, fake_backend, client)
    assert summary["started"] == summary["complete_pairs"] == 1
    assert summary["scored_pairs"] == 0
    assert summary["stop_reason"]
    assert client.attempts == 2
    reports = json.loads((tmp_path / "reports.json").read_bytes())
    assert reports[0]["scoring_status"] == "failed"
    assert all(row["feedback"] is None for row in reports[0]["arms"].values())
    assert not (tmp_path / "questions" / "0001").exists()


@pytest.mark.parametrize("count", [25, 50])
def test_page_identity_is_frozen_and_nonoverlapping(count):
    first = runner.batch_identity("500_v1", 0, count)
    second = runner.batch_identity("500_v1", count, count)
    assert first != second
    assert first.endswith(f"_0000_{count:04d}")
    assert second.endswith(f"_{count:04d}_{2 * count:04d}")


@pytest.mark.parametrize(
    "series,start,count",
    [
        ("500_v0", 0, 25),
        ("100_v1", 0, 25),
        ("../500_v1", 0, 25),
        ("500_v1", -1, 25),
        ("500_v1", True, 25),
        ("500_v1", 0, True),
        ("500_v1", 0, 0),
        ("500_v1", 0, 51),
    ],
)
def test_invalid_identity_rejected(series, start, count):
    with pytest.raises(ValueError):
        runner.batch_identity(series, start, count)


def put_plan(runs, run_id, **extra):
    directory = runs / run_id
    directory.mkdir()
    runner.write_json(
        directory / "launch_plan.json",
        {
            "protocol": runner.PROTOCOL,
            "model": runner.PILOT_MODEL,
            **extra,
        },
    )
    return directory


def test_claimed_questions_and_manifest_change_cannot_replay(tmp_path):
    put_plan(
        tmp_path,
        runner.batch_identity("500_v1", 0, 25),
        manifest_sha256="frozen",
        question_ids=["q0", "q1"],
    )
    with pytest.raises(ValueError, match="already claimed"):
        runner.check_no_replay(tmp_path, "500_v1", "frozen", ["q1", "q2"])
    with pytest.raises(ValueError, match="change dataset"):
        runner.check_no_replay(tmp_path, "500_v1", "changed", ["q2"])
    runner.check_no_replay(tmp_path, "500_v1", "frozen", ["q2"])


def test_explicit_continuation_only_exempts_proven_prior_claims(tmp_path, monkeypatch):
    from growrag.experiments import shared_continuation

    parent = runner.batch_identity("500_v1", 0, 25)
    put_plan(tmp_path, parent, manifest_sha256="frozen", question_ids=["q0", "q1"])
    proof = {"validated_prior_runs": [parent], "question_ids": ["q1"]}
    observed = []

    def verified(*args, **kwargs):
        observed.append(kwargs)
        return proof

    monkeypatch.setattr(shared_continuation, "verify_unstarted_continuation", verified)
    assert (
        runner.check_no_replay(tmp_path, "500_v1", "frozen", ["q1"], continuation_of=parent)
        == proof
    )
    assert observed[0]["question_ids"] == ["q1"]
    assert observed[0]["protocol"] == runner.PROTOCOL
    put_plan(
        tmp_path,
        runner.batch_identity("500_v1", 1, 1),
        manifest_sha256="frozen",
        question_ids=["q1"],
    )
    with pytest.raises(ValueError, match="already claimed"):
        runner.check_no_replay(tmp_path, "500_v1", "frozen", ["q1"], continuation_of=parent)


def test_continuation_cannot_change_series(tmp_path):
    with pytest.raises(ValueError, match="same frozen series"):
        runner.check_no_replay(
            tmp_path,
            "500_v2",
            "frozen",
            ["q1"],
            continuation_of=runner.batch_identity("500_v1", 0, 25),
        )


def test_failed_continuation_proof_cannot_release_claim(tmp_path, monkeypatch):
    from growrag.experiments import shared_continuation

    def rejected(*args, **kwargs):
        raise ValueError("already touched")

    monkeypatch.setattr(shared_continuation, "verify_unstarted_continuation", rejected)
    with pytest.raises(ValueError, match="already touched"):
        runner.check_no_replay(
            tmp_path,
            "500_v1",
            "frozen",
            ["q1"],
            continuation_of=runner.batch_identity("500_v1", 0, 25),
        )


def test_existing_batch_claim_rejects_before_manifest_or_secret(tmp_path, monkeypatch):
    run_id = runner.batch_identity("500_v1", 0, 25)
    runner.write_json(tmp_path / f"{run_id}.claim.json", {"synthetic": True})
    monkeypatch.setattr(runner, "_sha", lambda *a: pytest.fail("no data access"))
    monkeypatch.setattr(runner, "read_local_bailian_settings", lambda *a: pytest.fail("no keys"))
    with pytest.raises(FileExistsError, match="already exists"):
        runner.main(
            [
                "--manifest",
                "unused",
                "--upstream",
                "unused",
                "--index-path",
                "unused",
                "--runs-root",
                str(tmp_path),
                "--expected-manifest-sha256",
                "unused",
                "--series",
                "500_v1",
                "--start",
                "0",
                "--count",
                "25",
                "--allow-network",
            ]
        )


def test_history_includes_all_s2g_versions_and_previous_shared_batch(tmp_path, monkeypatch):
    directory = put_plan(tmp_path, runner.batch_identity("500_v1", 0, 25))
    runner.write_json(directory / "final_budget.json", {"api_requests": 1, "calls": [call()]})
    observed = []

    def reconcile(runs, *, reviewed_extra_ledgers):
        observed.extend(reviewed_extra_ledgers)
        return {"prior_reserved_cny": 0.01, "prior_unknown_cost_requests": 1}

    monkeypatch.setattr(runner, "reconcile_history", reconcile)
    result = runner.reviewed_history(tmp_path)
    assert result["prior_unknown_cost_requests"] == 1
    assert observed[: len(runner.HISTORICAL_ROOTS)] == list(runner.HISTORICAL_ROOTS)
    for name in (runner.RUN_ID, runner.JSON_RUN_ID, runner.SCHEMA_RUN_ID, runner.CAPACITY_RUN_ID):
        assert observed.count(f"{name}/final_budget.json") == 1
    assert observed[-1] == f"{directory.name}/final_budget.json"


@pytest.mark.parametrize("change", [{"protocol": "unknown"}, {"model": "other-model"}])
def test_history_rejects_unreviewed_shared_run(tmp_path, monkeypatch, change):
    directory = put_plan(tmp_path, runner.batch_identity("500_v1", 0, 25), **change)
    runner.write_json(directory / "final_budget.json", {"api_requests": 1, "calls": [call()]})
    monkeypatch.setattr(runner, "reconcile_history", lambda *a, **k: pytest.fail("no reconcile"))
    with pytest.raises(ValueError, match="unreviewed"):
        runner.reviewed_history(tmp_path)


def test_history_rejects_zero_request_ledger_with_pending_calls(tmp_path, monkeypatch):
    directory = put_plan(tmp_path, runner.batch_identity("500_v1", 0, 25))
    runner.write_json(directory / "final_budget.json", {"api_requests": 0, "calls": [call()]})
    monkeypatch.setattr(runner, "reconcile_history", lambda *a, **k: pytest.fail("no reconcile"))
    with pytest.raises(ValueError, match="pending"):
        runner.reviewed_history(tmp_path)


def test_history_rejects_unfinished_request_intent_before_new_batch(tmp_path, monkeypatch):
    directory = put_plan(tmp_path, runner.batch_identity("500_v1", 0, 25))
    journal = directory / "request_journal"
    journal.mkdir()
    runner.write_json(
        journal / "0000_intent.json",
        {"potential_reserved_cny": 0.01, "status": "pending_no_automatic_retry"},
    )
    monkeypatch.setattr(runner, "reconcile_history", lambda *a, **k: pytest.fail("no reconcile"))
    with pytest.raises(ValueError):
        runner.reviewed_history(tmp_path)


def report(base_em, s2g_em, *, complete=True, actual=0.002):
    arms = {}
    for arm, em in zip(runner.ARMS, (base_em, s2g_em), strict=True):
        arms[arm] = {
            "status": "completed" if complete else "failed",
            "result": {"retrieval_rounds": 1 if arm == runner.ARMS[0] else 2},
            "feedback": {
                "answer_em": em,
                "answer_f1": em,
                "unscorable_annotation": em is None,
            }
            if complete
            else None,
            "calls": [call(actual=actual)],
        }
    return {"complete_pair": complete, "arms": arms}


def test_summary_counts_repairs_harms_and_excludes_annotation_unknowns():
    reports = [report(0, 1), report(1, 0), report(1, 1), report(None, None)]
    calls = [c for r in reports for a in r["arms"].values() for c in a["calls"]]
    summary = runner.summarize(reports, calls, run_id="synthetic", planned=4, stop_reason=None)
    assert summary["complete_pairs"] == 4 and summary["scored_pairs"] == 3
    assert summary["unscorable_annotation_pairs"] == 1
    assert summary["paired_em"] == {"repairs": 1, "harms": 1}
    assert all(a["em"] == pytest.approx(2 / 3) for a in summary["arms"].values())
    assert summary["arms"][runner.ARMS[1]]["retrieval_rounds_total"] == 6
    assert summary["all_actual_calls_including_interrupted"]["api_requests"] == 8
    assert summary["benchmark_reproduction"] is False


def test_summary_preserves_failed_and_unattributed_call_cost():
    reports = [report(0, 1), report(None, None, complete=False, actual=None)]
    calls = [c for r in reports for a in r["arms"].values() for c in a["calls"]]
    calls.append(call())  # An interrupted request may precede a per-question report.
    summary = runner.summarize(reports, calls, run_id="synthetic", planned=3, stop_reason="failed")
    assert summary["scored_pairs"] == 1
    assert summary["all_actual_calls_including_interrupted"]["api_requests"] == 5
    assert summary["all_actual_calls_including_interrupted"]["estimated_actual_cny"] is None
    assert all(a["all_started_cost"]["api_requests"] == 2 for a in summary["arms"].values())
    assert all(
        a["all_started_cost"]["estimated_actual_cny"] is None for a in summary["arms"].values()
    )


def capacity_client(tmp_path, client_type=runner.SharedBudgetClient):
    config = ChatConfig("https://example.invalid/v1", "fake", "NEVER_READ", 10)
    delegate = SimpleNamespace(config=config, attempts=0, transport_source="fake_test")
    observed = []

    def complete(messages, *, trace_id, prompt_version):
        observed.append((delegate.config, json.loads(json.dumps(messages)), prompt_version))
        delegate.attempts += 1
        return ChatResponse(
            "synthetic", "fake", "fake", None, None, 10, 4, 0, Path("unused"), "fake_test"
        )

    delegate.complete = complete
    client = client_type(delegate, PriceLimits(), tmp_path / "journal")
    client.schema_stages = client.qwen_output_caps = True
    return client, delegate, config, observed


def test_shared_v5_changes_only_answer_capacity_not_legacy_or_author_prompt(tmp_path):
    client, delegate, config, observed = capacity_client(tmp_path)
    messages = [
        {"role": "system", "content": "Frozen synthetic author system prompt."},
        {"role": "user", "content": "Frozen synthetic author query and evidence."},
    ]
    frozen_messages = json.loads(json.dumps(messages))
    for stage, requested in legacy.AUTHOR_OUTPUT_CAPS.items():
        client.complete_author(
            messages,
            trace_id=f"v5-{stage}",
            prompt_version=PROMPT_VERSIONS[stage],
            max_output_tokens=requested,
            temperature=0,
            top_p=1,
        )
        assert client.config == delegate.config == config
        assert messages == frozen_messages
    assert [item[0].max_output_tokens for item in observed] == [768, 128, 1024]
    assert [item[0].json_schema_mode for item in observed] == [True, True, False]
    assert not any(item[0].json_object_mode for item in observed)
    assert all(item[1] == frozen_messages for item in observed)
    assert [item[2] for item in observed] == list(PROMPT_VERSIONS.values())
    assert client.generation_profile == "qwen_capacity_v5_judge768_extract128_answer1024"
    assert legacy.QWEN_OUTPUT_CAPS == {"judge": 768, "extract": 128, "answer": 256}
    assert legacy.AUTHOR_OUTPUT_CAPS == {"judge": 256, "extract": 64, "answer": 128}
    assert runner.SHARED_OUTPUT_CAPS is not legacy.QWEN_OUTPUT_CAPS
    assert client.report()["api_requests"] == 3
    expected_reserve = sum(
        (
            (request_input_bytes(c, messages, prompt_version=version) + 1024) * 0.2
            + c.max_output_tokens * 0.8
        )
        / 1_000_000
        for c, _, version in observed
    )
    assert client.reserved_cny == pytest.approx(expected_reserve)


def test_original_v4_client_still_uses_answer_256(tmp_path):
    client, _, _, observed = capacity_client(tmp_path, legacy.AuthorBudgetClient)
    client.complete_author(
        [{"role": "user", "content": "unchanged"}],
        trace_id="original-v4",
        prompt_version=PROMPT_VERSIONS["answer"],
        max_output_tokens=128,
        temperature=0,
        top_p=1,
    )
    assert observed[0][0].max_output_tokens == 256
    assert client.generation_profile == "qwen_capacity_v4_judge768_extract128_answer256"


@pytest.mark.parametrize(
    "version,cap",
    [
        ("unknown", 128),
        (PROMPT_VERSIONS["answer"], 1024),
        (PROMPT_VERSIONS["answer"], 256),
        (PROMPT_VERSIONS["answer"], True),
        (PROMPT_VERSIONS["extract"], 128),
    ],
)
def test_v5_rejects_unreviewed_original_stage_caps_before_any_call(tmp_path, version, cap):
    client, delegate, _, observed = capacity_client(tmp_path)
    with pytest.raises(ValueError):
        client.complete_author(
            [{"role": "user", "content": "no transport"}],
            trace_id="rejected-cap",
            prompt_version=version,
            max_output_tokens=cap,
            temperature=0,
            top_p=1,
        )
    assert delegate.attempts == 0 and not observed and not client.calls


@pytest.mark.parametrize("json_mode,schema_mode", [(True, True), (False, False)])
def test_v5_cannot_bypass_strict_schema_configuration(tmp_path, json_mode, schema_mode):
    client, delegate, _, _ = capacity_client(tmp_path)
    client.json_stages, client.schema_stages = json_mode, schema_mode
    with pytest.raises(ValueError, match="strict schema"):
        client.author_output_cap(PROMPT_VERSIONS["answer"], 128)
    assert delegate.attempts == 0


def test_reviewed_v1_failed_capacity_run_remains_in_v2_budget(tmp_path, monkeypatch):
    assert runner.PROTOCOL == "growrag-s2g-shared-qwen-capacity-v3"
    directory = put_plan(
        tmp_path,
        runner.batch_identity("32_v1", 0, 32),
        protocol="growrag-s2g-shared-qwen-capacity-v1",
    )
    runner.write_json(
        directory / "final_budget.json",
        {"api_requests": 1, "calls": [call()], "block_reason": "truncated_response"},
    )
    observed = []

    def reconcile(runs, *, reviewed_extra_ledgers):
        observed.extend(reviewed_extra_ledgers)
        return {"prior_reserved_cny": 0.01}

    monkeypatch.setattr(runner, "reconcile_history", reconcile)
    assert runner.reviewed_history(tmp_path)["prior_reserved_cny"] == 0.01
    assert f"{directory.name}/final_budget.json" in observed
    assert runner.batch_identity("32_v2", 0, 32) != directory.name
