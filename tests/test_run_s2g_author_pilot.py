"""Synthetic no-network tests for the visible author-API pilot runner."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import run_s2g_author_pilot as pilot
from growrag.experiments.api_client import ChatConfig, ChatResponse
from growrag.experiments.budget import PriceLimits
from growrag.experiments.hotpot import parse_hotpot_example
from growrag.experiments.protocol import Evidence
from growrag.experiments.s2g_author_api import PROMPT_VERSIONS

UPSTREAM = Path("external/s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6")


def example():
    return parse_hotpot_example(
        {
            "_id": "synthetic-question",
            "question": "When was Northbridge founded?",
            "context": [["Northbridge", ["Northbridge was founded in 1901.", "It is a school."]]],
            "answer": "SECRET-GOLD-MARKER",
            "supporting_facts": [["Northbridge", 0]],
            "type": "bridge",
        },
        dataset="synthetic-only-not-real-data",
    )


@dataclass
class FakeClient:
    fail: bool = False
    block_reason: str | None = None
    attempts: int = 0

    def __post_init__(self):
        self.calls, self.messages = [], []

    def complete(self, messages, *, trace_id, prompt_version):
        assert "SECRET-GOLD-MARKER" not in json.dumps(messages)
        self.messages.append(messages)
        self.attempts += 1
        self.calls.append(
            {
                "api_requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_actual_cny": 0,
                "reserved_cny": 0,
            }
        )
        if self.fail:
            raise RuntimeError("synthetic failure")
        if prompt_version == PROMPT_VERSIONS["judge"]:
            text = json.dumps({"sufficient": True, "gap_items": []})
        elif prompt_version == PROMPT_VERSIONS["extract"]:
            text = json.dumps({"evidence_global_ids": [1]})
        else:
            text = "Answer: 1901\nRationale: The source gives this date."
        return ChatResponse(
            text, "fake", "fake", "synthetic", "synthetic", 0, 0, 0.0, Path("unused"), "fake_test"
        )


def test_index_ranks_whole_documents_not_sentences():
    index = pilot.LocalDocumentIndex(example().candidate_context)
    docs = index("Northbridge", 6)
    assert len(docs) == 1
    assert "1901" in docs[0].text and "It is a school." in docs[0].text
    assert len(index.index.corpus) == 1


def test_author_sentence_index_never_used_as_hotpot_index():
    evidence = Evidence("x", "T", 10, "One full sentence.")
    assert pilot.aligned_sources(
        (evidence,), [{"title": "T", "sentence_id": 1, "text": "One full sentence."}]
    ) == (evidence,)
    assert (
        pilot.aligned_sources((evidence,), [{"title": "T", "sentence_id": 10, "text": "One full"}])
        == ()
    )


@pytest.mark.skipif(not UPSTREAM.exists(), reason="author snapshot not redistributed")
def test_synthetic_pair_gold_only_after_both_arms_and_live_logs(tmp_path):
    client = FakeClient()
    report = pilot.execute_question(
        example(), client, UPSTREAM, tmp_path / "q", pilot.ProgressLog(tmp_path)
    )
    assert report["complete_pair"]
    assert client.attempts == 5  # BASE; empty judge, extract, sufficient judge, answer.
    assert all(report["arms"][a]["feedback"]["answer_em"] == 0 for a in pilot.ARMS)
    assert report["offline_gold_answers"] == ["SECRET-GOLD-MARKER"]
    live = (tmp_path / "live.log").read_text(encoding="utf-8")
    assert '"kind": "query"' in live
    assert "SECRET-GOLD-MARKER" not in live
    assert (tmp_path / "events.jsonl").exists()


@pytest.mark.skipif(not UPSTREAM.exists(), reason="author snapshot not redistributed")
def test_failure_stops_next_arm_and_never_scores_as_zero(tmp_path):
    client = FakeClient(fail=True)
    report = pilot.execute_question(
        example(), client, UPSTREAM, tmp_path / "q", pilot.ProgressLog(tmp_path)
    )
    assert not report["complete_pair"] and client.attempts == 1
    assert report["arms"][pilot.ARMS[0]]["status"] == "failed"
    assert report["arms"][pilot.ARMS[1]]["status"] == "not_executed"
    assert all(report["arms"][a]["feedback"] is None for a in pilot.ARMS)
    assert "offline_gold_answers" not in report


def test_author_caps_share_one_ledger_and_restore_config(tmp_path):
    config = ChatConfig("https://example.invalid/v1", "fake", "NEVER_READ", 10)
    delegate = SimpleNamespace(config=config, transport_source="fake_test", attempts=0)
    observed = []

    def complete(messages, *, trace_id, prompt_version):
        observed.append(delegate.config)
        delegate.attempts += 1
        return ChatResponse(
            "Answer: test", "fake", "fake", None, None, 10, 4, 0, Path("synthetic"), "fake_test"
        )

    delegate.complete = complete
    client = pilot.AuthorBudgetClient(delegate, PriceLimits(), tmp_path / "journal")
    for cap in (64, 128, 256):
        client.complete_author(
            [{"role": "user", "content": "synthetic"}],
            trace_id=f"test-{cap}",
            prompt_version="test",
            max_output_tokens=cap,
            temperature=0,
            top_p=1,
        )
        assert client.config == config and delegate.config == config
    assert [c.max_output_tokens for c in observed] == [64, 128, 256]
    assert len(client.calls) == 3 and client.report()["api_requests"] == 3
    assert len(list((tmp_path / "journal").glob("*_intent.json"))) == 3


@pytest.mark.parametrize(
    "json_stages,schema_stages", [(False, False), (True, False), (False, True)]
)
def test_json_mode_only_applies_to_author_structured_stages(tmp_path, json_stages, schema_stages):
    config = ChatConfig("https://example.invalid/v1", "fake", "NEVER_READ", 10)
    delegate = SimpleNamespace(config=config, transport_source="fake_test", attempts=0)
    observed = []

    def complete(messages, *, trace_id, prompt_version):
        observed.append(delegate.config)
        delegate.attempts += 1
        return ChatResponse(
            "synthetic", "fake", "fake", None, None, 10, 4, 0, Path("unused"), "fake_test"
        )

    delegate.complete = complete
    client = pilot.AuthorBudgetClient(delegate, PriceLimits(), tmp_path / "journal")
    client.json_stages = json_stages
    client.schema_stages = schema_stages
    for stage, cap in (("judge", 256), ("extract", 64), ("answer", 128)):
        client.complete_author(
            [{"role": "user", "content": "synthetic"}],
            trace_id=f"v2-{stage}",
            prompt_version=PROMPT_VERSIONS[stage],
            max_output_tokens=cap,
            temperature=0,
            top_p=1,
        )
        assert client.config == config and delegate.config == config
    assert [c.json_object_mode for c in observed] == [json_stages, json_stages, False]
    assert [c.json_schema_mode for c in observed] == [schema_stages, schema_stages, False]
    assert [c.max_output_tokens for c in observed] == [256, 64, 128]
    assert len(client.calls) == 3


@pytest.mark.skipif(not UPSTREAM.exists(), reason="author snapshot not redistributed")
def test_plan_never_reads_secret_or_calls_models(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "load_debug", lambda _: ([example()] * 8, {"synthetic": True}))
    monkeypatch.setattr(pilot, "reconcile_history", lambda *a, **k: {"prior_reserved_cny": 1})
    monkeypatch.setattr(pilot, "source_snapshot", lambda *a: {"sha256": "synthetic"})
    monkeypatch.setattr(pilot, "_git_state", lambda: {"commit": "synthetic"})
    monkeypatch.setattr(pilot, "read_local_bailian_settings", lambda *a: pytest.fail("no keys"))
    output = tmp_path / "plan"
    assert (
        pilot.main(
            [
                "--upstream",
                str(UPSTREAM),
                "--manifest",
                "unused",
                "--runs-root",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    plan = json.loads((output / "plan.json").read_text())
    assert plan["worst_case_calls"] == 88
    assert not plan["official_dev_test_used"]


def test_existing_claim_rejects_before_data_access(tmp_path, monkeypatch):
    (tmp_path / f"{pilot.RUN_ID}.claim.json").write_text("{}")
    monkeypatch.setattr(pilot, "load_debug", lambda _: pytest.fail("no data"))
    with pytest.raises(ValueError, match="already claimed"):
        pilot.main(
            [
                "--upstream",
                str(UPSTREAM),
                "--manifest",
                "unused",
                "--runs-root",
                str(tmp_path),
                "--output",
                str(tmp_path / pilot.RUN_ID),
                "--allow-network",
            ]
        )


@pytest.mark.skipif(not UPSTREAM.exists(), reason="author snapshot not redistributed")
@pytest.mark.parametrize("schema_stages", [False, True])
def test_v2_v3_plan_keeps_prior_cost_and_separate_identity(tmp_path, monkeypatch, schema_stages):
    observed_roots = []

    def history(root, *, reviewed_extra_ledgers):
        observed_roots.extend(reviewed_extra_ledgers)
        return {"prior_reserved_cny": 1}

    monkeypatch.setattr(pilot, "load_debug", lambda _: ([example()] * 8, {"synthetic": True}))
    monkeypatch.setattr(pilot, "reconcile_history", history)
    monkeypatch.setattr(pilot, "source_snapshot", lambda *a: {"sha256": "synthetic"})
    monkeypatch.setattr(pilot, "_git_state", lambda: {"commit": "synthetic"})
    monkeypatch.setattr(pilot, "read_local_bailian_settings", lambda *a: pytest.fail("no keys"))
    (tmp_path / f"{pilot.RUN_ID}.claim.json").write_text("{}")
    output = tmp_path / "v2-plan"
    assert (
        pilot.main(
            [
                "--upstream",
                str(UPSTREAM),
                "--manifest",
                "unused",
                "--runs-root",
                str(tmp_path),
                "--output",
                str(output),
                "--schema-stages" if schema_stages else "--json-stages",
            ]
        )
        == 0
    )
    plan = json.loads((output / "plan.json").read_text())
    assert plan["run_id"] == (pilot.SCHEMA_RUN_ID if schema_stages else pilot.JSON_RUN_ID)
    format_key = "json_schema_mode" if schema_stages else "json_object_mode"
    assert plan["decoding"][format_key] == "judge/extract only"
    expected_roots = [*pilot.HISTORY_ROOTS, f"{pilot.RUN_ID}/final_budget.json"]
    if schema_stages:
        expected_roots.append(f"{pilot.JSON_RUN_ID}/final_budget.json")
        assert set(plan["output_schemas"]) == {"judge", "extract"}
        assert plan["v3_decision"]
    else:
        assert plan["v2_authorization"]
    assert observed_roots == expected_roots
    assert not plan["official_dev_test_used"]
