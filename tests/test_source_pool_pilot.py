"""Source-only contracts with explicit mock calls; no credentials or live requests."""

import copy
import json
from dataclasses import replace

import pytest
from test_pre_pilot import (
    PRIVATE_ERROR,
    ScriptedClient,
    example,
    example_steps,
    extraction_step,
    read_json,
)

from growrag.experiments import source_pool_pilot as pool
from growrag.experiments.api_client import APIRequestError, ChatConfig
from growrag.experiments.budget import BudgetedChatClient, PriceLimits
from growrag.experiments.data_protocol import role_for_question
from growrag.experiments.pre_pilot import EXTRACTION_VERSION, extract_pre_card
from growrag.experiments.representation_manifest import build_representation_manifest
from growrag.experiments.source_pool_pilot import run_source_pool, validate_source_pool_inputs


@pytest.fixture(scope="module")
def population():
    examples = []
    for index in range(500):
        item = example(f"source-{index}", f"Institution{index}")
        examples.append(
            replace(
                item,
                question=replace(item.question, dataset="hotpotqa-synthetic-train"),
                question_type="bridge" if index % 2 == 0 else "comparison",
            )
        )
    return tuple(examples)


def source_batch(population, count=4):
    manifest = build_representation_manifest(
        population,
        source_sha256="a" * 64,
        official_split="train",
        complete_train_declared=True,
        source_count=count,
        source_cap=max(8, count),
        target_count=4,
        debug_count=2,
    )
    by_id = {example.question.question_id: example for example in population}
    return tuple(by_id[qid] for qid in manifest["selected"]["source"]), manifest


def budgeted(steps, *, limits=None):
    return BudgetedChatClient(ScriptedClient(steps), limits or PriceLimits(budget_cny=5))


def test_sources_only_admission_exact_body_counts_and_no_gold_in_prompts(tmp_path, population):
    sources, manifest = source_batch(population)
    steps = []
    for number, item in enumerate(sources):
        steps.extend(example_steps(item, base_answer="1900" if number == 1 else "1800"))
        if number != 1:
            steps.append(extraction_step(item))
    client = budgeted(steps)
    result = run_source_pool(sources, client, tmp_path / "pool", manifest=manifest)
    summary = result.summary
    assert not client.delegate.steps
    assert summary["status"] == "completed"
    assert summary["processed_source_count"] == 4
    assert summary["eligible_source_count"] == summary["extracted_count"] == 3
    assert len(result.cards) == len(result.views) == summary["candidate_count"] == 3
    assert len(result.source_records) == 4
    assert summary["nonduplicate_canonical_body_count"] == 1
    assert summary["rejection_reason_counts"]["no_answer_or_support_increment"] == 1
    assert summary["manual_pattern_review_required"] is True
    assert summary["proposed_feasibility_gate"]["passed"] is None
    assert summary["proposed_feasibility_gate"]["pattern_condition_met"] is None
    assert summary["targets_executed"] == 0
    assert summary["automatic_expansion"] is False
    assert not (tmp_path / "pool" / "targets").exists()
    assert len(list((tmp_path / "pool" / "sources").iterdir())) == 4
    assert read_json(tmp_path / "pool" / "final_budget.json")["api_requests"] == 15
    assert read_json(tmp_path / "pool" / "sources" / "000" / "view.json")
    for call in client.delegate.calls:
        text = json.dumps(call["messages"])
        assert "GOLD_ONLY_" not in text and "supporting_facts" not in text
        assert "answer_em" not in text and "support_recall" not in text
        if call["version"] == EXTRACTION_VERSION:
            assert set(call["payload"]) == {"original_query", "executed_query"}


def test_64_source_preflight_does_not_use_old_32_limit(population):
    sources, manifest = source_batch(population, 64)
    validate_source_pool_inputs(sources, manifest)


@pytest.mark.parametrize(
    "problem",
    [
        "split",
        "schema",
        "wrong_gold",
        "missing_gold",
        "ids",
        "source_role",
        "dataset",
        "bad_sha",
        "source_cap",
        "expansion",
        "counts",
        "target_overlap",
        "duplicate_text",
    ],
)
def test_all_preflight_errors_precede_calls_or_output(tmp_path, population, problem):
    sources, manifest = source_batch(population)
    manifest = copy.deepcopy(manifest)
    if problem == "split":
        manifest["official_split"] = "test"
    elif problem == "schema":
        manifest["schema_version"] = "growrag-pre-query-manifest-v1"
    elif problem == "wrong_gold":
        sources = (
            replace(sources[0], gold=replace(sources[0].gold, question_id="other")),
            *sources[1:],
        )
    elif problem == "missing_gold":
        sources = (replace(sources[0], gold=None), *sources[1:])
    elif problem == "ids":
        manifest["selected"]["source"].reverse()
    elif problem == "source_role":
        wrong = next(e for e in population if role_for_question(e.question.text) != "memory_seed")
        sources = (
            replace(sources[0], question=replace(sources[0].question, text=wrong.question.text)),
            *sources[1:],
        )
    elif problem == "dataset":
        sources = (
            replace(sources[0], question=replace(sources[0].question, dataset="hotpot-test")),
            *sources[1:],
        )
    elif problem == "bad_sha":
        manifest["source_sha256"] = "not-a-hash"
    elif problem == "source_cap":
        manifest["source_cap"] = 258
    elif problem == "expansion":
        manifest["source_expansion_order"].reverse()
    elif problem == "counts":
        manifest["selected_counts"]["source"]["by_type"]["bridge"] = 0
    elif problem == "target_overlap":
        manifest["selected"]["target"] = [sources[0].question.question_id]
    else:
        sources = (
            sources[0],
            replace(
                sources[1], question=replace(sources[1].question, text=sources[0].question.text)
            ),
            *sources[2:],
        )
    client = budgeted([])
    with pytest.raises((TypeError, ValueError)):
        run_source_pool(sources, client, tmp_path / "pool", manifest=manifest)
    assert client.attempts == 0 and not (tmp_path / "pool").exists()


def test_bad_extraction_json_is_logged_without_retry_or_fake_card(tmp_path, population):
    sources, manifest = source_batch(population, 2)
    client = budgeted(
        [
            *example_steps(sources[0]),
            extraction_step(sources[0], "NOT JSON"),
            *example_steps(sources[1]),
            extraction_step(sources[1]),
        ]
    )
    result = run_source_pool(sources, client, tmp_path, manifest=manifest)
    assert result.summary["status"] == "completed"
    assert result.summary["extracted_count"] == len(result.cards) == 1
    assert result.summary["eligible_source_count"] == 2
    assert result.summary["rejection_reason_counts"]["invalid_extraction"] == 1
    assert result.summary["sources"][0]["extraction_event"]["status"] == "error"
    assert (tmp_path / "sources" / "000" / "extraction_event.json").exists()
    assert not (tmp_path / "sources" / "000" / "card.json").exists()
    assert client.attempts == 8


@pytest.mark.parametrize("phase", ["source", "extract", "budget"])
def test_transport_or_budget_failure_stops_before_next_source(tmp_path, population, phase):
    sources, manifest = source_batch(population)
    failure = APIRequestError(PRIVATE_ERROR, transport_source="mock", api_requests=0)
    if phase == "extract":
        step = extraction_step(sources[0], failure)
        steps = [*example_steps(sources[0]), step]
    else:
        step = example_steps(sources[0])[0]
        step.output = failure
        steps = [step]
    client = budgeted(steps, limits=PriceLimits(budget_cny=1e-12 if phase == "budget" else 5))
    result = run_source_pool(sources, client, tmp_path, manifest=manifest)
    assert result.summary["status"] == "stopped"
    assert result.summary["processed_source_count"] == 1
    assert len(result.summary["unprocessed_source_ids"]) == 3
    assert not (tmp_path / "sources" / "001").exists()
    assert client.attempts == {"source": 1, "extract": 4, "budget": 0}[phase]
    assert not result.cards
    assert read_json(tmp_path / "final_budget.json")["block_reason"] == client.block_reason
    for path in tmp_path.rglob("*.json"):
        assert PRIVATE_ERROR not in path.read_text(encoding="utf-8")


def test_shared_prior_ledger_is_retained_without_reset(tmp_path, population):
    sources, manifest = source_batch(population, 2)
    client = budgeted(
        [
            extraction_step(sources[0]),
            *example_steps(sources[0]),
            extraction_step(sources[0]),
            *example_steps(sources[1]),
            extraction_step(sources[1]),
        ]
    )
    extract_pre_card(client, sources[0].question, "prior separately authorized mock request")
    previous_cost = client.estimated_actual_cny
    result = run_source_pool(sources, client, tmp_path, manifest=manifest)
    assert result.summary["budget_at_start"]["api_requests"] == 1
    assert result.summary["budget"]["api_requests"] == 9
    assert client.estimated_actual_cny > previous_cost


def test_persistence_failure_prevents_further_charges(tmp_path, population, monkeypatch):
    sources, manifest = source_batch(population)
    client = budgeted([*example_steps(sources[0]), extraction_step(sources[0])])
    original = pool.write_json

    def failed_write(path, value):
        if path.name == "card.json":
            raise OSError("synthetic disk failure")
        original(path, value)

    monkeypatch.setattr(pool, "write_json", failed_write)
    with pytest.raises(OSError):
        run_source_pool(sources, client, tmp_path, manifest=manifest)
    assert client.attempts == 4
    assert not (tmp_path / "sources" / "001").exists()
    assert (tmp_path / "final_budget.json").exists()
    assert not (tmp_path / "source_pool.json").exists()


def test_no_implicit_resume_of_existing_or_blocked_pool(tmp_path, population):
    sources, manifest = source_batch(population)
    client = budgeted([])
    (tmp_path / "sources").mkdir()
    with pytest.raises(FileExistsError):
        run_source_pool(sources, client, tmp_path, manifest=manifest)
    client.block_reason = "transport_failure"
    with pytest.raises(ValueError, match="already blocked"):
        run_source_pool(sources, client, tmp_path / "new", manifest=manifest)
    assert client.attempts == 0


def test_real_client_requires_opt_in_without_network(tmp_path, population):
    sources, manifest = source_batch(population)
    client = budgeted([])
    client.transport_source = "live_api"
    client.config = ChatConfig(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        pool.PILOT_MODEL,
        "UNUSED_TEST_KEY",
        max_calls=256,
        max_output_tokens=768,
    )
    with pytest.raises(ValueError, match="allow_real"):
        run_source_pool(sources, client, tmp_path, manifest=manifest)
    assert client.attempts == 0
