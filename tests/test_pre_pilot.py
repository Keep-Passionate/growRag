"""Small PRE batch contracts with scripted ChatResponse objects, never a live API.

The answer strings below are fixtures, not evidence of model effectiveness.
Local BM25, source admission, card compilation and artifact writing are real.
"""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from growrag.experience.cards import CardLifecycle
from growrag.experiments import pre_pilot
from growrag.experiments.api_client import APIRequestError, ChatConfig, ChatResponse
from growrag.experiments.budget import BudgetedChatClient, PriceLimits
from growrag.experiments.hotpot import parse_hotpot_example
from growrag.experiments.llm_adapters import READER_PROMPT_VERSION
from growrag.experiments.pre_pilot import (
    EXTRACTION_VERSION,
    PRE_INTENT,
    extract_pre_card,
    lexical_select,
    run_example,
    run_pre_batch,
)
from growrag.experiments.protocol import Action, BackendCallError, RuntimeQuestion
from growrag.query_actions import PAIRED_ACTION_PROMPT_VERSION
from growrag.query_operators import ParaphraseBody

CARD_FIELDS = {
    "query_pattern": "A question asks when an organization was established.",
    "preconditions": ["The question explicitly identifies an organization."],
    "contraindications": ["The request is about a different type of event."],
    "body": {
        "rewrite_rule": "Express the establishment question as an equivalent founding phrase.",
        "preserve": ["original entity", "requested time scope", "relation direction"],
    },
}
MANIFEST = {"schema_version": "synthetic-test-only", "official_split": "train"}
PRIVATE_ERROR = "PRIVATE_EXCEPTION_BODY_MUST_NOT_BE_PERSISTED"


def example(identifier, entity, *, text=None):
    result = parse_hotpot_example(
        {
            "_id": identifier,
            "question": text or f"When was {entity} University established?",
            "answer": "1900",
            "supporting_facts": [[entity, 0]],
            "context": [[entity, [f"{entity} University was founded in 1900."]]],
        },
        dataset="synthetic-train-only",
    )
    # This alternative exists only in the offline gold record, never the corpus.
    return replace(
        result,
        gold=replace(result.gold, answers=("1900", f"GOLD_ONLY_{identifier}")),
    )


def action_order(identifier):
    return tuple(
        sorted(
            Action,
            key=lambda action: hashlib.sha256(f"42:{identifier}:{action.value}".encode()).digest(),
        )
    )


@dataclass
class Step:
    version: str
    question: str
    output: object
    action: Action | None = None
    directory: Path | None = None
    identifier: str | None = None


def example_steps(item, *, reuse=False, base_answer="1800", directory=None):
    steps = []
    for action in action_order(item.question.question_id):
        if action is Action.REUSE and not reuse:
            continue
        if action is not Action.BASE:
            steps.append(
                Step(
                    PAIRED_ACTION_PROMPT_VERSION,
                    item.question.text,
                    {"query": item.question.text + " founding chronology"},
                    action,
                    directory,
                    item.question.question_id,
                )
            )
        steps.append(
            Step(
                READER_PROMPT_VERSION,
                item.question.text,
                {
                    "answer": base_answer if action is Action.BASE else "1900",
                    "cited_evidence_ids": [item.candidate_context[0].evidence_id],
                },
                action,
                directory,
                item.question.question_id,
            )
        )
    return steps


def extraction_step(item, output=None):
    return Step(EXTRACTION_VERSION, item.question.text, CARD_FIELDS if output is None else output)


class ScriptedClient:
    """An explicit mock transport, with no network client or credential access."""

    transport_source = "mock"

    def __init__(self, steps, *, before_call=None):
        self.config = ChatConfig(
            "https://example.invalid/v1",
            "scripted-model-v1",
            "UNUSED_TEST_KEY",
            max_calls=256,
            max_output_tokens=128,
            temperature=0,
        )
        self.steps = list(steps)
        self.calls = []
        self.before_call = before_call

    @property
    def attempts(self):
        return len(self.calls)

    def complete(self, messages, *, trace_id, prompt_version):
        assert self.steps, "unexpected additional model invocation"
        step = self.steps.pop(0)
        assert prompt_version == step.version
        assert [message["role"] for message in messages] == ["system", "user"]
        payload = json.loads(messages[1]["content"])
        assert payload.get("original_query", payload.get("original_question")) == step.question
        if self.before_call:
            self.before_call(step, payload)
        self.calls.append(
            {"messages": messages, "payload": payload, "version": prompt_version, "step": step}
        )
        if isinstance(step.output, Exception):
            raise step.output
        return ChatResponse(
            step.output if isinstance(step.output, str) else json.dumps(step.output),
            self.config.model,
            self.config.model,
            f"mock-response-{self.attempts}",
            f"mock-request-{self.attempts}",
            11,
            7,
            0.0,
            Path("MOCK_ONLY_NOT_A_LIVE_AUDIT.json"),
            "mock",
        )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_prior_arms_saved(step, _payload):
    if step.directory is None:
        return
    assert (step.directory / "spec.json").exists()
    assert not (step.directory / "completed").exists()
    order = action_order(step.identifier)
    for action in order[: order.index(step.action)]:
        assert (step.directory / f"arm_{action.value}.json").exists()
    assert not (step.directory / f"arm_{step.action.value}.json").exists()


def test_complete_source_admission_extraction_freeze_and_target_paths(tmp_path, monkeypatch):
    source = example("source-a", "Northbridge")
    rejected = example("source-b", "Southbridge")
    target = example("target-a", "Eastbridge")
    unmatched = example(
        "target-b", "Westbridge", text="Which year did the orb acquire its pigment?"
    )
    unmatched = replace(
        unmatched,
        candidate_context=(
            replace(unmatched.candidate_context[0], text="The orb acquired its pigment in 1900."),
        ),
    )
    directory = tmp_path / "batch"
    captured_candidates, frozen_snapshots = [], []
    original_select = pre_pilot.lexical_select

    def selection_spy(question, candidates):
        assert not any(step.question == question.text for step in (c["step"] for c in client.calls))
        assert (directory / "sources" / "001" / "admission.json").exists()
        frozen_snapshots.append((directory / "frozen_library.json").read_bytes())
        captured_candidates.append(tuple(candidates))
        return original_select(question, candidates)

    monkeypatch.setattr(pre_pilot, "lexical_select", selection_spy)

    def before_call(step, payload):
        assert_prior_arms_saved(step, payload)
        if step.question in {target.question.text, unmatched.question.text}:
            index = "000" if step.question == target.question.text else "001"
            assert (directory / "targets" / f"{index}_selection.json").exists()
            assert (directory / "frozen_library.json").read_bytes() == frozen_snapshots[0]

    steps = [
        *example_steps(source, directory=directory / "sources" / "000"),
        extraction_step(source),
        *example_steps(rejected, base_answer="1900", directory=directory / "sources" / "001"),
        *example_steps(target, reuse=True, directory=directory / "targets" / "000"),
        *example_steps(unmatched, directory=directory / "targets" / "001"),
    ]
    client = ScriptedClient(steps, before_call=before_call)
    summary = run_pre_batch(
        (source, rejected), (target, unmatched), client, directory, manifest=MANIFEST
    )

    assert not client.steps
    assert summary["execution_kind"] == "mock" and summary["candidate_count"] == 1
    assert summary["sources"][0]["admission"]["eligible"]
    assert not summary["sources"][1]["admission"]["eligible"]
    assert summary["sources"][1]["card_id"] is None
    library = read_json(directory / "frozen_library.json")
    assert summary["library_sha256"] == pre_pilot.digest(library)
    assert len(library["cards"]) == len(library["views"]) == 1
    assert (
        frozen_snapshots[0]
        == frozen_snapshots[1]
        == (directory / "frozen_library.json").read_bytes()
    )
    card = library["cards"][0]
    assert card["lifecycle_state"] == CardLifecycle.CANDIDATE
    assert card["validation"]["matched_trials"] == 0
    assert card["provenance"]["source_type"] == "pre_independent_pair.v1"
    assert [ref["source_id"] for ref in card["provenance"]["source_refs"]] == [
        "pre-source-source-a"
    ]
    assert library["views"][0]["source_query_ids"] == [source.question.question_id]
    assert captured_candidates[0] == captured_candidates[1]
    selected, no_card = summary["targets"]
    assert selected["selection"]["proposed_route"] == "REUSE"
    assert selected["selection"]["conditions_status"] == "not_semantically_verified"
    assert selected["report"]["arms"]["REUSE"]["executed_action"] == "REUSE"
    assert no_card["selection"]["proposed_route"] == "BASE"
    assert no_card["report"]["arms"]["BASE"]["completed"]
    skipped = no_card["report"]["arms"]["REUSE"]
    assert skipped["stop_reason"] == "no_eligible_memory"
    assert skipped["executed_action"] is skipped["answer"] is skipped["feedback"] is None
    assert skipped["rag_calls"] == skipped["rewrite_calls"] == 0
    assert skipped["usage"] == {"input_tokens": 0, "output_tokens": 0, "api_requests": 0}
    assert not no_card["report"]["contrasts"]["REUSE_minus_BASE"]["available"]

    extracts = [call for call in client.calls if call["version"] == EXTRACTION_VERSION]
    assert len(extracts) == 1
    assert extracts[0]["payload"] == {
        "original_query": source.question.text,
        "executed_query": source.question.text + " founding chronology",
    }
    rewrites = [call for call in client.calls if call["version"] == PAIRED_ACTION_PROMPT_VERSION]
    assert len({call["messages"][0]["content"] for call in rewrites}) == 1
    for call in rewrites:
        payload = call["payload"]
        assert payload["current_evidence"] == payload["previous_queries"] == []
        assert payload["form"] == "paraphrase" and payload["intent"] == PRE_INTENT
        assert ("optional_historical_procedure" in payload) == (call["step"].action is Action.REUSE)
    for call in client.calls:
        serialized = json.dumps(call["messages"])
        assert "GOLD_ONLY_" not in serialized
        assert "supporting_facts" not in serialized
        assert "answer_em" not in serialized and "support_recall" not in serialized
        if call["step"].question in {target.question.text, unmatched.question.text}:
            assert source.question.text not in serialized
            assert "1800" not in serialized

    # A frozen source card is isolated by both identity and normalized text.
    candidates = captured_candidates[0]
    for question in (source.question, RuntimeQuestion("changed-id", source.question.text.upper())):
        view, selection = lexical_select(question, candidates)
        assert view is None and selection["candidates"] == []


def test_no_successful_source_does_not_extract_or_fabricate_a_reuse_arm(tmp_path):
    source, target = example("source", "Northbridge"), example("target", "Eastbridge")
    client = ScriptedClient([*example_steps(source, base_answer="1900"), *example_steps(target)])
    summary = run_pre_batch((source,), (target,), client, tmp_path, manifest=MANIFEST)
    assert not client.steps and len(client.calls) == 6
    assert summary["candidate_count"] == 0
    assert read_json(tmp_path / "frozen_library.json")["cards"] == []
    assert "no_answer_or_support_increment" in summary["sources"][0]["admission"]["reasons"]
    assert summary["sources"][0]["extraction_event"] is None
    for folder in ("sources", "targets"):
        result = read_json(tmp_path / folder / "000" / "arm_REUSE.json")["result"]
        assert result["state"]["rounds"] == result["events"] == []
        assert result["stop_reason"] == "no_eligible_memory"


def test_run_example_persists_each_arm_before_next_call_and_refuses_overwrite(tmp_path):
    item = example("persistence", "Northbridge")
    directory = tmp_path / "one"
    client = ScriptedClient(
        example_steps(item, directory=directory), before_call=assert_prior_arms_saved
    )
    run = run_example(item, client, None, directory)
    assert tuple(arm.planned_action for arm in run.arms) == action_order(item.question.question_id)
    assert not client.steps
    assert {p.name for p in (directory / "completed").iterdir()} == {
        "run.json",
        "report.json",
        "report.md",
    }
    count = len(client.calls)
    with pytest.raises(FileExistsError):
        run_example(item, client, None, directory)
    assert len(client.calls) == count


def test_arm_persistence_failure_stops_before_any_later_model_call(tmp_path, monkeypatch):
    item = example("write-failure", "Northbridge")
    directory = tmp_path / "one"
    client = ScriptedClient(example_steps(item))
    original_write = pre_pilot.write_json
    first = action_order(item.question.question_id)[0]

    def failing_write(path, value):
        if path.name == f"arm_{first.value}.json":
            raise OSError("synthetic disk unavailable")
        original_write(path, value)

    monkeypatch.setattr(pre_pilot, "write_json", failing_write)
    with pytest.raises(OSError, match="disk unavailable"):
        run_example(item, client, None, directory)
    expected_calls = {Action.BASE: 1, Action.FRESH: 2, Action.REUSE: 0}[first]
    assert len(client.calls) == expected_calls
    assert not (directory / "completed").exists()


@pytest.mark.parametrize("failed_version", [PAIRED_ACTION_PROMPT_VERSION, READER_PROMPT_VERSION])
def test_source_component_failure_retains_records_and_continues_other_examples(
    tmp_path, failed_version
):
    source, target = example("broken-source", "Northbridge"), example("target", "Eastbridge")
    steps = example_steps(source)
    index = next(i for i, step in enumerate(steps) if step.version == failed_version)
    failed_action = steps[index].action
    steps[index].output = ValueError(PRIVATE_ERROR)
    if failed_version == PAIRED_ACTION_PROMPT_VERSION:
        del steps[index + 1]  # Failed rewrite cannot incur a reader call.
    client = ScriptedClient([*steps, *example_steps(target)])
    summary = run_pre_batch((source,), (target,), client, tmp_path, manifest=MANIFEST)
    assert not client.steps and summary["candidate_count"] == 0
    assert not summary["sources"][0]["admission"]["eligible"]
    source_report = read_json(tmp_path / "sources" / "000" / "completed" / "report.json")
    assert not source_report["arms"][failed_action.value]["completed"]
    assert summary["targets"][0]["report"]["completed_arm_count"] == 2
    assert (tmp_path / "sources" / "000" / "source_record.json").exists()
    for path in tmp_path.rglob("*.json"):
        assert PRIVATE_ERROR not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", [ValueError(PRIVATE_ERROR), "not valid JSON"])
def test_extraction_failure_is_recorded_without_retry_or_candidate(tmp_path, failure):
    source, target = example("source", "Northbridge"), example("target", "Eastbridge")
    client = ScriptedClient(
        [*example_steps(source), extraction_step(source, failure), *example_steps(target)]
    )
    summary = run_pre_batch((source,), (target,), client, tmp_path, manifest=MANIFEST)
    row = summary["sources"][0]
    assert not client.steps and len(client.calls) == 7
    assert row["admission"]["eligible"] and row["card_id"] is None
    assert row["extraction_event"]["status"] == "error"
    assert summary["candidate_count"] == 0
    assert summary["targets"][0]["selection"]["proposed_route"] == "BASE"
    assert sum(call["version"] == EXTRACTION_VERSION for call in client.calls) == 1
    for path in tmp_path.rglob("*.json"):
        assert PRIVATE_ERROR not in path.read_text(encoding="utf-8")


def test_extract_card_accepts_only_query_strings_and_returns_typed_body():
    item = example("extract", "Northbridge")
    client = ScriptedClient([extraction_step(item)])
    result = extract_pre_card(client, item.question, "alternative organization founding date")
    assert isinstance(result.value["body"], ParaphraseBody)
    assert result.value["preconditions"] == tuple(CARD_FIELDS["preconditions"])
    assert result.usage.api_requests == 0 and result.transport_source == "mock"
    assert set(client.calls[0]["payload"]) == {"original_query", "executed_query"}
    assert client.calls[0]["payload"]["executed_query"] == "alternative organization founding date"


@pytest.mark.parametrize(
    "output",
    [
        [],
        {**CARD_FIELDS, "answer": "1900"},
        {**CARD_FIELDS, "query_pattern": "x" * 401},
        {**CARD_FIELDS, "preconditions": []},
        {**CARD_FIELDS, "contraindications": ["one"] * 4},
        {**CARD_FIELDS, "body": {"rewrite_rule": "rule", "preserve": []}},
        '{"query_pattern":"one","query_pattern":"duplicate"}',
    ],
)
def test_malformed_extraction_fails_closed_after_one_mock_response(output):
    item = example("extract", "Northbridge")
    client = ScriptedClient([extraction_step(item, output)])
    with pytest.raises(BackendCallError, match="invalid PRE card extraction") as caught:
        extract_pre_card(client, item.question, "alternative founding date")
    assert len(client.calls) == 1 and not client.steps
    assert caught.value.usage.api_requests == 0
    assert caught.value.request_id == "mock-request-1"


@pytest.mark.parametrize("problem", ["same_id", "same_text", "wrong_gold", "missing_gold"])
def test_invalid_source_target_inputs_fail_before_any_model_call(tmp_path, problem):
    source, target = example("source", "Northbridge"), example("target", "Eastbridge")
    if problem == "same_id":
        target = replace(target, question=replace(target.question, question_id="source"))
    elif problem == "same_text":
        target = replace(
            target, question=replace(target.question, text=source.question.text.upper())
        )
    elif problem == "wrong_gold":
        target = replace(target, gold=replace(target.gold, question_id="other"))
    else:
        target = replace(target, gold=None)
    client = ScriptedClient([])
    with pytest.raises((TypeError, ValueError)):
        run_pre_batch((source,), (target,), client, tmp_path / "batch", manifest=MANIFEST)
    assert client.calls == [] and not (tmp_path / "batch").exists()


@pytest.mark.parametrize("failure_kind", ["transport", "ordinary", "budget"])
def test_shared_budget_fault_blocks_all_subsequent_transport_attempts(tmp_path, failure_kind):
    source, target = example("source", "Northbridge"), example("target", "Eastbridge")
    first = example_steps(source)[0]
    first.output = (
        APIRequestError(PRIVATE_ERROR, api_requests=0, transport_source="mock")
        if failure_kind == "transport"
        else RuntimeError(PRIVATE_ERROR)
    )
    delegate = ScriptedClient([first])
    limits = PriceLimits(budget_cny=0.00000001 if failure_kind == "budget" else 5.0)
    client = BudgetedChatClient(delegate, limits)
    summary = run_pre_batch((source,), (target,), client, tmp_path, manifest=MANIFEST)
    assert delegate.attempts == (0 if failure_kind == "budget" else 1)
    assert summary["candidate_count"] == 0
    assert summary["targets"][0]["report"]["completed_arm_count"] == 0
    assert (
        summary["budget"]["block_reason"]
        == {
            "transport": "transport_failure",
            "ordinary": "unexpected_failure",
            "budget": "estimated_budget_limit",
        }[failure_kind]
    )
    assert "ALL source BASE/FRESH" in summary["budget"]["cost_scope"]
    assert read_json(tmp_path / "final_budget.json")["block_reason"] == client.block_reason
    for folder in ("sources", "targets"):
        assert len(list((tmp_path / folder / "000").glob("arm_*.json"))) == 3
    for path in tmp_path.rglob("*.json"):
        assert PRIVATE_ERROR not in path.read_text(encoding="utf-8")


def test_existing_partial_batch_is_not_automatically_recharged(tmp_path):
    source, target = example("source", "Northbridge"), example("target", "Eastbridge")
    (tmp_path / "sources").mkdir()
    client = ScriptedClient([])
    with pytest.raises(FileExistsError, match="no automatic rerun"):
        run_pre_batch((source,), (target,), client, tmp_path, manifest=MANIFEST)
    assert client.calls == []
