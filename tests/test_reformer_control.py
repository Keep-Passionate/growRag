"""Offline example-field controls; no paid client or redistributed author prompts."""

import copy
import inspect
import json
from pathlib import Path

import pytest

from growrag.experiments.api_client import APIRequestError, ChatResponse
from growrag.experiments.reformer_api import PROMPT_VERSIONS
from growrag.experiments.reformer_control import (
    CONTROL_ID,
    SELECT_PROMPT_VERSION,
    ReFormeRControlAPI,
    parse_pattern_id,
    selection_digest,
)
from growrag.experiments.s2g_author_api import AuthorDocument

UPSTREAM = Path("external/reformer_author_snapshot/aminbigdeli-ReFormeR-72e5245")
READER = Path("external/s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6")
QUESTION = "Which synthetic town is older?"
REWRITE = "Northbridge Eastbridge foundation dates"
ANSWER = "Answer: Northbridge\nRationale: The supplied sentence states this."
DOCS = tuple(
    AuthorDocument(f"doc-{i}", f"Title {i}", f"Unique retrieved text marker {i}.") for i in range(6)
)


class FakeClient:
    def __init__(self, outputs):
        self.outputs, self.calls, self.block_reason = list(outputs), [], None

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": copy.deepcopy(messages), **kwargs})
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return ChatResponse(
            content=output,
            requested_model="test-fake-model",
            returned_model="test-fake-model",
            response_id="synthetic-response",
            request_id="synthetic-request",
            input_tokens=20,
            output_tokens=10,
            elapsed_seconds=0.0,
            audit_path=Path("synthetic-no-network"),
            transport_source="test_fake",
        )


@pytest.fixture
def upstream():
    if not UPSTREAM.exists() or not READER.exists():
        pytest.skip("pinned author snapshots are intentionally not redistributed")
    return UPSTREAM


def make_adapter(upstream, outputs, *, docs=DOCS, event_callback=None):
    client, calls = FakeClient(outputs), []

    def retrieve(query, k):
        calls.append((query, k))
        return docs[:k]

    adapter = ReFormeRControlAPI(upstream, READER, client, retrieve, event_callback=event_callback)
    return adapter, client, calls


def paired(upstream, *, event_callback=None):
    adapter, client, retrieval = make_adapter(
        upstream,
        ['{"pattern_id": 2}', REWRITE, ANSWER, REWRITE, ANSWER],
        event_callback=event_callback,
    )
    selection = adapter.select(QUESTION, "q1/selection")
    included = adapter.run_selected(QUESTION, "q1/with", selection, include_examples=True)
    excluded = adapter.run_selected(QUESTION, "q1/without", selection, include_examples=False)
    return adapter, client, retrieval, selection, included, excluded


@pytest.mark.parametrize("pattern_id", range(10))
def test_each_zero_based_id_is_accepted(pattern_id):
    assert parse_pattern_id(json.dumps({"pattern_id": pattern_id})) == pattern_id


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '```json\n{"pattern_id":0}\n```',
        "[]",
        "null",
        "{}",
        '{"pattern_id": true}',
        '{"pattern_id": 1.0}',
        '{"pattern_id": "1"}',
        '{"pattern_id": -1}',
        '{"pattern_id": 10}',
        '{"pattern_id": null}',
        '{"pattern_id": 0, "rule": "injected"}',
        '{"pattern_id": 0, "pattern_id": 1}',
        '{"pattern_id": NaN}',
        '{"pattern_id": 0} trailing explanation',
        None,
    ],
)
def test_invalid_ids_extra_fields_and_salvage_fail_closed(content):
    with pytest.raises(ValueError):
        parse_pattern_id(content)


def test_one_selection_two_arms_share_pattern_only_examples_differ(upstream):
    adapter, client, retrieval, selection, included, excluded = paired(upstream)
    assert len(client.calls) == 5
    assert retrieval == [
        (QUESTION, 3),
        (QUESTION + " " + REWRITE, 6),
        (QUESTION + " " + REWRITE, 6),
    ]
    assert selection["api_calls"] == selection["retrieval_rounds"] == 1
    assert selection["selected_pattern"] == adapter.patterns[2]
    assert selection["selection_sha256"] == selection_digest(selection)
    assert selection["provenance"]["executed_author_methods"] == {}
    assert included["selected_pattern"] == adapter.patterns[2]
    expected = copy.deepcopy(adapter.patterns[2])
    expected["examples"] = []
    assert excluded["selected_pattern"] == expected
    assert included["example_count"] > 0 and excluded["example_count"] == 0
    for arm in (included, excluded):
        assert arm["answer"] == "northbridge"
        assert arm["question"] == QUESTION
        assert arm["pattern_id"] == arm["selected_pattern_id"] == 2
        assert arm["api_calls"] == 2 and arm["retrieval_rounds"] == 1
        assert arm["logical_retrieval_rounds"] == 2
        assert arm["initial_selector_documents"] == selection["initial_selector_documents"]
        assert arm["shared_selection_sha256"] == selection["selection_sha256"]
        assert arm["canonical_action_match"] is True
        assert arm["author_fallback_stages"] == []
        assert arm["provenance"]["baseline_id"] == CONTROL_ID
        assert arm["provenance"]["examples_are_new_memory"] is False
        assert arm["provenance"]["author_selection_executed"] is False
        assert "select_best_pattern" not in arm["provenance"]["executed_author_methods"]
        assert not any(event.get("stage") == "select_id" for event in arm["events"])
        json.dumps(arm)


def test_messages_only_examples_change_after_shared_selection(upstream):
    adapter, client, _, selection, _, _ = paired(upstream)
    prompts = json.loads((upstream / "reformer/prompts.json").read_bytes())
    original = prompts["user_prompts"]["pattern_selection"]["template"].format(
        query=QUESTION,
        documents_text="\n\n".join(doc.text for doc in DOCS[:3]),
        patterns_json=json.dumps(adapter.patterns, indent=2),
    )
    old_contract = "Return only the JSON of the selected pattern."
    assert (
        client.calls[0]["messages"][0]["content"] == prompts["system_prompts"]["pattern_selection"]
    )
    assert client.calls[0]["messages"][1]["content"].startswith(original[: -len(old_contract)])
    assert '"pattern_id": N' in client.calls[0]["messages"][1]["content"]
    for position, examples in ((1, selection["selected_pattern"]["examples"]), (3, [])):
        expected = prompts["user_prompts"]["pattern_application"]["template"].format(
            query=QUESTION,
            transformation_rule=selection["selected_pattern"]["transformation_rule"],
            examples_json=json.dumps(examples, indent=2),
        )
        assert client.calls[position]["messages"] == [
            {"role": "system", "content": prompts["system_prompts"]["pattern_application"]},
            {"role": "user", "content": expected},
        ]
        assert client.calls[position]["prompt_version"] == PROMPT_VERSIONS["rewrite"]
        assert all(doc.text not in expected for doc in DOCS)
    assert client.calls[0]["prompt_version"] == SELECT_PROMPT_VERSION
    assert client.calls[2]["messages"] == client.calls[4]["messages"]
    assert REWRITE not in client.calls[2]["messages"][1]["content"]


def test_trace_ids_events_and_returned_snapshots_remain_separate(upstream):
    adapter, client, _, selection, included, excluded = paired(upstream)
    assert [call["trace_id"] for call in client.calls] == [
        "q1/selection/select_id",
        "q1/with/rewrite",
        "q1/with/reader/s2g-author/01-answer",
        "q1/without/rewrite",
        "q1/without/reader/s2g-author/01-answer",
    ]
    for result, prefix in (
        (selection, "q1/selection"),
        (included, "q1/with"),
        (excluded, "q1/without"),
    ):
        assert all(event["question_id"].startswith(prefix) for event in result["events"])
    previous = json.dumps([selection, included, excluded], sort_keys=True)
    client.outputs = ['{"pattern_id":0}']
    next_selection = adapter.select("Unrelated question?", "q2/selection")
    assert json.dumps([selection, included, excluded], sort_keys=True) == previous
    assert all(event["question_id"] == "q2/selection" for event in next_selection["events"])
    assert QUESTION not in client.calls[-1]["messages"][1]["content"]


@pytest.mark.parametrize("field", ["selected_pattern", "initial_selector_documents", "pattern_id"])
def test_altered_selection_snapshot_stops_before_paid_execution(upstream, field):
    adapter, client, retrieval = make_adapter(upstream, ['{"pattern_id":2}'])
    selection = adapter.select(QUESTION, "q1/selection")
    selection[field] = "tampered"
    with pytest.raises(ValueError, match="snapshot was modified"):
        adapter.run_selected(QUESTION, "q1/arm", selection, include_examples=True)
    assert len(client.calls) == 1 and len(retrieval) == 1


@pytest.mark.parametrize("field", ["transformation_rule", "examples", "pattern_name"])
def test_resealed_untrusted_payload_cannot_change_canonical_action(upstream, field):
    adapter, client, _ = make_adapter(upstream, ['{"pattern_id":2}'])
    selection = adapter.select(QUESTION, "q1/selection")
    selection["selected_pattern"][field] = "injected"
    selection["selection_sha256"] = selection_digest(selection)
    with pytest.raises(ValueError, match="canonical frozen pattern"):
        adapter.run_selected(QUESTION, "q1/arm", selection, include_examples=True)
    assert len(client.calls) == 1


def test_selection_for_another_question_cannot_be_reused(upstream):
    adapter, client, _ = make_adapter(upstream, ['{"pattern_id":2}'])
    selection = adapter.select(QUESTION, "q1/selection")
    with pytest.raises(ValueError, match="same question"):
        adapter.run_selected("A different question?", "q2/arm", selection, include_examples=True)
    assert len(client.calls) == 1


def test_callback_mutation_cannot_affect_patterns_prompts_or_snapshots(upstream):
    def corrupting_callback(event):
        event["kind"] = "corrupt"
        if "selected_pattern" in event:
            event["selected_pattern"]["examples"] = ["corrupt"]
        if "messages" in event:
            event["messages"][0]["content"] = "corrupt"

    adapter, client, _, selection, included, excluded = paired(
        upstream, event_callback=corrupting_callback
    )
    for result in (selection, included, excluded):
        assert all(event["kind"] != "corrupt" for event in result["events"])
    assert selection["selected_pattern"] == included["selected_pattern"] == adapter.patterns[2]
    assert "corrupt" not in json.dumps(client.calls)


def test_no_initial_documents_still_has_explicit_id_selection_no_hidden_default(upstream):
    adapter, client, retrieval = make_adapter(upstream, ['{"pattern_id":9}'], docs=())
    selection = adapter.select(QUESTION, "q1/selection")
    assert selection["pattern_id"] == 9 and not selection["initial_selector_documents"]
    assert len(client.calls) == len(retrieval) == 1


def test_bad_selector_response_keeps_raw_evidence_and_stops(upstream):
    adapter, client, retrieval = make_adapter(upstream, ['{"pattern_id":99}'])
    with pytest.raises(ValueError, match="from 0 through 9"):
        adapter.select(QUESTION, "q1/selection")
    assert len(client.calls) == len(retrieval) == 1
    assert any(event.get("content") == '{"pattern_id":99}' for event in adapter.events)
    assert adapter.events[-1]["kind"] == "selection_validation_failure"
    assert adapter.fallbacks == []


@pytest.mark.parametrize("stage", ["selection", "rewrite", "answer"])
def test_transport_failure_stops_and_is_not_hidden(upstream, stage):
    outputs = [] if stage == "selection" else ['{"pattern_id":2}']
    if stage == "answer":
        outputs.append(REWRITE)
    adapter, client, _ = make_adapter(upstream, outputs + [APIRequestError("secret")])
    with pytest.raises((APIRequestError, RuntimeError)):
        selection = adapter.select(QUESTION, "q1/selection")
        adapter.run_selected(QUESTION, "q1/arm", selection, include_examples=True)
    assert len(client.calls) == {"selection": 1, "rewrite": 2, "answer": 3}[stage]
    assert "secret" not in json.dumps(adapter.events)


def test_final_search_revalidates_shared_document_identity(upstream):
    adapter, client, _ = make_adapter(upstream, ['{"pattern_id":2}', REWRITE])
    selection = adapter.select(QUESTION, "q1/selection")
    adapter.index = lambda *_: [AuthorDocument(DOCS[0].doc_id, "Changed", "Changed text")]
    with pytest.raises(ValueError, match="identity changed"):
        adapter.run_selected(QUESTION, "q1/arm", selection, include_examples=True)
    assert len(client.calls) == 2


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_examples_flag_requires_boolean_before_any_arm_calls(upstream, flag):
    adapter, client, _ = make_adapter(upstream, ['{"pattern_id":2}'])
    selection = adapter.select(QUESTION, "q1/selection")
    with pytest.raises(ValueError, match="must be boolean"):
        adapter.run_selected(QUESTION, "q1/arm", selection, include_examples=flag)
    assert len(client.calls) == 1


def test_gold_free_interfaces_and_canonical_source_cannot_be_changed_by_result(upstream):
    assert list(inspect.signature(ReFormeRControlAPI.select).parameters) == [
        "self",
        "question",
        "trace_id",
    ]
    assert list(inspect.signature(ReFormeRControlAPI.run_selected).parameters) == [
        "self",
        "question",
        "trace_id",
        "selection",
        "include_examples",
    ]
    adapter, client, _, selection, included, _ = paired(upstream)
    original = copy.deepcopy(adapter.patterns[2])
    included["selected_pattern"]["examples"].clear()
    assert selection["selected_pattern"] == original == adapter.patterns[2]
    client.outputs = [REWRITE, ANSWER]
    again = adapter.run_selected(QUESTION, "q1/recheck", selection, include_examples=True)
    assert again["selected_pattern"] == original
