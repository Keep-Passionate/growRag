"""Synthetic A0 prompt/Reader/runtime tests; no credentials, corpus, gold or API."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import a0_runtime as a0
from growrag.experiments import operator_model as legacy
from growrag.experiments.api_client import APIRequestError
from growrag.experiments.history_context import REWRITE_PROMPT_VERSION, SELECT_PROMPT_VERSION
from growrag.experiments.history_runtime import FILL_VERSION as V1_FILL_VERSION
from growrag.experiments.history_runtime import HistoryContractClient
from growrag.experiments.history_runtime_v2 import FILL_PROMPT, FILL_VERSION
from growrag.experiments.operator_model_v2 import answer_episode_v2
from growrag.experiments.operator_model_v3 import ModelOperatorPlannerV3, planner_prompt
from growrag.experiments.operator_schemas import READER_VERSION
from growrag.experiments.operator_schemas_v3 import PLANNER_VERSIONS
from growrag.experiments.output_schemas import registry_manifest, response_format_for
from growrag.experiments.protocol import Evidence, RuntimeQuestion
from growrag.history_library import FrozenHistoryLibrary, HistoryCard, HistoryRecord
from growrag.macro_operators import (
    GapField,
    OperatorRegistry,
    OperatorSpec,
    QueryStep,
    RuntimeState,
)
from growrag.operator_bank import operator_to_dict
from growrag.operator_loop import Observation

QUESTION = RuntimeQuestion("synthetic-a0", "Where was the founder of Cedar Lab born?")
EVIDENCE = (Evidence("current-1", "Cedar Lab", 0, "Avery Finch founded Cedar Lab."),)
STOP = {"reason": "no_useful_query", "intent": "lookup", "constraints": [], "actions": []}
ANSWER = {"answer": "Stonebridge", "supported": True, "evidence_ids": ["E1"]}


class SyntheticDurable:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.config = SimpleNamespace(json_object_mode=True, json_schema_mode=False)
        self.requests, self.calls = [], []
        self.block_reason = None

    def complete(self, messages, *, trace_id, prompt_version):
        self.requests.append(
            {"messages": deepcopy(messages), "trace_id": trace_id, "prompt_version": prompt_version}
        )
        if not self.outputs:
            pytest.fail("unexpected retry or fallback")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            self.block_reason = "transport_failed"
            raise output
        self.calls.append(
            {
                "trace_id": trace_id,
                "prompt_version": prompt_version,
                "status": "completed",
                "api_requests": 1,
                "input_tokens": 17,
                "output_tokens": 9,
                "reserved_cny": 0.01,
                "estimated_actual_cny": 0.0000106,
            }
        )
        self.response = SimpleNamespace(
            content=output if isinstance(output, str) else json.dumps(output),
            request_id="synthetic-unpaid",
            input_tokens=17,
            output_tokens=9,
        )
        return self.response


def state():
    return Observation(QUESTION, EVIDENCE, (), 2, 1)


def contract(outputs, events=None):
    delegate = SyntheticDurable(outputs)
    return a0.A0ContractClient(
        delegate, on_record=None if events is None else events.append
    ), delegate


def test_focused_wrapper_retains_v3_grammar_and_only_changes_wire_prompt_version():
    events = []
    focused, delegate = contract([STOP])
    result = a0.FocusedFreshPlanner(focused, trace_prefix="new", on_record=events.append)(state())
    original, old = contract([STOP])
    reference = ModelOperatorPlannerV3(original, mode="fresh", trace_prefix="old")(state())
    assert result == reference
    assert delegate.requests[0]["prompt_version"] == a0.FOCUSED_PLANNER_VERSION
    assert old.requests[0]["prompt_version"] == PLANNER_VERSIONS["fresh"]
    assert old.requests[0]["messages"][0]["content"] == planner_prompt("fresh")
    assert a0.focused_planner_prompt().startswith(planner_prompt("fresh"))
    assert "not a mandate to make every query short" in a0.focused_planner_prompt()
    assert delegate.requests[0]["messages"][1] == old.requests[0]["messages"][1]
    assert events[0]["prompt_version"] == a0.FOCUSED_PLANNER_VERSION


def test_focused_example_executes_standalone_grounded_query_with_original_goal():
    client, _ = contract([deepcopy(a0._FOCUSED_EXAMPLE)])
    proposal = a0.FocusedFreshPlanner(client, trace_prefix="synthetic")(state())
    registry = OperatorRegistry()
    registry.register(proposal.spec)
    plan = registry.plan(
        proposal.spec.operator_id,
        proposal.spec.version,
        goal=proposal.goal,
        gap=proposal.gap,
        state=RuntimeState(EVIDENCE, proposal.bindings, 2),
    )
    assert proposal.goal.original_question == QUESTION.text
    assert proposal.goal.constraints == ("Find the birthplace of the founder of Cedar Lab",)
    assert [request.query for request in plan.requests] == ["Avery Finch birthplace"]
    assert proposal.bindings[0].evidence_ids == ("current-1",)


@pytest.mark.parametrize("patch", ["unseen_binding", "unknown_id", "duplicate_gap", "multi_action"])
def test_focused_rejects_invalid_actions_without_repair_or_retry(patch):
    output = deepcopy(a0._FOCUSED_EXAMPLE)
    action = output["actions"][0]
    if patch == "unseen_binding":
        action["bindings"][0]["value"] = "Imaginary Founder"
    elif patch == "unknown_id":
        action["bindings"][0]["evidence_ids"] = ["missing"]
    elif patch == "duplicate_gap":
        action["gap_entries"] *= 2
    else:
        output["actions"] *= 2
    events = []
    delegate = SyntheticDurable([output])
    client = a0.A0ContractClient(delegate, on_record=events.append)
    with pytest.raises(a0.A0LocalOutputError):
        a0.FocusedFreshPlanner(client, trace_prefix="synthetic")(state())
    assert len(delegate.requests) == len(delegate.calls) == 1
    assert json.loads(events[0]["raw_content"]) == output
    assert delegate.block_reason is None


def test_focused_binding_cannot_cite_text_outside_unchanged_visible_prefix():
    evidence = (Evidence("current-1", "Source", 0, "x" * 20000 + " Avery Finch"),)
    client, delegate = contract([deepcopy(a0._FOCUSED_EXAMPLE)])
    with pytest.raises(a0.A0LocalOutputError, match="absent from the visible cited window"):
        a0.FocusedFreshPlanner(client, trace_prefix="hidden-tail")(
            replace(state(), evidence=evidence)
        )
    assert len(delegate.calls) == 1 and delegate.block_reason is None


def test_new_schemas_are_independent_and_unknown_version_fails_before_transport():
    before = registry_manifest()
    client, delegate = contract([STOP])
    assert isinstance(client, HistoryContractClient)
    assert client.calls is delegate.calls
    with pytest.raises(ValueError):
        client.complete([], trace_id="none", prompt_version="unregistered")
    assert delegate.requests == []
    for version in (a0.FOCUSED_PLANNER_VERSION, a0.SHORT_READER_VERSION):
        assert a0.schema_for(version)["additionalProperties"] is False
        with pytest.raises(ValueError):
            response_format_for(version)
    assert registry_manifest() == before


@pytest.mark.parametrize(
    "version,output",
    [
        (SELECT_PROMPT_VERSION, {"selected_card_id": None, "reason": "no_suitable_card"}),
        (REWRITE_PROMPT_VERSION, {"query": "current entity location"}),
        (PLANNER_VERSIONS["fresh"], STOP),
        (READER_VERSION, {"answer": "", "supported": False, "evidence_ids": []}),
        (
            V1_FILL_VERSION,
            {"intent": "lookup", "constraints": [], "gap_entries": [], "bindings": []},
        ),
    ],
)
def test_legacy_dispatch_and_response_metadata_stay_compatible(version, output):
    client, delegate = contract([output])
    messages = legacy._messages("original unchanged JSON", {"test": "same"})
    response = client.complete(messages, trace_id="legacy", prompt_version=version)
    assert response is delegate.response
    assert delegate.requests[0]["messages"] == messages
    assert response.request_id == "synthetic-unpaid"


def test_reader_aliases_preserve_exact_old_prefix_content_windows_and_raw_response():
    evidence = tuple(
        Evidence(f"long-corpus-document-{number}-" + "z" * 160, "标题" * 20, number, "内容" * 2500)
        for number in range(26)
    )
    old = SyntheticDurable([{"answer": "", "supported": False, "evidence_ids": []}])
    answer_episode_v2(old, QUESTION, evidence, trace_id="old-reader")
    old_payload = json.loads(old.requests[0]["messages"][1]["content"])
    raw = '{ "answer": "Stonebridge", "supported": true, "evidence_ids": ["E2", "E1"] }'
    new = SyntheticDurable([raw])
    events = []
    result = a0.answer_episode_shortrefs(
        new, QUESTION, evidence, trace_id="new-reader", on_record=events.append
    )
    payload = json.loads(new.requests[0]["messages"][1]["content"])
    assert len(payload["evidence"]) == len(old_payload["evidence"]) == 24
    assert payload["evidence_window_omitted_count"] == old_payload["evidence_window_omitted_count"]
    for number, (row, previous) in enumerate(
        zip(payload["evidence"], old_payload["evidence"], strict=True), 1
    ):
        assert row == {**previous, "evidence_id": f"E{number}"}
        assert previous["window"]["text_truncated"]
    assert result["evidence_ids"] == [evidence[1].evidence_id, evidence[0].evidence_id]
    assert result["evidence_refs"] == ["E2", "E1"]
    assert result["raw_content"] == events[1]["raw_content"] == raw
    assert events[0]["alias_to_evidence_id"] == result["alias_to_evidence_id"]
    assert events[2]["output"] == result
    wire_text = new.requests[0]["messages"][1]["content"]
    assert all(item.evidence_id not in wire_text for item in evidence)


@pytest.mark.parametrize(
    "refs",
    [
        ["E1", "E1"],
        ["E2"],
        ["E01"],
        ["E0"],
        ["e1"],
        ["E 1"],
        [" E1"],
        ["E1 "],
        ["[E1]"],
        ["current-1"],
        ["E١"],
        [1],
    ],
)
def test_reader_rejects_duplicate_unknown_and_noncanonical_labels(refs):
    events = []
    raw = json.dumps({**ANSWER, "evidence_ids": refs})
    delegate = SyntheticDurable([raw])
    client = a0.A0ContractClient(delegate, on_record=events.append)
    with pytest.raises(a0.A0LocalOutputError):
        a0.answer_episode_shortrefs(client, QUESTION, EVIDENCE, trace_id="bad-ref")
    assert len(delegate.calls) == len(delegate.requests) == 1
    assert events[0]["raw_content"] == raw
    assert events[-1]["kind"] == "local_contract_failure"
    assert events[-1]["scope"] == "arm"
    assert delegate.block_reason is None


@pytest.mark.parametrize(
    "output",
    [
        {**ANSWER, "supported": "true"},
        {**ANSWER, "supported": False},
        {**ANSWER, "answer": " "},
        {**ANSWER, "evidence_ids": []},
        {**ANSWER, "extra": "not allowed"},
        {"answer": "Stonebridge", "supported": True},
        '{"answer":"x","answer":"y","supported":true,"evidence_ids":["E1"]}',
        '{"answer":NaN,"supported":true,"evidence_ids":["E1"]}',
        "```json\n{}\n```",
    ],
)
def test_reader_strict_keys_types_and_support_consistency(output):
    client = SyntheticDurable([output])
    events = []
    with pytest.raises(a0.A0LocalOutputError):
        a0.answer_episode_shortrefs(
            client, QUESTION, EVIDENCE, trace_id="invalid", on_record=events.append
        )
    assert len(client.requests) == 1
    assert [event["stage"] for event in events] == ["reader_request", "raw_wire"]


def test_empty_evidence_can_abstain_without_fabricated_refs():
    client = SyntheticDurable([{"answer": "", "supported": False, "evidence_ids": []}])
    result = a0.answer_episode_shortrefs(client, QUESTION, (), trace_id="empty")
    assert result["alias_to_evidence_id"] == {}
    assert result["evidence_ids"] == result["visible_evidence_ids"] == []


def test_reader_contract_rejects_noncanonical_request_before_transport():
    client, delegate = contract([])
    payload = {
        "original_question": QUESTION.text,
        "evidence": [{"evidence_id": "E01", "text": "Evidence"}],
        "evidence_window_omitted_count": 0,
    }
    with pytest.raises(ValueError, match="canonical and consecutive"):
        client.complete(
            legacy._messages(a0.SHORT_READER_PROMPT, payload),
            trace_id="noncanonical-request",
            prompt_version=a0.SHORT_READER_VERSION,
        )
    assert delegate.requests == []


def test_reader_callback_cannot_mutate_mapping_or_wire_payload():
    client = SyntheticDurable([ANSWER])

    def malicious_callback(event):
        event.get("alias_to_evidence_id", {}).clear()
        event.get("payload", {}).clear()
        event.get("output", {}).clear()

    result = a0.answer_episode_shortrefs(
        client, QUESTION, EVIDENCE, trace_id="immutable", on_record=malicious_callback
    )
    assert result["evidence_ids"] == ["current-1"]
    assert result["alias_to_evidence_id"] == {"E1": "current-1"}


def test_no_credential_or_transport_before_runtime_and_evidence_validation():
    delegate = SyntheticDurable([])
    for question, evidence in [(object(), ()), (QUESTION, EVIDENCE * 2), (QUESTION, [])]:
        with pytest.raises((TypeError, ValueError)):
            a0.answer_episode_shortrefs(delegate, question, evidence, trace_id="invalid")
    with pytest.raises(ValueError, match="prompt exceeds") as caught:
        a0.answer_episode_shortrefs(
            delegate, RuntimeQuestion("large", "x" * 40000), EVIDENCE, trace_id="large"
        )
    assert not isinstance(caught.value, a0.A0LocalOutputError)
    assert delegate.requests == []


def library():
    spec = OperatorSpec(
        "BIRTHPLACE",
        "1",
        ("lookup",),
        (GapField("entity"),),
        (QueryStep("first", "{entity} birthplace"), QueryStep("second", "{entity} biography")),
    )
    cards = [
        HistoryCard(
            f"T{number:02d}",
            "1",
            "Birthplace lookup",
            "Find the current birthplace.",
            "Search the current entity.",
            operator_spec=replace(spec, operator_id=f"S{number}"),
        )
        for number in range(9)
    ]
    return FrozenHistoryLibrary(
        "synthetic-a0",
        (),
        tuple(
            HistoryRecord(card, "reference", "synthetic/manual", "0" * 64, status="published")
            for card in cards
        ),
    )


def fresh_action():
    return {
        "reason": "missing_evidence",
        "intent": "lookup",
        "constraints": [],
        "actions": [
            {
                "selected_operator": None,
                "operator": operator_to_dict(library().published_cards[0].operator_spec),
                "gap_entries": [{"name": "entity", "value": "Cedar Lab"}],
                "bindings": [],
            }
        ],
    }


@pytest.mark.parametrize("method", a0.METHODS)
def test_all_arms_share_short_reader_top6_and_unchanged_bounded_executor(tmp_path, method):
    outputs = []
    if method in {"fresh_original", "fresh_focused"}:
        outputs.append(fresh_action())
    elif method == "history_body8":
        outputs.extend(
            [
                {"selected_card_id": "T00", "reason": "condition_match"},
                {
                    "intent": "lookup",
                    "constraints": [],
                    "gap_entries": [{"name": "entity", "value": "Cedar Lab"}],
                    "bindings": [],
                },
            ]
        )
    outputs.append(ANSWER)
    events, searches = [], []
    delegate = SyntheticDurable(outputs)
    client = a0.A0ContractClient(delegate, on_record=events.append)

    def index(query, top_k):
        searches.append((query, top_k))
        return [SimpleNamespace(doc_id=f"doc/{len(searches)}", title="Source", text="Evidence.")]

    target = tmp_path / f"{method}.json"
    report = a0.execute_arm(
        QUESTION,
        method,
        index,
        client,
        library=library(),
        trace=method,
        log=events.append,
        target=target,
    )
    assert json.loads(json.dumps(report)) == json.loads(target.read_text(encoding="utf-8"))
    assert report["status"] == "completed" and report["method"] == method
    assert report["reader"]["prompt_version"] == a0.SHORT_READER_VERSION
    assert report["reader"]["evidence_ids"] == ["doc/1"]
    assert len(searches) == (1 if method == "base" else 3)
    assert all(top_k == 6 for _, top_k in searches)
    assert len(report["episode"]["proposals"]) <= 2
    assert report["gold_loaded"] is False and report["memory_updated"] is False
    if method == "history_body8":
        ranking = next(event for event in events if event["kind"] == "history_candidates")
        assert len(ranking["ranking"]) == 8
        fill = next(
            request for request in delegate.requests if request["prompt_version"] == FILL_VERSION
        )
        assert fill["messages"][0]["content"] == FILL_PROMPT
        payload = json.loads(fill["messages"][1]["content"])
        assert payload["allowed_binding_names"] == []
        assert payload["allowed_evidence_ids"] == ["doc/1"]
    count = len(delegate.requests)
    with pytest.raises(FileExistsError):
        a0.execute_arm(
            QUESTION,
            method,
            index,
            client,
            library=library(),
            trace=method,
            log=events.append,
            target=target,
        )
    assert len(delegate.requests) == count


def test_failed_reader_is_persisted_and_next_independent_arm_is_possible(tmp_path):
    events = []
    delegate = SyntheticDurable([{**ANSWER, "evidence_ids": ["E01"]}, ANSWER])
    client = a0.A0ContractClient(delegate, on_record=events.append)

    def index(query, top_k):
        return [SimpleNamespace(doc_id="doc/1", title="Source", text="Evidence.")]

    target = tmp_path / "failed.json"
    with pytest.raises(ValueError):
        a0.execute_arm(
            QUESTION,
            "base",
            index,
            client,
            library=None,
            trace="failed",
            log=events.append,
            target=target,
        )
    failed = json.loads(target.read_text(encoding="utf-8"))
    assert failed["status"] == "failed" and failed["error_stage"] == "reader"
    assert failed["episode"] is not None and failed["reader"] is None
    assert failed["calls"][0]["status"] == "completed"
    assert delegate.block_reason is None
    success = a0.execute_arm(
        QUESTION,
        "base",
        index,
        client,
        library=None,
        trace="independent",
        log=events.append,
        target=tmp_path / "independent.json",
    )
    assert success["status"] == "completed" and len(success["calls"]) == 1
    assert len(delegate.requests) == 2


def test_focused_two_decisions_use_initial_plus_two_retrievals_only(tmp_path):
    first, second = fresh_action(), fresh_action()
    first["actions"][0]["operator"]["steps"] = first["actions"][0]["operator"]["steps"][:1]
    second["actions"][0]["operator"]["steps"] = second["actions"][0]["operator"]["steps"][1:]
    client, delegate = contract([first, second, ANSWER])
    events, searches = [], []

    def index(query, top_k):
        searches.append(query)
        return [SimpleNamespace(doc_id=f"doc/{len(searches)}", title="Source", text="Evidence.")]

    report = a0.execute_arm(
        QUESTION, "fresh_focused", index, client, library=None, trace="two-decisions",
        log=events.append, target=tmp_path / "two-decisions.json",
    )
    assert len(report["episode"]["searches"]) == 3
    assert len(report["episode"]["proposals"]) == 2
    assert report["episode"]["stop_reason"] == "retrieval_budget"
    planner_payloads = [
        json.loads(request["messages"][1]["content"])
        for request in delegate.requests
        if request["prompt_version"] == a0.FOCUSED_PLANNER_VERSION
    ]
    assert [row["remaining_retrievals"] for row in planner_payloads] == [2, 1]


def test_executor_rejection_marks_failed_arm_and_never_falls_back_to_reader(tmp_path):
    output = fresh_action()
    output["actions"][0]["gap_entries"] = []
    client, delegate = contract([output])
    events = []

    def index(query, top_k):
        return [SimpleNamespace(doc_id="doc/1", title="Source", text="Evidence.")]

    target = tmp_path / "rejected-plan.json"
    with pytest.raises(a0.A0LocalOutputError, match="rejected by unchanged executor"):
        a0.execute_arm(
            QUESTION, "fresh_focused", index, client, library=None, trace="rejected-plan",
            log=events.append, target=target,
        )
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["error_stage"] == "episode"
    assert report["episode"]["stop_reason"] == "plan_rejected"
    assert report["reader"] is None and len(delegate.requests) == 1
    assert delegate.block_reason is None


def test_fill_scope_failure_is_arm_local_and_has_raw_accounted_record(tmp_path):
    outputs = [
        {"selected_card_id": "T00", "reason": "condition_match"},
        {
            "intent": "lookup",
            "constraints": [],
            "gap_entries": [{"name": "entity", "value": "Cedar Lab"}],
            "bindings": [{"name": "invented", "value": "Evidence", "evidence_ids": ["doc/1"]}],
        },
    ]
    events = []
    delegate = SyntheticDurable(outputs)
    client = a0.A0ContractClient(delegate, on_record=events.append)

    def index(query, top_k):
        return [SimpleNamespace(doc_id="doc/1", title="Source", text="Evidence.")]

    target = tmp_path / "fill-failed.json"
    with pytest.raises(a0.A0LocalOutputError, match="not declared"):
        a0.execute_arm(
            QUESTION,
            "history_body8",
            index,
            client,
            library=library(),
            trace="bad-fill",
            log=events.append,
            target=target,
        )
    assert delegate.block_reason is None and len(delegate.calls) == 2
    assert events[-1]["kind"] == "local_contract_failure"
    assert events[-1]["prompt_version"] == FILL_VERSION
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.parametrize("failure", ["retriever_exception", "changed_evidence"])
def test_retrieval_integrity_errors_after_completed_planner_are_not_local_outputs(
    tmp_path, failure
):
    client, delegate = contract([fresh_action()])
    searches = []

    def index(query, top_k):
        searches.append(query)
        if len(searches) > 1 and failure == "retriever_exception":
            raise ValueError("synthetic index integrity failure")
        return [SimpleNamespace(doc_id="same-id", title="Source", text=f"Text {len(searches)}")]

    with pytest.raises(ValueError) as caught:
        a0.execute_arm(
            QUESTION, "fresh_original", index, client, library=None, trace="retrieval-failed",
            log=lambda event: None, target=tmp_path / "retrieval-failed.json",
        )
    assert not isinstance(caught.value, a0.A0LocalOutputError)
    assert len(delegate.calls) == 1 and delegate.block_reason is None


def test_history_fill_semantic_failure_gets_local_output_marker(tmp_path):
    client, delegate = contract([
        {"selected_card_id": "T00", "reason": "condition_match"},
        {"intent": "lookup", "constraints": [], "gap_entries": [], "bindings": []},
    ])

    def index(query, top_k):
        return [SimpleNamespace(doc_id="doc/1", title="Source", text="Evidence.")]

    with pytest.raises(a0.A0LocalOutputError):
        a0.execute_arm(
            QUESTION, "history_body8", index, client, library=library(), trace="invalid-gap",
            log=lambda event: None, target=tmp_path / "invalid-gap.json",
        )
    assert len(delegate.calls) == 2 and delegate.block_reason is None


def test_planner_log_failure_is_not_classified_as_a_model_output_failure():
    client, delegate = contract([fresh_action()])

    def record(event):
        if event.get("stage") == "normalized_parser_input":
            raise ValueError("synthetic audit callback failure")

    with pytest.raises(ValueError) as caught:
        a0.A0FreshPlanner(client, trace_prefix="audit-failed", on_record=record)(state())
    assert not isinstance(caught.value, a0.A0LocalOutputError)
    assert len(delegate.calls) == 1


def test_transport_failure_preserves_block_and_does_not_fabricate_response(tmp_path):
    delegate = SyntheticDurable([APIRequestError("synthetic transport failure")])
    client = a0.A0ContractClient(delegate)
    events = []

    def index(query, top_k):
        return [SimpleNamespace(doc_id="doc/1", title="Source", text="Evidence.")]

    target = tmp_path / "transport-failed.json"
    with pytest.raises(APIRequestError):
        a0.execute_arm(
            QUESTION,
            "base",
            index,
            client,
            library=None,
            trace="transport",
            log=events.append,
            target=target,
        )
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["reader"] is None
    assert delegate.block_reason == "transport_failed"
    assert len(delegate.requests) == 1 and report["calls"] == []


def test_historical_model_modules_remain_byte_frozen():
    root = Path(__file__).resolve().parents[1]
    frozen = {
        "src/growrag/experiments/operator_model_v2.py": (
            "03b71d0be5c1905161124e00f27bd85f547fa229c386ec16f0d62715c4e1a981"
        ),
        "src/growrag/experiments/operator_schemas.py": (
            "8ae38ca9b462edeae32c6e1fb6d108294f3910abd6faaccb2b483171840283a5"
        ),
    }
    for name, digest in frozen.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
