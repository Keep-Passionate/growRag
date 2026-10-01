"""FILL compatibility contracts with synthetic data; no API, gold, or replay."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from growrag.experiments import history_runtime as v1
from growrag.experiments import history_runtime_v2 as v2
from growrag.experiments.api_client import APIRequestError, ChatConfig, ChatResponse
from growrag.experiments.budget import PriceLimits
from growrag.experiments.history_context import (
    REWRITE_PROMPT_VERSION,
    SELECT_PROMPT_VERSION,
    _bounded_messages,
)
from growrag.experiments.operator_schemas import READER_VERSION
from growrag.experiments.operator_schemas_v3 import PLANNER_VERSIONS
from growrag.experiments.output_schemas import registry_manifest, response_format_for
from growrag.experiments.protocol import Evidence, RuntimeQuestion
from growrag.experiments.run_pre_opportunity import DurableBudgetClient
from growrag.history_library import (
    FrozenHistoryLibrary,
    HistoryCard,
    HistoryRecord,
    card_view,
)
from growrag.macro_operators import GapField, OperatorSpec, QueryStep
from growrag.operator_loop import Observation

QUESTION = RuntimeQuestion("synthetic-v2", "Where was the founder of Cedar Lab born?")
EVIDENCE = (Evidence("current-1", "Cedar Lab", 0, "Avery Finch founded Cedar Lab."),)


class SyntheticDurable:
    def __init__(self, responses):
        self.responses = list(responses)
        self.config = SimpleNamespace(json_object_mode=True, json_schema_mode=False)
        self.calls, self.requests, self.journal = [], [], []
        self.block_reason = None
        self.reserved_cny = self.estimated_actual_cny = 0.0

    def complete(self, messages, *, trace_id, prompt_version):
        if self.block_reason:
            raise APIRequestError("synthetic budget blocked")
        self.requests.append(
            {"messages": deepcopy(messages), "trace_id": trace_id, "prompt_version": prompt_version}
        )
        self.journal.append({"trace_id": trace_id, "prompt_version": prompt_version})
        if not self.responses:
            pytest.fail("unexpected retry or fallback")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
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
        self.reserved_cny += 0.01
        self.estimated_actual_cny += 0.0000106
        self.response = SimpleNamespace(
            content=value if isinstance(value, str) else json.dumps(value),
            input_tokens=17,
            output_tokens=9,
        )
        return self.response


@pytest.fixture
def card():
    return HistoryCard(
        "LOOKUP",
        "1",
        "Lookup",
        "Current entity lookup.",
        "Fill the current entity.",
        operator_spec=OperatorSpec(
            "LOOKUP_SPEC",
            "1",
            ("lookup",),
            (GapField("entity"),),
            (QueryStep("search", "{entity} location"),),
        ),
    )


@pytest.fixture
def bridge(card):
    return replace(
        card,
        card_id="BRIDGE",
        operator_spec=OperatorSpec(
            "BRIDGE_SPEC",
            "1",
            ("lookup",),
            (),
            (QueryStep("next", "{founder} birthplace", requires_bindings=("founder",)),),
        ),
    )


def _library(card):
    return FrozenHistoryLibrary(
        "synthetic-history-v2",
        (),
        (HistoryRecord(card, "reference", "synthetic/manual", "0" * 64, status="published"),),
    )


def _fill(*, gap=None, bindings=None):
    return {
        "intent": "lookup",
        "constraints": [],
        "gap_entries": [{"name": "entity", "value": "Cedar Lab"}] if gap is None else gap,
        "bindings": [] if bindings is None else bindings,
    }


def _binding(name="founder", ids=None):
    return {
        "name": name,
        "value": "Avery Finch",
        "evidence_ids": ["current-1"] if ids is None else ids,
    }


def _messages(card):
    return _bounded_messages(
        v1.FILL_PROMPT,
        {
            "original_question": QUESTION.text,
            "evidence": [
                {
                    "evidence_id": "current-1",
                    "title": "Cedar Lab",
                    "sentence_id": 0,
                    "text": "Avery Finch founded Cedar Lab.",
                }
            ],
            "previous_queries": [QUESTION.text],
            "remaining_retrievals": 1,
            "selected_card": card_view(card, "examples"),
        },
    )


def _wire(responses, events=None):
    delegate = SyntheticDurable(responses)
    contract = v1.HistoryContractClient(
        delegate, on_record=None if events is None else events.append
    )
    return v2.FillWireV2Client(contract), delegate


def _selection(card):
    return {"selected_card_id": card.card_id, "reason": "uncertain_match"}


def _state():
    return Observation(QUESTION, EVIDENCE, (), 1, 1)


def test_fill_changes_only_declared_system_payload_additions_and_exact_wire_version(card):
    messages = _messages(card)
    original = deepcopy(messages)
    events = []
    wire, delegate = _wire([_fill()], events)
    response = wire.complete(messages, trace_id="synthetic/fill", prompt_version=v1.FILL_VERSION)
    sent = delegate.requests[0]
    assert messages == original
    assert sent["prompt_version"] == "growrag-history-base-fill-v2" == v2.FILL_VERSION
    assert sent["trace_id"] == "synthetic/fill"
    assert sent["messages"][0] == {"role": "system", "content": v2.FILL_PROMPT}
    payload = json.loads(sent["messages"][1]["content"])
    assert payload.pop("allowed_binding_names") == []
    assert payload.pop("allowed_evidence_ids") == ["current-1"]
    assert payload == json.loads(original[1]["content"])
    assert response is delegate.response
    assert wire.calls is delegate.calls
    assert delegate.calls[0]["prompt_version"] == v2.FILL_VERSION
    assert delegate.journal[0]["prompt_version"] == v2.FILL_VERSION
    assert events[-1]["prompt_version"] == v2.FILL_VERSION
    assert "If allowed_binding_names is [], return bindings: [] exactly." in v2.FILL_PROMPT
    assert "original_question is a field, NOT an" in v2.FILL_PROMPT


def test_allowed_names_are_derived_from_original_selected_spec(bridge):
    wire, delegate = _wire([_fill(gap=[], bindings=[_binding()])])
    wire.complete(_messages(bridge), trace_id="synthetic/bridge", prompt_version=v1.FILL_VERSION)
    payload = json.loads(delegate.requests[0]["messages"][1]["content"])
    assert payload["allowed_binding_names"] == ["founder"]
    assert payload["selected_card"] == card_view(bridge, "examples")


@pytest.mark.parametrize(
    "version,response",
    [
        (SELECT_PROMPT_VERSION, {"selected_card_id": None, "reason": "no_suitable_card"}),
        (REWRITE_PROMPT_VERSION, {"query": "Current query"}),
        (READER_VERSION, {"answer": "", "supported": False, "evidence_ids": []}),
        (
            PLANNER_VERSIONS["fresh"],
            {"reason": "no_useful_query", "intent": "lookup", "constraints": [], "actions": []},
        ),
    ],
)
def test_nonfill_requests_and_response_identity_are_unchanged(version, response):
    messages = [
        {"role": "system", "content": "JSON 合同"},
        {"role": "user", "content": '{ "exact_bytes" : "kept" }'},
    ]
    original = deepcopy(messages)
    wire, delegate = _wire([response])
    result = wire.complete(messages, trace_id="synthetic/nonfill", prompt_version=version)
    assert delegate.requests == [
        {"messages": original, "trace_id": "synthetic/nonfill", "prompt_version": version}
    ]
    assert messages == original
    assert result is delegate.response


def test_instance_adapter_does_not_mutate_original_or_global_schema_registry(card):
    before = registry_manifest()
    legacy, new = SyntheticDurable([_fill()]), SyntheticDurable([_fill()])
    original = v1.HistoryContractClient(legacy)
    wire = v2.FillWireV2Client(v1.HistoryContractClient(new))
    original.complete(_messages(card), trace_id="old", prompt_version=v1.FILL_VERSION)
    wire.complete(_messages(card), trace_id="new", prompt_version=v1.FILL_VERSION)
    assert legacy.requests[0]["prompt_version"] == v1.FILL_VERSION
    assert new.requests[0]["prompt_version"] == v2.FILL_VERSION
    assert registry_manifest() == before
    assert v2.schema_for_v2(v2.FILL_VERSION) == v1.schema_for(v1.FILL_VERSION)
    with pytest.raises(ValueError):
        response_format_for(v2.FILL_VERSION)


@pytest.mark.parametrize("version", [v1.FILL_VERSION, "growrag-history-base-fill-v20", "unknown"])
def test_v2_scoring_schema_rejects_old_fill_and_unknown_versions(version):
    with pytest.raises(ValueError):
        v2.schema_for_v2(version)


@pytest.mark.parametrize(
    "version",
    [SELECT_PROMPT_VERSION, REWRITE_PROMPT_VERSION, READER_VERSION, PLANNER_VERSIONS["fresh"]],
)
def test_v2_scoring_schema_keeps_nonfill_shapes_exact(version):
    assert v2.schema_for_v2(version) == v1.schema_for(version)


@pytest.mark.parametrize(
    "mutation",
    ["system", "payload_extra", "card_rule", "duplicate_evidence", "reserved_evidence"],
)
def test_unexpected_original_fill_request_fails_before_budget_or_network(card, mutation):
    messages = _messages(card)
    if mutation == "system":
        messages[0]["content"] += " silent new instruction"
    else:
        payload = json.loads(messages[1]["content"])
        if mutation == "payload_extra":
            payload["allowed_binding_names"] = ["invented"]
        elif mutation == "card_rule":
            payload["selected_card"]["action_kind"] = "rewrite_rule"
        elif mutation == "duplicate_evidence":
            payload["evidence"].append(deepcopy(payload["evidence"][0]))
        else:
            payload["evidence"][0]["evidence_id"] = "original_question"
        messages[1]["content"] = json.dumps(payload)
    wire, delegate = _wire([])
    with pytest.raises((ValueError, TypeError, KeyError)):
        wire.complete(messages, trace_id="bad-input", prompt_version=v1.FILL_VERSION)
    assert delegate.requests == [] and delegate.calls == [] and delegate.journal == []
    assert delegate.reserved_cny == 0


@pytest.mark.parametrize(
    "response",
    [
        _fill(bindings=[_binding("undeclared")]),
        {**_fill(), "query": "silent rewrite"},
        {**_fill(), "allowed_binding_names": []},
        "not JSON",
        '{"intent":"lookup","intent":"x"}',
    ],
)
def test_invalid_output_is_rejected_not_cleaned_and_http_costs_survive(card, response):
    events = []
    wire, delegate = _wire([response], events)
    with pytest.raises((ValueError, TypeError, KeyError)):
        wire.complete(_messages(card), trace_id="synthetic/failure", prompt_version=v1.FILL_VERSION)
    assert delegate.block_reason == "local_output_contract_failure"
    assert delegate.calls[0]["status"] == "completed"
    assert delegate.calls[0]["api_requests"] == 1
    assert delegate.calls[0]["input_tokens"] == 17 and delegate.calls[0]["output_tokens"] == 9
    assert delegate.calls[0]["prompt_version"] == v2.FILL_VERSION
    assert delegate.reserved_cny == pytest.approx(0.01)
    assert delegate.estimated_actual_cny == pytest.approx(0.0000106)
    assert events[-1]["kind"] == "local_contract_failure"
    assert events[-1]["prompt_version"] == v2.FILL_VERSION
    with pytest.raises(APIRequestError, match="blocked"):
        wire.complete(_messages(card), trace_id="synthetic/after", prompt_version=v1.FILL_VERSION)
    assert len(delegate.requests) == 1


@pytest.mark.parametrize("ids", [["original_question"], ["unseen"], ["current-1", "unseen"]])
def test_binding_citations_must_be_actual_current_visible_ids(bridge, ids):
    wire, delegate = _wire([_fill(gap=[], bindings=[_binding(ids=ids)])])
    with pytest.raises(ValueError, match="visible evidence"):
        wire.complete(_messages(bridge), trace_id="bad-id", prompt_version=v1.FILL_VERSION)
    assert delegate.block_reason == "local_output_contract_failure"


def test_gap_entries_cannot_be_duplicated_as_bindings(bridge):
    wire, delegate = _wire(
        [_fill(gap=[{"name": "founder", "value": "Avery Finch"}], bindings=[_binding()])]
    )
    with pytest.raises(ValueError, match="also be returned"):
        wire.complete(_messages(bridge), trace_id="bad-overlap", prompt_version=v1.FILL_VERSION)
    assert delegate.block_reason == "local_output_contract_failure"


@pytest.mark.parametrize("kind", ["template", "bridge", "rule", "stop"])
def test_original_planner_semantics_and_selected_parameters_are_unchanged(card, bridge, kind):
    selected = bridge if kind == "bridge" else card
    if kind == "rule":
        selected = replace(card, operator_spec=None)
    if kind == "stop":
        responses = [{"selected_card_id": None, "reason": "no_suitable_card"}]
    else:
        second = (
            {"query": "Cedar Lab director birthplace"}
            if kind == "rule"
            else (_fill(gap=[], bindings=[_binding()]) if kind == "bridge" else _fill())
        )
        responses = [_selection(selected), second]
    library = _library(selected)
    snapshot = library.to_json()
    old, new = SyntheticDurable(responses), SyntheticDurable(responses)
    original = v1.HistoryPlanner(v1.HistoryContractClient(old), library, trace_prefix="synthetic")
    adapted = v2.HistoryPlannerV2(v1.HistoryContractClient(new), library, trace_prefix="synthetic")
    assert original(_state()) == adapted(_state())
    assert old.requests[0] == new.requests[0]
    assert library.to_json() == snapshot
    if kind in {"rule", "stop"}:
        assert old.requests == new.requests
    else:
        assert old.requests[1]["prompt_version"] == v1.FILL_VERSION
        assert new.requests[1]["prompt_version"] == v2.FILL_VERSION


@pytest.mark.parametrize(
    "bad_fill",
    [
        _fill(gap=[{"name": "entity", "value": True}]),
        _fill(gap=[{"name": "undeclared", "value": "Cedar Lab"}]),
        _fill(gap=[{"name": "entity", "value": "A"}, {"name": "entity", "value": "B"}]),
    ],
)
def test_existing_gap_type_scope_and_duplicate_validation_remains(card, bad_fill):
    delegate = SyntheticDurable([_selection(card), bad_fill])
    planner = v2.HistoryPlannerV2(
        v1.HistoryContractClient(delegate), _library(card), trace_prefix="synthetic"
    )
    with pytest.raises((ValueError, TypeError)):
        planner(_state())
    assert len(delegate.requests) == 2
    assert delegate.calls[-1]["status"] == "completed"


def test_transport_failure_never_retries_or_fabricates_a_fill(card):
    wire, delegate = _wire([APIRequestError("synthetic transport failure")])
    with pytest.raises(APIRequestError, match="transport failure"):
        wire.complete(_messages(card), trace_id="failed-wire", prompt_version=v1.FILL_VERSION)
    assert len(delegate.requests) == 1
    assert delegate.requests[0]["prompt_version"] == v2.FILL_VERSION
    assert delegate.calls == []


@pytest.mark.parametrize("invalid", [False, True])
def test_real_budget_journal_records_new_wire_version_and_keeps_completed_costs(
    card, tmp_path, invalid
):
    class SyntheticTransport:
        transport_source = "synthetic_no_network"

        def __init__(self):
            self.attempts = 0
            self.config = ChatConfig(
                "https://synthetic.invalid",
                "synthetic-model",
                "UNUSED_SYNTHETIC_KEY",
                max_calls=2,
                max_output_tokens=64,
                json_object_mode=True,
            )

        def complete(self, messages, *, trace_id, prompt_version):
            self.attempts += 1
            value = _fill(bindings=[_binding("undeclared")]) if invalid else _fill()
            audit = tmp_path / "synthetic_http.json"
            audit.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "prompt_version": prompt_version,
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "transport_source": self.transport_source,
                    }
                ),
                encoding="utf-8",
            )
            return ChatResponse(
                json.dumps(value),
                "synthetic-model",
                "synthetic-model",
                "synthetic-response",
                "synthetic-request",
                10,
                5,
                0.001,
                audit,
                self.transport_source,
            )

    transport = SyntheticTransport()
    budget = DurableBudgetClient(transport, PriceLimits(budget_cny=1), tmp_path / "journal")
    wire = v2.FillWireV2Client(v1.HistoryContractClient(budget))
    if invalid:
        with pytest.raises(ValueError, match="not declared"):
            wire.complete(
                _messages(card), trace_id="synthetic/durable", prompt_version=v1.FILL_VERSION
            )
    else:
        wire.complete(_messages(card), trace_id="synthetic/durable", prompt_version=v1.FILL_VERSION)
    intent = json.loads((tmp_path / "journal" / "0000_intent.json").read_text(encoding="utf-8"))
    after = json.loads((tmp_path / "journal" / "0000_after.json").read_text(encoding="utf-8"))
    http = json.loads((tmp_path / "synthetic_http.json").read_text(encoding="utf-8"))
    assert intent["prompt_version"] == after["calls"][0]["prompt_version"] == v2.FILL_VERSION
    assert http["prompt_version"] == v2.FILL_VERSION and http["status"] == "completed"
    assert after["calls"][0]["status"] == "completed"
    assert after["api_requests"] == transport.attempts == 1
    assert budget.report()["estimated_actual_cny"] == pytest.approx(0.000006)
    assert budget.report()["reserved_cny"] > 0
    assert budget.block_reason == ("local_output_contract_failure" if invalid else None)
