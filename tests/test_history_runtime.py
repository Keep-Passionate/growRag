"""Synthetic history-only execution contracts; never call an API or load gold."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from growrag.experiments import history_runtime as runtime
from growrag.experiments.api_client import APIRequestError
from growrag.experiments.history_context import (
    REWRITE_PROMPT_VERSION,
    SELECT_PROMPT_VERSION,
    prepare_history_context,
)
from growrag.experiments.operator_model_v2 import answer_episode
from growrag.experiments.operator_schemas import READER_VERSION
from growrag.experiments.operator_schemas_v3 import PLANNER_VERSIONS
from growrag.experiments.protocol import Evidence, RuntimeQuestion
from growrag.history_library import (
    REPRESENTATIONS,
    FrozenHistoryLibrary,
    HistoryCard,
    HistoryCondition,
    HistoryExample,
    HistoryRecord,
)
from growrag.macro_operators import GapField, OperatorSpec, QueryStep
from growrag.operator_loop import Observation, run_operator_episode

QUESTION = RuntimeQuestion("synthetic-question", "Where was the founder of Cedar Lab born?")
EVIDENCE = (Evidence("current-1", "Cedar Lab", 0, "Avery Finch founded Cedar Lab."),)


class SyntheticDelegate:
    """A paid-looking response ledger is synthetic, not a mocked effectiveness result."""

    def __init__(self, responses, *, object_mode=True, schema_mode=False):
        self.responses = list(responses)
        self.config = SimpleNamespace(json_object_mode=object_mode, json_schema_mode=schema_mode)
        self.calls = []
        self.requests = []
        self.block_reason = None
        self.estimated_actual_cny = 0.0
        self.reserved_cny = 0.0

    def complete(self, messages, *, trace_id, prompt_version):
        if self.block_reason:
            raise APIRequestError("synthetic client blocked; no request sent")
        self.requests.append(
            {"messages": deepcopy(messages), "trace_id": trace_id, "version": prompt_version}
        )
        if not self.responses:
            pytest.fail("unexpected additional model request or fallback")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        content = value if isinstance(value, str) else json.dumps(value)
        self.calls.append(
            {
                "trace_id": trace_id,
                "prompt_version": prompt_version,
                "status": "completed",
                "api_requests": 1,
                "input_tokens": 7,
                "output_tokens": 3,
                "estimated_actual_cny": 0.0000038,
                "reserved_cny": 0.001,
            }
        )
        self.estimated_actual_cny += 0.0000038
        self.reserved_cny += 0.001
        return SimpleNamespace(content=content, input_tokens=7, output_tokens=3)


def _library(*cards, candidate_ids=()):
    return FrozenHistoryLibrary(
        "synthetic-history-runtime",
        (),
        tuple(
            HistoryRecord(
                card,
                "reference",
                "synthetic/manual",
                "0" * 64,
                status="candidate" if card.card_id in candidate_ids else "published",
            )
            for card in cards
        ),
    )


@pytest.fixture
def rule():
    return HistoryCard(
        "SEMANTIC_CARD",
        "1",
        "Clarify wording",
        "Rephrase the current target without historical entities.",
        "Use synonymous current-question terms without changing its objective.",
        (HistoryExample("Who founded HistoricalPine?", "HistoricalPine founder HistoricalPerson"),),
        (HistoryCondition("wording_unclear", "Current wording benefits from clarification."),),
    )


@pytest.fixture
def template():
    return HistoryCard(
        "LOOKUP_CARD",
        "1",
        "Location lookup",
        "Search for an explicitly supplied current entity.",
        "Fill entity and render the unchanged location template.",
        (HistoryExample("Where is HistoricalPine?", "HistoricalPine location"),),
        (HistoryCondition("entity_known", "A current entity is known."),),
        OperatorSpec(
            "LOCATION_SPEC",
            "1",
            ("lookup",),
            (GapField("entity"),),
            (QueryStep("search", "{entity} location"),),
        ),
    )


@pytest.fixture
def bridge(template):
    return replace(
        template,
        card_id="BRIDGE_CARD",
        operator_spec=OperatorSpec(
            "BRIDGE_SPEC",
            "1",
            ("lookup",),
            (),
            (QueryStep("search", "{founder} birthplace", requires_bindings=("founder",)),),
        ),
    )


def _selection(card_id, reason="uncertain_match"):
    return {"selected_card_id": card_id, "reason": reason}


def _fill(*, gap=None, bindings=None, intent="lookup"):
    return {
        "intent": intent,
        "constraints": [],
        "gap_entries": [{"name": "entity", "value": "Cedar Lab"}] if gap is None else gap,
        "bindings": [] if bindings is None else bindings,
    }


def _binding(name="founder", value="Avery Finch", ids=None):
    return {"name": name, "value": value, "evidence_ids": ["current-1"] if ids is None else ids}


def _state(*, evidence=EVIDENCE, remaining=1):
    return Observation(QUESTION, evidence, (), remaining, 1)


def _planner(library, responses, *, origin="reuse", records=None):
    delegate = SyntheticDelegate(responses)
    client = runtime.HistoryContractClient(delegate)
    planner = runtime.HistoryPlanner(
        client,
        library,
        trace_prefix="synthetic-run/question",
        origin=origin,
        on_record=None if records is None else records.append,
    )
    return planner, client, delegate


def test_pure_history_rule_action_is_transient_without_create_or_library_write(rule):
    library = _library(rule)
    original = library.to_json()
    records = []
    planner, _, delegate = _planner(
        library,
        [_selection(rule.card_id), {"query": "Cedar Lab founder birthplace"}],
        records=records,
    )
    proposal = planner(_state())
    assert proposal.origin == "reuse"
    assert proposal.goal.original_question == QUESTION.text
    assert proposal.gap["query_text"] == f"{QUESTION.text} Cedar Lab founder birthplace"
    assert proposal.spec.operator_id.startswith("RULE_QUERY_")
    assert library.to_json() == original
    assert [r["version"] for r in delegate.requests] == [
        SELECT_PROMPT_VERSION,
        REWRITE_PROMPT_VERSION,
    ]
    assert all(r["version"] != PLANNER_VERSIONS["fresh"] for r in delegate.requests)
    executed = next(row for row in records if row["kind"] == "history_execution")
    assert executed["selected_card_id"] == rule.card_id
    assert executed["selected_card_version"] == rule.version
    assert executed["action_kind"] == "rewrite_rule"


@pytest.mark.parametrize("origin", ["reuse", "static"])
def test_template_executes_unchanged_selected_spec_and_logs_identity(template, origin):
    records = []
    planner, _, delegate = _planner(
        _library(template), [_selection(template.card_id), _fill()], origin=origin, records=records
    )
    result = planner(_state())
    assert result.spec == template.operator_spec
    assert result.origin == origin
    assert result.gap == {"entity": "Cedar Lab"}
    assert result.bindings == ()
    assert [r["version"] for r in delegate.requests] == [
        SELECT_PROMPT_VERSION,
        runtime.FILL_VERSION,
    ]
    payload = json.loads(delegate.requests[1]["messages"][1]["content"])
    assert payload["selected_card"]["card_id"] == template.card_id
    assert payload["selected_card"]["operator_spec"]["operator_id"] == "LOCATION_SPEC"
    assert records[-1]["selected_card_id"] == template.card_id


@pytest.mark.parametrize("kind", ["rule", "template"])
def test_selector_projection_does_not_change_execution_messages_or_action(
    kind, rule, template, monkeypatch
):
    card = rule if kind == "rule" else template
    responses = [
        _selection(card.card_id),
        {"query": "Cedar Lab director"} if kind == "rule" else _fill(),
    ]
    executions, proposals, projections = [], [], []
    for representation in REPRESENTATIONS:

        def projected(*args, _projection=representation, **kwargs):
            kwargs["representation"] = _projection
            return prepare_history_context(*args, **kwargs)

        monkeypatch.setattr(runtime, "prepare_history_context", projected)
        planner, _, delegate = _planner(_library(card), responses)
        proposals.append(planner(_state()))
        projections.append(delegate.requests[0]["messages"])
        executions.append(delegate.requests[1]["messages"])
    assert executions[0] == executions[1] == executions[2]
    assert proposals[0] == proposals[1] == proposals[2]
    assert projections[0] != projections[1] != projections[2]


def test_valid_bridge_binding_uses_only_visible_current_evidence(bridge):
    planner, _, _ = _planner(
        _library(bridge), [_selection(bridge.card_id), _fill(gap=[], bindings=[_binding()])]
    )
    result = planner(_state())
    assert result.spec == bridge.operator_spec
    assert result.bindings[0].value == "Avery Finch"
    assert result.bindings[0].evidence_ids == ("current-1",)


@pytest.mark.parametrize(
    "fill",
    [
        _fill(gap=[{"name": "undeclared", "value": "Cedar Lab"}]),
        _fill(gap=[]),
        _fill(gap=[{"name": "entity", "value": "A"}, {"name": "entity", "value": "B"}]),
        _fill(gap=[{"name": "entity", "value": True}]),
        _fill(intent="not-supported"),
        {**_fill(), "operator_spec": {}},
        {**_fill(), "query": "replace the selected template"},
    ],
)
def test_template_rejects_extra_missing_duplicate_or_wrong_type_fields(template, fill):
    planner, _, delegate = _planner(_library(template), [_selection(template.card_id), fill])
    with pytest.raises((ValueError, TypeError, KeyError)):
        planner(_state())
    assert len(delegate.requests) == 2  # No repair call, retry, or FRESH fallback.


@pytest.mark.parametrize(
    "binding",
    [_binding(ids=["unseen"]), _binding(value="HistoricalPerson"), _binding(ids=[])],
)
def test_bridge_rejects_unseen_or_unsupported_bindings(bridge, binding):
    planner, _, delegate = _planner(
        _library(bridge), [_selection(bridge.card_id), _fill(gap=[], bindings=[binding])]
    )
    with pytest.raises((ValueError, TypeError)):
        planner(_state())
    assert len(delegate.requests) == 2


def test_bridge_rejects_binding_found_only_in_unseen_text_tail(bridge):
    hidden = (Evidence("current-1", "Cedar Lab", 0, "padding " * 10000 + "Avery Finch"),)
    planner, _, _ = _planner(
        _library(bridge), [_selection(bridge.card_id), _fill(gap=[], bindings=[_binding()])]
    )
    with pytest.raises(ValueError, match="visible window"):
        planner(_state(evidence=hidden))


def test_template_rejects_binding_not_declared_by_selected_spec(template):
    planner, _, _ = _planner(
        _library(template),
        [_selection(template.card_id), _fill(bindings=[_binding(name="undeclared")])],
    )
    with pytest.raises(ValueError):
        planner(_state())


def test_missing_required_bridge_binding_is_not_filled_by_guessing(bridge):
    planner, _, delegate = _planner(
        _library(bridge), [_selection(bridge.card_id), _fill(gap=[], bindings=[])]
    )
    with pytest.raises(ValueError, match="unbound"):
        planner(_state())
    assert len(delegate.requests) == 2


@pytest.mark.parametrize("selected", ["NOT_IN_LIBRARY", "CANDIDATE_CARD"])
def test_unavailable_selected_card_is_rejected_without_execution_or_fallback(rule, selected):
    candidate = replace(rule, card_id="CANDIDATE_CARD")
    library = _library(rule, candidate, candidate_ids=(candidate.card_id,))
    planner, _, delegate = _planner(library, [_selection(selected)])
    with pytest.raises(ValueError, match="offered published"):
        planner(_state())
    assert len(delegate.requests) == 1


def test_refusal_stops_without_create_fill_or_rewrite(rule):
    planner, _, delegate = _planner(_library(rule), [_selection(None, "no_suitable_card")])
    result = run_operator_episode(QUESTION, lambda q, k: EVIDENCE, planner, retrieval_budget=3)
    assert result.stop_reason == "controller_stop_claim"
    assert result.retrieval_calls == 1
    assert result.proposals[0].origin == "stop"
    assert len(delegate.requests) == 1


def test_empty_library_cannot_create_an_action():
    planner, _, delegate = _planner(_library(), [_selection(None, "no_suitable_card")])
    assert planner(_state()).origin == "stop"
    payload = json.loads(delegate.requests[0]["messages"][1]["content"])
    assert payload["candidate_cards"] == []
    assert len(delegate.requests) == 1


def test_repeated_rule_query_stops_before_a_second_search(rule):
    planner, _, delegate = _planner(
        _library(rule), [_selection(rule.card_id), {"query": QUESTION.text}]
    )
    result = run_operator_episode(QUESTION, lambda q, k: EVIDENCE, planner)
    assert result.stop_reason == "repeated_query_same_fixed_topk"
    assert result.retrieval_calls == 1
    assert len(delegate.requests) == 2


def test_no_new_evidence_stops_without_another_selection_or_fresh(rule):
    planner, _, delegate = _planner(
        _library(rule), [_selection(rule.card_id), {"query": "Cedar Lab founder birthplace"}]
    )
    result = run_operator_episode(QUESTION, lambda q, k: EVIDENCE, planner)
    assert result.stop_reason == "no_new_evidence"
    assert result.retrieval_calls == 2
    assert len(delegate.requests) == 2


def test_rule_history_examples_are_not_automatically_given_to_reader(rule):
    library = _library(rule)
    old = library.fingerprint
    planner, client, delegate = _planner(
        library,
        [
            _selection(rule.card_id),
            {"query": "Cedar Lab founder birthplace"},
            {"answer": "Fableton", "supported": True, "evidence_ids": ["current-2"]},
        ],
    )
    birth = Evidence("current-2", "Avery Finch", 0, "Avery Finch was born in Fableton.")

    def retrieve(query, top_k):
        return EVIDENCE if query == QUESTION.text else (birth,)

    episode = run_operator_episode(QUESTION, retrieve, planner, retrieval_budget=2)
    answer = answer_episode(client, QUESTION, episode.evidence, trace_id="synthetic-run/reader")
    assert answer["answer"] == "Fableton"
    reader_payload = json.loads(delegate.requests[-1]["messages"][1]["content"])
    assert set(reader_payload) == {"original_question", "evidence", "evidence_window_omitted_count"}
    assert reader_payload["original_question"] == QUESTION.text
    assert "HistoricalPine" not in json.dumps(reader_payload)
    assert "HistoricalPerson" not in json.dumps(reader_payload)
    assert library.fingerprint == old
    assert delegate.requests[-1]["version"] == READER_VERSION


def test_shortlist_ignores_examples_card_identity_and_reference_provenance(rule):
    question = RuntimeQuestion("synthetic", "MAGICENTITY")
    baseline = _library(rule)
    score = runtime.shortlist_cards(question, baseline)[0]["lexical_score"]
    changed = replace(
        rule,
        card_id="MAGICENTITY",
        version="MAGICENTITY",
        examples=(HistoryExample("MAGICENTITY", "MAGICENTITY MAGICENTITY"),),
    )
    assert runtime.shortlist_cards(question, _library(changed))[0]["lexical_score"] == score


def test_shortlist_ignores_nested_operator_id_and_version(template):
    question = RuntimeQuestion("synthetic", "MAGICENTITY")
    score = runtime.shortlist_cards(question, _library(template))[0]["lexical_score"]
    changed = replace(
        template,
        operator_spec=replace(
            template.operator_spec, operator_id="MAGICENTITY", version="MAGICENTITY"
        ),
    )
    assert runtime.shortlist_cards(question, _library(changed))[0]["lexical_score"] == score


def test_shortlist_only_published_and_deterministic_tie_break(rule):
    cards = [replace(rule, card_id=f"CARD_{letter}") for letter in "DBAC"]
    library = _library(*cards, candidate_ids=("CARD_D",))
    assert [x["card_id"] for x in runtime.shortlist_cards(QUESTION, library)] == [
        "CARD_A",
        "CARD_B",
        "CARD_C",
    ]


@pytest.mark.parametrize("limit", [True, 0, 4, 1.0])
def test_shortlist_limit_is_bounded_and_typed(rule, limit):
    with pytest.raises(ValueError):
        runtime.shortlist_cards(QUESTION, _library(rule), limit=limit)


@pytest.mark.parametrize("object_mode,schema_mode", [(False, False), (False, True), (True, True)])
def test_contract_client_requires_explicit_json_object_mode(object_mode, schema_mode):
    delegate = SyntheticDelegate([], object_mode=object_mode, schema_mode=schema_mode)
    with pytest.raises(ValueError, match="JSON-object"):
        runtime.HistoryContractClient(delegate)
    assert delegate.requests == []


@pytest.mark.parametrize(
    "version",
    [
        "unknown",
        SELECT_PROMPT_VERSION + "-changed",
        PLANNER_VERSIONS["memory"],
        PLANNER_VERSIONS["static"],
    ],
)
def test_unregistered_versions_fail_before_request_or_budget_reservation(version):
    delegate = SyntheticDelegate([])
    client = runtime.HistoryContractClient(delegate)
    with pytest.raises(ValueError, match="not registered"):
        client.complete([], trace_id="synthetic/unknown", prompt_version=version)
    assert delegate.requests == [] and delegate.calls == []
    assert delegate.reserved_cny == 0


@pytest.mark.parametrize(
    "version",
    [
        SELECT_PROMPT_VERSION,
        REWRITE_PROMPT_VERSION,
        runtime.FILL_VERSION,
        PLANNER_VERSIONS["fresh"],
        READER_VERSION,
    ],
)
def test_exact_registered_local_schema_versions(version):
    schema = runtime.schema_for(version)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == set(schema["required"])


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '```json\n{"selected_card_id":null,"reason":"no_suitable_card"}\n```',
        '{"selected_card_id":null,"selected_card_id":"OTHER","reason":"no_suitable_card"}',
        {"selected_card_id": 3, "reason": "uncertain_match"},
        {"selected_card_id": None, "reason": "invented"},
        {**_selection(None, "no_suitable_card"), "query": "invented"},
    ],
)
def test_local_contract_failure_preserves_http_completion_cost_and_blocks_more_calls(response):
    delegate = SyntheticDelegate([response])
    records = []
    client = runtime.HistoryContractClient(delegate, on_record=records.append)
    with pytest.raises((ValueError, TypeError, KeyError)):
        client.complete([], trace_id="synthetic/bad", prompt_version=SELECT_PROMPT_VERSION)
    assert delegate.block_reason == "local_output_contract_failure"
    assert client.calls is delegate.calls
    assert len(client.calls) == 1
    assert client.calls[0]["status"] == "completed"
    assert client.calls[0]["api_requests"] == 1
    assert client.calls[0]["input_tokens"] == 7 and client.calls[0]["output_tokens"] == 3
    assert delegate.estimated_actual_cny == pytest.approx(0.0000038)
    assert delegate.reserved_cny == pytest.approx(0.001)
    assert records[-1]["kind"] == "local_contract_failure"
    with pytest.raises(APIRequestError, match="blocked"):
        client.complete([], trace_id="synthetic/after", prompt_version=SELECT_PROMPT_VERSION)
    assert len(delegate.requests) == 1


def test_successful_local_contract_does_not_mutate_transport_usage():
    delegate = SyntheticDelegate([_selection(None, "no_suitable_card")])
    records = []
    client = runtime.HistoryContractClient(delegate, on_record=records.append)
    result = client.complete([], trace_id="synthetic/ok", prompt_version=SELECT_PROMPT_VERSION)
    assert json.loads(result.content) == _selection(None, "no_suitable_card")
    assert delegate.block_reason is None
    assert records[-1]["kind"] == "local_contract_pass"
    assert delegate.calls[0]["status"] == "completed"


def test_transport_failure_propagates_without_rule_or_fresh_fallback(rule):
    planner, _, delegate = _planner(
        _library(rule), [APIRequestError("synthetic transport failure")]
    )
    with pytest.raises(APIRequestError, match="synthetic transport failure"):
        planner(_state())
    assert len(delegate.requests) == 1
    assert delegate.calls == []
