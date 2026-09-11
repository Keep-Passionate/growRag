"""Offline runner/adaptor isolation tests. Every transport here is synthetic."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_representation_views import _bundle

from growrag.experiments.api_client import ChatConfig, ChatResponse
from growrag.experiments.protocol import (
    Answer,
    BackendCallError,
    CallResult,
    Evidence,
    ExecutionKind,
    RuntimeQuestion,
)
from growrag.experiments.query_comparison import ComparisonComponents
from growrag.experiments.representation_runner import (
    CANONICAL_VERSION,
    SELECTOR_VERSION,
    APICanonicalQueryGenerator,
    APIRepresentationSelector,
    RepresentationSpec,
    canonical_payload,
    model_fingerprint,
    run_representation_comparison,
)
from growrag.experiments.representation_views import (
    CandidateSet,
    RepresentationKind,
    retrieve_candidates,
)
from growrag.outer_loop import RagReply


def _spec(count=2):
    sources = tuple(_bundle(label) for label in ("alpha", "beta", "gamma")[:count])
    return RepresentationSpec(
        retrieve_candidates(
            RuntimeQuestion("target-id", "When was another University founded?", "synthetic"),
            sources,
        ),
        "SOURCE_INTENT_SENTINEL",
        "rag-v1",
        "generator-v1",
        "selector-v1",
    )


class _Harness:
    def __init__(self, spec, choices=("c1", "c1", "c1", "FRESH")):
        self.spec = spec
        self.choices = choices
        self.calls = []
        self.created_selectors = []

    def factory(self, route):
        harness = self

        class Backend:
            execution_kind = ExecutionKind.MOCK

            def run(self, request):
                harness.calls.append(("rag", route, request))
                return CallResult(
                    RagReply(Answer(route), (Evidence("e1", "target doc", 0, "target evidence"),)),
                    transport_source="mock",
                )

        class Generator:
            execution_kind = ExecutionKind.MOCK
            comparison_protocol_id = CANONICAL_VERSION

            def generate(self, question, decision, *, evidence=(), previous_queries=()):
                assert evidence == previous_queries == ()
                payload = canonical_payload(question, decision)
                harness.calls.append(("rewrite", route, payload))
                return CallResult(f"{route} independent lookup", transport_source="mock")

        return ComparisonComponents(
            Backend(),
            None if route == "BASE" else Generator(),
            self.spec.rag_fingerprint,
            None if route == "BASE" else self.spec.generator_fingerprint,
        )

    def selector_factory(self):
        choice = self.choices[len(self.created_selectors)]
        harness = self

        class Selector:
            execution_kind = ExecutionKind.MOCK
            fingerprint = harness.spec.selector_fingerprint

            def select(self, payload):
                harness.calls.append(("select", payload))
                if isinstance(choice, Exception):
                    raise choice
                return CallResult(choice, transport_source="mock")

        selector = Selector()
        self.created_selectors.append(selector)
        return selector

    def save_selections(self, selections):
        self.calls.append(("save_selections", selections))

    def save_outcome(self, outcome):
        self.calls.append(("save_outcome", outcome))

    def run(self, **kwargs):
        defaults = {
            "execution_kind": ExecutionKind.MOCK,
            "on_selections": self.save_selections,
            "on_outcome": self.save_outcome,
        }
        return run_representation_comparison(
            self.spec, self.factory, self.selector_factory, **(defaults | kwargs)
        )


def test_all_selections_are_saved_before_any_route_and_each_route_executes_once():
    harness = _Harness(_spec(3))
    result = harness.run()
    assert [call[0] for call in harness.calls[:5]] == ["select"] * 4 + ["save_selections"]
    assert [selection.selected_id for selection in result.selections] == ["c1"] * 3 + ["FRESH"]
    assert [selection.status for selection in result.selections] == ["selected"] * 3 + ["refused"]
    assert {outcome.route_id for outcome in result.outcomes} == {"BASE", "FRESH", "c1", "c2", "c3"}
    assert all(outcome.status == "completed" for outcome in result.outcomes)
    rag_calls = [call for call in harness.calls if call[0] == "rag"]
    rewrite_calls = [call for call in harness.calls if call[0] == "rewrite"]
    assert len(rag_calls) == 5 and len(rewrite_calls) == 4
    assert sum(call[1] == "c1" for call in rag_calls) == 1
    assert all(call[2].question == harness.spec.candidates.question for call in rag_calls)
    assert len({outcome.cache_key for outcome in result.outcomes}) == 5
    assert result.to_dict()["synthetic_demo"] is True
    assert result.execution_kind is ExecutionKind.MOCK


def test_canonical_execution_payload_has_only_body_not_selection_view_or_conditions():
    harness = _Harness(_spec())
    harness.run()
    for _, route, payload in (call for call in harness.calls if call[0] == "rewrite"):
        if route == "FRESH":
            assert set(payload) == {"original_question", "form", "intent"}
        else:
            assert set(payload) == {
                "original_question",
                "form",
                "intent",
                "optional_historical_procedure",
            }
            assert set(payload["optional_historical_procedure"]) == {"body"}
            assert payload["optional_historical_procedure"]["body"] == {
                "rewrite_rule": "SOURCE_ACTION_SENTINEL",
                "preserve": ["entity"],
            }
        raw = json.dumps(payload)
        for forbidden in (
            "SOURCE_CONDITION_SENTINEL",
            "SOURCE_PATTERN_SENTINEL",
            "SOURCE_ANSWER_SENTINEL",
            "SOURCE_CRITERION_SENTINEL",
            "source-alpha",
            "memory-alpha",
            "candidate_id",
            "representation",
            "rationale",
            "Unknown: other relationships",
            "source_question",
        ):
            assert forbidden not in raw


@pytest.mark.parametrize(
    "invalid", ["c9", " c1", None, {"candidate_id": "c1"}, ValueError("broken")]
)
def test_invalid_selector_outputs_become_explicit_error_not_successful_refusal(invalid):
    harness = _Harness(_spec(), choices=(invalid,) * 4)
    result = harness.run()
    assert all(
        row.status == "selector_error" and row.selected_id == "FRESH" for row in result.selections
    )
    assert all(row.event is not None and row.event.status == "error" for row in result.selections)
    assert all(outcome.status == "completed" for outcome in result.outcomes)


def test_no_candidate_makes_no_selector_call_but_keeps_base_and_fresh():
    harness = _Harness(_spec(0), choices=())
    result = harness.run()
    assert not harness.created_selectors
    assert not any(call[0] == "select" for call in harness.calls)
    assert all(row.status == "no_candidates" and row.event is None for row in result.selections)
    assert {row.route_id for row in result.outcomes} == {"BASE", "FRESH"}
    assert sum(call[0] == "rag" for call in harness.calls) == 2


def test_view_change_changes_selection_spec_but_not_execution_cache_identity():
    spec = _spec()
    candidate = spec.candidates.candidates[0]
    modified_bundle = replace(
        candidate.bundle,
        views=tuple(
            replace(view, text="Different condition wording, not a different action.")
            if view.kind is RepresentationKind.M5
            else view
            for view in candidate.bundle.views
        ),
    )
    modified_candidates = replace(
        spec.candidates,
        candidates=(replace(candidate, bundle=modified_bundle), *spec.candidates.candidates[1:]),
    )
    changed = replace(spec, candidates=modified_candidates)
    assert spec.fingerprint != changed.fingerprint
    for route in spec.route_ids:
        assert spec.outcome_key(route, ExecutionKind.MOCK) == changed.outcome_key(
            route, ExecutionKind.MOCK
        )
    assert spec.route_ids == changed.route_ids
    selector_changed = replace(spec, selector_fingerprint="another selector")
    assert spec.fingerprint != selector_changed.fingerprint
    assert spec.outcome_key("c1", ExecutionKind.MOCK) == selector_changed.outcome_key(
        "c1", ExecutionKind.MOCK
    )


@pytest.mark.parametrize(
    "change", ["query", "body", "model", "index", "seed", "execution_kind", "memory_version"]
)
def test_outcome_cache_key_binds_execution_inputs(change):
    spec = _spec()
    changed, kind = spec, ExecutionKind.MOCK
    if change == "query":
        changed = replace(
            spec,
            candidates=replace(
                spec.candidates, question=RuntimeQuestion("new", "Another university date?")
            ),
        )
    elif change in ("body", "memory_version"):
        candidate = spec.candidates.candidates[0]
        action = candidate.canonical_action
        if change == "body":
            payload = json.loads(action.text)
            payload["body"]["rewrite_rule"] = "Use a different procedural rewrite."
            action = replace(action, text=json.dumps(payload))
        else:
            action = replace(action, memory_id=action.memory_id + "-new-version")
        bundle = replace(candidate.bundle, canonical_action=action)
        changed = replace(
            spec,
            candidates=replace(
                spec.candidates,
                candidates=(replace(candidate, bundle=bundle), *spec.candidates.candidates[1:]),
            ),
        )
    elif change == "model":
        changed = replace(spec, generator_fingerprint="generator-v2")
    elif change == "index":
        changed = replace(spec, rag_fingerprint="different-index")
    elif change == "seed":
        changed = replace(spec, seed=43)
    else:
        kind = ExecutionKind.REAL
    assert spec.outcome_key("c1", ExecutionKind.MOCK) != changed.outcome_key("c1", kind)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"allow_real": True},
        {"allow_real": True, "on_selections": lambda _: None},
        {"allow_real": True, "on_outcome": lambda _: None},
    ],
)
def test_real_run_opt_in_and_both_persistence_callbacks_required_before_factories(kwargs):
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("should not construct a runtime")

    with pytest.raises(ValueError, match="allow_real|callbacks"):
        run_representation_comparison(
            _spec(), forbidden, forbidden, execution_kind=ExecutionKind.REAL, **kwargs
        )
    assert not calls


@pytest.mark.parametrize(
    "corruption",
    [
        "rag_config",
        "generator_config",
        "backend_kind",
        "generator_kind",
        "protocol",
        "selector_kind",
        "selector_config",
        "shared_backend",
        "shared_generator",
        "shared_selector",
        "base_generator",
        "factory_type",
    ],
)
def test_incompatible_or_shared_components_rejected_before_any_model_call(corruption):
    harness = _Harness(_spec())
    original_factory, original_selector = harness.factory, harness.selector_factory
    shared = original_factory("c1")
    shared_selector = original_selector()
    harness.created_selectors.clear()

    def factory(route):
        value = original_factory(route)
        if corruption == "factory_type":
            return None
        if corruption == "rag_config":
            return replace(value, rag_fingerprint="other")
        if corruption == "backend_kind":
            value.backend.execution_kind = ExecutionKind.REAL
        if corruption == "shared_backend":
            return replace(value, backend=shared.backend)
        if corruption == "base_generator" and route == "BASE":
            return replace(value, generator=shared.generator, generator_fingerprint="generator-v1")
        if route != "BASE":
            if corruption == "generator_config":
                return replace(value, generator_fingerprint="other")
            if corruption == "generator_kind":
                value.generator.execution_kind = ExecutionKind.REAL
            if corruption == "protocol":
                value.generator.comparison_protocol_id = "wrong-prompt"
            if corruption == "shared_generator":
                return replace(value, generator=shared.generator)
        return value

    def selector_factory():
        value = original_selector()
        if corruption == "selector_kind":
            value.execution_kind = ExecutionKind.REAL
        elif corruption == "selector_config":
            value.fingerprint = "other"
        elif corruption == "shared_selector":
            return shared_selector
        return value

    with pytest.raises((ValueError, TypeError)):
        run_representation_comparison(
            harness.spec, factory, selector_factory, execution_kind=ExecutionKind.MOCK
        )
    assert not harness.calls


def test_stop_before_work_keeps_explicit_unexecuted_records():
    harness = _Harness(_spec())
    result = harness.run(stop_requested=lambda: True)
    assert all(row.status == "stopped" and row.event is None for row in result.selections)
    assert all(row.status == "stopped" and row.result is None for row in result.outcomes)
    assert not any(call[0] in {"select", "rewrite", "rag"} for call in harness.calls)
    assert len(result.outcomes) == 4


def test_stop_after_one_outcome_keeps_remaining_records_without_running_them():
    harness = _Harness(_spec())
    state = {"stop": False}

    def persist(outcome):
        harness.save_outcome(outcome)
        state["stop"] = True

    result = harness.run(on_outcome=persist, stop_requested=lambda: state["stop"])
    assert result.outcomes[0].status == "completed"
    assert all(row.status == "stopped" and row.result is None for row in result.outcomes[1:])
    assert sum(call[0] == "rag" for call in harness.calls) == 1


def test_failed_selection_persistence_prevents_every_execution():
    harness = _Harness(_spec())

    def failed(_):
        raise OSError("storage unavailable")

    with pytest.raises(OSError, match="storage unavailable"):
        harness.run(on_selections=failed)
    assert [call[0] for call in harness.calls] == ["select"] * 4


def test_failed_outcome_persistence_prevents_later_routes():
    harness = _Harness(_spec())

    def failed(_):
        raise OSError("storage unavailable")

    with pytest.raises(OSError, match="storage unavailable"):
        harness.run(on_outcome=failed)
    assert sum(call[0] == "rag" for call in harness.calls) == 1


class _ChatClient:
    transport_source = "mock"

    def __init__(self, content):
        self.content = content
        self.requests = []
        self.config = ChatConfig("https://example.invalid/v1", "mock-model-v1", "NEVER_READ", 4)

    def complete(self, messages, *, trace_id, prompt_version):
        self.requests.append((messages, trace_id, prompt_version))
        return ChatResponse(
            self.content,
            self.config.model,
            self.config.model,
            "mock-response",
            "mock-request",
            10,
            3,
            0,
            Path("synthetic-audit-only.json"),
            "mock",
        )


def test_api_adapters_project_inputs_and_use_versioned_prompts_without_network():
    spec = _spec()
    selector_client = _ChatClient('{"candidate_id":"c1"}')
    selector = APIRepresentationSelector(selector_client)
    result = selector.select(spec.candidates.selector_payload(RepresentationKind.M5))
    assert result.value == "c1" and result.transport_source == "mock"
    assert result.usage.api_requests == 0
    messages, trace, version = selector_client.requests[0]
    assert version == SELECTOR_VERSION and trace.startswith("select:")
    assert json.loads(messages[1]["content"]) == spec.candidates.selector_payload(
        RepresentationKind.M5
    )
    generator_client = _ChatClient('{"query":"  rewritten search  "}')
    generator = APICanonicalQueryGenerator(generator_client)
    result = generator.generate(spec.candidates.question, spec.decision("c1"))
    assert result.value == "rewritten search" and result.usage.api_requests == 0
    messages, trace, version = generator_client.requests[0]
    assert version == CANONICAL_VERSION and trace.startswith("canonical_rewrite:")
    assert json.loads(messages[1]["content"]) == canonical_payload(
        spec.candidates.question, spec.decision("c1")
    )
    assert "SOURCE_CONDITION_SENTINEL" not in messages[1]["content"]
    assert generator.execution_kind is selector.execution_kind is ExecutionKind.MOCK


@pytest.mark.parametrize(
    "content",
    [
        '{"candidate_id":"c1","candidate_id":"FRESH"}',
        '{"candidate_id":"c9"}',
        '{"candidate_id":"c1","reason":"extra"}',
        '{"candidate_id":null}',
        '{"candidate_id":["c1"]}',
        "[]",
        "not json",
    ],
)
def test_selector_adapter_rejects_bad_or_duplicate_json_keys_without_retry(content):
    client = _ChatClient(content)
    with pytest.raises(BackendCallError, match="invalid selector"):
        APIRepresentationSelector(client).select(
            _spec().candidates.selector_payload(RepresentationKind.M1)
        )
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "content",
    [
        '{"query":"first","query":"second"}',
        '{"query":""}',
        '{"query":null}',
        '{"query":"valid","reason":"extra"}',
        "[]",
        "not json",
        json.dumps({"query": "x" * 2001}),
    ],
)
def test_canonical_adapter_rejects_bad_or_duplicate_json_keys_without_retry(content):
    spec, client = _spec(), _ChatClient(content)
    with pytest.raises(BackendCallError, match="invalid canonical query"):
        APICanonicalQueryGenerator(client).generate(
            spec.candidates.question, spec.decision("FRESH")
        )
    assert len(client.requests) == 1


@pytest.mark.parametrize("extra", ["gold", "gap", "type", "reason", "representation"])
def test_selector_adapter_rejects_nonwhitelisted_input_before_transport(extra):
    client = _ChatClient('{"candidate_id":"FRESH"}')
    payload = _spec().candidates.selector_payload(RepresentationKind.M2)
    payload[extra] = "forbidden"
    with pytest.raises(ValueError):
        APIRepresentationSelector(client).select(payload)
    assert not client.requests


def test_adapter_rejects_wrong_stage_and_response_provenance():
    spec, client = _spec(), _ChatClient('{"query":"lookup"}')
    generator = APICanonicalQueryGenerator(client)
    with pytest.raises(ValueError, match="before retrieval"):
        generator.generate(
            spec.candidates.question, spec.decision("FRESH"), previous_queries=("old",)
        )
    assert not client.requests
    original = client.complete

    def mismatched(*args, **kwargs):
        return replace(original(*args, **kwargs), transport_source="live_api")

    client.complete = mismatched
    with pytest.raises(BackendCallError, match="provenance mismatch"):
        generator.generate(spec.candidates.question, spec.decision("FRESH"))


def test_nonsecret_model_fingerprint_changes_with_model_and_decoding_configuration():
    client = _ChatClient('{"query":"lookup"}')
    original = model_fingerprint(client, CANONICAL_VERSION)
    client.config = replace(client.config, model="mock-model-v2")
    assert original != model_fingerprint(client, CANONICAL_VERSION)
    client.config = replace(client.config, model="mock-model-v1", temperature=0.2)
    assert original != model_fingerprint(client, CANONICAL_VERSION)
    assert original != model_fingerprint(_ChatClient(""), "new-prompt")


def test_spec_rejects_bad_metadata_and_changed_experiment_intent():
    spec = _spec()
    for changes in ({"seed": True}, {"seed": -1}, {"intent": "other"}, {"rag_fingerprint": ""}):
        with pytest.raises(ValueError):
            replace(spec, **changes)
    with pytest.raises(TypeError):
        replace(spec, candidates={"gold": "leaked"})
    assert CandidateSet(spec.candidates.question, ()).candidates == ()


def test_real_opt_in_does_not_allow_mock_components_to_be_reported_as_real():
    harness = _Harness(_spec())
    with pytest.raises(ValueError, match="provenance mismatch|synthetic source memory"):
        harness.run(execution_kind=ExecutionKind.REAL, allow_real=True)
    assert not harness.calls


def test_selector_response_provenance_mismatch_is_recorded_as_error():
    harness = _Harness(_spec())
    original = harness.selector_factory

    def factory():
        selector = original()
        select = selector.select

        def mismatched(payload):
            return replace(select(payload), transport_source="live_api")

        selector.select = mismatched
        return selector

    harness.selector_factory = factory
    result = harness.run()
    assert all(row.status == "selector_error" for row in result.selections)
    assert all(row.selected_id == "FRESH" for row in result.selections)
    assert all(row.event.status == "error" for row in result.selections)


@pytest.mark.parametrize("component", ["rewrite", "rag"])
def test_execution_failure_remains_a_failed_outcome_without_automatic_retry(component):
    harness = _Harness(_spec())
    original_factory = harness.factory

    def factory(route):
        value = original_factory(route)
        if route == "c1":

            def fail(*args, **kwargs):
                harness.calls.append(("intentional_failure", route, component))
                raise BackendCallError("synthetic failure", transport_source="mock")

            if component == "rewrite":
                value.generator.generate = fail
            else:
                value.backend.run = fail
        return value

    harness.factory = factory
    result = harness.run()
    outcomes = {row.route_id: row for row in result.outcomes}
    assert outcomes["c1"].status == "execution_failed"
    assert outcomes["c1"].result is not None
    assert any(event.status == "error" for event in outcomes["c1"].result.events)
    assert not outcomes["c1"].result.state.rounds
    assert all(row.status == "completed" for key, row in outcomes.items() if key != "c1")
    assert sum(call[0] == "intentional_failure" for call in harness.calls) == 1


@pytest.mark.parametrize(
    "corruption", ["extra_metadata", "nonneutral_id", "missing_text", "empty_pool"]
)
def test_selector_adapter_rejects_invalid_candidate_shape_without_transport(corruption):
    client = _ChatClient('{"candidate_id":"FRESH"}')
    payload = _spec().candidates.selector_payload(RepresentationKind.M2)
    if corruption == "extra_metadata":
        payload["candidates"][0]["source_id"] = "private-source-id"
    elif corruption == "nonneutral_id":
        payload["candidates"][0]["candidate_id"] = "memory-alpha:v1"
    elif corruption == "missing_text":
        del payload["candidates"][0]["text"]
    else:
        payload["candidates"] = []
    with pytest.raises(ValueError):
        APIRepresentationSelector(client).select(payload)
    assert not client.requests
