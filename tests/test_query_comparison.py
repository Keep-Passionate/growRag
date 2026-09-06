"""PRE-retrieval comparison contract tests; all execution is explicitly mock."""

import hashlib
import json
from dataclasses import replace

import pytest
from test_query_card_views import EXPAND, PARAPHRASE, MockClient, compile_view, typed_card

from growrag.experience import ActivationStage
from growrag.experiments.comparison_report import comparison_report, save_comparison
from growrag.experiments.protocol import (
    Action,
    Answer,
    BackendCallError,
    CallEvent,
    CallResult,
    Evidence,
    ExecutionKind,
    GoldRecord,
    RuntimeQuestion,
    Usage,
)
from growrag.experiments.query_comparison import (
    ComparisonComponents,
    QueryComparisonSpec,
    run_query_comparison,
)
from growrag.outer_loop import RagReply
from growrag.query_actions import PAIRED_ACTION_PROMPT_VERSION, APISingleQueryGenerator
from growrag.query_operators import RewriteForm

QUESTION = RuntimeQuestion("comparison-target", "When was Northbridge University established?")
EVIDENCE = (Evidence("target-e", "Northbridge", 0, "Northbridge was established in 1900."),)
GOLD = GoldRecord(QUESTION.question_id, ("1900",), (("Northbridge", 0),))
INTENT = "align search vocabulary"  # Experiment instruction, not inferred from card.


def make_spec(body=PARAPHRASE, **overrides):
    values = {
        "question": QUESTION,
        "memory": compile_view(typed_card(body)),
        "form": body.form,
        "intent": INTENT,
        "rag_fingerprint": "fixed-rag-v1",
        "generator_fingerprint": "fixed-generator-v1",
    }
    values.update(overrides)
    return QueryComparisonSpec(**values)


class MockBackend:
    execution_kind = ExecutionKind.MOCK

    def __init__(self, action, trace, *, fail=False, evidence=EVIDENCE, components=()):
        self.action, self.trace, self.fail = action, trace, fail
        self.evidence, self.components = evidence, components
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        self.trace.append(("rag", self.action))
        if self.fail:
            raise BackendCallError(
                "scripted backend failure",
                usage=Usage(3, 2, 0),
                request_id="mock-failed-request",
                audit_path="not-written-mock-audit",
                transport_source="mock",
            )
        citations = ("target-e",) if self.evidence else ()
        return CallResult(
            RagReply(Answer("1900", citations), self.evidence, self.components),
            usage=Usage(11, 7, 0),
            transport_source="mock",
        )


class CountedGenerator:
    execution_kind = ExecutionKind.MOCK
    comparison_protocol_id = PAIRED_ACTION_PROMPT_VERSION

    def __init__(self, action, trace, *, unchanged=False, fail=False):
        self.action, self.trace = action, trace
        self.unchanged, self.fail, self.calls = unchanged, fail, []

    def generate(self, question, decision, **kwargs):
        self.calls.append((question, decision, kwargs))
        self.trace.append(("rewrite", self.action))
        if self.fail:
            raise BackendCallError(
                "scripted rewrite failure", usage=Usage(4, 1, 0), transport_source="mock"
            )
        query = question.text.upper() if self.unchanged else "Northbridge university founding date"
        return CallResult(query, usage=Usage(5, 2, 0), transport_source="mock")


class Factory:
    def __init__(
        self,
        *,
        api_adapter=False,
        fail_backend=None,
        fail_generator=None,
        unchanged=False,
        evidence=EVIDENCE,
        components=(),
    ):
        self.api_adapter = api_adapter
        self.fail_backend, self.fail_generator = fail_backend, fail_generator
        self.unchanged, self.evidence, self.components = unchanged, evidence, components
        self.trace, self.bundles, self.clients = [], {}, {}

    def __call__(self, action):
        self.trace.append(("construct", action))
        backend = MockBackend(
            action,
            self.trace,
            fail=action is self.fail_backend,
            evidence=self.evidence,
            components=self.components,
        )
        generator = None
        if action is not Action.BASE:
            if self.api_adapter:
                client = self.clients[action] = MockClient()
                generator = APISingleQueryGenerator(client, paired_prompt=True)
            else:
                generator = CountedGenerator(
                    action, self.trace, unchanged=self.unchanged, fail=action is self.fail_generator
                )
        bundle = ComparisonComponents(
            backend, generator, "fixed-rag-v1", "fixed-generator-v1" if generator else None
        )
        self.bundles[action] = bundle
        return bundle


def run(factory=None, spec=None):
    return run_query_comparison(
        spec or make_spec(), factory or Factory(), execution_kind=ExecutionKind.MOCK
    )


@pytest.mark.parametrize("body", [PARAPHRASE, EXPAND])
def test_pre_three_arms_start_empty_with_fixed_form_intent_and_no_history_in_fresh(body):
    factory, spec = Factory(), make_spec(body)
    result = run(factory, spec)
    assert tuple(arm.planned_action for arm in result.arms) == spec.action_order
    assert factory.trace[:3] == [("construct", action) for action in spec.action_order]
    assert len({id(bundle.backend) for bundle in factory.bundles.values()}) == 3
    assert factory.bundles[Action.BASE].generator is None
    for action in (Action.FRESH, Action.REUSE):
        generator = factory.bundles[action].generator
        question, decision, arguments = generator.calls[0]
        assert question == QUESTION
        assert decision.form == spec.form
        assert decision.intent == INTENT
        assert arguments == {"evidence": (), "previous_queries": ()}
        assert decision.memory == (spec.memory if action is Action.REUSE else None)
    for bundle in factory.bundles.values():
        assert len(bundle.backend.requests) == 1
        assert bundle.backend.requests[0].question == QUESTION
    assert all(len(arm.result.state.rounds) == 1 for arm in result.arms)
    assert all(arm.result.stop_reason == "unassessed" for arm in result.arms)
    assert result.to_dict()["comparison_scope"] == "pre_retrieval_empty_prefix"


def test_paired_api_adapters_share_system_prompt_and_version_only_reuse_receives_card():
    factory = Factory(api_adapter=True)
    result = run(factory)
    fresh_messages, fresh_meta = factory.clients[Action.FRESH].requests[0]
    reuse_messages, reuse_meta = factory.clients[Action.REUSE].requests[0]
    assert fresh_messages[0] == reuse_messages[0]
    assert (
        fresh_meta["prompt_version"] == reuse_meta["prompt_version"] == PAIRED_ACTION_PROMPT_VERSION
    )
    fresh, reuse = (
        json.loads(fresh_messages[1]["content"]),
        json.loads(reuse_messages[1]["content"]),
    )
    assert "optional_historical_procedure" not in fresh
    assert reuse.pop("optional_historical_procedure")["memory_id"] == result.spec.memory.memory_id
    assert fresh == reuse
    assert fresh["current_evidence"] == fresh["previous_queries"] == []
    assert "gap" not in fresh
    assert comparison_report(result)["total_experiment_usage"]["api_requests"] == 0


@pytest.mark.parametrize(
    "corruption",
    [
        "rag_config",
        "generator_config",
        "backend_shared",
        "generator_shared",
        "backend_kind",
        "generator_kind",
        "unpaired_prompt",
        "base_generator",
    ],
)
def test_invalid_components_are_rejected_before_any_arm_executes(corruption):
    factory, cached = Factory(), {}

    def corrupt(action):
        bundle = factory(action)
        if corruption == "rag_config":
            bundle = replace(bundle, rag_fingerprint="other-rag")
        elif corruption == "backend_kind":
            bundle.backend.execution_kind = ExecutionKind.REAL
        elif corruption == "backend_shared":
            bundle = replace(bundle, backend=cached.setdefault("backend", bundle.backend))
        elif action is Action.BASE and corruption == "base_generator":
            bundle = replace(bundle, generator=CountedGenerator(action, factory.trace))
        elif action is not Action.BASE:
            if corruption == "generator_config":
                bundle = replace(bundle, generator_fingerprint="other-generator")
            elif corruption == "generator_kind":
                bundle.generator.execution_kind = ExecutionKind.REAL
            elif corruption == "generator_shared":
                bundle = replace(bundle, generator=cached.setdefault("generator", bundle.generator))
            elif corruption == "unpaired_prompt":
                bundle.generator.comparison_protocol_id = None
        return bundle

    with pytest.raises(ValueError):
        run(corrupt)
    assert all(operation == "construct" for operation, _ in factory.trace)


def test_real_execution_is_blocked_by_default_before_factory_construction():
    constructed = []
    with pytest.raises(ValueError, match="allow_real"):
        run_query_comparison(
            make_spec(),
            lambda action: constructed.append(action),
            execution_kind=ExecutionKind.REAL,
        )
    assert constructed == []


@pytest.mark.parametrize("field", ["form", "intent"])
def test_card_cannot_change_independently_fixed_experiment_instruction(field):
    mismatch = {"form": RewriteForm.EXPAND, "intent": "a different experimental instruction"}
    with pytest.raises(ValueError, match="independently fixed"):
        make_spec(**{field: mismatch[field]})


def test_non_pre_card_and_pre_card_requiring_evidence_are_rejected():
    view = compile_view(typed_card(stage=ActivationStage.POST_RETRIEVAL))
    with pytest.raises(ValueError, match="PRE_RETRIEVAL"):
        make_spec(memory=view)
    body = replace(EXPAND, requires_current_evidence=True)
    with pytest.raises(ValueError, match="requires current evidence"):
        make_spec(body)


@pytest.mark.parametrize(
    "question",
    [RuntimeQuestion("source-2", "Different text"), RuntimeQuestion("new-id", " Question 2!!! ")],
)
def test_all_sources_including_second_and_renamed_source_are_rejected(question):
    with pytest.raises(ValueError, match="source question"):
        make_spec(question=question)


def test_execution_order_is_question_and_seed_deterministic_not_result_dependent():
    spec = make_spec(seed=12)
    expected = tuple(
        sorted(
            Action,
            key=lambda action: hashlib.sha256(
                f"12:{QUESTION.question_id}:{action.value}".encode()
            ).digest(),
        )
    )
    factory = Factory()
    first = run(factory, spec)
    second = run(Factory(fail_backend=expected[0]), spec)
    assert first.spec.action_order == second.spec.action_order == expected
    assert [action for operation, action in factory.trace if operation == "rag"] == list(expected)
    assert first.spec.fingerprint == replace(spec).fingerprint
    assert first.spec.fingerprint != replace(spec, seed=13).fingerprint


def test_single_backend_error_retains_cost_audit_and_other_arms_continue():
    spec = make_spec()
    failed_action = spec.action_order[0]
    factory = Factory(fail_backend=failed_action)
    result = run(factory, spec)
    report = comparison_report(result, gold=GOLD)
    assert len(result.arms) == 3
    failed = report["arms"][failed_action.value]
    assert failed["completed"] is False
    assert failed["stop_reason"] == "rag_error"
    assert failed["feedback"] is None
    assert failed["rag_calls"] == 1
    assert failed["usage"]["input_tokens"] == 3 + (0 if failed_action is Action.BASE else 5)
    assert failed["usage"]["output_tokens"] == 2 + (0 if failed_action is Action.BASE else 2)
    failure_event = next(
        arm for arm in result.arms if arm.planned_action is failed_action
    ).result.events[-1]
    assert failure_event.request_id == "mock-failed-request"
    assert failure_event.audit_path == "not-written-mock-audit"
    for action in Action:
        assert len(factory.bundles[action].backend.requests) == 1
        if action is not failed_action:
            assert report["arms"][action.value]["completed"] is True


def test_rewrite_error_keeps_known_cost_without_running_its_backend():
    factory = Factory(fail_generator=Action.REUSE)
    report = comparison_report(run(factory), gold=GOLD)
    failed = report["arms"]["REUSE"]
    assert failed["usage"] == {"input_tokens": 4, "output_tokens": 1, "api_requests": 0}
    assert failed["stop_reason"] == "rewrite_error"
    assert failed["rewrite_calls"] == 1 and failed["rag_calls"] == 0
    assert factory.bundles[Action.REUSE].backend.requests == []
    assert report["arms"]["BASE"]["completed"] is True
    assert report["arms"]["FRESH"]["completed"] is True
    assert report["contrasts"]["REUSE_minus_FRESH"]["available"] is False


def test_rewrite_returning_original_query_executes_base_but_keeps_planned_arm_and_cost():
    report = comparison_report(run(Factory(unchanged=True)), gold=GOLD)
    for action in (Action.FRESH, Action.REUSE):
        row = report["arms"][action.value]
        assert row["planned_action"] == action.value
        assert row["executed_action"] == "BASE"
        assert row["query"] == QUESTION.text
        assert row["rewrite_calls"] == 1
        assert row["usage"] == {"input_tokens": 16, "output_tokens": 9, "api_requests": 0}


def test_gold_is_only_post_run_evaluation_and_does_not_change_recorded_execution():
    factory = Factory()
    result = run(factory)
    trace_before, record_before = list(factory.trace), result.to_dict()
    unknown = comparison_report(result)
    correct = comparison_report(result, gold=GOLD)
    different = comparison_report(result, gold=replace(GOLD, answers=("different",)))
    assert unknown["gold_available"] is False
    assert all(row["feedback"] is None for row in unknown["arms"].values())
    assert correct["arms"]["BASE"]["feedback"]["answer_em"] == 1
    assert different["arms"]["BASE"]["feedback"]["answer_em"] == 0
    assert result.to_dict() == record_before
    assert factory.trace == trace_before
    assert "gold" not in result.to_dict()["spec"]
    with pytest.raises(ValueError, match="belong"):
        comparison_report(result, gold=replace(GOLD, question_id="another-target"))
    with pytest.raises(TypeError, match="gold-free"):
        make_spec(question=GOLD)


def test_unexposed_evidence_is_unknown_not_zero_support_but_answer_metric_is_available():
    report = comparison_report(run(Factory(evidence=None)), gold=GOLD)
    for row in report["arms"].values():
        assert row["completed"] is True
        assert row["stop_reason"] == "evidence_unavailable"
        assert row["evidence"] is None
        feedback = row["feedback"]
        assert feedback["answer_em"] == 1
        for field in (
            "retrieved_gold_support_recall",
            "new_gold_support",
            "cited_gold_support_precision",
            "cited_gold_support_recall",
            "citation_ids_resolve",
            "answer_supported",
        ):
            assert feedback[field] is None
    assert report["contrasts"]["REUSE_minus_FRESH"]["support_recall_delta"] is None
    assert report["contrasts"]["REUSE_minus_FRESH"]["answer_em_delta"] == 0


def test_exposed_empty_evidence_is_zero_recall_not_unavailable():
    report = comparison_report(run(Factory(evidence=())), gold=GOLD)
    assert report["arms"]["BASE"]["evidence"] == []
    assert report["arms"]["BASE"]["feedback"]["retrieved_gold_support_recall"] == 0


def test_component_details_are_not_double_counted_in_experiment_usage():
    component = CallEvent(
        operation="rag.answer",
        execution_kind=ExecutionKind.MOCK,
        status="ok",
        elapsed_seconds=0.001,
        usage=Usage(11, 7, 0),
        transport_source="mock",
    )
    result = run(Factory(components=(component,)))
    assert all(arm.result.component_events == (component,) for arm in result.arms)
    report = comparison_report(result)
    assert report["total_experiment_usage"] == {
        "input_tokens": 3 * 11 + 2 * 5,
        "output_tokens": 3 * 7 + 2 * 2,
        "api_requests": 0,
    }
    assert report["arms"]["BASE"]["usage"]["input_tokens"] == 11
    assert report["arms"]["REUSE"]["usage"]["input_tokens"] == 16


def test_unknown_usage_does_not_silently_become_zero():
    result = run(Factory(api_adapter=True))
    usage = comparison_report(result)["total_experiment_usage"]
    assert usage["input_tokens"] is None
    assert usage["output_tokens"] is None
    assert usage["api_requests"] == 0


def test_save_writes_explicit_mock_artifacts_and_refuses_overwrite(tmp_path):
    result = run()
    directory = save_comparison(result, tmp_path / "comparison", gold=GOLD)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert set(before) == {"run.json", "report.json", "report.md"}
    saved = json.loads(before["run.json"])
    assert saved["synthetic_demo"] is True
    assert saved["execution_kind"] == "mock"
    assert saved["spec_fingerprint"] == result.spec.fingerprint
    assert json.loads(before["report.json"])["gold_available"] is True
    with pytest.raises(FileExistsError):
        save_comparison(result, directory, gold=GOLD)
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_invalid_gold_does_not_create_partial_output_directory(tmp_path):
    directory = tmp_path / "not-created"
    with pytest.raises(ValueError):
        save_comparison(run(), directory, gold=replace(GOLD, question_id="wrong"))
    assert not directory.exists()


def test_human_report_marks_demo_and_escapes_model_markup():
    from growrag.experiments.comparison_report import render_comparison_markdown

    result = run()
    report = comparison_report(result, gold=GOLD)
    report["arms"]["REUSE"]["answer"] = "<script>not markup</script>|field![image](https://x)"
    markdown = render_comparison_markdown(result, report)
    assert "手写测试替身" in markdown
    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown
    assert "&#124;field" in markdown
    assert "![image](https://x)" not in markdown


@pytest.mark.parametrize("action", [Action.FRESH, Action.REUSE])
@pytest.mark.parametrize("component", ["generator", "backend"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_unexpected_component_error_is_isolated_sanitized_and_cost_remains_unknown(
    tmp_path, action, component, error_type
):
    marker = "PRIVATE-EXCEPTION-BODY-MUST-NOT-BE-SAVED"
    factory = Factory()

    def broken_components(current_action):
        bundle = factory(current_action)
        if current_action is action:
            if component == "generator":

                def broken_generate(question, decision, **kwargs):
                    bundle.generator.calls.append((question, decision, kwargs))
                    factory.trace.append(("rewrite", current_action))
                    raise error_type(marker)

                bundle.generator.generate = broken_generate
            else:

                def broken_backend(request):
                    bundle.backend.requests.append(request)
                    factory.trace.append(("rag", current_action))
                    raise error_type(marker)

                bundle.backend.run = broken_backend
        return bundle

    result = run(broken_components)
    report = comparison_report(result, gold=GOLD)
    assert len(result.arms) == 3
    assert set(report["arms"]) == {item.value for item in Action}
    failed = report["arms"][action.value]
    assert failed["completed"] is False
    assert failed["feedback"] is None
    assert failed["stop_reason"] == ("rewrite_error" if component == "generator" else "rag_error")
    assert failed["usage"] == {
        "input_tokens": None,
        "output_tokens": None,
        "api_requests": None,
    }
    arm = next(item for item in result.arms if item.planned_action is action)
    assert arm.result.events[-1].status == "error"
    assert arm.result.events[-1].usage == Usage()
    if component == "backend":
        assert arm.result.events[0].operation == "rewrite"
        assert arm.result.events[0].status == "ok"
        assert arm.result.events[0].usage == Usage(5, 2, 0)
        assert failed["rag_calls"] == failed["rewrite_calls"] == 1
    else:
        assert failed["rag_calls"] == 0 and failed["rewrite_calls"] == 1
        assert factory.bundles[action].backend.requests == []
    for other in Action:
        if other is not action:
            assert report["arms"][other.value]["completed"] is True
            assert len(factory.bundles[other].backend.requests) == 1
    # Completed prior calls are retained in events; an unknown call prevents a
    # falsely precise aggregate even though this test itself uses no real API.
    assert report["total_experiment_usage"] == {
        "input_tokens": None,
        "output_tokens": None,
        "api_requests": None,
    }
    directory = save_comparison(result, tmp_path / "isolated-error", gold=GOLD)
    for path in directory.iterdir():
        assert marker not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("audited_envelope", [False, True])
def test_invalid_backend_return_is_isolated_and_preserves_only_known_usage(
    tmp_path, audited_envelope
):
    marker = "INVALID-RETURN-CONTENT-MUST-NOT-BE-SERIALIZED"
    factory = Factory()

    def invalid_reply_factory(action):
        bundle = factory(action)
        if action is Action.REUSE:

            def invalid_backend(request):
                bundle.backend.requests.append(request)
                invalid_value = {"not_a_rag_reply": marker}
                if not audited_envelope:
                    return invalid_value
                return CallResult(
                    invalid_value,
                    usage=Usage(13, 9, 0),
                    request_id="known-invalid-response-request",
                    transport_source="mock",
                )

            bundle.backend.run = invalid_backend
        return bundle

    result = run(invalid_reply_factory)
    report = comparison_report(result, gold=GOLD)
    failed = report["arms"]["REUSE"]
    assert failed["completed"] is False
    assert failed["stop_reason"] == "rag_error"
    assert failed["feedback"] is None
    assert failed["answer"] is None
    arm = next(item for item in result.arms if item.planned_action is Action.REUSE)
    assert arm.result.events[0].usage == Usage(5, 2, 0)
    expected_error_usage = Usage(13, 9, 0) if audited_envelope else Usage()
    assert arm.result.events[-1].status == "error"
    assert arm.result.events[-1].usage == expected_error_usage
    if audited_envelope:
        assert arm.result.events[-1].request_id == "known-invalid-response-request"
        assert failed["usage"] == {"input_tokens": 18, "output_tokens": 11, "api_requests": 0}
    else:
        assert failed["usage"] == {
            "input_tokens": None,
            "output_tokens": None,
            "api_requests": None,
        }
    assert report["arms"]["BASE"]["completed"] is True
    assert report["arms"]["FRESH"]["completed"] is True
    directory = save_comparison(result, tmp_path / "invalid-response", gold=GOLD)
    for path in directory.iterdir():
        assert marker not in path.read_text(encoding="utf-8")
