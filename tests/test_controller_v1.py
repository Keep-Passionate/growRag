"""Mock-only contracts: these tests do not establish judge or routing quality."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.controller import (
    ASSESS_PARSER_VERSION,
    APIEvidenceAssessor,
    APIGapQueryGenerator,
    APIRoutingPolicy,
    EvidenceAssessment,
    GapAppendQueryGenerator,
    Requirement,
)
from growrag.experience.cards import ActivationStage
from growrag.experience.query_views import CardMemoryView
from growrag.experiments.api_client import APIRequestError, ChatResponse
from growrag.experiments.data_protocol import normalize_question
from growrag.experiments.protocol import (
    Action,
    Answer,
    BackendCallError,
    Evidence,
    GoldRecord,
    RuntimeQuestion,
)
from growrag.outer_loop import Feedback, LoopState, RagReply, RoundRecord
from growrag.query_actions import RewriteDecision, RewriteForm

Q = RuntimeQuestion("target", "Which university employed the inventor of Widget A?")
E = Evidence("e1", "Widget A", 0, "Ada invented Widget A.")
E2 = Evidence("e2", "Ada", 0, "Ada worked at College X.")


class Client:
    transport_source = "mock"
    config = SimpleNamespace(base_url="https://test.invalid/v1", model="TEST")

    def __init__(self, value):
        self.content = value if isinstance(value, str) else json.dumps(value)
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        return ChatResponse(
            self.content,
            "TEST",
            "TEST",
            "response-1",
            None,
            10,
            3,
            0,
            Path("test-not-written"),
            self.transport_source,
        )


def assessment(*, sufficient=False):
    return EvidenceAssessment(
        (Requirement("Identify inventor", "supported", ("e1",)),)
        if sufficient
        else (
            Requirement("Identify inventor", "supported", ("e1",)),
            Requirement("Identify employer", "missing", ()),
        ),
        sufficient,
        None,
        "" if sufficient else "Inventor employer missing",
        "" if sufficient else "Find the inventor's employer",
        "Evidence covers inventor",
    )


def post_state():
    reply = RagReply(Answer(""), (E,))
    return LoopState(Q, (RoundRecord(RewriteDecision(), Q.text, reply, Feedback(False)),), (E,))


def card(name="m", *, stage=ActivationStage.PRE_RETRIEVAL, source="source", text="Other question?"):
    payload = {
        "schema": "growrag-query-card-view-v1",
        "form": "paraphrase",
        "intent": "align wording",
        "conditions": {
            "stage": stage.value,
            "query_pattern": "university employed inventor",
            "gap_pattern": None,
            "preconditions": ["Explicit institution relation"],
            "contraindications": [],
        },
        "body": {"rewrite_rule": "Use employer relation", "preserve": ["entities", "objective"]},
        "success_criterion": "Retrieve evidence of employer",
    }
    return CardMemoryView(
        name,
        source,
        "source-step",
        "typed_card_diagnostic_v1",
        json.dumps(payload),
        (source,),
        (hashlib.sha256(normalize_question(text).encode()).hexdigest(),),
        RewriteForm.PARAPHRASE,
        "align wording",
        stage,
        False,
    )


def route(action="FRESH", memory_id=None, *, checks=None):
    return {
        "action": action,
        "memory_id": memory_id,
        "reason": "Preserve employer relation",
        "condition_checks": checks or [],
    }


def valid_checks(status="supported"):
    return [
        {
            "condition": value,
            "status": status,
            "critical": True,
            "reason": "Observed in the question",
        }
        for value in (
            "Query matches: university employed inventor",
            "Required: Explicit institution relation",
        )
    ]


def test_assessor_current_reader_context_not_accumulated_union():
    client = Client(asdict(assessment(sufficient=True)))
    judge = APIEvidenceAssessor(client)
    # e2 was seen previously but is NOT in this Reader's current context.
    result = judge(
        replace(post_state(), observed_evidence=(E, E2)), RagReply(Answer("Ada", ("e1",)), (E,))
    )
    payload = json.loads(client.requests[0][0][1]["content"])
    assert payload["reader_context"] == [asdict(E)]
    assert payload["previous_evidence_ids"] == ["e1", "e2"]
    assert result.value.sufficient is True
    assert judge.latest_question == Q
    assert judge.records[0]["metadata"]["usage"]["api_requests"] == 0


@pytest.mark.parametrize("bad_ids", [[], ["unavailable"], ["e1", "e1"], [0]])
def test_supported_requires_valid_unique_current_evidence_ids(bad_ids):
    raw = json.loads(json.dumps(asdict(assessment(sufficient=True))))
    raw["requirements"][0]["evidence_ids"] = bad_ids
    client = Client(raw)
    client.transport_source = "live_api"  # Fake transport only; no real network.
    judge = APIEvidenceAssessor(client)
    with pytest.raises(BackendCallError) as error:
        judge(LoopState(Q), RagReply(Answer("Ada", ("e1",)), (E,)))
    assert error.value.usage.api_requests == 1
    assert len(client.requests) == 1
    assert judge.latest is None
    assert judge.records[0]["status"] == "invalid_response"


def test_sufficiency_cannot_be_true_with_missing_requirements_or_empty_answer():
    with pytest.raises(ValueError):
        replace(assessment(), sufficient=True)
    judge = APIEvidenceAssessor(Client(asdict(assessment(sufficient=True))))
    with pytest.raises(BackendCallError):
        judge(LoopState(Q), RagReply(Answer(""), (E,)))


def test_seed_is_local_bound_to_question_and_records_are_independent():
    client = Client({})
    first, second = APIEvidenceAssessor(client), APIEvidenceAssessor(client)
    first.seed(Q, assessment())
    second.seed(Q, first.latest)
    assert second.latest is first.latest  # frozen value is safe to share
    first.records.append({"local": True})
    assert second.records == [] and not client.requests
    with pytest.raises(TypeError):
        second.seed(GoldRecord("q", ("gold",)), assessment())


def test_gold_change_does_not_change_pre_runtime_inputs():
    payloads = []
    for answer in ("gold-one", "gold-two"):
        gold = GoldRecord(Q.question_id, (answer,))
        client = Client(route())
        policy = APIRoutingPolicy(client, (), APIEvidenceAssessor(client))
        policy(LoopState(Q))
        payloads.append(json.loads(client.requests[0][0][1]["content"]))
        assert gold.answers[0] not in json.dumps(payloads[-1])
    assert payloads[0] == payloads[1]
    assert "current_evidence" not in payloads[0] and "assessment" not in payloads[0]


def test_candidate_memory_is_not_enabled_by_default():
    client = Client(route())
    policy = APIRoutingPolicy(client, (card(),), APIEvidenceAssessor(client))
    policy(LoopState(Q))
    assert policy.records[0]["candidate_ids"] == []


def test_recall_filters_same_id_same_text_and_post_stage_before_top_three():
    client = Client(route())
    candidates = (
        card("same-id", source=Q.question_id),
        card("same-text", text=Q.text),
        card("post", stage=ActivationStage.POST_RETRIEVAL),
        *(card(str(i)) for i in range(5)),
    )
    policy = APIRoutingPolicy(
        client, candidates, APIEvidenceAssessor(client), allow_candidate_memory=True
    )
    policy(LoopState(Q))
    assert policy.records[0]["candidate_ids"] == ["0", "1", "2"]


def test_pre_card_does_not_become_a_post_card_and_reflective_never_reads_memory():
    client = Client(route())
    judge = APIEvidenceAssessor(client)
    judge.seed(Q, assessment())
    adaptive = APIRoutingPolicy(client, (card(),), judge, allow_candidate_memory=True)
    assert adaptive(post_state()).action is Action.FRESH
    assert adaptive.records[0]["candidate_ids"] == []
    assert adaptive.records[0]["status"] == "local_rule" and not client.requests
    before = len(client.requests)
    reflective = APIRoutingPolicy(
        client, (card(),), judge, mode="reflective", allow_candidate_memory=True
    )
    assert reflective(LoopState(Q)).action is Action.BASE
    assert reflective(post_state()).intent == assessment().next_intent
    assert len(client.requests) == before
    assert all(row["candidate_ids"] == [] for row in reflective.records)


@pytest.mark.parametrize("status", ["unknown", "conflicted"])
def test_unknown_or_conflicting_critical_condition_rejects_reuse(status):
    client = Client(route("REUSE", "m", checks=valid_checks(status)))
    policy = APIRoutingPolicy(
        client, (card(),), APIEvidenceAssessor(client), allow_candidate_memory=True
    )
    assert policy(LoopState(Q)).value.action is Action.BASE
    assert policy.records[0]["fallback_reason"]


def test_reuse_needs_all_card_checks_not_just_a_freeform_rationale():
    for checks in ([], valid_checks()[:1]):
        client = Client(route("REUSE", "m", checks=checks))
        policy = APIRoutingPolicy(
            client, (card(),), APIEvidenceAssessor(client), allow_candidate_memory=True
        )
        assert policy(LoopState(Q)).value.action is Action.BASE
    client = Client(route("REUSE", "m", checks=valid_checks()))
    policy = APIRoutingPolicy(
        client, (card(),), APIEvidenceAssessor(client), allow_candidate_memory=True
    )
    assert policy(LoopState(Q)).value.action is Action.REUSE


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        '{"action":"BASE","action":"FRESH"}',
        route("REUSE", "nonexistent", checks=valid_checks()),
    ],
)
def test_invalid_route_does_not_retry_and_keeps_audit(raw):
    client = Client(raw)
    policy = APIRoutingPolicy(client, (), APIEvidenceAssessor(client))
    with pytest.raises(BackendCallError):
        policy(LoopState(Q))
    assert len(client.requests) == 1 and policy.records[0]["metadata"]


def test_base_returns_original_without_rewrite_call():
    client = Client({})
    generator = APIGapQueryGenerator(client, APIEvidenceAssessor(client))
    result = generator.generate(Q, RewriteDecision())
    assert result.value == Q.text and result.usage.api_requests == 0
    assert not client.requests


def test_pre_fresh_never_receives_memory_or_stale_previous_assessment():
    client = Client({"query": "Widget A inventor employer university"})
    judge = APIEvidenceAssessor(client)
    judge.seed(RuntimeQuestion("another", "Another question"), assessment())
    generator = APIGapQueryGenerator(client, judge)
    decision = RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "align words")
    generator.generate(Q, decision)
    payload = generator.records[0]["payload"]
    assert payload["stage"] == "PRE" and payload["gap"] is None
    assert payload["supported_current_evidence"] == [] and "canonical_body" not in payload
    with pytest.raises(ValueError):
        generator.generate(Q, decision, evidence=(E,))
    with pytest.raises(ValueError):
        generator.generate(Q, decision, evidence=(E,), previous_queries=(Q.text,))
    assert len(client.requests) == 1


def test_post_gap_binding_separates_supported_and_observed_current_evidence():
    client = Client({"query": "Ada employer university"})
    judge = APIEvidenceAssessor(client)
    judge.seed(Q, assessment())
    generator = APIGapQueryGenerator(client, judge)
    generator.generate(
        Q,
        RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "find employer"),
        evidence=(E, E2),
        previous_queries=(Q.text,),
    )
    payload = generator.records[0]["payload"]
    assert payload["supported_current_evidence"] == [asdict(E)]
    assert payload["observed_current_evidence"] == [asdict(E), asdict(E2)]
    assert payload["gap"] == assessment().gap and "canonical_body" not in payload


def test_reuse_execution_receives_only_canonical_body_not_condition_or_source():
    client = Client({"query": "alternative wording"})
    selected = card()
    generator = APIGapQueryGenerator(client, APIEvidenceAssessor(client))
    generator.generate(Q, RewriteDecision(Action.REUSE, selected.form, selected.intent, selected))
    payload = generator.records[0]["payload"]
    assert payload["canonical_body"] == json.loads(selected.text)["body"]
    assert "conditions" not in payload and selected.source_query_id not in json.dumps(payload)
    judge = generator.assessor
    judge.seed(Q, assessment())
    with pytest.raises(ValueError):
        generator.generate(
            Q,
            RewriteDecision(Action.REUSE, selected.form, selected.intent, selected),
            evidence=(E,),
            previous_queries=(Q.text,),
        )
    assert len(client.requests) == 1


def test_invalid_query_cost_is_not_erased_or_retried():
    client = Client({"query": "x", "extra": "not allowed"})
    client.transport_source = "live_api"  # Fixture only: complete never performs I/O.
    generator = APIGapQueryGenerator(client, APIEvidenceAssessor(client))
    with pytest.raises(BackendCallError) as error:
        generator.generate(Q, RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "align"))
    assert error.value.usage.api_requests == 1
    assert len(client.requests) == 1 and generator.records[0]["status"] == "invalid_response"


def test_pre_route_does_not_use_same_question_seed_from_an_earlier_post_branch():
    client = Client(route())
    judge = APIEvidenceAssessor(client)
    judge.seed(Q, assessment())
    policy = APIRoutingPolicy(client, (), judge)
    decision = policy(LoopState(Q)).value
    assert decision.intent != assessment().next_intent
    payload = policy.records[0]["payload"]
    assert "assessment" not in payload and "current_evidence" not in payload


def test_api_error_retains_unknown_cost_without_retry_or_old_assessment():
    class FailingClient(Client):
        transport_source = "live_api"

        def complete(self, messages, **kwargs):
            self.requests.append((messages, kwargs))
            raise APIRequestError("server failed", api_requests=1, request_id="failed-1")

    client = FailingClient({})
    judge = APIEvidenceAssessor(client)
    judge.seed(Q, assessment())
    with pytest.raises(BackendCallError) as error:
        judge(LoopState(Q), RagReply(Answer(""), (E,)))
    assert error.value.usage.api_requests == 1
    assert error.value.usage.input_tokens is None and judge.latest is None
    assert len(client.requests) == 1
    assert judge.records[0]["status"] == "transport_error"
    assert judge.records[0]["metadata"]["request_id"] == "failed-1"


def test_duplicate_condition_checks_are_rejected_not_overwritten():
    checks = valid_checks()
    client = Client(route("REUSE", "m", checks=[*checks, checks[0]]))
    policy = APIRoutingPolicy(
        client, (card(),), APIEvidenceAssessor(client), allow_candidate_memory=True
    )
    with pytest.raises(BackendCallError):
        policy(LoopState(Q))


@pytest.mark.parametrize(
    "patch", [{"sufficient": 1}, {"useful_gain": "true"}, {"requirements": []}, {"gap": 0}]
)
def test_assessment_fields_are_strict_not_coerced(patch):
    with pytest.raises((ValueError, TypeError)):
        replace(assessment(), **patch)


def test_repair_does_not_silently_drop_unavailable_supported_reference():
    client = Client({"query": "new query"})
    judge = APIEvidenceAssessor(client)
    judge.seed(Q, assessment())
    generator = APIGapQueryGenerator(client, judge)
    with pytest.raises(ValueError):
        generator.generate(
            Q,
            RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "find"),
            evidence=(E2,),
            previous_queries=(Q.text,),
        )
    assert not client.requests


def test_missing_requirement_retains_partial_entity_clue_without_gold():
    client = Client({"query": "Ada employer university"})
    judge = APIEvidenceAssessor(client)
    judge.seed(
        Q,
        EvidenceAssessment(
            (Requirement("Find inventor and employer", "missing", ("e1",)),),
            False,
            None,
            "Employer missing but inventor Ada is observed",
            "Find employer",
            "Partial evidence",
        ),
    )
    generator = APIGapQueryGenerator(client, judge)
    generator.generate(
        Q,
        RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "fill gap"),
        evidence=(E,),
        previous_queries=(Q.text,),
    )
    payload = generator.records[0]["payload"]
    assert payload["supported_current_evidence"] == []
    assert payload["observed_current_evidence"] == [asdict(E)]
    assert payload["assessment_requirements"][0]["status"] == "missing"
    assert "gold" not in json.dumps(payload) and "canonical_body" not in payload
    assert "partial clue" in generator.records[0]["prompt"]


def test_local_gap_append_has_zero_rewrite_requests_and_at_most_three_gaps():
    client = Client({})
    judge = APIEvidenceAssessor(client)
    judge.seed(
        Q,
        EvidenceAssessment(
            tuple(Requirement(f"Need relation {i}", "missing", ()) for i in range(4)),
            False,
            None,
            "Combined gap",
            "Find missing relations",
            "Missing information",
        ),
    )
    generator = GapAppendQueryGenerator(judge)
    result = generator.generate(
        Q,
        RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "fill gap"),
        evidence=(E,),
        previous_queries=(Q.text,),
    )
    assert result.value == Q.text + " Need relation 0 Need relation 1 Need relation 2"
    assert result.usage.api_requests == 0 and result.provider == "local"
    assert result.transport_source == "local_compute" and not client.requests
    assert generator.records[0]["payload"]["appended_requirements"] == [
        "Need relation 0",
        "Need relation 1",
        "Need relation 2",
    ]
    assert E.text == "Ada invented Widget A."


def test_local_gap_append_uses_gap_fallback_and_rejects_pre_base_or_reuse():
    client = Client({})
    judge = APIEvidenceAssessor(client)
    judge.seed(
        Q,
        replace(
            assessment(sufficient=True),
            sufficient=False,
            gap="Answer still lacks support",
            next_intent="Check answer",
        ),
    )
    generator = GapAppendQueryGenerator(judge)
    fresh = RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "check")
    result = generator.generate(Q, fresh, evidence=(E,), previous_queries=(Q.text,))
    assert result.value.endswith("Answer still lacks support")
    with pytest.raises(ValueError):
        generator.generate(Q, fresh)
    with pytest.raises(ValueError):
        generator.generate(Q, RewriteDecision(), evidence=(E,), previous_queries=(Q.text,))
    selected = card()
    with pytest.raises(ValueError):
        generator.generate(
            Q,
            RewriteDecision(Action.REUSE, selected.form, selected.intent, selected),
            evidence=(E,),
            previous_queries=(Q.text,),
        )
    assert not client.requests


def test_local_gap_append_overlength_is_explicit_not_silently_truncated():
    client = Client({})
    judge = APIEvidenceAssessor(client)
    long_question = RuntimeQuestion("long", "q" * 1990)
    judge.seed(long_question, assessment())
    generator = GapAppendQueryGenerator(judge)
    with pytest.raises(BackendCallError) as error:
        generator.generate(
            long_question,
            RewriteDecision(Action.FRESH, RewriteForm.PARAPHRASE, "fill"),
            evidence=(E,),
            previous_queries=(long_question.text,),
        )
    assert error.value.usage.api_requests == 0
    assert generator.records[0]["status"] == "length_guard"
    assert generator.records[0]["query"].startswith(long_question.text)
    assert len(generator.records[0]["query"]) > 2000 and not client.requests


@pytest.mark.parametrize(
    "status,expected",
    [("supported", Action.REUSE), ("conflicted", Action.BASE), ("unknown", Action.BASE)],
)
def test_contraindication_check_is_a_positive_claim_of_absence(status, expected):
    selected = card()
    raw = json.loads(selected.text)
    raw["conditions"]["contraindications"] = ["Question asks about birthplace"]
    selected = replace(selected, text=json.dumps(raw))
    check = {
        "condition": "Contraindication is absent: Question asks about birthplace",
        "status": status,
        "critical": True,
        "reason": "Observed scope is employment, not birthplace",
    }
    client = Client(route("REUSE", "m", checks=[*valid_checks(), check]))
    policy = APIRoutingPolicy(
        client, (selected,), APIEvidenceAssessor(client), allow_candidate_memory=True
    )
    assert policy(LoopState(Q)).value.action is expected
    assert check["condition"] in policy.records[0]["payload"]["candidates"][0]["required_checks"]
    assert policy.records[0]["prompt_version"] == "growrag-three-way-route-v2"
    assert "conflicted means X holds" in policy.records[0]["prompt"]


def test_missing_reason_uses_explicit_local_placeholder_without_another_call():
    raw = asdict(assessment())
    del raw["reason"]
    client = Client(raw)
    judge = APIEvidenceAssessor(client)
    result = judge(LoopState(Q), RagReply(Answer(""), (E,)))
    assert result.value.reason == "Explanation omitted by model; inspect requirements and gap."
    assert result.value.sufficient is False
    assert len(client.requests) == 1 and result.usage.api_requests == 0
    record = judge.records[0]
    assert record["parser_version"] == ASSESS_PARSER_VERSION
    assert record["normalization"] == {"reason_missing": True, "reason_origin": "local_placeholder"}
    assert "reason" not in json.loads(record["response"])


def test_missing_reason_cannot_relax_sufficiency_or_evidence_constraints():
    raw = asdict(assessment())
    del raw["reason"]
    raw["sufficient"] = True
    client = Client(raw)
    judge = APIEvidenceAssessor(client)
    with pytest.raises(BackendCallError):
        judge(LoopState(Q), RagReply(Answer("Ada", ("e1",)), (E,)))
    assert judge.latest is None and len(client.requests) == 1


@pytest.mark.parametrize("reason", [None, 7, True, [], {}, ""])
def test_present_but_invalid_reason_is_not_replaced(reason):
    raw = asdict(assessment())
    raw["reason"] = reason
    client = Client(raw)
    judge = APIEvidenceAssessor(client)
    with pytest.raises(BackendCallError):
        judge(LoopState(Q), RagReply(Answer(""), (E,)))
    assert "normalization" not in judge.records[0] and len(client.requests) == 1


@pytest.mark.parametrize(
    "field", ["requirements", "sufficient", "useful_gain", "gap", "next_intent"]
)
def test_only_reason_is_optional_all_core_fields_remain_required(field):
    raw = asdict(assessment())
    del raw[field]
    client = Client(raw)
    judge = APIEvidenceAssessor(client)
    with pytest.raises(BackendCallError):
        judge(LoopState(Q), RagReply(Answer(""), (E,)))
    assert len(client.requests) == 1
