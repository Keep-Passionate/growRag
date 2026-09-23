"""Offline contracts only; fabricated responses do not prove model quality."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.controller import APIEvidenceAssessor, EvidenceAssessment, Requirement
from growrag.experiments.api_client import ChatResponse
from growrag.experiments.judge_smoke import (
    SMOKE_VERSION,
    evaluate_case,
    input_fingerprint,
    judge_input,
    smoke_cases,
    smoke_manifest,
)
from growrag.experiments.protocol import Answer, Evidence


def expected_assessment(case):
    """A test double produced from labels; used ONLY in these no-network tests."""
    expected = case.expected
    requirements = (
        tuple(
            Requirement("Fixture-supported fact", "supported", (reference,))
            for reference in expected.supported_evidence_ids
        )
        if expected.sufficient
        else (Requirement("Fixture-unresolved fact", expected.unresolved_statuses[0], ()),)
    )
    return EvidenceAssessment(
        requirements,
        expected.sufficient,
        expected.useful_gain,
        "" if expected.sufficient else "Fixture missing or contradictory relation",
        "" if expected.sufficient else "Retrieve the requested relation",
        "Fabricated fixture result, not a real model judgement.",
    )


class MockClient:
    transport_source = "mock"
    config = SimpleNamespace(base_url="https://test.invalid/v1", model="MOCK")

    def __init__(self, assessment):
        self.assessment = assessment
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        return ChatResponse(
            json.dumps(asdict(self.assessment)),
            "MOCK",
            "MOCK",
            "fixture-response",
            None,
            10,
            10,
            0,
            Path("not-written-smoke-fixture"),
            "mock",
        )


def test_cases_are_eight_deterministic_immutable_invented_examples():
    cases = smoke_cases()
    assert len(cases) == 8
    assert cases == smoke_cases()
    assert len({case.case_id for case in cases}) == 8
    assert len({input_fingerprint(case) for case in cases}) == 8
    assert all(case.state.question.dataset == SMOKE_VERSION for case in cases)
    assert all(case.reply.evidence and case.reply.answer.text for case in cases)
    assert [bool(case.state.rounds) for case in cases] == [False] * 7 + [True]
    assert [case.expected.sufficient for case in cases] == [
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize("case", smoke_cases(), ids=lambda case: case.case_id)
def test_assessor_input_matches_fingerprint_payload_and_never_contains_labels(case):
    client = MockClient(expected_assessment(case))
    judge = APIEvidenceAssessor(client)
    judge(case.state, case.reply)
    actual = json.loads(client.requests[0][0][1]["content"])
    assert actual == json.loads(json.dumps(judge_input(case)))
    assert set(actual) == {
        "original_question",
        "reader_context",
        "current_answer",
        "previous_evidence_ids",
        "has_previous_round",
    }
    serialized = json.dumps(actual)
    assert "expected" not in actual and "important_requirements" not in actual
    assert case.case_id not in serialized
    assert case.purpose not in serialized
    assert evaluate_case(case, judge.latest)["semantic_pass"] is True


def test_label_or_description_edits_cannot_change_input_fingerprint():
    case = smoke_cases()[0]
    altered = replace(
        case,
        purpose="Different offline description",
        expected=replace(case.expected, sufficient=False, important_requirements=("Secret gold",)),
    )
    assert judge_input(altered) == judge_input(case)
    assert input_fingerprint(altered) == input_fingerprint(case)


def test_fingerprint_changes_with_actual_judge_observations():
    case = smoke_cases()[2]
    changed_answer = replace(case, reply=replace(case.reply, answer=Answer("Harbor Museum")))
    changed_order = replace(case, reply=replace(case.reply, evidence=case.reply.evidence[::-1]))
    evidence = case.reply.evidence[0]
    changed_text = replace(
        case,
        reply=replace(
            case.reply,
            evidence=(replace(evidence, text="Belltower opened in 1911."), case.reply.evidence[1]),
        ),
    )
    fingerprints = {
        input_fingerprint(item) for item in (case, changed_answer, changed_order, changed_text)
    }
    assert len(fingerprints) == 4


def test_gain_case_binds_real_prior_round_but_omits_prior_answer_and_feedback():
    case = smoke_cases()[-1]
    payload = judge_input(case)
    assert payload["previous_evidence_ids"] == ["nivo:0"]
    assert payload["has_previous_round"] is True
    assert len(case.state.rounds) == 1
    assert case.state.rounds[0].reply.evidence == case.state.observed_evidence
    assert "rounds" not in payload and "feedback" not in json.dumps(payload)
    altered = replace(case, state=replace(case.state, rounds=(), observed_evidence=()))
    assert input_fingerprint(case) != input_fingerprint(altered)


def test_evaluation_reports_wrong_sufficiency_without_modifying_response():
    case = smoke_cases()[0]
    value = replace(expected_assessment(case), sufficient=False, gap="No", next_intent="Check")
    before = asdict(value)
    report = evaluate_case(case, value)
    assert report["semantic_pass"] is False
    assert report["mismatches"] == ["sufficient_mismatch"]
    assert asdict(value) == before


def test_evaluation_rejects_unsupported_sources_even_if_sufficient_flag_matches():
    case = smoke_cases()[2]
    value = replace(
        expected_assessment(case),
        requirements=(Requirement("Only one value", "supported", ("bell:0",)),),
    )
    report = evaluate_case(case, value)
    assert report["semantic_pass"] is False
    assert report["missing_supported_evidence_ids"] == ["harbor:0"]


def test_evaluation_requires_explicit_conflict_for_conflict_case():
    case = smoke_cases()[4]
    value = replace(expected_assessment(case), requirements=(Requirement("Date", "missing", ()),))
    assert evaluate_case(case, value)["mismatches"] == ["expected_unresolved_requirement_missing"]


def test_evaluation_checks_gain_and_first_round_null():
    for case in (smoke_cases()[0], smoke_cases()[-1]):
        value = replace(expected_assessment(case), useful_gain=False)
        assert evaluate_case(case, value)["mismatches"] == ["useful_gain_mismatch"]


def test_manifest_is_deterministic_offline_metadata_with_explicit_scope():
    manifest = smoke_manifest()
    assert manifest == smoke_manifest()
    assert manifest["schema"] == SMOKE_VERSION
    assert len(manifest["cases"]) == 8
    assert manifest["cases"][0]["expected"]["sufficient"] is True
    assert "not full semantic certification" in manifest["scope"]
    json.dumps(manifest)


@pytest.mark.parametrize("value", [None, {}, "case", Evidence("e", "T", 0, "text")])
def test_helpers_reject_untyped_inputs(value):
    with pytest.raises(TypeError):
        judge_input(value)
    with pytest.raises(TypeError):
        evaluate_case(smoke_cases()[0], value)
