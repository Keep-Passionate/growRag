"""Offline API-envelope checks; never import or execute the upstream snapshot.

The author parser is intentionally permissive. This migration-only validation
boundary must distinguish malformed responses from genuinely wrong answers.
"""

import json

import pytest

from growrag.experiments.run_s2g_author_pilot import validate_author_response


def response(stage, content):
    return {"kind": "api_response", "stage": stage, "content": content}


def test_non_response_events_do_not_trigger_protocol_validation():
    validate_author_response({"kind": "api_request", "stage": "judge"})


@pytest.mark.parametrize(
    "payload",
    [
        {"sufficient": True, "gap_items": []},
        {
            "sufficient": False,
            "gap_items": [{"target": "an entity", "slot": "founding year"}],
        },
    ],
)
def test_well_formed_judge_response_is_accepted(payload):
    validate_author_response(response("judge", json.dumps(payload)))


@pytest.mark.parametrize(
    "payload",
    [
        {"sufficient": "false", "gap_items": []},
        {"sufficient": False},
        {"sufficient": False, "gap_items": "missing year"},
    ],
)
def test_malformed_judge_cannot_silently_become_a_valid_verdict(payload):
    with pytest.raises(ValueError):
        validate_author_response(response("judge", json.dumps(payload)))


@pytest.mark.parametrize("ids", [[], [1, 3, 6]])
def test_well_formed_extractor_response_is_accepted(ids):
    validate_author_response(response("extract", json.dumps({"evidence_global_ids": ids})))


@pytest.mark.parametrize("ids", [["1"], [True], 1])
def test_extractor_requires_a_list_of_actual_integers(ids):
    # In Python bool subclasses int; accepting True here would silently select ID 1.
    with pytest.raises(ValueError):
        validate_author_response(response("extract", json.dumps({"evidence_global_ids": ids})))


def test_author_answer_and_rationale_format_is_accepted():
    validate_author_response(
        response("answer", "Answer: Northbridge\nRationale: The evidence names this town.")
    )


@pytest.mark.parametrize(
    "content",
    [
        "Answer: Northbridge",
        "Rationale: The evidence names Northbridge.",
        "answer: Northbridge\nrationale: A source says so.",
        "Answer : Northbridge\nRationale : A source says so.",
        "Answer: \nRationale: No answer present.",
    ],
)
def test_missing_author_answer_fields_are_protocol_failures_not_em_zero(content):
    with pytest.raises(ValueError):
        validate_author_response(response("answer", content))
