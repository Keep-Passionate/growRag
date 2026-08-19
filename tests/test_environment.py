from __future__ import annotations

from dataclasses import replace

import pytest

from growrag.models import EnvironmentFingerprint
from growrag.selection import (
    EnvironmentCompatibilityMode,
    EnvironmentCompatibilityPolicy,
    match_environments,
)

REFERENCE = EnvironmentFingerprint(
    corpus_id="wiki",
    corpus_version="2026-08",
    retriever_id="bm25",
    retriever_version="1.0",
    rewriter_id="source-rewriter",
    rewriter_version="1",
    index_id="wiki-index",
    index_version="sha256:abc",
    analyzer_id="unicode-words",
    analyzer_version="1",
)


def test_strict_mode_requires_every_retrieval_field_to_match() -> None:
    current = replace(REFERENCE, retriever_version="1.1")

    report = match_environments(REFERENCE, current)

    assert report.mode is EnvironmentCompatibilityMode.STRICT
    assert report.compatible is False
    assert report.score == 0.0
    assert report.hard_mismatches == ("retriever_version",)
    assert report.soft_mismatches == ()


def test_strict_mode_ignores_rewriter_provenance_fields() -> None:
    current = replace(
        REFERENCE,
        rewriter_id="target-applier",
        rewriter_version="different",
    )

    report = match_environments(REFERENCE, current)

    assert report.compatible is True
    assert report.score == 1.0
    assert report.hard_mismatches == ()


def test_tiered_mode_turns_soft_differences_into_penalties_and_warnings() -> None:
    policy = EnvironmentCompatibilityPolicy(
        mode=EnvironmentCompatibilityMode.TIERED,
        minimum_score=0.70,
        soft_mismatch_penalty=0.20,
    )
    current = replace(REFERENCE, retriever_version="1.1")

    report = policy.evaluate(REFERENCE, current)

    assert report.compatible is True
    assert report.score == pytest.approx(0.80)
    assert report.hard_mismatches == ()
    assert report.soft_mismatches == ("retriever_version",)


def test_tiered_mode_never_relaxes_corpus_index_or_retriever_identity() -> None:
    policy = EnvironmentCompatibilityPolicy(mode="TIERED", minimum_score=0.0)
    current = replace(REFERENCE, index_version="sha256:different")

    report = policy.evaluate(REFERENCE, current)

    assert report.mode is EnvironmentCompatibilityMode.TIERED
    assert report.compatible is False
    assert report.score == 0.0
    assert report.hard_mismatches == ("index_version",)
    assert report.soft_mismatches == ()


@pytest.mark.parametrize("unknown_value", ["", "UNSET", "unknown", "unspecified"])
def test_unknown_values_fail_closed_even_for_normally_soft_fields(
    unknown_value: str,
) -> None:
    policy = EnvironmentCompatibilityPolicy(mode="tiered", minimum_score=0.0)
    current = replace(REFERENCE, analyzer_version=unknown_value)

    report = policy.evaluate(REFERENCE, current)

    assert report.compatible is False
    assert report.score == 0.0
    assert report.hard_mismatches == ("analyzer_version",)
    assert report.soft_mismatches == ()


def test_tiered_score_threshold_controls_whether_soft_drift_is_accepted() -> None:
    current = replace(
        REFERENCE,
        retriever_version="1.1",
        analyzer_version="2",
    )
    rejected = EnvironmentCompatibilityPolicy(
        mode="tiered",
        minimum_score=0.70,
        soft_mismatch_penalty=0.20,
    ).evaluate(REFERENCE, current)
    accepted = EnvironmentCompatibilityPolicy(
        mode="tiered",
        minimum_score=0.60,
        soft_mismatch_penalty=0.20,
    ).evaluate(REFERENCE, current)

    assert rejected.score == pytest.approx(0.60)
    assert rejected.soft_mismatches == ("retriever_version", "analyzer_version")
    assert rejected.compatible is False
    assert accepted.compatible is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_score", -0.1),
        ("minimum_score", 1.1),
        ("soft_mismatch_penalty", -0.1),
        ("soft_mismatch_penalty", 1.1),
    ],
)
def test_policy_rejects_invalid_scores(field: str, value: float) -> None:
    arguments = {field: value}

    with pytest.raises(ValueError, match=field):
        EnvironmentCompatibilityPolicy(**arguments)
