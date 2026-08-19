from __future__ import annotations

import pytest

from growrag.models import EnvironmentFingerprint, QueryTransformation
from growrag.rewrite import ApplicationError, PrecomputedPlanApplier


def experience() -> QueryTransformation:
    return QueryTransformation(
        experience_id="exp-1",
        source_query_id="source-1",
        source_query="Who was born first, A or B?",
        transformed_query="A birth date; B birth date",
        atomic_units=("decompose comparison into two attributes",),
        environment=EnvironmentFingerprint(
            corpus_id="wiki",
            corpus_version="1",
            retriever_id="bm25",
            retriever_version="1",
            rewriter_id="fixture",
            rewriter_version="1",
            index_id="wiki-bm25",
            index_version="1",
            analyzer_id="lucene-en",
            analyzer_version="1",
        ),
        provenance="unit-test",
    )


def test_precomputed_applier_builds_target_specific_plan() -> None:
    applier = PrecomputedPlanApplier(
        plans={
            ("target-1", "exp-1"): "C birth date; D birth date",
        }
    )

    prepared = applier.prepare(
        target_query_id="target-1",
        target_query="Who was born first, C or D?",
        experience=experience(),
    )

    assert prepared.transformed_query == "C birth date; D birth date"
    assert prepared.transformed_query != experience().transformed_query
    assert prepared.experience_id == "exp-1"


def test_precomputed_applier_fails_closed_when_plan_is_missing() -> None:
    with pytest.raises(ApplicationError, match="missing precomputed plan"):
        PrecomputedPlanApplier(plans={}).prepare(
            target_query_id="target-1",
            target_query="question",
            experience=experience(),
        )


def test_source_episode_cannot_count_as_target_application() -> None:
    with pytest.raises(ApplicationError, match="source query"):
        PrecomputedPlanApplier(plans={}).prepare(
            target_query_id="source-1",
            target_query="question",
            experience=experience(),
        )
