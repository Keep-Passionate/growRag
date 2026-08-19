from growrag.models import EnvironmentFingerprint, QueryTransformation
from growrag.selection.candidates import LexicalCandidateRecallScorer, rank_experiences

ENVIRONMENT = EnvironmentFingerprint(
    corpus_id="wiki",
    corpus_version="1",
    retriever_id="bm25",
    retriever_version="1",
    rewriter_id="fixture",
    rewriter_version="1",
)


def experience(experience_id: str, source_query: str) -> QueryTransformation:
    return QueryTransformation(
        experience_id=experience_id,
        source_query_id=f"source-{experience_id}",
        source_query=source_query,
        transformed_query=f"transformed {source_query}",
        atomic_units=("fixture",),
        environment=ENVIRONMENT,
        provenance="unit-test",
    )


def test_rank_experiences_honors_fixed_candidate_budget() -> None:
    ranked = rank_experiences(
        "alpha beta",
        [
            experience("far", "unrelated terms"),
            experience("near", "alpha beta gamma"),
            experience("exact", "alpha beta"),
        ],
        scorer=LexicalCandidateRecallScorer(),
        max_candidates=2,
    )
    assert [item.experience.experience_id for item in ranked] == ["exact", "near"]
    assert [item.rank for item in ranked] == [1, 2]


def test_rank_experiences_rejects_invalid_budget() -> None:
    try:
        rank_experiences(
            "query",
            [],
            scorer=LexicalCandidateRecallScorer(),
            max_candidates=0,
        )
    except ValueError as exc:
        assert "max_candidates" in str(exc)
    else:
        raise AssertionError("invalid budget should fail")
