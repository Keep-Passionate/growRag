import pytest

from growrag.models import EnvironmentFingerprint, PreparedReuseCandidate


def environment(
    *,
    corpus_version: str = "1",
    rewriter_id: str = "writer-a",
) -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        corpus_id="wiki",
        corpus_version=corpus_version,
        retriever_id="bm25",
        retriever_version="1",
        rewriter_id=rewriter_id,
        rewriter_version="1",
        index_id="wiki-bm25",
        index_version="1",
        analyzer_id="lucene-en",
        analyzer_version="1",
    )


def test_retrieval_compatibility_ignores_rewriter_provenance() -> None:
    assert environment(rewriter_id="writer-a").retrieval_compatible_with(
        environment(rewriter_id="writer-b")
    )


def test_retrieval_compatibility_rejects_corpus_version_change() -> None:
    assert not environment(corpus_version="1").retrieval_compatible_with(
        environment(corpus_version="2")
    )


def test_unknown_index_does_not_count_as_compatible() -> None:
    unknown = EnvironmentFingerprint(
        corpus_id="wiki",
        corpus_version="1",
        retriever_id="bm25",
        retriever_version="1",
        rewriter_id="writer-a",
        rewriter_version="1",
    )

    assert not unknown.retrieval_compatible_with(unknown)


def test_prepared_candidate_rejects_empty_target_variant() -> None:
    with pytest.raises(ValueError, match="transformed_query"):
        PreparedReuseCandidate(
            target_query_id="target-1",
            target_query="question",
            experience_id="exp-1",
            transformed_query=" ",
            generated_by="fixture",
        )
