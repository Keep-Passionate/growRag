import pytest

from growrag.models import (
    SUPPORTED_GAP_CATEGORIES,
    EnvironmentFingerprint,
    PreparedReuseCandidate,
    QueryTransformation,
)


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


def transformation(**overrides) -> QueryTransformation:
    values = {
        "experience_id": "exp-1",
        "source_query_id": "source-1",
        "source_query": "Who wrote the book?",
        "transformed_query": "book title author",
        "atomic_units": ("add authorship relation",),
        "environment": environment(),
        "provenance": "paired source evidence",
        "applicability_signature": ("authorship",),
    }
    values.update(overrides)
    return QueryTransformation(**values)


def test_source_experience_card_defaults_are_backward_compatible() -> None:
    card = transformation()

    assert card.diagnosed_failure == ""
    assert card.gap_categories == ()
    assert card.contraindication_signature == ()
    assert card.atomic_units == ("add authorship relation",)
    assert card.applicability_signature == ("authorship",)


def test_gap_and_contraindication_metadata_are_normalized() -> None:
    card = transformation(
        diagnosed_failure="  The bridge entity was absent.  ",
        gap_categories=("Bridge Entity", "Evidence-Span", "RELATION"),
        contraindication_signature=(
            "  Answer Already Explicit  ",
            "Temporal   Question",
        ),
    )

    assert card.diagnosed_failure == "The bridge entity was absent."
    assert card.gap_categories == ("bridge_entity", "evidence_span", "relation")
    assert card.contraindication_signature == (
        "answer already explicit",
        "temporal question",
    )
    assert SUPPORTED_GAP_CATEGORIES == {
        "bridge_entity",
        "attribute",
        "relation",
        "evidence_span",
        "other",
    }


def test_positive_applicability_signatures_use_the_same_normalization() -> None:
    card = transformation(applicability_signature=("  Two   Entity  ", "Comparison"))

    assert card.applicability_signature == ("two entity", "comparison")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gap_categories", ("unsupported",), "unsupported gap category"),
        ("gap_categories", ("relation", " RELATION "), "duplicate normalized gap"),
        ("gap_categories", (" ",), "must not be blank"),
        (
            "contraindication_signature",
            ("Temporal Question", " temporal   question "),
            "duplicate normalized contraindication_signature",
        ),
        ("contraindication_signature", ("",), "must not be blank"),
    ],
)
def test_source_experience_card_rejects_invalid_gap_metadata(
    field: str,
    value: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        transformation(**{field: value})
