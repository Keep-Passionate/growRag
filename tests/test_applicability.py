from dataclasses import replace

from growrag.models import EnvironmentFingerprint, QueryTransformation
from growrag.selection.applicability import SignatureApplicabilityScorer


def make_experience(
    *,
    signature: tuple[str, ...] = ("comparison", "two_entity"),
) -> QueryTransformation:
    return QueryTransformation(
        experience_id="exp-1",
        source_query_id="source-1",
        source_query="which film director was born first alpha or beta",
        transformed_query="alpha birth date beta birth date",
        atomic_units=("split comparison into entity attributes",),
        environment=EnvironmentFingerprint(
            corpus_id="wiki",
            corpus_version="1",
            retriever_id="bm25",
            retriever_version="1",
            rewriter_id="fixture",
            rewriter_version="1",
        ),
        provenance="unit-test",
        transformation_type="comparison_decomposition",
        applicability_signature=signature,
    )


def test_signature_scorer_measures_declared_condition_coverage() -> None:
    estimate = SignatureApplicabilityScorer().score(
        "which actor was born first gamma or delta",
        make_experience(),
        current_signatures=frozenset({"comparison", "two_entity", "person"}),
    )
    assert estimate.score == 1.0
    assert estimate.matched_signatures == ("comparison", "two_entity")


def test_partial_signature_match_is_not_treated_as_full_fit() -> None:
    estimate = SignatureApplicabilityScorer().score(
        "when was gamma born",
        make_experience(),
        current_signatures=frozenset({"two_entity"}),
    )
    assert estimate.score == 0.5


def test_missing_signature_features_fail_closed() -> None:
    estimate = SignatureApplicabilityScorer().score(
        "question",
        make_experience(),
        current_signatures=frozenset(),
    )
    assert estimate.score == 0.0
    assert estimate.missing_features


def test_declared_contraindication_overrides_positive_signature_match() -> None:
    experience = replace(
        make_experience(),
        contraindication_signature=("answer_already_explicit",),
    )

    estimate = SignatureApplicabilityScorer().score(
        "which actor was born first gamma or delta",
        experience,
        current_signatures=frozenset({"comparison", "two_entity", "answer_already_explicit"}),
    )

    assert estimate.score == 0.0
    assert estimate.matched_signatures == ("comparison", "two_entity")
    assert estimate.matched_contraindications == ("answer_already_explicit",)
    assert not estimate.missing_features
