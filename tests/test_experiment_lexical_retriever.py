"""Tests exercise ranking mechanics, not research effectiveness."""

from dataclasses import FrozenInstanceError

import pytest

from growrag.experiments.lexical_retriever import BM25SentenceRetriever
from growrag.experiments.protocol import Evidence, ExecutionKind


def sentence(identity: str, text: str, title: str = "Document") -> Evidence:
    return Evidence(identity, title, 0, text)


def ids(index: BM25SentenceRetriever, query: str, top_k: int = 10) -> tuple[str, ...]:
    return tuple(item.evidence_id for item in index.retrieve(query, top_k=top_k).value)


def test_manually_computable_equal_length_frequency_ranking() -> None:
    # With b=0, the document length disappears. For k1=1, the apple factor is
    # tf*2/(tf+1): tf=2 gives 4/3 and tf=1 gives 1. The shared positive IDF
    # preserves this ordering. This tests a known BM25 formula, not a model.
    index = BM25SentenceRetriever(
        (sentence("a", "apple pear"), sentence("b", "apple apple")), k1=1, b=0
    )
    assert ids(index, "apple") == ("b", "a")


def test_length_normalization_changes_ranking() -> None:
    short = sentence("z", "apple")
    long = sentence("a", "apple pear pear pear pear pear pear pear")
    assert ids(BM25SentenceRetriever((short, long), b=0), "apple") == ("a", "z")
    assert ids(BM25SentenceRetriever((short, long), b=1), "apple") == ("z", "a")


def test_query_change_ranks_different_real_local_evidence() -> None:
    index = BM25SentenceRetriever(
        (
            sentence("author", "Aster was written by Mira."),
            sentence("birth", "Mira was born in Lumen."),
            sentence("award", "Mira won the fictional prize."),
        )
    )
    assert ids(index, "Aster written", top_k=1) == ("author",)
    assert ids(index, "Mira born", top_k=1) == ("birth",)
    result = index.retrieve("Mira born", top_k=1)
    assert index.execution_kind is ExecutionKind.REAL
    assert result.transport_source == "local_compute"
    assert result.usage.api_requests == 0
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None


def test_title_is_part_of_searchable_document() -> None:
    index = BM25SentenceRetriever((sentence("a", "Born in Lumen.", title="Mira"),))
    assert ids(index, "mira") == ("a",)


def test_zero_scores_never_pad_results() -> None:
    index = BM25SentenceRetriever((sentence("a", "apple"), sentence("b", "pear")))
    assert ids(index, "apple", top_k=100) == ("a",)
    assert ids(index, "unseen", top_k=100) == ()


def test_ties_and_fingerprints_are_input_order_independent() -> None:
    a, b = sentence("a", "apple"), sentence("b", "apple")
    first, second = BM25SentenceRetriever((b, a)), BM25SentenceRetriever((a, b))
    assert ids(first, "apple") == ids(second, "apple") == ("a", "b")
    assert first.corpus_fingerprint == second.corpus_fingerprint
    assert first.context_fingerprint == second.context_fingerprint


def test_fingerprints_track_content_and_retrieval_configuration() -> None:
    original = BM25SentenceRetriever((sentence("a", "apple"),))
    changed = BM25SentenceRetriever((sentence("a", "pear"),))
    configured = BM25SentenceRetriever(original.corpus, k1=2)
    assert original.corpus_fingerprint != changed.corpus_fingerprint
    assert original.context_fingerprint != changed.context_fingerprint
    assert original.corpus_fingerprint == configured.corpus_fingerprint
    assert original.context_fingerprint != configured.context_fingerprint


def test_index_and_term_counts_cannot_be_mutated() -> None:
    index = BM25SentenceRetriever((sentence("a", "apple"),))
    with pytest.raises(FrozenInstanceError):
        index.b = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        index._term_counts[0]["apple"] = 42  # type: ignore[index]


def test_casefold_punctuation_and_repeated_terms_are_stable() -> None:
    index = BM25SentenceRetriever((sentence("a", "Äpfel, apple!"), sentence("b", "pear")))
    assert ids(index, "ÄPFEL") == ("a",)
    assert ids(index, "APPLE apple") == ids(index, "apple")


@pytest.mark.parametrize("bad", [(), ("not evidence",), [sentence("a", "apple")]])
def test_rejects_empty_or_mutable_or_malformed_corpus(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BM25SentenceRetriever(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("different", [False, True])
def test_duplicate_ids_rejected_even_when_content_identical(different: bool) -> None:
    first = sentence("a", "apple")
    second = sentence("a", "pear") if different else first
    with pytest.raises(ValueError, match="unique"):
        BM25SentenceRetriever((first, second))


def test_rejects_entirely_tokenless_corpus() -> None:
    with pytest.raises(ValueError, match="alphanumeric"):
        BM25SentenceRetriever((sentence("a", "...", title="!!!"),))


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan"), "1.2"])
def test_rejects_invalid_k1(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BM25SentenceRetriever((sentence("a", "apple"),), k1=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-0.1, 1.1, True, float("inf"), float("nan"), "0.75"])
def test_rejects_invalid_b(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BM25SentenceRetriever((sentence("a", "apple"),), b=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_rejects_invalid_top_k(value: object) -> None:
    index = BM25SentenceRetriever((sentence("a", "apple"),))
    with pytest.raises((TypeError, ValueError)):
        index.retrieve("apple", top_k=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["", " \n\t ", "!?!", None, 12])
def test_rejects_empty_tokenless_or_nonstring_query(value: object) -> None:
    index = BM25SentenceRetriever((sentence("a", "apple"),))
    with pytest.raises((TypeError, ValueError)):
        index.retrieve(value, top_k=1)  # type: ignore[arg-type]
