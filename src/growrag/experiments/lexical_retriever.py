"""Deterministic local BM25 over an explicitly supplied sentence collection.

This is a small HotpotQA distractor-diagnostic component, NOT a full-Wikipedia
retrieval system or a reproduction of any upstream paper. Its input is only
Evidence; no answers or supporting-fact annotations enter ranking.

Each document is ``title + sentence``. Unicode alphanumeric tokens are
case-folded, without stemming or stop-word removal. Distinct query terms each
contribute once, using positive Robertson/Lucene-style IDF::

    log(1 + (N - df + 0.5) / (df + 0.5))

Only documents with strictly positive scores are returned; zero-score items
never pad top_k. Ties are broken by evidence_id, independent of input order.
REAL means actual local computation, not a model API call or validated quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import ClassVar

from .protocol import CallResult, Evidence, ExecutionKind, Usage

_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_ALGORITHM_VERSION = "growrag-title-sentence-bm25-v1"


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class BM25SentenceRetriever:
    """An immutable, gold-free sentence index for bounded diagnostics.

    ``corpus_fingerprint`` identifies canonical Evidence contents only.
    ``context_fingerprint`` also covers ranking and tokenization configuration;
    use it for DecisionState so parameter changes do not masquerade as the same
    retrieval environment. An empty or entirely tokenless corpus is invalid.
    """

    corpus: tuple[Evidence, ...]
    k1: float = field(default=1.2, kw_only=True)
    b: float = field(default=0.75, kw_only=True)
    execution_kind: ClassVar[ExecutionKind] = ExecutionKind.REAL
    corpus_fingerprint: str = field(init=False)
    context_fingerprint: str = field(init=False)
    _term_counts: tuple[Mapping[str, int], ...] = field(init=False, repr=False)
    _document_frequency: Mapping[str, int] = field(init=False, repr=False)
    _lengths: tuple[int, ...] = field(init=False, repr=False)
    _average_length: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, tuple):
            raise TypeError("corpus must be an immutable tuple of Evidence")
        if not self.corpus:
            raise ValueError("corpus cannot be empty")
        if not all(isinstance(item, Evidence) for item in self.corpus):
            raise TypeError("corpus must contain Evidence objects")
        if len({item.evidence_id for item in self.corpus}) != len(self.corpus):
            raise ValueError("corpus evidence IDs must be unique, including identical duplicates")

        k1 = _finite_number(self.k1, "k1")
        b = _finite_number(self.b, "b")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        canonical = tuple(sorted(self.corpus, key=lambda item: item.evidence_id))
        counts = tuple(Counter(_tokens(f"{item.title} {item.text}")) for item in canonical)
        lengths = tuple(sum(count.values()) for count in counts)
        if not sum(lengths):
            raise ValueError("corpus must contain at least one alphanumeric token")
        frequencies: Counter[str] = Counter()
        for count in counts:
            frequencies.update(count.keys())

        corpus_fingerprint = _digest([asdict(item) for item in canonical])
        context_fingerprint = _digest(
            {
                "corpus": corpus_fingerprint,
                "algorithm": _ALGORITHM_VERSION,
                "k1": k1,
                "b": b,
                "query_terms": "distinct",
                "zero_score": "exclude",
                "tie_break": "evidence_id",
            }
        )
        object.__setattr__(self, "corpus", canonical)
        object.__setattr__(self, "k1", k1)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "corpus_fingerprint", corpus_fingerprint)
        object.__setattr__(self, "context_fingerprint", context_fingerprint)
        object.__setattr__(self, "_term_counts", tuple(MappingProxyType(c) for c in counts))
        object.__setattr__(self, "_document_frequency", MappingProxyType(frequencies))
        object.__setattr__(self, "_lengths", lengths)
        object.__setattr__(self, "_average_length", sum(lengths) / len(lengths))

    def retrieve(self, query: str, *, top_k: int) -> CallResult[tuple[Evidence, ...]]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        terms = tuple(sorted(set(_tokens(query))))
        if not terms:
            raise ValueError("query must contain at least one alphanumeric token")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        ranked: list[tuple[float, Evidence]] = []
        total = len(self.corpus)
        for item, counts, length in zip(self.corpus, self._term_counts, self._lengths, strict=True):
            score = 0.0
            normalization = self.k1 * (1 - self.b + self.b * length / self._average_length)
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                df = self._document_frequency[term]
                idf = math.log1p((total - df + 0.5) / (df + 0.5))
                score += idf * (tf * (self.k1 + 1) / (tf + normalization))
            if score > 0:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].evidence_id))
        return CallResult(
            value=tuple(item for _, item in ranked[:top_k]),
            usage=Usage(api_requests=0),
            provider="local",
            model=_ALGORITHM_VERSION,
            transport_source="local_compute",
        )
