"""Method-neutral records used by GrowRAG experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Action(StrEnum):
    """Mutually exclusive actions evaluated for one target query."""

    DIRECT = "direct"
    REUSE = "reuse"
    FRESH = "fresh"


@dataclass(frozen=True, slots=True)
class EnvironmentFingerprint:
    """The environment in which a transformation was validated."""

    corpus_id: str
    corpus_version: str
    retriever_id: str
    retriever_version: str
    rewriter_id: str
    rewriter_version: str


@dataclass(frozen=True, slots=True)
class QueryTransformation:
    """A concrete, provenance-carrying query transformation.

    This record deliberately stores no global trust score. Historical reliability
    and current-query applicability are separate experimental quantities.
    """

    experience_id: str
    source_query_id: str
    source_query: str
    transformed_query: str
    atomic_units: tuple[str, ...]
    environment: EnvironmentFingerprint
    provenance: str

