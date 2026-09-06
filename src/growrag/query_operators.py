"""Small immutable action bodies, independent of RAG and model transports."""

from dataclasses import dataclass
from enum import StrEnum


class RewriteForm(StrEnum):
    KEEP = "keep"
    PARAPHRASE = "paraphrase"
    EXPAND = "expand"
    DECOMPOSE = "decompose"  # Reserved; no multi-query execution yet.


def _text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f"{field} must be nonempty text of at most 2000 characters")


def _terms(values: tuple[str, ...], field: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field} must be a nonempty immutable tuple")
    for value in values:
        _text(value, field)
    normalized = [" ".join(value.split()).casefold() for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} cannot contain duplicate entries")


@dataclass(frozen=True, slots=True)
class ParaphraseBody:
    """A wording transformation, not a requirement to discover missing facts."""

    rewrite_rule: str
    preserve: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.rewrite_rule, "rewrite_rule")
        _terms(self.preserve, "preserve")

    @property
    def form(self) -> RewriteForm:
        return RewriteForm.PARAPHRASE


@dataclass(frozen=True, slots=True)
class ExpansionBody:
    """Kinds of terms to add and the rule for grounding factual additions."""

    term_roles: tuple[str, ...]
    grounding_rule: str
    requires_current_evidence: bool = False

    def __post_init__(self) -> None:
        _terms(self.term_roles, "term_roles")
        _text(self.grounding_rule, "grounding_rule")
        if type(self.requires_current_evidence) is not bool:
            raise TypeError("requires_current_evidence must be bool")

    @property
    def form(self) -> RewriteForm:
        return RewriteForm.EXPAND


ActionBody = ParaphraseBody | ExpansionBody
