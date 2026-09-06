"""Local HotpotQA parsing and offline feedback, separate from runtime decisions.

Format: https://github.com/hotpotqa/hotpot#json-format
Metrics follow the public Hotpot answer-normalization/yes-no conventions. This
module is not a full replacement for the official benchmark evaluation script.
Context is the supplied candidate pool, NOT the complete fullwiki corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .protocol import Answer, Evidence, GoldRecord, RuntimeQuestion


@dataclass(frozen=True, slots=True)
class HotpotExample:
    question: RuntimeQuestion
    candidate_context: tuple[Evidence, ...]
    gold: GoldRecord | None
    # Analysis only: official hidden test does not expose these fields.
    question_type: str | None = None
    difficulty: str | None = None


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def parse_hotpot_example(record: dict, *, dataset: str) -> HotpotExample:
    """Separate query/text from gold; never infer a gap from supporting_facts."""
    if not isinstance(record, dict):
        raise ValueError("Hotpot example must be an object")
    question_id = _text(record.get("_id"), "_id")
    question = RuntimeQuestion(
        question_id=question_id,
        text=_text(record.get("question"), "question"),
        dataset=_text(dataset, "dataset"),
    )
    context = record.get("context")
    if not isinstance(context, list):
        raise ValueError("context must be a list")
    evidence = []
    titles = set()
    for paragraph in context:
        if not isinstance(paragraph, list) or len(paragraph) != 2:
            raise ValueError("each context paragraph must be [title, sentences]")
        title = _text(paragraph[0], "title")
        if title in titles:
            raise ValueError("duplicate paragraph title would make sentence references ambiguous")
        titles.add(title)
        if not isinstance(paragraph[1], list):
            raise ValueError("sentences must be a list")
        for index, sentence in enumerate(paragraph[1]):
            if not isinstance(sentence, str):
                raise ValueError("sentence must be text")
            # Keep original sentence positions, even when an empty sentence is omitted.
            if not sentence.strip():
                continue
            identity = json.dumps([title, index, sentence], ensure_ascii=False).encode("utf-8")
            evidence.append(Evidence(hashlib.sha256(identity).hexdigest(), title, index, sentence))
    gold = None
    if "answer" in record:
        answer = _text(record["answer"], "answer")
        supporting = record.get("supporting_facts")
        if not isinstance(supporting, list):
            raise ValueError("a labelled example must supply supporting_facts")
        facts = []
        for fact in supporting:
            if (
                not isinstance(fact, list)
                or len(fact) != 2
                or type(fact[1]) is not int
                or fact[1] < 0
            ):
                raise ValueError("supporting fact must be [title, nonnegative sentence index]")
            facts.append((_text(fact[0], "support title"), fact[1]))
        # Do NOT require gold support to be present in context: fullwiki may miss it.
        gold = GoldRecord(question_id, (answer,), tuple(dict.fromkeys(facts)))
    elif "supporting_facts" in record:
        raise ValueError("partly labelled examples are not supported")
    return HotpotExample(
        question,
        tuple(evidence),
        gold,
        record.get("type") if isinstance(record.get("type"), str) else None,
        record.get("level") if isinstance(record.get("level"), str) else None,
    )


def load_hotpot(path: Path, *, dataset: str) -> tuple[HotpotExample, ...]:
    """Read an explicitly chosen local file. No download, split, or label creation."""
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Hotpot top-level JSON must be a list")
    examples = tuple(parse_hotpot_example(record, dataset=dataset) for record in records)
    identifiers = [example.question.question_id for example in examples]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate question IDs")
    return examples


@dataclass(frozen=True, slots=True)
class LayeredFeedback:
    """Independent observations, not an automatic promotion or causality label."""

    question_id: str
    answer_em: float
    answer_f1: float
    retrieved_gold_support_recall: float | None
    new_gold_support: tuple[tuple[str, int], ...]
    cited_gold_support_precision: float | None
    cited_gold_support_recall: float | None
    citation_ids_resolve: bool
    # Resolvable citations and answer match do NOT establish textual entailment.
    answer_supported: bool | None = None
    feedback_origin: str = "offline_gold"


def _normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def _answer_f1(prediction: str, reference: str) -> float:
    predicted, expected = _normalize(prediction), _normalize(reference)
    if predicted != expected and {predicted, expected} & {"yes", "no", "noanswer"}:
        return 0.0
    p_tokens, g_tokens = predicted.split(), expected.split()
    overlap = sum((Counter(p_tokens) & Counter(g_tokens)).values())
    return 2 * overlap / (len(p_tokens) + len(g_tokens)) if overlap else 0.0


def evaluate_layered_feedback(
    question: RuntimeQuestion,
    before: tuple[Evidence, ...],
    after: tuple[Evidence, ...],
    answer: Answer,
    gold: GoldRecord,
) -> LayeredFeedback:
    """Evaluate AFTER branches execute; never feed this object to a selector.

    ``after`` is the complete active evidence available to this branch's reader.
    Missing annotated support is not proof all alternative evidence is invalid.
    """
    if question.question_id != gold.question_id:
        raise ValueError("gold belongs to a different question")
    if not gold.answers:
        raise ValueError("gold answers cannot be empty")
    seen: dict[str, Evidence] = {}
    for item in (*before, *after):
        if item.evidence_id in seen and seen[item.evidence_id] != item:
            raise ValueError("same evidence ID has inconsistent content")
        seen[item.evidence_id] = item
    before_refs = {(item.title, item.sentence_id) for item in before}
    after_refs = {(item.title, item.sentence_id) for item in after}
    annotated = set(gold.supporting_facts)
    by_id = {item.evidence_id: item for item in after}
    cited_ids = set(answer.cited_evidence_ids)
    cited_refs = {
        (by_id[identifier].title, by_id[identifier].sentence_id)
        for identifier in cited_ids
        if identifier in by_id
    }
    # Invalid IDs count against citation precision, rather than disappearing.
    unresolved = cited_ids - by_id.keys()
    cited_count = len(cited_refs) + len(unresolved)
    return LayeredFeedback(
        question_id=question.question_id,
        answer_em=float(any(_normalize(answer.text) == _normalize(ref) for ref in gold.answers)),
        answer_f1=max(_answer_f1(answer.text, ref) for ref in gold.answers),
        retrieved_gold_support_recall=(
            len(after_refs & annotated) / len(annotated) if annotated else None
        ),
        new_gold_support=tuple(sorted((after_refs - before_refs) & annotated)),
        cited_gold_support_precision=(
            len(cited_refs & annotated) / cited_count if cited_count else None
        ),
        cited_gold_support_recall=(
            len(cited_refs & annotated) / len(annotated) if annotated else None
        ),
        citation_ids_resolve=not unresolved,
    )
