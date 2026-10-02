"""Pure lexical candidate-ranking diagnostics; neither semantics nor QPP.

只比较固定已发布经验库的排序，不改卡片、动作、反馈或适用条件。legacy复刻
原JSON词集；action策略仅取原动作正文。零分仍返回，不能解释为适用或不适用。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from growrag.history_library import FrozenHistoryLibrary, HistoryCard, card_view

POLICIES = ("legacy_json_jaccard", "action_text_jaccard", "action_text_bm25")
BM25_K1 = 1.2
BM25_B = 0.75
MAX_LIMIT = 44
# Fixed function-word list, not learned from questions or effectiveness labels.
# 保留no/not/without等否定词；不添加领域词、同义词、词干化或实体知识。
ENGLISH_STOPWORDS = frozenset(
    "a an and are as at be been being but by can could did do does for from had has have "
    "he her him his how i in into is it its me my of on or our she should than that the "
    "their them there these they this those to was we were what when where which who "
    "whom why will with would you your".split()
)
_ACTION_TOKEN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def action_text(card: HistoryCard) -> str:
    """原动作文本；模板外壳的别名/通用描述不参与排序。

    模板无独立constraints字段，使用实际when条件及依赖名，不推测适用条件。
    规则保留原name/description/rule；示例、来源、身份与反馈均不进入正文。
    """
    if type(card) is not HistoryCard:
        raise TypeError("action_text requires a HistoryCard")
    if card.operator_spec is None:
        return "\n".join((card.name, card.description, card.transformation_rule))
    spec = card.operator_spec
    parts = [field.name for field in spec.gap_schema]
    for step in spec.steps:
        parts.append(step.template)
        parts.extend(f"{item.field} {item.value}" for item in step.when)
        parts.extend(step.requires_bindings)
    return "\n".join(parts)


def _legacy_terms(card):
    # Exact shortlist_cards lexical projection, including its original JSON keys.
    view = card_view(card, "rule")
    del view["card_id"], view["version"]
    if view["operator_spec"] is not None:
        del view["operator_spec"]["operator_id"], view["operator_spec"]["version"]
        for step in view["operator_spec"]["steps"]:
            del step["step_id"]
    return set(re.findall(r"\w+", json.dumps(view, ensure_ascii=False).casefold()))


def _action_tokens(text, *, remove_stopwords=False):
    tokens = _ACTION_TOKEN.findall(text.casefold())
    return tuple(
        token for token in tokens if not remove_stopwords or token not in ENGLISH_STOPWORDS
    )


def _jaccard(query, documents):
    return [len(query & terms) / max(1, len(query | terms)) for terms in documents]


def _bm25(query, documents):
    counts = [Counter(tokens) for tokens in documents]
    lengths = [sum(row.values()) for row in counts]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    if not average or not query:
        return [0.0] * len(documents)
    frequencies = Counter(term for row in counts for term in row)
    scores = []
    # Distinct sorted query terms keep scores deterministic; no query-TF boost.
    for row, length in zip(counts, lengths, strict=True):
        score = 0.0
        normalization = BM25_K1 * (1 - BM25_B + BM25_B * length / average)
        for term in sorted(query):
            tf = row.get(term, 0)
            if tf:
                df = frequencies[term]
                idf = math.log1p((len(documents) - df + 0.5) / (df + 0.5))
                score += idf * (tf * (BM25_K1 + 1) / (tf + normalization))
        scores.append(score)
    return scores


def rank_cards(
    question_text: str, library: FrozenHistoryLibrary, policy: str, limit: int = MAX_LIMIT
) -> tuple[dict, ...]:
    """返回{card_id, score}，固定card_id破同分；不排除零分候选。

    Jaccard action词集不去停用词；BM25同正文去固定英文停用词，因此两者不仅
    权重不同。空库/空查询合法。limit只裁剪最终排序，不改变BM25统计的全集。
    """
    if type(question_text) is not str:
        raise TypeError("question_text must be a string")
    if type(library) is not FrozenHistoryLibrary:
        raise TypeError("ranking requires a FrozenHistoryLibrary")
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("unknown registered lexical policy")
    if type(limit) is not int or not 1 <= limit <= MAX_LIMIT:
        raise ValueError("limit must be an integer within [1, 44]")
    cards = library.published_cards
    if policy == "legacy_json_jaccard":
        query = set(re.findall(r"\w+", question_text.casefold()))
        scores = _jaccard(query, [_legacy_terms(card) for card in cards])
    elif policy == "action_text_jaccard":
        query = set(_action_tokens(question_text))
        scores = _jaccard(query, [set(_action_tokens(action_text(card))) for card in cards])
    else:
        query = set(_action_tokens(question_text, remove_stopwords=True))
        documents = [_action_tokens(action_text(card), remove_stopwords=True) for card in cards]
        scores = _bm25(query, documents)
    ranked = [
        {"card_id": card.card_id, "score": score} for card, score in zip(cards, scores, strict=True)
    ]
    return tuple(sorted(ranked, key=lambda row: (-row["score"], row["card_id"]))[:limit])
