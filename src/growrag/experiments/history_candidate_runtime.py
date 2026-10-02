"""A-stage candidate-only intervention; unchanged selection prompt and action executor.

不在这里加领域规则、生成新动作、训练选择器或改变模板约束。不同实验臂只
改变候选检索与数量；selected card仍通过原v2规划器解析和执行。
"""

from __future__ import annotations

from .history_rank_diagnostics import rank_cards
from .history_runtime import shortlist_cards
from .history_runtime_v2 import HistoryPlannerV2

METHODS = {
    "history_legacy3": ("legacy_json_jaccard", 3),
    "history_body3": ("action_text_bm25", 3),
    "history_body8": ("action_text_bm25", 8),
}


class CandidateHistoryPlanner(HistoryPlannerV2):
    def __init__(self, *args, method, **kwargs):
        if method not in METHODS:
            raise ValueError("unregistered candidate method")
        self.method = method
        super().__init__(*args, **kwargs)

    def candidate_ranking(self, state):
        policy, limit = METHODS[self.method]
        if self.method == "history_legacy3":
            return shortlist_cards(state.question, self.library)
        return tuple(
            {"card_id": row["card_id"], "lexical_score": row["score"]}
            for row in rank_cards(state.question.text, self.library, policy, limit)
        )


def planner_factory(method):
    """Per-instance constructor; no global function or schema monkey-patching."""
    if method not in METHODS:
        raise ValueError("unregistered candidate method")

    def construct(*args, **kwargs):
        return CandidateHistoryPlanner(*args, method=method, **kwargs)

    return construct
