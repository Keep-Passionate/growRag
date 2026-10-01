"""合成示例：三种经验展示→同一模板执行；零API，不是HotpotQA实验成绩。"""

from __future__ import annotations

import json

from growrag.experiments.history_context import (
    plan_selected_template,
    prepare_history_context,
    resolve_history_selection,
)
from growrag.experiments.protocol import RuntimeQuestion
from growrag.history_library import (
    REPRESENTATIONS,
    FrozenHistoryLibrary,
    HistoryCard,
    HistoryCondition,
    HistoryExample,
    HistoryRecord,
)
from growrag.macro_operators import GapField, GoalContract, OperatorSpec, QueryStep, RuntimeState


def build_demo():
    question = RuntimeQuestion("synthetic-q", "Who acted in Cedar Lights?", "synthetic")
    spec = OperatorSpec(
        "CAST_LOOKUP",
        "1",
        ("lookup",),
        (GapField("film_title"),),
        (QueryStep("cast", "{film_title} cast"),),
    )
    card = HistoryCard(
        "CAST_CARD",
        "1",
        "查询演员表",
        "已知电影名，查询演员表；仅为手写合成示例。",
        "用当前电影名构造演员表查询，不携带历史电影事实。",
        (HistoryExample("Who acted in Pine Lights?", "Pine Lights cast"),),
        (HistoryCondition("needs_cast", "当前问题需要演员信息"),),
        spec,
    )
    library = FrozenHistoryLibrary(
        "synthetic-only",
        (),
        (HistoryRecord(card, "reference", "synthetic/manual", "0" * 64, status="published"),),
    )
    views = {}
    for representation in REPRESENTATIONS:
        context = prepare_history_context(
            library, question, offered_ids=(card.card_id,), representation=representation
        )
        selected = resolve_history_selection(
            '{"selected_card_id":"CAST_CARD","reason":"uncertain_match"}',
            context,
            library,
        )
        plan = plan_selected_template(
            library,
            context,
            selected.card_id,
            goal=GoalContract(question.text, "lookup"),
            gap={"film_title": "Cedar Lights"},
            state=RuntimeState(remaining_retrievals=1),
        )
        views[representation] = {
            "candidate_view": context.payload["candidate_cards"][0],
            "selection_is_scripted_not_model_generated": True,
            "compiled_queries": [request.query for request in plan.requests],
            "selection_request_utf8_bytes": len(
                json.dumps(
                    context.messages(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ),
        }
    return {"execution_kind": "synthetic", "api_calls": 0, "hotpot_questions": 0, "views": views}


if __name__ == "__main__":
    print(json.dumps(build_demo(), ensure_ascii=False, indent=2))
