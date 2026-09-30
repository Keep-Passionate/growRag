"""零 API 的合成演示：展示计划与保留规则，不报告真实问答成绩。

从仓库根目录运行；需将 PYTHONPATH 设为 src。本文例子完全虚构。
"""

from __future__ import annotations

import json
from dataclasses import asdict

from growrag.experiments.protocol import Evidence
from growrag.macro_operators import (
    GapField,
    GoalContract,
    GroundedBinding,
    OperatorSpec,
    QueryStep,
    RuntimeState,
    seed_registry,
)
from growrag.operator_retention import MetricObservation, OperatorOutcome, assess_retention


def build_demo() -> dict:
    registry = seed_registry()
    compare = registry.plan(
        "COMPARE_ALIGNED",
        "1",
        goal=GoalContract("Which was founded earlier, Cedar Lab or Pine Lab?", "compare"),
        gap={
            "left": "Cedar Lab",
            "right": "Pine Lab",
            "attribute": "founding year",
            "left_missing": False,
            "right_missing": True,
        },
        state=RuntimeState(remaining_retrievals=1),
    )
    bridge_goal = GoalContract("Where was the founder of Cedar Lab born?", "bridge")
    bridge_gap = {
        "anchor": "Cedar Lab",
        "bridge_relation": "founder",
        "target_relation": "birthplace",
        "bridge_missing": True,
    }
    first = registry.plan("BRIDGE_HOP", "1", goal=bridge_goal, gap=bridge_gap, state=RuntimeState())
    # 手写的当前证据，仅用于演示第二跳依赖；不是模型返回或实际检索结果。
    evidence = Evidence("synthetic-1", "Cedar Lab", 0, "Mira Vale founded Cedar Lab.")
    second = registry.plan(
        "BRIDGE_HOP",
        "1",
        goal=bridge_goal,
        gap={**bridge_gap, "bridge_missing": False},
        state=RuntimeState(
            evidence=(evidence,),
            bindings=(GroundedBinding("bridge", "Mira Vale", ("synthetic-1",)),),
        ),
    )
    # 第四种算子不需要修改调度器。这里只证明可扩展，不声称自动学会新算子。
    registry.register(
        OperatorSpec(
            "TIME_SCOPED_LOOKUP",
            "1",
            ("lookup",),
            (GapField("entity"), GapField("attribute"), GapField("year", "integer")),
            (QueryStep("lookup", "{entity} {attribute} in {year} {constraints}"),),
        )
    )
    fourth = registry.plan(
        "TIME_SCOPED_LOOKUP",
        "1",
        goal=GoalContract("Who directed Cedar Lab in 2005?", "lookup"),
        gap={"entity": "Cedar Lab", "attribute": "director", "year": 2005},
        state=RuntimeState(),
    )

    def toy_outcome(version: str, value: float) -> OperatorOutcome:
        return OperatorOutcome(
            version,
            "synthetic-state",
            "synthetic-protocol",
            f"synthetic://{version}",
            (MetricObservation("toy_quality", value),),
            feedback_source="proxy",
        )

    retention = assess_retention(
        toy_outcome("candidate-v2", 0.70),
        toy_outcome("historical-v1", 0.55),
        fresh=toy_outcome("fresh-v1", 0.80),
    )
    return {
        "kind": "synthetic_offline_demo_not_experiment",
        "api_calls": 0,
        "actual_retrieval_calls": 0,
        "plans_not_executed": {
            "compare_only_missing_side": asdict(compare),
            "bridge_first_hop": asdict(first),
            "bridge_after_supplied_evidence": asdict(second),
            "registered_fourth_operator": asdict(fourth),
        },
        "toy_retention": {
            **asdict(retention),
            "trusted": retention.trusted,
            "execution_authorized": retention.execution_authorized,
        },
    }


if __name__ == "__main__":
    print(json.dumps(build_demo(), ensure_ascii=False, indent=2))
