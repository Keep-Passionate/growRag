"""有界算子执行循环：只接受原问和当前证据，不接收gold或训练接口。

模型/规则决定器由调用者提供，检索器同样可替换。一个批次含两条query就消费
两次预算。此处只执行计划和累计证据；最终Reader及离线评分在外部。
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter

from .experiments.protocol import Evidence, RuntimeQuestion
from .macro_operators import (
    GoalContract,
    GroundedBinding,
    OperatorRegistry,
    OperatorSpec,
    RuntimeState,
)


@dataclass(frozen=True)
class ActionProposal:
    goal: GoalContract
    spec: OperatorSpec | None
    gap: dict = field(default_factory=dict)
    bindings: tuple[GroundedBinding, ...] = ()
    origin: str = "fresh"
    reason: str = ""

    def __post_init__(self):
        if not isinstance(self.goal, GoalContract):
            raise TypeError("goal must be GoalContract")
        if self.spec is not None and not isinstance(self.spec, OperatorSpec):
            raise TypeError("spec must be OperatorSpec or None for stop")
        if type(self.gap) is not dict or type(self.bindings) is not tuple:
            raise TypeError("gap/bindings types invalid")
        if any(not isinstance(binding, GroundedBinding) for binding in self.bindings):
            raise TypeError("bindings must contain GroundedBinding objects")
        # 保留普通 dict 便于 asdict/JSON 审计，但不借用调用者可随后修改的容器。
        object.__setattr__(self, "gap", deepcopy(self.gap))
        if self.origin not in {"fresh", "reuse", "static", "stop"}:
            raise ValueError("unknown origin")
        if not isinstance(self.reason, str) or len(self.reason) > 2000:
            raise ValueError("reason must be bounded text")
        if self.spec is None and (self.origin != "stop" or self.gap or self.bindings):
            raise ValueError("stop cannot contain action parameters")
        if self.spec is not None and self.origin == "stop":
            raise ValueError("stop cannot contain an operator")


@dataclass(frozen=True)
class SearchEvent:
    query: str
    evidence_ids: tuple[str, ...]
    new_evidence_ids: tuple[str, ...]
    elapsed_seconds: float
    step: int


@dataclass(frozen=True)
class Observation:
    question: RuntimeQuestion
    evidence: tuple[Evidence, ...]
    searches: tuple[SearchEvent, ...]
    remaining_retrievals: int
    decision_number: int


@dataclass(frozen=True)
class EpisodeResult:
    question_id: str
    evidence: tuple[Evidence, ...]
    searches: tuple[SearchEvent, ...]
    proposals: tuple[ActionProposal, ...]
    stop_reason: str
    rejected_error: str | None = None

    @property
    def retrieval_calls(self):
        return len(self.searches)


def run_operator_episode(
    question: RuntimeQuestion,
    retrieve: Callable[[str, int], tuple[Evidence, ...]],
    decide: Callable[[Observation], ActionProposal],
    *,
    retrieval_budget: int = 3,
    max_decisions: int = 2,
    top_k: int = 6,
    on_event: Callable[[dict], None] | None = None,
) -> EpisodeResult:
    """初检一次，最多两次决策。预算按实际发出的query扣除，不按宏算子扣除。

    不缓存模型答案，不写长期记忆。语义未知，故stop只是模型声明。非法计划会
    留拒绝记录并停止该题修复；网络/检索/解析异常继续抛给持久化runner，禁止伪补。
    """
    if not isinstance(question, RuntimeQuestion):
        raise TypeError("runtime question required; no labelled example")
    if not callable(retrieve) or not callable(decide):
        raise TypeError("retrieve and decide must be callable")
    if on_event is not None and not callable(on_event):
        raise TypeError("on_event must be callable or None")
    for value, lower, upper in ((retrieval_budget, 1, 8), (max_decisions, 0, 4), (top_k, 1, 6)):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError("invalid bounded episode limits")
    emit = on_event or (lambda event: None)
    evidence: dict[str, Evidence] = {}
    searches, proposals = [], []
    registry = OperatorRegistry()
    versions: dict[tuple[str, str], OperatorSpec] = {}

    def search(query, step):
        # 整题第二道防线：即使上层批次编译发生改动，也不能重置已消耗预算。
        if len(searches) >= retrieval_budget:
            raise ValueError("episode retrieval budget exhausted before search")
        emit(
            {
                "kind": "operator_search_start",
                "query": query,
                "step": step,
                "retrieval_number": len(searches) + 1,
            }
        )
        started = perf_counter()
        rows = retrieve(query, top_k)
        if type(rows) is not tuple or any(not isinstance(row, Evidence) for row in rows):
            raise TypeError("retriever must return immutable Evidence tuple")
        if len(rows) > top_k or len({r.evidence_id for r in rows}) != len(rows):
            raise ValueError("retriever violated top-k or unique-ID contract")
        for row in rows:
            if row.evidence_id in evidence and evidence[row.evidence_id] != row:
                raise ValueError("same evidence ID changed content")
        added = []
        for row in rows:
            if row.evidence_id not in evidence:
                added.append(row.evidence_id)
                evidence[row.evidence_id] = row
        event = SearchEvent(
            query,
            tuple(row.evidence_id for row in rows),
            tuple(added),
            perf_counter() - started,
            step,
        )
        searches.append(event)
        emit({"kind": "operator_search", "search": event, "documents": rows})
        return len(added)

    search(question.text, 0)
    stop = "retrieval_budget" if len(searches) == retrieval_budget else "decision_limit"
    error_name = None
    for number in range(1, max_decisions + 1):
        remaining = retrieval_budget - len(searches)
        if remaining <= 0:
            stop = "retrieval_budget"
            break
        observation = Observation(
            question, tuple(evidence.values()), tuple(searches), remaining, number
        )
        proposal = decide(observation)
        if not isinstance(proposal, ActionProposal):
            raise TypeError("decider must return ActionProposal")
        # 日志回调或调用者持有的 proposal 不应改变即将执行的参数和审计记录。
        proposal = deepcopy(proposal)
        proposals.append(proposal)
        emit({"kind": "operator_proposal", "proposal": deepcopy(proposal), "decision": number})
        if proposal.goal.original_question != question.text:
            stop, error_name = "plan_rejected", "original_question_changed"
            emit({"kind": "operator_rejection", "error_type": error_name, "decision": number})
            break
        if proposal.spec is None:
            stop = "controller_stop_claim"
            break
        try:
            key = (proposal.spec.operator_id, proposal.spec.version)
            if key in versions:
                if versions[key] != proposal.spec:
                    raise ValueError("same operator version changed within the episode")
            else:
                registry.register(proposal.spec)
                versions[key] = proposal.spec
            state = RuntimeState(tuple(evidence.values()), proposal.bindings, remaining)
            plan = registry.plan(
                proposal.spec.operator_id,
                proposal.spec.version,
                goal=proposal.goal,
                gap=proposal.gap,
                state=state,
            )
        except (ValueError, TypeError, KeyError) as error:
            stop, error_name = "plan_rejected", type(error).__name__
            emit({"kind": "operator_rejection", "error_type": error_name, "decision": number})
            break
        if not plan.requests:
            stop = "empty_plan_not_sufficiency_proof"
            break
        # 当前检索是确定性的同索引top-k，无排除/翻页；重复query不会带来新文档。
        normalized = [" ".join(r.query.casefold().split()) for r in plan.requests]
        previous = {" ".join(r.query.casefold().split()) for r in searches}
        if len(set(normalized)) != len(normalized) or set(normalized) & previous:
            stop = "repeated_query_same_fixed_topk"
            break
        gain = sum(search(request.query, number) for request in plan.requests)
        if gain == 0:
            stop = "no_new_evidence"
            break
        if len(searches) == retrieval_budget:
            stop = "retrieval_budget"
            break
    result = EpisodeResult(
        question.question_id,
        tuple(evidence.values()),
        tuple(searches),
        tuple(proposals),
        stop,
        error_name,
    )
    emit({"kind": "operator_episode_end", "stop_reason": stop, "retrieval_calls": len(searches)})
    return result
