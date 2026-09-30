"""Declarative, versioned macro operators that only plan the next search batch.

No model, retrieval, evaluation, historical retention or operator induction runs
here. A caller supplies an operator-specific gap and current-question evidence.
Lexical binding checks establish traceability, not semantic truth or entailment.
The goal is retained separately: a legal subquery need not repeat the full goal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from string import Formatter

from .experiments.protocol import Evidence

_RESERVED = frozenset({"original_question", "constraints"})
_KINDS = {"text": str, "bool": bool, "integer": int}
_FORMATTER = Formatter()


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f"{label} must be nonempty text of at most 2000 characters")


def _name(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise ValueError("names must be plain identifiers")


def _tuple_of(values: object, kind: type, label: str, *, nonempty: bool = False) -> None:
    if type(values) is not tuple or any(not isinstance(item, kind) for item in values):
        raise TypeError(f"{label} must be an immutable tuple of {kind.__name__}")
    if nonempty and not values:
        raise ValueError(f"{label} cannot be empty")


def _unique(values: tuple, label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label}")


def _checked_value(kind: str, value: object, label: str) -> None:
    if type(value) is not _KINDS[kind]:
        raise TypeError(f"{label} must have declared type {kind}")
    if kind == "text":
        _text(value, label)


def _placeholders(template: str) -> tuple[str, ...]:
    if not re.fullmatch(r"(?:[^{}]|\{\{|\}\}|\{[A-Za-z][A-Za-z0-9_]*\})*", template):
        raise ValueError("only plain placeholders are permitted")
    names = []
    for _, field, format_spec, conversion in _FORMATTER.parse(template):
        if field is not None:
            _name(field)
            if format_spec or conversion:
                raise ValueError("only plain placeholders are permitted")
            names.append(field)
    return tuple(names)


@dataclass(frozen=True, slots=True)
class GoalContract:
    """原始目标独立保存；子查询可以省略已知部分，但不替代这个目标。"""

    original_question: str
    intent: str
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.original_question, "original_question")
        _text(self.intent, "intent")
        _tuple_of(self.constraints, str, "constraints")
        for constraint in self.constraints:
            _text(constraint, "constraint")


@dataclass(frozen=True, slots=True)
class GroundedBinding:
    name: str
    value: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _name(self.name)
        if self.name in _RESERVED:
            raise ValueError("binding name is reserved")
        _text(self.value, "binding value")
        _tuple_of(self.evidence_ids, str, "evidence_ids", nonempty=True)
        _unique(self.evidence_ids, "evidence_ids")


@dataclass(frozen=True, slots=True)
class RuntimeState:
    evidence: tuple[Evidence, ...] = ()
    bindings: tuple[GroundedBinding, ...] = ()
    remaining_retrievals: int = 1

    def __post_init__(self) -> None:
        _tuple_of(self.evidence, Evidence, "evidence")
        _tuple_of(self.bindings, GroundedBinding, "bindings")
        if type(self.remaining_retrievals) is not int or self.remaining_retrievals < 0:
            raise ValueError("remaining_retrievals must be a nonnegative integer")
        _unique(tuple(item.evidence_id for item in self.evidence), "evidence identity")
        _unique(tuple(item.name for item in self.bindings), "binding name")
        by_id = {item.evidence_id: item for item in self.evidence}
        for binding in self.bindings:
            if any(identity not in by_id for identity in binding.evidence_ids):
                raise ValueError("binding cites evidence outside the current state")
            phrase = " ".join(binding.value.casefold().split())
            pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
            if not any(
                re.search(
                    pattern, " ".join(f"{by_id[key].title} {by_id[key].text}".casefold().split())
                )
                for key in binding.evidence_ids
            ):
                raise ValueError("binding lacks lexical support in its cited current evidence")


@dataclass(frozen=True, slots=True)
class GapField:
    """声明某个算子自己的 gap 字段；不把所有算子压成统一的缺失事实列表。"""

    name: str
    kind: str = "text"
    required: bool = True

    def __post_init__(self) -> None:
        _name(self.name)
        if self.name in _RESERVED:
            raise ValueError("gap field name is reserved")
        if self.kind not in _KINDS:
            raise ValueError("gap field type must be text, bool or integer")
        if type(self.required) is not bool:
            raise TypeError("required must be bool")


@dataclass(frozen=True, slots=True)
class FieldEquals:
    field: str
    value: str | bool | int

    def __post_init__(self) -> None:
        _name(self.field)
        if type(self.value) not in (str, bool, int):
            raise TypeError("condition value must be text, bool or integer")


@dataclass(frozen=True, slots=True)
class QueryStep:
    step_id: str
    template: str
    when: tuple[FieldEquals, ...] = ()
    requires_bindings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _name(self.step_id)
        _text(self.template, "template")
        _placeholders(self.template)
        _tuple_of(self.when, FieldEquals, "when")
        _unique(tuple(condition.field for condition in self.when), "condition field")
        _tuple_of(self.requires_bindings, str, "requires_bindings")
        _unique(self.requires_bindings, "required binding")
        for name in self.requires_bindings:
            _name(name)
            if name in _RESERVED:
                raise ValueError("binding name is reserved")


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """只注册不可变数据规范；新名字、新字段仍走同一个受限模板解释器。"""

    operator_id: str
    version: str
    supported_intents: tuple[str, ...]
    gap_schema: tuple[GapField, ...]
    steps: tuple[QueryStep, ...]

    def __post_init__(self) -> None:
        _name(self.operator_id)
        _text(self.version, "version")
        _tuple_of(self.supported_intents, str, "supported_intents", nonempty=True)
        _unique(self.supported_intents, "intent")
        for intent in self.supported_intents:
            _text(intent, "intent")
        _tuple_of(self.gap_schema, GapField, "gap_schema")
        _unique(tuple(field.name for field in self.gap_schema), "gap field")
        _tuple_of(self.steps, QueryStep, "steps", nonempty=True)
        if len(self.steps) > 16:
            raise ValueError("operators are limited to 16 query steps")
        _unique(tuple(step.step_id for step in self.steps), "step id")
        fields = {field.name: field for field in self.gap_schema}
        for step in self.steps:
            if set(step.requires_bindings) & set(fields):
                raise ValueError("gap fields cannot impersonate required bindings")
            allowed = _RESERVED | fields.keys() | set(step.requires_bindings)
            if set(_placeholders(step.template)) - allowed:
                raise ValueError("template references an undeclared field or binding")
            for condition in step.when:
                if condition.field not in fields:
                    raise ValueError("condition references an undeclared gap field")
                _checked_value(fields[condition.field].kind, condition.value, condition.field)


@dataclass(frozen=True, slots=True)
class SearchRequest:
    step_id: str
    query: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchPlan:
    operator_id: str
    version: str
    goal: GoalContract
    requests: tuple[SearchRequest, ...]

    @property
    def retrieval_cost(self) -> int:
        return len(self.requests)


class OperatorRegistry:
    """Register data specifications; there is no dispatch on seed operator names."""

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        if not isinstance(spec, OperatorSpec):
            raise TypeError("register requires an OperatorSpec")
        key = (spec.operator_id, spec.version)
        if key in self._specs:
            raise ValueError("an operator version cannot be overwritten")
        self._specs[key] = spec

    def plan(
        self,
        operator_id: str,
        version: str,
        *,
        goal: GoalContract,
        gap: Mapping[str, object],
        state: RuntimeState,
    ) -> SearchPlan:
        """编译下一批查询，不执行检索；每条请求计一次预算，整批超额则拒绝。

        intent 检查只是目标类型约束，不是语义证明。中间实体必须通过
        requires_bindings 声明依赖，且绑定已在 RuntimeState 核对当前证据来源。
        """
        if not isinstance(goal, GoalContract) or not isinstance(state, RuntimeState):
            raise TypeError("plan requires a GoalContract and RuntimeState")
        spec = self._specs[(operator_id, version)]
        if goal.intent not in spec.supported_intents:
            raise ValueError("operator does not support the goal intent")
        if not isinstance(gap, Mapping):
            raise TypeError("gap must be a field mapping")
        fields = {field.name: field for field in spec.gap_schema}
        if set(gap) - set(fields):
            raise ValueError("unknown gap fields")
        for field in spec.gap_schema:
            if field.name not in gap:
                if field.required:
                    raise ValueError(f"missing required gap field: {field.name}")
            else:
                _checked_value(field.kind, gap[field.name], field.name)
        bindings = {binding.name: binding for binding in state.bindings}
        values = dict(gap)
        values.update(
            original_question=goal.original_question, constraints=" ".join(goal.constraints)
        )
        requests = []
        for step in spec.steps:
            if any(
                condition.field not in gap or gap[condition.field] != condition.value
                for condition in step.when
            ):
                continue
            if any(name not in bindings for name in step.requires_bindings):
                raise ValueError(f"unbound dependency in step: {step.step_id}")
            step_values = {
                **values,
                **{name: bindings[name].value for name in step.requires_bindings},
            }
            if set(_placeholders(step.template)) - step_values.keys():
                raise ValueError(f"missing optional gap field used by step: {step.step_id}")
            query = step.template.format_map(step_values).strip()
            _text(query, "compiled query")
            evidence_ids = tuple(
                dict.fromkeys(
                    key for name in step.requires_bindings for key in bindings[name].evidence_ids
                )
            )
            requests.append(SearchRequest(step.step_id, query, evidence_ids))
        if len(requests) > state.remaining_retrievals:
            raise ValueError("next batch exceeds remaining retrieval budget")
        return SearchPlan(operator_id, version, goal, tuple(requests))


def seed_registry() -> OperatorRegistry:
    """Three illustrative specs, not learned operators or validated search policies.

    Gap values are caller declarations. In particular, a 'missing' flag is not
    an automatically verified completeness claim. Bridge execution is two calls
    to plan, separated by external retrieval and an evidence-backed binding.
    """
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec(
            "CONCAT_AUGMENT",
            "1",
            ("lookup", "compare", "bridge"),
            (GapField("search_terms"),),
            (QueryStep("augment", "{original_question} {search_terms}"),),
        )
    )
    registry.register(
        OperatorSpec(
            "COMPARE_ALIGNED",
            "1",
            ("compare",),
            (
                GapField("left"),
                GapField("right"),
                GapField("attribute"),
                GapField("left_missing", "bool"),
                GapField("right_missing", "bool"),
            ),
            (
                QueryStep(
                    "left", "{left} {attribute} {constraints}", (FieldEquals("left_missing", True),)
                ),
                QueryStep(
                    "right",
                    "{right} {attribute} {constraints}",
                    (FieldEquals("right_missing", True),),
                ),
            ),
        )
    )
    registry.register(
        OperatorSpec(
            "BRIDGE_HOP",
            "1",
            ("bridge",),
            (
                GapField("anchor"),
                GapField("bridge_relation"),
                GapField("target_relation"),
                GapField("bridge_missing", "bool"),
            ),
            (
                QueryStep(
                    "find_bridge",
                    "{anchor} {bridge_relation} {constraints}",
                    (FieldEquals("bridge_missing", True),),
                ),
                QueryStep(
                    "follow_bridge",
                    "{bridge} {target_relation} {constraints}",
                    (FieldEquals("bridge_missing", False),),
                    ("bridge",),
                ),
            ),
        )
    )
    return registry
