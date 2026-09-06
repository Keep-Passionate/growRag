"""Gold-free PRE-retrieval BASE/FRESH/REUSE comparison over independent RAGs.

One full RAG opportunity per arm; one generation opportunity per rewrite arm.
Equal opportunities do NOT mean equal tokens, latency or money. Factories must
be side-effect-free on construction and provide stateless, isolated backends.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

from growrag.experience.cards import ActivationStage
from growrag.experience.query_views import CardMemoryView
from growrag.experiments.protocol import Action, ExecutionKind, RuntimeQuestion
from growrag.outer_loop import LoopResult, RagBackend, run_outer_loop
from growrag.query_actions import (
    PAIRED_ACTION_PROMPT_VERSION,
    QueryGenerator,
    RewriteDecision,
)
from growrag.query_operators import RewriteForm

SCHEMA_VERSION = "growrag-pre-query-comparison-v1"


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


@dataclass(frozen=True, slots=True)
class QueryComparisonSpec:
    question: RuntimeQuestion
    memory: CardMemoryView
    form: RewriteForm
    # Experiment-level instruction fixed BEFORE selecting a card, not copied from it.
    intent: str
    rag_fingerprint: str
    generator_fingerprint: str
    seed: int = 42

    def __post_init__(self) -> None:
        if not isinstance(self.question, RuntimeQuestion):
            raise TypeError("comparison accepts a gold-free RuntimeQuestion")
        if not isinstance(self.memory, CardMemoryView):
            raise TypeError("comparison requires one preselected typed diagnostic card")
        for name in ("intent", "rag_fingerprint", "generator_fingerprint"):
            _text(getattr(self, name), name)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.form not in (RewriteForm.PARAPHRASE, RewriteForm.EXPAND):
            raise ValueError("only single-query forms are implemented")
        if not isinstance(self.form, RewriteForm):
            raise TypeError("form must be a RewriteForm")
        if self.memory.form != self.form or self.memory.intent != self.intent:
            raise ValueError("card must match the independently fixed form and intent")
        if self.memory.is_source(self.question):
            raise ValueError("a source question cannot be used as an independent target")
        if self.memory.stage is not ActivationStage.PRE_RETRIEVAL:
            raise ValueError("this protocol is PRE_RETRIEVAL only; no shared-prefix repair yet")
        self.memory.check_stage(after_retrieval=False, evidence=())

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def action_order(self) -> tuple[Action, ...]:
        """Stable, question-dependent arm ordering without response-driven changes."""
        return tuple(
            sorted(
                Action,
                key=lambda action: hashlib.sha256(
                    f"{self.seed}:{self.question.question_id}:{action.value}".encode()
                ).digest(),
            )
        )


@dataclass(frozen=True, slots=True)
class ComparisonComponents:
    backend: RagBackend
    generator: QueryGenerator | None
    rag_fingerprint: str
    generator_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class ComparisonArm:
    planned_action: Action
    result: LoopResult


@dataclass(frozen=True, slots=True)
class QueryComparison:
    spec: QueryComparisonSpec
    execution_kind: ExecutionKind
    arms: tuple[ComparisonArm, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, QueryComparisonSpec) or not isinstance(
            self.execution_kind, ExecutionKind
        ):
            raise TypeError("comparison metadata must use typed records")
        if not isinstance(self.arms, tuple) or len(self.arms) != 3:
            raise ValueError("comparison must retain all three arms, including failures")
        if not all(isinstance(arm, ComparisonArm) for arm in self.arms):
            raise TypeError("arms must contain ComparisonArm records")
        if tuple(arm.planned_action for arm in self.arms) != self.spec.action_order:
            raise ValueError("comparison arms must match the frozen execution order")
        if any(arm.result.state.question != self.spec.question for arm in self.arms):
            raise ValueError("all arm results must belong to the same original question")

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "execution_kind": self.execution_kind.value,
            "synthetic_demo": self.execution_kind is ExecutionKind.MOCK,
            "notice": (
                "Scripted test doubles, NOT model performance evidence."
                if self.execution_kind is ExecutionKind.MOCK
                else "Audited adapter execution; not proof of a research effect."
            ),
            "comparison_scope": "pre_retrieval_empty_prefix",
            "budget_scope": "equal call opportunities, NOT equal tokens or money",
            "max_rag_calls_per_arm": 1,
            "max_rewrite_calls_per_rewrite_arm": 1,
            "generator_protocol": PAIRED_ACTION_PROMPT_VERSION,
            "spec_fingerprint": self.spec.fingerprint,
            "spec": asdict(self.spec),
            "action_order": self.spec.action_order,
            "arms": [asdict(arm) for arm in self.arms],
        }


ComponentFactory = Callable[[Action], ComparisonComponents]


def run_query_comparison(
    spec: QueryComparisonSpec,
    factory: ComponentFactory,
    *,
    execution_kind: ExecutionKind,
    allow_real: bool = False,
) -> QueryComparison:
    """Validate every arm before any execution; preserve ordinary backend failures.

    Fingerprints are declarations of frozen corpus/model/prompt/settings, not a
    sandbox proving arbitrary third-party adapters truthful or stateless.
    REAL requires explicit opt-in; the API transport still enforces its own cap.
    """
    if not isinstance(spec, QueryComparisonSpec):
        raise TypeError("a QueryComparisonSpec is required")
    if not isinstance(execution_kind, ExecutionKind) or type(allow_real) is not bool:
        raise TypeError("execution provenance and real opt-in must be explicit")
    if execution_kind is ExecutionKind.REAL and not allow_real:
        raise ValueError("real comparison requires explicit allow_real=True")
    components: dict[Action, ComparisonComponents] = {}
    for action in spec.action_order:
        bundle = factory(action)
        if not isinstance(bundle, ComparisonComponents):
            raise TypeError("factory must return ComparisonComponents")
        if getattr(bundle.backend, "execution_kind", None) is not execution_kind:
            raise ValueError("RAG provenance is missing or mixed")
        if bundle.rag_fingerprint != spec.rag_fingerprint:
            raise ValueError("all arms must use the declared frozen RAG configuration")
        if any(bundle.backend is other.backend for other in components.values()):
            raise ValueError("each arm needs an independent backend instance")
        if action is Action.BASE:
            if bundle.generator is not None or bundle.generator_fingerprint is not None:
                raise ValueError("BASE has no generator or hidden generation budget")
        else:
            generator = bundle.generator
            if getattr(generator, "execution_kind", None) is not execution_kind:
                raise ValueError("generator provenance is missing or mixed")
            if bundle.generator_fingerprint != spec.generator_fingerprint:
                raise ValueError("rewrite arms must use the declared generator configuration")
            if getattr(generator, "comparison_protocol_id", None) != PAIRED_ACTION_PROMPT_VERSION:
                raise ValueError("rewrite arms require the shared paired prompt protocol")
            if any(generator is other.generator for other in components.values()):
                raise ValueError("rewrite arms need independent generator instances")
        components[action] = bundle

    arms = []
    for action in spec.action_order:
        bundle = components[action]
        decision = (
            RewriteDecision()
            if action is Action.BASE
            else RewriteDecision(
                action, spec.form, spec.intent, spec.memory if action is Action.REUSE else None
            )
        )
        result = run_outer_loop(
            spec.question,
            bundle.backend,
            generator=bundle.generator,
            policy=lambda state, selected=decision: selected,
            max_rag_calls=1,
        )
        arms.append(ComparisonArm(action, result))
    return QueryComparison(spec, execution_kind, tuple(arms))
