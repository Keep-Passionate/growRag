"""Run a scripted, zero-API comparison demo. NOT HotpotQA or LLM effectiveness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from growrag.experience.cards import ActivationStage
from growrag.experience.query_views import CardMemoryView
from growrag.experiments.api_client import ChatResponse
from growrag.experiments.comparison_report import comparison_report, save_comparison
from growrag.experiments.data_protocol import normalize_question
from growrag.experiments.protocol import (
    Action,
    Answer,
    CallResult,
    Evidence,
    ExecutionKind,
    GoldRecord,
    RuntimeQuestion,
    Usage,
)
from growrag.experiments.query_comparison import (
    ComparisonComponents,
    QueryComparisonSpec,
    run_query_comparison,
)
from growrag.outer_loop import RagReply
from growrag.query_actions import APISingleQueryGenerator
from growrag.query_operators import RewriteForm

NOTICE = "SCRIPTED DEMO: handwritten card, query outputs and answers; zero API; no research scores."
INTENT = "terminology alignment"  # Fixed by this demo protocol, not extracted from a card.
EVIDENCE = Evidence("fictional-1", "Briar Observatory", 0, "Ada founded Briar Observatory.")


class ScriptedClient:
    transport_source = "mock"
    config = SimpleNamespace(base_url="https://scripted-demo.invalid/v1", model="SCRIPTED-NOT-LLM")

    def complete(self, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        query = (
            "Briar Observatory established by"
            if "optional_historical_procedure" in payload
            else "Briar Observatory founder"
        )
        return ChatResponse(
            json.dumps({"query": query}),
            "SCRIPTED-NOT-LLM",
            "SCRIPTED-NOT-LLM",
            "scripted-not-network",
            None,
            0,
            0,
            0,
            Path("mock-no-network-log"),
            "mock",
        )


class ScriptedRag:
    execution_kind = ExecutionKind.MOCK

    def run(self, request):
        return CallResult(
            RagReply(Answer("Ada", (EVIDENCE.evidence_id,)), (EVIDENCE,)),
            usage=Usage(0, 0, 0),
            transport_source="mock",
        )


def make_spec() -> QueryComparisonSpec:
    payload = {
        "schema": "growrag-query-card-view-v1",
        "form": "paraphrase",
        "intent": INTENT,
        "conditions": {
            "stage": "pre_retrieval",
            "query_pattern": "founding questions",
            "gap_pattern": None,
            "preconditions": [],
            "contraindications": [],
        },
        "body": {
            "rewrite_rule": "Express a founding question with equivalent search vocabulary.",
            "preserve": ["current entity", "relation direction"],
        },
        "success_criterion": "Preserve the current information need.",
    }
    view = CardMemoryView(
        "synthetic-card@v1",
        "synthetic-source",
        "synthetic-episode#step-2",
        "typed_card_diagnostic_v1",
        json.dumps(payload),
        ("synthetic-source",),
        (hashlib.sha256(normalize_question("Who founded Cedar Library?").encode()).hexdigest(),),
        RewriteForm.PARAPHRASE,
        INTENT,
        ActivationStage.PRE_RETRIEVAL,
        False,
    )
    return QueryComparisonSpec(
        RuntimeQuestion("synthetic-target", "Who founded Briar Observatory?", "synthetic"),
        view,
        RewriteForm.PARAPHRASE,
        INTENT,
        "scripted-rag-v1",
        "scripted-generator-v1",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new directory")
    spec = make_spec()

    def factory(action):
        return ComparisonComponents(
            ScriptedRag(),
            None
            if action is Action.BASE
            else APISingleQueryGenerator(ScriptedClient(), paired_prompt=True),
            spec.rag_fingerprint,
            None if action is Action.BASE else spec.generator_fingerprint,
        )

    run = run_query_comparison(spec, factory, execution_kind=ExecutionKind.MOCK)
    gold = GoldRecord(spec.question.question_id, ("Ada",), ((EVIDENCE.title, 0),))
    save_comparison(run, args.output, gold=gold)
    report = comparison_report(run, gold=gold)
    print(
        json.dumps(
            {
                "notice": NOTICE,
                "output": str(args.output),
                "queries": {action: row["query"] for action, row in report["arms"].items()},
                "api_requests": report["total_experiment_usage"]["api_requests"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
