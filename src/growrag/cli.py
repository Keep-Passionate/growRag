"""Command-line entry point for the GrowRAG v1 research harness."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from growrag import __version__
from growrag.config import ConfigError, load_pilot_config, load_runtime_config
from growrag.demo import build_demo_decision
from growrag.evaluation.oracle import analyze_oracle, read_oracle_csv
from growrag.experience import SCHEMA_VERSION, SnapshotValidationError, load_snapshot
from growrag.pipeline import TrustedReuseLayer
from growrag.runtime_io import (
    RuntimeIOError,
    load_plan_bundle,
    load_requests,
    write_decision_jsonl,
)
from growrag.selection import (
    LexicalCandidateRecallScorer,
    SignatureApplicabilityScorer,
    TrustedReuseGate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="growrag", description="GrowRAG v1 research tools")
    parser.add_argument("--version", action="version", version=f"growrag {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("demo", help="run a dependency-free trusted-gate demonstration")

    oracle = commands.add_parser("oracle", help="analyze an offline candidate-set CSV")
    oracle.add_argument("--input", required=True, type=Path)
    oracle.add_argument("--output", required=True, type=Path)
    oracle.add_argument("--good-threshold", type=float, default=0.5)

    config = commands.add_parser("config-validate", help="validate a strict TOML config")
    config.add_argument("--config", required=True, type=Path)
    config.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="inspect a template without requiring runtime identities",
    )

    memory = commands.add_parser("memory-validate", help="validate a memory snapshot")
    memory.add_argument("--memory", required=True, type=Path)

    decide = commands.add_parser(
        "decide",
        help="choose DIRECT or REUSE queries from frozen file inputs",
    )
    decide.add_argument("--config", required=True, type=Path)
    decide.add_argument("--memory", required=True, type=Path)
    decide.add_argument("--requests", required=True, type=Path)
    decide.add_argument("--plans", required=True, type=Path)
    decide.add_argument("--output", required=True, type=Path)
    return parser


def _run_demo() -> int:
    print(json.dumps(asdict(build_demo_decision()), ensure_ascii=False, indent=2, default=str))
    return 0


def _run_oracle(input_path: Path, output_path: Path, good_threshold: float) -> int:
    analysis = analyze_oracle(read_oracle_csv(input_path), good_threshold=good_threshold)
    payload = analysis.to_dict()
    payload["source_csv"] = str(input_path.resolve())
    resolved_output = output_path.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Offline candidate-set oracle report written to {resolved_output}")
    return 0


def _run_config_validate(path: Path, *, allow_placeholders: bool) -> int:
    config = load_pilot_config(path) if allow_placeholders else load_runtime_config(path)
    print(
        json.dumps(
            {
                "config": str(path.resolve()),
                "experiment": config.experiment.name,
                "runtime_ready": not allow_placeholders,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _run_memory_validate(path: Path) -> int:
    snapshot = load_snapshot(path)
    state_counts = Counter(record.state.value for record in snapshot.ledger.records)
    print(
        json.dumps(
            {
                "memory": str(path.resolve()),
                "memory_snapshot_id": snapshot.memory_snapshot_id,
                "schema_version": snapshot.schema_version,
                "runtime_ready": snapshot.schema_version == SCHEMA_VERSION,
                "record_count": len(snapshot.ledger),
                "state_counts": dict(sorted(state_counts.items())),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _run_decide(
    *,
    config_path: Path,
    memory_path: Path,
    requests_path: Path,
    plans_path: Path,
    output_path: Path,
) -> int:
    config = load_runtime_config(config_path)
    if not config.actions.direct or not config.actions.reuse or config.actions.fresh:
        raise ConfigError(
            "runtime decide requires actions.direct=true, actions.reuse=true, "
            "and actions.fresh=false"
        )
    snapshot = load_snapshot(memory_path).require_runtime_ready()
    requests = load_requests(requests_path)
    plans = load_plan_bundle(plans_path)
    if (plans.applier_id, plans.application_version) != (
        config.application.applier_id,
        config.application.application_version,
    ):
        raise RuntimeIOError("plan bundle applier identity/version differs from config")
    if config.candidate_recall.scorer != "source-query-jaccard-v1":
        raise ConfigError("unsupported candidate recall scorer: " + config.candidate_recall.scorer)
    if config.applicability.scorer != "signature-coverage-v2":
        raise ConfigError("unsupported applicability scorer: " + config.applicability.scorer)

    applier = plans.to_applier(
        retrieval_query_count=config.query_budget.max_retrieval_queries,
        requested_top_k=config.query_budget.max_top_k,
        context_token_budget=config.query_budget.max_context_tokens,
    )
    layer = TrustedReuseLayer(
        applier=applier,
        recall_scorer=LexicalCandidateRecallScorer(),
        applicability_scorer=SignatureApplicabilityScorer(),
        gate=TrustedReuseGate(
            policy=config.gate_policy,
            budget=config.query_budget,
            environment_policy=config.environment_compatibility,
        ),
        max_candidates=config.candidate_recall.maximum_candidates,
    )
    effective_config = {
        "environment": asdict(config.environment),
        "environment_compatibility": asdict(config.environment_compatibility),
        "gate_policy": asdict(config.gate_policy),
        "query_budget": asdict(config.query_budget),
        "candidate_recall": asdict(config.candidate_recall),
        "applicability": asdict(config.applicability),
        "application": asdict(config.application),
    }
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    artifacts: list[dict[str, object]] = []
    for request in requests:
        result = layer.decide(
            records=snapshot.ledger,
            target_query_id=request.target_query_id,
            target_query=request.target_query,
            current_environment=config.environment,
            current_signatures=frozenset(request.signatures),
        )
        artifacts.append(
            {
                "artifact_type": "trusted_reuse_decision",
                "schema_version": 2,
                "growrag_version": __version__,
                "config_sha256": config_sha256,
                "memory_snapshot_id": snapshot.memory_snapshot_id,
                "request": asdict(request),
                "effective_config": effective_config,
                "recall": {
                    "scorer_id": result.recall_scorer_id,
                    "max_candidates": result.max_candidates,
                    "eligible_experience_ids": list(result.eligible_experience_ids),
                    "candidate_experience_ids": list(result.candidate_experience_ids),
                },
                "portfolio": {
                    "enabled": result.portfolio_enabled,
                    "hot_experience_ids": list(result.hot_experience_ids),
                    "cold_experience_ids": list(result.cold_experience_ids),
                    "expired_experience_ids": list(result.expired_experience_ids),
                    "environment_rejected_experience_ids": list(
                        result.environment_rejected_experience_ids
                    ),
                },
                "decision": asdict(result.decision),
            }
        )
    write_decision_jsonl(output_path, artifacts)
    print(f"Trusted reuse decisions written to {output_path.resolve()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            return _run_demo()
        if args.command == "oracle":
            return _run_oracle(args.input, args.output, args.good_threshold)
        if args.command == "config-validate":
            return _run_config_validate(
                args.config,
                allow_placeholders=args.allow_placeholders,
            )
        if args.command == "memory-validate":
            return _run_memory_validate(args.memory)
        if args.command == "decide":
            return _run_decide(
                config_path=args.config,
                memory_path=args.memory,
                requests_path=args.requests,
                plans_path=args.plans,
                output_path=args.output,
            )
    except (ConfigError, RuntimeIOError, SnapshotValidationError, OSError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"unhandled command: {args.command}")
