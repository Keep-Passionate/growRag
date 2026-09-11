"""Source-only PRE admission diagnostic; no target selection or memory training.

The caller supplies one shared budget ledger. This module never resets it,
loads credentials, retries failed calls, or expands the frozen source ID list.
Mock transports exercise contracts only and are not model-effectiveness data.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from growrag.experience.cards import ExperienceCard
from growrag.experience.pre_sources import (
    PreSourceRecord,
    evaluate_pre_source,
    make_pre_source_candidate,
    pre_source_card_to_view,
)
from growrag.experience.query_views import CardMemoryView
from growrag.outer_loop import _call_failure, _event
from growrag.query_operators import RewriteForm

from .budget import BudgetedChatClient
from .data_protocol import DATASET_ID, SPLIT_VERSION, normalize_question, role_for_question
from .hotpot import HotpotExample
from .lexical_retriever import BM25SentenceRetriever
from .llm_adapters import _execution_kind
from .pre_pilot import (
    EVIDENCE_CONTRACT,
    EXTRACTION_VERSION,
    PRE_INTENT,
    digest,
    extract_pre_card,
    run_example,
    write_json,
)
from .protocol import Action, BackendCallError, ExecutionKind, GoldRecord
from .representation_manifest import SCHEMA_VERSION as MANIFEST_SCHEMA
from .run_pilot import PILOT_MODEL

MAX_SOURCES = 256
BUDGET_SCOPE = "ALL calls on the caller's shared ledger; no source-stage budget reset"


@dataclass(frozen=True, slots=True)
class SourcePoolResult:
    source_records: tuple[PreSourceRecord, ...]
    cards: tuple[ExperienceCard, ...]
    views: tuple[CardMemoryView, ...]
    summary: dict


def validate_source_pool_inputs(sources: tuple[HotpotExample, ...], manifest: dict) -> None:
    """Zero-network preflight; the file loader must separately verify raw SHA-256."""
    if not isinstance(sources, tuple) or not 0 < len(sources) <= MAX_SOURCES:
        raise ValueError("sources must be a nonempty tuple of at most 256 examples")
    if not isinstance(manifest, dict) or any(
        manifest.get(key) != value
        for key, value in {
            "schema_version": MANIFEST_SCHEMA,
            "dataset": DATASET_ID,
            "official_split": "train",
            "split_version": SPLIT_VERSION,
            "seed": 42,
            "selected_roles": {
                "source": "memory_seed",
                "target": "calibration_dev",
                "debug": "calibration_dev",
                "check": "calibration_dev",
            },
        }.items()
    ):
        raise ValueError("a train-only representation manifest with frozen roles is required")
    if manifest.get("complete_train_declared") is not True or any(
        manifest.get(key) is not False
        for key in ("official_validation_used", "official_test_used", "selector_train_used")
    ):
        raise ValueError("only declared complete train sources are permitted")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("source_sha256", ""))):
        raise ValueError("manifest must bind the original data SHA-256")
    cap = manifest.get("source_cap")
    if type(cap) is not int or not len(sources) <= cap <= MAX_SOURCES:
        raise ValueError("source cap is invalid or exceeds 256")
    ids, normalized, types = [], [], []
    for example in sources:
        if not isinstance(example, HotpotExample) or not isinstance(example.gold, GoldRecord):
            raise TypeError("source examples require matching offline gold")
        if example.gold.question_id != example.question.question_id:
            raise ValueError("gold belongs to a different source question")
        if not example.gold.supporting_facts or any(not a.strip() for a in example.gold.answers):
            raise ValueError("source gold needs nonempty answers and supporting facts")
        if (
            example.question.dataset != manifest.get("input_dataset_label")
            or role_for_question(example.question.text, seed=42) != "memory_seed"
        ):
            raise ValueError("source dataset or memory_seed role does not match manifest")
        if example.question_type not in {"bridge", "comparison"}:
            raise ValueError("source stratification type is missing")
        BM25SentenceRetriever(example.candidate_context)
        ids.append(example.question.question_id)
        normalized.append(normalize_question(example.question.text))
        types.append(example.question_type)
    if len(set(ids)) != len(ids) or len(set(normalized)) != len(normalized):
        raise ValueError("source IDs and normalized questions must be unique")
    if manifest.get("selected", {}).get("source") != ids:
        raise ValueError("source IDs or their order differ from the frozen manifest")
    expansion = manifest.get("source_expansion_order")
    if not isinstance(expansion, list) or len(expansion) != cap or expansion[: len(ids)] != ids:
        raise ValueError("sources must equal the frozen expansion prefix, not a replacement")
    if len(set(expansion)) != cap:
        raise ValueError("source expansion order contains duplicate IDs")
    expected_types = ["bridge" if i % 2 == 0 else "comparison" for i in range(len(sources))]
    if types != expected_types:
        raise ValueError("source order must preserve frozen type alternation")
    expected_counts = {"total": len(ids), "by_type": dict(Counter(types))}
    if manifest.get("selected_counts", {}).get("source") != expected_counts:
        raise ValueError("source type counts differ from the manifest")
    target_ids = manifest.get("selected", {}).get("target", [])
    if not isinstance(target_ids, list) or set(ids) & set(target_ids):
        raise ValueError("source and reserved target IDs must be disjoint")


def _validate_client(client: BudgetedChatClient, *, allow_real: bool) -> ExecutionKind:
    if not isinstance(client, BudgetedChatClient):
        raise TypeError("one caller-owned BudgetedChatClient is required, including mock runs")
    if type(allow_real) is not bool:
        raise TypeError("allow_real must be an explicit boolean")
    kind = _execution_kind(client)
    if client.block_reason:
        raise ValueError("shared budget is already blocked; no reset or automatic retry")
    if kind is ExecutionKind.REAL:
        host = urlsplit(client.config.base_url).hostname or ""
        if not allow_real:
            raise ValueError("real execution requires allow_real=True")
        if host != "dashscope.aliyuncs.com" and not host.endswith(".cn-beijing.maas.aliyuncs.com"):
            raise ValueError("this price profile only permits the Beijing endpoint")
        if (
            client.config.model != PILOT_MODEL
            or client.config.max_calls > 256
            or client.config.max_output_tokens > 768
            or client.limits.budget_cny > 5
            or client.limits.input_per_million_cny != 0.2
            or client.limits.output_per_million_cny != 0.8
        ):
            raise ValueError("live run differs from the frozen snapshot, budget or price limits")
    return kind


def _ledger(client: BudgetedChatClient) -> dict:
    return {**client.report(), "cost_scope": BUDGET_SCOPE}


def _canonical_body(card: ExperienceCard) -> str:
    """Exact normalized procedural identity, NOT semantic diversity validation."""
    body = card.repair.action_body
    return digest(
        {
            "form": body.form.value,
            "intent": normalize_question(card.repair.intent),
            "rewrite_rule": normalize_question(body.rewrite_rule),
            "preserve": sorted(normalize_question(item) for item in body.preserve),
        }
    )


def run_source_pool(
    sources: tuple[HotpotExample, ...],
    client: BudgetedChatClient,
    directory: Path,
    *,
    manifest: dict,
    allow_real: bool = False,
) -> SourcePoolResult:
    """Observe exactly the manifest sources; ordinary no-gain admission continues.

    Failed source calls or a blocked budget stop the pool after saving that
    source's audit. Malformed extraction JSON is recorded without fabricated
    output or a retry. Persistence failures raise before any further API call.
    The result's cards remain CANDIDATE, never trusted serving memory.
    """
    validate_source_pool_inputs(sources, manifest)
    kind = _validate_client(client, allow_real=allow_real)
    directory = Path(directory)
    for name in ("sources", "source_pool_manifest.json", "final_budget.json", "source_pool.json"):
        if (directory / name).exists():
            raise FileExistsError("source pool output already exists; no automatic resume")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "source_pool_manifest.json", manifest)
    (directory / "sources").mkdir()
    records, cards, views, rows = [], [], [], []
    rejections: Counter[str] = Counter()
    status, stop_reason, extracted_count = "completed", None, 0
    budget_start = _ledger(client)
    write_json(directory / "initial_budget.json", budget_start)
    try:
        for number, example in enumerate(sources):
            path = directory / "sources" / f"{number:03d}"
            run = run_example(example, client, None, path, seed=42, allow_real=allow_real)
            arms = {arm.planned_action: arm.result for arm in run.arms}
            source = PreSourceRecord(
                f"pre-source-{example.question.question_id}",
                example.question,
                arms[Action.BASE],
                arms[Action.FRESH],
                RewriteForm.PARAPHRASE,
                PRE_INTENT,
                run.spec.rag_fingerprint,
                run.spec.generator_fingerprint,
                kind,
            )
            admission = evaluate_pre_source(source, example.gold)
            write_json(path / "source_record.json", source.to_dict())
            records.append(source)
            rejections.update(admission.reasons)
            row = {
                "question_id": example.question.question_id,
                "admission": asdict(admission),
                "card_id": None,
                "extraction_status": "not_eligible",
                "extraction_event": None,
            }
            failed_source = any(
                event.status == "error"
                for arm in (source.base, source.fresh)
                for event in (*arm.events, *arm.component_events)
            )
            if admission.eligible and not client.block_reason and not failed_source:
                start, extracted = perf_counter(), None
                try:
                    query = source.fresh.state.rounds[-1].search_query
                    extracted = extract_pre_card(client, example.question, query)
                    event = _event("extract_card", kind, start, extracted)
                    extracted_count += 1
                    write_json(
                        path / "extraction.json",
                        {
                            "event": asdict(event),
                            "fields": {k: v for k, v in extracted.value.items() if k != "body"},
                            "body": asdict(extracted.value["body"]),
                        },
                    )
                    card = make_pre_source_candidate(
                        source,
                        gold=example.gold,
                        card_id=f"pre-card-{example.question.question_id}",
                        version="v1",
                        created_at=datetime.now(UTC).isoformat(),
                        **extracted.value,
                        evidence_contract=EVIDENCE_CONTRACT,
                        extraction_prompt_version=EXTRACTION_VERSION,
                    )
                    view = pre_source_card_to_view(card, sources={source.source_id: source})
                    write_json(path / "card.json", asdict(card))
                    write_json(path / "view.json", asdict(view))
                    cards.append(card)
                    views.append(view)
                    row.update(card_id=card.versioned_id, extraction_status="card_created")
                except (BackendCallError, ValueError, TypeError) as error:
                    event = _event("extract_card", kind, start, _call_failure(error, extracted))
                    row["extraction_status"] = (
                        "budget_or_transport_failed"
                        if client.block_reason
                        else "invalid_extraction"
                    )
                    rejections[row["extraction_status"]] += 1
                row["extraction_event"] = asdict(event)
                write_json(path / "extraction_event.json", asdict(event))
            write_json(path / "admission.json", row)
            write_json(path / "budget_after_source.json", _ledger(client))
            rows.append(row)
            print(f"Source {number + 1}/{len(sources)}: {row['extraction_status']}", flush=True)
            if client.block_reason or failed_source:
                status = "stopped"
                stop_reason = client.block_reason or "source_call_failed"
                break
        library = {
            "schema_version": "growrag-source-pool-library-v1",
            "execution_kind": kind.value,
            "cards": [asdict(card) for card in cards],
            "views": [asdict(view) for view in views],
            "notice": "Source candidates only; no target transfer or serving trust was validated.",
        }
        write_json(directory / "source_library.json", library)
        unique_bodies = len({_canonical_body(card) for card in cards})
        summary = {
            "schema_version": "growrag-source-pool-pilot-v1",
            "status": status,
            "stop_reason": stop_reason,
            "execution_kind": kind.value,
            "requested_source_count": len(sources),
            "processed_source_count": len(records),
            "unprocessed_source_ids": [e.question.question_id for e in sources[len(records) :]],
            "eligible_source_count": sum(row["admission"]["eligible"] for row in rows),
            "extracted_count": extracted_count,
            "candidate_count": len(cards),
            "nonduplicate_canonical_body_count": unique_bodies,
            "canonical_identity_notice": "Exact normalized body identity, not semantic diversity.",
            "rejection_reason_counts": dict(sorted(rejections.items())),
            "manual_pattern_review_required": True,
            "proposed_feasibility_gate": {
                "minimum_nonduplicate_cards": 6,
                "minimum_manually_reviewed_patterns": 2,
                "card_count_condition_met": unique_bodies >= 6,
                "pattern_condition_met": None,
                "passed": None,
                "notice": "Proposal only; model query_pattern strings do not validate diversity.",
            },
            "library_sha256": digest(library),
            "sources": rows,
            "budget_at_start": budget_start,
            "budget": _ledger(client),
            "targets_executed": 0,
            "automatic_expansion": False,
            "notice": "Train-only source diagnostic; not a representation ranking or benchmark.",
        }
        write_json(directory / "source_pool.json", summary)
        return SourcePoolResult(tuple(records), tuple(cards), tuple(views), summary)
    finally:
        write_json(directory / "final_budget.json", _ledger(client))
