"""Independent-source runtime and global post-collection feedback.

The run_memory_source CLI reuses execute_pair unchanged. All 64 predictions stay
free of feedback until global collection completes, then train labels scan once.
"""

from __future__ import annotations

import hashlib
import json
import math
import mmap
from datetime import UTC, datetime
from pathlib import Path

from .fresh_dev_manifest import SOURCE_SHA, _ids, _project, _record_spans, _sha
from .memory_source_plan import COUNTS, SCHEMA
from .pre_pilot import write_json
from .protocol import RuntimeQuestion
from .representation_runner import fingerprint
from .run_pilot import PILOT_MODEL
from .run_reformer_pilot import FROZEN_MANIFEST_SHA
from .run_shared_s2g import (
    ARMS,
    GENERATION_PROFILE,
    SHARED_OUTPUT_CAPS,
    execute_pair,
    summarize,
)
from .s2g_author_api import PROMPT_VERSIONS
from .shared_hotpot_scale import _gold
from .shared_s2g_corpus import (
    ExactGold,
    ExactSupport,
    SharedBM25Index,
    SharedRuntime,
    _manifest,
    _verify_file,
    score_result,
)

PROTOCOL = "growrag-independent-memory-source-collection-v1"
PREFIX = "2026-09-27_memory_source_v1_"
PLAN_SHA = "e3f3d1e22ebd1f7289d7ae5c336ee1e1894adb67573038b487867a06e3ee3362"
MAX_BATCH = 16
SERIES_CAP_CNY = 3.0
SOURCE_COUNT = 64


def batch_identity(start, count):
    if (
        type(start) is not int
        or type(count) is not int
        or start < 0
        or not 1 <= count <= MAX_BATCH
        or start + count > SOURCE_COUNT
    ):
        raise ValueError("only 1-16 frozen source questions within offsets 0-63")
    return f"{PREFIX}{start:04d}_{start + count:04d}"


def source_plan(path):
    path = Path(path)
    if _sha(path) != PLAN_SHA:
        raise ValueError("source plan hash mismatch")
    plan = json.loads(path.read_bytes())
    if (
        plan.get("schema_version") != SCHEMA
        or plan.get("official_split") != "train"
        or plan.get("official_dev_test_used") is not False
        or plan.get("counts") != COUNTS
        or plan.get("source_manifest_sha256") != FROZEN_MANIFEST_SHA
        or plan.get("source_data_sha256") != SOURCE_SHA
    ):
        raise ValueError("unreviewed source plan")
    seen = set()
    for role, n in COUNTS.items():
        ids = plan["roles"][role]
        if len(ids) != n or len(set(ids)) != n or set(ids) & seen:
            raise ValueError("role counts, uniqueness or separation mismatch")
        seen.update(ids)
    if seen & set(plan["forbidden_question_ids"]):
        raise ValueError("old development/exposed ID cannot become a source")
    return plan


def load_source_questions(manifest_path):
    manifest_path = Path(manifest_path)
    plan = source_plan(manifest_path)
    path = manifest_path.parent / "source_runtime_questions.jsonl"
    spec = plan["artifacts"][path.name]
    if spec.get("contains_gold") is not False or spec.get("rows") != SOURCE_COUNT:
        raise ValueError("runtime contract must explicitly exclude gold")
    if _sha(path) != spec["sha256"]:
        raise ValueError("source runtime file hash mismatch")
    rows = [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
    if any(set(r) != {"question_id", "text", "dataset"} for r in rows):
        raise ValueError("runtime contains a label or unallowlisted field")
    if [r["question_id"] for r in rows] != plan["roles"]["source"]:
        raise ValueError("runtime must contain only ordered source64")
    return tuple(RuntimeQuestion(**row) for row in rows), plan


def load_source_runtime(manifest_path, corpus_manifest, index_path):
    """Load source64 text only, plus the existing corpus; no old500 question text."""
    questions, plan = load_source_questions(manifest_path)
    root, corpus, _ = _manifest(corpus_manifest, FROZEN_MANIFEST_SHA)
    corpus_path = _verify_file(root, corpus, "corpus.jsonl")
    spec = corpus["artifacts"]["corpus.jsonl"]
    if spec["sha256"] != plan["corpus_sha256"]:
        raise ValueError("source collection must keep the same corpus")
    # Use the reviewed index exactly; never silently build a new index in a live run.
    if not Path(index_path).is_file():
        raise ValueError("existing frozen shared index required")
    index = SharedBM25Index(index_path, corpus_path, spec["sha256"], spec["rows"], k1=1.2, b=0.75)
    return SharedRuntime(
        questions,
        index,
        {
            "source_plan_sha256": PLAN_SHA,
            "corpus_manifest_sha256": FROZEN_MANIFEST_SHA,
            "corpus_sha256": index.corpus_sha256,
            "document_count": spec["rows"],
            "question_count": len(questions),
            "index_path": str(index.path),
            "retrieval_config": dict(index.retrieval_config),
            "context_fingerprint": index.context_fingerprint,
            "gold_loaded": False,
            "calibration_probe_text_loaded": False,
        },
    )


def collect_batch(questions, runtime, client, upstream, output, progress, *, run_id, start):
    """Reuse exactly the existing BASE+S2G executor; not one label is loaded here."""
    reports, started = [], 0
    try:
        for offset, question in enumerate(questions, start):
            if client.block_reason:
                break
            started += 1
            directory = output / "questions" / f"{offset:04d}"
            arms = execute_pair(
                question, runtime.index, client, upstream, directory, progress, run_id=run_id
            )
            report = {
                "question_id": question.question_id,
                "question": question.text,
                "offset": offset,
                "arms": arms,
                "complete_pair": all(arms[a]["status"] == "completed" for a in ARMS),
                "scoring_status": "deferred_until_entire_source_collection_frozen",
            }
            reports.append(report)
            write_json(directory / "prediction_report.json", report)
            progress(
                {
                    "kind": "question_complete",
                    "question_id": question.question_id,
                    "status": "completed" if report["complete_pair"] else "failed",
                    "requests": client.attempts,
                }
            )
        write_json(
            output / "predictions_frozen.json",
            {
                "run_id": run_id,
                "question_ids": [r["question_id"] for r in reports],
                "reports_sha256_before_scoring": fingerprint(reports),
                "created_utc": datetime.now(UTC).isoformat(),
            },
        )
    finally:
        write_json(output / "predictions.json", reports)
        summary = summarize(
            reports,
            client.calls,
            run_id=run_id,
            planned=len(questions),
            stop_reason=client.block_reason,
            started=started,
        )
        summary.update(
            protocol=PROTOCOL,
            purpose="source_trajectory_collection_only",
            gold_loaded=False,
            memory_cards_created=0,
            notice="Frozen source64, BASE+S2G migration; "
            "not evaluation scores or a memory library.",
        )
        write_json(output / "summary.json", summary)
    return summary


def recheck_exposure(runs, plan):
    """New foreign claims since splitting must not silently expose source IDs."""
    wanted = set(plan["roles"]["source"])
    for path in [*Path(runs).glob("*/launch_plan.json"), *Path(runs).glob("*.claim.json")]:
        identity = path.parent.name if path.name == "launch_plan.json" else path.name
        if identity.startswith(PREFIX):
            continue
        if wanted & _ids(json.loads(path.read_bytes())):
            raise ValueError("source ID has a foreign experimental exposure after splitting")


def verify_collection(runs, plan, questions=None):
    """All 64 must finish before labels are opened; failures need separate decisions."""
    reports, hashes, configurations, all_calls = [], {}, set(), set()
    question_map = {q.question_id: q.text for q in questions} if questions is not None else {}
    recheck_exposure(runs, plan)
    for root in sorted(Path(runs).glob(f"{PREFIX}*")):
        if not root.is_dir():
            continue
        launch = json.loads((root / "launch_plan.json").read_bytes())
        if launch["protocol"] != PROTOCOL or launch["manifest_sha256"] != PLAN_SHA:
            raise ValueError("unreviewed collection launch")
        start, count = launch["start"], launch["count"]
        if (
            root.name != batch_identity(start, count)
            or launch["run_id"] != root.name
            or launch["question_ids"] != plan["roles"]["source"][start : start + count]
            or launch["max_calls"] != 11 * count
            or launch["model"] != PILOT_MODEL
            or not launch["git"].get("commit")
            or launch["git"].get("worktree_dirty") is not False
            or launch.get("generation_profile") != GENERATION_PROFILE
            or launch.get("backend_output_caps") != SHARED_OUTPUT_CAPS
            or launch.get("max_retrieval_rounds") != 4
            or launch.get("top_docs") != 6
            or launch.get("gap_profile") != "paper_k1"
            or launch.get("temperature") != 0
            or launch.get("top_p") != 1
            or launch.get("enable_thinking") is not False
        ):
            raise ValueError("launch source identity or limits mismatch")
        claim = json.loads((Path(runs) / f"{root.name}.claim.json").read_bytes())
        if (
            claim.get("protocol") != PROTOCOL
            or claim.get("manifest_sha256") != PLAN_SHA
            or claim.get("question_ids") != launch["question_ids"]
            or claim.get("plan_sha256") != fingerprint(launch)
        ):
            raise ValueError("exclusive source claim mismatch")
        snapshot = json.loads((root / "source_snapshot.json").read_bytes())
        if (
            not snapshot.get("files")
            or fingerprint(snapshot["files"]) != snapshot["sha256"]
            or snapshot["sha256"] != launch["source_sha256"]
            or any(
                hashlib.sha256(v["text"].encode()).hexdigest() != v["sha256"]
                for v in snapshot["files"].values()
            )
        ):
            raise ValueError("source snapshot changed")
        configurations.add(
            fingerprint(
                {
                    k: launch.get(k)
                    for k in (
                        "source_sha256",
                        "model",
                        "generation_profile",
                        "backend_output_caps",
                        "author",
                        "index_path",
                        "runtime_metadata",
                        "temperature",
                        "top_p",
                        "enable_thinking",
                    )
                }
            )
        )
        if not (root / "final_budget.json").is_file():
            raise ValueError("collection run still pending")
        events = [
            json.loads(line)
            for line in (root / "events.jsonl").read_bytes().splitlines()
            if line.strip()
        ]
        if (
            not events
            or events[-1].get("kind") != "exit"
            or events[-1].get("status") != "completed"
        ):
            raise ValueError("source run did not exit successfully")
        ledger = json.loads((root / "final_budget.json").read_bytes())
        batch = json.loads((root / "predictions.json").read_bytes())
        seal = json.loads((root / "predictions_frozen.json").read_bytes())
        if (
            fingerprint(batch) != seal["reports_sha256_before_scoring"]
            or seal.get("run_id") != root.name
            or seal.get("question_ids") != launch["question_ids"]
            or [r["question_id"] for r in batch] != launch["question_ids"]
            or [r["offset"] for r in batch] != list(range(start, start + count))
        ):
            raise ValueError("collection prediction seal mismatch")
        owned = []
        for row in batch:
            if (
                not row["complete_pair"]
                or any(row["arms"][a]["status"] != "completed" for a in ARMS)
                or any(row["arms"][a].get("feedback") is not None for a in ARMS)
                or "offline_gold_answers" in row
            ):
                raise ValueError("source collection is incomplete or already labelled")
            path = root / "questions" / f"{row['offset']:04d}" / "prediction_report.json"
            if json.loads(path.read_bytes()) != row:
                raise ValueError("per-question prediction mismatch")
            if question_map and row["question"] != question_map[row["question_id"]]:
                raise ValueError("prediction question differs from frozen source text")
            for arm in ARMS:
                outcome = row["arms"][arm]
                if json.loads((path.parent / f"{arm}_execution.json").read_bytes()) != outcome:
                    raise ValueError("arm execution differs from frozen prediction")
                if outcome["result"].get("question_id") != row["question_id"]:
                    raise ValueError("arm answer has wrong source identity")
                calls = outcome["calls"]
                if not calls or (arm == ARMS[0] and len(calls) != 1) or len(calls) > 10:
                    raise ValueError("source arm call count mismatch")
                for call in calls:
                    trace = call["trace_id"]
                    stages = {value: name for name, value in PROMPT_VERSIONS.items()}
                    stage = stages.get(call.get("prompt_version"))
                    if (
                        stage is None
                        or call.get("api_requests") != 1
                        or any(
                            type(call.get(k)) is not int or call[k] < 0
                            for k in ("input_tokens", "output_tokens")
                        )
                        or any(
                            isinstance(call.get(k), bool)
                            or not isinstance(call.get(k), (int, float))
                            or not math.isfinite(call[k])
                            or call[k] < 0
                            for k in ("reserved_cny", "estimated_actual_cny")
                        )
                    ):
                        raise ValueError("invalid source call role or usage")
                    if trace in all_calls or not trace.startswith(
                        f"{root.name}/{row['question_id']}/{arm}/"
                    ):
                        raise ValueError("duplicate or foreign source call")
                    all_calls.add(trace)
                    audit_path = Path(call["audit_path"]).resolve()
                    if not audit_path.is_relative_to((root / "api_audit").resolve()):
                        raise ValueError("API audit escaped source run")
                    audit = json.loads(audit_path.read_bytes())
                    request = audit.get("request", {})
                    if (
                        audit.get("status") != "completed"
                        or audit.get("http_status") != 200
                        or audit.get("retry_count") != 0
                        or audit.get("request", {}).get("model") != PILOT_MODEL
                        or audit.get("response", {}).get("model") != PILOT_MODEL
                        or audit.get("network_attempted") is not True
                        or audit.get("transport_source") != "live_api"
                        or request.get("max_tokens") != SHARED_OUTPUT_CAPS[stage]
                        or request.get("temperature") != 0
                        or request.get("top_p") != 1
                        or request.get("enable_thinking") is not False
                        or request.get("stream") is not False
                        or any(
                            audit.get(k) != call.get(k)
                            for k in (
                                "trace_id",
                                "prompt_version",
                                "api_requests",
                                "input_tokens",
                                "output_tokens",
                            )
                        )
                    ):
                        raise ValueError("source API audit differs from ledger")
                owned.extend(calls)
            hashes[str(path.resolve())] = _sha(path)
        if owned != ledger["calls"] or len(owned) != ledger["api_requests"]:
            raise ValueError("source ledger not uniquely owned by predictions")
        if not math.isclose(
            sum(c["reserved_cny"] for c in owned), ledger["reserved_cny"], abs_tol=1e-12
        ):
            raise ValueError("source reservation sum mismatch")
        reports.extend(batch)
    reports.sort(key=lambda r: r["offset"])
    if [r["offset"] for r in reports] != list(range(SOURCE_COUNT)) or [
        r["question_id"] for r in reports
    ] != plan["roles"]["source"]:
        raise ValueError("complete source64 collection required before any gold")
    if len(configurations) != 1:
        raise ValueError("source batches changed model, corpus, prompt or source code")
    return reports, hashes


def project_source_gold(raw_path, questions):
    """AFTER global seal only: project labels for the explicit source64 allowlist."""
    if _sha(raw_path) != SOURCE_SHA:
        raise ValueError("raw official train mirror changed")
    by_id = {q.question_id: q for q in questions}
    found = {}
    with (
        Path(raw_path).open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        for start, end in _record_spans(mapped):
            raw = mapped[start:end]
            qid = _project(raw, {"_id"}).get("_id")
            if qid not in by_id:
                continue
            if qid in found:
                raise ValueError("duplicate source ID in raw train")
            projected = _project(raw, {"_id", "answer", "supporting_facts", "context"})
            projected["question"] = by_id[qid].text
            gold = _gold(projected)
            found[qid] = ExactGold(
                qid,
                tuple(gold["answers"]),
                tuple(ExactSupport(**p) for p in gold["exact_support"]),
                gold["annotation_status"],
                gold.get("annotation_issue"),
            )
    if found.keys() != by_id.keys():
        raise ValueError("source ID missing from raw train")
    return found


def score_collection(runs, source_manifest, runtime, raw_path, output):
    """Separate offline action. Never modifies immutable runtime prediction files."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    plan = source_plan(source_manifest)
    reports, hashes = verify_collection(runs, plan, runtime.questions)
    if [q.question_id for q in runtime.questions] != plan["roles"]["source"]:
        raise ValueError("offline source runtime mismatch")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "collection_frozen.json",
        {
            "protocol": PROTOCOL,
            "source_plan_sha256": PLAN_SHA,
            "question_ids": plan["roles"]["source"],
            "prediction_artifact_sha256": hashes,
            "created_utc": datetime.now(UTC).isoformat(),
            "gold_not_loaded_yet": True,
        },
    )
    gold = project_source_gold(raw_path, runtime.questions)
    for row in reports:
        scores = {
            a: score_result(
                row["arms"][a]["result"],
                gold[row["question_id"]],
                runtime.index,
                retained_mode="raw" if a == ARMS[0] else "sources",
            )
            for a in ARMS
        }
        for arm in ARMS:
            row["arms"][arm]["feedback"] = scores[arm]
        row["offline_gold_answers"] = list(gold[row["question_id"]].answers)
        row["scoring_status"] = "completed"
    write_json(output / "scored_source_reports.json", reports)
    return reports
