"""Independent-source runtime and global post-collection feedback.

The run_memory_source CLI reuses execute_pair unchanged. All 64 predictions stay
free of feedback until global collection completes, then train labels scan once.
"""

from __future__ import annotations

import ast
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
CORE_PATH = "src/growrag/experiments/source_collection_core.py"
RUNNER_PATH = "src/growrag/experiments/run_memory_source.py"
RUNTIME_FUNCTIONS = (
    "batch_identity",
    "source_plan",
    "load_source_questions",
    "load_source_runtime",
    "collect_batch",
    "recheck_exposure",
)
RUNTIME_CONSTANTS = (
    "PROTOCOL",
    "PREFIX",
    "PLAN_SHA",
    "MAX_BATCH",
    "SERIES_CAP_CNY",
    "SOURCE_COUNT",
)


def execution_signature(snapshot):
    """Separate audit/continuation changes from the unchanged execution method.

    Keep every other src file exact, plus the runtime functions/constants in this
    module. The full per-run snapshot remains independently verified and saved.
    Older launches need not have recorded this derived signature explicitly.
    """
    files = snapshot["files"]
    tree = ast.parse(files[CORE_PATH]["text"])
    selected, imports = {}, []
    for node in tree.body:
        names = []
        if isinstance(node, ast.ImportFrom):
            imports.append(ast.dump(node, include_attributes=False))
        elif isinstance(node, ast.Import):
            aliases = [alias for alias in node.names if alias.name != "ast"]
            if aliases:
                imports.append(ast.dump(ast.Import(names=aliases), include_attributes=False))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [node.name] if node.name in RUNTIME_FUNCTIONS else []
        elif isinstance(node, ast.Assign):
            names = [
                target.id
                for target in node.targets
                if isinstance(target, ast.Name) and target.id in RUNTIME_CONSTANTS
            ]
        for name in names:
            if name in selected:
                raise ValueError("duplicate source runtime definition")
            selected[name] = ast.dump(node, include_attributes=False)
    if set(selected) != set(RUNTIME_FUNCTIONS) | set(RUNTIME_CONSTANTS):
        raise ValueError("source snapshot lacks locked runtime definitions")
    return fingerprint(
        {
            "files": {
                name: entry["sha256"]
                for name, entry in files.items()
                if name not in {CORE_PATH, RUNNER_PATH}
            },
            "core_runtime_ast": selected,
            "core_runtime_imports": imports,
        }
    )


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


def _verify_continuation_links(batches):
    """Overlapping claims are allowed only for explicitly proved unstarted IDs."""
    by_name = {root.name: (root, launch, batch) for root, launch, batch in batches}
    for _root, launch, _ in batches:
        prior = [
            item
            for item in batches
            if item[1]["start"] < launch["start"]
            and set(item[1]["question_ids"]) & set(launch["question_ids"])
        ]
        parent = launch.get("continuation_of")
        proof = launch.get("continuation_proof")
        if not prior:
            if parent is not None or proof is not None:
                raise ValueError("continuation proof has no overlapping parent claim")
            continue
        if len(prior) != 1 or parent != prior[0][0].name or not isinstance(proof, dict):
            raise ValueError("overlapping source claims lack a unique continuation proof")
        previous_root, previous_launch, previous_rows = by_name[parent]
        untouched = previous_launch["question_ids"][len(previous_rows) :]
        required_records = (
            "launch_plan.json",
            "predictions.json",
            "predictions_frozen.json",
            "final_budget.json",
            "events.jsonl",
        )
        if (
            proof.get("schema_version") != "growrag-memory-source-unstarted-continuation-v1"
            or proof.get("prior_run_id") != parent
            or proof.get("parent_run_id") != parent
            or proof.get("question_ids") != launch["question_ids"]
            or proof.get("continued_question_ids") != launch["question_ids"]
            or proof.get("released_question_ids") != untouched
            or not set(launch["question_ids"]).issubset(untouched)
            or any(
                proof.get("prior_record_sha256", {}).get(name) != _sha(previous_root / name)
                for name in required_records
            )
        ):
            raise ValueError("continuation proof differs from frozen unstarted parent prefix")


def verify_collection(runs, plan, questions=None):
    """Seal every source status before labels; failed attempts remain unscored.

    Recovery admits a closed failed prefix and explicit continuation of only its
    unstarted suffix. It never retries, silently drops, or assigns EM=0 to failure.
    """
    reports, hashes, configurations, all_calls, batches = [], {}, set(), set(), []
    question_map = {q.question_id: q.text for q in questions} if questions is not None else {}
    recheck_exposure(runs, plan)
    for claim_path in Path(runs).glob(f"{PREFIX}*.claim.json"):
        if not (
            Path(runs) / claim_path.name.removesuffix(".claim.json") / "launch_plan.json"
        ).is_file():
            raise ValueError("orphan source claim requires reconciliation")
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
        signature = execution_signature(snapshot)
        if launch.get("execution_signature", signature) != signature:
            raise ValueError("declared execution signature differs from snapshot")
        configurations.add(
            fingerprint(
                {
                    "execution_signature": signature,
                    **{
                        k: launch.get(k)
                        for k in (
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
                    },
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
        ledger = json.loads((root / "final_budget.json").read_bytes())
        batch = json.loads((root / "predictions.json").read_bytes())
        seal = json.loads((root / "predictions_frozen.json").read_bytes())
        actual_ids = [r["question_id"] for r in batch]
        if (
            not batch
            or len(batch) > count
            or fingerprint(batch) != seal["reports_sha256_before_scoring"]
            or seal.get("run_id") != root.name
            or seal.get("question_ids") != actual_ids
            or actual_ids != launch["question_ids"][: len(batch)]
            or [r["offset"] for r in batch] != list(range(start, start + len(batch)))
        ):
            raise ValueError("collection prediction seal mismatch")
        if {p.name for p in (root / "questions").iterdir()} != {
            f"{row['offset']:04d}" for row in batch
        }:
            raise ValueError("unowned source question artifacts could hide a started question")
        failed_rows = [r for r in batch if not r["complete_pair"]]
        expected_exit = "failed" if failed_rows else "completed"
        if (
            not events
            or events[-1].get("kind") != "exit"
            or events[-1].get("status") != expected_exit
            or len(failed_rows) > 1
            or (failed_rows and failed_rows[0] is not batch[-1])
            or (failed_rows and ledger.get("block_reason") != "transport_failure")
            or (not failed_rows and (len(batch) != count or ledger.get("block_reason")))
        ):
            raise ValueError("source run has an unreviewed termination or incomplete prefix")
        owned, owned_audits = [], set()
        for row in batch:
            if (
                type(row["complete_pair"]) is not bool
                or row["complete_pair"]
                != all(row["arms"][a]["status"] == "completed" for a in ARMS)
                or any(row["arms"][a].get("feedback") is not None for a in ARMS)
                or "offline_gold_answers" in row
            ):
                raise ValueError("source completion status inconsistent or already labelled")
            path = root / "questions" / f"{row['offset']:04d}" / "prediction_report.json"
            if json.loads(path.read_bytes()) != row:
                raise ValueError("per-question prediction mismatch")
            if question_map and row["question"] != question_map[row["question_id"]]:
                raise ValueError("prediction question differs from frozen source text")
            arm_failed = False
            for arm in ARMS:
                outcome = row["arms"][arm]
                status = outcome.get("status")
                execution_path = path.parent / f"{arm}_execution.json"
                if status == "not_executed":
                    if (
                        not arm_failed
                        or outcome.get("calls") != []
                        or outcome.get("result") is not None
                    ):
                        raise ValueError("unexecuted arm must follow a failed arm and own no calls")
                    if (
                        execution_path.exists()
                        and json.loads(execution_path.read_bytes()) != outcome
                    ):
                        raise ValueError("unexecuted arm differs from prediction")
                    continue
                if status not in {"completed", "failed"} or arm_failed:
                    raise ValueError("source arm continued after failure or has unknown status")
                if json.loads(execution_path.read_bytes()) != outcome:
                    raise ValueError("arm execution differs from frozen prediction")
                if (
                    status == "completed"
                    and outcome["result"].get("question_id") != row["question_id"]
                ):
                    raise ValueError("arm answer has wrong source identity")
                if status == "failed":
                    arm_failed = True
                    if (
                        outcome.get("result") is not None
                        or outcome.get("error_type") != "APIRequestError"
                    ):
                        raise ValueError("only audited length-failed attempts are reviewed")
                calls = outcome["calls"]
                if not calls or (arm == ARMS[0] and len(calls) != 1) or len(calls) > 10:
                    raise ValueError("source arm call count mismatch")
                for call_index, call in enumerate(calls):
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
                    owned_audits.add(audit_path)
                    request = audit.get("request", {})
                    failed_call = status == "failed" and call_index == len(calls) - 1
                    expected_status = "failed" if failed_call else "completed"
                    if failed_call and (
                        audit.get("error_type") != "APIRequestError"
                        or audit.get("response", {}).get("choices", [{}])[0].get("finish_reason")
                        != "length"
                        or call.get("output_tokens") != SHARED_OUTPUT_CAPS[stage]
                    ):
                        raise ValueError("failed source call is not the reviewed length failure")
                    if (
                        audit.get("status") != expected_status
                        or call.get("status", "completed") != expected_status
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
        if owned_audits != {p.resolve() for p in (root / "api_audit").glob("*.json")}:
            raise ValueError("unowned source API audit could hide a started question")
        if not math.isclose(
            sum(c["reserved_cny"] for c in owned), ledger["reserved_cny"], abs_tol=1e-12
        ):
            raise ValueError("source reservation sum mismatch")
        reports.extend(batch)
        batches.append((root, launch, batch))
    reports.sort(key=lambda r: r["offset"])
    if [r["offset"] for r in reports] != list(range(SOURCE_COUNT)) or [
        r["question_id"] for r in reports
    ] != plan["roles"]["source"]:
        raise ValueError("complete source64 collection required before any gold")
    if len(configurations) != 1:
        raise ValueError("source batches changed model, corpus, prompt or execution code")
    _verify_continuation_links(batches)
    return reports, hashes


def project_source_gold(raw_path, questions):
    """AFTER global status seal only: project labels for completed source IDs."""
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
    completed_ids = [row["question_id"] for row in reports if row["complete_pair"]]
    failed_ids = [row["question_id"] for row in reports if not row["complete_pair"]]
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "collection_frozen.json",
        {
            "protocol": PROTOCOL,
            "source_plan_sha256": PLAN_SHA,
            "question_ids": plan["roles"]["source"],
            "completed_question_ids": completed_ids,
            "failed_question_ids": failed_ids,
            "scoring_policy": "completed-pairs-only-after-all-source-statuses-frozen-v2",
            "prediction_artifact_sha256": hashes,
            "created_utc": datetime.now(UTC).isoformat(),
            "gold_not_loaded_yet": True,
        },
    )
    selected_questions = tuple(q for q in runtime.questions if q.question_id in completed_ids)
    gold = project_source_gold(raw_path, selected_questions)
    for row in reports:
        if not row["complete_pair"]:
            row["scoring_status"] = "not_scored_incomplete_pair"
            continue
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
