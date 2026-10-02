"""A0 offline audit and paired feedback; default never opens labels or calls an API.

Every registered question has an explicit terminal state. Only fully paired IDs
are projected from train gold, after external freeze/terminal seals, raw requests,
journals, costs and offline execution replay have passed. Failed costs remain real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from growrag.history_library import FrozenHistoryLibrary

from . import a0_study as study
from .a0_runtime import METHODS, SHORT_READER_VERSION, A0ContractClient, execute_arm, schema_for
from .api_client import _token_count
from .protocol import RuntimeQuestion
from .representation_runner import fingerprint
from .score_history_calibration import _inside, _money, _read, _require, _sha
from .score_operator_sources import score_source_report

OUTPUT = "runs/a0_query_construction_feedback_v1"
METRICS = ("answer_em", "answer_f1", "raw_support_recall", "visible_support_recall")
PAIRS = (
    ("history_body8", "fresh_original"),
    ("history_body8", "fresh_focused"),
    ("fresh_focused", "fresh_original"),
    ("fresh_original", "base"),
    ("fresh_focused", "base"),
    ("history_body8", "base"),
)


def totals(calls):
    """Unknown usage/cost stays unknown, alongside explicitly named known subtotals."""
    result = {"api_requests": sum(c.get("api_requests", 0) for c in calls)}
    for key in ("input_tokens", "output_tokens", "estimated_actual_cny", "reserved_cny"):
        known = [_money(c[key]) for c in calls if c.get(key) is not None]
        result[key] = sum(known) if len(known) == len(calls) else None
        result[f"known_{key}"] = sum(known)
    result["unknown_cost_requests"] = sum(c.get("estimated_actual_cny") is None for c in calls)
    return result


def _same_amount(actual, expected, message):
    _require(
        actual is None
        if expected is None
        else actual is not None and math.isclose(_money(actual), expected, abs_tol=1e-10),
        message,
    )


def _without_elapsed(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {k: _without_elapsed(v) for k, v in value.items() if k != "elapsed_seconds"}
    if isinstance(value, (tuple, list)):
        return [_without_elapsed(v) for v in value]
    return value


def audit_http(call, directory, plan, track, seen, prefix):
    trace = call.get("trace_id", "")
    _require(trace.startswith(prefix) and trace not in seen, "duplicate/wrong API trace")
    seen.add(trace)
    name = hashlib.sha256(trace.encode()).hexdigest() + ".json"
    _require(Path(call.get("audit_path", "")).name == name, "ledger audit identity mismatch")
    data = _read(track(directory / "api_audit" / name))
    _require(
        data.get("trace_id") == trace
        and data.get("prompt_version") == call.get("prompt_version")
        and data.get("transport_source") == "live_api"
        and data.get("network_attempted") is True
        and data.get("api_requests") == call.get("api_requests") == 1
        and data.get("retry_count") == 0
        and data.get("status") == call.get("status")
        and data.get("status") in {"completed", "failed"},
        "raw HTTP/ledger terminal mismatch",
    )
    request = data["request"]
    expected = {
        "model": plan["model"],
        "max_tokens": plan["max_output_tokens"],
        "stream": False,
        "enable_thinking": False,
        "temperature": 0,
        "top_p": 1,
        "response_format": {"type": "json_object"},
    }
    _require(
        set(request) == {*expected, "messages"}
        and all(request[k] == v for k, v in expected.items()),
        "HTTP configuration drift",
    )
    raw = json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8")
    _require(hashlib.sha256(raw).hexdigest() == data.get("request_sha256"), "request hash drift")
    for field in ("input_tokens", "output_tokens"):
        _require(data.get(field) == call.get(field), "raw/ledger token mismatch")
        if call.get(field) is not None:
            _require(type(call[field]) is int and call[field] >= 0, "invalid token usage")
    cost = call.get("estimated_actual_cny")
    if cost is not None:
        expected_cost = (
            _money(call["input_tokens"]) * 0.2 + _money(call["output_tokens"]) * 0.8
        ) / 1_000_000
        _same_amount(cost, expected_cost, "declared-price cost mismatch")
        _require(
            cost <= _money(call["reserved_cny"])
            or call.get("validation_status") == "reservation_assumption_exceeded",
            "reservation exceeded without a recorded budget failure",
        )
    response = data.get("response")
    content = None
    if data["status"] == "completed":
        _require(
            data.get("http_status") == 200
            and data.get("finish_reason") == "stop"
            and data.get("response_redacted") is False,
            "abnormal completed HTTP",
        )
        _require(
            response.get("model") == call.get("returned_model")
            and (
                response.get("model") == plan["model"]
                or call.get("validation_status") == "model_snapshot_mismatch"
            ),
            "returned model mismatch without a recorded budget failure",
        )
        usage = response.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        _require(
            _token_count(usage.get("prompt_tokens")) == call["input_tokens"]
            and _token_count(usage.get("completion_tokens")) == call["output_tokens"],
            "raw usage mismatch",
        )
        choices = response.get("choices", [])
        _require(
            len(choices) == 1 and choices[0].get("finish_reason") == "stop",
            "invalid completion count/finish",
        )
        content = choices[0]["message"]["content"]
        _require(isinstance(content, str) and content.strip(), "empty raw completion")
        # Deliberately do not parse the content here: malformed local outputs must
        # remain auditable failed arms and are tested through actual runtime replay.
    else:
        _require(isinstance(data.get("error_type"), str), "failed HTTP lacks error record")
    if call.get("validation_status") is not None:
        _require(
            call["validation_status"]
            in {"unknown_usage", "model_snapshot_mismatch", "reservation_assumption_exceeded"},
            "unregistered budget validation failure",
        )
        content = None  # Budget rejected this response before the contract client saw it.
    return call, request, content


def replay_report(report, captured, local_events, library, prefix):
    """Re-execute the actual A0 arm from saved messages/responses and retrievals only."""
    _require(
        all(content is not None for _, _, content in captured),
        "transport failures have no complete content replay",
    )
    searches = [e for e in local_events if e.get("kind") == "operator_search"]

    class RecordedClient:
        def __init__(self):
            self.position, self.calls, self.block_reason = 0, [], None
            self.config = SimpleNamespace(json_object_mode=True, json_schema_mode=False)

        def complete(self, messages, *, trace_id, prompt_version):
            _require(self.position < len(captured), "replay requested unrecorded model output")
            call, request, content = captured[self.position]
            _require(
                messages == request["messages"]
                and trace_id == call["trace_id"]
                and prompt_version == call["prompt_version"],
                "replay request differs",
            )
            self.position += 1
            self.calls.append(deepcopy(call))
            return SimpleNamespace(content=content)

    client = RecordedClient()
    position = 0
    emitted = []

    def index(query, top_k):
        nonlocal position
        _require(position < len(searches), "replay requested unrecorded retrieval")
        event = searches[position]
        _require(event["search"]["query"] == query and top_k == 6, "replay query/top-k differs")
        docs = event["documents"]
        _require(
            [d["evidence_id"] for d in docs] == event["search"]["evidence_ids"],
            "retrieval event document order differs",
        )
        position += 1
        return tuple(
            SimpleNamespace(doc_id=d["evidence_id"], title=d["title"], text=d["text"]) for d in docs
        )

    # Never write in an original prediction directory. Temporary output is private
    # and automatically removed; it cannot be discovered as a live run ledger.
    with TemporaryDirectory(prefix="growrag-a0-offline-replay-") as temporary:
        target = Path(temporary) / "replay.json"
        caught = None
        try:
            execute_arm(
                RuntimeQuestion(**report["question"]),
                report["method"],
                index,
                A0ContractClient(client),
                library=library,
                trace=prefix.rstrip("/"),
                log=emitted.append,
                target=target,
            )
        except (ValueError, TypeError, KeyError) as error:
            caught = type(error).__name__
        _require(target.is_file(), "replay did not produce an auditable terminal")
        replayed = _read(target)
    _require(_without_elapsed(replayed) == _without_elapsed(report), "offline replay differs")
    _require(
        client.position == len(captured) and position == len(searches),
        "replay left unused responses/retrievals",
    )
    _require((caught is None) == (report["status"] == "completed"), "replay terminal differs")
    # Check saved attribution against runtime-emitted events, not only final answer.
    keys = {"history_candidates", "history_selection", "history_execution", "operator_search"}
    expected_events = [
        {k: v for k, v in e.items() if k not in {"utc", "question_id", "arm"}}
        for e in local_events
        if e.get("kind") in keys
    ]
    actual_events = [e for e in emitted if e.get("kind") in keys]
    _require(
        _without_elapsed(actual_events) == _without_elapsed(expected_events),
        "replay attribution/search log differs",
    )
    readers = [(c, req) for c, req, _ in captured if c["prompt_version"] == SHORT_READER_VERSION]
    return fingerprint(readers[-1][1]["messages"]) if readers else None


def audit_journal(directory, ledger, captured, track, *, refusal_prefixes=()):
    journal = directory / "request_journal"
    present = {p.name for p in journal.glob("*.json")} if journal.exists() else set()
    if not captured and not present:
        return
    names = {
        f"{i:04d}_{suffix}.json" for i in range(len(captured)) for suffix in ("intent", "after")
    }
    refusal_names = {f"{len(captured):04d}_{suffix}.json" for suffix in ("intent", "after")}
    refusal = present == names | refusal_names
    _require(present == names or refusal, "journal coverage mismatch")
    prior = 0.0
    for number, (call, request, _) in enumerate(captured):
        intent = _read(track(journal / f"{number:04d}_intent.json"))
        after = _read(track(journal / f"{number:04d}_after.json"))
        _require(
            intent.get("trace_id") == call["trace_id"]
            and intent.get("prompt_version") == call["prompt_version"]
            and intent.get("request_fingerprint") == fingerprint(request["messages"])
            and intent.get("status") == "pending_no_automatic_retry",
            "journal intent differs",
        )
        _same_amount(
            intent.get("potential_reserved_cny"), call["reserved_cny"], "intent reserve differs"
        )
        _same_amount(intent.get("prior_reserved_cny"), prior, "intent prior reserve differs")
        _require(after.get("calls") == ledger["calls"][: number + 1], "journal call prefix differs")
        values = totals(after["calls"])
        for key in ("api_requests", "reserved_cny", "estimated_actual_cny"):
            _same_amount(after.get(key), values[key], "journal total differs")
        prior += call["reserved_cny"]
    if refusal:
        intent = _read(track(journal / f"{len(captured):04d}_intent.json"))
        after = _read(track(journal / f"{len(captured):04d}_after.json"))
        _require(
            after.get("block_reason")
            in {"elapsed_time_limit", "estimated_budget_limit", "prompt_size_limit"}
            and after.get("calls") == ledger["calls"]
            and intent.get("status") == "pending_no_automatic_retry"
            and any(intent.get("trace_id", "").startswith(prefix) for prefix in refusal_prefixes)
            and intent.get("trace_id") not in {c[0]["trace_id"] for c in captured}
            and re.fullmatch(r"[0-9a-f]{64}", intent.get("request_fingerprint", "")),
            "unaccounted no-request refusal",
        )
        schema_for(intent.get("prompt_version"))
        _money(intent.get("potential_reserved_cny"))
        _same_amount(intent.get("prior_reserved_cny"), prior, "refusal prior reserve differs")
        _same_amount(after.get("reserved_cny"), prior, "refusal charged an unsent request")
        _same_amount(
            after.get("api_requests"),
            totals(ledger["calls"])["api_requests"],
            "refusal changed HTTP count",
        )
    _require(after == ledger, "last journal differs from final ledger")


def collect(project, *, expected_freeze_sha256, expected_terminal_sha256):
    project = Path(project).resolve(strict=True)
    for value in (expected_freeze_sha256, expected_terminal_sha256):
        _require(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "external freeze and terminal SHA256 required",
        )
    inputs = {}

    def track(path):
        path = _inside(project, Path(path))
        inputs[str(path)] = _sha(path)
        return path

    frozen, manifest, questions, _ = study.load_freeze(project, expected_freeze_sha256)
    track(project / study.FREEZE)
    track(project / study.PROTOCOL_NOTE)
    track(project / study.BUNDLE / "manifest.json")
    track(project / study.BUNDLE / "runtime_questions.jsonl")
    library_path = track(project / manifest["library"]["path"])
    _require(_sha(library_path) == frozen["library_sha256"], "frozen library SHA differs")
    library = FrozenHistoryLibrary.from_json(library_path.read_text(encoding="utf-8"))
    _require(
        library.fingerprint == manifest["library"]["fingerprint"], "library fingerprint differs"
    )
    terminal_path = track(project / study.OUTPUT / "TERMINAL.json")
    _require(_sha(terminal_path) == expected_terminal_sha256, "external terminal SHA differs")
    terminal = _read(terminal_path)
    aggregate_plan = _read(track(project / study.OUTPUT / "launch_plan.json"))
    _require(
        aggregate_plan
        == {
            "protocol": study.PROTOCOL,
            "question_ids": frozen["question_ids"],
            "methods": list(METHODS),
            "freeze_sha256": expected_freeze_sha256,
            "gold_loaded": False,
        },
        "aggregate launch differs",
    )
    _require(
        terminal.get("protocol") == study.PROTOCOL
        and terminal.get("planned_questions") == study.COUNT
        and terminal.get("freeze_sha256") == expected_freeze_sha256
        and terminal.get("status") in {"completed", "stopped"}
        and terminal.get("gold_loaded") is False
        and terminal.get("memory_updated") is False,
        "global terminal policy differs",
    )
    runs = project / "runs"
    records, all_calls, seen, all_terminal, run_names = [], [], set(), [], []
    unstarted_batch_seen, consecutive_failures, previous_batch_stopped = False, 0, False
    sealed_batches = terminal.get("batch_summary_sha256", {})
    _require(type(sealed_batches) is dict, "missing batch terminal seals")
    for start, count in study.BATCHES:
        run_id = f"{study.PREFIX}v1_{start:04d}_{start + count:04d}"
        chosen = questions[start : start + count]
        directory = runs / run_id
        if run_id not in sealed_batches:
            unstarted_batch_seen = True
            _require(
                not directory.exists() and not (runs / f"{run_id}.claim.json").exists(),
                "unsealed attempted batch; labels remain closed",
            )
            rows = [
                {
                    "question_id": q.question_id,
                    "arms": {m: {"status": "not_attempted", "path": None} for m in METHODS},
                }
                for q in chosen
            ]
            all_terminal.extend(rows)
            records.extend(
                {
                    "question_id": q.question_id,
                    "question": q.text,
                    "methods": {},
                    "usage": {},
                    "states": rows[i]["arms"],
                }
                for i, q in enumerate(chosen)
            )
            continue
        _require(
            not unstarted_batch_seen and not previous_batch_stopped,
            "batches resumed after an unstarted/stopped batch",
        )
        run_names.append(run_id)
        summary_path = track(directory / "SUMMARY.json")
        _require(_sha(summary_path) == sealed_batches[run_id], "batch summary SHA differs")
        summary = _read(summary_path)
        plan = _read(track(directory / "launch_plan.json"))
        claim = _read(track(runs / f"{run_id}.claim.json"))
        expected = {
            **study.CONFIGURATION,
            "protocol": study.PROTOCOL,
            "run_id": run_id,
            "phase": "a0_development",
            "question_ids": [q.question_id for q in chosen],
            "methods": list(METHODS),
            "start": start,
            "count": count,
            "freeze_sha256": expected_freeze_sha256,
            "source_sha256": frozen["source_sha256"],
            "bundle_sha256": frozen["bundle_sha256"],
            "library_sha256": frozen["library_sha256"],
            "gold_loaded": False,
        }
        _require(
            all(plan.get(k) == v for k, v in expected.items())
            and claim == {**plan, "plan_sha256": fingerprint(plan)},
            "launch/claim differs",
        )
        snapshot = _read(track(directory / "source_snapshot.json"))
        _require(
            snapshot.get("sha256") == frozen["source_sha256"] == fingerprint(snapshot["files"]),
            "batch source snapshot differs",
        )
        ledger = _read(track(directory / "final_budget.json"))
        prior = _read(track(directory / "prior_budget.json"))
        _same_amount(
            plan.get("prior_reserved_cny"), prior["prior_reserved_cny"], "prior budget differs"
        )
        event_path = directory / "events.jsonl"
        events = [
            json.loads(line) for line in track(event_path).read_text(encoding="utf-8").splitlines()
        ]
        rows, hashes = study.terminal_rows(directory, chosen)
        _require(
            summary.get("terminal") == rows and summary.get("prediction_sha256") == hashes,
            "batch terminal/report seal differs",
        )
        _require(
            summary.get("protocol") == study.PROTOCOL
            and summary.get("run_id") == run_id
            and summary.get("status") in {"completed", "stopped"}
            and summary.get("source_sha256") == frozen["source_sha256"]
            and summary.get("gold_loaded") is False
            and summary.get("memory_updated") is False,
            "batch terminal policy differs",
        )
        paired = sum(all(a["status"] == "completed" for a in r["arms"].values()) for r in rows)
        _require(summary.get("complete_paired_questions") == paired, "paired count differs")
        states = [(r["question_id"], m, r["arms"][m]["status"]) for r in rows for m in METHODS]
        missing_seen = False
        for qid, method, status in states:
            if status == "not_attempted":
                failed_before_report = [
                    e
                    for e in events
                    if e.get("kind") == "arm_failed"
                    and e.get("question_id") == qid
                    and e.get("arm") == method
                ]
                _require(len(failed_before_report) <= 1, "duplicate pre-report failure event")
                if failed_before_report:
                    _require(
                        not missing_seen and consecutive_failures < 3,
                        "failure after prior missing/stop terminal",
                    )
                    consecutive_failures += 1
                missing_seen = True
                continue
            _require(not missing_seen, "attempted arm after missing terminal")
            _require(consecutive_failures < 3, "execution continued after three failed arms")
            consecutive_failures = consecutive_failures + 1 if status == "failed" else 0
        _require(
            summary.get("trailing_consecutive_failed_arms") == consecutive_failures,
            "trailing consecutive failure counter differs",
        )
        if summary["status"] == "completed":
            _require(
                not missing_seen and summary.get("stop_reason") is None,
                "completed batch has missing arms or stop reason",
            )
        else:
            _require(bool(summary.get("stop_reason")), "stopped batch lacks reason")
        previous_batch_stopped = summary["status"] == "stopped" or start == 0 and paired != count
        all_terminal.extend(rows)
        batch_calls, captured_batch, refusal_prefixes = [], [], []
        for question, row in zip(chosen, rows, strict=True):
            item = {
                "question_id": question.question_id,
                "question": question.text,
                "methods": {},
                "usage": {},
                "states": row["arms"],
            }
            for method in METHODS:
                state = row["arms"][method]
                if state["status"] == "not_attempted":
                    continue
                report = _read(track(directory / state["path"]))
                _require(
                    report.get("status") in {"completed", "failed"}
                    and report.get("question") == asdict(question)
                    and report.get("method") == report.get("arm") == method
                    and report.get("gold_loaded") is False
                    and report.get("memory_updated") is False
                    and report.get("feedback") is None,
                    "report identity/gold policy differs",
                )
                prefix = f"{run_id}/{question.question_id}/{method}/"
                captured = [
                    audit_http(c, directory, plan, track, seen, prefix) for c in report["calls"]
                ]
                local = [
                    e
                    for e in events
                    if e.get("question_id") == question.question_id and e.get("arm") == method
                ]
                reader_hash = None
                if report["status"] == "failed":
                    refusal_prefixes.append(prefix)
                if (
                    captured
                    and all(content is not None for _, _, content in captured)
                    and report.get("error_type") != "APIRequestError"
                ):
                    reader_hash = replay_report(report, captured, local, library, prefix)
                else:
                    _require(
                        report["status"] == "failed"
                        and report.get("error_type")
                        and report.get("error_stage") in {"planner_setup", "episode", "reader"}
                        and any(e.get("kind") == "arm_failed" for e in local),
                        "transport/budget failure lacks terminal evidence",
                    )
                if report["status"] == "completed":
                    _require(
                        reader_hash is not None and bool(captured), "completed arm lacks Reader"
                    )
                choices = [e for e in local if e.get("kind") == "history_selection"]
                executions = [e for e in local if e.get("kind") == "history_execution"]
                searches = [e["search"] for e in local if e.get("kind") == "operator_search"]
                item["methods"][method] = report
                item["usage"][method] = {
                    **totals(report["calls"]),
                    "reader_input_sha256": reader_hash,
                    "retrieval_calls": len(searches),
                    "queries": [s["query"] for s in searches],
                    "selected_cards": [
                        {k: e[k] for k in ("selected_card_id", "reason")} for e in choices
                    ],
                    "executed_cards": [
                        {
                            "card_id": e["selected_card_id"],
                            "action_kind": e["action_kind"],
                            "decision": e["decision"],
                            "search_executed": any(s["step"] == e["decision"] for s in searches),
                        }
                        for e in executions
                    ],
                }
                batch_calls.extend(report["calls"])
                captured_batch.extend(captured)
            records.append(item)
        _require(ledger.get("calls") == batch_calls, "ledger omitted/extra/reordered calls")
        audit_journal(directory, ledger, captured_batch, track, refusal_prefixes=refusal_prefixes)
        values = totals(batch_calls)
        for key in ("api_requests", "reserved_cny", "estimated_actual_cny"):
            _same_amount(ledger.get(key), values[key], "ledger total differs")
            _same_amount(summary.get(key), values[key], "summary total differs")
        all_calls.extend(batch_calls)
    _require(set(run_names) == set(sealed_batches), "unknown sealed batch")
    _require(
        all_terminal == terminal.get("terminal") and len(records) == study.COUNT,
        "global terminal does not cover registered cohort",
    )
    _require([r["question_id"] for r in records] == frozen["question_ids"], "cohort order differs")
    if terminal["status"] == "completed":
        _require(
            len(run_names) == len(study.BATCHES) and terminal.get("stop_reason") is None,
            "completed terminal missing batches",
        )
    else:
        _require(bool(terminal.get("stop_reason")), "stopped terminal lacks reason")
    reconciled = study.reviewed_history(runs)
    audited = {
        c["trace_id"] for c in reconciled["calls"] if c["trace_id"].split("/", 1)[0] in run_names
    }
    _require(audited == seen, "project audit/request coverage differs")
    for root in reconciled["roots"]:
        track(Path(root["path"]))
    paired_ids = [
        r["question_id"]
        for r in records
        if all(r["states"][m]["status"] == "completed" for m in METHODS)
    ]
    _require(all(_sha(Path(p)) == digest for p, digest in inputs.items()), "audit inputs changed")
    return {
        "manifest": manifest,
        "records": records,
        "paired_ids": paired_ids,
        "inputs": inputs,
        "totals": totals(all_calls),
        "source_sha256": frozen["source_sha256"],
        "terminal": terminal,
        "model": study.CONFIGURATION["model"],
    }


def load_gold(project, manifest, ids):
    """Project only paired IDs; caller must already hold a successful collect result."""
    _require(
        ids
        and len(ids) == len(set(ids))
        and set(ids) <= set(manifest["question_ids"])
        and ids == [qid for qid in manifest["question_ids"] if qid in set(ids)]
        and manifest["official_split"] == "train",
        "gold projection not registered paired train IDs",
    )
    import pyarrow.dataset as ds

    entry = manifest["source_provenance"]
    path = _inside(project, project / entry["path"])
    _require(_sha(path) == entry["sha256"], "gold provenance differs")
    provenance = _read(path)
    _require(provenance.get("official_split") == "train", "gold provenance is not train")
    recorded = {r["path"].replace("\\", "/"): r["sha256"] for r in manifest["parquet_inputs"]}
    paths, inputs = [], {str(path): _sha(path)}
    for shard in provenance["shards"]:
        part = _inside(project, path.parent / shard["file"])
        _require(
            part.parent == path.parent
            and _sha(part) == shard["sha256"] == recorded.get(part.relative_to(project).as_posix()),
            "gold shard hash differs",
        )
        paths.append(str(part))
        inputs[str(part)] = _sha(part)
    _require(
        len(paths) == len(set(paths)) == len(recorded) == len(manifest["parquet_inputs"]) and paths,
        "gold shard coverage differs",
    )
    rows = (
        ds.dataset(paths, format="parquet")
        .to_table(
            columns=["id", "answer", "context", "supporting_facts"],
            filter=ds.field("id").isin(ids),
        )
        .to_pylist()
    )
    _require(
        len(rows) == len(ids) and {r["id"] for r in rows} == set(ids), "gold ID coverage differs"
    )
    return {r["id"]: r for r in rows}, inputs


def _mean(values):
    known = [v for v in values if v is not None]
    return {"mean": mean(known) if known else None, "valid_n": len(known)}


def summarize(records, gold):
    rows = []
    for item in records:
        qid = item["question_id"]
        if qid not in gold:
            continue
        _require(
            all(item["states"][m]["status"] == "completed" for m in METHODS),
            "gold supplied for an unpaired question",
        )
        canonical, first_by_hash, raw, controlled = {}, {}, {}, {}
        for method in METHODS:
            key = item["usage"][method]["reader_input_sha256"]
            _require(isinstance(key, str) and len(key) == 64, "missing Reader input fingerprint")
            first_by_hash.setdefault(key, method)
            canonical[method] = first_by_hash[key]
            report = item["methods"][method]
            raw[method] = score_source_report(report, gold[qid])
            copied = deepcopy(report)
            # Same WIRE input may map E1 to a different full ID. Keep this arm's
            # evidence windows and mapping; normalize only answer/support/refs.
            first = item["methods"][canonical[method]]["reader"]
            reader = copied["reader"]
            reader["answer"], reader["supported"] = first["answer"], first["supported"]
            reader["evidence_refs"] = deepcopy(first["evidence_refs"])
            reader["evidence_ids"] = [
                reader["alias_to_evidence_id"][ref] for ref in first["evidence_refs"]
            ]
            controlled[method] = score_source_report(copied, gold[qid])
        rows.append(
            {
                "question_id": qid,
                "question": item["question"],
                "reference_answer": gold[qid]["answer"],
                "raw_scores": raw,
                "controlled_reader_scores": controlled,
                "canonical_reader_method": canonical,
                "answers": {m: item["methods"][m]["reader"]["answer"] for m in METHODS},
                "usage": item["usage"],
            }
        )
    _require({r["question_id"] for r in rows} == set(gold), "gold/paired rows differ")
    methods = {}
    for method in METHODS:
        calls = [c for r in records for c in r["methods"].get(method, {}).get("calls", [])]
        states = [r["states"][method]["status"] for r in records]
        methods[method] = {
            "availability": {s: states.count(s) for s in ("completed", "failed", "not_attempted")},
            "all_attempted_cost": totals(calls),
            "paired_questions": len(rows),
            **{
                label: {metric: _mean([r[key][method][metric] for r in rows]) for metric in METRICS}
                for label, key in (
                    ("raw", "raw_scores"),
                    ("controlled_reader", "controlled_reader_scores"),
                )
            },
        }
    pairs = {}
    for left, right in PAIRS:
        same = [
            r["usage"][left]["reader_input_sha256"] == r["usage"][right]["reader_input_sha256"]
            for r in rows
        ]
        values = {
            "identical_reader_input": sum(same),
            "changed_reader_input": len(rows) - sum(same),
        }
        for label, key in (
            ("raw", "raw_scores"),
            ("controlled_reader", "controlled_reader_scores"),
        ):
            em = [(r[key][right]["answer_em"], r[key][left]["answer_em"]) for r in rows]
            values[label] = {
                "repaired": em.count((0.0, 1.0)),
                "harmed": em.count((1.0, 0.0)),
                "delta": {
                    metric: _mean(
                        [
                            None
                            if r[key][left][metric] is None or r[key][right][metric] is None
                            else r[key][left][metric] - r[key][right][metric]
                            for r in rows
                        ]
                    )
                    for metric in METRICS
                },
            }
        pairs[f"{left}_minus_{right}"] = values
    return rows, {
        "planned_questions": len(records),
        "joint_complete_questions": len(rows),
        "methods": methods,
        "paired": pairs,
        "primary_metric_policy": "controlled_reader_first_by_fixed_method_order",
        "reader_control_notice": "Same exact Reader wire messages, first response by METHODS; "
        "raw predictions and all actual costs retained. No deployed cache or cost saving.",
        "missing_notice": "Only joint-complete IDs have gold opened; availability is not accuracy.",
    }


def run(project, *, score=False, expected_freeze_sha256, expected_terminal_sha256):
    project = Path(project).resolve(strict=True)
    output = project / OUTPUT
    if score and output.exists():
        raise FileExistsError("feedback exists; no overwrite or rescoring")
    checked = collect(
        project,
        expected_freeze_sha256=expected_freeze_sha256,
        expected_terminal_sha256=expected_terminal_sha256,
    )
    public = {
        "status": "preflight_passed",
        "planned_questions": study.COUNT,
        "joint_complete_questions": len(checked["paired_ids"]),
        "gold_loaded": False,
        "api_calls": 0,
        "memory_updated": False,
        "totals": checked["totals"],
    }
    if not score:
        return public
    _require(bool(checked["paired_ids"]), "no paired IDs; gold remains closed")
    gold, gold_inputs = load_gold(project, checked["manifest"], checked["paired_ids"])
    rows, summary = summarize(checked["records"], gold)
    inputs = {**checked["inputs"], **gold_inputs}
    _require(
        all(_sha(Path(p)) == digest for p, digest in inputs.items()),
        "inputs changed during scoring",
    )
    summary.update(
        protocol=study.PROTOCOL,
        model=checked["model"],
        totals=checked["totals"],
        api_calls=0,
        memory_updated=False,
        prediction_modified=False,
        gold_ids=checked["paired_ids"],
        source_sha256=checked["source_sha256"],
        external_freeze_sha256=expected_freeze_sha256,
        external_terminal_sha256=expected_terminal_sha256,
        split="official_train_independent_a0_development",
        labels_opened_after_terminal_audit=True,
    )
    missing = [
        {"question_id": r["question_id"], "states": r["states"], "usage": r["usage"]}
        for r in checked["records"]
        if r["question_id"] not in gold
    ]
    output.mkdir(exist_ok=False)
    for name, value in (
        ("SUMMARY.json", summary),
        ("per_question.json", rows),
        ("missing_questions.json", missing),
        ("audit.json", {"inputs": inputs}),
    ):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / "feedback_frozen.json").open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "files": {
                    p.name: _sha(p) for p in output.iterdir() if p.name != "feedback_frozen.json"
                },
                "api_calls": 0,
                "memory_updated": False,
            },
            handle,
            indent=2,
        )
    return {**public, "status": "scored", "gold_loaded": True, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--score", action="store_true")
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--expected-terminal-sha256", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                Path.cwd(),
                score=args.score,
                expected_freeze_sha256=args.expected_freeze_sha256,
                expected_terminal_sha256=args.expected_terminal_sha256,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
