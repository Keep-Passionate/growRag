"""No-gold terminal audit for the interrupted, immutable candidate study.

只审核预先固定的45道完整五路题、1个失败BASE和24个未启动路径。复用原
合同审计/离线回放，不补齐引用、不重跑、不读取标签、不给出准确率。源码
snapshot和三份原SUMMARY外部硬钉；本脚本位于src之外，不改变运行时源码。
默认只读预检，--write才创建独占的技术审计目录，绝不授权续跑或评分。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from growrag.experiments import score_history_candidates as frozen
from growrag.experiments.shared_continuation import _event_mentions
from growrag.history_library import FrozenHistoryLibrary

SOURCE_SHA256 = "fcf241750b2c9db5e1c733bda2038f9c0d1ae010739f0d9c699a141582bc679e"
FAILED_ID = "5ac28dae5542996366519a06"
COMPLETE_BY_BATCH = (5, 20, 20)
SUMMARY_PINS = (
    "234bf85d01fc22d185ecd17a735af7bb27f1900578fb6c4005ea96ea994a1c90",
    "8ce36cd64c7cd20f3fe2db3d6da6236155db0df68c5a7bf2885c3a9ad4297c67",
    "90afe150ddcf6a8aa759c2f2bb0f94c7e5d120ae5f53d70fe97856c20fa71f07",
)
OUTPUT = "runs/history_candidates45_terminal_audit_v1"
SCHEMA = "growrag-history-candidates45-terminal-audit-v1"


def _check(condition, message):
    frozen._require(condition, message)


def _failed_reader(report, directory, plan, inputs, seen):
    """Audit the original schema-valid response and its exact local rejection.

    The model's answer is neither scored nor exposed. A missing final character
    is diagnosed only; there is no prefix matching or replacement of that ID.
    """
    calls = report.get("calls")
    trace = f"{plan['run_id']}/{FAILED_ID}/base/reader"
    _check(
        report.get("status") == "failed"
        and report.get("error_type") == "ValueError"
        and report.get("reader") is None
        and type(calls) is list
        and len(calls) == 1,
        "unexpected failed BASE state",
    )
    call = calls[0]
    _check(
        call.get("trace_id") == trace and call.get("prompt_version") == frozen.READER_VERSION,
        "failed BASE did not make exactly the original Reader request",
    )
    request, value, _ = frozen._http(
        call, directory, plan, inputs, seen, f"{plan['run_id']}/{FAILED_ID}/base/"
    )
    episode = report["episode"]
    evidence = tuple(frozen.Evidence(**row) for row in episode["evidence"])
    visible = frozen.visible_evidence(evidence)
    payload = {
        "original_question": report["question"]["text"],
        "evidence": visible,
        "evidence_window_omitted_count": len(evidence) - len(visible),
    }
    _check(
        request["messages"] == frozen._messages(frozen.READER_PROMPT, payload),
        "failed Reader request changed",
    )
    visible_ids = {row["evidence_id"] for row in visible}
    unknown = [identity for identity in value["evidence_ids"] if identity not in visible_ids]
    _check(
        value["supported"] is True
        and len(unknown) == 1
        and len(set(value["evidence_ids"])) == len(value["evidence_ids"])
        and len(unknown[0]) == 63
        and sum(identity[:-1] == unknown[0] for identity in visible_ids) == 1,
        "failure is not the pinned missing-last-character unseen citation",
    )
    _check(
        len(episode["searches"]) == 1
        and episode["searches"][0]["step"] == 0
        and episode["searches"][0]["query"] == report["question"]["text"]
        and episode["proposals"] == []
        and episode["stop_reason"] == "retrieval_budget"
        and episode["rejected_error"] is None,
        "failed BASE retrieval episode changed",
    )
    return {
        **frozen._totals(calls),
        "question_id": FAILED_ID,
        "method": "base",
        "transport_status": "completed_http200",
        "execution_status": "failed",
        "error_type": "ValueError",
        "failure_reason": "reader_cites_unseen_evidence_id",
        "unknown_citation_count": 1,
        "unknown_id_length": 63,
        "diagnosis_only_missing_last_character": True,
        "id_repaired": False,
        "reader_answer_exposed": False,
        "accuracy": None,
        "retrieval_calls": 1,
    }


def _journal(directory, calls, ledger, inputs, untouched, tracked):
    """Every request intent/after pair and raw HTTP must have exactly one owner."""
    folder = frozen._inside(directory, directory / "request_journal" / "0000_intent.json").parent
    expected = {f"{i:04d}_{kind}.json" for i in range(len(calls)) for kind in ("intent", "after")}
    _check(
        {p.name for p in folder.iterdir()} == expected, "journal has missing/extra pending intent"
    )
    canonical_audits = {hashlib.sha256(c["trace_id"].encode()).hexdigest() + ".json" for c in calls}
    _check(
        {p.name for p in (directory / "api_audit").iterdir()} == canonical_audits,
        "raw HTTP has missing/extra owner",
    )
    for index, call in enumerate(calls):
        intent = frozen._read(tracked(folder / f"{index:04d}_intent.json"))
        after = frozen._read(tracked(folder / f"{index:04d}_after.json"))
        audit_path = (
            directory
            / "api_audit"
            / (hashlib.sha256(call["trace_id"].encode()).hexdigest() + ".json")
        )
        audit = frozen._read(audit_path)
        _check(
            not _event_mentions(intent, untouched)
            and not _event_mentions(after, untouched)
            and not _event_mentions(audit, untouched),
            "untouched ID appears in request evidence",
        )
        _check(
            intent.get("trace_id") == call["trace_id"]
            and intent.get("prompt_version") == call["prompt_version"]
            and intent.get("status") == "pending_no_automatic_retry"
            and intent.get("request_fingerprint")
            == frozen.fingerprint(audit["request"]["messages"])
            and math.isclose(
                frozen._money(intent.get("potential_reserved_cny")),
                call["reserved_cny"],
                abs_tol=1e-10,
            )
            and math.isclose(
                frozen._money(intent.get("prior_reserved_cny")),
                sum(c["reserved_cny"] for c in calls[:index]),
                abs_tol=1e-10,
            ),
            "intent differs from original HTTP/reservation",
        )
        prefix = calls[: index + 1]
        _check(
            after.get("calls") == prefix and after.get("block_reason") is None,
            "after journal differs from completed call prefix",
        )
        for key, value in frozen._totals(prefix).items():
            _check(
                math.isclose(frozen._money(after.get(key)), value, abs_tol=1e-10),
                "after journal cost differs from original prefix",
            )
    # The single already-observed semantic failure sets the stop flag AFTER the
    # successful HTTP's durable after record. No other finalization change allowed.
    final_after = frozen._read(folder / f"{len(calls) - 1:04d}_after.json")
    _check(
        ledger == {**final_after, "block_reason": ledger["block_reason"]},
        "final ledger differs beyond the documented stop flag",
    )


def collect(project):
    """Fixed terminal state only. It is impossible to request a chosen subset."""
    project = Path(project).resolve(strict=True)
    inputs = {}

    def tracked(path):
        path = path if isinstance(path, Path) else Path(path)
        path = frozen._inside(project, path if path.is_absolute() else project / path)
        inputs[str(path)] = frozen._sha(path)
        return path

    manifest, questions, corpus = frozen.load_bundle(
        project, expected_sha=frozen.study.BUNDLE_SHA256
    )
    ids = manifest["question_ids"]
    _check(
        len(ids) == len(set(ids)) == 50
        and [q.question_id for q in questions] == ids
        and ids[45] == FAILED_ID,
        "terminal scope differs from fixed ordered50",
    )
    untouched, complete = set(ids[46:]), ids[:45]
    by_id = {q.question_id: asdict(q) for q in questions}
    for relative in (
        f"{frozen.BUNDLE}/manifest.json",
        f"{frozen.BUNDLE}/manifest.sha256",
        f"{frozen.BUNDLE}/{manifest['runtime_artifact']['path']}",
    ):
        tracked(relative)
    tracked(corpus)
    for key in ("library", "source_manifest", "background_manifest"):
        tracked(manifest[key]["path"])
    library = FrozenHistoryLibrary.from_json(
        (project / manifest["library"]["path"]).read_text(encoding="utf-8")
    )
    _check(
        library.fingerprint == manifest["library"]["fingerprint"]
        and not set(ids) & set(library.allowed_source_ids)
        and all(not set(r.source_qids) & set(ids) for r in library.records),
        "library role drift",
    )
    snapshot = frozen.source_snapshot(project)
    _check(snapshot["sha256"] == SOURCE_SHA256, "current source differs from frozen runtime")
    for relative in snapshot["files"]:
        tracked(relative)
    runs = project / "runs"
    expected_runs = {frozen.study.run_id(s, n) for s, n in frozen.study.BATCHES}
    for path in runs.rglob("*.claim.json"):
        claim = frozen._read(frozen._inside(project, path))
        if set(ids) & set(claim.get("question_ids", [])):
            _check(
                path.parent == runs and path.name.removesuffix(".claim.json") in expected_runs,
                "duplicate development claim",
            )
    for path in runs.glob("*/launch_plan.json"):
        if set(ids) & set(frozen._read(frozen._inside(project, path)).get("question_ids", [])):
            _check(path.parent.name in expected_runs, "duplicate development launch")
    probe_sha = frozen.study.verify_probe(runs, SOURCE_SHA256)
    probe_root = (project / frozen.PROBE).parent
    tracked(probe_root.with_name(probe_root.name + ".claim.json"))
    for path in probe_root.rglob("*"):
        if path.is_file():
            tracked(path)
    seen, all_calls, records, prior, failed = set(), [], [], {}, None
    for batch_number, ((start, count), completed) in enumerate(
        zip(frozen.study.BATCHES, COMPLETE_BY_BATCH, strict=True)
    ):
        run_id = frozen.study.run_id(start, count)
        directory = runs / run_id
        plan = frozen._read(tracked(directory / "launch_plan.json"))
        claim = frozen._read(tracked(runs / f"{run_id}.claim.json"))
        summary_path = tracked(directory / "SUMMARY.json")
        _check(
            frozen._sha(summary_path) == SUMMARY_PINS[batch_number],
            "original SUMMARY bytes changed",
        )
        summary = frozen._read(summary_path)
        saved = frozen._read(tracked(directory / "source_snapshot.json"))
        ledger = frozen._read(tracked(directory / "final_budget.json"))
        events = [
            json.loads(line)
            for line in tracked(directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        batch_ids, paired_ids = ids[start : start + count], ids[start : start + completed]
        last = batch_number == 2
        expected = {
            **frozen.study.CONFIGURATION,
            "protocol": frozen.study.PROTOCOL,
            "phase": "candidate_development",
            "run_id": run_id,
            "start": start,
            "count": count,
            "methods": list(frozen.study.METHODS),
            "question_ids": batch_ids,
            "bundle_sha256": frozen.study.BUNDLE_SHA256,
            "library_sha256": manifest["library"]["sha256"],
            "library_fingerprint": library.fingerprint,
            "source_sha256": SOURCE_SHA256,
            "gold_loaded": False,
            "memory_updates": False,
            "official_split": "train",
            "probe_summary_sha256": probe_sha,
            "prior_batch_summary_sha256": prior,
        }
        _check(
            all(plan.get(k) == v for k, v in expected.items())
            and claim == {**plan, "plan_sha256": frozen.fingerprint(plan)},
            "launch/claim drift",
        )
        _check(
            saved == snapshot and saved.get("sha256") == frozen.fingerprint(saved.get("files")),
            "saved/current source snapshots differ",
        )
        normal = {
            "protocol": frozen.study.PROTOCOL,
            "run_id": run_id,
            "status": "failed" if last else "completed",
            "failure_type": "ValueError" if last else None,
            "planned_questions": count,
            "completed_questions": completed,
            "source_sha256": SOURCE_SHA256,
            "gold_loaded": False,
            "memory_updated": False,
            "new_operator_induction": False,
        }
        _check(
            all(summary.get(k) == v for k, v in normal.items()), "unexpected terminal batch state"
        )
        terminal, names = [], set()
        for qid in batch_ids:
            states = {}
            for method in frozen.study.METHODS:
                status = (
                    "completed"
                    if qid in paired_ids
                    else "failed"
                    if qid == FAILED_ID and method == "base"
                    else "not_attempted"
                )
                name = f"{qid}_{method}.json" if status != "not_attempted" else None
                states[method] = {"status": status, "path": name}
                if name:
                    names.add(name)
            terminal.append({"question_id": qid, "methods": states})
        _check(
            summary.get("terminal") == terminal
            and set(summary.get("prediction_sha256", {})) == names,
            "45 complete / 1 failed / 24 unattempted coverage changed",
        )
        actual = {p.name for p in directory.glob("*.json") if p.name[:24] in set(ids)}
        raw_names = {p.name for p in (directory / "raw_execution").glob("*.json")}
        _check(actual == names == raw_names, "stray or missing top/raw prediction")
        _check(
            [e["question_id"] for e in events if e.get("kind") == "question_completed"]
            == paired_ids,
            "completion events differ from fixed sequential prefix",
        )
        stops = [e for e in events if e.get("kind") == "batch_stopped"]
        _check(
            (
                len(stops) == 1
                and stops[0].get("question_id") == FAILED_ID
                and stops[0].get("arm") == "base"
                and stops[0].get("error_type") == "ValueError"
            )
            if last
            else not stops,
            "unexpected stop event",
        )
        _check(
            not any(_event_mentions(e, untouched) for e in events)
            and not any(
                e.get("question_id") == FAILED_ID and e.get("arm") in frozen.study.METHODS[1:]
                for e in events
            ),
            "not-attempted question/method was actually touched",
        )
        batch_calls = []
        for qid in batch_ids:
            row = {"question_id": qid, "usage": {}}
            for method in frozen.study.METHODS:
                if terminal[batch_ids.index(qid)]["methods"][method]["status"] == "not_attempted":
                    continue
                name = f"{qid}_{method}.json"
                path = tracked(directory / name)
                report = frozen._read(path)
                _check(
                    frozen._sha(path) == summary["prediction_sha256"][name]
                    and frozen._read(tracked(directory / "raw_execution" / name))
                    == {k: v for k, v in report.items() if k != "method"}
                    and report.get("question_id") == qid
                    and report.get("question") == by_id[qid]
                    and report.get("method") == method
                    and report.get("arm") == frozen.study.underlying_arm(method)
                    and report.get("gold_loaded") is False
                    and report.get("memory_updated") is False
                    and report.get("feedback") is None
                    and report["episode"]["question_id"] == qid,
                    "prediction bytes/identity/role/raw seal drift",
                )
                if qid == FAILED_ID:
                    failed = _failed_reader(report, directory, plan, inputs, seen)
                else:
                    _check(report.get("status") == "completed", "paired method not complete")
                    row["usage"][method] = frozen._audit_method(
                        report, directory, plan, events, library, inputs, seen
                    )
                batch_calls.extend(report["calls"])
            if qid in paired_ids:
                records.append(row)
        _check(
            ledger.get("calls") == batch_calls
            and ledger.get("block_reason") == ("experiment_execution_failure" if last else None),
            "ledger owner/terminal block reason drift",
        )
        for key, value in frozen._totals(batch_calls).items():
            _check(
                math.isclose(frozen._money(ledger.get(key)), value, abs_tol=1e-10)
                and math.isclose(frozen._money(summary.get(key)), value, abs_tol=1e-10),
                "ledger/summary cost drift",
            )
        _journal(directory, batch_calls, ledger, inputs, untouched, tracked)
        all_calls.extend(batch_calls)
        prior = {**prior, run_id: frozen._sha(summary_path)}
    _check(
        [r["question_id"] for r in records] == complete and failed is not None,
        "fixed terminal scope incomplete",
    )
    reconciled = frozen.study.reviewed_history(runs)
    covered = {
        c["trace_id"]
        for c in reconciled["calls"]
        if c["trace_id"].split("/", 1)[0] in expected_runs
    }
    _check(covered == seen, "project request coverage differs from terminal ownership")
    methods = {}
    per_question = []
    for row in records:
        per_question.append(
            {
                "question_id": row["question_id"],
                "methods": {
                    method: {
                        "status": "completed",
                        "offline_contract_replay": usage["offline_contract_replay"],
                        "api_requests": usage["api_requests"],
                        "retrieval_calls": usage["retrieval_calls"],
                        "estimated_actual_cny": usage["estimated_actual_cny"],
                        "reserved_cny": usage["reserved_cny"],
                        "stop_reason": usage["stop_reason"],
                        "reader_input_sha256": usage["reader_input_sha256"],
                        "prepared_card_ids": [c["card_id"] for c in usage["executed_cards"]],
                        "actually_searched_learned_cards": sum(
                            c["source_kind"] == "learned" and c["search_executed"]
                            for c in usage["executed_cards"]
                        ),
                    }
                    for method, usage in row["usage"].items()
                },
            }
        )
    for method in frozen.study.METHODS:
        usages = [row["usage"][method] for row in records]
        methods[method] = {
            key: sum(u[key] for u in usages)
            for key in (
                "api_requests",
                "input_tokens",
                "output_tokens",
                "estimated_actual_cny",
                "reserved_cny",
                "retrieval_calls",
            )
        }
        methods[method]["completed_questions"] = 45
        methods[method]["stop_reasons"] = dict(Counter(u["stop_reason"] for u in usages))
    public = {
        "schema_version": SCHEMA,
        "status": "interrupted_stage_terminal_audit_passed",
        "planned_questions": 50,
        "complete_paired_questions": 45,
        "completed_method_reports": 225,
        "failed_method_reports": 1,
        "not_attempted_method_paths": 24,
        "not_started_question_ids": ids[46:],
        "failed_reader": failed,
        "methods": methods,
        "totals_including_failed_http": frozen._totals(all_calls),
        "source_sha256": SOURCE_SHA256,
        "summary_sha256_pins": list(SUMMARY_PINS),
        "gold_loaded": False,
        "api_calls": 0,
        "accuracy": None,
        "reader_answers_exposed": False,
        "memory_updated": False,
        "prediction_modified": False,
        "continuation_authorized": False,
        "score_gate_amended": False,
        "notice": (
            "Technical audit only; interrupted45 pairs are not a completed50 benchmark. "
            "Original full250 scoring gate remains closed. No ID repair, replayed API, "
            "continuation or label-based scoring is authorized."
        ),
    }
    return public, per_question, inputs


def run(project, *, write=False):
    project = Path(project).resolve(strict=True)
    output = (project / OUTPUT).resolve()
    _check(output.is_relative_to(project / "runs"), "terminal output escapes runs")
    if write and output.exists():
        raise FileExistsError("terminal audit exists; never overwrite")
    summary, rows, inputs = collect(project)
    if not write:
        return summary
    inputs[str(Path(__file__).resolve())] = frozen._sha(Path(__file__))
    _check(
        all(frozen._sha(Path(path)) == digest for path, digest in inputs.items()),
        "inputs changed during terminal audit",
    )
    output.mkdir(exist_ok=False)
    for name, value in (
        ("SUMMARY.json", summary),
        ("per_question_technical.json", rows),
        ("audit_inputs.json", {"inputs": inputs}),
    ):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / "terminal_audit_frozen.json").open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "files": {
                    p.name: frozen._sha(p)
                    for p in output.iterdir()
                    if p.name != "terminal_audit_frozen.json"
                },
                "gold_loaded": False,
                "api_calls": 0,
            },
            handle,
            indent=2,
        )
    return {**summary, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="save exclusive no-answer technical audit"
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(Path.cwd(), write=args.write), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
