"""Synthetic interrupted50 only. No real labels, credential, API or source edits."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from test_score_history_candidates import sealed as _sealed_fixture
from test_score_history_candidates import write

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audits/audit_history_candidate_terminal.py"
SPEC = importlib.util.spec_from_file_location("history_candidate_terminal_audit", SCRIPT)
auditor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auditor)
complete_fixture = _sealed_fixture


@pytest.fixture
def interrupted(complete_fixture, monkeypatch):
    root, _, directories, manifest = complete_fixture
    frozen = auditor.frozen
    ids = manifest["question_ids"]
    failed_id = ids[45]
    directory = directories[-1]
    summary_path = directory / "SUMMARY.json"
    summary = frozen._read(summary_path)
    report_path = directory / f"{failed_id}_base.json"
    report = frozen._read(report_path)
    report.update(status="failed", error_type="ValueError", reader=None)
    write(report_path, report)
    write(
        directory / "raw_execution" / report_path.name,
        {k: v for k, v in report.items() if k != "method"},
    )
    call = report["calls"][0]
    raw_path = (
        directory / "api_audit" / (hashlib.sha256(call["trace_id"].encode()).hexdigest() + ".json")
    )
    raw = frozen._read(raw_path)
    value = json.loads(raw["response"]["choices"][0]["message"]["content"])
    value["evidence_ids"] = [value["evidence_ids"][0][:-1]]
    raw["response"]["choices"][0]["message"]["content"] = json.dumps(value)
    write(raw_path, raw)
    for qid in ids[45:]:
        for method in frozen.study.METHODS:
            name = f"{qid}_{method}.json"
            if qid == failed_id and method == "base":
                continue
            removed = frozen._read(directory / name)
            for removed_call in removed["calls"]:
                (
                    directory
                    / "api_audit"
                    / (hashlib.sha256(removed_call["trace_id"].encode()).hexdigest() + ".json")
                ).unlink()
            (directory / name).unlink()
            (directory / "raw_execution" / name).unlink()
    calls = []
    terminal, hashes = [], {}
    for qid in ids[25:]:
        row = {"question_id": qid, "methods": {}}
        for method in frozen.study.METHODS:
            name = f"{qid}_{method}.json"
            path = directory / name
            if path.exists():
                current = frozen._read(path)
                hashes[name] = frozen._sha(path)
                calls.extend(current["calls"])
                row["methods"][method] = {"status": current["status"], "path": name}
            else:
                row["methods"][method] = {"status": "not_attempted", "path": None}
        terminal.append(row)
    summary.update(
        status="failed",
        failure_type="ValueError",
        completed_questions=20,
        terminal=terminal,
        prediction_sha256=hashes,
        **frozen._totals(calls),
    )
    write(summary_path, summary)
    ledger = frozen._read(directory / "final_budget.json")
    ledger.update(**frozen._totals(calls), calls=calls, block_reason="experiment_execution_failure")
    write(directory / "final_budget.json", ledger)
    events_path = directory / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    events = [event for event in events if event.get("question_id") in ids[25:45]]
    events += [
        {"kind": "question_start", "question_id": failed_id, "arm": "setup"},
        {
            "kind": "batch_stopped",
            "question_id": failed_id,
            "arm": "base",
            "error_type": "ValueError",
        },
    ]
    events_path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    for current_directory in directories:
        current_ledger = frozen._read(current_directory / "final_budget.json")
        for index, current_call in enumerate(current_ledger["calls"]):
            prior_calls = current_ledger["calls"][:index]
            raw = frozen._read(
                current_directory
                / "api_audit"
                / (hashlib.sha256(current_call["trace_id"].encode()).hexdigest() + ".json")
            )
            write(
                current_directory / "request_journal" / f"{index:04d}_intent.json",
                {
                    "trace_id": current_call["trace_id"],
                    "prompt_version": current_call["prompt_version"],
                    "request_fingerprint": frozen.fingerprint(raw["request"]["messages"]),
                    "potential_reserved_cny": current_call["reserved_cny"],
                    "prior_reserved_cny": sum(c["reserved_cny"] for c in prior_calls),
                    "status": "pending_no_automatic_retry",
                },
            )
            prefix = current_ledger["calls"][: index + 1]
            after = {
                **current_ledger,
                **frozen._totals(prefix),
                "calls": prefix,
                "block_reason": None,
            }
            write(current_directory / "request_journal" / f"{index:04d}_after.json", after)
    monkeypatch.setattr(auditor, "FAILED_ID", failed_id)
    monkeypatch.setattr(auditor, "SOURCE_SHA256", frozen.source_snapshot(root)["sha256"])
    monkeypatch.setattr(
        auditor, "SUMMARY_PINS", tuple(frozen._sha(d / "SUMMARY.json") for d in directories)
    )
    monkeypatch.setattr(frozen, "_load_gold", lambda *args: pytest.fail("no labels allowed"))
    return root, directories, manifest


def test_terminal_state_no_gold_and_no_answers(interrupted):
    root, _, manifest = interrupted
    summary, rows, inputs = auditor.collect(root)
    assert summary["complete_paired_questions"] == 45
    assert summary["completed_method_reports"] == 225
    assert summary["failed_method_reports"] == 1
    assert summary["not_attempted_method_paths"] == 24
    assert summary["not_started_question_ids"] == manifest["question_ids"][46:]
    assert summary["gold_loaded"] is False and summary["api_calls"] == 0
    assert summary["accuracy"] is None and summary["reader_answers_exposed"] is False
    assert summary["continuation_authorized"] is False and summary["score_gate_amended"] is False
    assert summary["failed_reader"]["id_repaired"] is False
    assert summary["totals_including_failed_http"]["api_requests"] == 407
    assert sum(m["api_requests"] for m in summary["methods"].values()) == 406

    def inspect(node):
        if isinstance(node, dict):
            assert not set(node) & {
                "answer",
                "answers",
                "reference_answer",
                "raw_content",
                "evidence",
            }
            for child in node.values():
                inspect(child)
        elif isinstance(node, list):
            for child in node:
                inspect(child)

    inspect([summary, rows])
    assert len(rows) == 45 and len(inputs) > 225
    assert not (root / auditor.OUTPUT).exists()


def test_write_is_exclusive_preserves_source_and_original_artifacts(interrupted):
    root, directories, _ = interrupted
    frozen = auditor.frozen
    before_source = frozen.source_snapshot(root)
    before = {str(p): frozen._sha(p) for d in directories for p in d.rglob("*") if p.is_file()}
    result = auditor.run(root, write=True)
    assert result["status"] == "interrupted_stage_terminal_audit_passed"
    assert frozen.source_snapshot(root) == before_source
    assert before == {
        str(p): frozen._sha(p) for d in directories for p in d.rglob("*") if p.is_file()
    }
    assert (root / auditor.OUTPUT / "terminal_audit_frozen.json").is_file()
    with pytest.raises(FileExistsError):
        auditor.run(root, write=True)


@pytest.mark.parametrize(
    "defect", ["source", "summary", "raw", "pending", "extra_claim", "fail_not_citation"]
)
def test_terminal_audit_fails_closed(interrupted, defect):
    root, directories, manifest = interrupted
    frozen = auditor.frozen
    last = directories[-1]
    if defect == "source":
        (root / "src/example.py").write_text("# drift\n", encoding="utf-8")
    elif defect == "summary":
        value = frozen._read(last / "SUMMARY.json")
        value["status"] = "completed"
        write(last / "SUMMARY.json", value)
    elif defect == "raw":
        path = last / "raw_execution" / f"{manifest['question_ids'][25]}_base.json"
        value = frozen._read(path)
        value["question_id"] = "wrong"
        write(path, value)
    elif defect == "pending":
        write(last / "request_journal/9999_intent.json", {"trace_id": "stray"})
    elif defect == "extra_claim":
        write(
            root / "runs/replacement.claim.json", {"question_ids": [manifest["question_ids"][46]]}
        )
    else:
        report = frozen._read(last / f"{auditor.FAILED_ID}_base.json")
        call = report["calls"][0]
        path = (
            last / "api_audit" / (hashlib.sha256(call["trace_id"].encode()).hexdigest() + ".json")
        )
        value = frozen._read(path)
        response = json.loads(value["response"]["choices"][0]["message"]["content"])
        response["evidence_ids"] = [report["episode"]["evidence"][0]["evidence_id"]]
        value["response"]["choices"][0]["message"]["content"] = json.dumps(response)
        write(path, value)
    with pytest.raises(ValueError):
        auditor.collect(root)
    assert not (root / auditor.OUTPUT).exists()


def test_original_full250_scorer_gate_stays_closed(interrupted):
    root, _, _ = interrupted
    with pytest.raises(ValueError, match="all 250"):
        auditor.frozen.run(root, score=True)
