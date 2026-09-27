"""纯合成数据；测试不读题库、不调用 API，也不改变任何真实 run。"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "trajectory_stats",
    Path(__file__).resolve().parents[1] / "scripts" / "audits" / "trajectory_stats.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def report():
    events, docs, sources = [], [], []
    queries = ["Alpha  Beta", "alpha beta", "alpha beta"]
    for number, query in enumerate(queries, 1):
        doc = {"doc_id": f"doc-{number}", "title": "same title", "text": "sample"}
        docs.append(doc)
        events.append(
            {
                "kind": "retrieval",
                "event_index": len(events),
                "round": number,
                "queries": [query],
                "documents": [[doc]],
            }
        )
        selected = (
            []
            if number in (1, 3)
            else [
                {
                    "doc_id": doc["doc_id"],
                    "sentence_id": 1,
                    "text": "sample",
                    "title": "same title",
                }
            ]
        )
        sources.extend(selected)
        events.append(
            {
                "kind": "extraction",
                "event_index": len(events),
                "round": number,
                "sources": selected,
            }
        )
    base = {
        "status": "completed",
        "calls": [],
        "feedback": {"answer_em": 0.0, "answer_f1": 0.5, "unscorable_annotation": False},
    }
    s2g = copy.deepcopy(base)
    s2g["feedback"].update(answer_em=1.0, answer_f1=1.0)
    s2g["result"] = {
        "events": events,
        "retrieval_rounds": 3,
        "stop_reason": "max_turns",
        "sources": sources,
        "retrieved_documents": docs,
        "api_calls": 0,
    }
    return {
        "question_id": "synthetic",
        "offset": 0,
        "scoring_status": "completed",
        "complete_pair": True,
        "arms": {audit.BASE: base, audit.S2G: s2g},
    }


def test_repeat_does_not_imply_repeated_docs_and_terminal_zero_is_not_continue():
    row = audit.inspect_trace(report(), "synthetic_batch")
    assert row["exact_repeat_rounds"] == 1
    assert row["normalized_repeat_rounds"] == 2
    assert row["normalized_repeat_rounds_with_new_docs"] == 2
    assert row["zero_new_retention_continue_rounds"] == 1
    assert [r["new_document_ids_count"] for r in row["rounds"]] == [1, 1, 1]
    assert not row["rounds"][-1]["zero_new_pointers_but_continued"]


def test_bad_annotation_keeps_behavior_but_not_oracle():
    sample = report()
    for arm in sample["arms"].values():
        arm["feedback"] = {"unscorable_annotation": True, "answer_em": None, "answer_f1": None}
    row = audit.inspect_trace(sample, "synthetic_batch")
    assert row["annotation_invalid"] and not row["scorable_pair"]
    assert row["retrieval_rounds"] == 3
    assert audit.metric_upper_bounds([row])["n"] == 0


def test_pointer_outside_passed_documents_rejected():
    sample = report()
    sample["arms"][audit.S2G]["result"]["events"][3]["sources"][0]["doc_id"] = "not-passed"
    with pytest.raises(ValueError, match="not passed"):
        audit.inspect_trace(sample, "synthetic_batch")


def test_unknown_cost_is_not_zero():
    assert audit.totals([{"input_tokens": 4}])["estimated_actual_cny"] is None
    assert audit.totals([{"input_tokens": 4}])["input_tokens"] == 4


def test_oracle_is_above_constant_but_not_implemented_router():
    one = audit.inspect_trace(report(), "synthetic_batch")
    two = copy.deepcopy(one)
    two["metrics"][audit.BASE] = {"answer_em": 1.0, "answer_f1": 1.0}
    two["metrics"][audit.S2G] = {"answer_em": 0.0, "answer_f1": 0.5}
    upper = audit.metric_upper_bounds([one, two])
    assert upper["answer_em"]["best_constant"] == 0.5
    assert upper["answer_em"]["posthoc_oracle"] == 1.0
    assert upper["answer_em"]["oracle_minus_best_constant"] == 0.5
    assert upper["paired_em"] == {"repairs_base0_s2g1": 1, "harms_base1_s2g0": 1}


def test_unsealed_and_claim_are_not_read(tmp_path):
    (tmp_path / (audit.PREFIX + "0000_0025")).mkdir()
    (tmp_path / (audit.PREFIX + "0000_0025.claim.json")).touch()
    summary, rows = audit.analyze(tmp_path)
    assert not rows
    assert summary["unsealed_skipped"] == [audit.PREFIX + "0000_0025"]


def test_elapsed_keeps_unknown_and_defines_percentile():
    value = audit.elapsed_stats([float(n) for n in range(1, 11)] + [None])
    assert value["records"] == 11
    assert value["known_records"] == 10 and value["unknown_records"] == 1
    assert value["mean_seconds"] == value["median_seconds"] == 5.5
    assert value["p90_seconds_nearest_rank"] == 9
    assert value["known_sum_seconds"] == 55


def synthetic_plan(monkeypatch):
    files = {
        f"src/growrag/experiments/{name}.py": {"sha256": audit.digest("pass\n"), "text": "pass\n"}
        for name in ("s2g_author_api", "shared_s2g_corpus", "api_client", "run_s2g_author_pilot")
    }
    snapshot = {"files": files, "sha256": audit.canonical_sha(files)}
    monkeypatch.setattr(audit, "REVIEWED_SOURCE_SNAPSHOTS", {snapshot["sha256"]})
    plan = {key: "synthetic" for key in audit.PLAN_FIELDS}
    plan.update(
        manifest_sha256=audit.EXPECTED_MANIFEST,
        source_sha256=snapshot["sha256"],
        backend_output_caps={"judge": 768, "extract": 128, "answer": 1024},
        answer_length_policy="synthetic",
        arms=[audit.BASE, audit.S2G],
        no_training_no_memory_updates=True,
        official_dev_test_used=False,
        price_input_cny_per_million=0.2,
        price_output_cny_per_million=0.8,
        author={
            "upstream_commit": "synthetic",
            "source_sha256": {},
            "author_prompt_sha256": {},
            "sentence_splitter": "regex",
            "remove_repeat_docs": True,
            "dedup_key": "document_id",
            "trained_author_judge_used": False,
            "gap_profile": "paper_k1",
            "retrieval_backend": "mock",
            "backend_generation_settings": "mock",
            "gold_provided_to_author_loop": False,
        },
        runtime_metadata={
            key: "synthetic"
            for key in (
                "manifest_sha256",
                "corpus_sha256",
                "document_count",
                "question_count",
                "retrieval_config",
                "context_fingerprint",
            )
        },
    )
    return plan, snapshot


def seal(directory, plan, snapshot, reports, calls):
    directory.mkdir()
    content = {
        "launch_plan.json": plan,
        "source_snapshot.json": snapshot,
        "reports.json": reports,
        "summary.json": {
            "complete_pairs": sum(r["complete_pair"] for r in reports),
            "started_without_report": 0,
        },
        "final_budget.json": {"calls": calls, "prior_reserved_cny": 999},
        "events.jsonl": {"kind": "exit", "status": "completed"},
    }
    for name, value in content.items():
        (directory / name).write_text(json.dumps(value), encoding="utf-8")


def test_method_provenance_missing_source_corruption_and_changed_caps(monkeypatch):
    plan, snapshot = synthetic_plan(monkeypatch)
    original = audit.method_signature(plan, snapshot)
    changed = copy.deepcopy(plan)
    changed["backend_output_caps"]["judge"] = 256
    assert audit.method_signature(changed, snapshot) != original
    del changed["author"]["sentence_splitter"]
    with pytest.raises(ValueError, match="missing method provenance"):
        audit.method_signature(changed, snapshot)
    snapshot["files"][next(iter(snapshot["files"]))]["text"] += "# tampered\n"
    with pytest.raises(ValueError, match="aggregate hash mismatch"):
        audit.method_signature(plan, snapshot)
    plan["source_sha256"] = "unreviewed"
    with pytest.raises(ValueError, match="unreviewed source"):
        audit.method_signature(plan, snapshot)


def test_sealed_failed_and_resumed_batches_cost_latency_and_mixed_guard(tmp_path, monkeypatch):
    plan, snapshot = synthetic_plan(monkeypatch)
    good = report()
    for a, seconds in ((audit.BASE, 2), (audit.S2G, 3)):
        good["arms"][a]["elapsed_seconds"] = seconds
    failed = copy.deepcopy(good)
    failed.update(question_id="failed", offset=1, complete_pair=False)
    failed["arms"][audit.S2G].update(status="failed", elapsed_seconds=90)
    call = {
        "trace_id": "only-paid-attempt",
        "status": "completed",
        "api_requests": 1,
        "input_tokens": 8,
        "output_tokens": 2,
        "estimated_actual_cny": 0.04,
    }
    failed["arms"][audit.S2G]["calls"] = [call]
    first = tmp_path / (audit.PREFIX + "0000_0025")
    seal(first, plan, snapshot, [good, failed], [call])
    resumed = copy.deepcopy(good)
    resumed.update(question_id="new-unstarted", offset=2)
    second = tmp_path / (audit.PREFIX + "0002_0025")
    seal(second, plan, snapshot, [resumed], [])
    summary, rows = audit.analyze(tmp_path)
    assert len(rows) == 2 and len(summary["incomplete_pairs"]) == 1
    assert summary["posthoc_metric_upper_bounds"]["n"] == 2
    assert (
        summary["all_sealed_ledger_calls_including_incomplete_or_unreported"][
            "estimated_actual_cny"
        ]
        == 0.04
    )
    latency = summary["latency"]
    assert latency["complete_pair_arms"][audit.S2G]["mean_seconds"] == 3
    assert latency["failed_reported_arms_only"][audit.S2G]["known_sum_seconds"] == 90
    assert (
        latency["all_reported_attempted_arms_including_failed"][audit.S2G]["known_sum_seconds"]
        == 96
    )
    plan["author"]["sentence_splitter"] = "different"
    (second / "launch_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="mixed method provenance"):
        audit.analyze(tmp_path)
