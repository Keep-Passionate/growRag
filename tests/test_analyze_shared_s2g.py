"""No APIs/gold needed: synthetic durable run artifacts cover aggregation invariants."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from growrag.experiments import analyze_shared_s2g as analysis

ARMS = analysis.ARMS


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def context(tmp_path):
    ids = [f"q{i}" for i in range(32)]
    manifest = {
        "question_ids": ids,
        "question_types": {q: "bridge" if i % 2 == 0 else "comparison" for i, q in enumerate(ids)},
        "official_split": "train",
        "role": "development",
    }
    path = tmp_path / "data" / "manifest.json"
    dump(path, manifest)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  manifest.json\n")
    runs = tmp_path / "runs"
    runs.mkdir()
    return path, runs, digest, manifest


def call(run, qid, arm, *, failed=False, unknown=False):
    return {
        "trace_id": f"{run}/{qid}/{arm}/01-answer",
        "status": "failed" if failed else "completed",
        "api_requests": 1,
        "input_tokens": None if unknown else 10,
        "output_tokens": None if unknown else 5,
        "reserved_cny": 0.1,
        "estimated_actual_cny": None if unknown else 0.01,
    }


def arm_row(run, qid, arm, em, *, status="completed", invalid=False):
    calls = [] if status == "not_executed" else [call(run, qid, arm, failed=status == "failed")]
    feedback = (
        None
        if status != "completed"
        else {
            "answer_em": None if invalid else em,
            "answer_f1": None if invalid else em,
            "raw_support_recall": None if invalid else 1.0,
            "retained_support_recall": None if invalid else (1.0 if arm == ARMS[0] else 0.5),
            "unscorable_annotation": invalid,
            "annotation_status": "invalid" if invalid else "valid",
        }
    )
    return {
        "status": status,
        "calls": calls,
        "feedback": feedback,
        "result": {
            "answer": "A",
            "stop_reason": "sufficient",
            "retrieval_rounds": 1 if arm == ARMS[0] else 2,
        },
        "elapsed_seconds": 1.0,
    }


def add_batch(
    context, start, results, *, count=None, series="32_v3", protocol="protocol-v3", profile="v5"
):
    _, runs, digest, manifest = context
    count = len(results) if count is None else count
    run = f"2026-09-27_s2g_shared{series}_{start:04d}_{start + count:04d}"
    directory = runs / run
    launch = {
        "run_id": run,
        "series": series,
        "protocol": protocol,
        "manifest_sha256": digest,
        "generation_profile": profile,
        "model": "pinned-qwen",
        "backend_output_caps": {"answer": 1024},
        "max_retrieval_rounds": 4,
        "top_docs": 6,
        "gap_profile": "paper_k1",
        "arms": list(ARMS),
        "source_sha256": "source",
        "start": start,
        "count": count,
        "question_ids": manifest["question_ids"][start : start + count],
        "historical_budget": {"estimated_actual_cny": 9999},
    }
    rows = []
    for offset, (base, s2g, mode) in enumerate(results, start):
        qid = manifest["question_ids"][offset]
        invalid = mode == "invalid"
        arms = {
            a: arm_row(run, qid, a, em, invalid=invalid)
            for a, em in zip(ARMS, (base, s2g), strict=True)
        }
        if mode == "failed":
            arms[ARMS[0]] = arm_row(run, qid, ARMS[0], 0, status="failed")
            arms[ARMS[1]] = arm_row(run, qid, ARMS[1], 0, status="not_executed")
        rows.append(
            {
                "question_id": qid,
                "question_type": manifest["question_types"][qid],
                "offset": offset,
                "arms": arms,
                "complete_pair": mode != "failed",
                "scoring_status": "completed",
            }
        )
    dump(directory / "launch_plan.json", launch)
    dump(directory / "reports.json", rows)
    dump(
        directory / "final_budget.json",
        {"calls": [c for r in rows for a in ARMS for c in r["arms"][a]["calls"]]},
    )
    return directory


def analyze(context, **kwargs):
    path, runs, digest, _ = context
    return analysis.analyze_series(
        path,
        runs,
        series="32_v3",
        protocol="protocol-v3",
        generation_profile="v5",
        expected_manifest_sha256=digest,
        **kwargs,
    )


def test_paired_cohort_denominators_repairs_harms_types_and_all_cost(context):
    add_batch(
        context,
        0,
        [(0, 1, "ok"), (1, 0, "ok"), (1, 1, "ok"), (0, 0, "invalid"), (0, 0, "failed")],
        count=6,
    )
    report = analyze(context)
    assert report["planned_questions"] == 32 and report["claimed_questions"] == 6
    assert report["paired_scored_questions"] == 3
    assert report["question_status_counts"] == {
        "scored_pair": 3,
        "unscorable_annotation": 1,
        "execution_failed": 1,
        "claimed_no_execution_record": 1,
        "unclaimed": 26,
    }
    for arm in ARMS:
        assert report["effect"]["arms"][arm]["answer_em"] == {"mean": 2 / 3, "n": 3}
    assert report["effect"]["paired_em"]["repairs"] == report["effect"]["paired_em"]["harms"] == 1
    assert report["effect"]["paired_em"]["repair_rate_among_base_wrong"] == 1
    assert report["effect"]["paired_em"]["harm_rate_among_base_correct"] == 0.5
    assert report["by_question_type"]["bridge"]["paired_scored_n"] == 2
    assert report["all_attempt_cost"]["recorded_attempts"] == 9
    assert report["all_attempt_cost"]["failed_attempts"] == 1
    assert report["all_attempt_cost"]["known_estimated_cny"] == pytest.approx(0.09)
    assert report["diagnostic_indices"]["paired_harm"]["examples"][0]["question_id"] == "q1"
    assert report["diagnostic_indices"]["s2g_stopped_sufficient_but_em_wrong"]["count"] == 1


def test_multiple_batches_and_other_series_do_not_mix(context):
    add_batch(context, 0, [(0, 1, "ok")])
    add_batch(context, 1, [(1, 1, "ok")])
    add_batch(context, 0, [(0, 0, "failed")], series="32_v1", protocol="old", profile="old")
    report = analyze(context)
    assert report["paired_scored_questions"] == 2
    assert len(report["ignored_other_series"]) == 1
    assert report["all_attempt_cost"]["recorded_attempts"] == 4


@pytest.mark.parametrize(
    "field,value", [("protocol", "old"), ("generation_profile", "old"), ("manifest_sha256", "bad")]
)
def test_same_series_rejects_mixed_protocol_dataset_and_profile(context, field, value):
    directory = add_batch(context, 0, [(0, 1, "ok")])
    path = directory / "launch_plan.json"
    launch = json.loads(path.read_text())
    launch[field] = value
    dump(path, launch)
    with pytest.raises(ValueError, match="mixes"):
        analyze(context)


def test_same_series_rejects_duplicate_claimed_questions_even_without_reports(context):
    add_batch(context, 0, [(0, 1, "ok")], count=2)
    add_batch(context, 1, [(1, 1, "ok")], count=2)
    with pytest.raises(ValueError, match="repeated question"):
        analyze(context)


def test_interrupted_unreported_calls_still_count_and_unknown_cost_is_not_zero(context):
    directory = add_batch(context, 0, [], count=1)
    pending = call(directory.name, "q0", ARMS[0], failed=True, unknown=True)
    dump(directory / "final_budget.json", {"calls": [pending]})
    report = analyze(context)
    assert report["question_status_counts"]["started_without_report"] == 1
    assert report["all_attempt_cost"]["unknown_cost_attempts"] == 1
    assert report["all_attempt_cost"]["unknown_input_token_attempts"] == 1
    assert report["all_attempt_cost"]["known_input_tokens"] == 0
    assert report["execution"][ARMS[0]]["all_attempt_cost"]["recorded_attempts"] == 1


def test_missing_final_ledger_is_explicit_incomplete_lower_bound(context):
    directory = add_batch(context, 0, [(1, 1, "ok")])
    (directory / "final_budget.json").unlink()
    report = analyze(context)
    assert not report["cost_completeness"]["all_matching_batches_have_final_budget"]
    assert report["all_attempt_cost"]["recorded_attempts"] == 2


@pytest.mark.parametrize("change", ["omit", "duplicate", "alter"])
def test_final_ledger_must_reconcile_with_report_attempts(context, change):
    directory = add_batch(context, 0, [(1, 1, "ok")])
    path = directory / "final_budget.json"
    ledger = json.loads(path.read_text())
    if change == "omit":
        ledger["calls"].pop()
    elif change == "duplicate":
        ledger["calls"].append(copy.deepcopy(ledger["calls"][0]))
    else:
        ledger["calls"][0]["input_tokens"] = 1000
    dump(path, ledger)
    with pytest.raises(ValueError, match="ledger"):
        analyze(context)


def test_diagnostic_examples_are_capped_but_counts_are_not(context):
    add_batch(context, 0, [(0, 1, "ok")] * 3)
    report = analyze(context, max_examples=1)
    assert report["diagnostic_indices"]["paired_repair"]["count"] == 3
    assert len(report["diagnostic_indices"]["paired_repair"]["examples"]) == 1


def test_output_is_new_only_and_no_gold_corpus_reads(context, monkeypatch, tmp_path):
    add_batch(context, 0, [(0, 1, "ok")])
    original = Path.read_bytes

    def guarded(path):
        assert path.name not in {"gold.jsonl", "corpus.jsonl", "runtime_questions.jsonl"}
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    report = analyze(context)
    output = tmp_path / "analysis"
    analysis.write_analysis(report, output)
    assert (output / "analysis.json").exists()
    markdown = (output / "analysis.md").read_text(encoding="utf-8")
    assert "32" in markdown and "未" in markdown
    with pytest.raises(FileExistsError):
        analysis.write_analysis(report, output)


def test_fixed_series_size_rejects_32_manifest_as_500(context):
    path, runs, digest, _ = context
    with pytest.raises(ValueError, match="series size"):
        analysis.analyze_series(
            path,
            runs,
            series="500_v1",
            protocol="P",
            generation_profile="G",
            expected_manifest_sha256=digest,
        )


def test_unknown_api_request_count_and_zero_request_attempt_cost():
    report = analysis._cost(
        [
            {"api_requests": None, "status": "pending", "reserved_cny": 0.1},
            {"api_requests": 0, "status": "failed", "reserved_cny": 0.1},
        ]
    )
    assert report["unknown_api_request_attempts"] == 1
    assert report["unknown_cost_attempts"] == 1
    assert report["known_api_requests"] == 0
