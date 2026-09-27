"""Synthetic aggregate/manifest data only; no private gold or live API access."""

import copy
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "missing_bounds",
    Path(__file__).resolve().parents[1] / "scripts" / "audits" / "missing_bounds.py",
)
BOUNDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOUNDS)


def fixtures():
    ids = [f"synthetic-{i}" for i in range(500)]
    ids[404] = next(iter(BOUNDS.INVALID_IDS))
    manifest = {
        "question_ids": ids,
        "role": "development",
        "official_split": "train",
        "annotation_summary": {
            "invalid": 1,
            "valid": 499,
            "invalid_question_ids": [ids[404]],
            "replacement_count": 0,
        },
    }
    rows = [
        {"question_id": qid, "offset": i, "analysis_status": "unclaimed", "arms": {}}
        for i, qid in enumerate(ids)
    ]
    for i, (base, s2g) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        rows[i].update(
            analysis_status="scored_pair",
            arms={
                arm: {
                    "status": "completed",
                    "unscorable_annotation": False,
                    "annotation_status": "valid",
                    "feedback": {"answer_em": em, "answer_f1": em},
                }
                for arm, em in zip(BOUNDS.ARMS, (base, s2g), strict=True)
            },
        )
    rows[404]["analysis_status"] = "unscorable_annotation"
    analysis = {
        "schema_version": "growrag-shared-s2g-series-analysis-v1",
        "series": "500_v1",
        "manifest_sha256": BOUNDS.MANIFEST_SHA,
        "planned_questions": 500,
        "paired_scored_questions": 2,
        "effect": {"paired_scored_n": 2},
        "question_index": rows,
    }
    refresh(analysis)
    return analysis, manifest


def refresh(analysis):
    rows = analysis["question_index"]
    analysis["question_status_counts"] = dict(Counter(r["analysis_status"] for r in rows))
    n = sum(r["analysis_status"] == "scored_pair" for r in rows)
    analysis["paired_scored_questions"] = analysis["effect"]["paired_scored_n"] = n


def test_valid499_bounds_and_bad_item_never_enters_missing_denominator():
    analysis, manifest = fixtures()
    result = BOUNDS.compute_bounds(analysis, manifest)
    assert result["planned_execution_cost_cohort_n"] == 500
    assert result["valid_metric_cohort_n"] == 499
    assert result["observed_paired_n"] == 2 and result["missing_valid_pairs"] == 497
    assert not set(result["missing_valid_question_ids"]) & BOUNDS.INVALID_IDS
    metric = result["metrics"]["answer_em"]
    assert metric["arms"][BOUNDS.ARMS[0]]["lower"] == pytest.approx(1 / 499)
    assert metric["arms"][BOUNDS.ARMS[0]]["upper"] == pytest.approx(498 / 499)
    assert metric["s2g_minus_base"]["lower"] == pytest.approx(-497 / 499)
    assert metric["s2g_minus_base"]["upper"] == pytest.approx(497 / 499)
    assert "not imputation or confidence interval" in result["method"]


def test_fractional_f1_uses_score_sums_not_em_counts():
    analysis, manifest = fixtures()
    analysis["question_index"][0]["arms"][BOUNDS.ARMS[1]]["feedback"]["answer_f1"] = 0.25
    analysis["question_index"][1]["arms"][BOUNDS.ARMS[0]]["feedback"]["answer_f1"] = 0.4
    value = BOUNDS.compute_bounds(analysis, manifest)["metrics"]["answer_f1"]
    assert value["arms"][BOUNDS.ARMS[0]]["known_score_sum"] == 1.4
    assert value["arms"][BOUNDS.ARMS[1]]["known_score_sum"] == 1.25
    assert value["s2g_minus_base"]["lower"] == pytest.approx((-0.15 - 497) / 499)


def test_full_observation_collapses_bounds_and_zero_observation_is_wide():
    analysis, manifest = fixtures()
    prototype = analysis["question_index"][0]
    for row in analysis["question_index"]:
        if row["question_id"] not in BOUNDS.INVALID_IDS:
            row.update(analysis_status="scored_pair", arms=copy.deepcopy(prototype["arms"]))
    refresh(analysis)
    observed = BOUNDS.compute_bounds(analysis, manifest)["metrics"]["answer_em"]
    assert observed["missing_valid_pairs"] == 0
    assert observed["s2g_minus_base"]["lower"] == observed["s2g_minus_base"]["upper"] == -1
    for row in analysis["question_index"]:
        if row["question_id"] not in BOUNDS.INVALID_IDS:
            row.update(analysis_status="unclaimed", arms={})
    refresh(analysis)
    empty = BOUNDS.compute_bounds(analysis, manifest)["metrics"]["answer_em"]
    assert empty["s2g_minus_base"]["lower"] == -1
    assert empty["s2g_minus_base"]["upper"] == 1
    assert empty["s2g_minus_base"]["observed_paired_mean_delta"] is None


def test_failed_invalid_question_is_quality_excluded_but_execution_counted():
    analysis, manifest = fixtures()
    analysis["question_index"][404]["analysis_status"] = "execution_failed"
    refresh(analysis)
    result = BOUNDS.compute_bounds(analysis, manifest)
    assert result["all_planned_status_counts"]["execution_failed"] == 1
    assert "execution_failed" not in result["missing_valid_status_counts"]
    assert result["missing_valid_pairs"] == 497


@pytest.mark.parametrize("mutation", ["id", "count", "replacement"])
def test_annotation_exclusion_cannot_be_changed_after_observing_results(mutation):
    analysis, manifest = fixtures()
    quality = manifest["annotation_summary"]
    if mutation == "id":
        quality["invalid_question_ids"] = [manifest["question_ids"][0]]
    elif mutation == "count":
        quality["valid"] = 498
    else:
        quality["replacement_count"] = 1
    with pytest.raises(ValueError, match="preregistered"):
        BOUNDS.compute_bounds(analysis, manifest)


def test_unregistered_bad_annotation_requires_audit_not_extra_exclusion():
    analysis, manifest = fixtures()
    analysis["question_index"][2]["arms"] = {BOUNDS.ARMS[0]: {"unscorable_annotation": True}}
    with pytest.raises(ValueError, match="unregistered"):
        BOUNDS.compute_bounds(analysis, manifest)


def test_invalid_question_cannot_be_scored():
    analysis, manifest = fixtures()
    analysis["question_index"][404].update(analysis_status="scored_pair", arms={})
    refresh(analysis)
    with pytest.raises(ValueError, match="invalid question was scored"):
        BOUNDS.compute_bounds(analysis, manifest)


@pytest.mark.parametrize("mutation", ["offset", "duplicate", "denominator"])
def test_wrong_cohort_or_denominator_fails_closed(mutation):
    analysis, manifest = fixtures()
    if mutation == "offset":
        analysis["question_index"][10]["offset"] = 11
    elif mutation == "duplicate":
        analysis["question_index"][10]["question_id"] = manifest["question_ids"][11]
    else:
        analysis["paired_scored_questions"] = 3
    with pytest.raises(ValueError, match="cohort|denominator"):
        BOUNDS.compute_bounds(analysis, manifest)


@pytest.mark.parametrize(
    "metric,value",
    [("answer_em", True), ("answer_em", 0.5), ("answer_f1", float("nan")), ("answer_f1", 1.1)],
)
def test_invalid_scores_rejected(metric, value):
    analysis, manifest = fixtures()
    analysis["question_index"][0]["arms"][BOUNDS.ARMS[0]]["feedback"][metric] = value
    with pytest.raises(ValueError, match="score|EM"):
        BOUNDS.compute_bounds(analysis, manifest)


def test_failed_single_arm_score_is_not_silently_used():
    analysis, manifest = fixtures()
    analysis["question_index"][2].update(
        analysis_status="execution_failed",
        arms={BOUNDS.ARMS[0]: {"status": "completed", "feedback": {"answer_em": 1}}},
    )
    refresh(analysis)
    with pytest.raises(ValueError, match="partial scores"):
        BOUNDS.compute_bounds(analysis, manifest)


def test_loader_rejects_wrong_hash_before_reading_any_analysis(monkeypatch):
    reads = []

    def read(path):
        reads.append(path.name)
        return b"{}"

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError, match="SHA"):
        BOUNDS.load_and_compute(Path("analysis.json"), Path("manifest.json"))
    assert reads == ["manifest.json"]


def test_loader_reads_only_two_declared_inputs_and_does_not_modify_them(monkeypatch):
    analysis, manifest = fixtures()
    manifest_raw = json.dumps(manifest).encode()
    digest = hashlib.sha256(manifest_raw).hexdigest()
    monkeypatch.setattr(BOUNDS, "MANIFEST_SHA", digest)
    analysis["manifest_sha256"] = digest
    before = copy.deepcopy((analysis, manifest))
    files = {"manifest.json": manifest_raw, "analysis.json": json.dumps(analysis).encode()}
    reads = []

    def read(path):
        reads.append(path.name)
        assert path.name in files, "unexpected file such as gold/corpus was accessed"
        return files[path.name]

    monkeypatch.setattr(Path, "read_bytes", read)
    result = BOUNDS.load_and_compute(Path("analysis.json"), Path("manifest.json"))
    assert reads == ["manifest.json", "analysis.json"]
    assert result["input_sha256"]["manifest.json"] == digest
    assert result["valid_metric_cohort_n"] == 499
    assert (analysis, manifest) == before
