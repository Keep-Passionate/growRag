"""Missing-outcome sensitivity bounds for the frozen shared500 experiment.

Read ONLY aggregate analysis.json and the preregistered public manifest.json.
Never read gold, run a model, repair an answer, or relabel a failed question.
The intervals are deterministic worst/best-case bounds, NOT confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

MANIFEST_SHA = "2fb117c03b64be3ab12398a297c82ce3315600c14e33f5dcd56c5c6c5c59e064"
INVALID_IDS = frozenset({"5ab273ee5542997061209606"})
ARMS = ("BASE1_AUTHOR_READER", "S2G_AUTHOR_API4")
METRICS = ("answer_em", "answer_f1")
STATUSES = frozenset(
    {
        "scored_pair",
        "execution_failed",
        "incomplete_pair",
        "unscorable_annotation",
        "scoring_failed",
        "complete_unscored",
        "unclaimed",
        "started_without_report",
        "claimed_no_execution_record",
    }
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _score(value, metric):
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1,
        "score must be a finite number in [0, 1]",
    )
    _require(metric != "answer_em" or value in (0, 1), "EM must be 0 or 1")
    return float(value)


def compute_bounds(analysis, manifest):
    """Pure function. Input-file hash verification is performed by load_and_compute.

    Even a failed execution of the preregistered invalid question is excluded
    from the 499 metric cohort, but remains in the 500 execution/cost cohort.
    No observed single-arm answer is scored when the paired score is missing.
    """
    _require(
        analysis.get("schema_version") == "growrag-shared-s2g-series-analysis-v1"
        and analysis.get("series") == "500_v1"
        and analysis.get("manifest_sha256") == MANIFEST_SHA,
        "unexpected aggregate schema, series, or frozen manifest identity",
    )
    ids = manifest.get("question_ids")
    _require(
        isinstance(ids, list)
        and len(ids) == 500
        and all(isinstance(qid, str) and qid for qid in ids)
        and len(set(ids)) == 500,
        "manifest requires 500 unique question IDs",
    )
    _require(
        manifest.get("role") == "development" and manifest.get("official_split") == "train",
        "not the frozen train-development scope",
    )
    quality = manifest.get("annotation_summary") or {}
    invalid = quality.get("invalid_question_ids")
    _require(
        isinstance(invalid, list)
        and len(invalid) == len(set(invalid))
        and set(invalid) == INVALID_IDS
        and INVALID_IDS <= set(ids)
        and ids[404] in INVALID_IDS
        and type(quality.get("invalid")) is int
        and quality["invalid"] == 1
        and type(quality.get("valid")) is int
        and quality["valid"] == 499
        and type(quality.get("replacement_count")) is int
        and quality["replacement_count"] == 0,
        "preregistered annotation exclusion changed or is inconsistent",
    )
    rows = analysis.get("question_index")
    _require(
        isinstance(rows, list)
        and len(rows) == 500
        and [row.get("question_id") for row in rows] == ids
        and all(type(row.get("offset")) is int and row["offset"] == i for i, row in enumerate(rows))
        and analysis.get("planned_questions") == 500,
        "aggregate question cohort/order/offset differs from frozen 500",
    )
    statuses = Counter(row.get("analysis_status") for row in rows)
    _require(set(statuses) <= STATUSES, "unknown analysis status")
    _require(
        analysis.get("question_status_counts") == dict(statuses),
        "aggregate status counts disagree with its question index",
    )
    scored, missing = [], []
    for row in rows:
        qid, status = row["question_id"], row["analysis_status"]
        excluded = qid in INVALID_IDS
        arms = row.get("arms") or {}
        for arm in arms.values():
            bad_flag = arm.get("unscorable_annotation", False)
            label_status = arm.get("annotation_status")
            _require(type(bad_flag) is bool, "annotation flag must be boolean")
            _require(label_status in (None, "valid", "invalid"), "unknown annotation status")
            _require(
                not (bad_flag or label_status == "invalid") or excluded,
                "unregistered bad annotation: do not silently expand exclusions",
            )
            _require(
                not (excluded and label_status == "valid"),
                "preregistered invalid question cannot acquire a valid scoring label",
            )
        _require(
            status != "unscorable_annotation" or excluded,
            "unregistered unscorable question",
        )
        if status == "scored_pair":
            _require(not excluded, "preregistered invalid question was scored")
            _require(set(arms) == set(ARMS), "scored pair lacks both arms")
            for arm in ARMS:
                _require(arms[arm].get("status") == "completed", "scored arm did not complete")
                feedback = arms[arm].get("feedback") or {}
                for metric in METRICS:
                    _score(feedback.get(metric), metric)
            scored.append(row)
        else:
            # Atomic paired scoring is part of the frozen runner contract.
            _require(
                all(
                    (arm.get("feedback") or {}).get(metric) is None
                    for arm in arms.values()
                    for metric in METRICS
                ),
                "unscored row contains partial scores; require audit, not silent use",
            )
            if not excluded:
                missing.append(row)
    n, m, denominator = len(scored), len(missing), 499
    _require(n + m == denominator, "valid-cohort partition failed")
    _require(
        analysis.get("paired_scored_questions") == n
        and (analysis.get("effect") or {}).get("paired_scored_n") == n,
        "aggregate scored-pair denominator disagrees",
    )
    metrics = {}
    for metric in METRICS:
        sums = {
            arm: math.fsum(row["arms"][arm]["feedback"][metric] for row in scored) for arm in ARMS
        }
        delta = math.fsum(
            row["arms"][ARMS[1]]["feedback"][metric] - row["arms"][ARMS[0]]["feedback"][metric]
            for row in scored
        )
        metrics[metric] = {
            "observed_paired_n": n,
            "missing_valid_pairs": m,
            "valid_cohort_denominator": denominator,
            "arms": {
                arm: {
                    "known_score_sum": sums[arm],
                    "observed_paired_mean": sums[arm] / n if n else None,
                    "lower": sums[arm] / denominator,
                    "upper": (sums[arm] + m) / denominator,
                }
                for arm in ARMS
            },
            "s2g_minus_base": {
                "known_paired_delta_sum": delta,
                "observed_paired_mean_delta": delta / n if n else None,
                "lower": (delta - m) / denominator,
                "upper": (delta + m) / denominator,
            },
        }
    return {
        "schema_version": "growrag-shared500-missing-outcome-bounds-v1",
        "method": "bounded missing outcomes sensitivity; not imputation or confidence interval",
        "manifest_sha256": MANIFEST_SHA,
        "planned_execution_cost_cohort_n": 500,
        "preregistered_invalid_n": 1,
        "preregistered_invalid_question_ids": sorted(INVALID_IDS),
        "valid_metric_cohort_n": denominator,
        "observed_paired_n": n,
        "missing_valid_pairs": m,
        "all_planned_status_counts": dict(statuses),
        "missing_valid_status_counts": dict(Counter(r["analysis_status"] for r in missing)),
        "missing_valid_question_ids": [r["question_id"] for r in missing],
        "metrics": metrics,
        "source_sealing_independently_verified": False,
        "notices": [
            "EM/F1 point estimates remain restricted to the identical observed paired cohort.",
            "Each unobserved valid-question score may range over [0,1]; "
            "its paired difference over [-1,1].",
            "Endpoints are hypothetical possibilities, not actual scores assigned to failures.",
            "The preregistered invalid item is outside both observed and missing metric cohorts; "
            "no corrected gold is invented.",
            "The invalid item and all failures remain in the 500-question "
            "execution and cost accounting.",
            "No gold is loaded and no singly completed arm is newly scored.",
            "Bounds do not measure sampling uncertainty, "
            "statistical significance, or generalization.",
            "Unfinished/unclaimed valid questions also widen the bounds; "
            "final reporting still requires a separate sealing audit.",
            "Cost and source-seal verification belong to the independent series audit; "
            "this tool does not recompute them.",
        ],
    }


def load_and_compute(analysis_path, manifest_path):
    """Exactly two local reads; no gold/corpus/index/API access."""
    analysis_path, manifest_path = Path(analysis_path), Path(manifest_path)
    _require(analysis_path.name == "analysis.json", "expected aggregate analysis.json")
    _require(manifest_path.name == "manifest.json", "expected public manifest.json")
    manifest_raw = manifest_path.read_bytes()
    _require(hashlib.sha256(manifest_raw).hexdigest() == MANIFEST_SHA, "manifest file SHA mismatch")
    analysis_raw = analysis_path.read_bytes()
    result = compute_bounds(json.loads(analysis_raw), json.loads(manifest_raw))
    result["input_sha256"] = {
        "manifest.json": MANIFEST_SHA,
        "analysis.json": hashlib.sha256(analysis_raw).hexdigest(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            load_and_compute(args.analysis, args.manifest),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
