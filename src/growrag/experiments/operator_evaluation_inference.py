"""Fixed-policy paired inference for already scored, frozen evaluation records.

本模块不读文件、不访问模型、不更新记忆。每指标先固定七臂共同可评分题集，
十项对比共享同一批题的重采样，避免独立抽样破坏配对关系。题目独立性及缺失
机制并未因此被验证；区间不能解释为模型随机波动或建库抽样波动。
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .operator_evaluation_summary import ARMS, METRICS, summarize_evaluation
from .representation_runner import fingerprint

SCHEMA = "growrag-operator-evaluation-inference-v1"
_SEED = 20260930
_RESAMPLES = 10000
_CHUNK = 256
_ALPHA = 0.05
_COMPARISONS = (
    ("fresh", "base"),
    ("static", "base"),
    *((arm, ref) for arm in ARMS if arm.startswith("memory") for ref in ("fresh", "static")),
)
_WARNINGS = (
    "Seven-arm complete-case subsets may be selected by non-random failures or missing labels; "
    "results do not automatically generalize to excluded questions.",
    "Question resampling assumes independent sampling units; entity/document clustering and "
    "question independence have not been verified or adjusted.",
    "Intervals describe question-resampling variation conditional on these predictions and "
    "frozen banks, not model-generation randomness or memory-source sampling variation.",
    "Percentile intervals are pointwise, not simultaneous intervals for ten comparisons. "
    "Only the ten predeclared EM exact-test p-values form the Holm family, not all four metrics.",
    "A percentile interval excluding zero is not an EM Holm rejection; neither outcome "
    "establishes a winner, causality or equivalence.",
)


def analysis_policy() -> dict:
    """Return a fresh JSON-compatible fixed policy; no CLI or data-driven tuning."""
    return {
        "schema_version": SCHEMA,
        "arms": list(ARMS),
        "metrics": list(METRICS),
        "comparisons": [
            {
                "name": f"{candidate}_minus_{reference}",
                "candidate": candidate,
                "reference": reference,
            }
            for candidate, reference in _COMPARISONS
        ],
        "difference": "candidate minus reference, original 0..1 metric units",
        "denominator": "per-metric seven-arm common-scorable questions in report order",
        "missing_data": "exclude from that metric's common set; never impute zero",
        "bootstrap": {
            "resamples": _RESAMPLES,
            "seed": _SEED,
            "bit_generator": "PCG64",
            "rng_reset": "same seed reset independently for each metric",
            "sampling_unit": "question with replacement, paired across all ten comparisons",
            "batch_size_max": _CHUNK,
            "interval": "two-sided pointwise percentile",
            "confidence_level": 0.95,
            "quantiles": [0.025, 0.975],
            "quantile_method": "linear",
            "index_digest_encoding": "row-major little-endian int64",
            "n_below_two": "CI unavailable, not zero uncertainty",
        },
        "em_test": {
            "test": "exact McNemar, two-sided binomial conditional on discordants",
            "null_probability": 0.5,
            "no_valid_questions": "p unavailable",
            "no_discordants": "p=1",
            "multiplicity": "Holm step-down",
            "family_size": len(_COMPARISONS),
            "alpha": _ALPHA,
            "missing_p": "internally use 1 to retain fixed family; output remains unavailable",
        },
    }


def _bootstrap_means(deltas, *, resamples=_RESAMPLES, seed=_SEED, chunk_size=_CHUNK):
    """Private testable primitive; production always uses the fixed defaults.

    Each row is a question and each column a predeclared comparison. One shared
    index matrix per chunk is applied to all columns, never per-arm resampling.
    Only <=256 resamples of question indices exist at once.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    if (
        deltas.ndim != 2
        or deltas.shape[0] < 2
        or deltas.shape[1] == 0
        or not np.isfinite(deltas).all()
        or type(resamples) is not int
        or resamples < 1
        or type(chunk_size) is not int
        or not 1 <= chunk_size <= _CHUNK
    ):
        raise ValueError("finite paired matrix, n>=2, positive resamples and chunk<=256 required")
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty((resamples, deltas.shape[1]), dtype=np.float64)
    digest = hashlib.sha256()
    n = deltas.shape[0]
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = rng.integers(0, n, size=(stop - start, n), dtype=np.int64)
        digest.update(indices.astype("<i8", copy=False).tobytes(order="C"))
        means[start:stop] = deltas[indices].mean(axis=1)
    return means, digest.hexdigest()


def _mcnemar_exact(repairs: int, harms: int) -> float:
    """Exact symmetric binomial two-sided tail; counts describe discordant pairs."""
    if any(type(value) is not int or value < 0 for value in (repairs, harms)):
        raise ValueError("repair/harm counts must be nonnegative integers")
    discordant = repairs + harms
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(repairs, harms) + 1))
    return min(1.0, 2 * tail / (1 << discordant))


def _holm(pvalues) -> list[float | None]:
    """Keep all ten planned hypotheses even when some tests are unavailable."""
    if len(pvalues) != len(_COMPARISONS) or any(
        value is not None
        and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1)
        for value in pvalues
    ):
        raise ValueError("ten finite p-values in [0,1] or explicit None required")
    values = [1.0 if value is None else float(value) for value in pvalues]
    adjusted = [None] * len(values)
    running = 0.0
    for rank, index in enumerate(
        sorted(range(len(values)), key=lambda index: (values[index], index))
    ):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        if pvalues[index] is not None:
            adjusted[index] = running
    return adjusted


def infer_evaluation(reports, feedback) -> dict:
    """Pure fixed analysis after the existing summary's strict input validation.

    We deliberately leave summary's pair-specific descriptive denominators
    untouched. This separate inference artifact uses seven-way common sets only.
    """
    summary = summarize_evaluation(reports, feedback)
    policy = analysis_policy()
    comparison_names = [row["name"] for row in policy["comparisons"]]
    if list(summary["paired"]) != comparison_names:
        raise ValueError("summary comparisons differ from fixed inference policy")
    total = len(reports)
    metrics = {}
    for metric in METRICS:
        denominator = summary["denominators"]["all_arms_common"][metric]
        ids = denominator["question_ids"]
        n = len(ids)
        deltas = np.array(
            [
                [
                    feedback[qid][candidate][metric] - feedback[qid][reference][metric]
                    for candidate, reference in _COMPARISONS
                ]
                for qid in ids
            ],
            dtype=np.float64,
        ).reshape(n, len(_COMPARISONS))
        quantiles, index_sha = None, None
        if n >= 2:
            means, index_sha = _bootstrap_means(deltas)
            quantiles = np.quantile(means, [0.025, 0.975], axis=0, method="linear")
        comparisons = {}
        for column, (candidate, reference) in enumerate(_COMPARISONS):
            warnings = []
            interval = None
            if n < 2:
                warnings.append("fewer_than_two_common_questions_ci_unavailable")
            else:
                interval = {
                    "lower": float(quantiles[0, column]),
                    "upper": float(quantiles[1, column]),
                }
                if np.all(deltas[:, column] == deltas[0, column]):
                    warnings.append("constant_observed_paired_differences")
                if interval["lower"] == interval["upper"]:
                    warnings.append("degenerate_percentile_interval_is_not_zero_uncertainty")
            result = {
                "candidate": candidate,
                "reference": reference,
                "n": n,
                "mean_delta": float(deltas[:, column].mean()) if n else None,
                "ci95": interval,
                "warnings": warnings,
            }
            if metric == "answer_em":
                repairs = int(np.count_nonzero(deltas[:, column] == 1))
                harms = int(np.count_nonzero(deltas[:, column] == -1))
                result.update(
                    repairs=repairs,
                    harms=harms,
                    discordant_pairs=repairs + harms,
                    p_exact=_mcnemar_exact(repairs, harms) if n else None,
                )
            comparisons[f"{candidate}_minus_{reference}"] = result
        if metric == "answer_em":
            adjusted = _holm([comparisons[name]["p_exact"] for name in comparison_names])
            for name, value in zip(comparison_names, adjusted, strict=True):
                comparisons[name].update(
                    p_holm=value,
                    reject_holm_0_05=None if value is None else value <= _ALPHA,
                )
        metrics[metric] = {
            "denominator": denominator,
            "missing_questions": total - n,
            "missing_by_arm": {
                arm: total - summary["arms"][arm]["own_scorable"][metric]["n"] for arm in ARMS
            },
            "bootstrap_resamples_generated": _RESAMPLES if n >= 2 else 0,
            "shared_resampling_indices_sha256": index_sha,
            "comparisons": comparisons,
        }
    return {
        "schema_version": SCHEMA,
        "policy": policy,
        "policy_sha256": fingerprint(policy),
        "numpy_version": np.__version__,
        "question_count": total,
        "metrics": metrics,
        "warnings": list(_WARNINGS),
        "independence_verified": False,
    }
