"""Tiny offline QPP regression diagnostic, never a rewrite/reuse gate.

Predicts annotated support-sentence recall of the ORIGINAL query under the
existing local BM25 top-4 sentence retriever. This is not a probability of
correctness, evidence sufficiency, rewrite gain, or memory usefulness.

Features use only query text and pre-ranking statistics of that question's
allowed distractor candidate pool. Gold and ranked results enter label creation
only. Corpus statistics are not learned across questions or from memory_seed.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from collections import Counter
from pathlib import Path

from .experiments.data_protocol import DATASET_ID, SPLIT_VERSION, build_manifest
from .experiments.hotpot import HotpotExample, parse_hotpot_example
from .experiments.lexical_retriever import BM25SentenceRetriever
from .experiments.protocol import Evidence, GoldRecord, RuntimeQuestion

FEATURE_VERSION = "growrag-pre-retrieval-lexical-five-v1"
MODEL_VERSION = "growrag-original-query-support-recall-ridge-v1"
FEATURE_NAMES = (
    "log1p_query_token_count",
    "unique_token_ratio",
    "mean_distinct_query_idf",
    "max_distinct_query_idf",
    "distinct_query_oov_fraction",
)
TOP_K = 4
NOTICE = (
    "Tiny convenience sample from official train; engineering feasibility only, "
    "not effectiveness evidence. Predicts original-query local BM25 top-4 "
    "annotated support recall, NOT rewrite/reuse gain or correctness probability. "
    "Not enabled in runtime routing."
)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


def extract_features(query: str, corpus: tuple[Evidence, ...]) -> tuple[float, ...]:
    """Five pre-ranking features; accepts neither GoldRecord nor ranked results.

    IDF uses log(1 + (N - df + .5)/(df + .5)), including df=0 for OOV
    terms. Mean/max/OOV use DISTINCT query tokens. Tokenization matches local
    BM25's Unicode case-folded alphanumeric title+sentence representation.
    """
    if not isinstance(query, str) or not (tokens := _tokens(query)):
        raise ValueError("query must have alphanumeric text")
    if not isinstance(corpus, tuple) or not corpus:
        raise ValueError("corpus must be a nonempty immutable Evidence tuple")
    if not all(isinstance(item, Evidence) for item in corpus):
        raise TypeError("corpus must contain Evidence only")
    if len({item.evidence_id for item in corpus}) != len(corpus):
        raise ValueError("duplicate corpus evidence IDs")
    frequencies: Counter[str] = Counter()
    for item in corpus:
        frequencies.update(set(_tokens(f"{item.title} {item.text}")))
    distinct = sorted(set(tokens))
    idfs = [
        math.log1p((len(corpus) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
        for term in distinct
    ]
    return (
        math.log1p(len(tokens)),
        len(distinct) / len(tokens),
        sum(idfs) / len(idfs),
        max(idfs),
        sum(term not in frequencies for term in distinct) / len(distinct),
    )


def offline_support_recall(
    question: RuntimeQuestion, retriever: BM25SentenceRetriever, gold: GoldRecord
) -> float:
    """Create a label offline by actually ranking; never used inside features."""
    if question.question_id != gold.question_id:
        raise ValueError("gold belongs to another question")
    annotated = set(gold.supporting_facts)
    if not annotated:
        raise ValueError("support-recall label is undefined without annotated support")
    retrieved = retriever.retrieve(question.text, top_k=TOP_K).value
    refs = {(item.title, item.sentence_id) for item in retrieved}
    return len(refs & annotated) / len(annotated)


def _load_roles(path: Path) -> tuple[dict[str, tuple[HotpotExample, ...]], dict]:
    path = path.resolve()
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("schema_version") != "growrag-hotpot-manifest-v1"
        or manifest.get("dataset") != DATASET_ID
        or manifest.get("official_split") != "train"
        or manifest.get("configuration") != "distractor"
        or manifest.get("split_version") != SPLIT_VERSION
    ):
        raise ValueError("only the existing frozen Hotpot train manifest is allowed")
    data_path = (path.parent / manifest["data_file"]).resolve()
    if data_path.parent != path.parent:
        raise ValueError("manifest data file must stay in its directory")
    raw = data_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["data_sha256"]:
        raise ValueError("data checksum does not match frozen manifest")
    records = json.loads(raw)
    expected = build_manifest(records, seed=manifest["seed"])
    for key in (
        "roles",
        "selected",
        "role_counts",
        "record_count",
        "role_weights",
        "official_validation_used",
        "official_test_used",
    ):
        if manifest[key] != expected[key]:
            raise ValueError("manifest roles changed from the frozen split algorithm")
    by_id = {record["_id"]: record for record in records}
    roles = {
        role: tuple(
            parse_hotpot_example(by_id[qid], dataset="hotpotqa-distractor-train-preview-200")
            for qid in manifest["roles"][role]
        )
        for role in ("selector_train", "calibration_dev")
    }
    if any(not values for values in roles.values()):
        raise ValueError("training and internal development roles must both be nonempty")
    return roles, {
        "dataset": DATASET_ID,
        "official_split": "train",
        "configuration": "distractor",
        "data_sha256": manifest["data_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "split_version": SPLIT_VERSION,
        "seed": manifest["seed"],
        "unused_memory_seed_count": manifest["role_counts"]["memory_seed"],
        "official_dev_used": False,
        "official_test_used": False,
    }


def predict_from_json(artifact: dict, query: str, corpus: tuple[Evidence, ...]) -> float:
    """Portable JSON-only inference without sklearn or pickle; output is clipped."""
    if (
        artifact.get("schema_version") != MODEL_VERSION
        or artifact.get("feature_version") != FEATURE_VERSION
        or artifact.get("feature_names") != list(FEATURE_NAMES)
    ):
        raise ValueError("incompatible QPP feature/model version")
    vectors = [artifact[key] for key in ("scaler_mean", "scaler_scale", "ridge_coef")]
    if any(len(vector) != len(FEATURE_NAMES) for vector in vectors):
        raise ValueError("invalid QPP coefficient shape")
    if not all(math.isfinite(float(value)) for vector in vectors for value in vector):
        raise ValueError("QPP coefficients must be finite")
    if any(float(value) <= 0 for value in artifact["scaler_scale"]):
        raise ValueError("QPP scales must be positive")
    intercept = float(artifact["ridge_intercept"])
    if not math.isfinite(intercept):
        raise ValueError("QPP intercept must be finite")
    features = extract_features(query, corpus)
    raw = intercept + sum(
        (value - mean) / scale * coefficient
        for value, mean, scale, coefficient in zip(features, *vectors, strict=True)
    )
    return min(1.0, max(0.0, raw))


def train_from_manifest(manifest_path: Path, output_dir: Path) -> dict:
    """Fit once on selector_train; calibration_dev only reports MAE/MSE.

    No search, threshold selection, prompt changes, network calls or memory
    writes. Dependencies are optional and imported only for offline fitting.
    """
    import numpy as np
    import sklearn
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    roles, provenance = _load_roles(Path(manifest_path))
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite an existing QPP run")
    rows = {}
    for role, examples in roles.items():
        rows[role] = []
        for example in examples:
            # Feature creation is deliberately before offline ranking/gold access.
            features = extract_features(example.question.text, example.candidate_context)
            retriever = BM25SentenceRetriever(example.candidate_context)
            if example.gold is None:
                raise ValueError("offline training labels are missing")
            label = offline_support_recall(example.question, retriever, example.gold)
            rows[role].append(
                {
                    "question_id": example.question.question_id,
                    "features": list(features),
                    "label_support_recall_at_4": label,
                    "corpus_fingerprint": retriever.corpus_fingerprint,
                    "retriever_context_fingerprint": retriever.context_fingerprint,
                }
            )
    train_rows = rows["selector_train"]
    train_x = np.asarray([row["features"] for row in train_rows], dtype=float)
    train_y = np.asarray([row["label_support_recall_at_4"] for row in train_rows], dtype=float)
    pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, solver="svd"))])
    pipeline.fit(train_x, train_y)
    baseline = float(train_y.mean())
    artifact = {
        "schema_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": pipeline["scaler"].mean_.tolist(),
        "scaler_scale": pipeline["scaler"].scale_.tolist(),
        "ridge_coef": pipeline["ridge"].coef_.tolist(),
        "ridge_intercept": float(pipeline["ridge"].intercept_),
        "ridge_alpha": 1.0,
        "ridge_solver": "svd",
        "prediction_clip": [0.0, 1.0],
        "is_probability": False,
        "runtime_gate_enabled": False,
        "label": "original_query_local_bm25_top_4_annotated_support_sentence_recall",
        "corpus_statistics_scope": "each_question_allowed_title_sentence_candidate_pool",
        "idf_oov_definition": "positive_bm25_idf_with_df_zero; distinct_query_tokens",
        "fit_question_ids": [row["question_id"] for row in train_rows],
        "provenance": provenance,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "source_sha256": {
            "preflight_qpp.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "lexical_retriever.py": hashlib.sha256(
                (Path(__file__).parent / "experiments" / "lexical_retriever.py").read_bytes()
            ).hexdigest(),
        },
        "notice": NOTICE,
    }
    metrics = {}
    for role, role_rows in rows.items():
        x = np.asarray([row["features"] for row in role_rows], dtype=float)
        y = np.asarray([row["label_support_recall_at_4"] for row in role_rows], dtype=float)
        raw = pipeline.predict(x)
        predictions = np.clip(raw, 0.0, 1.0)
        constant = np.full(len(y), baseline)
        metrics[role] = {
            "n": len(y),
            "ridge_clipped": {
                "mae": float(mean_absolute_error(y, predictions)),
                "mse": float(mean_squared_error(y, predictions)),
            },
            "training_mean_constant": {
                "mae": float(mean_absolute_error(y, constant)),
                "mse": float(mean_squared_error(y, constant)),
            },
        }
        for row, unclipped, clipped in zip(role_rows, raw, predictions, strict=True):
            row.update(
                prediction_unclipped=float(unclipped),
                prediction_clipped=float(clipped),
                training_mean_constant=baseline,
            )
    report = {
        "schema_version": "growrag-preflight-qpp-report-v1",
        "notice": NOTICE,
        "model": artifact,
        "training_mean_constant": baseline,
        "hyperparameter_search": False,
        "fit_or_tune_on_calibration_dev": False,
        "api_requests": 0,
        "metrics": metrics,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, value in (("model.json", artifact), ("report.json", report)):
        (output_dir / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
    return report
