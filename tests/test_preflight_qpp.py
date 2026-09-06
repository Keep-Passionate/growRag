import hashlib
import json
import math

import pytest

from growrag.experiments.data_protocol import build_manifest
from growrag.experiments.hotpot import parse_hotpot_example
from growrag.experiments.lexical_retriever import BM25SentenceRetriever
from growrag.experiments.protocol import Evidence, GoldRecord, RuntimeQuestion
from growrag.preflight_qpp import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    MODEL_VERSION,
    extract_features,
    offline_support_recall,
    predict_from_json,
    train_from_manifest,
)


def corpus():
    return (Evidence("a", "Doc", 0, "alpha beta"), Evidence("b", "Doc", 1, "beta gamma"))


def create_manifest(tmp_path):
    records = [
        {
            "_id": f"q{i}",
            "question": f"What happened in year {i}?",
            "answer": "Answer",
            "context": [["Doc", [f"In year {i} alpha occurred.", "Unrelated event."]]],
            "supporting_facts": [["Doc", 0]],
        }
        for i in range(200)
    ]
    raw = json.dumps(records).encode()
    (tmp_path / "data.json").write_bytes(raw)
    manifest = build_manifest(records)
    manifest.update(data_file="data.json", data_sha256=hashlib.sha256(raw).hexdigest())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest, records


def test_features_do_not_rank_or_accept_gold(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pre-retrieval features must not rank")

    monkeypatch.setattr(BM25SentenceRetriever, "retrieve", forbidden)
    features = extract_features("alpha alpha missing", corpus())
    idf_present, idf_missing = math.log(2), math.log(6)
    assert features == pytest.approx(
        (math.log(4), 2 / 3, (idf_present + idf_missing) / 2, idf_missing, 0.5)
    )
    with pytest.raises(TypeError):
        extract_features("alpha", (GoldRecord("q", ("a",)),))


def test_feature_case_order_and_duplicate_token_conventions():
    assert extract_features("ALPHA beta", corpus()) == extract_features(
        "alpha BETA", corpus()[::-1]
    )
    assert (
        extract_features("alpha alpha beta", corpus())[2:]
        == extract_features("alpha beta", corpus())[2:]
    )
    with pytest.raises(ValueError):
        extract_features("???", corpus())


def test_label_uses_real_bm25_and_gold_only_offline():
    question = RuntimeQuestion("q", "alpha")
    gold = GoldRecord("q", ("unused answer",), (("Doc", 0), ("Doc", 1)))
    assert offline_support_recall(question, BM25SentenceRetriever(corpus()), gold) == 0.5
    with pytest.raises(ValueError, match="another question"):
        offline_support_recall(
            question, BM25SentenceRetriever(corpus()), GoldRecord("other", ("a",))
        )
    with pytest.raises(ValueError, match="undefined"):
        offline_support_recall(question, BM25SentenceRetriever(corpus()), GoldRecord("q", ("a",)))


def test_json_inference_validates_versions_and_clips():
    model = {
        "schema_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": [0.0] * 5,
        "scaler_scale": [1.0] * 5,
        "ridge_coef": [0.0] * 5,
        "ridge_intercept": 2.0,
    }
    assert predict_from_json(model, "alpha", corpus()) == 1.0
    model["ridge_intercept"] = -1.0
    assert predict_from_json(model, "alpha", corpus()) == 0.0
    model["scaler_scale"][0] = 0
    with pytest.raises(ValueError, match="positive"):
        predict_from_json(model, "alpha", corpus())


def test_fitting_is_train_only_and_json_round_trip(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    path, manifest, records = create_manifest(tmp_path)
    seen_queries = []

    def traced_features(query, candidate_corpus):
        seen_queries.append(query)
        return extract_features(query, candidate_corpus)

    monkeypatch.setattr("growrag.preflight_qpp.extract_features", traced_features)
    report = train_from_manifest(path, tmp_path / "out")
    model = json.loads((tmp_path / "out" / "model.json").read_text())
    assert model["fit_question_ids"] == manifest["roles"]["selector_train"]
    assert set(report["rows"]) == {"selector_train", "calibration_dev"}
    assert report["api_requests"] == 0
    assert report["fit_or_tune_on_calibration_dev"] is False
    assert model["runtime_gate_enabled"] is False
    assert "NOT rewrite/reuse gain" in report["notice"]
    by_id = {record["_id"]: record for record in records}
    expected_queries = {
        by_id[qid]["question"]
        for role in ("selector_train", "calibration_dev")
        for qid in manifest["roles"][role]
    }
    assert set(seen_queries) == expected_queries
    assert len(seen_queries) == len(expected_queries)
    for role, rows in report["rows"].items():
        assert [row["question_id"] for row in rows] == manifest["roles"][role]
        for row in rows:
            example = parse_hotpot_example(by_id[row["question_id"]], dataset="synthetic")
            value = predict_from_json(model, example.question.text, example.candidate_context)
            assert value == pytest.approx(row["prediction_clipped"])
    with pytest.raises(FileExistsError):
        train_from_manifest(path, tmp_path / "out")


def test_dev_inputs_and_labels_do_not_change_fitted_coefficients(tmp_path):
    pytest.importorskip("sklearn")
    path, manifest, records = create_manifest(tmp_path)
    first = train_from_manifest(path, tmp_path / "first")
    dev_ids = set(manifest["roles"]["calibration_dev"])
    for record in records:
        if record["_id"] in dev_ids:
            record["supporting_facts"] = [["Absent", 0]]
            record["context"] = [["Unseen", ["Completely unrelated words."]]]
    raw = json.dumps(records).encode()
    (tmp_path / "data.json").write_bytes(raw)
    manifest["data_sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(manifest))
    second = train_from_manifest(path, tmp_path / "second")
    for key in ("scaler_mean", "scaler_scale", "ridge_coef", "ridge_intercept"):
        assert first["model"][key] == second["model"][key]
    assert first["metrics"]["calibration_dev"] != second["metrics"]["calibration_dev"]


@pytest.mark.parametrize("change", ["official_dev", "roles", "checksum", "escape"])
def test_corrupt_or_evaluation_manifest_rejected(tmp_path, change):
    pytest.importorskip("sklearn")
    path, manifest, _ = create_manifest(tmp_path)
    if change == "official_dev":
        manifest["official_split"] = "validation"
    elif change == "roles":
        manifest["roles"]["selector_train"] = manifest["roles"]["memory_seed"]
    elif change == "checksum":
        manifest["data_sha256"] = "wrong"
    else:
        manifest["data_file"] = "../outside.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        train_from_manifest(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()
