"""Source label gate and exact support coverage tested with synthetic records only."""

import json
from copy import deepcopy

import pytest

from growrag.experiments import score_operator_sources as module
from growrag.experiments.operator_execution_signature import (
    METHOD_FILES,
    execution_configuration,
)
from growrag.experiments.operator_execution_signature import (
    SCHEMA as EXECUTION_SCHEMA,
)
from growrag.experiments.representation_runner import fingerprint
from growrag.experiments.shared_hotpot_dev import document_id


def _qid(number):
    return f"{number:024x}"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _gold(qid=None):
    return {
        "id": qid or _qid(1),
        "answer": "The Blue Team",
        "context": {"title": ["Same title"], "sentences": [["Opening.", "", "The Blue Team won."]]},
        "supporting_facts": {"title": ["Same title"], "sent_id": [2]},
    }


def _report(qid=None, arm="fresh", *, answer="Blue Team", end=None):
    qid = qid or _qid(1)
    sentences = _gold()["context"]["sentences"][0]
    text = "\n".join(sentences)
    identity = document_id("Same title", sentences)
    return {
        "status": "completed",
        "question_id": qid,
        "arm": arm,
        "episode": {
            "question_id": qid,
            "evidence": [
                {"evidence_id": identity, "title": "Same title", "text": text, "sentence_id": 0}
            ],
        },
        "reader": {
            "answer": answer,
            "visible_evidence_ids": [identity],
            "evidence_windows": [
                {
                    "evidence_id": identity,
                    "text_start": 0,
                    "text_end": len(text) if end is None else end,
                    "original_text_chars": len(text),
                }
            ],
        },
        "feedback": None,
        "memory_updated": False,
    }


def _launch(
    runs,
    name,
    ids,
    manifest_sha="a" * 64,
    *,
    model="fixed-qwen",
    phase="source",
    arms=module.ARMS,
    reports=None,
    status="completed",
):
    folder = runs / f"{module.PREFIX}{name}"
    signature = {
        "schema": EXECUTION_SCHEMA,
        "configuration": execution_configuration(),
        "files": dict.fromkeys(METHOD_FILES, "a" * 64),
    }
    signature["sha256"] = fingerprint(signature)
    plan = {
        "protocol": module.PROTOCOL,
        "phase": phase,
        "manifest_sha256": manifest_sha,
        "model": model,
        "question_ids": ids,
        "arms": list(arms),
        "gold_loaded": False,
        "memory_updates": False,
        "execution_signature": signature,
    }
    reports = (
        reports
        if reports is not None
        else [{"question_id": qid, "arms": {arm: _report(qid, arm) for arm in arms}} for qid in ids]
    )
    _write(folder / "launch_plan.json", plan)
    _write(folder / "predictions.json", reports)
    _write(
        folder / "predictions_frozen.json",
        {
            "phase": phase,
            "sha256": fingerprint(reports),
            "question_ids": [item["question_id"] for item in reports],
            "gold_loaded": False,
            "status": status,
            "cleanup_errors": [],
        },
    )
    return folder


def _manifest(ids):
    return {"roles": {"source": ids, "calibration": [_qid(20)], "evaluation": [_qid(30)]}}


def test_complete_sources_are_ordered_and_enriched_without_altering_predictions(tmp_path):
    _launch(tmp_path, "second", [_qid(2)])
    first = _launch(tmp_path, "first", [_qid(1)])
    original = (first / "predictions.json").read_bytes()
    records, model, audit = module.collect_frozen_sources(
        _manifest([_qid(1), _qid(2)]), "a" * 64, tmp_path
    )
    assert [item["question_id"] for item in records] == [_qid(1), _qid(2)]
    assert all(
        item["phase"] == "source" and item["protocol"] == module.PROTOCOL for item in records
    )
    assert model == "fixed-qwen" and len(audit) == 6
    assert (first / "predictions.json").read_bytes() == original


def test_missing_source_question_rejects_before_labels(tmp_path):
    _launch(tmp_path, "only", [_qid(1)])
    with pytest.raises(ValueError, match="not all source"):
        module.collect_frozen_sources(_manifest([_qid(1), _qid(2)]), "a" * 64, tmp_path)


def test_failure_is_preserved_but_unstarted_arms_without_failure_are_not_terminal(tmp_path):
    report = {
        "question_id": _qid(1),
        "arms": {
            "base": _report(arm="base"),
            "fresh": {"status": "failed", "feedback": None, "error_type": "TimeoutError"},
        },
    }
    _launch(tmp_path, "failed", [_qid(1)], reports=[report], status="failed")
    records, _, _ = module.collect_frozen_sources(_manifest([_qid(1)]), "a" * 64, tmp_path)
    assert records[0]["arms"]["fresh"]["status"] == "failed"
    assert "static" not in records[0]["arms"]


def test_only_base_without_failed_repair_does_not_unlock_gold(tmp_path):
    _launch(tmp_path, "only_base", [_qid(1)], arms=("base",))
    with pytest.raises(ValueError, match="unstarted arms"):
        module.collect_frozen_sources(_manifest([_qid(1)]), "a" * 64, tmp_path)


@pytest.mark.parametrize("mutation", ["fingerprint", "model", "manifest", "gold", "cleanup"])
def test_mismatched_source_certificates_reject(tmp_path, mutation):
    folder = _launch(tmp_path, "a", [_qid(1)])
    if mutation in {"fingerprint", "cleanup"}:
        path = folder / "predictions_frozen.json"
        item = json.loads(path.read_text())
        item["sha256" if mutation == "fingerprint" else "cleanup_errors"] = (
            "b" * 64 if mutation == "fingerprint" else ["ValueError"]
        )
    else:
        path = folder / "launch_plan.json"
        item = json.loads(path.read_text())
        key = {"model": "model", "manifest": "manifest_sha256", "gold": "gold_loaded"}[mutation]
        item[key] = {"model": "", "manifest": "b" * 64, "gold": True}[mutation]
    _write(path, item)
    with pytest.raises(ValueError):
        module.collect_frozen_sources(_manifest([_qid(1)]), "a" * 64, tmp_path)


def test_different_models_and_duplicate_retries_are_rejected(tmp_path):
    _launch(tmp_path, "a", [_qid(1)])
    _launch(tmp_path, "b", [_qid(2)], model="other-model")
    with pytest.raises(ValueError, match="one fixed model"):
        module.collect_frozen_sources(_manifest([_qid(1), _qid(2)]), "a" * 64, tmp_path)
    _launch(tmp_path, "retry", [_qid(1)])
    with pytest.raises(ValueError, match="duplicate source"):
        module.collect_frozen_sources(_manifest([_qid(1), _qid(2)]), "a" * 64, tmp_path)


def test_same_model_but_changed_method_is_not_pooled(tmp_path):
    _launch(tmp_path, "a", [_qid(1)])
    folder = _launch(tmp_path, "b", [_qid(2)])
    path = folder / "launch_plan.json"
    plan = json.loads(path.read_text())
    signature = plan["execution_signature"]
    signature["files"][METHOD_FILES[0]] = "b" * 64
    del signature["sha256"]
    signature["sha256"] = fingerprint(signature)
    _write(path, plan)
    with pytest.raises(ValueError, match="execution signature"):
        module.collect_frozen_sources(_manifest([_qid(1), _qid(2)]), "a" * 64, tmp_path)


def test_calibration_and_evaluation_launches_are_not_scored(tmp_path):
    _launch(tmp_path, "a", [_qid(1)])
    _launch(tmp_path, "calibration", [_qid(20)], phase="calibration")
    _launch(tmp_path, "evaluation", [_qid(30)], phase="evaluation")
    records, _, audit = module.collect_frozen_sources(_manifest([_qid(1)]), "a" * 64, tmp_path)
    assert len(records) == 1 and len(audit) == 3


def test_answer_normalization_and_original_sentence_two_coverage():
    result = module.score_source_report(_report(), _gold())
    assert result["answer_em"] == 1.0 and result["answer_f1"] == 1.0
    assert result["raw_support_recall"] == 1.0
    assert result["visible_support_recall"] == 1.0


def test_reader_prefix_that_cuts_gold_sentence_has_no_complete_visible_coverage():
    result = module.score_source_report(_report(end=13), _gold())
    assert result["raw_support_recall"] == 1.0
    assert result["visible_support_recall"] == 0.0


def test_same_title_different_version_is_not_gold_support():
    report = _report()
    item = report["episode"]["evidence"][0]
    identity = document_id("Same title", ["Different document version."])
    item.update(evidence_id=identity, text="Different document version.")
    report["reader"].update(
        visible_evidence_ids=[identity],
        evidence_windows=[
            {
                "evidence_id": identity,
                "text_start": 0,
                "text_end": len(item["text"]),
                "original_text_chars": len(item["text"]),
            }
        ],
    )
    result = module.score_source_report(report, _gold())
    assert result["raw_support_recall"] == 0.0 and result["visible_support_recall"] == 0.0


def test_claimed_correct_doc_id_with_changed_content_is_unknown_not_positive():
    report = _report()
    report["episode"]["evidence"][0]["text"] = "Changed text"
    result = module.score_source_report(report, _gold())
    assert result["answer_em"] == 1.0
    assert result["raw_support_recall"] is None
    assert result["issue"] == "evidence_content_id_mismatch"


def test_missing_window_metadata_is_explicit_unknown_not_zero():
    report = _report()
    del report["reader"]["evidence_windows"]
    result = module.score_source_report(report, _gold())
    assert result["raw_support_recall"] == 1.0 and result["visible_support_recall"] is None
    assert result["issue"] == "visible_windows_not_recorded"


@pytest.mark.parametrize("report", [None, {"status": "failed"}])
def test_failed_or_unattempted_arms_never_get_fake_zero_scores(report):
    result = module.score_source_report(report, _gold())
    assert all(
        result[key] is None
        for key in ("answer_em", "answer_f1", "raw_support_recall", "visible_support_recall")
    )


@pytest.mark.parametrize(
    "change", ["empty_answer", "missing_support", "invalid_index", "ambiguous_title"]
)
def test_invalid_annotation_kept_as_unknown(change):
    gold = _gold()
    if change == "empty_answer":
        gold["answer"] = ""
    elif change == "missing_support":
        gold["supporting_facts"] = {"title": [], "sent_id": []}
    elif change == "invalid_index":
        gold["supporting_facts"]["sent_id"] = [30]
    else:
        gold["context"]["title"].append("Same title")
        gold["context"]["sentences"].append(["Other version"])
    result = module.score_source_report(_report(), gold)
    assert result["annotation_status"] == "invalid"
    assert result["answer_em"] is None and result["raw_support_recall"] is None


@pytest.fixture
def score_bundle(tmp_path, monkeypatch):
    manifest_path = tmp_path / "data/manifest.json"
    manifest = _manifest([_qid(1), _qid(2)])
    _write(manifest_path, manifest)
    runs = tmp_path / "runs"
    _launch(runs, "one", [_qid(1), _qid(2)], manifest_sha=module._sha(manifest_path))
    monkeypatch.setattr(module, "load_inputs", lambda path, phase: (manifest, ()))
    return tmp_path, manifest_path, runs, tmp_path / "feedback"


def test_default_preflight_never_opens_gold(score_bundle, monkeypatch):
    monkeypatch.setattr(module, "_load_source_gold", lambda *args: pytest.fail("gold opened"))
    result = module.score_operator_sources(*score_bundle)
    assert result["gold_loaded"] is False
    assert not score_bundle[-1].exists()


def test_incomplete_predictions_cannot_open_gold_even_with_explicit_write(
    score_bundle, monkeypatch
):
    root, manifest, runs, output = score_bundle
    monkeypatch.setattr(
        module, "load_inputs", lambda path, phase: (_manifest([_qid(1), _qid(2), _qid(3)]), ())
    )
    monkeypatch.setattr(module, "_load_source_gold", lambda *args: pytest.fail("gold opened"))
    with pytest.raises(ValueError, match="not all source"):
        module.score_operator_sources(root, manifest, runs, output, write=True)
    assert not output.exists()


def test_complete_predictions_produce_scalar_feedback_and_separate_audit(score_bundle, monkeypatch):
    monkeypatch.setattr(
        module, "_load_source_gold", lambda *args: ({_qid(i): _gold(_qid(i)) for i in (1, 2)}, [])
    )
    result = module.score_operator_sources(*score_bundle, write=True)
    output = score_bundle[-1]
    feedback = json.loads((output / "feedback.json").read_text())
    assert set(feedback[_qid(1)]["fresh"]) == {"answer_em", "answer_f1", "raw_support_recall"}
    audit = json.loads((output / "scoring_audits.json").read_text())
    assert audit[_qid(1)]["fresh"]["visible_support_recall"] == 1.0
    assert result["gold_loaded"] is True and result["calibration_evaluation_gold_loaded"] is False
    assert result["raw_predictions_modified"] is False
    with pytest.raises(FileExistsError):
        module.score_operator_sources(*score_bundle, write=True)


def test_label_projection_uses_only_source_ids_and_manifest_pinned_train_parquet(
    tmp_path, monkeypatch
):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow.dataset as ds

    folder = tmp_path / "data/hotpotqa/official_train_v1_1"
    folder.mkdir(parents=True)
    shard = folder / "train.parquet"
    rows = [_gold(_qid(i)) for i in (1, 20, 30)]
    pq.write_table(pa.Table.from_pylist(rows), shard)
    provenance = folder / "mirror_provenance.json"
    _write(
        provenance,
        {"official_split": "train", "shards": [{"file": shard.name, "sha256": module._sha(shard)}]},
    )
    manifest = _manifest([_qid(1)])
    manifest["input_files"] = [
        {"path": p.relative_to(tmp_path).as_posix(), "sha256": module._sha(p)}
        for p in (shard, provenance)
    ]
    original = ds.dataset
    called = []

    class Checked:
        def __init__(self, dataset):
            self.dataset = dataset

        def to_table(self, *, columns, filter):
            assert columns == ["id", "answer", "supporting_facts", "context"]
            table = self.dataset.to_table(columns=columns, filter=filter)
            assert table["id"].to_pylist() == [_qid(1)]
            called.append(True)
            return table

    monkeypatch.setattr(ds, "dataset", lambda *args, **kwargs: Checked(original(*args, **kwargs)))
    gold, audit = module._load_source_gold(tmp_path, manifest)
    assert set(gold) == {_qid(1)} and len(audit) == 1 and called == [True]
    changed = deepcopy(manifest)
    changed["input_files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="parquet differs"):
        module._load_source_gold(tmp_path, changed)
