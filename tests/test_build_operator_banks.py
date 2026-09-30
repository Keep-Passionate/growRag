"""Synthetic handoff and prefix-isolation tests; no real labels/API are used."""

import json
from dataclasses import asdict

import pytest

from growrag.experiments import build_operator_banks as module
from growrag.experiments.operator_source_bank import SOURCE_PROTOCOL
from growrag.experiments.protocol import Evidence, RuntimeQuestion
from growrag.macro_operators import GapField, GoalContract, OperatorSpec, QueryStep
from growrag.operator_bank import FrozenOperatorBank
from growrag.operator_loop import ActionProposal, run_operator_episode


def _qid(number):
    return f"{number:024x}"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _complete_source(qid):
    question = RuntimeQuestion(qid, "Where was Alice born?")
    goal = GoalContract(question.text, "lookup")
    spec = OperatorSpec(
        "PROPOSED",
        "1",
        ("lookup",),
        (GapField("entity"),),
        (QueryStep("search", "{entity} birthplace"),),
    )
    initial = Evidence("e0", "Biography", 0, "Alice worked here.")

    def retrieve(query, top_k):
        return (initial,) if query == question.text else (Evidence("e1", "Place", 0, "New fact."),)

    arms = {}
    for arm in ("base", "fresh", "static"):
        decide = (
            (lambda state: ActionProposal(goal, spec, {"entity": "Alice"}))
            if arm == "fresh"
            else (lambda state: ActionProposal(goal, None, origin="stop"))
        )
        result = run_operator_episode(
            question, retrieve, decide, retrieval_budget=1 if arm == "base" else 3, max_decisions=1
        )
        arms[arm] = {
            "question_id": qid,
            "arm": arm,
            "status": "completed",
            "episode": json.loads(json.dumps(asdict(result))),
        }
    return {"question_id": qid, "phase": "source", "protocol": SOURCE_PROTOCOL, "arms": arms}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source = [_qid(i) for i in range(500)]
    manifest = {
        "roles": {"source": source, "calibration": [_qid(600)], "evaluation": [_qid(700)]},
        "nested_source_ids": {str(size): source[:size] for size in module.SOURCE_SIZES},
    }
    manifest_path = tmp_path / "data/manifest.json"
    _write(manifest_path, manifest)
    monkeypatch.setattr(module, "load_inputs", lambda path, phase: (manifest, ()))
    reports = [
        {
            "question_id": qid,
            "phase": "source",
            "protocol": SOURCE_PROTOCOL,
            "arms": {arm: {"status": "failed"} for arm in ("base", "fresh", "static")},
        }
        for qid in source
    ]
    feedback = {
        qid: {arm: dict.fromkeys(module.METRICS) for arm in ("base", "fresh", "static")}
        for qid in source
    }
    folder = tmp_path / "scored"
    prediction = tmp_path / "runs/example/predictions.json"
    _write(prediction, reports)
    audit = {
        "schema_version": module.FEEDBACK_SCHEMA,
        "phase": "source",
        "protocol": SOURCE_PROTOCOL,
        "manifest_sha256": module._sha(manifest_path),
        "source_ids": source,
        "source_count": 500,
        "gold_loaded": True,
        "calibration_evaluation_gold_loaded": False,
        "raw_predictions_modified": False,
        "model": "fixed-qwen",
        "prediction_inputs": [{"path": str(prediction), "sha256": module._sha(prediction)}],
    }

    def freeze():
        _write(folder / "feedback.json", feedback)
        _write(folder / "source_reports.json", reports)
        audit.update(
            feedback_sha256=module.fingerprint(feedback),
            source_reports_sha256=module.fingerprint(reports),
            artifacts={
                name: {"sha256": module._sha(folder / name)}
                for name in ("feedback.json", "source_reports.json")
            },
        )
        _write(folder / "audit.json", audit)
        return module._sha(folder / "audit.json")

    freeze()
    return (
        tmp_path,
        manifest_path,
        folder,
        tmp_path / "banks",
        manifest,
        reports,
        feedback,
        audit,
        freeze,
    )


def _build(data, **kwargs):
    root, manifest, feedback, output, *_ = data
    return module.build_operator_banks(
        root,
        manifest,
        feedback,
        output,
        expected_audit_sha256=module._sha(feedback / "audit.json"),
        **kwargs,
    )


def _make_win(data, number):
    qid = data[4]["roles"]["source"][number]
    data[5][number] = _complete_source(qid)
    data[6][qid] = {
        "base": dict.fromkeys(module.METRICS, 0),
        "static": dict.fromkeys(module.METRICS, 0),
        "fresh": dict.fromkeys(module.METRICS, 1),
    }


def test_empty_banks_are_legal_and_dry_run_does_not_write_or_authorize_evaluation(inputs):
    result = _build(inputs)
    assert not inputs[3].exists()
    assert result["source_sizes"] == [50, 100, 250, 500]
    assert result["new_gold_loaded"] is False and result["evaluation_authorized"] is False
    assert result["api_calls"] == 0
    for item in result["banks"].values():
        assert item["retained_record_count"] == item["published_count"] == 0
        assert item["empty_published_bank"] is True
        assert item["empty_bank_policy"] == "fresh_fallback_without_resampling"


def test_four_output_banks_have_exact_prefix_scope_fingerprints_and_audits(inputs):
    result = _build(inputs, write=True)
    for size in module.SOURCE_SIZES:
        path = inputs[3] / f"bank_{size}.json"
        bank = FrozenOperatorBank.from_json(path.read_text(encoding="utf-8"))
        assert set(bank.allowed_source_ids) == set(inputs[4]["roles"]["source"][:size])
        assert result["banks"][str(size)]["sha256"] == module._sha(path)
        assert (inputs[3] / f"bank_{size}_audit.json").exists()
    assert (inputs[3] / "bank_bundle.sha256").read_text().strip() == (
        f"{module._sha(inputs[3] / 'bank_bundle.json')}  bank_bundle.json"
    )


def test_prefixes_are_built_before_aggregation_not_truncated_from_largest(inputs):
    _make_win(inputs, 0)
    _make_win(inputs, 60)
    inputs[-1]()
    result = _build(inputs, write=True)
    small = FrozenOperatorBank.from_json((inputs[3] / "bank_50.json").read_text())
    large = FrozenOperatorBank.from_json((inputs[3] / "bank_100.json").read_text())
    assert small.records[0].source_qids == (_qid(0),)
    assert large.records[0].source_qids == (_qid(0), _qid(60))
    assert result["banks"]["50"]["published_count"] == 1
    assert result["banks"]["50"]["cross_question_transfer_verified"] is False


@pytest.mark.parametrize("phase", ["calibration", "evaluation"])
def test_non_source_reports_rejected_even_if_all_input_hashes_resigned(inputs, phase):
    inputs[5][0]["phase"] = phase
    inputs[-1]()
    with pytest.raises(ValueError, match="cannot build memory"):
        _build(inputs)


@pytest.mark.parametrize("target", ["reports", "feedback"])
def test_calibration_or_eval_ids_cannot_enter_source_scope(inputs, target):
    if target == "reports":
        inputs[5][0]["question_id"] = _qid(600)
    else:
        inputs[6][_qid(700)] = inputs[6].pop(_qid(0))
    inputs[-1]()
    with pytest.raises(ValueError, match="source"):
        _build(inputs)


def test_reordered_or_changed_nested_prefix_rejected(inputs):
    inputs[4]["nested_source_ids"]["50"] = list(reversed(inputs[4]["nested_source_ids"]["50"]))
    with pytest.raises(ValueError, match="nested source"):
        _build(inputs)


def test_scoring_audit_must_match_explicit_handoff_sha(inputs):
    with pytest.raises(ValueError, match="audit SHA mismatch"):
        module.build_operator_banks(*inputs[:4], expected_audit_sha256="f" * 64)
    with pytest.raises(ValueError, match="explicit scoring audit"):
        module.build_operator_banks(*inputs[:4], expected_audit_sha256="bad")


def test_changed_feedback_file_or_semantic_fingerprint_rejected(inputs):
    path = inputs[2] / "feedback.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="artifact SHA mismatch"):
        _build(inputs)
    inputs[-1]()
    audit = inputs[7]
    audit["feedback_sha256"] = "f" * 64
    _write(inputs[2] / "audit.json", audit)
    with pytest.raises(ValueError, match="content fingerprint"):
        _build(inputs)


@pytest.mark.parametrize(
    "key,value",
    [
        ("phase", "calibration"),
        ("gold_loaded", False),
        ("calibration_evaluation_gold_loaded", True),
        ("raw_predictions_modified", True),
        ("model", ""),
        ("source_count", 499),
    ],
)
def test_invalid_scoring_provenance_rejected(inputs, key, value):
    inputs[7][key] = value
    inputs[-1]()
    with pytest.raises(ValueError, match="complete approved source"):
        _build(inputs)


@pytest.mark.parametrize(
    "change", ["answer_text", "bool", "fractional_em", "missing_metric", "extra_arm"]
)
def test_only_exact_scalar_feedback_enters_extractor(inputs, change):
    values = inputs[6][_qid(0)]["fresh"]
    if change == "answer_text":
        values["answer"] = "label text must not enter bank"
    elif change == "bool":
        values["answer_em"] = True
    elif change == "fractional_em":
        values["answer_em"] = 0.5
    elif change == "missing_metric":
        del values["raw_support_recall"]
    else:
        inputs[6][_qid(0)]["memory500"] = dict(values)
    inputs[-1]()
    with pytest.raises(ValueError, match="feedback|scalar"):
        _build(inputs)


def test_old_or_partial_output_is_never_overwritten(inputs):
    inputs[3].mkdir()
    keep = inputs[3] / "keep.txt"
    keep.write_text("keep")
    with pytest.raises(FileExistsError):
        _build(inputs, write=True)
    assert keep.read_text() == "keep"


def test_prediction_change_after_scoring_requires_rescore_not_silent_reuse(inputs):
    prediction = inputs[0] / "runs/example/predictions.json"
    prediction.write_text("[]")
    with pytest.raises(ValueError, match="prediction changed"):
        _build(inputs)


def test_gold_files_are_not_reopened_as_prediction_audit_inputs(inputs):
    path = inputs[0] / "fake_gold.json"
    path.write_text("must not be parsed")
    inputs[7]["prediction_inputs"] = [{"path": str(path), "sha256": module._sha(path)}]
    inputs[-1]()
    with pytest.raises(ValueError, match="prediction-only"):
        _build(inputs)


def test_builder_does_not_open_gold_provenance_paths(inputs):
    inputs[7]["gold_inputs"] = [{"path": "Z:/nonexistent-sealed-gold.parquet", "sha256": "f" * 64}]
    inputs[-1]()
    assert _build(inputs)["new_gold_loaded"] is False
