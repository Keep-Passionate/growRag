"""Synthetic v3 source500 -> scalar scoring -> independent banks; no real gold/API."""

import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from growrag.experiments import build_operator_banks as banks
from growrag.experiments import score_operator_sources as scoring
from growrag.experiments.operator_model_v3 import ModelOperatorPlannerV3, encode_wire_plan
from growrag.experiments.operator_profiles import (
    ACTION_LIST,
    ACTION_LIST_METHOD_FILES,
    ACTION_LIST_SCHEMA,
    LEGACY,
    STRUCTURED,
    action_list_execution_configuration,
)
from growrag.experiments.operator_source_bank import build_source_bank
from growrag.experiments.protocol import Evidence, RuntimeQuestion
from growrag.macro_operators import GapField, OperatorSpec, QueryStep
from growrag.operator_bank import FrozenOperatorBank, operator_to_dict
from growrag.operator_loop import run_operator_episode


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _qid(number):
    return f"{number:024x}"


def _signature(digest="a" * 64):
    value = {
        "schema": ACTION_LIST_SCHEMA,
        "configuration": action_list_execution_configuration(),
        "files": dict.fromkeys(ACTION_LIST_METHOD_FILES, digest),
    }
    return {**value, "sha256": scoring.fingerprint(value)}


def _failed(qid):
    return {
        "question_id": qid,
        "arms": {arm: {"status": "failed", "feedback": None} for arm in scoring.ARMS},
    }


def _launch(runs, name, ids, manifest_sha, *, reports=None, signature=None, phase="source"):
    folder = runs / f"{ACTION_LIST.prefix}{name}"
    _write(
        folder / "launch_plan.json",
        {
            "protocol": ACTION_LIST.protocol,
            "profile": ACTION_LIST.name,
            "phase": phase,
            "model": "synthetic-fixed-model",
            "manifest_sha256": manifest_sha,
            "question_ids": ids,
            "arms": list(scoring.ARMS),
            "gold_loaded": False,
            "memory_updates": False,
            "execution_signature": signature or _signature(),
        },
    )
    values = reports if reports is not None else [_failed(qid) for qid in ids]
    _write(folder / "predictions.json", values)
    _write(
        folder / "predictions_frozen.json",
        {
            "phase": phase,
            "sha256": scoring.fingerprint(values),
            "question_ids": ids,
            "gold_loaded": False,
            "status": "completed",
            "cleanup_errors": [],
        },
    )
    return folder


class _SyntheticClient:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=json.dumps(self.value), request_id="unpaid-synthetic")


def _executed(qid):
    """Use the actual action-list decoder and unchanged execution DSL, not handbuilt events."""
    question = RuntimeQuestion(qid, "Where was Alice born?")
    stop = {
        "decision": "stop",
        "reason": "no_useful_query",
        "intent": "lookup",
        "constraints": [],
        "operator": None,
        "selected_operator": None,
        "gap": {},
        "bindings": [],
    }
    spec = OperatorSpec(
        "PROPOSED",
        "1",
        ("lookup",),
        (GapField("entity"),),
        (QueryStep("search", "{entity} birthplace"),),
    )

    def retrieve(query, top_k):
        return (
            (Evidence("e0", "Biography", 0, "Alice worked here."),)
            if query == question.text
            else (Evidence("e1", "Place", 0, "Alice was born here."),)
        )

    arms = {}
    for arm in scoring.ARMS:
        plan = (
            stop
            if arm != "fresh"
            else {
                **stop,
                "decision": "act",
                "reason": "missing_evidence",
                "operator": operator_to_dict(spec),
                "gap": {"entity": "Alice"},
            }
        )
        mode = "static" if arm == "static" else "fresh"
        client = _SyntheticClient(encode_wire_plan(plan, mode=mode))
        planner = ModelOperatorPlannerV3(client, mode=mode, trace_prefix="synthetic")
        episode = run_operator_episode(
            question,
            retrieve,
            planner,
            retrieval_budget=1 if arm == "base" else 3,
            max_decisions=1,
        )
        assert client.calls == (0 if arm == "base" else 1)
        arms[arm] = {
            "status": "completed",
            "question_id": qid,
            "arm": arm,
            "episode": json.loads(json.dumps(asdict(episode))),
            "reader": {"answer": "Blue Team" if arm == "fresh" else "unknown"},
            "feedback": None,
            "memory_updated": False,
        }
    return {"question_id": qid, "arms": arms}


@pytest.fixture
def source500(tmp_path, monkeypatch):
    ids = [_qid(i) for i in range(500)]
    manifest = {
        "roles": {"source": ids, "calibration": [_qid(600)], "evaluation": [_qid(700)]},
        "nested_source_ids": {str(size): ids[:size] for size in banks.SOURCE_SIZES},
    }
    path = tmp_path / "manifest.json"
    _write(path, manifest)
    for module in (scoring, banks):
        monkeypatch.setattr(module, "load_inputs", lambda path, phase: (manifest, ()))
    reads = []

    def synthetic_gold(root, received):
        assert received == manifest
        reads.append(len(ids))
        return {
            qid: {
                "id": qid,
                "answer": "Blue Team",
                "context": {"title": ["Fact"], "sentences": [["Blue Team won."]]},
                "supporting_facts": {"title": ["Fact"], "sent_id": [0]},
            }
            for qid in ids
        }, []

    monkeypatch.setattr(scoring, "_load_source_gold", synthetic_gold)
    reports = [_failed(qid) for qid in ids]
    reports[0], reports[60] = _executed(ids[0]), _executed(ids[60])
    launch = _launch(tmp_path / "runs", "source", ids, scoring._sha(path), reports=reports)
    return SimpleNamespace(
        root=tmp_path,
        manifest=manifest,
        manifest_path=path,
        reports=reports,
        launch=launch,
        reads=reads,
        feedback=tmp_path / "feedback",
        output=tmp_path / "banks",
    )


def _score(data, **kwargs):
    return scoring.score_operator_sources(
        data.root,
        data.manifest_path,
        data.root / "runs",
        data.feedback,
        profile=ACTION_LIST.name,
        **kwargs,
    )


def _build(data, **kwargs):
    return banks.build_operator_banks(
        data.root,
        data.manifest_path,
        data.feedback,
        data.output,
        expected_audit_sha256=scoring._sha(data.feedback / "audit.json"),
        profile=ACTION_LIST.name,
        **kwargs,
    )


def test_v3_dry_run_leaves_all_real_labels_sealed_and_legacy_schema_unchanged(source500):
    result = _score(source500)
    assert result["profile"] == ACTION_LIST.name
    assert result["protocol"] == ACTION_LIST.protocol
    assert result["execution_signature_sha256"] == _signature()["sha256"]
    assert result["source_count"] == 500 and result["gold_loaded"] is False
    assert not source500.feedback.exists() and source500.reads == []


def test_v3_499_terminal_sources_cannot_read_labels(source500):
    ids = source500.manifest["roles"]["source"][:-1]
    _launch(
        source500.root / "runs",
        "source",
        ids,
        scoring._sha(source500.manifest_path),
        reports=source500.reports[:-1],
    )
    with pytest.raises(ValueError, match="not all source IDs"):
        _score(source500, write=True)
    assert source500.reads == [] and not source500.feedback.exists()


def test_v3_changed_input_between_collection_and_label_gate_remains_sealed(source500, monkeypatch):
    original = scoring.collect_frozen_sources

    def changed(*args, **kwargs):
        collected = original(*args, **kwargs)
        path = source500.launch / "predictions.json"
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return collected

    monkeypatch.setattr(scoring, "collect_frozen_sources", changed)
    with pytest.raises(ValueError, match="before label gate"):
        _score(source500, write=True)
    assert source500.reads == [] and not source500.feedback.exists()


def test_v3_complete_scoring_and_four_independent_prefix_banks(source500, monkeypatch):
    result = _score(source500, write=True)
    assert source500.reads == [500]
    assert result["gold_loaded"] is True and result["api_calls"] == 0
    feedback = _read(source500.feedback / "feedback.json")
    assert all(
        set(metrics) == set(banks.METRICS)
        for arms in feedback.values()
        for metrics in arms.values()
    )
    assert feedback[_qid(0)]["fresh"]["answer_em"] == 1
    assert feedback[_qid(1)]["fresh"]["answer_em"] is None

    def no_labels(*args):
        pytest.fail("bank construction cannot open labels")

    monkeypatch.setattr(scoring, "_load_source_gold", no_labels)
    bundle = _build(source500, write=True)
    assert bundle["execution_signature_sha256"] == result["execution_signature_sha256"]
    assert bundle["profile"] == ACTION_LIST.name
    assert bundle["protocol"] == ACTION_LIST.protocol
    for size in banks.SOURCE_SIZES:
        bank = FrozenOperatorBank.from_json((source500.output / f"bank_{size}.json").read_text())
        assert bank.protocol_id == ACTION_LIST.protocol
        assert set(bank.allowed_source_ids) == set(source500.manifest["roles"]["source"][:size])
        assert bank.records[0].source_qids == ((_qid(0),) if size == 50 else (_qid(0), _qid(60)))
        assert bundle["banks"][str(size)]["cross_question_transfer_verified"] is False
    assert bundle["new_gold_loaded"] is False and bundle["evaluation_authorized"] is False


@pytest.mark.parametrize("field,value", [("profile", LEGACY.name), ("protocol", LEGACY.protocol)])
def test_v3_source_prefix_cannot_disguise_legacy_launch(source500, field, value):
    path = source500.launch / "launch_plan.json"
    plan = _read(path)
    plan[field] = value
    _write(path, plan)
    with pytest.raises(ValueError, match="profile|protocol"):
        _score(source500, write=True)
    assert source500.reads == []


def test_v3_two_source_batches_cannot_mix_method_signatures(source500):
    ids = source500.manifest["roles"]["source"]
    _launch(source500.root / "runs", "source", ids[:250], scoring._sha(source500.manifest_path))
    _launch(
        source500.root / "runs",
        "source_second",
        ids[250:],
        scoring._sha(source500.manifest_path),
        signature=_signature("b" * 64),
    )
    with pytest.raises(ValueError, match="fixed execution signature"):
        _score(source500, write=True)
    assert source500.reads == []


def test_v3_calibration_evaluation_and_legacy_source_do_not_satisfy_source_gate(source500):
    ids = source500.manifest["roles"]["source"]
    _launch(source500.root / "runs", "source", ids[:-1], scoring._sha(source500.manifest_path))
    for phase in ("calibration", "evaluation"):
        _launch(source500.root / "runs", phase, ids[-1:], "foreign", phase=phase)
    legacy = source500.root / "runs" / f"{LEGACY.prefix}source"
    _write(legacy / "launch_plan.json", {"protocol": LEGACY.protocol, "phase": "source"})
    with pytest.raises(ValueError, match="not all source IDs"):
        _score(source500, write=True)
    assert source500.reads == []


@pytest.mark.parametrize("tamper", ["signature", "profile", "report", "duplicate", "no_launch"])
def test_v3_builder_rejects_resigned_provenance_or_report_tampering(source500, tamper):
    _score(source500, write=True)
    path = source500.feedback / "audit.json"
    audit = _read(path)
    if tamper == "signature":
        audit["execution_signature_sha256"] = "b" * 64
    elif tamper == "profile":
        audit["profile"] = LEGACY.name
    elif tamper == "duplicate":
        audit["prediction_inputs"].append(deepcopy(audit["prediction_inputs"][0]))
    elif tamper == "no_launch":
        audit["prediction_inputs"] = [
            row
            for row in audit["prediction_inputs"]
            if not row["path"].endswith("launch_plan.json")
        ]
    else:
        report_path = source500.feedback / "source_reports.json"
        reports = _read(report_path)
        reports[0]["arms"]["fresh"]["reader"]["answer"] = "rewritten"
        _write(report_path, reports)
        audit["source_reports_sha256"] = scoring.fingerprint(reports)
        audit["artifacts"]["source_reports.json"]["sha256"] = scoring._sha(report_path)
    _write(path, audit)
    with pytest.raises(ValueError, match="source scoring|duplicate frozen"):
        _build(source500, write=True)
    assert not source500.output.exists()


@pytest.mark.parametrize("protocol", [LEGACY.protocol, STRUCTURED.protocol, "other"])
def test_v3_extractor_never_mixes_report_protocols(source500, protocol):
    qid = _qid(0)
    report = {**source500.reports[0], "phase": "source", "protocol": protocol}
    with pytest.raises(ValueError, match="source-phase"):
        build_source_bank((qid,), [report], {}, profile=ACTION_LIST.name)


def test_explicit_protocol_cannot_override_selected_extractor_profile():
    with pytest.raises(ValueError, match="protocol"):
        build_source_bank((_qid(0),), [], {}, protocol_id=ACTION_LIST.protocol)
    with pytest.raises(ValueError, match="protocol"):
        build_source_bank((_qid(0),), [], {}, protocol_id=LEGACY.protocol, profile=ACTION_LIST.name)


@pytest.mark.parametrize("operation", ["score", "collect", "extract", "build"])
def test_calibration_only_structured_profile_has_no_source_pipeline(operation, tmp_path):
    with pytest.raises(ValueError, match="calibration-only"):
        if operation == "score":
            scoring.score_operator_sources(
                tmp_path, tmp_path, tmp_path, tmp_path, profile=STRUCTURED.name
            )
        elif operation == "collect":
            scoring.collect_frozen_sources({}, "a" * 64, tmp_path, profile=STRUCTURED.name)
        elif operation == "extract":
            build_source_bank((_qid(0),), [], {}, profile=STRUCTURED.name)
        else:
            banks.build_operator_banks(
                tmp_path,
                tmp_path,
                tmp_path,
                tmp_path,
                expected_audit_sha256="a" * 64,
                profile=STRUCTURED.name,
            )
