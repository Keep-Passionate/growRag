"""Synthetic-only held-out scoring tests; no real evaluation labels are opened."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from growrag.experiments import score_operator_evaluation as module
from growrag.experiments.operator_evaluation_freeze import ANALYSIS_FILES, analysis_signature
from growrag.experiments.operator_execution_signature import METHOD_FILES, execution_signature
from growrag.macro_operators import GapField, OperatorSpec, QueryStep
from growrag.operator_bank import FrozenOperatorBank, OperatorRecord, operator_to_dict


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _qid(i):
    return f"{i:024x}"


def _report(qid, arm, status="completed"):
    return {
        "question_id": qid,
        "arm": arm,
        "status": status,
        "feedback": None,
        "memory_updated": False,
        "calls": [],
        "episode": {"question_id": qid, "evidence": [], "searches": [], "proposals": []}
        if status == "completed"
        else None,
        "reader": {"answer": "yes", "visible_evidence_ids": [], "evidence_windows": []}
        if status == "completed"
        else None,
        "error_type": "SyntheticFailure" if status == "failed" else None,
    }


def _gold(qid):
    return {
        "id": qid,
        "answer": "yes",
        "context": {"title": ["Synthetic"], "sentences": [["Evidence."]]},
        "supporting_facts": {"title": ["Synthetic"], "sent_id": [0]},
    }


def _setup(tmp_path, monkeypatch, count=3):
    monkeypatch.setattr(module, "_EVALUATION_COUNT", count)
    methods = (*METHOD_FILES, "src/growrag/experiments/run_operator_study.py")
    for name in methods:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic code", encoding="utf-8")
    signature = execution_signature(tmp_path)
    # The scorer verifies its actually imported analysis against the frozen copy.
    actual_root = Path(module.__file__).resolve().parents[3]
    for name in ANALYSIS_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((actual_root / name).read_bytes())
    ids = [_qid(i) for i in range(1000, 1000 + count)]
    source = [_qid(i) for i in range(500)]
    manifest = {
        "roles": {"source": source, "calibration": [_qid(600)], "evaluation": ids},
        "nested_source_ids": {str(size): source[:size] for size in (50, 100, 250, 500)},
    }
    manifest_path = tmp_path / "data/manifest.json"
    _write(manifest_path, manifest)
    runs = tmp_path / "runs"
    runs.mkdir()
    spec = OperatorSpec(
        "CURRENT_GAP",
        "1",
        ("lookup",),
        (GapField("term"),),
        (QueryStep("search", "{original_question} {term}"),),
    )
    banks = {}
    for size in (50, 100, 250, 500):
        record = OperatorRecord(
            spec,
            (source[0],),
            module.PROTOCOL,
            "audit/source-0",
            status="validated",
            validation_ref="audit/source-0/publication",
        )
        bank = FrozenOperatorBank(module.PROTOCOL, tuple(source[:size]), (record,))
        path = tmp_path / "banks" / f"bank_{size}.json"
        _write(path, bank.to_dict())
        banks[str(size)] = {"fingerprint": bank.fingerprint, "file_sha256": module._sha(path)}
    freeze = {
        "evaluation_ids": ids,
        "evaluation_order_sha256": module.fingerprint(ids),
        "expected": {"manifest": module._sha(manifest_path), "execution": signature["sha256"]},
        "execution_signature": signature,
        "runner_sha256": hashlib.sha256(b"synthetic code").hexdigest(),
        "paths": {"runs": "runs", "manifest": "data/manifest.json", "banks": "banks"},
        "banks": banks,
        "analysis": analysis_signature(tmp_path),
    }
    certificate = tmp_path / "certificate.json"
    _write(certificate, {"freeze": freeze})
    certsha = module._sha(certificate)
    monkeypatch.setattr(module, "_validate_freeze", lambda root, path, expected: deepcopy(freeze))
    snapshot_files = {
        name: {"text": "synthetic code", "sha256": hashlib.sha256(b"synthetic code").hexdigest()}
        for name in methods
    }
    snapshot = {"files": snapshot_files, "sha256": module.fingerprint(snapshot_files)}

    def launch(name, chosen, *, fail_at=None, omit_last=False, plan_change=None):
        folder = runs / f"{module.PREFIX}{name}"
        plan = {
            **signature["configuration"],
            "protocol": module.PROTOCOL,
            "phase": "evaluation",
            "evaluation_freeze_sha256": certsha,
            "evaluation_freeze_path": str(certificate),
            "evaluation_order_sha256": freeze["evaluation_order_sha256"],
            "manifest_sha256": freeze["expected"]["manifest"],
            "execution_signature": signature,
            "arms": list(module.ARMS),
            "question_ids": chosen,
            "bank_sha256": {
                f"memory{size}": entry["fingerprint"] for size, entry in freeze["banks"].items()
            },
            "bank_file_sha256": {
                f"memory{size}": entry["file_sha256"] for size, entry in freeze["banks"].items()
            },
            "source_sha256": snapshot["sha256"],
            "gold_loaded": False,
            "memory_updates": False,
        }
        plan.update(plan_change or {})
        reports = []
        for number, qid in enumerate(chosen):
            if omit_last and number == len(chosen) - 1:
                break
            arms = {}
            for arm in module.ARMS:
                status = "failed" if fail_at == (qid, arm) else "completed"
                arms[arm] = _report(qid, arm, status)
                _write(folder / f"{qid}_{arm}.json", arms[arm])
                if status == "failed":
                    break
            item = {"question_id": qid, "arms": arms}
            reports.append(item)
            _write(folder / f"checkpoint_{number:04d}.json", item)
            if any(outcome["status"] == "failed" for outcome in arms.values()):
                break
        _write(folder / "launch_plan.json", plan)
        _write(folder / "evaluation_freeze_verified.json", freeze)
        _write(folder / "source_snapshot.json", snapshot)
        _write(folder / "predictions.json", reports)
        _write(
            folder / "predictions_frozen.json",
            {
                "sha256": module.fingerprint(reports),
                "question_ids": [item["question_id"] for item in reports],
                "phase": "evaluation",
                "gold_loaded": False,
                "status": "failed" if fail_at or omit_last else "completed",
                "cleanup_errors": [],
            },
        )
        _write(folder / "final_budget.json", {"calls": [], "api_requests": 0})
        _write(
            folder / "events.jsonl",
            {
                "kind": "exit",
                "status": "failed" if fail_at or omit_last else "completed",
                "requests": 0,
            },
        )
        return folder

    return {
        "root": tmp_path,
        "manifest": manifest,
        "freeze": freeze,
        "runs": runs,
        "certificate": certificate,
        "sha": certsha,
        "ids": ids,
        "launch": launch,
        "spec": operator_to_dict(spec),
    }


@pytest.fixture
def data(tmp_path, monkeypatch):
    return _setup(tmp_path, monkeypatch)


def _collect(data):
    return module.collect_frozen_evaluation(
        data["root"],
        data["manifest"],
        data["freeze"],
        data["runs"],
        data["certificate"],
        data["sha"],
    )


def _score(data, *, write=False):
    return module.score_operator_evaluation(
        data["root"],
        data["certificate"],
        data["runs"],
        data["root"] / "scored",
        expected_certificate_sha256=data["sha"],
        write=write,
    )


def test_all500_seven_arm_synthetic_predictions_can_be_collected(tmp_path, monkeypatch):
    data = _setup(tmp_path, monkeypatch, count=500)
    for offset in range(0, 500, 25):
        data["launch"](f"batch{offset // 25:02d}", data["ids"][offset : offset + 25])
    reports, audited = _collect(data)
    assert len(reports) == 500 and sum(len(row["arms"]) for row in reports) == 3500
    assert len(audited) > 3500


def test_default_preflight_never_loads_gold_or_writes(data, monkeypatch):
    data["launch"]("whole", data["ids"])
    monkeypatch.setattr(module, "_load_dev_gold", lambda *args: pytest.fail("gold opened"))
    result = _score(data)
    assert result["gold_loaded"] is False and result["arm_count"] == 7
    assert result["all_questions_terminal"] is True
    assert result["all_arm_predictions_completed"] is True
    assert len(result["scoring_implementation"]) == 6
    assert {
        entry["path"].replace("\\", "/").rsplit("/", 1)[-1]
        for entry in result["scoring_implementation"]
    } == {
        "score_operator_evaluation.py",
        "operator_evaluation_summary.py",
        "operator_evaluation_inference.py",
        "score_operator_sources.py",
        "hotpot.py",
        "shared_hotpot_dev.py",
    }
    assert all(len(entry["sha256"]) == 64 for entry in result["scoring_implementation"])
    assert not (data["root"] / "scored").exists()


def test_missing_unstarted_question_cannot_unlock_labels(data, monkeypatch):
    data["launch"]("partial", data["ids"], omit_last=True)
    monkeypatch.setattr(module, "_load_dev_gold", lambda *args: pytest.fail("gold opened"))
    with pytest.raises(ValueError, match="need terminal reports"):
        _score(data, write=True)


@pytest.mark.parametrize("part", ["policy", "files", "numpy_version"])
def test_analysis_mismatch_rejected_before_labels(data, monkeypatch, part):
    if part == "numpy_version":
        data["freeze"]["analysis"][part] = "another-version"
    elif part == "files":
        data["freeze"]["analysis"][part][ANALYSIS_FILES[0]] = "f" * 64
    else:
        data["freeze"]["analysis"][part]["unexpected_change"] = True
    monkeypatch.setattr(module, "_load_dev_gold", lambda *args: pytest.fail("gold opened"))
    with pytest.raises(ValueError, match="scoring policy/code/library"):
        _score(data, write=True)
    assert not (data["root"] / "scored").exists()


def test_failed_last_question_explicitly_marks_unstarted_arms_unknown(data, monkeypatch):
    data["launch"]("whole", data["ids"], fail_at=(data["ids"][-1], "fresh"))
    reports, _ = _collect(data)
    last = reports[-1]["arms"]
    assert last["fresh"]["status"] == "failed"
    assert last["static"]["status"] == "failure_induced_unstarted"
    assert last["static"]["caused_by_arm"] == "fresh"
    monkeypatch.setattr(
        module, "_load_dev_gold", lambda *args: ({qid: _gold(qid) for qid in data["ids"]}, [])
    )
    audit = _score(data, write=True)
    assert audit["all_questions_terminal"] is True
    assert audit["all_arm_predictions_completed"] is False
    assert audit["execution_counts"]["fresh"] == {
        "completed": 2,
        "failed": 1,
        "failure_induced_unstarted": 0,
    }
    assert audit["execution_counts"]["static"] == {
        "completed": 2,
        "failed": 0,
        "failure_induced_unstarted": 1,
    }
    feedback = json.loads((data["root"] / "scored/feedback.json").read_text())
    assert feedback[data["ids"][-1]]["static"]["status"] == "failure_induced_unstarted"
    assert feedback[data["ids"][-1]]["static"]["answer_em"] is None
    assert feedback[data["ids"][-1]]["fresh"]["answer_f1"] is None


def test_repeated_question_is_rejected_instead_of_best_run_selection(data):
    data["launch"]("first", data["ids"])
    data["launch"]("repeat", [data["ids"][0]])
    with pytest.raises(ValueError, match="repeated"):
        _collect(data)


@pytest.mark.parametrize(
    "change",
    [
        {"evaluation_freeze_sha256": "f" * 64},
        {"manifest_sha256": "f" * 64},
        {"model": "other"},
        {"bank_sha256": {}},
        {"bank_file_sha256": {}},
        {"gold_loaded": True},
        {"memory_updates": True},
        {"top_k": 3},
        {"evaluation_order_sha256": "f" * 64},
        {"arms": list(reversed(module.ARMS))},
    ],
)
def test_frozen_launch_mismatches_fail_before_scoring(data, change):
    data["launch"]("wrong", data["ids"], plan_change=change)
    with pytest.raises(ValueError):
        _collect(data)


def test_interleaved_or_changed_batch_order_is_rejected(data):
    data["launch"]("wrong", [data["ids"][0], data["ids"][2]])
    with pytest.raises(ValueError, match="contiguous"):
        _collect(data)


def test_impossible_over25_question_batch_is_rejected(tmp_path, monkeypatch):
    data = _setup(tmp_path, monkeypatch, count=26)
    data["launch"]("oversized", data["ids"])
    with pytest.raises(ValueError, match="1..25"):
        _collect(data)


def test_source_overlap_rejected_before_any_labels(data):
    data["manifest"]["roles"]["source"] = [data["ids"][0]]
    with pytest.raises(ValueError, match="independent"):
        _collect(data)


def test_mutated_checkpoint_or_per_arm_report_is_detected(data):
    folder = data["launch"]("whole", data["ids"])
    _write(folder / "checkpoint_0000.json", {})
    with pytest.raises(ValueError, match="checkpoint"):
        _collect(data)


def test_evaluation_feedback_or_memory_update_in_arm_is_rejected(data):
    folder = data["launch"]("whole", data["ids"])
    reports = json.loads((folder / "predictions.json").read_text())
    reports[0]["arms"]["base"]["memory_updated"] = True
    _write(folder / "predictions.json", reports)
    _write(folder / "checkpoint_0000.json", reports[0])
    seal = json.loads((folder / "predictions_frozen.json").read_text())
    seal["sha256"] = module.fingerprint(reports)
    _write(folder / "predictions_frozen.json", seal)
    with pytest.raises(ValueError, match="invalid/updated"):
        _collect(data)


def test_runner_snapshot_text_must_match_frozen_sha(data):
    folder = data["launch"]("whole", data["ids"])
    snapshot = json.loads((folder / "source_snapshot.json").read_text())
    snapshot["files"]["src/growrag/experiments/run_operator_study.py"]["text"] = "changed wrapper"
    snapshot["sha256"] = module.fingerprint(snapshot["files"])
    _write(folder / "source_snapshot.json", snapshot)
    plan = json.loads((folder / "launch_plan.json").read_text())
    plan["source_sha256"] = snapshot["sha256"]
    _write(folder / "launch_plan.json", plan)
    with pytest.raises(ValueError, match="runner version"):
        _collect(data)


def _replace_episode(folder, data, arm, episode):
    """Keep all synthetic report copies/seals consistent to exercise semantic checks."""
    reports = json.loads((folder / "predictions.json").read_text())
    reports[0]["arms"][arm]["episode"] = episode
    _write(folder / "predictions.json", reports)
    _write(folder / "checkpoint_0000.json", reports[0])
    _write(folder / f"{data['ids'][0]}_{arm}.json", reports[0]["arms"][arm])
    seal = json.loads((folder / "predictions_frozen.json").read_text())
    seal["sha256"] = module.fingerprint(reports)
    _write(folder / "predictions_frozen.json", seal)


def _reuse_episode(data, *, executed=True):
    return {
        "question_id": data["ids"][0],
        "evidence": [],
        "searches": [{"step": 0}, *([{"step": 1}] if executed else [])],
        "proposals": [{"origin": "reuse", "spec": deepcopy(data["spec"])}],
    }


def test_executed_reuse_matches_complete_frozen_published_spec(data):
    folder = data["launch"]("whole", data["ids"])
    _replace_episode(folder, data, "memory50", _reuse_episode(data))
    reports, _ = _collect(data)
    assert reports[0]["arms"]["memory50"]["episode"]["searches"][-1]["step"] == 1


def test_same_operator_id_but_changed_rule_is_not_frozen_reuse(data):
    folder = data["launch"]("whole", data["ids"])
    episode = _reuse_episode(data)
    episode["proposals"][0]["spec"]["steps"][0]["template"] = "{term} changed"
    _replace_episode(folder, data, "memory50", episode)
    with pytest.raises(ValueError, match="executed reuse spec"):
        _collect(data)


def test_executed_reuse_cannot_claim_an_unpublished_spec(data):
    outcome = _report(data["ids"][0], "memory50")
    outcome["episode"] = _reuse_episode(data)
    with pytest.raises(ValueError, match="executed reuse spec"):
        module._check_executed_reuse(outcome, "memory50", {"memory50": []})


def test_executed_reuse_does_not_coerce_boolean_to_integer(data):
    outcome = _report(data["ids"][0], "memory50")
    outcome["episode"] = _reuse_episode(data)
    outcome["episode"]["proposals"][0]["spec"]["gap_schema"][0]["required"] = 1
    with pytest.raises(ValueError, match="executed reuse spec"):
        module._check_executed_reuse(outcome, "memory50", {"memory50": [data["spec"]]})


def test_nonmemory_arm_cannot_claim_reuse_even_if_unexecuted(data):
    folder = data["launch"]("whole", data["ids"])
    _replace_episode(folder, data, "fresh", _reuse_episode(data, executed=False))
    with pytest.raises(ValueError, match="non-memory arm"):
        _collect(data)


def test_unexecuted_rejected_proposal_is_not_treated_as_actual_reuse(data):
    folder = data["launch"]("whole", data["ids"])
    episode = _reuse_episode(data, executed=False)
    episode["proposals"][0]["spec"] = {"not": "published"}
    episode["stop_reason"] = "plan_rejected"
    _replace_episode(folder, data, "memory50", episode)
    _collect(data)


def test_failed_reader_retained_episode_still_requires_published_reuse(data):
    folder = data["launch"]("first", data["ids"], fail_at=(data["ids"][0], "memory50"))
    data["launch"]("later", data["ids"][1:])
    episode = _reuse_episode(data)
    episode["proposals"][0]["spec"]["version"] = "changed"
    _replace_episode(folder, data, "memory50", episode)
    with pytest.raises(ValueError, match="executed reuse spec"):
        _collect(data)


def test_search_cannot_point_to_missing_proposal(data):
    folder = data["launch"]("whole", data["ids"])
    episode = _reuse_episode(data)
    episode["searches"][1]["step"] = 2
    _replace_episode(folder, data, "memory50", episode)
    with pytest.raises(ValueError, match="absent proposal"):
        _collect(data)


def test_bank_file_modification_fails_before_labels(data):
    data["launch"]("whole", data["ids"])
    path = data["root"] / "banks/bank_50.json"
    bank = FrozenOperatorBank.from_json(path.read_text())
    empty = FrozenOperatorBank(bank.protocol_id, bank.allowed_source_ids, ())
    _write(path, empty.to_dict())
    with pytest.raises(ValueError, match="bank changed"):
        _collect(data)


def test_unassigned_failed_call_in_final_ledger_cannot_be_lost(data):
    folder = data["launch"]("whole", data["ids"])
    _write(folder / "final_budget.json", {"calls": [{"trace_id": "unassigned"}], "api_requests": 1})
    with pytest.raises(ValueError, match="owned calls differ"):
        _collect(data)


def test_final_exit_must_agree_with_request_ledger_and_seal(data):
    folder = data["launch"]("whole", data["ids"])
    _write(folder / "events.jsonl", {"kind": "exit", "status": "completed", "requests": 1})
    with pytest.raises(ValueError, match="terminal exit"):
        _collect(data)


def test_explicit_score_preserves_bad_annotation_and_never_overwrites(data, monkeypatch):
    data["launch"]("whole", data["ids"])
    gold = {qid: _gold(qid) for qid in data["ids"]}
    gold[data["ids"][1]]["supporting_facts"]["sent_id"][0] = 99
    monkeypatch.setattr(module, "_load_dev_gold", lambda *args: (gold, []))
    result = _score(data, write=True)
    feedback = json.loads((data["root"] / "scored/feedback.json").read_text())
    assert feedback[data["ids"][0]]["base"]["answer_em"] == 1.0
    assert feedback[data["ids"][1]]["base"]["answer_em"] is None
    assert feedback[data["ids"][1]]["base"]["annotation_status"] == "invalid"
    assert result["memory_updated"] is False and result["raw_predictions_modified"] is False
    assert "summary.json" in result["artifacts"]
    assert "inference.json" in result["artifacts"]
    inference = json.loads((data["root"] / "scored/inference.json").read_text())
    assert inference["question_count"] == len(data["ids"])
    assert inference["policy"] == result["analysis"]["policy"]
    with pytest.raises(FileExistsError):
        _score(data, write=True)


def test_input_or_bank_change_during_scoring_prevents_publication(data, monkeypatch):
    folder = data["launch"]("whole", data["ids"])

    def mutate(*args):
        (folder / "predictions.json").write_text("[]")
        return {qid: _gold(qid) for qid in data["ids"]}, []

    monkeypatch.setattr(module, "_load_dev_gold", mutate)
    with pytest.raises(ValueError, match="input changed"):
        _score(data, write=True)
    assert not (data["root"] / "scored").exists()


def test_dev_loader_filters_only_frozen_evaluation_ids_and_checks_source_hashes(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    folder = tmp_path / "data/hotpotqa/official_dev_v1"
    folder.mkdir(parents=True)
    shard = folder / "dev.parquet"
    pq.write_table(pa.Table.from_pylist([_gold(_qid(i)) for i in (1, 2, 3)]), shard)
    provenance = folder / "mirror_provenance.json"
    _write(
        provenance,
        {"official_split": "dev", "shards": [{"file": shard.name, "sha256": module._sha(shard)}]},
    )
    manifest = {
        "roles": {"evaluation": [_qid(2)]},
        "input_files": [
            {"path": p.relative_to(tmp_path).as_posix(), "sha256": module._sha(p)}
            for p in (provenance, shard)
        ],
    }
    gold, audit = module._load_dev_gold(tmp_path, manifest)
    assert set(gold) == {_qid(2)} and len(audit) == 2
    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="shard differs"):
        module._load_dev_gold(tmp_path, manifest)


def test_cli_default_outputs_summary_without_ids_and_does_not_score(data, monkeypatch, capsys):
    data["launch"]("whole", data["ids"])
    monkeypatch.setattr(module, "_load_dev_gold", lambda *args: pytest.fail("gold opened"))
    module.main(
        [
            "--root",
            str(data["root"]),
            "--certificate",
            str(data["certificate"]),
            "--expected-certificate-sha256",
            data["sha"],
            "--runs",
            "runs",
            "--output-dir",
            "scored",
        ]
    )
    text = capsys.readouterr().out
    assert data["ids"][0] not in text
    assert json.loads(text)["gold_loaded"] is False
