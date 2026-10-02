"""External45 analysis contracts; only synthetic gold, never actual batch labels."""

import copy
import importlib.util
from pathlib import Path

import pytest
from test_history_candidate_terminal import auditor as synthetic_terminal
from test_history_candidate_terminal import complete_fixture as _complete_fixture
from test_history_candidate_terminal import interrupted as _interrupted_fixture
from test_score_history_candidates import write

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audits/score_history_candidate_partial.py"
SPEC = importlib.util.spec_from_file_location("candidate_partial_analysis", SCRIPT)
partial = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(partial)
complete_fixture = _complete_fixture
interrupted = _interrupted_fixture


@pytest.fixture
def sealed_partial(interrupted, monkeypatch):
    root, directories, manifest = interrupted
    monkeypatch.setattr(partial, "terminal", synthetic_terminal)
    amendment = root / partial.AMENDMENT
    amendment.parent.mkdir(parents=True)
    amendment.write_text("# Synthetic pre-label fixed45 amendment\n", encoding="utf-8")
    monkeypatch.setattr(partial, "AMENDMENT_SHA256", partial.frozen._sha(amendment))
    monkeypatch.setattr(
        partial, "PREFIX_SHA256", partial.frozen.fingerprint(manifest["question_ids"][:45])
    )
    synthetic_terminal.run(root, write=True)
    monkeypatch.setattr(
        partial,
        "TERMINAL_SEAL_SHA256",
        partial.frozen._sha(root / synthetic_terminal.OUTPUT / "terminal_audit_frozen.json"),
    )
    return root, directories, manifest


def synthetic_gold(ids):
    return {
        qid: {
            "id": qid,
            "answer": "answer",
            "context": {"title": ["Synthetic"], "sentences": [["Synthetic answer."]]},
            "supporting_facts": {"title": ["Synthetic"], "sent_id": [0]},
        }
        for qid in ids
    }


def test_partial_preflight_never_opens_labels(sealed_partial, monkeypatch):
    root, _, _ = sealed_partial
    monkeypatch.setattr(partial, "_load_gold", lambda *args: pytest.fail("no labels in preflight"))
    result = partial.run(root)
    assert result["gold_loaded"] is False and result["api_calls"] == 0
    assert result["planned_questions"] == 50 and result["complete_paired_questions"] == 45
    assert result["completed_method_reports"] == 225
    assert result["failed_method_reports"] == 1 and result["not_attempted_method_paths"] == 24
    assert result["paired_scored_totals"]["api_requests"] == 406
    assert result["campaign_totals_including_failed_http"]["api_requests"] == 407
    assert not (root / partial.OUTPUT).exists()


def test_synthetic_score_keeps_raw_controlled_cost_missing_and_originals(
    sealed_partial, monkeypatch
):
    root, directories, manifest = sealed_partial
    frozen = partial.frozen
    original = {str(p): frozen._sha(p) for d in directories for p in d.rglob("*") if p.is_file()}
    snapshot = frozen.source_snapshot(root)
    projected = []

    def labels(project, current_manifest, ids):
        assert project == root and current_manifest == manifest
        assert ids == manifest["question_ids"][:45]
        projected.extend(ids)
        return synthetic_gold(ids), {}

    monkeypatch.setattr(partial, "_load_gold", labels)
    result = partial.run(root, score=True)
    folder = root / partial.OUTPUT
    summary = frozen._read(folder / "SUMMARY.json")
    rows = frozen._read(folder / "per_question.json")
    missing = frozen._read(folder / "missing_questions.json")
    seal = frozen._read(folder / "feedback_frozen.json")
    assert projected == manifest["question_ids"][:45]
    assert result["gold_loaded"] is True and len(rows) == 45
    assert summary["questions"] == 45 and summary["completed_methods"] == 225
    assert summary["planned_questions"] == 50 and summary["complete_paired_questions"] == 45
    assert summary["failed_method_reports"] == 1 and summary["not_attempted_method_paths"] == 24
    assert summary["methods"]["history_body8"]["raw"]["answer_em"]["mean"] == 1.0
    assert summary["methods"]["history_body8"][frozen.PRIMARY]["answer_em"]["mean"] == 0.0
    assert summary["paired_scored_totals"]["api_requests"] == 406
    assert summary["campaign_totals_including_failed_http"]["api_requests"] == 407
    assert summary["failed_reader_costs"]["id_repaired"] is False
    assert summary["memory_updated"] is False and summary["prediction_modified"] is False
    assert summary["api_calls"] == 0 and summary["full50_score_gate_amended"] is False
    assert summary["continuation_authorized"] is False
    assert [r["question_id"] for r in missing] == manifest["question_ids"][45:]
    assert all(r["labels_loaded"] is False and "answer" not in r for r in missing)
    assert missing[0]["methods"]["base"] == "failed"
    assert missing[0]["methods"]["fresh"] == "not_attempted"
    assert "feedback_frozen.json" not in seal["files"]
    assert set(seal["files"]) == {
        "per_question.json",
        "per_question.md",
        "SUMMARY.json",
        "missing_questions.json",
        "audit.json",
    }
    assert all(frozen._sha(folder / name) == sha for name, sha in seal["files"].items())
    assert original == {
        str(p): frozen._sha(p) for d in directories for p in d.rglob("*") if p.is_file()
    }
    assert frozen.source_snapshot(root) == snapshot
    assert not (root / frozen.OUTPUT).exists()
    assert (
        summary["candidate_policy_contrasts"]["history_body3_to_history_body8"]["raw"]["repaired"]
        == 0
    )
    assert (
        summary["candidate_policy_contrasts"]["history_body3_to_history_body8"][frozen.PRIMARY][
            "repaired"
        ]
        == 0
    )
    assert summary["base_fresh_wrong_history_correct"]["raw"]["history_body8"]["count"] == 0
    assert (
        summary["base_fresh_wrong_history_correct"][frozen.PRIMARY]["history_body8"]["count"] == 0
    )
    bounds = summary["full50_missing_result_bounds"]["by_metric_policy"]["raw"]["methods"][
        "history_body8"
    ]
    assert bounds["worst_full50"] == 0.9 and bounds["best_full50"] == 1.0
    with pytest.raises(FileExistsError):
        partial.run(root, score=True)
    assert len(projected) == 45
    with pytest.raises(ValueError, match="all 250"):
        frozen.run(root, score=True)


def test_original_input_drift_fails_before_gold(sealed_partial, monkeypatch):
    root, directories, manifest = sealed_partial
    path = directories[0] / f"{manifest['question_ids'][0]}_base.json"
    value = partial.frozen._read(path)
    value["reader"]["answer"] = "tampered"
    write(path, value)
    monkeypatch.setattr(partial, "_load_gold", lambda *args: pytest.fail("gold must stay closed"))
    with pytest.raises(ValueError, match="sealed inputs changed"):
        partial.run(root, score=True)
    assert not (root / partial.OUTPUT).exists()


def test_wrong_amendment_fails_before_terminal_or_labels(tmp_path, monkeypatch):
    path = tmp_path / partial.AMENDMENT
    path.parent.mkdir(parents=True)
    path.write_text("wrong amendment", encoding="utf-8")
    monkeypatch.setattr(partial, "_load_gold", lambda *args: pytest.fail("gold must stay closed"))
    with pytest.raises(ValueError, match="hard-pinned version"):
        partial.run(tmp_path, score=True)


def test_gold_projection_excludes_five_missing_and_pins_every_shard(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = [f"{i:024x}" for i in range(50)]
    monkeypatch.setattr(partial.terminal, "FAILED_ID", ids[45])
    parent = tmp_path / "data/synthetic_train"
    parent.mkdir(parents=True)
    shard = parent / "train.parquet"
    rows = list(synthetic_gold(ids).values())
    for row in rows[45:]:
        row["answer"] = "SYNTHETIC MISSING GOLD MUST NEVER APPEAR IN PROJECTION"
    pq.write_table(pa.Table.from_pylist(rows), shard)
    provenance = parent / "mirror_provenance.json"
    write(
        provenance,
        {
            "official_split": "train",
            "shards": [{"file": shard.name, "sha256": partial.frozen._sha(shard)}],
        },
    )
    manifest = {
        "question_ids": ids,
        "official_split": "train",
        "source_provenance": {
            "path": provenance.relative_to(tmp_path).as_posix(),
            "sha256": partial.frozen._sha(provenance),
        },
        "parquet_inputs": [
            {"path": shard.relative_to(tmp_path).as_posix(), "sha256": partial.frozen._sha(shard)}
        ],
    }
    gold, inputs = partial._load_gold(tmp_path, manifest, ids[:45])
    assert set(gold) == set(ids[:45]) and not set(gold) & set(ids[45:])
    assert all(v["answer"] == "answer" for v in gold.values())
    assert set(inputs) == {str(shard), str(provenance)}
    for invalid in (ids, ids[1:46], ids[:44], list(reversed(ids[:45]))):
        with pytest.raises(ValueError, match="fixed train45 prefix"):
            partial._load_gold(tmp_path, manifest, invalid)
    wrong = copy.deepcopy(manifest)
    wrong["parquet_inputs"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="shard/provenance/manifest"):
        partial._load_gold(tmp_path, wrong, ids[:45])


def test_missing_bounds_do_not_impute_unknown_scored_labels():
    methods = partial.frozen.study.METHODS
    rows = [
        {
            key: {m: {"answer_em": None if i == 0 and m == "base" else 0.0} for m in methods}
            for key in ("raw_scores", "controlled_reader_scores")
        }
        for i in range(45)
    ]
    result = partial._missing_bounds(rows)
    assert result["by_metric_policy"]["raw"]["methods"]["base"]["observed_valid_n"] == 44
    assert result["by_metric_policy"]["raw"]["methods"]["base"]["worst_full50"] is None
    assert (
        result["by_metric_policy"]["raw"]["paired"]["base_to_history_body3"]["worst_full50_delta"]
        is None
    )


def test_policy_contrasts_and_unique_history_opportunities_are_descriptive():
    rows = []
    outcomes = (
        {
            "base": 0.0,
            "fresh": 0.0,
            "history_legacy3": 0.0,
            "history_body3": 0.0,
            "history_body8": 1.0,
        },
        {
            "base": 1.0,
            "fresh": 0.0,
            "history_legacy3": 1.0,
            "history_body3": 1.0,
            "history_body8": 0.0,
        },
        {method: None for method in partial.frozen.study.METHODS},
    )
    for i, outcome in enumerate(outcomes):
        row = {"question_id": f"synthetic{i}", "usage": {}}
        for key in ("raw_scores", "controlled_reader_scores"):
            row[key] = {
                method: {metric: em for metric in partial.frozen.METRICS}
                for method, em in outcome.items()
            }
        for method in partial.frozen.study.METHODS:
            row["usage"][method] = {
                "reader_input_sha256": method,
                "api_requests": 1,
                "retrieval_calls": 1,
                "estimated_actual_cny": 0.001,
            }
        rows.append(row)
    summary = {}
    partial._extra_analysis(rows, summary)
    contrast = summary["candidate_policy_contrasts"]["history_body3_to_history_body8"]["raw"]
    assert contrast["repaired"] == contrast["harmed"] == 1
    assert contrast["delta"]["answer_em"] == {"mean": 0.0, "valid_n": 2}
    assert summary["base_fresh_wrong_history_correct"]["raw"]["history_body8"] == {
        "count": 1,
        "question_ids": ["synthetic0"],
    }
    assert summary["base_fresh_wrong_history_correct"]["raw"]["history_body3"]["count"] == 0
