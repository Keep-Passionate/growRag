"""Synthetic diagnostics: no real cohort, model, gold-file loading or API."""

import hashlib
import json
from copy import deepcopy
from statistics import mean

import pytest

from growrag.experiments import research_diagnostic as diagnostic


def row(qid, f=1, h=1, *, executed=False):
    scores = {
        m: {
            "answer_em": h if m == diagnostic.HISTORY else f,
            "answer_f1": h if m == diagnostic.HISTORY else f,
            "status": "completed",
            "raw_support_recall": 0.5,
            "visible_support_recall": 0.5,
        }
        for m in diagnostic.METHODS
    }
    return {
        "question_id": qid,
        "question": "Which fictional city is Avery from?",
        "reference_answer": "Stonebridge",
        "raw_scores": deepcopy(scores),
        "controlled_reader_scores": scores,
        "canonical_reader_method": {
            m: m if h != f and m == diagnostic.HISTORY else "base" for m in diagnostic.METHODS
        },
        "answers": {m: "raw " + m for m in diagnostic.METHODS},
        "model_support_claims": dict.fromkeys(diagnostic.METHODS, True),
        "usage": {
            m: {
                "reader_input_sha256": ("b" if h != f and m == diagnostic.HISTORY else "a") * 64,
                "api_requests": 2,
                "input_tokens": 10,
                "output_tokens": 5,
                "retrieval_calls": 2 if executed and m == diagnostic.HISTORY else 1,
                "estimated_actual_cny": 0.001,
                "reserved_cny": 0.01,
                "unknown_cost_requests": 0,
                "queries": ["synthetic"],
                "selected_cards": [],
                "executed_cards": [{"search_executed": executed}]
                if m == diagnostic.HISTORY
                else [],
            }
            for m in diagnostic.METHODS
        },
    }


def summary(rows):
    return {
        "gold_ids": [r["question_id"] for r in rows],
        "joint_complete_questions": len(rows),
        "methods": {
            m: {
                label: {
                    metric: {"mean": mean(r[key][m][metric] for r in rows)}
                    for metric in ("answer_em", "answer_f1")
                }
                for key, label in (
                    ("raw_scores", "raw"),
                    ("controlled_reader_scores", "controlled_reader"),
                )
            }
            for m in diagnostic.METHODS
        },
    }


def test_gain_loss_oracle():
    rows = [row("win", f=0, h=1), row("loss", f=1, h=0), row("tie")]
    result = diagnostic.paired_metrics(rows, "controlled_reader_scores")
    assert result["f1_positive_gain"] == pytest.approx(1 / 3)
    assert result["f1_negative_loss"] == pytest.approx(1 / 3)
    assert result["f1_net_gain"] == 0
    assert result["fixed_outcome_oracle_f1"] == 1
    assert result["em_cells"]["history_only"] == 1


def test_zero_opportunity_is_not_fabricated():
    result = diagnostic.paired_metrics([row("loss", f=1, h=0)], "controlled_reader_scores")
    assert result["em_cells"]["history_only"] == 0
    assert result["f1_positive_gain"] == 0
    assert result["fresh_f1"] == result["fixed_outcome_oracle_f1"]


def test_chosen_or_repeated_is_not_extra_search():
    r = row("synthetic")
    r["usage"][diagnostic.HISTORY]["selected_cards"] = [{"selected_card_id": "card1"}]
    r["usage"][diagnostic.HISTORY]["executed_cards"] = [{"search_executed": False}]
    assert not diagnostic.has_history_search(r)
    assert "history_no_extra_search" in diagnostic.observations(r)


def test_raw_and_controlled_answers_are_distinct():
    flat = diagnostic.flat_row(row("synthetic"))
    assert flat["history_body8_raw_answer"] == "raw history_body8"
    assert flat["history_body8_controlled_answer"] == "raw base"


def test_sampling_is_deterministic_and_keeps_all_executions():
    rows = [row(f"correct{i}", executed=i < 8) for i in range(30)]
    rows += [row(f"wrong{i}", f=0, h=0, executed=i < 7) for i in range(25)]
    rows += [row(f"loss{i}", f=1, h=0, executed=i == 0) for i in range(13)]
    selected = diagnostic.select_diagnostic(rows)
    assert len(selected) == 50
    selected_ids = {r["question_id"] for r in selected}
    assert selected_ids == {r["question_id"] for r in diagnostic.select_diagnostic(rows[::-1])}
    assert {r["question_id"] for r in rows if diagnostic.has_history_search(r)} <= selected_ids


def test_insufficient_quota_fails_instead_of_borrowing_ids():
    with pytest.raises(ValueError, match="quota"):
        diagnostic.select_diagnostic([row("only-one")])


@pytest.mark.parametrize(
    "change", ["duplicates", "coverage", "mean", "failed", "nan", "bool", "canonical"]
)
def test_row_integrity_rejects_drift(change):
    rows = [row("a"), row("b")]
    s = summary(rows)
    if change == "duplicates":
        rows[1]["question_id"] = "a"
    elif change == "coverage":
        s["gold_ids"] = ["a"]
    elif change == "mean":
        s["methods"]["base"]["raw"]["answer_f1"]["mean"] = 0
    elif change == "failed":
        rows[1]["raw_scores"]["base"]["status"] = "failed"
    elif change in ("nan", "bool"):
        rows[1]["raw_scores"]["base"]["answer_f1"] = float("nan") if change == "nan" else True
    else:
        rows[1]["usage"][diagnostic.HISTORY]["reader_input_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        diagnostic.validate_rows(rows, s)


def test_valid_rows_pass():
    rows = [row("a"), row("b", f=0, h=0)]
    diagnostic.validate_rows(rows, summary(rows))


def test_export_seal_mismatch_stops_before_output(tmp_path):
    folder = tmp_path / diagnostic.FEEDBACK
    folder.mkdir(parents=True)
    files = {
        name: "b" * 64
        for name in ("audit.json", "SUMMARY.json", "per_question.json", "missing_questions.json")
    }
    for name in files:
        (folder / name).write_text("{}", encoding="utf-8")
    (folder / "feedback_frozen.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    with pytest.raises(ValueError, match="seal drift"):
        diagnostic.run(tmp_path)
    assert not (tmp_path / diagnostic.OUTPUT).exists()


def test_never_overwrite_existing_diagnostic(tmp_path):
    (tmp_path / diagnostic.OUTPUT).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        diagnostic.run(tmp_path)


def test_output_stays_inside_runs(tmp_path):
    with pytest.raises(ValueError, match="direct runs child"):
        diagnostic.run(tmp_path, "../elsewhere")


def test_hash_is_file_bytes(tmp_path):
    p = tmp_path / "known.txt"
    p.write_bytes(b"known")
    assert diagnostic._sha(p) == hashlib.sha256(b"known").hexdigest()


def test_complete_export_only_uses_sealed_rows(tmp_path, monkeypatch):
    """Complete export exercises hashes, reports, CSV and sealing without a gold reader."""
    monkeypatch.setattr(diagnostic, "QUOTAS", {"both_correct": 1, "both_wrong": 1})
    rows = [row("correct"), row("wrong", f=0, h=0), row("loss", f=1, h=0, executed=True)]
    feedback = tmp_path / diagnostic.FEEDBACK
    feedback.mkdir(parents=True)
    audited = {}

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return diagnostic._sha(path)

    s = summary(rows)
    s.update(
        prediction_modified=False,
        memory_updated=False,
        primary_metric_policy="synthetic_controlled",
        split="synthetic_train_development",
    )
    for name, key in (
        ("runs/a0_wire_v2_freeze_v1.json", "external_freeze_sha256"),
        ("runs/a0_wire_v2_v1/TERMINAL.json", "external_terminal_sha256"),
    ):
        p = tmp_path / name
        s[key] = write(p, {"synthetic": True})
        audited[str(p)] = s[key]
    for r in rows:
        for m in (diagnostic.FRESH, diagnostic.HISTORY):
            p = tmp_path / "runs/synthetic" / f"{r['question_id']}_{m}.json"
            report = {
                "question_id": r["question_id"],
                "method": m,
                "status": "completed",
                "gold_loaded": False,
                "memory_updated": False,
                "reader": {"answer": r["answers"][m]},
                "episode": {
                    "searches": [{"query": "synthetic"}],
                    "proposals": [],
                    "stop_reason": "synthetic_stop",
                },
            }
            audited[str(p)] = write(p, report)
    sealed = {}
    for name, value in (
        ("SUMMARY.json", s),
        ("per_question.json", rows),
        ("missing_questions.json", [{"question_id": "unopened"}]),
        ("audit.json", {"inputs": audited}),
    ):
        sealed[name] = write(feedback / name, value)
    write(feedback / "feedback_frozen.json", {"files": sealed})
    # There is no source gold file at all, including the unpaired ID.
    findings = diagnostic.run(tmp_path)
    assert findings["api_calls"] == 0 and findings["new_gold_opened"] is False
    assert findings["diagnostic_n"] == 3
    assert "unopened" not in findings["diagnostic_ids"]
    target = tmp_path / diagnostic.OUTPUT
    seal = diagnostic._read(target / "frozen.json")
    assert all(diagnostic._sha(target / name) == sha for name, sha in seal["files"].items())
    assert "# 3题" in (target / "CASEBOOK.md").read_text(encoding="utf-8")
    assert diagnostic.verify_export(tmp_path)["status"] == "verified"
    (target / "paired_99.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact drift"):
        diagnostic.verify_export(tmp_path)
