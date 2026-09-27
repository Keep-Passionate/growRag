"""Synthetic tests only; no real source labels or model calls."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "source_summary",
    Path(__file__).resolve().parents[1] / "scripts/audits/analyze_memory_sources.py",
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def fixture():
    rows, calls = [], []
    for offset in range(2):
        arms = {}
        for arm in M.ARMS:
            em = offset if arm == M.BASE else 1 - offset
            call = {
                "trace_id": f"q{offset}/{arm}",
                "api_requests": 1,
                "reserved_cny": 0.1,
                "estimated_actual_cny": 0.01,
                "input_tokens": 10,
                "output_tokens": 2,
            }
            calls.append(call)
            arms[arm] = {
                "status": "completed",
                "calls": [call],
                "elapsed_seconds": 1,
                "feedback": {
                    "answer_em": em,
                    "answer_f1": em,
                    "raw_support_recall": em,
                    "retained_support_recall": em,
                    "raw_support_hits": em,
                    "retained_support_hits": em,
                },
                "result": {
                    "answer": str(em),
                    "retrieval_rounds": 1,
                    "events": [{"kind": "retrieval", "queries": ["rewrite"]}]
                    if arm == M.S2G
                    else [],
                },
            }
        rows.append(
            {
                "offset": offset,
                "question_id": f"q{offset}",
                "question": f"question{offset}",
                "arms": arms,
                "scoring_status": "completed",
                "offline_gold_answers": ["synthetic"],
            }
        )
    before = copy.deepcopy(rows)
    for row in before:
        row.pop("offline_gold_answers")
        row["scoring_status"] = "deferred"
        for arm in M.ARMS:
            row["arms"][arm]["feedback"] = None
    return rows, before, calls, {"q0": "synthetic0.json", "q1": "synthetic1.json"}


def test_source_metrics_costs_queries():
    rows, summary = M.assemble(*fixture())
    assert summary["same_complete_case_n"] == 2
    assert summary["repairs"] == summary["harms"] == 1
    assert summary["actual_usage_all_source_attempts"]["api_requests"] == 4
    assert summary["support_deltas"]["raw_support_recall"]["mean"] == 0
    assert summary["memory_cards_created"] == 0
    assert rows[0]["arms"][M.BASE]["queries"] == ["question0"]
    assert rows[0]["arms"][M.S2G]["queries"] == ["rewrite"]


@pytest.mark.parametrize("mutation", ["answer", "duplicate", "orphan", "drop", "id"])
def test_prediction_or_ownership_changes_rejected(mutation):
    args = fixture()
    if mutation == "answer":
        args[0][0]["arms"][M.BASE]["result"]["answer"] = "posthoc"
    elif mutation == "duplicate":
        args[2].append(copy.deepcopy(args[2][0]))
    elif mutation == "orphan":
        args[2].append(dict(args[2][0], trace_id="orphan"))
    elif mutation == "drop":
        args[2].pop()
    else:
        args[0][0]["question_id"] = "heldout"
    with pytest.raises(ValueError):
        M.assemble(*args)


def test_unscorable_excluded_from_every_score_but_not_actual_cost():
    args = fixture()
    args[0][1]["arms"][M.BASE]["feedback"] = {"unscorable_annotation": True}
    _, summary = M.assemble(*args)
    assert summary["same_complete_case_n"] == 1
    assert summary["excluded_unscorable_n"] == 1
    assert summary["actual_usage_all_source_attempts"]["api_requests"] == 4


def test_unknown_usage_not_zero():
    args = fixture()
    args[2][0]["estimated_actual_cny"] = None
    args[1][0]["arms"][M.BASE]["calls"][0]["estimated_actual_cny"] = None
    _, summary = M.assemble(*args)
    assert summary["actual_usage_all_source_attempts"]["total_estimated_cost_cny"] is None


def test_technical_failure_is_not_zero_answer_or_bad_annotation(tmp_path):
    args = fixture()
    for cohort in (args[0], args[1]):
        outcome = cohort[1]["arms"][M.S2G]
        outcome["status"] = "failed"
        outcome["feedback"] = None
        outcome.pop("result")
    rows, summary = M.assemble(*args)
    assert summary["source_question_count"] == 2
    assert summary["same_complete_case_n"] == 1
    assert summary["technical_failure_pairs_n"] == 1
    assert summary["unscorable_annotation_pairs_n"] == 0
    assert summary["actual_usage_all_source_attempts"]["api_requests"] == 4
    assert rows[1]["deltas"] is None
    assert rows[1]["arms"][M.S2G]["feedback"] is None
    M.write_markdown(tmp_path, rows, summary)
    assert "技术失败1题" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_readable_source_entrypoints(tmp_path):
    rows, summary = M.assemble(*fixture())
    M.write_markdown(tmp_path, rows, summary)
    text = (tmp_path / "QUESTIONS.md").read_text(encoding="utf-8")
    assert "q0" in text and "q1" in text and "synthetic0.json" in text
    assert "来源筛选口径探索" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_source_policy_groups_are_nested_and_d_is_overlapping():
    rows = fixture()[0]
    s = rows[0]["arms"][M.S2G]["feedback"]
    s["raw_support_recall"], s["retained_support_recall"] = 0.5, 0.25
    result, flags = M.candidate_policy_exploration(rows)
    assert all(result["groups"][name]["question_ids"] == ["q0"] for name in ("A", "B", "C", "D"))
    assert result["details"]["D_both_incomplete"]["question_ids"] == ["q0"]
    assert flags["q0"] == {"A": True, "B": True, "C": True, "D": True}
    assert not any(flags["q1"].values())


def test_source_policy_unknown_support_is_not_incomplete_or_increment():
    rows = fixture()[0]
    s = rows[0]["arms"][M.S2G]["feedback"]
    s["raw_support_recall"], s["retained_support_recall"] = None, 1
    result, flags = M.candidate_policy_exploration(rows)
    assert flags["q0"] == {"A": True, "B": False, "C": False, "D": False}
    assert result["details"]["A_raw_delta_unknown"]["question_ids"] == ["q0"]
    assert result["details"]["A_support_field_unknown"]["question_ids"] == ["q0"]
    s["retained_support_recall"] = 0.5
    result, flags = M.candidate_policy_exploration(rows)
    assert flags["q0"]["D"]
    assert result["details"]["D_raw_incomplete"]["n"] == 0
    assert result["details"]["D_retained_incomplete"]["n"] == 1


def test_a_does_not_require_base_scoreable_and_c_does():
    rows = fixture()[0]
    rows[0]["arms"][M.BASE]["feedback"]["answer_em"] = None
    result, flags = M.candidate_policy_exploration(rows)
    assert flags["q0"] == {"A": True, "B": True, "C": False, "D": False}
    assert result["details"]["B_base_answer_unscorable"]["question_ids"] == ["q0"]


def test_source_policy_groups_report_failed_s2g_separately():
    rows = fixture()[0]
    rows[0]["arms"][M.S2G]["status"] = "failed"
    result, flags = M.candidate_policy_exploration(rows)
    assert result["groups"]["A"]["n"] == 0
    assert result["details"]["s2g_answer_unscorable"]["question_ids"] == ["q0"]
    assert not any(flags["q0"].values())


def test_actual_source64_manifest_normal_import_and_incomplete_blocks_labels(tmp_path, monkeypatch):
    """Integration: real frozen question metadata only; absent predictions forbid scoring."""
    manifest = M.ROOT / "data/hotpotqa/memory_sources_sep27_v1/manifest.json"
    if not manifest.exists():
        pytest.skip("local frozen source plan not present")
    assert M.core.__name__ == "growrag.experiments.source_collection_core"
    questions, plan = M.core.load_source_questions(manifest)
    assert len(questions) == 64
    assert [q.question_id for q in questions] == plan["roles"]["source"]
    assert all(not hasattr(q, "answer") for q in questions)
    monkeypatch.setattr(
        M.core, "project_source_gold", lambda *a: pytest.fail("summary must never read raw labels")
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="complete source64"):
        M.main(
            [
                "--runs-root",
                str(tmp_path),
                "--source-manifest",
                str(manifest),
                "--scored-dir",
                str(tmp_path / "DO_NOT_READ_LABELS"),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()
