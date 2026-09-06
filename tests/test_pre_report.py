"""Honest denominators and route accounting in the development report."""

from dataclasses import replace

from test_query_comparison import GOLD, QUESTION, make_spec, run

from growrag.experiments.comparison_report import comparison_report
from growrag.experiments.pre_report import render_batch, summarize_batch


def batch(*, no_card):
    spec = replace(make_spec(), memory=None) if no_card else make_spec()
    return {
        "execution_kind": "mock",
        "source_count": 1,
        "candidate_count": int(not no_card),
        "budget": None,
        "targets": [
            {
                "question_id": QUESTION.question_id,
                "question": "<script>![image](https://x)",
                "selection": {
                    "proposed_route": "BASE" if no_card else "REUSE",
                    "selected_memory_id": None if no_card else "fixture-card",
                },
                "report": comparison_report(run(spec=spec), gold=GOLD),
            }
        ],
    }


def test_unavailable_reuse_is_not_scored_as_its_base_fallback():
    result = batch(no_card=True)
    report = summarize_batch(result)
    assert report["no_card_targets"] == 1
    assert report["arms"]["REUSE"]["completed"] == 0
    assert report["arms"]["REUSE"]["correct"] == 0
    assert report["arms"]["REUSE"]["total_targets"] == 1
    assert report["arms"]["REUSE"]["mean_answer_f1_completed"] is None
    assert report["arms"]["LEXICAL_POLICY"]["correct"] == 1
    assert report["reuse_vs_fresh"]["available_pairs"] == 0


def test_available_pairs_and_identical_queries_are_both_reported():
    report = summarize_batch(batch(no_card=False))
    assert report["reuse_vs_fresh"]["available_pairs"] == 1
    assert report["reuse_vs_fresh"]["answer_f1_ties"] == 1
    assert report["reuse_vs_fresh"]["identical_executed_queries"] == 1


def test_report_escapes_model_text_and_marks_scripted_execution():
    markdown = render_batch(batch(no_card=False))
    assert "手写替身" in markdown
    assert "<script>" not in markdown
    assert "![image](https://x)" not in markdown
    assert "&lt;script&gt;" in markdown
