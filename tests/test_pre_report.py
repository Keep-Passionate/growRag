"""Honest denominators and route accounting in the development report."""

from dataclasses import replace

import pytest
from test_query_comparison import GOLD, QUESTION, Factory, make_spec, run

from growrag.experiments.comparison_report import comparison_report
from growrag.experiments.pre_report import render_batch, summarize_batch
from growrag.experiments.protocol import Action


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
    assert report["arms"]["REUSE"]["executed"] == 0
    assert report["arms"]["REUSE"]["no_eligible_memory"] == 1
    assert report["arms"]["REUSE"]["failed_count"] == 0
    assert report["arms"]["REUSE"]["other_uncompleted_count"] == 0
    assert report["arms"]["REUSE"]["correct"] == 0
    assert report["arms"]["REUSE"]["total_targets"] == 1
    assert report["arms"]["REUSE"]["mean_answer_f1_completed"] is None
    assert report["arms"]["LEXICAL_POLICY"]["correct"] == 1
    assert report["reuse_vs_fresh"]["available_pairs"] == 0
    assert "未执行（无合适经验）" in render_batch(result)
    assert "未完成" not in render_batch(result).split("## 逐题结果")[1].split("LEXICAL_POLICY")[0]


def test_available_pairs_and_identical_queries_are_both_reported():
    report = summarize_batch(batch(no_card=False))
    assert report["arms"]["REUSE"]["executed"] == 1
    assert report["arms"]["REUSE"]["completed"] == 1
    assert report["arms"]["REUSE"]["failed_count"] == 0
    assert report["arms"]["REUSE"]["no_eligible_memory"] == 0
    assert report["reuse_vs_fresh"]["available_pairs"] == 1
    assert report["reuse_vs_fresh"]["answer_f1_ties"] == 1
    assert report["reuse_vs_fresh"]["identical_executed_queries"] == 1


def test_report_escapes_model_text_and_marks_scripted_execution():
    markdown = render_batch(batch(no_card=False))
    assert "手写替身" in markdown
    assert "<script>" not in markdown
    assert "![image](https://x)" not in markdown
    assert "&lt;script&gt;" in markdown


@pytest.mark.parametrize("component", ["backend", "generator"])
def test_actual_failed_call_is_distinct_from_no_card_and_keeps_stage_reason(component):
    result = batch(no_card=False)
    factory = Factory(**{f"fail_{component}": Action.REUSE})
    result["targets"][0]["report"] = comparison_report(run(factory=factory), gold=GOLD)
    stats = summarize_batch(result)
    reuse = stats["arms"]["REUSE"]
    assert reuse["failed_count"] == 1
    assert reuse["executed"] == 1
    assert reuse["completed"] == reuse["no_eligible_memory"] == 0
    assert reuse["abstentions"] == 0
    assert reuse["mean_answer_f1_completed"] is None
    assert stats["arms"]["LEXICAL_POLICY"]["failed_count"] == 1
    expected = "RAG 阶段失败" if component == "backend" else "改写阶段失败"
    assert f"未完成（{expected}）" in render_batch(result)


def test_abstention_is_completed_and_scored_not_failed_or_unavailable():
    result = batch(no_card=False)
    row = result["targets"][0]["report"]["arms"]["REUSE"]
    row.update(answer="", abstained=True)
    row["feedback"].update(answer_em=0.0, answer_f1=0.0)
    reuse = summarize_batch(result)["arms"]["REUSE"]
    assert reuse["completed"] == reuse["executed"] == reuse["scored_count"] == 1
    assert reuse["abstentions"] == 1
    assert reuse["correct"] == reuse["failed_count"] == reuse["no_eligible_memory"] == 0
    assert reuse["mean_answer_f1_completed"] == 0.0
    assert "弃答" in render_batch(result)


def test_other_stopped_state_is_not_silently_counted_as_failure():
    result = batch(no_card=False)
    row = result["targets"][0]["report"]["arms"]["REUSE"]
    row.update(
        completed=False,
        feedback=None,
        abstained=None,
        stop_reason="policy_stop",
        rag_calls=0,
        rewrite_calls=0,
    )
    reuse = summarize_batch(result)["arms"]["REUSE"]
    assert reuse["other_uncompleted_count"] == 1
    assert reuse["completed"] == reuse["executed"] == reuse["failed_count"] == 0
    assert reuse["not_completed_count"] == 1
    assert "未完成（策略停止）" in render_batch(result)


def test_terminal_categories_account_for_all_rows_without_dropping_no_card_or_failure():
    result = batch(no_card=False)
    no_card = batch(no_card=True)["targets"][0]
    failed = batch(no_card=False)["targets"][0]
    failed["report"] = comparison_report(run(factory=Factory(fail_backend=Action.REUSE)), gold=GOLD)
    result["targets"].extend([no_card, failed])
    reuse = summarize_batch(result)["arms"]["REUSE"]
    assert reuse["total_targets"] == 3
    assert reuse["executed"] == 2
    assert reuse["completed"] == reuse["no_eligible_memory"] == reuse["failed_count"] == 1
    assert (
        sum(
            reuse[key]
            for key in (
                "completed",
                "no_eligible_memory",
                "failed_count",
                "other_uncompleted_count",
            )
        )
        == reuse["total_targets"]
    )
    assert reuse["mean_answer_f1_completed"] == 1.0


def test_legacy_minimal_row_without_call_counters_stays_unknown_not_invented():
    result = batch(no_card=False)
    result["targets"][0]["report"]["arms"]["REUSE"] = {
        "completed": False,
        "feedback": None,
        "abstained": None,
    }
    reuse = summarize_batch(result)["arms"]["REUSE"]
    assert reuse["execution_unknown_count"] == 1
    assert reuse["executed"] == reuse["failed_count"] == reuse["no_eligible_memory"] == 0
    assert reuse["other_uncompleted_count"] == 1
    assert "未完成（原因未记录）" in render_batch(result)


def test_completed_unscored_row_does_not_disappear_from_completion_count():
    result = batch(no_card=False)
    result["targets"][0]["report"]["arms"]["REUSE"]["feedback"] = None
    reuse = summarize_batch(result)["arms"]["REUSE"]
    assert reuse["completed"] == reuse["unscored_completed_count"] == 1
    assert reuse["scored_count"] == 0
    assert reuse["mean_answer_f1_completed"] is None
    assert "已完成（未评分）" in render_batch(result)


@pytest.mark.parametrize("reason", [None, "blocked-by-test"])
def test_budget_null_means_no_block_while_actual_block_reason_is_preserved(reason):
    result = batch(no_card=True)
    result["budget"] = {
        "api_requests": 3,
        "input_tokens": 10,
        "output_tokens": 5,
        "estimated_actual_cny": 0.001,
        "block_reason": reason,
    }
    markdown = render_batch(result)
    assert f"预算阻断原因：{'无阻断' if reason is None else reason}。" in markdown
    assert "预算阻断原因：未知" not in markdown


def test_missing_budget_block_field_does_not_claim_no_block():
    result = batch(no_card=True)
    result["budget"] = {
        "api_requests": 3,
        "input_tokens": 10,
        "output_tokens": 5,
        "estimated_actual_cny": 0.001,
    }
    assert "预算阻断原因：未知／未执行。" in render_batch(result)
