"""Synthetic report analysis only: no model client and no live API invocation."""

import json

import pytest

from growrag.experiments.bounded_analysis import analyze, load_inputs, main


def score(em, f1=None, support=0.5, added=()):
    return {
        "answer_em": em,
        "answer_f1": em if f1 is None else f1,
        "retrieved_gold_support_recall": support,
        "new_gold_support": list(added),
    }


def arm(
    *,
    scores=None,
    answers=None,
    judges=None,
    action="BASE",
    call="a",
    cost=0.1,
    status="completed",
    audit=None,
    prefix=None,
):
    scores = scores or [score(0)]
    answers = answers or ["raw"] * len(scores)
    judges = judges or [False] * len(scores)
    rounds = [
        {
            "decision": {
                "action": action,
                "memory": {"memory_id": "m"} if action == "REUSE" else None,
            },
            "search_query": f"query {index}",
            "reply": {
                "answer": {"text": answer},
                "component_events": (
                    [{"operation": "rag.answer", "audit_path": str(audit), "model": "fixed-model"}]
                    if audit
                    else []
                ),
            },
            "feedback": {"sufficient": judges[index]},
        }
        for index, answer in enumerate(answers)
    ]
    return {
        "status": status,
        "result": {"state": {"rounds": rounds}, "stop_reason": "budget"},
        "feedback": scores[-1] if status == "completed" else None,
        "round_feedback": scores if status == "completed" else [],
        "released_feedback": scores[-1],
        "answer_release": {"allowed_by_judge": judges[-1]},
        "route_records": [{"action": action, "selected_memory_id": "m"}],
        "calls": [
            {
                "trace_id": call,
                "api_requests": 1,
                "input_tokens": 10,
                "output_tokens": 5,
                "estimated_actual_cny": cost,
            }
        ],
        "shared_prefix": prefix,
    }


def report(qid, **arms):
    return {"question": {"question_id": qid, "text": "A synthetic question?"}, "arms": arms}


def test_pairwise_intersection_does_not_turn_failed_missing_scores_into_zero(tmp_path):
    rows = [
        report(
            "q1",
            BASE1=arm(scores=[score(0)], call="a"),
            ADAPTIVE_NO_MEMORY2=arm(scores=[score(0)], call="b"),
            ADAPTIVE_MEMORY2=arm(scores=[score(1)], call="c", action="REUSE"),
        ),
        report(
            "q2",
            BASE1=arm(scores=[score(1)], call="d"),
            ADAPTIVE_MEMORY2=arm(status="failed", call="e"),
        ),
    ]
    data = analyze([(r, tmp_path) for r in rows])
    memory = data["arms"]["ADAPTIVE_MEMORY2"]
    assert memory["scores"]["answer_em"] == {"n": 1, "mean": 1}
    assert memory["comparisons"]["BASE1"]["scored_intersection_n"] == 1
    assert memory["comparisons"]["BASE1"]["rescues_em_0_to_1"] == ["q1"]
    assert memory["comparisons"]["BASE1"]["harms_em_1_to_0"] == []
    assert memory["issued_reuse_question_rate_among_attempted"] == 0.5
    assert data["questions"][1]["arms"]["ADAPTIVE_MEMORY2"]["final_score"]["answer_em"] is None


def test_shared_prefix_cost_is_only_shadow_and_duplicate_call_counted_once(tmp_path):
    base = arm(call="shared", cost=0.1)
    branch = arm(
        call="second",
        cost=0.2,
        prefix={
            "source_arm": "BASE1",
            "cost": {"estimated_actual_cny": 0.1},
            "actually_collected_once": True,
        },
    )
    row = report("q", BASE1=base, REFLECTIVE2=branch, SHARED_COPY=base)
    data = analyze([(row, tmp_path)])
    assert data["observed_unique_reported_call_cost"]["recorded_attempts"] == 2
    assert data["observed_unique_reported_call_cost"]["estimated_actual_cny"] == pytest.approx(0.3)
    assert data["questions"][0]["arms"]["REFLECTIVE2"][
        "shadow_standalone_estimated_cny"
    ] == pytest.approx(0.3)


def test_partial_unknown_cost_is_not_zero_and_no_budget_is_not_known(tmp_path):
    data = analyze([(report("q", BASE1=arm(cost=None, status="failed")), tmp_path)])
    assert data["observed_unique_reported_call_cost"]["estimated_actual_cny"] is None
    assert data["authoritative_final_budget_totals"]["api_requests"] is None
    assert data["arms"]["BASE1"]["scores"]["answer_f1"] == {"n": 0, "mean": None}


def test_module_transitions_and_judge_gold_mismatch_are_descriptive(tmp_path):
    row = report(
        "q",
        BASE1=arm(call="a", scores=[score(0)], judges=[True]),
        REFLECTIVE2=arm(
            call="b",
            scores=[score(0), score(1, support=1, added=["fact"])],
            judges=[False, False],
            answers=["wrong", "gold-match"],
        ),
    )
    data = analyze([(row, tmp_path)])
    module = data["arms"]["REFLECTIVE2"]["modules"]
    assert module["initial_insufficient_with_second_round_n"] == 1
    assert module["answer_f1_improved_ids"] == ["q"]
    assert module["second_round_added_gold_support_ids"] == ["q"]
    assert module["gold_matching_raw_answer_withheld_ids"] == ["q"]
    assert data["arms"]["BASE1"]["modules"][
        "sufficient_signal_with_gold_nonmatching_raw_answer_ids"
    ] == ["q"]


def test_same_reader_input_different_answers_whitelists_transport_metadata(tmp_path):
    audit = tmp_path / "api_audit"
    audit.mkdir()
    for name in ("call-a", "call-b"):
        (audit / f"{name}.json").write_text(
            json.dumps(
                {
                    "request_sha256": "a" * 64,
                    "prompt_version": "reader-v2",
                    "endpoint": "SECRET_ENDPOINT",
                    "headers": {"Authorization": "SECRET_TOKEN"},
                    "request": {"private": "SECRET_BODY"},
                }
            ),
            encoding="utf-8",
        )
    row = report(
        "q",
        BASE1=arm(call="a", answers=["one"], audit=audit / "call-a.json"),
        FRESH1=arm(call="b", answers=["two"], audit=audit / "call-b.json"),
    )
    data = analyze([(row, tmp_path)])
    divergent = data["same_reader_request_different_answers"]
    assert len(divergent) == 1 and divergent[0]["request_sha256"] == "a" * 64
    assert {r["raw_answer"] for r in divergent[0]["occurrences"]} == {"one", "two"}
    assert "SECRET_" not in json.dumps(data)
    assert data["observed_unique_reported_call_cost"]["recorded_attempts"] == 2


def test_run_directory_needs_finalization_but_stopped_run_remains_explicit(tmp_path):
    (tmp_path / "reports.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_inputs([tmp_path])
    (tmp_path / "summary.json").write_text(
        json.dumps({"planned": 7, "started": 3, "reported": 3, "complete": 2, "not_started": 4}),
        encoding="utf-8",
    )
    (tmp_path / "final_budget.json").write_text(
        json.dumps({"api_requests": 35, "block_reason": "bounded_component_failure"}),
        encoding="utf-8",
    )
    rows, sources, budgets = load_inputs([tmp_path])
    data = analyze(rows, sources=sources, budgets=budgets)
    assert data["sources"][0]["not_started"] == 4
    assert data["sources"][0]["block_reason"] == "bounded_component_failure"
    assert data["authoritative_final_budget_totals"]["api_requests"] == 35


def test_cli_writes_exclusive_outputs_without_altering_input(tmp_path):
    source = tmp_path / "report.json"
    original = json.dumps(report("q", BASE1=arm()))
    source.write_text(original, encoding="utf-8")
    output = tmp_path / "analysis"
    main([str(source), "--output", str(output)])
    assert source.read_text(encoding="utf-8") == original
    assert (output / "analysis.md").is_file() and (output / "analysis.json").is_file()
    with pytest.raises(FileExistsError):
        main([str(source), "--output", str(output)])


def test_duplicate_question_attempts_cannot_silently_share_denominators(tmp_path):
    row = report("q", BASE1=arm())
    with pytest.raises(ValueError, match="duplicate question"):
        analyze([(row, tmp_path), (row, tmp_path)])
