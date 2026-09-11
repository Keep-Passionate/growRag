"""Post-hoc reporting checks over synthetic immutable runner fixtures only."""

import json
from dataclasses import replace

import pytest
from test_representation_runner import _Harness, _spec

from growrag.experiments.protocol import Answer, CallResult, GoldRecord, Usage
from growrag.experiments.representation_report import representation_report, save_representation_run


def _run(answers=None, *, choices=("c1", "c2", "FRESH", "c1")):
    """Construct labelled synthetic outcomes, not replacement real-model results."""
    run = _Harness(_spec(), choices=choices).run()
    if answers is None:
        return run
    outcomes = []
    for outcome in run.outcomes:
        result = outcome.result
        row = result.state.rounds[0]
        row = replace(row, reply=replace(row.reply, answer=Answer(answers[outcome.route_id])))
        outcomes.append(
            replace(outcome, result=replace(result, state=replace(result.state, rounds=(row,))))
        )
    return replace(run, outcomes=tuple(outcomes))


def _gold(run, answer="correct"):
    return GoldRecord(run.spec.candidates.question.question_id, (answer,), (("target doc", 0),))


def _known_usage(run):
    selections = tuple(
        replace(row, event=replace(row.event, usage=Usage(3, 1, 0))) for row in run.selections
    )
    outcomes = []
    for row in run.outcomes:
        result = row.result
        events = tuple(replace(event, usage=Usage(10, 2, 0)) for event in result.events)
        # Component traces are already included in the outer calls, not extra work.
        components = tuple(replace(event, usage=Usage(9999, 9999, 0)) for event in result.events)
        outcomes.append(
            replace(row, result=replace(result, events=events, component_events=components))
        )
    return replace(run, selections=selections, outcomes=tuple(outcomes))


def test_increment_is_against_fresh_with_rescue_rejection_and_oracle_regret():
    run = _run({"BASE": "correct", "FRESH": "wrong", "c1": "correct", "c2": "wrong"})
    report = representation_report(run, gold=_gold(run))
    m1, m2, m3 = (report["policies"][kind] for kind in ("M1", "M2", "M3"))
    assert m1["answer_f1_delta_vs_fresh"] == 1
    assert m1["rescued_vs_fresh"] is True and m1["harmed_vs_fresh"] is False
    assert m1["oracle_regret_f1"] == 0
    assert m2["answer_f1_delta_vs_fresh"] == 0  # Not -1 versus BASE.
    assert m2["oracle_regret_f1"] == 1
    assert m3["selected_id"] == "FRESH" and m3["answer_f1_delta_vs_fresh"] == 0
    assert m3["missed_beneficial_candidate"] is True
    assert m3["oracle_regret_f1"] == 1
    assert report["candidate_pool_has_benefit_vs_fresh"] is True
    assert report["offline_candidate_oracle_f1"] == 1
    assert report["synthetic_demo"] is True


def test_harm_is_fresh_correct_to_selected_wrong_not_base_background():
    run = _run({"BASE": "wrong", "FRESH": "correct", "c1": "wrong", "c2": "correct"})
    report = representation_report(run, gold=_gold(run))
    m1, m2, m3 = (report["policies"][kind] for kind in ("M1", "M2", "M3"))
    assert m1["answer_f1_delta_vs_fresh"] == -1
    assert m1["harmed_vs_fresh"] is True and m1["rescued_vs_fresh"] is False
    assert m2["answer_f1_delta_vs_fresh"] == 0 and m2["oracle_regret_f1"] == 0
    assert m3["missed_beneficial_candidate"] is False
    assert report["candidate_pool_has_benefit_vs_fresh"] is False


def test_partial_f1_does_not_become_an_exact_match_rescue():
    run = _run({"BASE": "wrong", "FRESH": "wrong", "c1": "red", "c2": "red blue"})
    report = representation_report(run, gold=_gold(run, "red blue"))
    m1 = report["policies"]["M1"]
    assert m1["answer_f1_delta_vs_fresh"] == pytest.approx(2 / 3)
    assert m1["rescued_vs_fresh"] is False
    assert m1["oracle_regret_f1"] == pytest.approx(1 / 3)


def test_actual_audit_cost_counts_each_execution_once_not_each_policy_or_component():
    run = _known_usage(_run())
    report = representation_report(run, gold=_gold(run, "c1"))
    # 4 RAG + 3 rewrites, plus 4 selector calls. Component events are not doubled.
    assert report["actual_audit_usage"] == {
        "input_tokens": 82,
        "output_tokens": 18,
        "api_requests": 0,
    }
    for policy in report["policies"].values():
        # One selector + the chosen rewrite and RAG, never all unselected routes.
        assert policy["path_usage_from_audit"] == {
            "input_tokens": 23,
            "output_tokens": 5,
            "api_requests": 0,
        }
        assert "latency" in policy["cost_note"] and "Reconstructed" in policy["cost_note"]
    assert (
        sum(p["path_usage_from_audit"]["input_tokens"] for p in report["policies"].values()) != 82
    )
    assert report["outcomes"]["BASE"]["rewrite_calls"] == 0
    assert report["outcomes"]["c1"]["rewrite_calls"] == 1
    assert report["selector_call_count"] == 4


def test_unknown_usage_propagates_per_metric_not_as_zero():
    run = _known_usage(_run())
    outcomes = []
    for row in run.outcomes:
        if row.route_id == "c1":
            events = list(row.result.events)
            events[0] = replace(events[0], usage=Usage(None, 2, 0))
            row = replace(row, result=replace(row.result, events=tuple(events)))
        outcomes.append(row)
    report = representation_report(replace(run, outcomes=tuple(outcomes)), gold=_gold(run))
    assert report["actual_audit_usage"]["input_tokens"] is None
    assert report["actual_audit_usage"]["output_tokens"] == 18
    assert report["policies"]["M1"]["path_usage_from_audit"]["input_tokens"] is None
    assert report["policies"]["M2"]["path_usage_from_audit"]["input_tokens"] == 23


def test_missing_evidence_is_unknown_but_answer_can_still_be_evaluated():
    run = _run({"BASE": "wrong", "FRESH": "wrong", "c1": "correct", "c2": "wrong"})
    outcomes = []
    for outcome in run.outcomes:
        if outcome.route_id == "c1":
            result = outcome.result
            row = replace(
                result.state.rounds[0], reply=replace(result.state.rounds[0].reply, evidence=None)
            )
            state = replace(result.state, rounds=(row,), observed_evidence=())
            outcome = replace(outcome, result=replace(result, state=state))
        outcomes.append(outcome)
    report = representation_report(replace(run, outcomes=tuple(outcomes)), gold=_gold(run))
    feedback = report["outcomes"]["c1"]["feedback"]
    assert feedback["answer_em"] == 1
    for key in (
        "retrieved_gold_support_recall",
        "new_gold_support",
        "cited_gold_support_precision",
        "cited_gold_support_recall",
        "citation_ids_resolve",
    ):
        assert feedback[key] is None
    assert report["policies"]["M1"]["support_recall_delta_vs_fresh"] is None
    assert report["policies"]["M1"]["answer_f1_delta_vs_fresh"] == 1


def test_empty_gold_support_does_not_turn_unavailable_support_score_into_zero():
    run = _run()
    report = representation_report(run, gold=replace(_gold(run), supporting_facts=()))
    assert all(
        row["feedback"]["retrieved_gold_support_recall"] is None
        for row in report["outcomes"].values()
    )
    assert all(row["support_recall_delta_vs_fresh"] is None for row in report["policies"].values())


@pytest.mark.parametrize("missing_route", ["c1", "c2", "FRESH"])
def test_unexecuted_outcomes_are_retained_without_inventing_scores(missing_route):
    run = _run()
    run = replace(
        run,
        outcomes=tuple(
            replace(row, result=None, status="stopped") if row.route_id == missing_route else row
            for row in run.outcomes
        ),
    )
    report = representation_report(run, gold=_gold(run, "c1"))
    row = report["outcomes"][missing_route]
    assert row["feedback"] is None and row["answer"] is None
    assert row["status"] == "stopped" and row["stop_reason"] == "not_executed"
    assert report["candidate_pool_complete"] is False
    assert report["offline_candidate_oracle_f1"] is None
    assert report["candidate_pool_has_benefit_vs_fresh"] is None
    for policy in report["policies"].values():
        if policy["selected_id"] == missing_route or missing_route == "FRESH":
            assert policy["paired_available"] is False
            assert policy["answer_f1_delta_vs_fresh"] is None
            assert policy["rescued_vs_fresh"] is None
            assert policy["harmed_vs_fresh"] is None


def test_failed_execution_retains_failed_call_cost_and_no_answer_score():
    run = _known_usage(_run())
    outcomes = []
    for row in run.outcomes:
        if row.route_id == "c1":
            result = row.result
            event = replace(
                result.events[0],
                status="error",
                error_type="SyntheticFailure",
                usage=Usage(7, 1, 0),
            )
            result = replace(
                result,
                state=replace(result.state, rounds=(), observed_evidence=()),
                events=(event,),
                component_events=(),
                stop_reason="error",
            )
            row = replace(row, result=result, status="execution_failed")
        outcomes.append(row)
    report = representation_report(replace(run, outcomes=tuple(outcomes)), gold=_gold(run))
    assert report["outcomes"]["c1"]["feedback"] is None
    assert report["outcomes"]["c1"]["usage"]["input_tokens"] == 7
    assert report["actual_audit_usage"]["input_tokens"] == 69  # 82 - 20 + 7.
    assert report["policies"]["M1"]["path_usage_from_audit"]["input_tokens"] == 10
    assert report["policies"]["M1"]["answer_em"] is None
    assert report["policies"]["M1"]["paired_available"] is False
    assert report["policies"]["M1"]["reuse_attempted"] is True
    assert report["policies"]["M1"]["reuse_executed"] is False
    assert report["outcomes"]["c1"]["usage"]["api_requests"] == 0


def test_selector_error_is_distinct_from_deliberate_refusal_even_with_fresh_fallback():
    run = _Harness(_spec(), choices=("invalid", "FRESH", "FRESH", "FRESH")).run()
    report = representation_report(run, gold=_gold(run, "FRESH"))
    m1, m2 = report["policies"]["M1"], report["policies"]["M2"]
    assert m1["selector_error"] is True and m2["selector_error"] is False
    assert m1["selection_status"] == "selector_error"
    assert m2["selection_status"] == "refused"
    assert m1["selected_id"] == "FRESH"
    assert m1["answer_f1_delta_vs_fresh"] == 0


def test_no_gold_produces_no_oracle_or_correctness_labels_and_does_not_change_run():
    run = _run()
    original = run.to_dict()
    report = representation_report(run)
    assert report["gold_available"] is False
    assert report["offline_candidate_oracle_f1"] is None
    assert all(row["feedback"] is None for row in report["outcomes"].values())
    assert all(row["answer_f1_delta_vs_fresh"] is None for row in report["policies"].values())
    assert run.to_dict() == original
    representation_report(run, gold=_gold(run))
    assert run.to_dict() == original


def test_wrong_question_gold_and_nonrun_input_are_rejected():
    run = _run()
    with pytest.raises(ValueError, match="this target"):
        representation_report(run, gold=replace(_gold(run), question_id="other"))
    with pytest.raises(ValueError, match="this target"):
        representation_report(run, gold={"answer": "correct"})
    with pytest.raises(TypeError):
        representation_report({"outcomes": []})


def test_no_candidates_still_reports_baseline_with_zero_selection_cost():
    run = _Harness(_spec(0), choices=()).run()
    report = representation_report(run, gold=_gold(run, "FRESH"))
    assert report["candidate_count"] == 0
    assert report["candidate_pool_complete"] is True
    assert report["candidate_pool_has_benefit_vs_fresh"] is False
    assert report["offline_candidate_oracle_f1"] == 1
    assert set(report["outcomes"]) == {"BASE", "FRESH"}
    assert all(
        policy["selection_status"] == "no_candidates" for policy in report["policies"].values()
    )
    assert all(policy["answer_f1_delta_vs_fresh"] == 0 for policy in report["policies"].values())
    assert report["selector_call_count"] == 0
    assert "All four selector calls" not in report["cost_scope"]
    assert all(
        not any(
            policy[key] for key in ("reused", "reuse_selected", "reuse_attempted", "reuse_executed")
        )
        for policy in report["policies"].values()
    )


def test_character_lengths_are_not_reported_as_tokens():
    run = _run()
    report = representation_report(run)
    for candidate, entry in zip(
        run.spec.candidates.candidates, report["representation_lengths"], strict=True
    ):
        assert entry["candidate_id"] == candidate.candidate_id
        for view in candidate.bundle.views:
            assert entry["views"][view.kind.value] == {"characters": len(view.text), "tokens": None}


def test_exclusive_save_never_overwrites_existing_artifacts(tmp_path):
    run = _run()
    directory = tmp_path / "new-run"
    report = save_representation_run(run, directory, gold=_gold(run))
    before = {name: (directory / name).read_bytes() for name in ("run.json", "report.json")}
    assert json.loads(before["report.json"])["question_id"] == report["question_id"]
    assert json.loads(before["run.json"])["synthetic_demo"] is True
    with pytest.raises(FileExistsError):
        save_representation_run(run, directory, gold=_gold(run, "different"))
    assert before == {name: (directory / name).read_bytes() for name in before}


def test_wrong_gold_is_rejected_before_creating_output_directory(tmp_path):
    run = _run()
    directory = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        save_representation_run(run, directory, gold=replace(_gold(run), question_id="other"))
    assert not directory.exists()


def test_same_query_answer_difference_is_marked_not_hidden_as_a_query_change():
    run = _run({"BASE": "wrong", "FRESH": "wrong", "c1": "correct", "c2": "wrong"})
    fresh = next(row for row in run.outcomes if row.route_id == "FRESH")
    outcomes = []
    for outcome in run.outcomes:
        if outcome.route_id == "c1":
            result = outcome.result
            row = replace(
                result.state.rounds[0], search_query=fresh.result.state.rounds[0].search_query
            )
            outcome = replace(
                outcome, result=replace(result, state=replace(result.state, rounds=(row,)))
            )
        outcomes.append(outcome)
    report = representation_report(replace(run, outcomes=tuple(outcomes)), gold=_gold(run))
    assert report["policies"]["M1"]["different_executed_query"] is False
    assert "Same-query answer variation is not rewrite credit" in report["notice"]


def test_lost_support_and_better_answer_are_separate_feedback_axes():
    run = _run({"BASE": "wrong", "FRESH": "wrong", "c1": "correct", "c2": "wrong"})
    outcomes = []
    for outcome in run.outcomes:
        if outcome.route_id == "c1":
            result = outcome.result
            row = replace(
                result.state.rounds[0], reply=replace(result.state.rounds[0].reply, evidence=())
            )
            state = replace(result.state, rounds=(row,), observed_evidence=())
            outcome = replace(outcome, result=replace(result, state=state))
        outcomes.append(outcome)
    report = representation_report(replace(run, outcomes=tuple(outcomes)), gold=_gold(run))
    assert report["policies"]["M1"]["answer_f1_delta_vs_fresh"] == 1
    assert report["policies"]["M1"]["support_recall_delta_vs_fresh"] == -1


def test_selected_but_stopped_reuse_is_not_counted_as_attempted_or_executed():
    run = _run()
    run = replace(
        run,
        outcomes=tuple(
            replace(row, result=None, status="stopped") if row.route_id == "c1" else row
            for row in run.outcomes
        ),
    )
    report = representation_report(run, gold=_gold(run))
    policy = report["policies"]["M1"]
    assert policy["reused"] is policy["reuse_selected"] is True
    assert policy["reuse_attempted"] is False
    assert policy["reuse_executed"] is False
    assert policy["paired_available"] is False
    assert "deprecated alias of reuse_selected" in report["reuse_metrics_note"]
    assert "not necessarily a network request" in report["reuse_metrics_note"]


def test_stop_before_selecting_reports_zero_calls_and_zero_reuse_activity():
    run = _Harness(_spec()).run(stop_requested=lambda: True)
    report = representation_report(run, gold=_gold(run))
    assert report["selector_call_count"] == 0
    assert report["actual_audit_usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "api_requests": 0,
    }
    for policy in report["policies"].values():
        assert policy["selection_status"] == "stopped"
        assert policy["reuse_selected"] is False
        assert policy["reuse_attempted"] is False
        assert policy["reuse_executed"] is False
        assert policy["answer_em"] is None


def test_unchanged_reuse_query_attempt_returns_base_without_claiming_reuse_execution():
    harness = _Harness(_spec())
    original = harness.factory

    def factory(route):
        components = original(route)
        if route == "c1":

            def unchanged(question, decision, *, evidence=(), previous_queries=()):
                assert evidence == previous_queries == ()
                return CallResult(question.text, transport_source="mock")

            components.generator.generate = unchanged
        return components

    harness.factory = factory
    run = harness.run()
    report = representation_report(run, gold=_gold(run))
    policy = report["policies"]["M1"]
    assert report["outcomes"]["c1"]["executed_action"] == "BASE"
    assert policy["reuse_selected"] is True
    assert policy["reuse_attempted"] is True
    assert policy["reuse_executed"] is False


def test_completed_changed_reuse_and_plain_fresh_have_distinct_activity_flags():
    run = _run()
    report = representation_report(run, gold=_gold(run))
    selected, fresh = report["policies"]["M1"], report["policies"]["M3"]
    assert all(selected[key] for key in ("reuse_selected", "reuse_attempted", "reuse_executed"))
    assert not any(fresh[key] for key in ("reuse_selected", "reuse_attempted", "reuse_executed"))
    assert report["outcomes"]["FRESH"]["rewrite_calls"] == 1
