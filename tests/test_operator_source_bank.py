"""Synthetic provenance/retention checks; these numbers are not benchmark results."""

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from growrag.experiments.operator_source_bank import SOURCE_PROTOCOL, build_source_bank
from growrag.experiments.protocol import Evidence, RuntimeQuestion
from growrag.macro_operators import GapField, GoalContract, OperatorSpec, QueryStep
from growrag.operator_loop import ActionProposal, run_operator_episode


def make_source(qid="s1", *, question="Where was Alice born?", templates=("{entity} birthplace",)):
    runtime = RuntimeQuestion(qid, question)
    base_evidence = Evidence("e0", "Biography", 0, "Alice was a researcher.")
    goal = GoalContract(question, "lookup")
    specs = tuple(
        OperatorSpec(
            f"PROPOSED_{i}", "1", ("lookup",), (GapField("entity"),), (QueryStep("search", text),)
        )
        for i, text in enumerate(templates)
    )
    count = 0

    def retrieve(query, top_k):
        nonlocal count
        if query == question:
            return (base_evidence,)
        count += 1
        return (Evidence(f"e{count}", "New document", 0, f"Alice has fact {count}."),)

    def fresh(state):
        return ActionProposal(goal, specs[state.decision_number - 1], {"entity": "Alice"})

    arms = {}
    for arm in ("base", "static", "fresh"):
        result = run_operator_episode(
            runtime,
            retrieve,
            fresh if arm == "fresh" else lambda state: ActionProposal(goal, None, origin="stop"),
            retrieval_budget=1 if arm == "base" else 3,
            max_decisions=len(specs) if arm == "fresh" else 1,
        )
        arms[arm] = {
            "status": "completed",
            "question_id": qid,
            "arm": arm,
            "episode": json.loads(json.dumps(asdict(result))),
        }
    return {"phase": "source", "protocol": SOURCE_PROTOCOL, "question_id": qid, "arms": arms}


def scored(qid="s1", *, fresh_em=1, fresh_f1=1, fresh_support=0.5):
    return {
        qid: {
            "base": {"answer_em": 0, "answer_f1": 0.1, "raw_support_recall": 0.25},
            "static": {"answer_em": 0, "answer_f1": 0.1, "raw_support_recall": 0.25},
            "fresh": {
                "answer_em": fresh_em,
                "answer_f1": fresh_f1,
                "raw_support_recall": fresh_support,
            },
        }
    }


def test_valid_source_is_exploratory_published_but_not_trusted():
    bank, audit = build_source_bank(("s1",), (make_source(),), scored())
    assert len(bank.records) == len(bank.published_specs) == 1
    record = bank.records[0]
    assert record.source_qids == ("s1",)
    entry = audit["operators"][record.source_trace_ref]
    assert entry["trusted"] is False
    assert entry["individual_operator_effect"] == "unknown"
    assert entry["attribution_scope"] == "episode_association"
    assert entry["publication"]["validation_scope"] == "source_schema_sanity_only"
    assert entry["publication"]["transfer_evidence"] == "single_source_unverified"
    assert entry["publication"]["semantic_drift_absence_proven"] is False
    event = entry["source_events"][0]
    assert event["proposal_number"] == 1
    assert event["pre_action_evidence_ids"] == ["e0"]
    assert event["executed_queries"] == ["Alice birthplace"]
    assert record.validation_ref == record.source_trace_ref + "#publication"


def test_source_inputs_are_not_mutated_and_bank_has_no_feedback_payload():
    report, feedback = make_source(), scored()
    before = deepcopy((report, feedback))
    bank, _ = build_source_bank(("s1",), (report,), feedback)
    assert (report, feedback) == before
    assert "answer_em" not in bank.to_json()
    assert "episode_delta" not in bank.to_json()


def test_future_450_sources_cannot_change_first_50_bank_or_audits():
    reports = [make_source(f"s{i}") for i in range(500)]
    feedback = {}
    for i in range(500):
        feedback.update(scored(f"s{i}"))
    prefix = tuple(f"s{i}" for i in range(50))
    small, a = build_source_bank(prefix, reports[:50], feedback)
    with_future, b = build_source_bank(prefix, reports, feedback)
    assert small == with_future
    assert small.fingerprint == with_future.fingerprint
    assert a == b
    assert small.records[0].source_qids == tuple(sorted(prefix))
    # A malformed future metric is deliberately never inspected by the small bank.
    feedback["s499"]["fresh"]["answer_em"] = "future-invalid"
    assert build_source_bank(prefix, reports, feedback)[0].fingerprint == small.fingerprint


def test_aggregation_occurs_only_after_source_prefix_filter():
    reports = (make_source("s1"), make_source("s2"))
    feedback = {**scored("s1", fresh_em=0, fresh_f1=0.5), **scored("s2")}
    small, _ = build_source_bank(("s1",), reports, feedback)
    larger, _ = build_source_bank(("s1", "s2"), reports, feedback)
    assert small.records[0].status == "candidate"
    assert small.published_specs == ()
    assert larger.records[0].status == "validated"
    assert larger.records[0].source_qids == ("s1", "s2")
    assert len(larger.records) == 1


@pytest.mark.parametrize("phase", ["calibration", "evaluation", "dev", None])
def test_non_source_phase_is_rejected_even_outside_prefix(phase):
    wrong = make_source("not-prefix")
    wrong["phase"] = phase
    with pytest.raises(ValueError, match="source-phase"):
        build_source_bank(("s1",), (make_source(), wrong), scored())


def test_wrong_protocol_and_duplicate_report_are_rejected():
    report = make_source()
    with pytest.raises(ValueError, match="duplicate"):
        build_source_bank(("s1",), (report, report), scored())
    report["protocol"] = "other-protocol"
    with pytest.raises(ValueError, match="protocol"):
        build_source_bank(("s1",), (report,), scored())


@pytest.mark.parametrize("missing_arm", ["base", "fresh", "static"])
def test_missing_or_failed_arm_is_not_a_zero_score(missing_arm):
    report = make_source()
    report["arms"][missing_arm] = {"status": "failed"}
    bank, audit = build_source_bank(("s1",), (report,), scored())
    assert bank.records == ()
    assert audit["questions"]["s1"]["status"] == "incomplete_arms"
    assert "episode_delta_vs_static" not in audit["questions"]["s1"]


def test_no_report_and_no_feedback_are_unknown_not_zero():
    bank, audit = build_source_bank(("s1", "s2"), (make_source(),), {})
    assert bank.records == ()
    assert audit["questions"]["s1"]["episode_delta_vs_static"] == dict.fromkeys(
        ("answer_em", "answer_f1", "raw_support_recall")
    )
    assert audit["questions"]["s2"]["status"] == "missing_report"


@pytest.mark.parametrize("metric", ["answer_em", "answer_f1", "raw_support_recall"])
def test_any_one_observed_metric_gain_can_retain(metric):
    feedback = scored(fresh_em=0, fresh_f1=0.1, fresh_support=0.25)
    feedback["s1"]["fresh"][metric] = 1
    bank, _ = build_source_bank(("s1",), (make_source(),), feedback)
    assert len(bank.records) == 1
    assert bool(bank.published_specs) is (metric == "answer_em")


def test_tradeoffs_remain_in_audit_instead_of_combined_reward():
    feedback = scored(fresh_em=1, fresh_f1=0, fresh_support=0)
    bank, audit = build_source_bank(("s1",), (make_source(),), feedback)
    assert len(bank.records) == 1
    assert audit["questions"]["s1"]["episode_delta_vs_static"] == {
        "answer_em": 1,
        "answer_f1": -0.1,
        "raw_support_recall": -0.25,
    }


def test_no_gain_does_not_create_a_record():
    bank, _ = build_source_bank(
        ("s1",), (make_source(),), scored(fresh_em=0, fresh_f1=0.1, fresh_support=0.25)
    )
    assert bank.records == ()


@pytest.mark.parametrize("value", [True, float("nan"), -1, 2, "1"])
def test_invalid_feedback_rejected_instead_of_coerced(value):
    feedback = scored()
    feedback["s1"]["fresh"]["answer_em"] = value
    with pytest.raises(ValueError, match="metric"):
        build_source_bank(("s1",), (make_source(),), feedback)


def test_feedback_text_or_unregistered_metric_is_rejected():
    feedback = scored()
    feedback["s1"]["fresh"]["answer"] = "a source label must not enter"
    with pytest.raises(ValueError, match="scalar metrics"):
        build_source_bank(("s1",), (make_source(),), feedback)


def test_only_really_executed_proposals_are_retained():
    report = make_source()
    proposal = deepcopy(report["arms"]["fresh"]["episode"]["proposals"][0])
    proposal["spec"]["steps"][0]["template"] = "{entity} unused"
    report["arms"]["fresh"]["episode"]["proposals"].append(proposal)
    bank, audit = build_source_bank(("s1",), (report,), scored())
    assert len(bank.records) == 1
    assert audit["questions"]["s1"]["retained"] == 1


def test_two_executed_actions_do_not_each_get_individual_causal_credit():
    report = make_source(templates=("{entity} birthplace", "{entity} citizenship"))
    bank, audit = build_source_bank(("s1",), (report,), scored())
    assert len(bank.records) == 2
    for entry in audit["operators"].values():
        assert entry["individual_operator_effect"] == "unknown"
        assert entry["source_events"][0]["attribution_scope"] == "episode_association"
    second = next(
        e["source_events"][0]
        for e in audit["operators"].values()
        if e["source_events"][0]["proposal_number"] == 2
    )
    assert second["pre_action_evidence_ids"] == ["e0", "e1"]
    assert second["remaining_retrievals"] == 1


@pytest.mark.parametrize("template", ["Alice {entity} birthplace", "Alice birthplace"])
def test_obvious_source_name_hardcoding_stays_candidate_only(template):
    bank, audit = build_source_bank(("s1",), (make_source(templates=(template,)),), scored())
    assert len(bank.records) == 1
    assert bank.published_specs == ()
    event = next(iter(audit["operators"].values()))["source_events"][0]
    assert "possible_source_entity_or_number_hardcoding" in event["publication_flags"]


def test_episode_rejection_blocks_publication_not_retention_of_earlier_executed_action():
    report = make_source()
    report["arms"]["fresh"]["episode"].update(
        stop_reason="plan_rejected", rejected_error="ValueError"
    )
    bank, audit = build_source_bank(("s1",), (report,), scored())
    assert len(bank.records) == 1 and bank.published_specs == ()
    assert (
        "episode_had_rejected_plan"
        in next(iter(audit["operators"].values()))["source_events"][0]["publication_flags"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "changed_query",
        "changed_goal",
        "future_binding",
        "extra_field",
        "injected_evidence",
        "missing_search_ids",
    ],
)
def test_malformed_or_leaky_trajectory_is_quarantined(mutation):
    report = make_source()
    episode = report["arms"]["fresh"]["episode"]
    proposal = episode["proposals"][0]
    if mutation == "changed_query":
        episode["searches"][1]["query"] = "different query"
    elif mutation == "changed_goal":
        proposal["goal"]["original_question"] = "A different task"
    elif mutation == "future_binding":
        proposal["bindings"] = [{"name": "bridge", "value": "Alice", "evidence_ids": ["e1"]}]
    elif mutation == "extra_field":
        proposal["spec"]["gold_answer"] = "forbidden"
    elif mutation == "injected_evidence":
        episode["evidence"].append(
            {"evidence_id": "extra", "title": "extra", "sentence_id": 0, "text": "never retrieved"}
        )
    else:
        episode["searches"][1]["evidence_ids"] = ["missing"]
    bank, audit = build_source_bank(("s1",), (report,), scored())
    assert bank.records == ()
    assert audit["questions"]["s1"]["status"] == "invalid_source_trajectory"


def test_only_completed_stop_without_executed_action_has_no_candidate():
    report = make_source()
    report["arms"]["fresh"]["episode"] = deepcopy(report["arms"]["static"]["episode"])
    bank, audit = build_source_bank(("s1",), (report,), scored())
    assert bank.records == ()
    assert audit["questions"]["s1"]["status"] == "episode_gain_observed"


def test_protocol_and_budget_are_fixed_not_implicit_experiment_options():
    with pytest.raises(ValueError, match="frozen"):
        build_source_bank(("s1",), (), {}, retrieval_budget=4)
    with pytest.raises(ValueError, match="frozen"):
        build_source_bank(("s1",), (), {}, protocol_id="v2")


def test_different_initial_evidence_invalidates_paired_source_comparison():
    report = make_source()
    report["arms"]["static"]["episode"]["evidence"][0]["text"] = "Different initial facts"
    bank, audit = build_source_bank(("s1",), (report,), scored())
    assert bank.records == ()
    assert audit["questions"]["s1"]["status"] == "invalid_source_trajectory"


def test_known_hardcoding_cannot_be_laundered_by_a_second_different_question():
    reports = (
        make_source("s1", templates=("Alice {entity} birthplace",)),
        make_source(
            "s2",
            question="Which researcher was born there?",
            templates=("Alice {entity} birthplace",),
        ),
    )
    bank, audit = build_source_bank(("s1", "s2"), reports, {**scored("s1"), **scored("s2")})
    assert len(bank.records) == 1 and bank.published_specs == ()
    publication = next(iter(audit["operators"].values()))["publication"]
    assert publication["blocking_flags"] == ["possible_source_entity_or_number_hardcoding"]


def test_valid_earlier_evidence_binding_can_be_replayed_without_gold():
    report = make_source()
    proposal = report["arms"]["fresh"]["episode"]["proposals"][0]
    proposal["spec"]["gap_schema"] = []
    proposal["spec"]["steps"][0].update(
        template="{bridge} birthplace", requires_bindings=["bridge"]
    )
    proposal["gap"] = {}
    proposal["bindings"] = [{"name": "bridge", "value": "Alice", "evidence_ids": ["e0"]}]
    bank, _ = build_source_bank(("s1",), (report,), scored())
    assert len(bank.published_specs) == 1
