"""Synthetic explicit-continuation audit, never real API calls or held-out labels."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from growrag.experiments.bounded_continuation import review_unstarted
from growrag.experiments.protocol import RuntimeQuestion

SCHEMA = "growrag-bounded-system-v1"


class GoldGuard:
    def __init__(self, index):
        self.question = RuntimeQuestion(f"q{index}", f"Synthetic question {index}?")

    @property
    def gold(self):
        raise AssertionError("review must never access any gold labels")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def modify(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    write(path, value)


def fixture(tmp_path, *, started=1):
    runs = tmp_path / "runs"
    prior = runs / "stopped"
    examples = tuple(GoldGuard(i) for i in range(4))
    frozen = {
        "data": {"question_ids": [e.question.question_id for e in examples], "sha256": "data"},
        "source": {"sha256": "memory"},
        "prompts": {"prompt-v1": {"text": "frozen", "sha256": "prompt"}},
        "model": "fixed-model-v1",
    }
    write(prior / "launch_plan.json", {"schema_version": SCHEMA, "arms": ["BASE", "MEM"], **frozen})
    write(
        prior / "summary.json",
        {
            "planned": 4,
            "started": started,
            "reported": started,
            "complete": started - 1,
            "not_started": 4 - started,
        },
    )
    calls = []
    for i in range(started):
        call = {"trace_id": f"paid-{i}", "api_requests": 1}
        calls.append(call)
        write(
            prior / "questions" / f"{i:03d}" / "report.json",
            {
                "schema_version": SCHEMA,
                "execution_kind": "real",
                "question": asdict(examples[i].question),
                "arms": {
                    "BASE": {"status": "completed", "feedback": {"em": 1}, "calls": [call]},
                    "MEM": {
                        "status": "failed" if i == started - 1 else "completed",
                        "feedback": None if i == started - 1 else {"em": 0},
                        "calls": [],
                    },
                },
            },
        )
    write(
        prior / "final_budget.json",
        {
            "api_requests": len(calls),
            "calls": calls,
            "block_reason": "bounded_component_failure",
        },
    )
    return runs, prior, examples, frozen


def test_only_untouched_suffix_returns_and_all_attempts_excluded(tmp_path):
    runs, prior, examples, frozen = fixture(tmp_path, started=2)
    before = {str(p): p.read_bytes() for p in prior.rglob("*.json")}
    remaining, provenance, ledger = review_unstarted(prior, runs, examples, **frozen)
    assert remaining == examples[2:]
    assert provenance["excluded_attempted_question_ids"] == ["q0", "q1"]
    assert provenance["remaining_question_ids"] == ["q2", "q3"]
    assert provenance["prior_complete"] == 1 and provenance["prior_started"] == 2
    assert ledger == "stopped/final_budget.json"
    assert all(len(digest) == 64 for digest in provenance["sha256"].values())
    assert before == {str(p): p.read_bytes() for p in prior.rglob("*.json")}


@pytest.mark.parametrize("field", ["data", "source", "prompts", "model"])
def test_configuration_changes_cannot_be_disguised_as_same_experiment(tmp_path, field):
    runs, prior, examples, frozen = fixture(tmp_path)
    modify(prior / "launch_plan.json", lambda plan: plan.update({field: "changed"}))
    with pytest.raises(ValueError, match=f"prior {field}"):
        review_unstarted(prior, runs, examples, **frozen)


def test_parser_only_code_metadata_change_does_not_change_frozen_prompts(tmp_path):
    runs, prior, examples, frozen = fixture(tmp_path)
    modify(prior / "launch_plan.json", lambda plan: plan.update({"parser": "old", "git": "old"}))
    remaining, _, _ = review_unstarted(prior, runs, list(examples), **frozen)
    assert remaining == examples[1:]


@pytest.mark.parametrize(
    "change",
    [
        {"started": 0, "reported": 0},
        {"started": 4, "reported": 4},
        {"started": True},
        {"started": 2, "reported": 1},
        {"not_started": 1},
        {"planned": 5},
        {"complete": 2},
    ],
)
def test_incomplete_or_fabricated_prefix_counts_fail_closed(tmp_path, change):
    runs, prior, examples, frozen = fixture(tmp_path)
    modify(prior / "summary.json", lambda summary: summary.update(change))
    with pytest.raises(ValueError):
        review_unstarted(prior, runs, examples, **frozen)


@pytest.mark.parametrize("field,value", [("question_id", "q2"), ("text", "different wording")])
def test_cannot_skip_or_reorder_questions_based_on_their_results(tmp_path, field, value):
    runs, prior, examples, frozen = fixture(tmp_path)
    modify(
        prior / "questions" / "000" / "report.json",
        lambda report: report["question"].update({field: value}),
    )
    with pytest.raises(ValueError, match="original attempted prefix"):
        review_unstarted(prior, runs, examples, **frozen)


def test_reordered_or_shortened_example_list_does_not_match_frozen_data_order(tmp_path):
    runs, prior, examples, frozen = fixture(tmp_path)
    with pytest.raises(ValueError, match="full frozen data order"):
        review_unstarted(prior, runs, examples[1:], **frozen)
    with pytest.raises(ValueError, match="full frozen data order"):
        review_unstarted(prior, runs, examples[::-1], **frozen)


def test_extra_started_directory_prevents_hiding_an_unreported_question(tmp_path):
    runs, prior, examples, frozen = fixture(tmp_path)
    (prior / "questions" / "001").mkdir()
    with pytest.raises(ValueError, match="contiguous attempted prefix"):
        review_unstarted(prior, runs, examples, **frozen)


def test_missing_report_cannot_be_treated_as_unstarted(tmp_path):
    runs, prior, examples, frozen = fixture(tmp_path)
    (prior / "questions" / "000" / "report.json").unlink()
    with pytest.raises(FileNotFoundError):
        review_unstarted(prior, runs, examples, **frozen)


@pytest.mark.parametrize(
    "change", [{"api_requests": 2}, {"block_reason": None}, {"block_reason": ""}]
)
def test_final_budget_must_be_a_stopped_audited_run(tmp_path, change):
    runs, prior, examples, frozen = fixture(tmp_path)
    modify(prior / "final_budget.json", lambda ledger: ledger.update(change))
    with pytest.raises(ValueError, match="stopped run"):
        review_unstarted(prior, runs, examples, **frozen)


def test_unreported_api_attempt_prevents_unsafe_continuation(tmp_path):
    runs, prior, examples, frozen = fixture(tmp_path)
    modify(
        prior / "final_budget.json",
        lambda ledger: ledger.update(
            {
                "api_requests": 2,
                "calls": [*ledger["calls"], {"trace_id": "unreported", "api_requests": 1}],
            }
        ),
    )
    with pytest.raises(ValueError, match="unique requests"):
        review_unstarted(prior, runs, examples, **frozen)


def test_prior_run_must_remain_inside_runs_root(tmp_path):
    runs, prior, examples, frozen = fixture(tmp_path)
    with pytest.raises(ValueError, match="strictly inside"):
        review_unstarted(prior, prior, examples, **frozen)
    with pytest.raises(ValueError, match="strictly inside"):
        review_unstarted(tmp_path, runs, examples, **frozen)


def test_schema_provenance_and_complete_count_are_not_inferred(tmp_path):
    runs, prior, examples, frozen = fixture(tmp_path)
    modify(
        prior / "questions" / "000" / "report.json",
        lambda report: report.update({"execution_kind": "mock"}),
    )
    with pytest.raises(ValueError, match="real execution"):
        review_unstarted(prior, runs, examples, **frozen)


def test_runtime_question_required_but_gold_never_touched(tmp_path):
    runs, prior, examples, frozen = fixture(tmp_path)
    bad = (SimpleNamespace(question="not typed"), *examples[1:])
    with pytest.raises(TypeError, match="RuntimeQuestion"):
        review_unstarted(prior, runs, bad, **frozen)
