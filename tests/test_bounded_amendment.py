"""Synthetic chain review: no real API, no held-out questions or gold access."""

import hashlib
import json
from dataclasses import asdict

import pytest

from growrag.experiments.bounded_amendment import review_amended_unstarted
from growrag.experiments.protocol import RuntimeQuestion

SCHEMA = "growrag-bounded-system-v1"


class GoldGuard:
    def __init__(self, index):
        self.question = RuntimeQuestion(f"q{index}", f"Synthetic question {index}?")

    @property
    def gold(self):
        raise AssertionError("amendment review must never access gold")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def modify(path, action):
    value = json.loads(path.read_text(encoding="utf-8"))
    action(value)
    write(path, value)


def fixture(tmp_path):
    runs = tmp_path / "runs"
    priors = (runs / "first", runs / "second")
    examples = tuple(GoldGuard(i) for i in range(8))
    frozen = {
        "data": {"question_ids": [f"q{i}" for i in range(8)], "sha256": "data"},
        "source": {"sha256": "memory"},
        "prompts": {"judge": {"text": "new minimal requirements", "sha256": "new"}},
        "model": "fixed-model-v1",
        "revisions": {"reason": "fix JSON and paired sharing", "json_mode": True},
    }
    offset = 0
    previous_hashes = None
    for index, (prior, started) in enumerate(zip(priors, (1, 3), strict=True)):
        planned = [f"q{i}" for i in range(offset, 8)]
        plan = {
            "schema_version": SCHEMA,
            "arms": ["BASE", "MEM"],
            "data": frozen["data"],
            "source": frozen["source"],
            "model": frozen["model"],
            "prompts": {"judge": {"text": "old prompt", "sha256": "old"}},
        }
        if index:
            plan["planned_question_ids"] = planned
            plan["continuation"] = {
                "prior_run": str(priors[0]),
                "remaining_question_ids": planned,
                "excluded_attempted_question_ids": ["q0"],
                "reviewed_extra_ledger": "first/final_budget.json",
                "prior_started": 1,
                "prior_complete": 0,
                "prior_api_requests": 1,
                "sha256": previous_hashes,
            }
        write(prior / "launch_plan.json", plan)
        write(
            prior / "summary.json",
            {
                "planned": len(planned),
                "started": started,
                "reported": started,
                "complete": started - 1,
                "not_started": len(planned) - started,
            },
        )
        calls = []
        for local in range(started):
            trace = f"paid-{index}-{local}"
            call = {
                "trace_id": trace,
                "api_requests": 1,
                "input_tokens": 20,
                "output_tokens": 5,
                "status": "completed",
                "audit_path": str(prior / "api_audit" / f"{trace}.json"),
            }
            calls.append(call)
            write(
                prior / "api_audit" / f"{trace}.json",
                {
                    **call,
                    "transport_source": "live_api",
                    "request": {"model": frozen["model"]},
                },
            )
            write(
                prior / "questions" / f"{local:03d}" / "report.json",
                {
                    "schema_version": SCHEMA,
                    "execution_kind": "real",
                    "question": asdict(examples[offset + local].question),
                    "arms": {
                        "BASE": {"feedback": {"em": 1}, "calls": [call]},
                        "MEM": {
                            "feedback": None if local == started - 1 else {"em": 0},
                            "calls": [],
                        },
                    },
                },
            )
        write(
            prior / "final_budget.json",
            {"api_requests": len(calls), "calls": calls, "block_reason": "failed judge"},
        )
        paths = [
            prior / "launch_plan.json",
            prior / "summary.json",
            prior / "final_budget.json",
            *(prior / "questions").glob("*/report.json"),
        ]
        previous_hashes = {
            str(path.relative_to(prior)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }
        offset += started
    return runs, priors, examples, frozen


def test_declared_amendment_returns_only_untouched_suffix_without_writes_or_gold(tmp_path):
    runs, priors, examples, frozen = fixture(tmp_path)
    before = {str(path): path.read_bytes() for path in runs.rglob("*.json")}
    remaining, provenance, ledgers = review_amended_unstarted(priors, runs, examples, **frozen)
    assert remaining == examples[4:]
    assert ledgers == ("first/final_budget.json", "second/final_budget.json")
    assert provenance["remaining_question_ids"] == ["q4", "q5", "q6", "q7"]
    assert provenance["excluded_attempted_question_ids"] == ["q0", "q1", "q2", "q3"]
    assert all(review["changed_prompt_keys"] == ["judge"] for review in provenance["prior_runs"])
    assert provenance["new_prompts"] == frozen["prompts"]
    assert before == {str(path): path.read_bytes() for path in runs.rglob("*.json")}


@pytest.mark.parametrize("field", ["data", "source", "model"])
def test_amendment_cannot_change_frozen_data_memory_or_model(tmp_path, field):
    runs, priors, examples, frozen = fixture(tmp_path)
    modify(priors[1] / "launch_plan.json", lambda plan: plan.update({field: "changed"}))
    with pytest.raises(ValueError, match=f"prior {field}"):
        review_amended_unstarted(priors, runs, examples, **frozen)


@pytest.mark.parametrize("reason", [None, "", "   ", 123])
def test_protocol_change_requires_explicit_reason(tmp_path, reason):
    runs, priors, examples, frozen = fixture(tmp_path)
    frozen["revisions"] = {"reason": reason}
    with pytest.raises(ValueError, match="explicit reason"):
        review_amended_unstarted(priors, runs, examples, **frozen)


def test_chain_hash_mismatch_preserves_old_failure_not_silent_amendment(tmp_path):
    runs, priors, examples, frozen = fixture(tmp_path)
    modify(priors[0] / "summary.json", lambda summary: summary.update({"extra": "edited"}))
    with pytest.raises(ValueError, match="authenticate"):
        review_amended_unstarted(priors, runs, examples, **frozen)


@pytest.mark.parametrize(
    "change", [{"planned": 8}, {"reported": 2}, {"not_started": 5}, {"started": True}]
)
def test_invalid_suffix_counts_rejected(tmp_path, change):
    runs, priors, examples, frozen = fixture(tmp_path)
    modify(priors[1] / "summary.json", lambda summary: summary.update(change))
    with pytest.raises(ValueError, match="stopped prefix"):
        review_amended_unstarted(priors, runs, examples, **frozen)


def test_cannot_skip_or_select_question_by_result(tmp_path):
    runs, priors, examples, frozen = fixture(tmp_path)
    modify(
        priors[1] / "questions" / "000" / "report.json",
        lambda report: report["question"].update({"question_id": "q4"}),
    )
    with pytest.raises(ValueError, match="original attempted prefix"):
        review_amended_unstarted(priors, runs, examples, **frozen)


def test_hidden_started_directory_prevents_claiming_unstarted(tmp_path):
    runs, priors, examples, frozen = fixture(tmp_path)
    (priors[1] / "questions" / "003").mkdir()
    with pytest.raises(ValueError, match="exact attempted prefix"):
        review_amended_unstarted(priors, runs, examples, **frozen)


def test_extra_unaccounted_live_audit_rejected(tmp_path):
    runs, priors, examples, frozen = fixture(tmp_path)
    write(priors[1] / "api_audit" / "extra.json", {"transport_source": "live_api"})
    with pytest.raises(ValueError, match="unaccounted audit"):
        review_amended_unstarted(priors, runs, examples, **frozen)


def test_ledger_call_usage_must_match_report_and_audit(tmp_path):
    runs, priors, examples, frozen = fixture(tmp_path)
    modify(
        priors[1] / "api_audit" / "paid-1-0.json",
        lambda audit: audit.update({"output_tokens": 999}),
    )
    with pytest.raises(ValueError, match="usage/status"):
        review_amended_unstarted(priors, runs, examples, **frozen)


def test_chain_order_cannot_be_reversed_or_start_at_continuation(tmp_path):
    runs, priors, examples, frozen = fixture(tmp_path)
    with pytest.raises(ValueError):
        review_amended_unstarted(priors[::-1], runs, examples, **frozen)
    with pytest.raises(ValueError):
        review_amended_unstarted(priors[1:], runs, examples, **frozen)


def test_chain_repeated_root_rejected(tmp_path):
    runs, priors, examples, frozen = fixture(tmp_path)
    with pytest.raises(ValueError, match="duplicate prior"):
        review_amended_unstarted((priors[0], priors[0]), runs, examples, **frozen)


def test_report_call_cannot_differ_from_ledger(tmp_path):
    runs, priors, examples, frozen = fixture(tmp_path)
    modify(
        priors[1] / "questions" / "000" / "report.json",
        lambda report: report["arms"]["BASE"]["calls"][0].update({"input_tokens": 30}),
    )
    with pytest.raises(ValueError, match="identical ledger calls"):
        review_amended_unstarted(priors, runs, examples, **frozen)


def test_arm_order_change_rejected(tmp_path):
    runs, priors, examples, frozen = fixture(tmp_path)
    modify(priors[1] / "launch_plan.json", lambda plan: plan.update({"arms": ["MEM", "BASE"]}))
    with pytest.raises(ValueError, match="arm order"):
        review_amended_unstarted(priors, runs, examples, **frozen)
