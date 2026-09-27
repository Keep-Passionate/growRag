"""Offline inheritance permits untouched claims, never replay of failed attempts."""

import json

import pytest

from growrag.experiments.shared_continuation import verify_unstarted_continuation

RUN = "2026-09-27_s2g_shared500_v1_0075_0100"
OPTIONS = {
    "manifest_sha256": "frozen",
    "generation_profile": "v5",
    "protocol": "v3",
    "model": "qwen",
    "question_ids": ["q80", "q81"],
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def prior(tmp_path):
    directory = tmp_path / RUN
    launch = {
        "run_id": RUN,
        "manifest_sha256": "frozen",
        "generation_profile": "v5",
        "protocol": "v3",
        "model": "qwen",
        "start": 75,
        "count": 25,
        "question_ids": [f"q{i}" for i in range(75, 100)],
    }
    call = {"trace_id": f"{RUN}/q79/S2G_AUTHOR_API4/01-judge", "status": "failed"}
    dump(directory / "launch_plan.json", launch)
    dump(
        directory / "reports.json",
        [
            {
                "question_id": "q79",
                "arms": {
                    "BASE1_AUTHOR_READER": {"calls": []},
                    "S2G_AUTHOR_API4": {"calls": [call]},
                },
            }
        ],
    )
    dump(directory / "final_budget.json", {"calls": [call]})
    dump(directory / "request_journal" / "0000_intent.json", {"trace_id": call["trace_id"]})
    dump(directory / "request_journal" / "0000_after.json", {"calls": [call]})
    dump(directory / "api_audit" / "0000.json", {"trace_id": call["trace_id"]})
    (directory / "events.jsonl").write_text(
        json.dumps({"kind": "api_request", "question_id": f"{RUN}/q79/S2G_AUTHOR_API4"})
        + "\n"
        + json.dumps({"kind": "exit", "status": "failed"})
        + "\n"
    )
    return tmp_path, directory


def verify(prior, **changes):
    root, _ = prior
    return verify_unstarted_continuation(root, RUN, **{**OPTIONS, **changes})


def test_untouched_only_proof_records_hashes_offsets_and_direct_parent(prior):
    proof = verify(prior)
    assert proof["question_ids"] == ["q80", "q81"]
    assert proof["checked_global_offsets"] == [80, 81]
    assert proof["validated_prior_runs"] == [RUN]
    assert set(proof["prior_record_sha256"]) == {
        "launch_plan.json",
        "reports.json",
        "final_budget.json",
        "events.jsonl",
    }
    assert all(len(value) == 64 for value in proof["prior_record_sha256"].values())


def test_failed_question_cannot_be_retried(prior):
    with pytest.raises(ValueError, match="prior report"):
        verify(prior, question_ids=["q79"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_sha256", "other"),
        ("model", "other"),
        ("protocol", "other"),
        ("generation_profile", "other"),
        ("question_ids", ["absent"]),
        ("question_ids", []),
        ("question_ids", ["q80", "q80"]),
    ],
)
def test_scope_mismatches_fail_closed(prior, field, value):
    with pytest.raises(ValueError):
        verify(prior, **{field: value})


@pytest.mark.parametrize("name", ["../old", "C:/old", "old", "../" + RUN])
def test_unsafe_parent_names_rejected(prior, name):
    with pytest.raises(ValueError, match="basename"):
        verify_unstarted_continuation(prior[0], name, **OPTIONS)


@pytest.mark.parametrize("artifact", ["reports.json", "final_budget.json", "events.jsonl"])
def test_missing_completion_artifacts_reject(prior, artifact):
    (prior[1] / artifact).unlink()
    with pytest.raises(FileNotFoundError):
        verify(prior)


@pytest.mark.parametrize(
    "event",
    [
        {"kind": "api_request", "trace_id": f"{RUN}/q80/S2G_AUTHOR_API4/01-judge"},
        {"kind": "arm_start", "question_id": "q80"},
        {"kind": "nested", "data": {"question_id": f"{RUN}/q80/BASE1_AUTHOR_READER"}},
    ],
)
def test_any_target_event_blocks_inheritance(prior, event):
    path = prior[1] / "events.jsonl"
    path.write_text(json.dumps(event) + "\n" + json.dumps({"kind": "exit", "status": "failed"}))
    with pytest.raises(ValueError, match="prior events"):
        verify(prior)


def test_unreported_call_also_blocks_inheritance(prior):
    path = prior[1] / "final_budget.json"
    budget = json.loads(path.read_text())
    budget["calls"].append({"trace_id": f"{RUN}/q80/BASE1_AUTHOR_READER/01-answer"})
    dump(path, budget)
    with pytest.raises(ValueError, match="prior call"):
        verify(prior)


@pytest.mark.parametrize("name", ["report.json", "BASE1_AUTHOR_READER_execution.json"])
def test_question_files_even_without_global_report_block_inheritance(prior, name):
    dump(prior[1] / "questions" / "0080" / name, {})
    with pytest.raises(ValueError, match="question directory"):
        verify(prior)


def test_empty_question_directory_also_blocks_inheritance(prior):
    (prior[1] / "questions" / "0080").mkdir(parents=True)
    with pytest.raises(ValueError, match="question directory"):
        verify(prior)


@pytest.mark.parametrize("folder", ["request_journal", "api_audit"])
def test_raw_target_evidence_blocks_even_if_absent_from_summary(prior, folder):
    filename = "0001_intent.json" if folder == "request_journal" else "extra.json"
    dump(prior[1] / folder / filename, {"trace_id": f"{RUN}/q80/S2G_AUTHOR_API4/01"})
    with pytest.raises(ValueError, match="prior request evidence"):
        verify(prior)


def test_missing_after_blocks_even_if_final_budget_exists(prior):
    (prior[1] / "request_journal" / "0000_after.json").unlink()
    with pytest.raises(ValueError, match="lacks corresponding after"):
        verify(prior)


def test_last_after_must_equal_final_budget(prior):
    path = prior[1] / "request_journal" / "0000_after.json"
    value = json.loads(path.read_text())
    value["reserved_cny"] = 999
    dump(path, value)
    with pytest.raises(ValueError, match="differs from final budget"):
        verify(prior)


def test_unknown_audit_trace_blocks_inheritance(prior):
    dump(prior[1] / "api_audit" / "extra.json", {"trace_id": f"{RUN}/q78/S2G_AUTHOR_API4/01"})
    with pytest.raises(ValueError, match="unaccounted"):
        verify(prior)


@pytest.mark.parametrize(
    "events",
    [
        [],
        [{"kind": "api_request"}],
        [{"kind": "exit", "status": "failed"}, {"kind": "api_request"}],
    ],
)
def test_unsealed_or_post_exit_activity_rejected(prior, events):
    (prior[1] / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match="closed|after terminal"):
        verify(prior)


def test_bounded_ancestry_proves_every_old_claim(prior):
    root, directory = prior
    child = "2026-09-27_s2g_shared500_v1_0080_0100"
    child_dir = root / child
    launch = json.loads((directory / "launch_plan.json").read_text())
    launch.update(
        run_id=child,
        start=80,
        count=20,
        question_ids=[f"q{i}" for i in range(80, 100)],
        continuation_of=RUN,
    )
    dump(child_dir / "launch_plan.json", launch)
    dump(child_dir / "reports.json", [])
    dump(child_dir / "final_budget.json", {"calls": []})
    (child_dir / "request_journal").mkdir()
    (child_dir / "api_audit").mkdir()
    (child_dir / "events.jsonl").write_text(json.dumps({"kind": "exit", "status": "failed"}))
    proof = verify_unstarted_continuation(root, child, **{**OPTIONS, "question_ids": ["q81"]})
    assert proof["validated_prior_runs"] == [child, RUN]
    launch["continuation_of"] = child
    dump(child_dir / "launch_plan.json", launch)
    with pytest.raises(ValueError, match="cycle"):
        verify_unstarted_continuation(root, child, **OPTIONS)
