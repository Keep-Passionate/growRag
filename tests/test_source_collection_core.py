"""Synthetic-only source collection tests; no live API and no real dataset labels."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import source_collection_core as core


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def synthetic_question(qid):
    return core.RuntimeQuestion(qid, "Synthetic source question " + qid, "synthetic")


def test_runtime_loads_only_source_order_and_no_labels(tmp_path, monkeypatch):
    rows = [{"question_id": "q0", "text": "question", "dataset": "synthetic"}]
    p = tmp_path / "source_runtime_questions.jsonl"
    p.write_text(json.dumps(rows[0]) + "\n")
    plan = {
        "roles": {"source": ["q0"]},
        "artifacts": {
            p.name: {
                "sha256": core._sha(p),
                "rows": 1,
                "contains_gold": False,
            }
        },
    }
    monkeypatch.setattr(core, "SOURCE_COUNT", 1)
    monkeypatch.setattr(core, "source_plan", lambda _: plan)
    questions, _ = core.load_source_questions(tmp_path / "manifest.json")
    assert [q.question_id for q in questions] == ["q0"]
    assert not hasattr(questions[0], "answer")
    rows[0]["answer"] = "DO NOT LOAD"
    p.write_text(json.dumps(rows[0]) + "\n")
    plan["artifacts"][p.name]["sha256"] = core._sha(p)
    with pytest.raises(ValueError, match="label"):
        core.load_source_questions(tmp_path / "manifest.json")


def test_non_sources_never_project_gold_fields(tmp_path, monkeypatch):
    selected = "a" * 24
    records = [
        {"_id": "b" * 24, "question": "reserved probe", "answer": "PROBE_SECRET"},
        {
            "_id": selected,
            "question": "source",
            "answer": "1901",
            "supporting_facts": [["Northbridge", 0]],
            "context": [["Northbridge", ["Northbridge was founded in 1901."]]],
        },
        {"_id": "c" * 24, "question": "old development", "answer": "OLD_SECRET"},
    ]
    path = tmp_path / "raw.json"
    dump(path, records)
    monkeypatch.setattr(core, "SOURCE_SHA", hashlib.sha256(path.read_bytes()).hexdigest())
    actual_project, fields_seen = core._project, []

    def spy(raw, fields):
        qid = actual_project(raw, {"_id"})["_id"]
        fields_seen.append((qid, fields))
        assert qid == selected or fields == {"_id"}
        return actual_project(raw, fields)

    monkeypatch.setattr(core, "_project", spy)
    result = core.project_source_gold(path, [synthetic_question(selected)])
    assert list(result) == [selected] and result[selected].answers == ("1901",)
    assert len(result[selected].exact_support) == 1
    assert [fields for qid, fields in fields_seen if qid == selected][-1] == {
        "_id",
        "answer",
        "supporting_facts",
        "context",
    }


def test_collection_never_invokes_gold_and_seals_batch(tmp_path, monkeypatch):
    questions = [synthetic_question("q0"), synthetic_question("q1")]
    client = SimpleNamespace(block_reason=None, calls=[], attempts=0)

    def execute(question, index, client, upstream, directory, progress, *, run_id):
        directory.mkdir(parents=True)
        return {
            arm: {
                "status": "completed",
                "feedback": None,
                "calls": [],
                "result": {
                    "question_id": question.question_id,
                    "answer": "synthetic",
                    "retrieval_rounds": 1,
                },
            }
            for arm in core.ARMS
        }

    monkeypatch.setattr(core, "execute_pair", execute)
    monkeypatch.setattr(
        core, "project_source_gold", lambda *a: pytest.fail("no gold during collection")
    )
    summary = core.collect_batch(
        questions,
        SimpleNamespace(index=None),
        client,
        None,
        tmp_path,
        lambda _: None,
        run_id="synthetic",
        start=0,
    )
    rows = json.loads((tmp_path / "predictions.json").read_bytes())
    freeze = json.loads((tmp_path / "predictions_frozen.json").read_bytes())
    assert freeze["reports_sha256_before_scoring"] == core.fingerprint(rows)
    assert summary["gold_loaded"] is False and summary["memory_cards_created"] == 0
    assert all(r["arms"][a]["feedback"] is None for r in rows for a in core.ARMS)


def test_incomplete_collection_blocks_before_gold(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "source_plan", lambda _: {"roles": {"source": ["q0"]}})
    monkeypatch.setattr(core, "project_source_gold", lambda *a: pytest.fail("no incomplete gold"))
    with pytest.raises(ValueError, match="complete source64"):
        core.score_collection(
            tmp_path,
            Path("manifest"),
            SimpleNamespace(questions=()),
            Path("raw"),
            tmp_path / "score",
        )
    assert not (tmp_path / "score").exists()


@pytest.mark.parametrize("start,count", [(0, 0), (0, 17), (64, 1), (63, 2), (True, 1), (0, True)])
def test_source_batch_bounds(start, count):
    with pytest.raises(ValueError):
        core.batch_identity(start, count)


def frozen_collection(tmp_path, monkeypatch, *, start=0, count=2, total=None):
    """Entire synthetic two-question collection; no real data or model outputs."""
    total = total or count
    monkeypatch.setattr(core, "SOURCE_COUNT", total)
    ids = [chr(97 + offset) * 24 for offset in range(total)]
    questions = [synthetic_question(qid) for qid in ids]
    chosen = questions[start : start + count]
    chosen_ids = [question.question_id for question in chosen]
    plan = {"roles": {"source": ids}}
    root = tmp_path / core.batch_identity(start, count)
    root.mkdir()
    snapshot = {
        "files": {
            "src/synthetic.py": {"text": "pass\n", "sha256": hashlib.sha256(b"pass\n").hexdigest()}
        }
    }
    source_text = Path(core.__file__).read_text(encoding="utf-8")
    snapshot["files"][core.CORE_PATH] = {
        "text": source_text,
        "sha256": hashlib.sha256(source_text.encode()).hexdigest(),
    }
    snapshot["sha256"] = core.fingerprint(snapshot["files"])
    launch = {
        "protocol": core.PROTOCOL,
        "manifest_sha256": core.PLAN_SHA,
        "start": start,
        "count": count,
        "run_id": root.name,
        "question_ids": chosen_ids,
        "max_calls": 11 * count,
        "model": core.PILOT_MODEL,
        "git": {"commit": "synthetic", "worktree_dirty": False},
        "generation_profile": core.GENERATION_PROFILE,
        "backend_output_caps": core.SHARED_OUTPUT_CAPS,
        "max_retrieval_rounds": 4,
        "top_docs": 6,
        "gap_profile": "paper_k1",
        "temperature": 0,
        "top_p": 1,
        "enable_thinking": False,
        "source_sha256": snapshot["sha256"],
        "author": {"commit": "synthetic-author"},
        "index_path": "synthetic-index",
        "runtime_metadata": {"corpus": "synthetic"},
    }
    dump(root / "launch_plan.json", launch)
    dump(
        tmp_path / f"{root.name}.claim.json",
        {
            "protocol": core.PROTOCOL,
            "manifest_sha256": core.PLAN_SHA,
            "question_ids": chosen_ids,
            "plan_sha256": core.fingerprint(launch),
        },
    )
    dump(root / "source_snapshot.json", snapshot)
    (root / "events.jsonl").write_text('{"kind":"exit","status":"completed"}\n')
    (root / "api_audit").mkdir()
    reports, calls = [], []
    for offset, question in enumerate(chosen, start):
        directory = root / "questions" / f"{offset:04d}"
        directory.mkdir(parents=True)
        arms = {}
        for arm in core.ARMS:
            trace = f"{root.name}/{question.question_id}/{arm}/answer"
            audit_path = root / "api_audit" / f"{offset}-{arm}.json"
            call = {
                "trace_id": trace,
                "audit_path": str(audit_path.resolve()),
                "prompt_version": "s2g-author-5d842a6-answer-api-v1",
                "api_requests": 1,
                "input_tokens": 5,
                "output_tokens": 2,
                "reserved_cny": 0.01,
                "estimated_actual_cny": 0.001,
                "status": "completed",
            }
            calls.append(call)
            dump(
                audit_path,
                {
                    "status": "completed",
                    "http_status": 200,
                    "retry_count": 0,
                    "network_attempted": True,
                    "transport_source": "live_api",
                    "response": {"model": core.PILOT_MODEL},
                    "request": {
                        "model": core.PILOT_MODEL,
                        "temperature": 0,
                        "top_p": 1,
                        "enable_thinking": False,
                        "stream": False,
                        "max_tokens": core.SHARED_OUTPUT_CAPS["answer"],
                    },
                    **{
                        k: call[k]
                        for k in (
                            "trace_id",
                            "prompt_version",
                            "api_requests",
                            "input_tokens",
                            "output_tokens",
                        )
                    },
                },
            )
            arms[arm] = {
                "status": "completed",
                "feedback": None,
                "calls": [call],
                "result": {"question_id": question.question_id, "answer": "synthetic"},
            }
            dump(directory / f"{arm}_execution.json", arms[arm])
        report = {
            "question_id": question.question_id,
            "question": question.text,
            "offset": offset,
            "complete_pair": True,
            "arms": arms,
        }
        reports.append(report)
        dump(directory / "prediction_report.json", report)
    dump(root / "predictions.json", reports)
    dump(
        root / "predictions_frozen.json",
        {
            "run_id": root.name,
            "question_ids": chosen_ids,
            "reports_sha256_before_scoring": core.fingerprint(reports),
        },
    )
    dump(
        root / "final_budget.json",
        {
            "calls": calls,
            "api_requests": 2 * count,
            "reserved_cny": sum(c["reserved_cny"] for c in calls),
        },
    )
    return root, plan, questions


def test_full_global_collection_verification(tmp_path, monkeypatch):
    _, plan, questions = frozen_collection(tmp_path, monkeypatch)
    reports, hashes = core.verify_collection(tmp_path, plan, questions)
    assert len(reports) == len(hashes) == 2
    assert all(r["arms"][a]["feedback"] is None for r in reports for a in core.ARMS)


@pytest.mark.parametrize(
    "mutation",
    [
        "claim",
        "source",
        "exit",
        "seal_id",
        "seal_hash",
        "per_question",
        "arm_file",
        "question_text",
        "audit_model",
        "audit_tokens",
        "audit_temperature",
        "ledger_total",
        "ledger_reserve",
        "ledger_orphan",
        "foreign_exposure",
        "incomplete",
    ],
)
def test_global_collection_tampering_blocks_before_gold(tmp_path, monkeypatch, mutation):
    root, plan, questions = frozen_collection(tmp_path, monkeypatch)
    reports = json.loads((root / "predictions.json").read_bytes())
    qpath = root / "questions/0000/prediction_report.json"
    if mutation == "foreign_exposure":
        dump(tmp_path / "foreign.claim.json", {"question_ids": [questions[0].question_id]})
    elif mutation == "exit":
        (root / "events.jsonl").write_text('{"kind":"exit","status":"failed"}\n')
    elif mutation == "incomplete":
        (root / "final_budget.json").unlink()
    elif mutation == "question_text":
        # Change both stored copies and reseal: frozen runtime text still rejects it.
        reports[0]["question"] = "Different synthetic question"
        dump(qpath, reports[0])
        dump(root / "predictions.json", reports)
        seal = json.loads((root / "predictions_frozen.json").read_bytes())
        seal["reports_sha256_before_scoring"] = core.fingerprint(reports)
        dump(root / "predictions_frozen.json", seal)
    else:
        path = {
            "claim": tmp_path / f"{root.name}.claim.json",
            "source": root / "source_snapshot.json",
            "seal_id": root / "predictions_frozen.json",
            "seal_hash": root / "predictions_frozen.json",
            "per_question": qpath,
            "arm_file": qpath.parent / f"{core.ARMS[0]}_execution.json",
            "audit_model": next((root / "api_audit").glob("*.json")),
            "audit_tokens": next((root / "api_audit").glob("*.json")),
            "audit_temperature": next((root / "api_audit").glob("*.json")),
            "ledger_total": root / "final_budget.json",
            "ledger_reserve": root / "final_budget.json",
            "ledger_orphan": root / "final_budget.json",
        }[mutation]
        value = json.loads(path.read_bytes())
        if mutation == "claim":
            value["plan_sha256"] = "changed"
        elif mutation == "source":
            value["files"]["src/synthetic.py"]["text"] = "changed"
        elif mutation == "seal_id":
            value["run_id"] = "changed"
        elif mutation == "seal_hash":
            value["reports_sha256_before_scoring"] = "changed"
        elif mutation == "per_question":
            value["offline_gold_answers"] = ["DO NOT LOAD"]
        elif mutation == "arm_file":
            value["feedback"] = {"answer_em": 1}
        elif mutation == "audit_model":
            value["request"]["model"] = "changed"
        elif mutation == "audit_tokens":
            value["input_tokens"] = 999
        elif mutation == "audit_temperature":
            value["request"]["temperature"] = 0.5
        elif mutation == "ledger_total":
            value["api_requests"] = 5
        elif mutation == "ledger_reserve":
            value["reserved_cny"] = 0.05
        else:
            value["calls"].append(dict(value["calls"][0], trace_id="orphan"))
        dump(path, value)
    monkeypatch.setattr(core, "source_plan", lambda _: plan)
    monkeypatch.setattr(
        core,
        "project_source_gold",
        lambda *a: pytest.fail("tampered collection must not read gold"),
    )
    with pytest.raises(ValueError):
        core.score_collection(
            tmp_path,
            Path("manifest"),
            SimpleNamespace(questions=questions),
            Path("raw"),
            tmp_path / "scored",
        )
    assert not (tmp_path / "scored").exists()


def mark_closed_length_failure(root, *, attempted=None, failed_arm=None):
    """Transform synthetic fixtures only, preserving a terminal failed prefix."""
    reports = json.loads((root / "predictions.json").read_bytes())
    attempted = attempted or len(reports)
    for row in reports[attempted:]:
        for outcome in row["arms"].values():
            for call in outcome["calls"]:
                Path(call["audit_path"]).unlink()
        directory = root / "questions" / f"{row['offset']:04d}"
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
    reports = reports[:attempted]
    row, failed_arm = reports[-1], failed_arm or core.ARMS[1]
    outcome = row["arms"][failed_arm]
    outcome.pop("result")
    outcome.update(status="failed", error_type="APIRequestError", events=[])
    call = outcome["calls"][-1]
    call.update(
        status="failed",
        prompt_version=core.PROMPT_VERSIONS["judge"],
        output_tokens=core.SHARED_OUTPUT_CAPS["judge"],
    )
    audit_path = Path(call["audit_path"])
    audit = json.loads(audit_path.read_bytes())
    audit.update(
        status="failed",
        error_type="APIRequestError",
        prompt_version=call["prompt_version"],
        output_tokens=call["output_tokens"],
    )
    audit["request"]["max_tokens"] = core.SHARED_OUTPUT_CAPS["judge"]
    audit["response"]["choices"] = [{"finish_reason": "length"}]
    dump(audit_path, audit)
    if failed_arm == core.ARMS[0]:
        for unstarted_call in row["arms"][core.ARMS[1]]["calls"]:
            Path(unstarted_call["audit_path"]).unlink()
        row["arms"][core.ARMS[1]] = {"status": "not_executed", "feedback": None, "calls": []}
    row["complete_pair"] = False
    directory = root / "questions" / f"{row['offset']:04d}"
    for arm, value in row["arms"].items():
        dump(directory / f"{arm}_execution.json", value)
    dump(directory / "prediction_report.json", row)
    dump(root / "predictions.json", reports)
    seal = json.loads((root / "predictions_frozen.json").read_bytes())
    seal.update(
        question_ids=[row["question_id"] for row in reports],
        reports_sha256_before_scoring=core.fingerprint(reports),
    )
    dump(root / "predictions_frozen.json", seal)
    calls = [c for row in reports for arm in core.ARMS for c in row["arms"][arm]["calls"]]
    dump(
        root / "final_budget.json",
        {
            "calls": calls,
            "api_requests": len(calls),
            "reserved_cny": sum(c["reserved_cny"] for c in calls),
            "block_reason": "transport_failure",
        },
    )
    (root / "events.jsonl").write_text('{"kind":"exit","status":"failed"}\n')
    return reports


def save_launch(root, launch):
    dump(root / "launch_plan.json", launch)
    claim_path = root.parent / f"{root.name}.claim.json"
    claim = json.loads(claim_path.read_bytes())
    claim["plan_sha256"] = core.fingerprint(launch)
    dump(claim_path, claim)


def synthetic_recovery(tmp_path, monkeypatch):
    parent, plan, questions = frozen_collection(tmp_path, monkeypatch, count=3, total=3)
    parent_rows = mark_closed_length_failure(parent, attempted=2)
    child, _, _ = frozen_collection(tmp_path, monkeypatch, start=2, count=1, total=3)
    launch = json.loads((child / "launch_plan.json").read_bytes())
    launch["continuation_of"] = parent.name
    launch["continuation_proof"] = {
        "schema_version": "growrag-memory-source-unstarted-continuation-v1",
        "prior_run_id": parent.name,
        "parent_run_id": parent.name,
        "question_ids": launch["question_ids"],
        "continued_question_ids": launch["question_ids"],
        "released_question_ids": plan["roles"]["source"][len(parent_rows) :],
        "prior_record_sha256": {
            name: core._sha(parent / name)
            for name in (
                "launch_plan.json",
                "predictions.json",
                "predictions_frozen.json",
                "final_budget.json",
                "events.jsonl",
            )
        },
    }
    save_launch(child, launch)
    return parent, child, plan, questions


def test_failed_prefix_explicit_unstarted_continuation_is_not_retry(tmp_path, monkeypatch):
    _, _, plan, questions = synthetic_recovery(tmp_path, monkeypatch)
    rows, hashes = core.verify_collection(tmp_path, plan, questions)
    assert [row["offset"] for row in rows] == [0, 1, 2]
    assert [row["complete_pair"] for row in rows] == [True, False, True]
    assert len(hashes) == 3
    assert sum(len(row["arms"][arm]["calls"]) for row in rows for arm in core.ARMS) == 6


@pytest.mark.parametrize(
    "mutation", ["missing", "parent_hash", "replay_id", "unused_api", "unused_directory"]
)
def test_continuation_proof_tampering_rejected(tmp_path, monkeypatch, mutation):
    parent, child, plan, questions = synthetic_recovery(tmp_path, monkeypatch)
    launch = json.loads((child / "launch_plan.json").read_bytes())
    if mutation == "missing":
        launch.pop("continuation_proof")
    elif mutation == "parent_hash":
        launch["continuation_proof"]["prior_record_sha256"]["final_budget.json"] = "wrong"
    elif mutation == "replay_id":
        launch["continuation_proof"]["continued_question_ids"] = [questions[1].question_id]
    elif mutation == "unused_api":
        dump(parent / "api_audit/unowned.json", {"trace_id": "unowned"})
    else:
        (parent / "questions/0002").mkdir()
    save_launch(child, launch)
    with pytest.raises(ValueError):
        core.verify_collection(tmp_path, plan, questions)


@pytest.mark.parametrize("mutation", ["audit_only", "runtime", "other_src", "declared_signature"])
def test_execution_signature_keeps_runtime_but_allows_audit_revision(
    tmp_path, monkeypatch, mutation
):
    _, child, plan, questions = synthetic_recovery(tmp_path, monkeypatch)
    snapshot = json.loads((child / "source_snapshot.json").read_bytes())
    original = core.execution_signature(snapshot)
    name = "src/synthetic.py" if mutation == "other_src" else core.CORE_PATH
    if mutation == "audit_only":
        snapshot["files"][name]["text"] += "\ndef offline_audit_addition():\n    pass\n"
    elif mutation == "runtime":
        snapshot["files"][name]["text"] = snapshot["files"][name]["text"].replace(
            "only 1-16 frozen source questions", "changed runtime source questions"
        )
    elif mutation == "other_src":
        snapshot["files"][name]["text"] = "print('changed method')\n"
    snapshot["files"][name]["sha256"] = hashlib.sha256(
        snapshot["files"][name]["text"].encode()
    ).hexdigest()
    snapshot["sha256"] = core.fingerprint(snapshot["files"])
    dump(child / "source_snapshot.json", snapshot)
    launch = json.loads((child / "launch_plan.json").read_bytes())
    launch["source_sha256"] = snapshot["sha256"]
    launch["execution_signature"] = (
        "wrong" if mutation == "declared_signature" else core.execution_signature(snapshot)
    )
    save_launch(child, launch)
    if mutation == "audit_only":
        assert core.execution_signature(snapshot) == original
        assert len(core.verify_collection(tmp_path, plan, questions)[0]) == 3
    else:
        with pytest.raises(ValueError):
            core.verify_collection(tmp_path, plan, questions)


@pytest.mark.parametrize(
    "mutation", ["finish", "http", "call_status", "output_cap", "block", "exit"]
)
def test_only_closed_audited_length_failure_is_admitted(tmp_path, monkeypatch, mutation):
    root, plan, questions = frozen_collection(tmp_path, monkeypatch)
    reports = mark_closed_length_failure(root)
    call = reports[-1]["arms"][core.ARMS[1]]["calls"][-1]
    audit_path = Path(call["audit_path"])
    audit = json.loads(audit_path.read_bytes())
    if mutation == "finish":
        audit["response"]["choices"][0]["finish_reason"] = "stop"
    elif mutation == "http":
        audit["http_status"] = 500
    elif mutation == "call_status":
        audit["status"] = "completed"
    elif mutation == "output_cap":
        audit["request"]["max_tokens"] += 1
    elif mutation == "block":
        ledger = json.loads((root / "final_budget.json").read_bytes())
        ledger["block_reason"] = None
        dump(root / "final_budget.json", ledger)
    else:
        (root / "events.jsonl").write_text('{"kind":"exit","status":"completed"}\n')
    dump(audit_path, audit)
    with pytest.raises(ValueError):
        core.verify_collection(tmp_path, plan, questions)


def test_not_executed_arm_must_own_no_call(tmp_path, monkeypatch):
    root, plan, questions = frozen_collection(tmp_path, monkeypatch)
    rows = mark_closed_length_failure(root, failed_arm=core.ARMS[0])
    verified, _ = core.verify_collection(tmp_path, plan, questions)
    assert verified[-1]["arms"][core.ARMS[1]]["calls"] == []
    rows[-1]["arms"][core.ARMS[1]]["calls"] = [deepcopy(rows[0]["arms"][core.ARMS[1]]["calls"][0])]
    dump(root / "predictions.json", rows)
    dump(root / "questions/0001/prediction_report.json", rows[-1])
    seal = json.loads((root / "predictions_frozen.json").read_bytes())
    seal["reports_sha256_before_scoring"] = core.fingerprint(rows)
    dump(root / "predictions_frozen.json", seal)
    with pytest.raises(ValueError, match="own no calls"):
        core.verify_collection(tmp_path, plan, questions)


def test_score_only_completed_sources_after_all_statuses_sealed(tmp_path, monkeypatch):
    root, plan, questions = frozen_collection(tmp_path, monkeypatch)
    mark_closed_length_failure(root)
    output = tmp_path / "scored"
    monkeypatch.setattr(core, "source_plan", lambda _: plan)

    def project(_raw, selected):
        seal = json.loads((output / "collection_frozen.json").read_bytes())
        assert seal["question_ids"] == plan["roles"]["source"]
        assert seal["failed_question_ids"] == [questions[1].question_id]
        assert [q.question_id for q in selected] == [questions[0].question_id]
        return {questions[0].question_id: SimpleNamespace(answers=("synthetic gold",))}

    monkeypatch.setattr(core, "project_source_gold", project)
    monkeypatch.setattr(
        core, "score_result", lambda *args, **kwargs: {"answer_em": 1, "answer_f1": 1}
    )
    rows = core.score_collection(
        tmp_path,
        Path("manifest"),
        SimpleNamespace(questions=questions, index=None),
        Path("raw"),
        output,
    )
    assert rows[0]["scoring_status"] == "completed"
    assert rows[1]["scoring_status"] == "not_scored_incomplete_pair"
    assert "offline_gold_answers" not in rows[1]
    assert all(rows[1]["arms"][arm]["feedback"] is None for arm in core.ARMS)
