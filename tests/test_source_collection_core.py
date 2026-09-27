"""Synthetic-only source collection tests; no live API and no real dataset labels."""

import hashlib
import json
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


def frozen_collection(tmp_path, monkeypatch):
    """Entire synthetic two-question collection; no real data or model outputs."""
    monkeypatch.setattr(core, "SOURCE_COUNT", 2)
    ids = ["a" * 24, "b" * 24]
    questions = [synthetic_question(qid) for qid in ids]
    plan = {"roles": {"source": ids}}
    root = tmp_path / core.batch_identity(0, 2)
    root.mkdir()
    snapshot = {
        "files": {
            "src/synthetic.py": {"text": "pass\n", "sha256": hashlib.sha256(b"pass\n").hexdigest()}
        }
    }
    snapshot["sha256"] = core.fingerprint(snapshot["files"])
    launch = {
        "protocol": core.PROTOCOL,
        "manifest_sha256": core.PLAN_SHA,
        "start": 0,
        "count": 2,
        "run_id": root.name,
        "question_ids": ids,
        "max_calls": 22,
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
            "question_ids": ids,
            "plan_sha256": core.fingerprint(launch),
        },
    )
    dump(root / "source_snapshot.json", snapshot)
    (root / "events.jsonl").write_text('{"kind":"exit","status":"completed"}\n')
    (root / "api_audit").mkdir()
    reports, calls = [], []
    for offset, question in enumerate(questions):
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
            "question_ids": ids,
            "reports_sha256_before_scoring": core.fingerprint(reports),
        },
    )
    dump(root / "final_budget.json", {"calls": calls, "api_requests": 4, "reserved_cny": 0.04})
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
