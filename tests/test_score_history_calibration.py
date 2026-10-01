"""All inputs are synthetic; no real question, gold file, credential or API is opened."""

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from growrag.experiments import prior_budget
from growrag.experiments import score_history_calibration as scoring
from growrag.experiments.history_context import SELECT_PROMPT, _bounded_messages
from growrag.experiments.protocol import Evidence
from growrag.experiments.shared_hotpot_dev import document_id
from growrag.history_library import FrozenHistoryLibrary, HistoryCard, HistoryRecord, card_view


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def update(path, **fields):
    value = scoring._read(path)
    value.update(fields)
    write(path, value)


@pytest.fixture
def sealed(tmp_path, monkeypatch, request):
    version = getattr(request, "param", "v1")
    profile = scoring.profile_for(version)
    batches, configuration, protocol = (
        profile[k] for k in ("batches", "configuration", "protocol")
    )
    project = tmp_path
    src = project / "src/example.py"
    src.parent.mkdir()
    src.write_text("# synthetic frozen method\n", encoding="utf-8")
    ids = [f"{number:024x}" for number in range(100)]
    questions = [
        {"question_id": qid, "text": f"Synthetic question {i}?", "dataset": "synthetic"}
        for i, qid in enumerate(ids)
    ]
    parent = project / scoring.MANIFEST
    parent.parent.mkdir(parents=True)
    runtime = parent.parent / "calibration_runtime_questions.jsonl"
    runtime.write_text("\n".join(json.dumps(q) for q in questions), encoding="utf-8")
    corpus = parent.parent / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    manifest = {
        "roles": {"source": ["source"], "calibration": ids, "evaluation": ["eval"]},
        "official_splits": {"calibration": "train"},
        "artifacts": {},
        "input_files": [],
    }
    for path, count in ((runtime, 100), (corpus, 1)):
        manifest["artifacts"][path.name] = {
            "contains_gold": False,
            "runtime_safe": True,
            "rows": count,
            "sha256": scoring._sha(path),
        }
    write(parent, manifest)
    card = HistoryCard("RULE", "1", "Rule", "Description", "Preserve question")
    library = FrozenHistoryLibrary(
        "synthetic",
        ("source",),
        (HistoryRecord(card, "reference", "synthetic", "a" * 64, status="published"),),
    )
    lib_path = project / scoring.LIBRARY
    lib_path.parent.mkdir(parents=True)
    lib_path.write_text(library.to_json(), encoding="utf-8")
    monkeypatch.setattr(scoring, "MANIFEST_SHA256", scoring._sha(parent))
    monkeypatch.setattr(scoring, "LIBRARY_SHA256", scoring._sha(lib_path))
    monkeypatch.setattr(scoring, "LIBRARY_FINGERPRINT", library.fingerprint)
    snapshot = scoring.source_snapshot(project)
    doc_id = document_id("Synthetic", ["Synthetic answer."])
    evidence = Evidence(doc_id, "Synthetic", 0, "Synthetic answer.")
    visible = scoring.visible_evidence((evidence,))
    gold = {
        qid: {
            "id": qid,
            "answer": "answer",
            "context": {"title": ["Synthetic"], "sentences": [["Synthetic answer."]]},
            "supporting_facts": {"title": ["Synthetic"], "sent_id": [0]},
        }
        for qid in scoring._scope(manifest, version)
    }
    directories = []
    for batch_number, (start, count) in enumerate(batches):
        run_id = f"{scoring.PREFIX}{version}_{start:04d}_{start + count:04d}"
        directory = project / "runs" / run_id
        directory.mkdir(parents=True)
        directories.append(directory)
        plan = {
            **configuration,
            "protocol": protocol,
            "phase": "calibration",
            "run_id": run_id,
            "start": start,
            "count": count,
            "arms": list(scoring.ARMS),
            "question_ids": ids[start : start + count],
            "manifest_sha256": scoring.MANIFEST_SHA256,
            "library_sha256": scoring.LIBRARY_SHA256,
            "library_fingerprint": library.fingerprint,
            "reference_fingerprint": library.fingerprint,
            "official_split": "train",
            "source_sha256": snapshot["sha256"],
            "gold_loaded": False,
            "memory_updates": False,
        }
        if batch_number:
            plan["first_batch_summary_sha256"] = scoring._sha(directories[0] / "SUMMARY.json")
        write(directory / "launch_plan.json", plan)
        write(
            directory.parent / f"{run_id}.claim.json",
            {**plan, "plan_sha256": scoring.fingerprint(plan)},
        )
        write(directory / "source_snapshot.json", snapshot)
        events, all_calls, terminal, hashes = [], [], [], {}

        def request(trace, version, messages, value, directory=directory):
            payload = {
                "model": configuration["model"],
                "max_tokens": 2048,
                "stream": False,
                "enable_thinking": False,
                "temperature": 0,
                "top_p": 1,
                "response_format": {"type": "json_object"},
                "messages": messages,
            }
            path = directory / "api_audit" / (hashlib.sha256(trace.encode()).hexdigest() + ".json")
            call = {
                "trace_id": trace,
                "prompt_version": version,
                "status": "completed",
                "api_requests": 1,
                "input_tokens": 100,
                "output_tokens": 20,
                "returned_model": configuration["model"],
                "audit_path": str(path),
                "reserved_cny": 0.001,
                "estimated_actual_cny": 0.000036,
            }
            response = {
                "model": configuration["model"],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
            }
            write(
                path,
                {
                    **call,
                    "transport_source": "live_api",
                    "network_attempted": True,
                    "http_status": 200,
                    "retry_count": 0,
                    "finish_reason": "stop",
                    "response_redacted": False,
                    "request": payload,
                    "response": response,
                    "request_sha256": hashlib.sha256(
                        json.dumps(payload, ensure_ascii=False).encode()
                    ).hexdigest(),
                },
            )
            return call

        for question in questions[start : start + count]:
            qid = question["question_id"]
            terminal.append({"question_id": qid, "arms": {}})
            for arm in scoring.ARMS:
                prefix = f"{run_id}/{qid}/{arm}"
                calls = []
                if arm in {"history", "static_rules"}:
                    selected = {"selected_card_id": None, "reason": "no_suitable_card"}
                    calls.append(
                        request(
                            prefix + "/history/1/select",
                            scoring.SELECT_PROMPT_VERSION,
                            _bounded_messages(
                                SELECT_PROMPT, {"candidate_cards": [card_view(card, "examples")]}
                            ),
                            selected,
                        )
                    )
                    events.append(
                        {"kind": "history_selection", "question_id": qid, "arm": arm, **selected}
                    )
                answer = {
                    "answer": "answer" if arm == "history" else "wrong",
                    "supported": True,
                    "evidence_ids": [doc_id],
                }
                payload = {
                    "original_question": question["text"],
                    "evidence": visible,
                    "evidence_window_omitted_count": 0,
                }
                calls.append(
                    request(
                        prefix + "/reader",
                        scoring.READER_VERSION,
                        scoring._messages(scoring.READER_PROMPT, payload),
                        answer,
                    )
                )
                report = {
                    "status": "completed",
                    "question_id": qid,
                    "question": question,
                    "arm": arm,
                    "memory_updated": False,
                    "gold_loaded": False,
                    "calls": calls,
                    "episode": {
                        "question_id": qid,
                        "evidence": [asdict(evidence)],
                        "searches": [{"query": question["text"], "step": 0}],
                    },
                    "reader": {
                        **answer,
                        "visible_evidence_ids": [doc_id],
                        "evidence_windows": [
                            {"evidence_id": e["evidence_id"], **e["window"]} for e in visible
                        ],
                    },
                }
                name = f"{qid}_{arm}.json"
                write(directory / name, report)
                hashes[name] = scoring._sha(directory / name)
                terminal[-1]["arms"][arm] = {"status": "completed", "path": name}
                all_calls.extend(calls)
            events.append({"kind": "question_completed", "question_id": qid})
        (directory / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events), encoding="utf-8"
        )
        totals = scoring._totals(all_calls)
        write(
            directory / "final_budget.json",
            {
                **totals,
                "calls": all_calls,
                "block_reason": None,
                "limits": {"input_per_million_cny": 0.2, "output_per_million_cny": 0.8},
            },
        )
        write(
            directory / "SUMMARY.json",
            {
                **totals,
                "protocol": protocol,
                "run_id": run_id,
                "status": "completed",
                "failure_type": None,
                "planned_questions": count,
                "completed_questions": count,
                "source_sha256": snapshot["sha256"],
                "gold_loaded": False,
                "memory_updated": False,
                "new_operator_induction": False,
                "terminal": terminal,
                "prediction_sha256": hashes,
            },
        )
    monkeypatch.setattr(prior_budget, "ROOT_LEDGERS", ())
    # Run the actual no-network HTTP/ledger reconciler, without unrelated project history.
    monkeypatch.setattr(
        scoring,
        "reviewed_history",
        lambda runs: prior_budget.reconcile_history(
            runs, reviewed_extra_ledgers=tuple(f"{p.name}/final_budget.json" for p in directories)
        ),
    )
    return project, directories, ids, gold


def test_complete_preflight_does_not_open_gold(sealed, monkeypatch):
    project, _, _, _ = sealed
    monkeypatch.setattr(
        scoring, "_load_calibration_gold", lambda *a, **kw: pytest.fail("gold opened")
    )
    result = scoring.run(project)
    assert result["status"] == "preflight_passed"
    assert result["totals"]["api_requests"] == 150
    assert result["gold_loaded"] is False
    assert not (project / scoring.OUTPUT).exists()


@pytest.mark.parametrize(
    "failure", ["failed", "missing_arm", "source", "claim", "raw_model", "orphan"]
)
def test_fail_closed_before_gold(sealed, monkeypatch, failure):
    project, directories, ids, _ = sealed
    directory = directories[1]
    if failure == "failed":
        update(directory / "SUMMARY.json", status="failed", failure_type="ValueError")
    elif failure == "missing_arm":
        summary = scoring._read(directory / "SUMMARY.json")
        del summary["terminal"][0]["arms"]["history"]
        write(directory / "SUMMARY.json", summary)
    elif failure == "source":
        (project / "src/example.py").write_text("# changed\n", encoding="utf-8")
    elif failure == "claim":
        write(project / "runs/another.claim.json", {"question_ids": [ids[40]]})
    elif failure == "raw_model":
        path = next((directory / "api_audit").glob("*.json"))
        value = scoring._read(path)
        value["response"]["model"] = "wrong-model"
        write(path, value)
    else:
        value = scoring._read(next((directory / "api_audit").glob("*.json")))
        write(project / "runs/unaccounted/api_audit/extra.json", value)
    monkeypatch.setattr(
        scoring, "_load_calibration_gold", lambda *a, **kw: pytest.fail("gold opened")
    )
    with pytest.raises(ValueError):
        scoring.run(project, score=True)
    assert not (project / scoring.OUTPUT).exists()


def test_reader_response_and_visible_window_are_bound(sealed):
    project, directories, ids, _ = sealed
    path = directories[1] / f"{ids[45]}_history.json"
    report = scoring._read(path)
    report["reader"]["answer"] = "silently edited"
    write(path, report)
    summary = scoring._read(directories[1] / "SUMMARY.json")
    summary["prediction_sha256"][path.name] = scoring._sha(path)
    write(directories[1] / "SUMMARY.json", summary)
    with pytest.raises(ValueError, match="answer differs"):
        scoring.collect(project)


def test_scoring_separates_same_reader_input_changes_and_preserves_inputs(sealed, monkeypatch):
    project, directories, _, gold = sealed
    before = {
        str(path): scoring._sha(path)
        for directory in directories
        for path in directory.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(scoring, "_load_calibration_gold", lambda p, m, ids, **kw: (gold, {}))
    result = scoring.run(project, score=True)
    output = project / scoring.OUTPUT
    summary = scoring._read(output / "SUMMARY.json")
    assert result["status"] == "scored"
    assert summary["completed_arms"] == 100
    assert summary["history_vs"]["base"]["repaired"] == 25
    assert summary["history_vs"]["base"]["same_input_repaired"] == 25
    assert summary["arms"]["history"]["answer_em"] == {"mean": 1.0, "valid_n": 25}
    assert summary["arms"]["history"]["visible_support_recall"]["mean"] == 1.0
    assert summary["api_calls"] == 0 and summary["memory_updated"] is False
    assert (output / "per_question.md").read_text(encoding="utf-8").count("## ") == 25
    assert all(scoring._sha(Path(path)) == digest for path, digest in before.items())
    frozen = scoring._read(output / "feedback_frozen.json")
    assert all(scoring._sha(output / name) == digest for name, digest in frozen["files"].items())
    with pytest.raises(FileExistsError):
        scoring.run(project, score=True)


def test_selected_card_events_must_match_actual_selection(sealed):
    project, directories, ids, _ = sealed
    path = directories[1] / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    event = next(
        e for e in rows if e.get("question_id") == ids[45] and e.get("kind") == "history_selection"
    )
    event["selected_card_id"] = "RULE"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="selection/execution"):
        scoring.collect(project)


def test_actual_rule_execution_requires_fill_request_and_frozen_version(sealed):
    project, directories, ids, _ = sealed
    directory, qid = directories[1], ids[45]
    report = scoring._read(directory / f"{qid}_history.json")
    select = report["calls"][0]
    select_path = Path(select["audit_path"])
    selected = {"selected_card_id": "RULE", "reason": "condition_match"}
    audit = scoring._read(select_path)
    audit["response"]["choices"][0]["message"]["content"] = json.dumps(selected)
    write(select_path, audit)
    followup = copy.deepcopy(select)
    followup["trace_id"] = select["trace_id"].removesuffix("select") + "rewrite"
    followup["prompt_version"] = scoring.REWRITE_PROMPT_VERSION
    followup_path = (
        directory
        / "api_audit"
        / (hashlib.sha256(followup["trace_id"].encode()).hexdigest() + ".json")
    )
    followup["audit_path"] = str(followup_path)
    audit.update(followup)
    audit["response"]["choices"][0]["message"]["content"] = json.dumps({"query": "synthetic query"})
    write(followup_path, audit)
    report["calls"].insert(1, followup)
    library = FrozenHistoryLibrary.from_json((project / scoring.LIBRARY).read_text())
    plan = scoring._read(directory / "launch_plan.json")
    events = [
        {"question_id": qid, "arm": "history", "kind": "history_selection", **selected},
        {
            "question_id": qid,
            "arm": "history",
            "kind": "history_execution",
            "selected_card_id": "RULE",
            "selected_card_version": "1",
            "action_kind": library.published_cards[0].action_kind,
        },
    ]
    usage = scoring._audit_arm(report, directory, plan, events, library, {}, set())
    assert usage["executed_cards"][0]["source_kind"] == "reference"
    assert usage["executed_cards"][0]["search_executed"] is False
    assert usage["api_requests"] == 3
    events[-1]["selected_card_version"] = "2"
    with pytest.raises(ValueError, match="identity/version"):
        scoring._audit_arm(report, directory, plan, events, library, {}, set())
    events[-1]["selected_card_version"] = "1"
    report["calls"].pop(1)
    with pytest.raises(ValueError, match="no matching fill/rewrite"):
        scoring._audit_arm(report, directory, plan, events, library, {}, set())


def test_unknown_annotations_stay_null_in_summary(sealed):
    project, _, ids, gold = sealed
    checked = scoring.collect(project)
    gold = copy.deepcopy(gold)
    gold[ids[40]]["answer"] = ""
    rows, summary = scoring.summarize(checked["records"], gold)
    assert rows[0]["scores"]["history"]["answer_em"] is None
    assert summary["arms"]["history"]["answer_em"]["valid_n"] == 24


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_gold_projection_is_fixed_train25_and_shard_pinned(tmp_path, version):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    project = tmp_path
    ids = [f"q{i}" for i in range(100)]
    directory = project / "data/hotpotqa/official_train_v1_1"
    directory.mkdir(parents=True)
    shard = directory / "train-0.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": qid,
                    "answer": "synthetic",
                    "context": "unused",
                    "supporting_facts": "unused",
                }
                for qid in ids
            ]
        ),
        shard,
    )
    provenance = directory / "mirror_provenance.json"
    write(
        provenance,
        {
            "official_split": "train",
            "shards": [{"file": shard.name, "sha256": scoring._sha(shard)}],
        },
    )
    manifest = {
        "roles": {"calibration": ids},
        "input_files": [
            {"path": p.relative_to(project).as_posix(), "sha256": scoring._sha(p)}
            for p in (shard, provenance)
        ],
    }
    selected_ids = scoring._scope(manifest, version)
    rows, inputs = scoring._load_calibration_gold(project, manifest, selected_ids, version=version)
    assert list(rows) == selected_ids and len(inputs) == 2
    with pytest.raises(ValueError, match="exactly predeclared"):
        scoring._load_calibration_gold(project, manifest, ids[:25], version=version)
    if version == "v2":
        with pytest.raises(ValueError, match="exactly predeclared"):
            scoring._load_calibration_gold(project, manifest, ids[40:65], version=version)
    manifest["input_files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="train shard"):
        scoring._load_calibration_gold(project, manifest, selected_ids, version=version)


@pytest.mark.parametrize("sealed", ["v2"], indirect=True)
def test_v2_full_gate_and_separate_feedback(sealed, monkeypatch):
    project, _, ids, gold = sealed
    checked = scoring.collect(project, version="v2")
    assert checked["ids"] == ids[65:90]
    assert checked["protocol"] == "growrag-history-calibration-v2"

    def labels(p, manifest, selected, *, version):
        assert version == "v2" and selected == ids[65:90]
        return gold, {}

    monkeypatch.setattr(scoring, "_load_calibration_gold", labels)
    result = scoring.run(project, version="v2", score=True)
    output = project / "runs/history_base25_feedback_v2"
    assert result["output"] == str(output)
    assert scoring._read(output / "SUMMARY.json")["split"] == "official_train_calibration_65_90"
    assert not (project / scoring.OUTPUT).exists()
    # A v2 result cannot accidentally satisfy the unchanged v1 gate.
    with pytest.raises(FileNotFoundError):
        scoring.collect(project)


@pytest.mark.parametrize("sealed", ["v2"], indirect=True)
@pytest.mark.parametrize("failure", ["failed", "protocol"])
def test_v2_failure_or_wrong_profile_cannot_open_gold(sealed, monkeypatch, failure):
    project, directories, _, _ = sealed
    path = directories[-1] / ("SUMMARY.json" if failure == "failed" else "launch_plan.json")
    update(
        path, **({"status": "failed"} if failure == "failed" else {"protocol": scoring.PROTOCOL})
    )
    monkeypatch.setattr(
        scoring, "_load_calibration_gold", lambda *a, **kw: pytest.fail("gold opened")
    )
    with pytest.raises(ValueError):
        scoring.run(project, version="v2", score=True)
    assert not (project / "runs/history_base25_feedback_v2").exists()


@pytest.mark.parametrize("sealed", ["v2"], indirect=True)
def test_v2_audit_rejects_v1_fill_wire(sealed):
    project, directories, ids, _ = sealed
    directory, qid = directories[-1], ids[70]
    report = scoring._read(directory / f"{qid}_history.json")
    call = report["calls"][0]
    call["prompt_version"] = scoring.FILL_VERSION
    path = Path(call["audit_path"])
    audit = scoring._read(path)
    audit["prompt_version"] = scoring.FILL_VERSION
    audit["response"]["choices"][0]["message"]["content"] = json.dumps(
        {"intent": "lookup", "constraints": [], "gap_entries": [], "bindings": []}
    )
    write(path, audit)
    library = FrozenHistoryLibrary.from_json((project / scoring.LIBRARY).read_text())
    with pytest.raises(ValueError, match="old FILL version"):
        scoring._audit_arm(
            report,
            directory,
            scoring._read(directory / "launch_plan.json"),
            [],
            library,
            {},
            set(),
            version="v2",
        )


def test_unregistered_version_is_rejected_before_reading_files(tmp_path):
    with pytest.raises(ValueError):
        scoring.collect(tmp_path, version="v3")
