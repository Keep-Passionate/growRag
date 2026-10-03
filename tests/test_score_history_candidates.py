"""Synthetic-only contracts: no real gold, question, credential, model or index."""

import copy
import hashlib
import json
from dataclasses import asdict

import pytest

from growrag.experiments import prior_budget
from growrag.experiments import score_history_candidates as scoring
from growrag.experiments.history_context import prepare_history_context
from growrag.experiments.operator_model_v3 import planner_prompt
from growrag.experiments.protocol import Evidence, RuntimeQuestion
from growrag.experiments.shared_hotpot_dev import document_id
from growrag.history_library import FrozenHistoryLibrary, HistoryCard, HistoryRecord
from growrag.macro_operators import GapField, GoalContract, OperatorSpec, QueryStep
from growrag.operator_loop import ActionProposal


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    """Full 50×5 fixture with real-format synthetic transport/ledger/episode seals."""
    root = tmp_path
    source = root / "src/example.py"
    source.parent.mkdir()
    source.write_text("# synthetic frozen method\n", encoding="utf-8")
    ids = [f"{n:024x}" for n in range(50)]
    questions = tuple(
        RuntimeQuestion(qid, f"Synthetic entity question {i}?", "synthetic")
        for i, qid in enumerate(ids)
    )
    parent = root / scoring.BUNDLE
    parent.mkdir(parents=True)
    runtime = parent / "runtime_questions.jsonl"
    runtime.write_text("\n".join(json.dumps(asdict(q)) for q in questions), encoding="utf-8")
    corpus = root / "data/synthetic/corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("{}\n", encoding="utf-8")
    template = OperatorSpec(
        "SYN_TEMPLATE",
        "1",
        ("lookup",),
        (GapField("entity"),),
        (QueryStep("search", "question {entity}"),),
    )
    library = FrozenHistoryLibrary(
        "synthetic",
        ("source",),
        tuple(
            HistoryRecord(
                HistoryCard(
                    f"RULE_{i}",
                    "1",
                    f"Question rule {i}",
                    "Preserve synthetic question",
                    "Preserve entities",
                ),
                "reference",
                "synthetic",
                "a" * 64,
                status="published",
            )
            for i in range(9)
        )
        + (
            HistoryRecord(
                HistoryCard(
                    "OP_TEMPLATE",
                    "1",
                    "Template",
                    "Description",
                    "Use template",
                    operator_spec=template,
                ),
                "learned",
                "synthetic",
                "a" * 64,
                source_qids=("source",),
                status="published",
            ),
        ),
    )
    library_path = root / "runs/synthetic_library/combined.json"
    library_path.parent.mkdir(parents=True)
    library_path.write_text(library.to_json(), encoding="utf-8")
    manifest = {
        "count": 50,
        "question_ids": ids,
        "official_split": "train",
        "runtime_artifact": {"path": runtime.name, "sha256": scoring._sha(runtime)},
        "library": {
            "path": library_path.relative_to(root).as_posix(),
            "sha256": scoring._sha(library_path),
            "fingerprint": library.fingerprint,
        },
    }
    for key in ("source_manifest", "background_manifest"):
        path = root / f"data/synthetic/{key}.json"
        write(path, {"synthetic": True})
        manifest[key] = {"path": path.relative_to(root).as_posix(), "sha256": scoring._sha(path)}
    write(parent / "manifest.json", manifest)
    (parent / "manifest.sha256").write_text(
        scoring._sha(parent / "manifest.json"), encoding="utf-8"
    )
    monkeypatch.setattr(scoring.study, "BUNDLE_SHA256", scoring._sha(parent / "manifest.json"))
    monkeypatch.setattr(
        scoring, "load_bundle", lambda *args, **kwargs: (copy.deepcopy(manifest), questions, corpus)
    )
    snapshot = scoring.source_snapshot(root)
    probe = root / scoring.PROBE
    write(probe, {"status": "completed", "source_sha256": snapshot["sha256"]})
    write(probe.parent.with_name(probe.parent.name + ".claim.json"), {"synthetic": True})
    monkeypatch.setattr(scoring.study, "verify_probe", lambda *args: scoring._sha(probe))
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
        for qid in ids
    }
    directories, prior = [], {}

    def request(directory, trace, version, messages, value):
        payload = {
            "model": scoring.study.CONFIGURATION["model"],
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
            "returned_model": payload["model"],
            "audit_path": str(path),
            "reserved_cny": 0.001,
            "estimated_actual_cny": 0.000036,
        }
        response = {
            "model": payload["model"],
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

    for start, count in scoring.study.BATCHES:
        run_id = scoring.study.run_id(start, count)
        directory = root / "runs" / run_id
        directory.mkdir(parents=True)
        directories.append(directory)
        plan = {
            **scoring.study.CONFIGURATION,
            "protocol": scoring.study.PROTOCOL,
            "phase": "candidate_development",
            "run_id": run_id,
            "start": start,
            "count": count,
            "methods": list(scoring.study.METHODS),
            "question_ids": ids[start : start + count],
            "bundle_sha256": scoring.study.BUNDLE_SHA256,
            "library_sha256": manifest["library"]["sha256"],
            "library_fingerprint": library.fingerprint,
            "source_sha256": snapshot["sha256"],
            "gold_loaded": False,
            "memory_updates": False,
            "official_split": "train",
            "probe_summary_sha256": scoring._sha(probe),
            "prior_batch_summary_sha256": copy.deepcopy(prior),
        }
        write(directory / "launch_plan.json", plan)
        write(
            directory.parent / (run_id + ".claim.json"),
            {**plan, "plan_sha256": scoring.fingerprint(plan)},
        )
        write(directory / "source_snapshot.json", snapshot)
        events, calls_all, hashes, terminal = [], [], {}, []
        for question in questions[start : start + count]:
            qid = question.question_id
            terminal.append({"question_id": qid, "methods": {}})
            for method in scoring.study.METHODS:
                prefix = f"{run_id}/{qid}/{method}"
                calls, proposals = [], []
                searches = [
                    {
                        "query": question.text,
                        "step": 0,
                        "evidence_ids": [doc_id],
                        "new_evidence_ids": [doc_id],
                        "elapsed_seconds": 0.01,
                    }
                ]
                executes_template = method == "history_body8" and qid == ids[0]
                if method in scoring.CANDIDATE_METHODS:
                    ranking = scoring._ranking(question, library, method)
                    context = prepare_history_context(
                        library,
                        question,
                        offered_ids=tuple(r["card_id"] for r in ranking),
                        representation="examples",
                        evidence=(evidence,),
                        previous_queries=(question.text,),
                        remaining_retrievals=2,
                    )
                    selected = {
                        "selected_card_id": "OP_TEMPLATE" if executes_template else None,
                        "reason": "condition_match" if executes_template else "no_suitable_card",
                    }
                    calls.append(
                        request(
                            directory,
                            prefix + "/history/1/select",
                            scoring.SELECT_PROMPT_VERSION,
                            context.messages(),
                            selected,
                        )
                    )
                    events += [
                        {
                            "kind": "history_candidates",
                            "question_id": qid,
                            "arm": method,
                            "decision": 1,
                            "ranking": list(ranking),
                            "payload": context.payload,
                            "context_fingerprint": context.fingerprint,
                        },
                        {
                            "kind": "history_selection",
                            "question_id": qid,
                            "arm": method,
                            **selected,
                        },
                    ]
                    if executes_template:
                        from growrag.experiments.history_context import _bounded_messages
                        from growrag.experiments.history_runtime import FILL_PROMPT
                        from growrag.history_library import card_view

                        card = next(
                            c for c in library.published_cards if c.card_id == "OP_TEMPLATE"
                        )
                        payload = {
                            "original_question": question.text,
                            "evidence": visible,
                            "previous_queries": [question.text],
                            "remaining_retrievals": 2,
                            "selected_card": card_view(card, "examples"),
                        }
                        messages, _, _ = scoring._fill_request(
                            _bounded_messages(FILL_PROMPT, payload)
                        )
                        value = {
                            "intent": "lookup",
                            "constraints": [],
                            "gap_entries": [{"name": "entity", "value": "Synthetic"}],
                            "bindings": [],
                        }
                        calls.append(
                            request(
                                directory,
                                prefix + "/history/1/fill",
                                scoring.FILL_VERSION,
                                messages,
                                value,
                            )
                        )
                        proposal = asdict(
                            ActionProposal(
                                GoalContract(question.text, "lookup"),
                                template,
                                {"entity": "Synthetic"},
                                origin="reuse",
                                reason="condition_match",
                            )
                        )
                        proposals.append(proposal)
                        events.append(
                            {
                                "kind": "history_execution",
                                "question_id": qid,
                                "arm": method,
                                "decision": 1,
                                "selected_card_id": card.card_id,
                                "selected_card_version": card.version,
                                "action_kind": card.action_kind,
                                "proposal": proposal,
                            }
                        )
                        searches.append(
                            {
                                "query": "question Synthetic",
                                "step": 1,
                                "evidence_ids": [doc_id],
                                "new_evidence_ids": [],
                                "elapsed_seconds": 0.02,
                            }
                        )
                    else:
                        proposals.append(
                            asdict(
                                ActionProposal(
                                    GoalContract(question.text, "lookup"),
                                    None,
                                    origin="stop",
                                    reason="no_suitable_card",
                                )
                            )
                        )
                elif method == "fresh":
                    payload = {
                        "mode": "fresh",
                        "original_question": question.text,
                        "evidence": visible,
                        "evidence_window_omitted_count": 0,
                        "previous_queries": [question.text],
                        "remaining_retrievals": 2,
                        "candidate_specs": [],
                        "candidate_shortlist_omitted_count": 0,
                    }
                    value = {
                        "reason": "evidence_sufficient",
                        "intent": "lookup",
                        "constraints": [],
                        "actions": [],
                    }
                    calls.append(
                        request(
                            directory,
                            prefix + "/plan/1",
                            scoring.PLANNER_VERSIONS["fresh"],
                            scoring._messages(planner_prompt("fresh"), payload),
                            value,
                        )
                    )
                    proposals.append(
                        asdict(
                            ActionProposal(
                                GoalContract(question.text, "lookup"),
                                None,
                                origin="stop",
                                reason="evidence_sufficient",
                            )
                        )
                    )
                answer = {
                    "answer": "wrong" if method == "base" else "answer",
                    "supported": True,
                    "evidence_ids": [doc_id],
                }
                payload = {
                    "original_question": question.text,
                    "evidence": visible,
                    "evidence_window_omitted_count": 0,
                }
                calls.append(
                    request(
                        directory,
                        prefix + "/reader",
                        scoring.READER_VERSION,
                        scoring._messages(scoring.READER_PROMPT, payload),
                        answer,
                    )
                )
                report = {
                    "status": "completed",
                    "question_id": qid,
                    "question": asdict(question),
                    "arm": scoring.study.underlying_arm(method),
                    "method": method,
                    "gold_loaded": False,
                    "memory_updated": False,
                    "calls": calls,
                    "episode": {
                        "question_id": qid,
                        "evidence": [asdict(evidence)],
                        "searches": searches,
                        "proposals": proposals,
                        "stop_reason": "no_new_evidence"
                        if executes_template
                        else "retrieval_budget"
                        if method == "base"
                        else "controller_stop_claim",
                        "rejected_error": None,
                    },
                    "reader": {
                        **answer,
                        "visible_evidence_ids": [doc_id],
                        "evidence_windows": [
                            {"evidence_id": row["evidence_id"], **row["window"]} for row in visible
                        ],
                    },
                }
                name = f"{qid}_{method}.json"
                write(directory / name, report)
                write(
                    directory / "raw_execution" / name,
                    {k: v for k, v in report.items() if k != "method"},
                )
                hashes[name] = scoring._sha(directory / name)
                terminal[-1]["methods"][method] = {"status": "completed", "path": name}
                calls_all.extend(calls)
            events.append({"kind": "question_completed", "question_id": qid})
        (directory / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events), encoding="utf-8"
        )
        totals = scoring._totals(calls_all)
        write(
            directory / "final_budget.json",
            {
                **totals,
                "calls": calls_all,
                "block_reason": None,
                "limits": {"input_per_million_cny": 0.2, "output_per_million_cny": 0.8},
            },
        )
        write(
            directory / "SUMMARY.json",
            {
                **totals,
                "protocol": scoring.study.PROTOCOL,
                "run_id": run_id,
                "status": "completed",
                "failure_type": None,
                "planned_questions": count,
                "completed_questions": count,
                "source_sha256": snapshot["sha256"],
                "gold_loaded": False,
                "memory_updated": False,
                "new_operator_induction": False,
                "prediction_sha256": hashes,
                "terminal": terminal,
            },
        )
        prior[run_id] = scoring._sha(directory / "SUMMARY.json")
    monkeypatch.setattr(prior_budget, "ROOT_LEDGERS", ())
    monkeypatch.setattr(
        scoring.study,
        "reviewed_history",
        lambda runs: prior_budget.reconcile_history(
            runs, reviewed_extra_ledgers=tuple(f"{d.name}/final_budget.json" for d in directories)
        ),
    )
    return root, gold, directories, manifest


def forbid_gold(*args, **kwargs):
    pytest.fail("gold loader must not run")


def first_report(sealed, method="history_body8"):
    root, _, directories, manifest = sealed
    qid = manifest["question_ids"][0]
    path = directories[0] / f"{qid}_{method}.json"
    return path, scoring._read(path)


def reseal_report(path, report):
    """Simulated malicious post-hoc change, including top/raw seals but not raw HTTP."""
    write(path, report)
    write(
        path.parent / "raw_execution" / path.name,
        {k: v for k, v in report.items() if k != "method"},
    )
    summary_path = path.parent / "SUMMARY.json"
    summary = scoring._read(summary_path)
    summary["prediction_sha256"][path.name] = scoring._sha(path)
    write(summary_path, summary)


def test_preflight_full250_without_gold(sealed, monkeypatch):
    root, _, _, _ = sealed
    monkeypatch.setattr(scoring, "_load_gold", forbid_gold)
    result = scoring.run(root)
    assert result["gold_loaded"] is False and result["questions"] == 50
    assert result["totals"]["api_requests"] == 451
    assert not (root / scoring.OUTPUT).exists()


@pytest.mark.parametrize(
    "defect", ["missing", "failed", "source", "claim", "model", "orphan", "raw_report"]
)
def test_collection_defects_fail_before_gold(sealed, monkeypatch, defect):
    root, _, directories, manifest = sealed
    monkeypatch.setattr(scoring, "_load_gold", forbid_gold)
    path, report = first_report(sealed)
    if defect == "missing":
        path.unlink()
    elif defect == "failed":
        report["status"] = "failed"
        reseal_report(path, report)
    elif defect == "source":
        (root / "src/example.py").write_text("# drift\n", encoding="utf-8")
    elif defect == "claim":
        write(root / "runs/renamed.claim.json", {"question_ids": [manifest["question_ids"][0]]})
    elif defect == "raw_report":
        raw = path.parent / "raw_execution" / path.name
        value = scoring._read(raw)
        value["arm"] = "fresh"
        write(raw, value)
    else:
        audit = (
            path.parent
            / "api_audit"
            / (hashlib.sha256(report["calls"][-1]["trace_id"].encode()).hexdigest() + ".json")
        )
        value = scoring._read(audit)
        if defect == "model":
            value["response"]["model"] = "different"
            write(audit, value)
        else:
            write(path.parent / "api_audit/orphan.json", value)
    with pytest.raises((ValueError, FileNotFoundError)):
        scoring.run(root, score=True)


@pytest.mark.parametrize(
    "defect", ["reader", "proposal", "query", "candidate_policy", "unused_response"]
)
def test_semantic_and_raw_transport_binding(sealed, monkeypatch, defect):
    root, _, directories, _ = sealed
    monkeypatch.setattr(scoring, "_load_gold", forbid_gold)
    path, report = first_report(sealed)
    if defect == "reader":
        report["reader"]["answer"] = "tampered"
    elif defect == "proposal":
        report["episode"]["proposals"][0]["reason"] = "tampered"
    elif defect == "query":
        report["episode"]["searches"][0]["query"] = "tampered"
    elif defect == "unused_response":
        report["calls"].insert(0, copy.deepcopy(report["calls"][0]))
    else:
        events_path = path.parent / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        event = next(
            e
            for e in events
            if e.get("kind") == "history_candidates" and e["arm"] == "history_body8"
        )
        event["ranking"].reverse()
        events_path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    reseal_report(path, report)
    with pytest.raises(ValueError):
        scoring.run(root, score=True)


def test_score_first_reader_normalization_preserves_raw_and_all_cost(sealed, monkeypatch):
    root, gold, directories, _ = sealed
    before = {str(p): scoring._sha(p) for d in directories for p in d.rglob("*") if p.is_file()}
    monkeypatch.setattr(scoring, "_load_gold", lambda *args: (gold, {}))
    result = scoring.run(root, score=True)
    summary = scoring._read(root / scoring.OUTPUT / "SUMMARY.json")
    rows = scoring._read(root / scoring.OUTPUT / "per_question.json")
    assert result["status"] == "scored" and summary["completed_methods"] == 250
    assert summary["methods"]["history_body8"]["raw"]["answer_em"]["mean"] == 1
    assert summary["methods"]["history_body8"][scoring.PRIMARY]["answer_em"]["mean"] == 0
    assert rows[0]["canonical_reader_method"] == {m: "base" for m in scoring.study.METHODS}
    assert summary["totals"]["api_requests"] == 451
    assert summary["methods"]["history_body8"]["api_requests"] == 101
    assert summary["methods"]["history_body8"]["actually_searched_learned_card_questions"] == 1
    assert rows[0]["usage"]["history_body8"]["executed_cards"][0]["queries"] == [
        "question Synthetic"
    ]
    assert summary["history_vs"]["history_body8"]["base"]["raw"]["repaired"] == 50
    assert summary["history_vs"]["history_body8"]["base"][scoring.PRIMARY]["repaired"] == 0
    assert before == {
        str(p): scoring._sha(p) for d in directories for p in d.rglob("*") if p.is_file()
    }
    assert (root / scoring.OUTPUT / "feedback_frozen.json").is_file()
    with pytest.raises(FileExistsError):
        scoring.run(root, score=True)


def test_primary_control_is_per_question_not_global_and_oracle_only_history(sealed):
    root, gold, _, _ = sealed
    checked = scoring.collect(root)
    records = checked["records"][:2]
    # Distinct first same-input group starts at FRESH, regardless of raw scores.
    records[0]["usage"]["fresh"]["reader_input_sha256"] = "other"
    records[0]["usage"]["history_body8"]["reader_input_sha256"] = "other"
    records[0]["methods"]["fresh"]["reader"]["answer"] = "wrong"
    records[0]["methods"]["history_body8"]["reader"]["answer"] = "answer"
    records[1]["methods"]["base"]["reader"]["answer"] = "answer"
    rows, summary = scoring.summarize(records, gold)
    assert rows[0]["canonical_reader_method"]["history_body8"] == "fresh"
    assert rows[0]["controlled_reader_scores"]["history_body8"]["answer_em"] == 0
    assert rows[1]["controlled_reader_scores"]["history_body8"]["answer_em"] == 1
    assert rows[0]["executed_policy_oracle"][scoring.PRIMARY]["answer_em"] == 0
    assert "only the three executed" in summary["oracle_notice"]


def test_bad_gold_stays_unknown_with_valid_denominator(sealed):
    root, gold, _, manifest = sealed
    records = scoring.collect(root)["records"]
    gold[manifest["question_ids"][0]]["supporting_facts"] = {"title": [], "sent_id": []}
    rows, summary = scoring.summarize(records, gold)
    assert rows[0]["raw_scores"]["base"]["answer_em"] is None
    assert summary["methods"]["base"][scoring.PRIMARY]["answer_em"]["valid_n"] == 49


def test_gold_loader_exact_train50_projection_and_shard_pins(sealed):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    root, gold, _, manifest = sealed
    parent = root / "data/synthetic_train"
    parent.mkdir()
    shard = parent / "train.parquet"
    rows = list(gold.values()) + [
        {**list(gold.values())[0], "id": "source", "answer": "never read"}
    ]
    pq.write_table(pa.Table.from_pylist(rows), shard)
    provenance = parent / "mirror_provenance.json"
    write(
        provenance,
        {
            "official_split": "train",
            "shards": [{"file": shard.name, "sha256": scoring._sha(shard)}],
        },
    )
    manifest["source_provenance"] = {
        "path": provenance.relative_to(root).as_posix(),
        "sha256": scoring._sha(provenance),
    }
    manifest["parquet_inputs"] = [
        {"path": shard.relative_to(root).as_posix(), "sha256": scoring._sha(shard)}
    ]
    projected, inputs = scoring._load_gold(root, manifest, manifest["question_ids"])
    assert set(projected) == set(manifest["question_ids"]) and "source" not in projected
    assert set(inputs) == {str(provenance), str(shard)}
    with pytest.raises(ValueError, match="fixed train50"):
        scoring._load_gold(root, manifest, manifest["question_ids"][:5])
    manifest["parquet_inputs"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="shard/provenance/manifest"):
        scoring._load_gold(root, manifest, manifest["question_ids"])


def test_fill_output_is_bound_to_compiled_query(sealed, monkeypatch):
    root, _, _, _ = sealed
    monkeypatch.setattr(scoring, "_load_gold", forbid_gold)
    path, report = first_report(sealed)
    call = next(c for c in report["calls"] if c["prompt_version"] == scoring.FILL_VERSION)
    audit_path = (
        path.parent
        / "api_audit"
        / (hashlib.sha256(call["trace_id"].encode()).hexdigest() + ".json")
    )
    audit = scoring._read(audit_path)
    value = json.loads(audit["response"]["choices"][0]["message"]["content"])
    value["gap_entries"][0]["value"] = "Wrong entity"
    audit["response"]["choices"][0]["message"]["content"] = json.dumps(value)
    write(audit_path, audit)
    with pytest.raises(ValueError, match="compiled query"):
        scoring.run(root, score=True)


def test_contract_replay_rejects_unused_recorded_response(sealed):
    root, _, _, manifest = sealed
    path, report = first_report(sealed, "base")
    library = FrozenHistoryLibrary.from_json(
        (root / manifest["library"]["path"]).read_text(encoding="utf-8")
    )
    call = report["calls"][-1]
    audit = scoring._read(
        path.parent
        / "api_audit"
        / (hashlib.sha256(call["trace_id"].encode()).hexdigest() + ".json")
    )
    content = audit["response"]["choices"][0]["message"]["content"]
    captured = [(call, audit["request"], content), (call, audit["request"], content)]
    prefix = call["trace_id"].removesuffix("reader")
    with pytest.raises(ValueError, match="unused responses"):
        scoring._contract_replay(report, captured, prefix, library)
