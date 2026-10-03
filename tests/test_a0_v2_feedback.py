"""Synthetic offline A0 feedback tests; no real cohort, gold files or API."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace

import pytest

from growrag.experiments import a0_runtime_v2 as runtime
from growrag.experiments import a0_v2_feedback as feedback
from growrag.experiments.history_context import SELECT_PROMPT_VERSION
from growrag.experiments.pre_pilot import write_json
from growrag.experiments.protocol import RuntimeQuestion
from growrag.history_library import FrozenHistoryLibrary

QUESTION = RuntimeQuestion("synthetic-feedback", "Where was Avery born?")
LIBRARY = FrozenHistoryLibrary("synthetic", (), ())
STOP = {"reason": "no_useful_query", "intent": "lookup", "constraints": [], "actions": []}


def jsonable(value):
    return json.loads(json.dumps(value, default=lambda x: asdict(x) if is_dataclass(x) else x))


class RecordedSynthetic:
    def __init__(self, answer="Stonebridge", bad_ref=False, *, supported=True, planner_output=None):
        self.answer, self.bad_ref = answer, bad_ref
        self.supported = supported
        self.planner_output = STOP if planner_output is None else deepcopy(planner_output)
        self.calls, self.captured = [], []
        self.block_reason = None
        self.config = SimpleNamespace(json_object_mode=True, json_schema_mode=False)

    def complete(self, messages, *, trace_id, prompt_version):
        if prompt_version == runtime.SHORT_READER_VERSION:
            value = {
                "candidate_answer": self.answer,
                "supported": self.supported,
                "evidence_ids": ["E99" if self.bad_ref else "E1"],
            }
        elif prompt_version == SELECT_PROMPT_VERSION:
            value = {"selected_card_id": None, "reason": "no_suitable_card"}
        else:
            value = self.planner_output
        content = json.dumps(value)
        name = hashlib.sha256(trace_id.encode()).hexdigest() + ".json"
        call = {
            "trace_id": trace_id,
            "prompt_version": prompt_version,
            "status": "completed",
            "api_requests": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "reserved_cny": 0.01,
            "estimated_actual_cny": 0.000006,
            "returned_model": "synthetic-model",
            "audit_path": name,
        }
        request = {
            "model": "synthetic-model",
            "max_tokens": 2048,
            "stream": False,
            "enable_thinking": False,
            "temperature": 0,
            "top_p": 1,
            "response_format": {"type": "json_object"},
            "messages": deepcopy(messages),
        }
        self.calls.append(call)
        self.captured.append((call, request, content))
        return SimpleNamespace(content=content)


def sample(
    tmp_path,
    method="base",
    *,
    bad_ref=False,
    answer="Stonebridge",
    run_id="synthetic",
    supported=True,
    planner_output=None,
):
    delegate = RecordedSynthetic(
        answer, bad_ref, supported=supported, planner_output=planner_output
    )
    events = []
    target = tmp_path / f"{method}.json"
    trace = f"{run_id}/{QUESTION.question_id}/{method}"

    def index(query, k):
        return (
            SimpleNamespace(
                doc_id="full-document-id", title="Avery", text="Avery was born in Stonebridge."
            ),
        )

    try:
        runtime.execute_arm(
            QUESTION,
            method,
            index,
            runtime.A0ContractClient(delegate),
            library=LIBRARY,
            trace=trace,
            log=events.append,
            target=target,
        )
    except ValueError:
        if not bad_ref:
            raise
    return (
        json.loads(target.read_text(encoding="utf-8")),
        delegate.captured,
        jsonable(events),
        trace + "/",
    )


@pytest.mark.parametrize("method", runtime.METHODS)
def test_actual_runtime_replay_all_methods_and_short_reference_mapping(tmp_path, method):
    report, captured, events, prefix = sample(tmp_path, method)
    value = feedback.replay_report(report, captured, events, LIBRARY, prefix)
    assert len(value) == 64
    assert report["reader"]["evidence_refs"] == ["E1"]
    assert report["reader"]["evidence_ids"] == ["full-document-id"]


@pytest.mark.parametrize("method", ["fresh_original", "fresh_focused"])
def test_nonempty_fresh_action_integer_wire_and_derived_intent_replay(tmp_path, method):
    wire = {
        "reason": "missing_evidence",
        "intent": "lookup",
        "constraints": ["Preserve the original goal"],
        "actions": [
            {
                "selected_operator": None,
                "operator": {
                    "operator_id": "YEAR_LOOKUP",
                    "version": "1",
                    "gap_schema": [
                        {"name": "year", "kind": "integer", "required": True},
                        {"name": "entity", "kind": "text", "required": True},
                    ],
                    "steps": [
                        {
                            "step_id": "search",
                            "template": "{entity} born {year}",
                            "when": [{"field": "year", "value": "1977"}],
                            "requires_bindings": [],
                        }
                    ],
                },
                "gap_entries": [
                    {"name": "year", "value": "1977"},
                    {"name": "entity", "value": "Avery"},
                ],
                "bindings": [],
            }
        ],
    }
    report, captured, events, prefix = sample(tmp_path, method, planner_output=wire)
    assert report["status"] == "completed"
    assert [search["query"] for search in report["episode"]["searches"]] == [
        QUESTION.text,
        "Avery born 1977",
    ]
    assert len(report["episode"]["proposals"]) == 1
    proposal = report["episode"]["proposals"][0]
    assert proposal["goal"]["intent"] == "lookup"
    assert proposal["spec"]["supported_intents"] == ["lookup"]
    assert proposal["gap"]["year"] == 1977 and type(proposal["gap"]["year"]) is int
    assert proposal["spec"]["steps"][0]["when"][0]["value"] == 1977
    expected_version = (
        runtime.FRESH_PLANNER_VERSION
        if method == "fresh_original"
        else runtime.FOCUSED_PLANNER_VERSION
    )
    assert captured[0][0]["prompt_version"] == expected_version
    assert json.loads(captured[0][2]) == wire
    assert "supported_intents" not in wire["actions"][0]["operator"]
    original = deepcopy((report, captured, events))
    digest = feedback.replay_report(report, captured, events, LIBRARY, prefix)
    assert digest == feedback.fingerprint(captured[-1][1]["messages"])
    assert (report, captured, events) == original
    assert feedback.totals(report["calls"])["api_requests"] == 2


@pytest.mark.parametrize(
    "tamper", ["messages", "query", "alias", "unused_response", "selected_event"]
)
def test_replay_rejects_saved_prompt_query_alias_or_attribution_tamper(tmp_path, tamper):
    report, captured, events, prefix = sample(tmp_path)
    if tamper == "messages":
        captured[0][1]["messages"][0]["content"] += " modified"
    elif tamper == "query":
        next(e for e in events if e.get("kind") == "operator_search")["search"]["query"] = (
            "different query"
        )
    elif tamper == "alias":
        report["reader"]["alias_to_evidence_id"]["E1"] = "other-document"
    elif tamper == "unused_response":
        captured.append(deepcopy(captured[-1]))
    else:
        events.append(
            {
                "kind": "history_selection",
                "selected_card_id": "invented",
                "reason": "condition_match",
            }
        )
    with pytest.raises(ValueError):
        feedback.replay_report(report, captured, events, LIBRARY, prefix)


def test_failed_local_reader_replays_without_repair_and_retains_cost(tmp_path):
    report, captured, events, prefix = sample(tmp_path, bad_ref=True)
    assert report["status"] == "failed" and report["reader"] is None
    feedback.replay_report(report, captured, events, LIBRARY, prefix)
    costs = feedback.totals(report["calls"])
    assert costs["api_requests"] == 1 and costs["estimated_actual_cny"] == pytest.approx(0.000006)


def test_unknown_failure_cost_is_not_zero_or_refunded():
    value = feedback.totals(
        [
            {"api_requests": 1, "reserved_cny": 0.05},
            {
                "api_requests": 1,
                "reserved_cny": 0.03,
                "input_tokens": 10,
                "output_tokens": 2,
                "estimated_actual_cny": 0.01,
            },
        ]
    )
    assert value["estimated_actual_cny"] is None and value["input_tokens"] is None
    assert value["known_estimated_actual_cny"] == 0.01
    assert value["reserved_cny"] == 0.08 and value["unknown_cost_requests"] == 1


def http_fixture(tmp_path, captured):
    call, request, content = captured[0]
    data = {
        "trace_id": call["trace_id"],
        "prompt_version": call["prompt_version"],
        "transport_source": "live_api",
        "network_attempted": True,
        "api_requests": 1,
        "retry_count": 0,
        "status": "completed",
        "http_status": 200,
        "finish_reason": "stop",
        "input_tokens": 10,
        "output_tokens": 5,
        "response_redacted": False,
        "request": request,
        "request_sha256": hashlib.sha256(
            json.dumps(request, ensure_ascii=False, allow_nan=False).encode()
        ).hexdigest(),
        "response": {
            "model": "synthetic-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        },
    }
    path = tmp_path / "api_audit" / call["audit_path"]
    path.parent.mkdir(exist_ok=True)
    write_json(path, data)
    return path, data


@pytest.mark.parametrize("tamper", [None, "model", "cost", "hash", "usage"])
def test_http_model_cost_hash_usage_contract(tmp_path, tamper):
    report, captured, _, prefix = sample(tmp_path)
    path, data = http_fixture(tmp_path, captured)
    if tamper == "model":
        data["response"]["model"] = "other"
    elif tamper == "cost":
        captured[0][0]["estimated_actual_cny"] = 0.01
    elif tamper == "hash":
        data["request_sha256"] = "a" * 64
    elif tamper == "usage":
        data["response"]["usage"]["prompt_tokens"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")

    def call():
        return feedback.audit_http(
            report["calls"][0] if tamper != "cost" else captured[0][0],
            tmp_path,
            {"model": "synthetic-model", "max_output_tokens": 2048},
            lambda p: p,
            set(),
            prefix,
        )

    if tamper:
        with pytest.raises(ValueError):
            call()
    else:
        assert call()[2] == captured[0][2]


def test_journal_checks_prefix_and_final_ledger(tmp_path):
    report, captured, _, _ = sample(tmp_path)
    call, request, _ = captured[0]
    ledger = {
        "calls": report["calls"],
        "api_requests": 1,
        "reserved_cny": 0.01,
        "estimated_actual_cny": 0.000006,
    }
    journal = tmp_path / "request_journal"
    journal.mkdir()
    write_json(
        journal / "0000_intent.json",
        {
            "trace_id": call["trace_id"],
            "prompt_version": call["prompt_version"],
            "request_fingerprint": feedback.fingerprint(request["messages"]),
            "status": "pending_no_automatic_retry",
            "potential_reserved_cny": 0.01,
            "prior_reserved_cny": 0,
        },
    )
    write_json(journal / "0000_after.json", ledger)
    feedback.audit_journal(tmp_path, ledger, captured, lambda p: p)
    wrong = deepcopy(ledger)
    wrong["calls"] = []
    (journal / "0000_after.json").write_text(json.dumps(wrong), encoding="utf-8")
    with pytest.raises(ValueError, match="prefix"):
        feedback.audit_journal(tmp_path, ledger, captured, lambda p: p)


def test_controlled_reader_first_method_not_best_answer_and_failed_costs_remain(tmp_path):
    item = {
        "question_id": QUESTION.question_id,
        "question": QUESTION.text,
        "methods": {},
        "usage": {},
        "states": {},
    }
    for method in runtime.METHODS:
        report, captured, events, prefix = sample(
            tmp_path, method, answer="wrong" if method == "base" else "Stonebridge"
        )
        digest = feedback.replay_report(report, captured, events, LIBRARY, prefix)
        item["methods"][method] = report
        item["usage"][method] = {"reader_input_sha256": digest}
        item["states"][method] = {"status": "completed"}
    failed = {
        "question_id": "failed",
        "question": "not labeled",
        "methods": {
            "base": {
                "status": "failed",
                "calls": [
                    {
                        "api_requests": 1,
                        "reserved_cny": 0.04,
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "estimated_actual_cny": 0.000006,
                    }
                ],
            }
        },
        "usage": {},
        "states": {
            m: {"status": "failed" if m == "base" else "not_attempted"} for m in runtime.METHODS
        },
    }
    gold = {
        QUESTION.question_id: {
            "answer": "Stonebridge",
            "context": {"title": ["Avery"], "sentences": [["Avery was born in Stonebridge."]]},
            "supporting_facts": {"title": ["Avery"], "sent_id": [0]},
        }
    }
    rows, summary = feedback.summarize([item, failed], gold)
    assert rows[0]["raw_scores"]["fresh_focused"]["answer_em"] == 1
    assert all(score["answer_em"] == 0 for score in rows[0]["controlled_reader_scores"].values())
    assert all(m == "base" for m in rows[0]["canonical_reader_method"].values())
    assert summary["methods"]["base"]["all_attempted_cost"]["api_requests"] == 2
    assert summary["methods"]["base"]["paired_only_cost"]["api_requests"] == 1
    assert summary["joint_complete_questions"] == 1


def test_unsupported_candidate_is_audit_only_never_scored(tmp_path):
    item = {
        "question_id": QUESTION.question_id,
        "question": QUESTION.text,
        "methods": {},
        "usage": {},
        "states": {},
    }
    for method in runtime.METHODS:
        report, _, _, _ = sample(tmp_path, method)
        report["reader"].update(candidate_answer="Stonebridge", supported=False, answer="")
        item["methods"][method] = report
        item["usage"][method] = {"reader_input_sha256": "a" * 64}
        item["states"][method] = {"status": "completed"}
    gold = {
        QUESTION.question_id: {
            "answer": "Stonebridge",
            "context": {"title": ["Avery"], "sentences": [["Avery was born in Stonebridge."]]},
            "supporting_facts": {"title": ["Avery"], "sent_id": [0]},
        }
    }
    rows, _ = feedback.summarize([item], gold)
    assert set(rows[0]["candidate_answers_audit_only"].values()) == {"Stonebridge"}
    assert set(rows[0]["answers"].values()) == {""}
    for label in ("raw_scores", "controlled_reader_scores"):
        assert all(s["answer_em"] == s["answer_f1"] == 0 for s in rows[0][label].values())


def test_controlled_reader_keeps_first_abstention_despite_later_supported_candidate(tmp_path):
    item = {
        "question_id": QUESTION.question_id,
        "question": QUESTION.text,
        "methods": {},
        "usage": {},
        "states": {},
    }
    for method in runtime.METHODS:
        report, captured, events, prefix = sample(tmp_path, method, supported=method != "base")
        digest = feedback.replay_report(report, captured, events, LIBRARY, prefix)
        assert report["reader"]["candidate_answer"] == "Stonebridge"
        assert report["reader"]["answer"] == ("" if method == "base" else "Stonebridge")
        item["methods"][method] = report
        item["usage"][method] = {"reader_input_sha256": digest}
        item["states"][method] = {"status": "completed"}
    assert len({usage["reader_input_sha256"] for usage in item["usage"].values()}) == 1
    original = deepcopy(item)
    gold = {
        QUESTION.question_id: {
            "answer": "Stonebridge",
            "context": {"title": ["Avery"], "sentences": [["Avery was born in Stonebridge."]]},
            "supporting_facts": {"title": ["Avery"], "sent_id": [0]},
        }
    }
    rows, summary = feedback.summarize([item], gold)
    assert rows[0]["raw_scores"]["base"]["answer_em"] == 0
    for method in runtime.METHODS[1:]:
        assert rows[0]["raw_scores"][method]["answer_em"] == 1
    assert all(score["answer_em"] == 0 for score in rows[0]["controlled_reader_scores"].values())
    assert all(score["answer_f1"] == 0 for score in rows[0]["controlled_reader_scores"].values())
    assert set(rows[0]["canonical_reader_method"].values()) == {"base"}
    assert set(rows[0]["candidate_answers_audit_only"].values()) == {"Stonebridge"}
    assert rows[0]["answers"] == {m: "" if m == "base" else "Stonebridge" for m in runtime.METHODS}
    assert summary["methods"]["fresh_original"]["controlled_reader"]["answer_em"]["mean"] == 0
    assert item == original


def test_failed_preflight_never_opens_gold(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("terminal tampered")

    monkeypatch.setattr(feedback, "collect", reject)
    monkeypatch.setattr(feedback, "load_gold", lambda *a: pytest.fail("gold gate escaped"))
    with pytest.raises(ValueError, match="terminal tampered"):
        feedback.run(
            tmp_path, score=True, expected_freeze_sha256="a" * 64, expected_terminal_sha256="b" * 64
        )


def test_default_preflight_and_zero_pairs_never_open_gold(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "collect", lambda *a, **k: {"paired_ids": [], "totals": {}})
    monkeypatch.setattr(feedback, "load_gold", lambda *a: pytest.fail("gold gate escaped"))
    assert (
        feedback.run(tmp_path, expected_freeze_sha256="a" * 64, expected_terminal_sha256="b" * 64)[
            "gold_loaded"
        ]
        is False
    )
    with pytest.raises(ValueError, match="no paired IDs"):
        feedback.run(
            tmp_path, score=True, expected_freeze_sha256="a" * 64, expected_terminal_sha256="b" * 64
        )


def test_external_seals_and_gold_projection_subset_required(tmp_path):
    with pytest.raises(ValueError, match="external"):
        feedback.collect(tmp_path, expected_freeze_sha256="bad", expected_terminal_sha256="a" * 64)
    with pytest.raises(ValueError, match="registered"):
        feedback.load_gold(
            tmp_path, {"official_split": "train", "question_ids": ["allowed"]}, ["hidden"]
        )


def test_zero_http_budget_refusal_is_auditable_and_uncharged(tmp_path):
    journal = tmp_path / "request_journal"
    journal.mkdir()
    ledger = {
        "calls": [],
        "api_requests": 0,
        "reserved_cny": 0,
        "estimated_actual_cny": 0,
        "block_reason": "estimated_budget_limit",
    }
    write_json(
        journal / "0000_intent.json",
        {
            "trace_id": "run/q/base/reader",
            "prompt_version": runtime.SHORT_READER_VERSION,
            "request_fingerprint": "a" * 64,
            "status": "pending_no_automatic_retry",
            "potential_reserved_cny": 0.01,
            "prior_reserved_cny": 0,
        },
    )
    write_json(journal / "0000_after.json", ledger)
    feedback.audit_journal(tmp_path, ledger, [], lambda p: p, refusal_prefixes=("run/q/base/",))
    with pytest.raises(ValueError, match="unaccounted"):
        feedback.audit_journal(tmp_path, ledger, [], lambda p: p)


@pytest.mark.parametrize("status", ["unknown_usage", "model_snapshot_mismatch"])
def test_completed_http_budget_validation_does_not_become_normal_replay(tmp_path, status):
    _, captured, _, prefix = sample(tmp_path)
    path, data = http_fixture(tmp_path, captured)
    call = captured[0][0]
    call["validation_status"] = status
    if status == "unknown_usage":
        call["input_tokens"] = data["input_tokens"] = None
        call["estimated_actual_cny"] = None
        data["response"]["usage"]["prompt_tokens"] = None
    else:
        call["returned_model"] = data["response"]["model"] = "unexpected-snapshot"
    path.write_text(json.dumps(data), encoding="utf-8")
    checked = feedback.audit_http(
        call,
        tmp_path,
        {"model": "synthetic-model", "max_output_tokens": 2048},
        lambda p: p,
        set(),
        prefix,
    )
    assert checked[2] is None


def synthetic_collection(tmp_path, monkeypatch):
    study = feedback.study
    monkeypatch.setattr(study, "COUNT", 1)
    monkeypatch.setattr(study, "BATCHES", ((0, 1),))
    monkeypatch.setitem(study.CONFIGURATION, "model", "synthetic-model")
    source_hash = feedback.fingerprint({})
    for relative in (
        study.FREEZE,
        study.PROTOCOL_NOTE,
        f"{study.BUNDLE}/manifest.json",
        f"{study.BUNDLE}/runtime_questions.jsonl",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    library_path = tmp_path / "library.json"
    library_path.write_text(LIBRARY.to_json(), encoding="utf-8")
    frozen = {
        "question_ids": [QUESTION.question_id],
        "source_sha256": source_hash,
        "bundle_sha256": "b" * 64,
        "library_sha256": feedback._sha(library_path),
    }
    manifest = {
        "question_ids": [QUESTION.question_id],
        "official_split": "train",
        "library": {
            "path": "library.json",
            "sha256": frozen["library_sha256"],
            "fingerprint": LIBRARY.fingerprint,
        },
    }
    monkeypatch.setattr(study, "load_freeze", lambda *a: (frozen, manifest, (QUESTION,), None))
    run_id = f"{study.PREFIX}v1_0000_0001"
    directory = tmp_path / "runs" / run_id
    directory.mkdir(parents=True)
    plan = {
        **study.CONFIGURATION,
        "protocol": study.PROTOCOL,
        "run_id": run_id,
        "phase": "a0_v2_development",
        "question_ids": [QUESTION.question_id],
        "methods": list(runtime.METHODS),
        "start": 0,
        "count": 1,
        "freeze_sha256": "a" * 64,
        "source_sha256": source_hash,
        "bundle_sha256": "b" * 64,
        "library_sha256": frozen["library_sha256"],
        "gold_loaded": False,
        "prior_reserved_cny": 0,
        "subcap_cny": 5,
    }
    write_json(directory / "launch_plan.json", plan)
    write_json(
        tmp_path / "runs" / f"{run_id}.claim.json",
        {**plan, "plan_sha256": feedback.fingerprint(plan)},
    )
    write_json(directory / "source_snapshot.json", {"files": {}, "sha256": source_hash})
    write_json(directory / "prior_budget.json", {"prior_reserved_cny": 0})
    all_calls, captures, events = [], [], []
    for method in runtime.METHODS:
        report, captured, local, _ = sample(directory, method, run_id=run_id)
        (directory / f"{method}.json").rename(directory / f"{QUESTION.question_id}_{method}.json")
        all_calls.extend(report["calls"])
        captures.extend(captured)
        events.extend(
            {"question_id": QUESTION.question_id, "arm": method, "utc": "synthetic", **e}
            for e in local
        )
        for capture in captured:
            http_fixture(directory, [capture])
    (directory / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8"
    )
    journal = directory / "request_journal"
    journal.mkdir()
    for number, (call, request, _) in enumerate(captures):
        prefix_calls = all_calls[: number + 1]
        values = feedback.totals(prefix_calls)
        ledger = {
            "calls": prefix_calls,
            **{k: values[k] for k in ("api_requests", "reserved_cny", "estimated_actual_cny")},
        }
        write_json(
            journal / f"{number:04d}_intent.json",
            {
                "trace_id": call["trace_id"],
                "prompt_version": call["prompt_version"],
                "request_fingerprint": feedback.fingerprint(request["messages"]),
                "status": "pending_no_automatic_retry",
                "potential_reserved_cny": 0.01,
                "prior_reserved_cny": sum(c["reserved_cny"] for c in all_calls[:number]),
            },
        )
        write_json(journal / f"{number:04d}_after.json", ledger)
    write_json(directory / "final_budget.json", ledger)
    rows, hashes = study.terminal_rows(directory, (QUESTION,))
    summary = {
        "protocol": study.PROTOCOL,
        "run_id": run_id,
        "terminal": rows,
        "prediction_sha256": hashes,
        "status": "completed",
        "stop_reason": None,
        "source_sha256": source_hash,
        "gold_loaded": False,
        "memory_updated": False,
        "complete_paired_questions": 1,
        "trailing_consecutive_failed_arms": 0,
        **{k: ledger[k] for k in ("api_requests", "reserved_cny", "estimated_actual_cny")},
    }
    write_json(directory / "SUMMARY.json", summary)
    aggregate = tmp_path / study.OUTPUT
    aggregate.mkdir()
    write_json(
        aggregate / "launch_plan.json",
        {
            "protocol": study.PROTOCOL,
            "question_ids": [QUESTION.question_id],
            "methods": list(runtime.METHODS),
            "freeze_sha256": "a" * 64,
            "gold_loaded": False,
        },
    )
    terminal = {
        "protocol": study.PROTOCOL,
        "planned_questions": 1,
        "status": "completed",
        "stop_reason": None,
        "freeze_sha256": "a" * 64,
        "batch_summary_sha256": {run_id: feedback._sha(directory / "SUMMARY.json")},
        "terminal": rows,
        "gold_loaded": False,
        "memory_updated": False,
    }
    write_json(aggregate / "TERMINAL.json", terminal)
    monkeypatch.setattr(study, "reviewed_history", lambda *a: {"calls": all_calls, "roots": []})
    return aggregate / "TERMINAL.json", directory


def test_full_collect_synthetic_seal_http_journal_replay_and_terminal(tmp_path, monkeypatch):
    terminal, directory = synthetic_collection(tmp_path, monkeypatch)
    checked = feedback.collect(
        tmp_path, expected_freeze_sha256="a" * 64, expected_terminal_sha256=feedback._sha(terminal)
    )
    assert checked["paired_ids"] == [QUESTION.question_id]
    assert checked["totals"]["api_requests"] == 7
    # A copied report modification fails before any label loader can be reached.
    report = directory / f"{QUESTION.question_id}_base.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    data["reader"]["answer"] = "tampered"
    report.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="seal"):
        feedback.collect(
            tmp_path,
            expected_freeze_sha256="a" * 64,
            expected_terminal_sha256=feedback._sha(terminal),
        )
