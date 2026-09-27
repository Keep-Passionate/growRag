"""Small synthetic states; no network, API, or private dataset required."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments.s2g_author_api import load_author_scope
from growrag.experiments.shared_s2g_corpus import CorpusDocument, ExactGold, ExactSupport

SPEC = importlib.util.spec_from_file_location(
    "query_probe",
    Path(__file__).resolve().parents[1] / "scripts" / "audits" / "query_replay_probe.py",
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _author_snapshot_or_skip():
    """Optional local snapshot integration; never download it in CI."""
    root = Path(__file__).resolve().parents[1]
    upstream = root / "external/s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6"
    if not all((upstream / name).is_file() for name in PROBE.PINNED_HASHES):
        pytest.skip("Optional pinned-author integration: local S2G snapshot is absent")
    return upstream


def documents():
    # Same title intentionally has two different versions.
    return {
        "a": CorpusDocument("a", "Same title", ("old content.",)),
        "b": CorpusDocument("b", "Same title", ("gold content.", "second support.")),
        "c": CorpusDocument("c", "Other", ("other content.",)),
    }


def row(doc):
    return {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text}


def gold(docs):
    return ExactGold(
        "qid",
        ("never used answer",),
        (
            ExactSupport("b", "Same title", 0, PROBE.digest(docs["b"].sentences[0].encode())),
            ExactSupport("b", "Same title", 1, PROBE.digest(docs["b"].sentences[1].encode())),
        ),
    )


def event(number, query, docs):
    return {
        "kind": "retrieval",
        "round": number,
        "queries": [query],
        "documents": [[row(doc) for doc in docs]],
    }


def test_both_choices_share_actual_past_not_counterfactual_history():
    docs = documents()
    calls = []

    def retrieve(query, past):
        calls.append((query, past))
        return [row(docs[{"gap1": "a", "gap2": "c", "original": "b"}[query]])]

    replay = PROBE.replay_question(
        "original",
        [event(1, "gap1", [docs["a"]]), event(2, "gap2", [docs["c"]])],
        retrieve,
        docs.__getitem__,
    )
    assert calls == [("gap1", ()), ("original", ()), ("gap2", ("a",)), ("original", ("a",))]
    scored = PROBE.score_question(replay, gold(docs), docs.__getitem__)
    assert [r["sentence_outcome"] for r in scored] == ["negative", "negative"]
    assert scored[0]["choices"]["executed"]["new_sentence_count"] == 0
    assert scored[0]["choices"]["original"]["new_sentence_count"] == 2
    assert scored[0]["choices"]["original"]["new_document_count"] == 1
    assert scored[1]["missing_gold_sentences_before"] == 2


def test_identical_query_is_retained_as_zero_and_seen_support_not_new():
    docs = documents()

    def retrieve(query, past):
        return [] if "b" in past else [row(docs["b"])]

    replay = PROBE.replay_question(
        "original",
        [event(1, "original", [docs["b"]]), event(2, "original", [])],
        retrieve,
        docs.__getitem__,
    )
    scored = PROBE.score_question(replay, gold(docs), docs.__getitem__)
    assert len(scored) == 2
    assert all(not r["query_text_changed"] and r["sentence_outcome"] == "zero" for r in scored)
    assert scored[1]["missing_gold_sentences_before"] == 0
    assert scored[1]["choices"]["executed"]["new_sentence_count"] == 0


def test_order_mismatch_is_uncomparable_even_with_same_document_set():
    docs = documents()
    replay = PROBE.replay_question(
        "original",
        [event(1, "gap", [docs["a"], docs["b"]])],
        lambda q, p: [row(docs["b"]), row(docs["a"])],
        docs.__getitem__,
    )
    assert replay["all_replays_match"] is False
    scored = PROBE.score_question(replay, gold(docs), docs.__getitem__)
    assert scored[0]["comparison_status"] == "uncomparable_replay_mismatch"
    assert "choices" not in scored[0]


def test_gold_version_hash_mismatch_is_never_semantically_guessed():
    docs = documents()
    bad = ExactGold("qid", ("unused",), (ExactSupport("b", "Same title", 0, "wronghash"),))
    with pytest.raises(ValueError, match="fingerprint"):
        PROBE.exact_targets(bad, docs.__getitem__)


def test_invalid_annotation_is_unscorable_not_zero():
    docs = documents()
    replay = PROBE.replay_question(
        "original",
        [event(1, "original", [docs["a"]])],
        lambda q, p: [row(docs["a"])],
        docs.__getitem__,
    )
    scored = PROBE.score_question(replay, ExactGold("qid", (), (), "invalid"), docs.__getitem__)
    assert scored[0]["comparison_status"] == "unscorable_annotation"
    assert "sentence_outcome" not in scored[0]


def test_author_function_really_uses_top50_then_excludes_seen_before_top6():
    scope = load_author_scope(_author_snapshot_or_skip())
    docs = [CorpusDocument(str(i), f"Doc {i}", (f"sentence {i}",)) for i in range(60)]
    searches = []

    def index(query, k):
        searches.append((query, k))
        return tuple(d.as_author_document() for d in docs[:k])

    replay = PROBE.AuthorReplay(scope, index)
    values = replay(" question ", [str(i) for i in range(4)])
    assert [v["doc_id"] for v in values] == [str(i) for i in range(4, 10)]
    assert searches == [("question", 50)]
    assert replay("question", [str(i) for i in range(50)]) == []
    assert replay.search_count == 1  # Cache is only an offline diagnostic optimization.
    assert replay.logical_requests == 2
    assert replay.cache_hits == 1


def test_full_probe_cannot_start_before_500_closed_reports(tmp_path, monkeypatch):
    index = tmp_path / "existing.sqlite"
    index.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--all-closed", "--index", str(index), "--output", str(tmp_path / "out")],
    )
    monkeypatch.setattr(PROBE, "closed_predictions", lambda *args: ([], [], {}, {"q0": 0}))
    with pytest.raises(ValueError, match="all 500"):
        PROBE.main()
    assert not (tmp_path / "out").exists()


def test_workers_above_two_are_rejected_before_any_io(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["probe", "--workers", "3", "--output", "unused"])
    with pytest.raises(SystemExit) as raised:
        PROBE.main()
    assert raised.value.code == 2


def test_per_question_replay_counts_logical_calls_separately_and_resets_cache():
    scope = load_author_scope(_author_snapshot_or_skip())
    docs = documents()
    queries = []

    class Index:
        def document(self, doc_id):
            return docs[doc_id]

        def __call__(self, query, k):
            queries.append((query, k))
            return (docs["a"].as_author_document(),)

    runtime = SimpleNamespace(
        questions=[SimpleNamespace(question_id="qid", text="original")],
        index=Index(),
        metadata={"context_fingerprint": "fixed"},
    )
    item = {
        "question_id": "qid",
        "run_id": "synthetic",
        "offset": 2,
        "reported_rounds": 1,
        "context_fingerprint": "fixed",
        "events": [event(1, "original", [docs["a"]])],
    }
    first = PROBE.replay_item(item, runtime, scope)
    second = PROBE.replay_item(item, runtime, scope)
    assert first["replay"] == second["replay"]
    assert first["logical_requests"] == 2
    assert first["index_calls"] == first["cache_hits"] == 1
    assert len(queries) == 2  # No across-question cache or model inference occurs.


def test_claim_file_is_ignored_but_real_incomplete_directory_is_reported(tmp_path):
    prefix = "2026-09-27_s2g_shared500_v1_0000_0025"
    (tmp_path / f"{prefix}.claim.json").write_text("arbitrary claim, not a batch")
    (tmp_path / prefix).mkdir()
    rows, excluded, _, offsets = PROBE.closed_predictions(tmp_path, None)
    assert rows == [] and offsets == {}
    assert excluded == [{"run_id": prefix, "reason": "not_closed"}]


def test_all500_guard_accepts_exact_coverage_and_rejects_same_count_wrong_ids_or_offsets():
    questions = [SimpleNamespace(question_id=f"q{i}") for i in range(500)]
    correct = {q.question_id: i for i, q in enumerate(questions)}
    PROBE.validate_report_coverage(correct, questions, require_all=True)
    wrong_id = {**correct}
    wrong_id["unrelated"] = wrong_id.pop("q499")
    with pytest.raises(ValueError, match="ID/offset"):
        PROBE.validate_report_coverage(wrong_id, questions, require_all=True)
    wrong_offset = {**correct, "q498": 499, "q499": 498}
    with pytest.raises(ValueError, match="ID/offset"):
        PROBE.validate_report_coverage(wrong_offset, questions, require_all=True)
    with pytest.raises(ValueError, match="exact frozen"):
        PROBE.validate_report_coverage({"q0": 0}, questions, require_all=True)


def test_completed_failure_event_must_also_match_frozen_launch(tmp_path):
    directory = tmp_path / "2026-09-27_s2g_shared500_v1_0000_0025"
    directory.mkdir()
    values = {
        "launch_plan.json": {
            "series": "500_v1",
            "manifest_sha256": PROBE.MANIFEST_SHA,
            "top_docs": 6,
            "max_retrieval_rounds": 4,
            "start": 0,
            "question_ids": ["q0"],
        },
        "reports.json": [{"question_id": "q0", "offset": 100, "complete_pair": False}],
        "final_budget.json": {},
    }
    for name, value in values.items():
        (directory / name).write_text(json.dumps(value))
    (directory / "events.jsonl").write_text(
        json.dumps({"kind": "question_complete", "question_id": "wrong", "status": "failed"})
        + "\n"
        + json.dumps({"kind": "exit"})
        + "\n"
    )
    with pytest.raises(ValueError, match="frozen launch"):
        PROBE.closed_predictions(tmp_path, None)


def test_pre_replay_loader_never_reads_labeled_reports_and_rejects_execution_feedback(
    tmp_path, monkeypatch
):
    directory = tmp_path / "2026-09-27_s2g_shared500_v1_0000_0001"
    question_dir = directory / "questions" / "0000"
    question_dir.mkdir(parents=True)
    launch = {
        "series": "500_v1",
        "manifest_sha256": PROBE.MANIFEST_SHA,
        "top_docs": 6,
        "max_retrieval_rounds": 4,
        "start": 0,
        "question_ids": ["q0"],
        "runtime_metadata": {"context_fingerprint": "fixed"},
    }
    (directory / "launch_plan.json").write_text(json.dumps(launch))
    (directory / "final_budget.json").write_text("{}")
    labeled_paths = {directory / "reports.json", question_dir / "report.json"}
    for path in labeled_paths:
        path.write_text('{"offline_gold_answers": ["must never be read"]}')
    (directory / "events.jsonl").write_text(
        json.dumps({"kind": "question_complete", "question_id": "q0", "status": "completed"})
        + "\n"
        + json.dumps({"kind": "exit"})
        + "\n"
    )
    result = {
        "question_id": "q0",
        "events": [],
        "retrieval_rounds": 0,
        "provenance": {
            "source_sha256": PROBE.PINNED_HASHES,
            "remove_repeat_docs": True,
            "dedup_key": "document_id",
            "top_documents": 6,
        },
    }
    for arm in PROBE.ARMS:
        (question_dir / f"{arm}_execution.json").write_text(
            json.dumps({"status": "completed", "feedback": None, "result": result})
        )
    original_read = Path.read_bytes

    def guarded_read(path):
        assert path not in labeled_paths, "pre-replay loader read a labeled report"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    selected, excluded, hashes, offsets = PROBE.closed_predictions(tmp_path, None)
    assert len(selected) == 1 and excluded == [] and offsets == {"q0": 0}
    assert not any(str(path) in hashes for path in labeled_paths)
    (question_dir / "S2G_AUTHOR_API4_execution.json").write_text(
        json.dumps({"status": "completed", "feedback": {"answer_em": 1}, "result": result})
    )
    with pytest.raises(ValueError, match="gold-free"):
        PROBE.closed_predictions(tmp_path, None)
