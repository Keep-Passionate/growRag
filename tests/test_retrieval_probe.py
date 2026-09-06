"""Synthetic retrieval-only ablations; no keys, model adapters or network calls."""

import json
from dataclasses import asdict, replace

import pytest
from test_pre_manifest import _records, _write_original

from growrag.experiments.hotpot import HotpotExample
from growrag.experiments.lexical_retriever import BM25SentenceRetriever
from growrag.experiments.pre_manifest import load_pre_examples
from growrag.experiments.protocol import Evidence, GoldRecord, RuntimeQuestion
from growrag.experiments.retrieval_probe import main, probe_queries


def tiny_example():
    return HotpotExample(
        RuntimeQuestion("query-1", "alpha beta", "synthetic"),
        (
            Evidence("e-a", "First", 0, "alpha alpha beta"),
            Evidence("e-b", "Second", 1, "alpha beta beta"),
            Evidence("e-c", "Third", 0, "gamma"),
        ),
        GoldRecord("query-1", ("NEVER_USE_THIS_GOLD_ANSWER",), (("First", 0), ("Third", 0))),
    )


def test_actual_bm25_reports_equal_order_equal_set_and_reversed_order():
    item = tiny_example()
    report = probe_queries(
        item,
        {"alpha": "alpha", "redundant": "alpha alpha unknownword", "beta": "beta"},
        top_k=2,
    )
    index = BM25SentenceRetriever(item.candidate_context)
    assert report["queries"]["alpha"]["evidence_ids"] == ["e-a", "e-b"]
    assert report["queries"]["beta"]["evidence_ids"] == ["e-b", "e-a"]
    assert report["queries"]["alpha"]["sentence_refs"] == [("First", 0), ("Second", 1)]
    assert report["queries"]["alpha"]["gold_support_recall"] == 0.5
    assert report["pairs"] == [
        {"left": "alpha", "right": "redundant", "same_order": True, "same_set": True},
        {"left": "alpha", "right": "beta", "same_order": False, "same_set": True},
        {"left": "redundant", "right": "beta", "same_order": False, "same_set": True},
    ]
    assert report["corpus_fingerprint"] == index.corpus_fingerprint
    assert report["context_fingerprint"] == index.context_fingerprint
    assert report["reader_calls"] == report["api_requests"] == 0
    assert report["transport_source"] == "local_compute"
    assert report["answer_effect"] == "unknown" and report["diagnostic_stage"] == "posthoc"
    assert "NEVER_USE_THIS_GOLD_ANSWER" not in json.dumps(report)


def test_top_k_and_nonoverlapping_queries_change_set_and_zero_hits_are_not_padded():
    report = probe_queries(tiny_example(), {"a": "alpha", "b": "beta", "oov": "zebra"}, 1)
    assert report["queries"]["a"]["evidence_ids"] == ["e-a"]
    assert report["queries"]["b"]["evidence_ids"] == ["e-b"]
    assert report["queries"]["oov"]["evidence_ids"] == []
    assert report["queries"]["oov"]["gold_support_recall"] == 0.0
    assert all(not pair["same_set"] and not pair["same_order"] for pair in report["pairs"])


@pytest.mark.parametrize("gold", [None, GoldRecord("query-1", ("unused",), ())])
def test_missing_gold_support_is_unknown_not_zero(gold):
    report = probe_queries(replace(tiny_example(), gold=gold), {"q": "alpha"})
    assert not report["gold_support_available"]
    assert report["queries"]["q"]["gold_support_recall"] is None


@pytest.mark.parametrize(
    "queries",
    [{}, {"name": "alpha", "NAME": "beta"}, {" name": "alpha"}, {"": "alpha"}, {"n": ""}],
)
def test_invalid_or_duplicate_names_and_empty_queries_are_rejected(queries):
    with pytest.raises(ValueError):
        probe_queries(tiny_example(), queries)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_top_k_is_strictly_positive_integer(top_k):
    with pytest.raises(ValueError, match="positive integer"):
        probe_queries(tiny_example(), {"q": "alpha"}, top_k)


def test_gold_does_not_change_rankings_or_enter_retriever(monkeypatch):
    item = tiny_example()
    captured = []
    original = BM25SentenceRetriever.retrieve

    def spy(self, query, *, top_k):
        captured.append((query, top_k))
        return original(self, query, top_k=top_k)

    monkeypatch.setattr(BM25SentenceRetriever, "retrieve", spy)
    first = probe_queries(item, {"q": "alpha"}, 2)
    other = replace(item, gold=GoldRecord("query-1", ("other hidden answer",), (("Second", 1),)))
    second = probe_queries(other, {"q": "alpha"}, 2)
    assert captured == [("alpha", 2), ("alpha", 2)]
    assert first["queries"]["q"]["evidence_ids"] == second["queries"]["q"]["evidence_ids"]
    assert first["context_fingerprint"] == second["context_fingerprint"]
    with pytest.raises(ValueError, match="different question"):
        probe_queries(replace(item, gold=replace(item.gold, question_id="different")), {"q": "a"})


def archived_record(item):
    index = BM25SentenceRetriever(item.candidate_context)
    question = asdict(item.question)
    record = {
        "schema_version": "pre_source_pair.v1",
        "source_id": "synthetic-pre-source",
        "question": question,
    }
    for action, query in (("BASE", item.question.text), ("FRESH", "fictional event")):
        record[action.lower()] = {
            "state": {
                "question": question.copy(),
                "rounds": [
                    {
                        "decision": {"action": action, "memory": None},
                        "search_query": query,
                        "reply": {
                            "evidence": [asdict(e) for e in index.retrieve(query, top_k=4).value],
                        },
                    }
                ],
            },
            "events": [{"status": "ok"}],
        }
    return record


@pytest.fixture
def cli_inputs(tmp_path):
    records = _records()
    for row in records:
        row["context"][0][1].append("A second fictional event happened.")
    manifest_path, _, records = _write_original(tmp_path, records)
    sources, targets, _ = load_pre_examples(manifest_path)
    record = archived_record(sources[0])
    archive = tmp_path / "source_record.json"
    archive.write_text(json.dumps(record), encoding="utf-8")
    output = tmp_path / "probe.json"
    args = [
        "--manifest",
        str(manifest_path),
        "--source-record",
        str(archive),
        "--output",
        str(output),
    ]
    return args, manifest_path, archive, output, record, sources, targets


def test_cli_verifies_replay_and_separates_frozen_arms_from_manual_posthoc(cli_inputs):
    args, manifest_path, archive, output, _, _, _ = cli_inputs
    before = (manifest_path.read_bytes(), archive.read_bytes())
    assert main([*args, "--candidate", "entities_only", "fictional event"]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["original_arm_replay_verified"]
    assert report["selected_source_role"] == "memory_seed"
    assert report["query_origins"] == {
        "BASE": "original_batch_frozen_arm",
        "FRESH": "original_batch_frozen_arm",
        "entities_only": "manual_posthoc",
    }
    pair = next(row for row in report["pairs"] if row["left"] == "FRESH")
    assert pair["same_order"] and pair["same_set"]
    assert all(
        len(report[key]) == 64
        for key in ("data_sha256", "original_manifest_sha256", "source_record_sha256")
    )
    assert before == (manifest_path.read_bytes(), archive.read_bytes())


@pytest.mark.parametrize("name", ["BASE", "fresh", "candidate"])
def test_cli_rejects_duplicate_candidates_and_reserved_names(cli_inputs, name):
    args, _, _, output, *_ = cli_inputs
    candidates = ["--candidate", name, "fictional event"]
    if name == "candidate":
        candidates *= 2
    with pytest.raises(ValueError, match="unique"):
        main([*args, *candidates])
    assert not output.exists()


@pytest.mark.parametrize(
    "problem",
    [
        "question_text",
        "target_id",
        "outside_first_32",
        "base_query",
        "evidence_order",
        "failed_arm",
    ],
)
def test_cli_rejects_archive_mismatches_without_output(cli_inputs, problem):
    args, manifest_path, archive, output, record, sources, targets = cli_inputs
    if problem == "question_text":
        record["question"]["text"] += " CHANGED"
    elif problem == "target_id":
        record = archived_record(targets[0])
    elif problem == "outside_first_32":
        original = json.loads(manifest_path.read_text(encoding="utf-8"))
        record["question"]["question_id"] = original["roles"]["memory_seed"][32]
        assert record["question"]["question_id"] not in {q.question.question_id for q in sources}
    elif problem == "base_query":
        record["base"]["state"]["rounds"][0]["search_query"] = "different"
    elif problem == "evidence_order":
        record["fresh"]["state"]["rounds"][0]["reply"]["evidence"].reverse()
    else:
        record["fresh"]["events"][0]["status"] = "error"
    archive.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        main(args)
    assert not output.exists()


def test_cli_rejects_changed_data_hash_and_existing_output(cli_inputs):
    args, manifest_path, _, output, *_ = cli_inputs
    data = manifest_path.parent / "data.json"
    data.write_bytes(data.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        main(args)
    assert not output.exists()
    output.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="no overwrite"):
        main(args)
    assert output.read_text(encoding="utf-8") == "preserve me"
