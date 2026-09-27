"""Synthetic, offline exact-version retrieval/feedback boundary contracts."""

import hashlib
import json
import math
from dataclasses import asdict, fields
from pathlib import Path

import pytest

from growrag.experiments import shared_s2g_corpus as shared
from growrag.experiments.lexical_retriever import BM25SentenceRetriever
from growrag.experiments.protocol import Evidence
from growrag.experiments.s2g_author_api import load_author_scope


def row(title, sentences):
    return {
        "doc_id": hashlib.sha256(shared._canonical([title, sentences])).hexdigest(),
        "title": title,
        "sentences": sentences,
    }


def dump_jsonl(path, values):
    path.write_bytes(b"".join(shared._canonical(value) + b"\n" for value in values))


def write_manifest(root, manifest):
    for name, info in manifest["artifacts"].items():
        raw = (root / name).read_bytes()
        info.update(
            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), rows=len(raw.splitlines())
        )
    raw = shared._canonical(manifest) + b"\n"
    (root / "manifest.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (root / "manifest.sha256").write_text(f"{digest}  manifest.json\n", encoding="ascii")
    return root / "manifest.json"


@pytest.fixture
def bundle(tmp_path):
    docs = [
        row("Alpha", [" Alpha was born in 1990. It is in East County.", "Its mayor is Ann."]),
        row("Alpha", ["Alpha was born in 2000.", "Its mayor is Bob."]),
        row("Beta", ["Beta is east of Alpha. Café café."]),
    ]
    dump_jsonl(tmp_path / "corpus.jsonl", docs)
    dump_jsonl(
        tmp_path / "runtime_questions.jsonl",
        [{"question_id": "q1", "text": "When was Alpha born?", "dataset": "synthetic"}],
    )
    dump_jsonl(
        tmp_path / "corpus_sources.jsonl",
        [{"doc_id": doc["doc_id"], "source_question_ids": ["q1"]} for doc in docs],
    )
    dump_jsonl(
        tmp_path / "gold.jsonl",
        [
            {
                "question_id": "q1",
                "answers": ["1990"],
                "supporting_facts": [["Alpha", 0]],
                "exact_support": [
                    {
                        "doc_id": docs[0]["doc_id"],
                        "title": "Alpha",
                        "sentence_index": 0,
                        "text_sha256": hashlib.sha256(docs[0]["sentences"][0].encode()).hexdigest(),
                    }
                ],
            }
        ],
    )
    manifest = {
        "schema_version": "growrag-shared-hotpot-development-v1",
        "official_split": "train",
        "role": "development",
        "official_dev_test_used": False,
        "selection_uses_gold": False,
        "corpus_selection_uses_gold": False,
        "old_check_question_or_gold_projected": False,
        "runtime_input_allowlist": ["runtime_questions.jsonl", "corpus.jsonl"],
        "question_ids": ["q1"],
        "selected_ids_sha256": shared._ids_digest(["q1"]),
        "target_ids_forbidden_for_training_or_memory": ["q1"],
        "excluded_question_ids": ["old"],
        "question_types": {"q1": "bridge"},
        "corpus_statistics": {"documents": 3},
        "artifacts": {
            name: {
                "contains_gold": name == "gold.jsonl",
                "runtime_safe": name in shared.RUNTIME_FILES,
            }
            for name in [
                "corpus.jsonl",
                "runtime_questions.jsonl",
                "gold.jsonl",
                "corpus_sources.jsonl",
            ]
        },
    }
    return tmp_path, docs, manifest, write_manifest(tmp_path, manifest)


def test_runtime_only_reads_allowlisted_artifacts_and_does_not_retain_metadata(bundle, monkeypatch):
    root, _, _, path = bundle
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        if self.name in {"gold.jsonl", "corpus_sources.jsonl"}:
            pytest.fail("runtime accessed forbidden data")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    runtime = shared.load_runtime(path)
    assert [q.question_id for q in runtime.questions] == ["q1"]
    assert {field.name for field in fields(runtime)} == {"questions", "index", "metadata"}
    assert set(asdict(runtime.questions[0])) == {"question_id", "text", "dataset"}
    assert not hasattr(runtime.index, "manifest") and not hasattr(runtime.index, "gold")
    assert not {"question_types", "source_question_ids", "answers"} & runtime.metadata.keys()
    assert runtime.metadata["document_count"] == 3
    assert runtime.index.path == root / "bm25_shared.sqlite"
    runtime.close()


@pytest.mark.parametrize("query", ["Alpha", "Alpha Alpha café", "east county", "1990", "unknown"])
def test_disk_bm25_rankings_match_existing_paragraph_units_exactly(bundle, query):
    _, docs, _, path = bundle
    runtime = shared.load_runtime(path)
    evidence = tuple(
        Evidence(doc["doc_id"], doc["title"], 0, "\n".join(doc["sentences"])) for doc in docs
    )
    reference = BM25SentenceRetriever(evidence)
    expected = reference.retrieve(query, top_k=50).value
    actual = runtime.index(query, 50)
    assert [doc.doc_id for doc in actual] == [item.evidence_id for item in expected]
    assert [doc.text for doc in actual] == [item.text for item in expected]
    runtime.close()


def test_same_title_versions_are_distinct_and_raw_sentence_bytes_survive(bundle):
    _, docs, _, path = bundle
    runtime = shared.load_runtime(path)
    found = runtime.index("Alpha", 50)
    assert len([doc for doc in found if doc.title == "Alpha"]) == 2
    doc = runtime.index.document(docs[0]["doc_id"])
    assert doc.sentences == tuple(docs[0]["sentences"])
    assert doc.sentences[0].startswith(" ")
    runtime.close()


def test_exact_scores_match_existing_positive_idf_and_distinct_query_terms(bundle):
    _, docs, _, path = bundle
    runtime = shared.load_runtime(path, k1=0.9, b=0.4)
    evidence = tuple(
        Evidence(doc["doc_id"], doc["title"], 0, "\n".join(doc["sentences"])) for doc in docs
    )
    reference = BM25SentenceRetriever(evidence, k1=0.9, b=0.4)
    expected = {}
    for item, counts, length in zip(
        reference.corpus, reference._term_counts, reference._lengths, strict=True
    ):
        score = 0.0
        for term in ("alpha", "born", "café"):
            if not counts.get(term):
                continue
            df = reference._document_frequency[term]
            idf = math.log1p((len(evidence) - df + 0.5) / (df + 0.5))
            norm = 0.9 * (1 - 0.4 + 0.4 * length / reference._average_length)
            score += idf * (counts[term] * (0.9 + 1) / (counts[term] + norm))
        if score:
            expected[item.evidence_id] = score
    actual = {
        doc.doc_id: score
        for doc, score in runtime.index.search_with_scores("Alpha BORN café Alpha café", 50)
    }
    assert actual == expected
    runtime.close()


def test_index_is_reused_and_wrong_config_is_not_silently_rebuilt(bundle, monkeypatch):
    _, _, _, path = bundle
    first = shared.load_runtime(path)
    index_path = first.index.path
    stamp = index_path.stat().st_mtime_ns
    first.close()
    monkeypatch.setattr(shared.SharedBM25Index, "_build", lambda *_: pytest.fail("must reuse"))
    second = shared.load_runtime(path)
    assert second.index.path.stat().st_mtime_ns == stamp
    second.close()
    with pytest.raises(ValueError, match="source/config mismatch"):
        shared.load_runtime(path, k1=0.9, b=0.4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("official_split", "validation"),
        ("runtime_input_allowlist", ["runtime_questions.jsonl", "gold.jsonl"]),
        ("selection_uses_gold", True),
        ("excluded_question_ids", ["q1"]),
        ("selected_ids_sha256", "0" * 64),
        ("target_ids_forbidden_for_training_or_memory", []),
    ],
)
def test_manifest_contract_fails_closed(bundle, field, value):
    root, _, manifest, _ = bundle
    manifest[field] = value
    path = write_manifest(root, manifest)
    with pytest.raises(ValueError):
        shared.load_runtime(path)


def test_manifest_and_artifact_tampering_rejected(bundle):
    root, _, _, path = bundle
    with pytest.raises(ValueError, match="manifest SHA"):
        shared.load_runtime(path, expected_manifest_sha256="0" * 64)
    with (root / "corpus.jsonl").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="artifact SHA/size"):
        shared.load_runtime(path)


@pytest.mark.parametrize(
    "name,extra",
    [
        ("runtime_questions.jsonl", {"answer": "1990"}),
        ("corpus.jsonl", {"source_question_ids": ["q1"]}),
    ],
)
def test_hidden_gold_or_ownership_fields_not_projected_away_silently(bundle, name, extra):
    root, _, manifest, _ = bundle
    values = [json.loads(line) for line in (root / name).read_text().splitlines()]
    values[0].update(extra)
    dump_jsonl(root / name, values)
    path = write_manifest(root, manifest)
    with pytest.raises(ValueError, match="unallowlisted|only doc_id"):
        shared.load_runtime(path)


def test_index_partial_or_hash_corruption_is_preserved_not_overwritten(bundle):
    root, _, _, path = bundle
    index = root / "partial.sqlite"
    index.write_text("partial")
    with pytest.raises(FileNotFoundError):
        shared.load_runtime(path, index_path=index)
    assert index.read_text() == "partial"
    runtime = shared.load_runtime(path)
    runtime.close()
    with (root / "bm25_shared.sqlite").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="index SHA"):
        shared.load_runtime(path)


def test_gold_is_explicit_after_execution_and_requires_completed_allowlist(bundle):
    _, _, _, path = bundle
    for ids in ([], ["unknown"], ["q1", "q1"]):
        with pytest.raises(ValueError, match="completed question allowlist"):
            shared.load_gold_after_execution(path, completed_question_ids=ids)
    gold = shared.load_gold_after_execution(path, completed_question_ids=["q1"])["q1"]
    assert gold.answers == ("1990",) and gold.annotation_status == "valid"


def result_for(doc, *, sources=()):
    return {
        "question_id": "q1",
        "answer": "1990",
        "retrieved_documents": [asdict(doc.as_author_document())],
        "sources": list(sources),
    }


def test_same_title_wrong_version_cannot_gain_gold_support(bundle):
    _, docs, _, path = bundle
    runtime = shared.load_runtime(path)
    gold = shared.load_gold_after_execution(path, completed_question_ids=["q1"])["q1"]
    wrong = runtime.index.document(docs[1]["doc_id"])
    feedback = shared.score_result(result_for(wrong), gold, runtime.index, retained_mode="raw")
    assert feedback["raw_support_recall"] == feedback["retained_support_recall"] == 0
    assert feedback["answer_em"] == 1  # 答案碰巧对，不代表拿到了标注原文版本。
    runtime.close()


def test_base_raw_mode_and_empty_s2g_sources_are_not_conflated(bundle):
    _, docs, _, path = bundle
    runtime = shared.load_runtime(path)
    gold = shared.load_gold_after_execution(path, completed_question_ids=["q1"])["q1"]
    result = result_for(runtime.index.document(docs[0]["doc_id"]))
    base = shared.score_result(result, gold, runtime.index, retained_mode="raw")
    s2g = shared.score_result(result, gold, runtime.index, retained_mode="sources")
    assert base["raw_support_recall"] == s2g["raw_support_recall"] == 1
    assert base["retained_support_recall"] == 1
    assert s2g["retained_support_recall"] == 0
    runtime.close()


def test_multiple_regex_units_must_cover_whole_gold_raw_sentence(bundle):
    _, docs, _, path = bundle
    runtime = shared.load_runtime(path)
    gold = shared.load_gold_after_execution(path, completed_question_ids=["q1"])["q1"]
    doc = runtime.index.document(docs[0]["doc_id"])
    source1 = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "sentence_id": 1,
        "text": "Alpha was born in 1990.",
    }
    source2 = {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "sentence_id": 2,
        "text": "It is in East County.",
    }
    partial = shared.score_result(
        result_for(doc, sources=[source1]), gold, runtime.index, retained_mode="sources"
    )
    complete = shared.score_result(
        result_for(doc, sources=[source1, source2]), gold, runtime.index, retained_mode="sources"
    )
    assert partial["retained_support_recall"] == 0 and partial["partial_raw_sentences"] == 1
    assert partial["unaligned_sources"] == 0
    assert complete["retained_support_recall"] == 1 and complete["partial_raw_sentences"] == 0
    assert complete["answer_supported"] is None
    runtime.close()


def test_invalid_source_pointer_is_reported_not_semantically_scored(bundle):
    _, docs, _, path = bundle
    runtime = shared.load_runtime(path)
    gold = shared.load_gold_after_execution(path, completed_question_ids=["q1"])["q1"]
    doc = runtime.index.document(docs[0]["doc_id"])
    result = result_for(
        doc,
        sources=[
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "sentence_id": 1,
                "text": "Fabricated evidence.",
            }
        ],
    )
    feedback = shared.score_result(result, gold, runtime.index, retained_mode="sources")
    assert feedback["unaligned_sources"] == 1 and feedback["retained_support_recall"] == 0
    assert feedback["answer_supported"] is None
    runtime.close()


def test_gold_support_text_fingerprint_checked_against_exact_corpus(bundle):
    root, _, manifest, _ = bundle
    values = [json.loads(line) for line in (root / "gold.jsonl").read_text().splitlines()]
    values[0]["exact_support"][0]["text_sha256"] = "0" * 64
    dump_jsonl(root / "gold.jsonl", values)
    path = write_manifest(root, manifest)
    runtime = shared.load_runtime(path)
    gold = shared.load_gold_after_execution(path, completed_question_ids=["q1"])["q1"]
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        shared.score_result({"question_id": "q1"}, gold, runtime.index, retained_mode="raw")
    runtime.close()


def test_scale_schema_and_invalid_annotations_are_supported_without_ownership_reads(bundle):
    root, docs, manifest, _ = bundle
    manifest["schema_version"] = "growrag-shared-hotpot-scale-development-v1"
    manifest.pop("selection_uses_gold")
    manifest.pop("corpus_selection_uses_gold")
    manifest["selection_uses_gold_or_difficulty"] = False
    artifact = manifest["artifacts"].pop("corpus_sources.jsonl")
    manifest["artifacts"]["corpus_source_edges.jsonl"] = artifact
    dump_jsonl(
        root / "corpus_source_edges.jsonl",
        [{"doc_id": doc["doc_id"], "source_question_id": "q1"} for doc in docs],
    )
    dump_jsonl(
        root / "gold.jsonl",
        [
            {
                "question_id": "q1",
                "answers": [],
                "supporting_facts": [],
                "exact_support": [],
                "annotation_status": "invalid",
                "annotation_issue": "missing support annotation",
            }
        ],
    )
    path = write_manifest(root, manifest)
    runtime = shared.load_runtime(path)
    gold = shared.load_gold_after_execution(path, completed_question_ids=["q1"])["q1"]
    feedback = shared.score_result(
        {"question_id": "q1", "answer": "X"}, gold, runtime.index, retained_mode="sources"
    )
    assert feedback["unscorable_annotation"]
    assert all(
        feedback[name] is None
        for name in ["answer_em", "answer_f1", "raw_support_recall", "retained_support_recall"]
    )
    runtime.close()


@pytest.mark.parametrize(
    "sentences",
    [
        [" Alpha. Beta!", " Gamma? Delta."],
        ["你好。世界！", "第二段?好。"],
        [" Mr. Smith is here.\nHe said yes.", "", " Another paragraph."],
        [" A\r\nB.", "End."],
    ],
)
def test_alignment_splitter_matches_real_author_regex_fallback(sentences):
    upstream = Path("external/s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6")
    if not upstream.exists():
        pytest.skip("pinned author snapshot intentionally not redistributed")
    author = load_author_scope(upstream)
    doc = shared._document(row("Title", sentences))
    _, units = shared._author_units_with_spans(doc)
    assert [part for part, _, _ in units] == author["split_wiki_sentences"](doc.text)


def test_invalid_runtime_index_parameters(bundle):
    _, _, _, path = bundle
    for kwargs in ({"k1": 0}, {"b": 2}, {"k1": float("nan")}):
        with pytest.raises(ValueError):
            shared.load_runtime(path, **kwargs)
    runtime = shared.load_runtime(path)
    with pytest.raises(ValueError):
        runtime.index("!!!", 3)
    with pytest.raises(ValueError):
        runtime.index("Alpha", True)
    runtime.close()
