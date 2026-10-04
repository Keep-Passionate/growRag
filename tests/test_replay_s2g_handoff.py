"""Offline audit checks; fixtures are synthetic and carry no QA performance claim."""

import ast
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audits/replay_s2g_handoff.py"
SPEC = importlib.util.spec_from_file_location("replay_s2g_handoff", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str


class Index:
    def __init__(self):
        self.calls = []

    def search_with_scores(self, query, k):
        self.calls.append((query, k))
        return [(Doc("d1", "Title", "First. Second."), 3.0)]


def trace():
    return {
        "question_id": "teaching-only",
        "run_id": "synthetic",
        "manifest_index": 0,
        "question": "Q",
        "events": [
            {
                "kind": "judge",
                "evidence_contexts": [""],
                "verdicts": [False],
                "gap_items": [[{"target": "X", "slot": "R"}]],
                "prior_retrieved_doc_ids": [],
            },
            {
                "kind": "query",
                "gap_items": [{"target": "X", "slot": "R"}],
                "gap_profile": "paper_k1",
                "query": "Q X R",
                "prior_retrieved_doc_ids": [],
            },
            {
                "kind": "retrieval",
                "round": 1,
                "queries": ["Q X R"],
                "documents": [[{"doc_id": "d1", "title": "Title", "text": "First. Second."}]],
                "prior_retrieved_doc_ids": [],
            },
            {
                "kind": "extraction",
                "pointers": [[[2]]],
                "sources": [
                    {"doc_id": "d1", "title": "Title", "sentence_id": 2, "text": "Second."}
                ],
                "prior_retrieved_doc_ids": ["d1"],
            },
            {
                "kind": "judge",
                "evidence_contexts": ["Second."],
                "verdicts": [True],
                "gap_items": [[]],
                "prior_retrieved_doc_ids": ["d1"],
            },
        ],
    }


def scope():
    return {
        "build_query_from_missing": lambda q, gaps, **kw: "Q X R",
        "split_wiki_sentences": lambda text: ["First.", "Second."],
        "merge_evidence_only": lambda titles, texts, pointers: "Second.",
        "append_evidence_context": lambda old, new: new,
    }


def test_exact_query_retrieval_and_pre_judge_window():
    index = Index()
    report = audit.replay_trace(trace(), index, scope())
    assert index.calls == [("Q X R", 50)]
    assert report["retrievals"][0]["observed_postfilter_ids"] == ["d1"]
    assert [c["kind"] for c in report["checks"]].count("pre_judge_context") == 2


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda t: t["events"][0].update(evidence_contexts=["Second."]), "pre-Judge"),
        (lambda t: t["events"][1].update(query="modified"), "exact query"),
        (lambda t: t["events"][2]["documents"][0][0].update(text="wrong"), "ordered retrieval"),
        (lambda t: t["events"][3]["sources"][0].update(text="wrong"), "pointer/text"),
        (lambda t: t["events"][4].update(prior_retrieved_doc_ids=[]), "prior document"),
    ],
)
def test_tampered_evidence_or_query_cannot_pass(change, error):
    example = trace()
    change(example)
    with pytest.raises(ValueError, match=error):
        audit.replay_trace(example, Index(), scope())


def test_retrieval_excludes_previous_docs_and_preserves_rank():
    example = trace()
    example["events"] = example["events"][:-1] + [
        {
            "kind": "judge",
            "evidence_contexts": ["Second."],
            "verdicts": [False],
            "gap_items": [[{"target": "X", "slot": "R"}]],
        },
        {
            "kind": "query",
            "gap_items": [{"target": "X", "slot": "R"}],
            "gap_profile": "paper_k1",
            "query": "Q X R",
        },
        {"kind": "retrieval", "round": 2, "queries": ["Q X R"], "documents": [[]]},
    ]
    report = audit.replay_trace(example, Index(), scope())
    assert report["retrievals"][1]["recomputed_raw_top50_ids"] == ["d1"]
    assert report["retrievals"][1]["observed_postfilter_ids"] == []


def test_constant_and_default_changes_are_detected():
    old = ast.parse("TOKEN = 'a'\ndef search(k=50): return TOKEN")
    different = ast.parse("TOKEN = 'b'\ndef search(k=6): return TOKEN")
    for name in ("TOKEN", "search"):
        assert ast.dump(audit._definition(old, name)) != ast.dump(
            audit._definition(different, name)
        )


def test_frozen_snapshot_tamper_fails_before_import(tmp_path):
    files = {"any": {"text": "x", "sha256": "wrong"}}
    sha = hashlib.sha256(
        json.dumps(
            files,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="aggregate mismatch"):
        audit.verify_source({"files": files, "sha256": sha}, {"source_sha256": "bad"}, tmp_path)
