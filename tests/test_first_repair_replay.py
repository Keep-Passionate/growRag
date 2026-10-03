"""Synthetic first-repair protocol tests, never loading a real gold row or model.

这些夹具只是检查实验程序，不是QA实验题；每个断言对应一个可解释的
归因约束：相同初态、实际首修、稳定去重、完整支持句与可见窗口的区别。
"""

import json
from copy import deepcopy

import pytest

from growrag.experiments import first_repair_replay as replay
from growrag.experiments.shared_s2g_corpus import CorpusDocument

QUESTION = "Which fictional academy did Avery attend?"
FRESH = "fresh_original"
HISTORY = "history_body8"


def search(query, ids, step):
    return {
        "query": query,
        "evidence_ids": list(ids),
        "new_evidence_ids": list(ids),
        "elapsed_seconds": 0.01,
        "step": step,
    }


def report(method, *, repaired=True):
    searches = [search(QUESTION, ["d1", "d2"], 0)]
    if repaired:
        searches += [search("Avery academy attended", ["d3", "d2"], 1)]
        searches[-1]["new_evidence_ids"] = ["d3"]
    return {
        "status": "completed",
        "question_id": "synthetic-question",
        "method": method,
        "question": {
            "question_id": "synthetic-question",
            "text": QUESTION,
            "dataset": "synthetic-fixture",
        },
        "gold_loaded": False,
        "memory_updated": False,
        "episode": {
            "question_id": "synthetic-question",
            "searches": searches,
            "proposals": [],
            "evidence": [
                {"evidence_id": identity, "title": identity, "sentence_id": 0, "text": identity}
                for identity in ("d1", "d2", "d3")
            ],
            "stop_reason": "bounded",
        },
    }


def document(identity, sentences, title=None):
    return CorpusDocument(identity, title or identity, tuple(sentences))


def gold_documents(*documents):
    return {d.doc_id: (d.title, list(d.sentences)) for d in documents}


def test_first_pair_preserves_actual_searches_and_no_action():
    fresh = report(FRESH)
    history = report(HISTORY, repaired=False)
    result = replay.first_pair(fresh, history, QUESTION)
    assert result["initial"] == fresh["episode"]["searches"][0]
    assert result[FRESH] == fresh["episode"]["searches"][1]
    assert result[HISTORY] is None


def test_blocked_proposal_does_not_count_as_retrieval():
    fresh, history = report(FRESH, repaired=False), report(HISTORY, repaired=False)
    history["episode"]["proposals"] = [
        {"query": "never actually executed", "selected_card_id": "fictional-card"}
    ]
    result = replay.first_pair(fresh, history, QUESTION)
    assert result[FRESH] is None
    assert result[HISTORY] is None


def test_first_pair_ignores_later_repair_content_for_first_step():
    fresh, history = report(FRESH), report(HISTORY)
    fresh["episode"]["searches"].append(search("fresh later query", ["df"], 2))
    history["episode"]["searches"].append(search("history later query", ["dh"], 2))
    result = replay.first_pair(fresh, history, QUESTION)
    assert result[FRESH]["step"] == 1
    assert result[HISTORY]["step"] == 1
    assert "later" not in result[FRESH]["query"]


@pytest.mark.parametrize("changed", ["query", "evidence_order", "evidence_ids", "new_ids"])
def test_initial_state_differences_cannot_be_silently_compared(changed):
    fresh, history = report(FRESH), report(HISTORY)
    initial = history["episode"]["searches"][0]
    if changed == "query":
        initial["query"] = "paraphrased initial query"
    elif changed == "evidence_order":
        initial["evidence_ids"].reverse()
    elif changed == "evidence_ids":
        initial["evidence_ids"][-1] = "other-source"
    else:
        initial["new_evidence_ids"] = ["d1"]
    with pytest.raises(ValueError):
        replay.first_pair(fresh, history, QUESTION)


@pytest.mark.parametrize("step", [-1, 0, 2, True, "1"])
def test_nonfirst_repair_step_is_not_renamed_first(step):
    fresh, history = report(FRESH), report(HISTORY)
    history["episode"]["searches"][1]["step"] = step
    with pytest.raises(ValueError):
        replay.first_pair(fresh, history, QUESTION)


def test_missing_initial_search_is_not_a_no_action_case():
    fresh, history = report(FRESH), report(HISTORY, repaired=False)
    history["episode"]["searches"] = []
    with pytest.raises(ValueError):
        replay.first_pair(fresh, history, QUESTION)


def test_question_text_is_bound_to_actual_initial_query():
    fresh, history = report(FRESH), report(HISTORY)
    with pytest.raises(ValueError):
        replay.first_pair(fresh, history, "a different question")


def test_common_missing_initial_content_is_not_verified_equality():
    fresh, history = report(FRESH), report(HISTORY)
    fresh["episode"]["evidence"] = []
    history["episode"]["evidence"] = []
    with pytest.raises(ValueError):
        replay.first_pair(fresh, history, QUESTION)


def test_common_initial_content_drift_is_rejected():
    fresh, history = report(FRESH), report(HISTORY)
    history["episode"]["evidence"][0]["text"] = "Altered source text."
    with pytest.raises(ValueError):
        replay.first_pair(fresh, history, QUESTION)


def test_merge_keeps_initial_order_then_new_ranked_documents():
    first, second, third = (document(i, ["Fictional sentence."]) for i in ("d1", "d2", "d3"))
    initial, extra = (first, second), (second, third, first)
    merged = replay.merge_documents(initial, extra)
    assert merged == (first, second, third)
    assert initial == (first, second)
    assert extra == (second, third, first)


@pytest.mark.parametrize("changed", ["title", "text"])
def test_merge_rejects_same_id_content_drift(changed):
    original = document("d1", ["Original fictional content."], "Academy")
    variant = document(
        "d1",
        ["Altered fictional content."] if changed == "text" else original.sentences,
        "Different title" if changed == "title" else original.title,
    )
    with pytest.raises(ValueError):
        replay.merge_documents((original,), (variant,))


def test_no_action_union_is_exact_initial_documents():
    initial = (document("d1", ["No repair ran."]),)
    assert replay.merge_documents(initial, ()) == initial


def test_raw_sentence_coverage_not_title_or_document_count():
    first = document("d1", ["First support.", "Another support.", "Unannotated."])
    second = document("d2", ["Third support."])
    documents = gold_documents(first, second)
    targets = {("d1", 0), ("d1", 1), ("d2", 0)}
    result = replay.coverage((first,), documents, targets)
    assert result["support_total"] == 3
    assert result["raw_supported"] == 2
    assert result["visible_supported"] == 2
    assert result["raw_support_recall"] == pytest.approx(2 / 3)
    assert result["visible_support_recall"] == pytest.approx(2 / 3)


def test_unannotated_extra_document_is_not_a_support_gain():
    support = document("support", ["Annotated support."])
    distractor = document("distractor", ["Unannotated extra paragraph."])
    gold = gold_documents(support)
    targets = {("support", 0)}
    before = replay.coverage((support,), gold, targets)
    after = replay.coverage((support, distractor), gold, targets)
    assert before["raw_support_recall"] == after["raw_support_recall"] == 1
    assert before["visible_support_recall"] == after["visible_support_recall"] == 1


def test_raw_support_monotonic_under_verified_union():
    first, second = document("d1", ["First support."]), document("d2", ["Second support."])
    gold = gold_documents(first, second)
    targets = {("d1", 0), ("d2", 0)}
    before = replay.coverage((first,), gold, targets)
    union = replay.merge_documents((first,), (second, first))
    after = replay.coverage(union, gold, targets)
    assert before["raw_support_recall"] == 0.5
    assert after["raw_support_recall"] == 1


def test_visible_support_can_decrease_when_budget_is_shared():
    # 已有证据没有丢失，但加入长段落之后，每条可见窗口变短。
    support = document("support", ["x" * 9000, "Late complete support sentence."])
    distractor = document("distractor", ["y" * 9000])
    gold, targets = gold_documents(support), {("support", 1)}
    before = replay.coverage((support,), gold, targets)
    after = replay.coverage((support, distractor), gold, targets)
    assert before["raw_support_recall"] == after["raw_support_recall"] == 1
    assert before["visible_support_recall"] == 1
    assert after["visible_support_recall"] == 0


def test_partial_sentence_prefix_is_not_full_support():
    support = document("support", ["x" * 16000])
    result = replay.coverage((support,), gold_documents(support), {("support", 0)})
    assert result["raw_support_recall"] == 1
    assert result["visible_support_recall"] == 0


@pytest.mark.parametrize("changed", ["title", "text"])
def test_coverage_rejects_gold_known_document_identity_drift(changed):
    original = document("d1", ["Verified fictional text."], "Verified title")
    altered = document(
        "d1",
        ["Different fictional text."] if changed == "text" else original.sentences,
        "Different title" if changed == "title" else original.title,
    )
    with pytest.raises(ValueError):
        replay.coverage((altered,), gold_documents(original), {("d1", 0)})


def test_coverage_does_not_mutate_inputs():
    item = document("d1", ["First support."])
    gold, targets = gold_documents(item), {("d1", 0)}
    snapshot = deepcopy((gold, targets))
    replay.coverage((item,), gold, targets)
    assert (gold, targets) == snapshot


def test_empty_support_annotation_is_not_zero_success():
    with pytest.raises(ValueError):
        replay.coverage((), {}, set())


class FakeIndex:
    """Tiny in-memory retrieval contract fixture, not an experimental model."""

    def __init__(self, *args):
        self.documents = (document("d1", ["First source."]), document("d2", ["Second source."]))
        self.callback_documents = tuple(d.as_author_document() for d in self.documents)
        self.calls = []
        self.closed = False

    def __call__(self, query, top_k):
        self.calls.append((query, top_k))
        return self.callback_documents

    def document(self, identity):
        return next(d for d in self.documents if d.doc_id == identity)

    def close(self):
        self.closed = True


def pool_for(documents):
    return {
        d.doc_id: {"evidence_id": d.doc_id, "title": d.title, "sentence_id": 0, "text": d.text}
        for d in documents
    }


def test_replay_executes_saved_query_at_exact_top_six_and_returns_source_sentences():
    index = FakeIndex()
    saved = search("Saved first query", ["d1", "d2"], 1)
    documents, elapsed = replay.replay_search(index, saved, [pool_for(index.documents)])
    assert index.calls == [("Saved first query", 6)]
    assert documents == index.documents
    assert all(isinstance(d, CorpusDocument) for d in documents)
    assert elapsed >= 0


def test_replay_rejects_correct_set_in_wrong_order():
    index = FakeIndex()
    saved = search("Saved first query", ["d2", "d1"], 1)
    with pytest.raises(ValueError, match="ordered IDs"):
        replay.replay_search(index, saved, [pool_for(index.documents)])


@pytest.mark.parametrize("changed", ["missing", "title", "text"])
def test_replay_checks_every_source_pool_not_only_first(changed):
    index = FakeIndex()
    first = pool_for(index.documents)
    second = deepcopy(first)
    if changed == "missing":
        del second["d2"]
    else:
        second["d2"][changed] = "Changed original source."
    saved = search("Saved first query", ["d1", "d2"], 1)
    with pytest.raises(ValueError, match="content differs"):
        replay.replay_search(index, saved, [first, second])


def test_callback_text_mismatch_cannot_be_hidden_by_document_lookup():
    index = FakeIndex()
    index.callback_documents = (
        document("d1", ["Changed callback text."]).as_author_document(),
        index.callback_documents[1],
    )
    saved = search("Saved first query", ["d1", "d2"], 1)
    with pytest.raises(ValueError, match="callback/source"):
        replay.replay_search(index, saved, [pool_for(index.documents)])


def test_run_never_overwrites_existing_output(tmp_path, monkeypatch):
    target = tmp_path / replay.OUTPUT
    target.mkdir(parents=True)
    (target / "original.txt").write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        replay, "load_inputs", lambda project: pytest.fail("must stop before inputs")
    )
    with pytest.raises(FileExistsError):
        replay.run(tmp_path)
    assert (target / "original.txt").read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("output", ["../outside", "runs/nested/output", "knowledge/output"])
def test_output_path_is_a_direct_runs_child(tmp_path, output, monkeypatch):
    monkeypatch.setattr(
        replay, "load_inputs", lambda project: pytest.fail("must stop before inputs")
    )
    with pytest.raises(ValueError, match="direct runs child"):
        replay.run(tmp_path, output)


def test_retrieval_failure_is_preserved_and_prevents_gold_projection(tmp_path, monkeypatch):
    (tmp_path / "runs").mkdir()
    index = FakeIndex()
    manifest = {
        "corpus_ref": {"index_path": "index", "path": "corpus", "sha256": "a" * 64, "rows": 2}
    }
    monkeypatch.setattr(
        replay,
        "load_inputs",
        lambda project: ([{"question_id": "synthetic"}], {"synthetic": {}}, manifest, {}),
    )
    monkeypatch.setattr(replay, "SharedBM25Index", lambda *args: index)

    def fail_retrieval(*args):
        raise ValueError("synthetic replay mismatch")

    monkeypatch.setattr(replay, "replay_case", fail_retrieval)
    monkeypatch.setattr(
        replay, "load_gold", lambda *args: pytest.fail("gold must stay closed after replay failure")
    )
    with pytest.raises(ValueError, match="synthetic replay mismatch"):
        replay.run(tmp_path)
    failure = json.loads((tmp_path / replay.OUTPUT / "FAILED.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "retrieval"
    assert failure["error"] == "synthetic replay mismatch"
    assert not (tmp_path / replay.OUTPUT / "SUMMARY.json").exists()
    assert index.closed
