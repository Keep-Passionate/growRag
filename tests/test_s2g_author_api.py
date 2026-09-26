"""Offline transport tests: real pinned author functions, fake model responses only."""

import ast
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import s2g_author_api as bridge

UPSTREAM = Path("external/s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6")
FALSE = '{"sufficient": false, "gap_items": []}'
TRUE = '{"sufficient": true, "gap_items": []}'
PICK = '{"evidence_global_ids": [1]}'
ANSWER = "Answer: Northbridge\nRationale: The supplied sentence states this."
DOCS = (
    bridge.AuthorDocument("doc-a", "Northbridge", "Northbridge was founded in 1920. It is a town."),
    bridge.AuthorDocument("doc-b", "Eastbridge", "Eastbridge was founded in 1940."),
)


class FakeClient:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return SimpleNamespace(
            content=output,
            requested_model="fake-model",
            returned_model="fake-model",
            response_id="fake-response",
            request_id="fake-request",
            input_tokens=20,
            output_tokens=10,
            elapsed_seconds=0.0,
            audit_path=Path("synthetic-no-network"),
            transport_source="test_fake",
        )


@pytest.fixture
def upstream():
    if not UPSTREAM.exists():
        pytest.skip("author source intentionally not redistributed; acquire pinned snapshot first")
    return UPSTREAM


def make_adapter(upstream, outputs, **kwargs):
    client = FakeClient(outputs)
    retrieval_calls = []

    def retrieve(query, k):
        retrieval_calls.append((query, k))
        return DOCS[:k]

    adapter = bridge.S2GAuthorAPI(upstream, client, retrieve, **kwargs)
    return adapter, client, retrieval_calls


def test_original_main_batch_really_executes_without_gold(upstream):
    adapter, client, retrieval = make_adapter(upstream, [FALSE, PICK, TRUE, ANSWER])
    result = adapter.run("Which town is older?", "train-q1")
    assert result["answer"] == "northbridge"
    assert result["stop_reason"] == "sufficient"
    assert result["retrieval_rounds"] == 1 and result["api_calls"] == 4
    assert len(client.calls) == 4 and retrieval == [("Which town is older?", 50)]
    assert {
        "main_batch",
        "call_suff_gate_batch",
        "bm25_search_batch",
        "concat_and_pick_sentences_batch",
        "merge_evidence_only",
    } <= set(result["executed_author_functions"])
    assert result["sources"] == [
        {
            "doc_id": "doc-a",
            "title": "Northbridge",
            "sentence_id": 1,
            "text": "Northbridge was founded in 1920.",
        }
    ]
    assert result["author_rows"][-1]["Turn"] == 2  # 作者 Turn 是 judge 次数，不是检索次数。
    for row in result["author_rows"]:
        assert not any(key.startswith(("Gold", "Correct")) for key in row)
    assert not result["provenance"]["gold_provided_to_author_loop"]
    assert result["provenance"]["max_model_calls"] == 10
    json.dumps(result)


def test_main_loop_bytecode_unchanged_and_gate_only_backend_is_transformed(upstream):
    scope = bridge.load_author_scope(upstream)
    path = upstream / "inference/inference_bm25.py"
    tree = ast.parse(path.read_bytes())
    original = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main_batch"
    )
    reference = {}
    exec(compile(ast.Module(body=[original], type_ignores=[]), str(path), "exec"), reference)
    assert scope["main_batch"].__code__.co_code == reference["main_batch"].__code__.co_code
    assert "_api_gate_backend" in scope["call_suff_gate_batch"].__code__.co_names
    assert "torch" not in scope["call_suff_gate_batch"].__code__.co_names
    assert "safe_json_load" in scope["call_suff_gate_batch"].__code__.co_names
    assert "extract_gap_items" in scope["call_suff_gate_batch"].__code__.co_names
    for prohibited in ("torch", "pd", "OpenAI", "os", "tokenizer", "accelerator"):
        assert prohibited not in scope


def test_empty_first_true_verdict_uses_original_force_retrieval_guard(upstream):
    adapter, _, retrieval = make_adapter(upstream, [TRUE, PICK, TRUE, ANSWER])
    result = adapter.run("Q", "q")
    assert retrieval[0][0] == "Q"
    assert result["author_rows"][0]["Gate Output"] is False
    assert result["author_rows"][0]["SuffMissingFacts"] == []
    assert "should_force_first_retrieval" in result["executed_author_functions"]


@pytest.mark.parametrize(
    "profile,expected",
    [
        ("code_default", "Q Alpha year Beta date"),
        ("paper_k1", "Q Alpha year"),
    ],
)
def test_declared_query_profiles_preserve_author_gap_builder(upstream, profile, expected):
    gaps = json.dumps(
        {
            "sufficient": False,
            "gap_items": [
                {"target": "Alpha", "slot": "year"},
                {"target": "Beta", "slot": "date"},
            ],
        }
    )
    adapter, _, retrieval = make_adapter(upstream, [gaps, PICK, TRUE, ANSWER], gap_profile=profile)
    result = adapter.run("Q", "q")
    assert retrieval[0][0] == expected
    assert result["provenance"]["gap_profile"] == profile
    # 这是作者真实行为：第一次 judge=False 时首条 query 可以不是原问题。
    assert result["events"][3]["kind"] == "judge"


def test_four_rounds_end_with_fifth_judge_then_forced_answer(upstream):
    outputs = [item for _ in range(4) for item in (FALSE, PICK)] + [FALSE, ANSWER]
    adapter, client, retrieval = make_adapter(upstream, outputs)
    result = adapter.run("Q", "q")
    assert result["stop_reason"] == "max_turns"
    assert result["retrieval_rounds"] == 4 and len(retrieval) == 4
    assert len(client.calls) == 10 and result["api_calls"] == 10
    assert len(result["author_rows"]) == 5
    assert result["answer"] == "northbridge"  # 保留强制回答，未暗中加 abstain。


def test_empty_retrieval_skips_extractor_but_not_author_bounded_loop(upstream):
    client = FakeClient([FALSE] * 5 + [ANSWER])
    adapter = bridge.S2GAuthorAPI(upstream, client, lambda *_: [])
    result = adapter.run("Q", "q")
    assert result["api_calls"] == 6 and result["retrieval_rounds"] == 4
    assert result["stop_reason"] == "max_turns" and result["sources"] == []
    assert result["evidence_context"] == ""
    assert all("extract" not in call["prompt_version"] for call in client.calls)


def test_author_parser_salvages_json_and_discards_unknown_pointer_ids(upstream):
    selected = '```json\n{"evidence global ids": [999, 3, 1, 1, -1, "bad"]}\n```'
    adapter, _, _ = make_adapter(upstream, ["not json", selected, TRUE, ANSWER])
    result = adapter.run("Q", "q")
    assert [(s["doc_id"], s["sentence_id"]) for s in result["sources"]] == [
        ("doc-a", 1),
        ("doc-b", 1),
    ]
    assert result["author_rows"][0]["SuffMissingFacts"] == []


def test_original_bool_coercion_is_not_silently_repaired(upstream):
    adapter, _, _ = make_adapter(upstream, [FALSE, PICK, '{"sufficient":"false"}', ANSWER])
    result = adapter.run("Q", "q")
    # 当前作者代码 bool("false") == True；后续改进必须独立命名、比较，不能改基线。
    assert result["stop_reason"] == "sufficient"
    assert result["author_rows"][-1]["Gate Output"] is True


def test_document_dedup_uses_id_not_title(upstream):
    outputs = [FALSE, PICK, FALSE, FALSE, FALSE, FALSE, ANSWER]
    adapter, _, _ = make_adapter(upstream, outputs, remove_repeat_docs=True)
    result = adapter.run("Q", "q")
    assert len(result["retrieved_documents"]) == 2
    assert result["retrieval_rounds"] == 4
    assert result["provenance"]["dedup_key"] == "document_id"


def test_same_title_with_distinct_ids_is_not_title_deduplicated(upstream):
    docs = [bridge.AuthorDocument("a", "Same", "A."), bridge.AuthorDocument("b", "Same", "B.")]
    client = FakeClient([FALSE, '{"evidence_global_ids":[1,2]}', TRUE, ANSWER])
    adapter = bridge.S2GAuthorAPI(upstream, client, lambda *_: docs, remove_repeat_docs=True)
    result = adapter.run("Q", "q")
    assert [d["doc_id"] for d in result["retrieved_documents"]] == ["a", "b"]


def test_events_are_live_and_callback_cannot_mutate_internal_log(upstream):
    events = []

    def callback(event):
        events.append(dict(event))
        event["kind"] = "corrupted outside"

    adapter, client, _ = make_adapter(
        upstream, [FALSE, PICK, TRUE, ANSWER], event_callback=callback
    )
    complete = client.complete

    def checked_complete(*args, **kwargs):
        assert events[-1]["kind"] == "api_request"
        return complete(*args, **kwargs)

    client.complete = checked_complete
    result = adapter.run("Q", "q")
    assert all(event["kind"] != "corrupted outside" for event in result["events"])
    assert [event["event_index"] for event in events] == list(range(len(events)))
    assert events[-1]["kind"] == "finish"
    assert all("timestamp_utc" in event for event in events)


def test_failure_is_not_retried_and_error_text_not_logged(upstream):
    adapter, client, retrieval = make_adapter(upstream, [RuntimeError("secret-do-not-log")])
    with pytest.raises(RuntimeError, match="secret-do-not-log"):
        adapter.run("Q", "q")
    assert len(client.calls) == 1 and not retrieval
    assert adapter.events[-1]["kind"] == "api_failure"
    assert "secret-do-not-log" not in json.dumps(adapter.events)


def test_per_stage_author_caps_are_passed_to_optional_transport(upstream):
    adapter, client, _ = make_adapter(upstream, [FALSE, PICK, TRUE, ANSWER])
    client.complete_author = client.complete
    result = adapter.run("Q", "q")
    assert [call["max_output_tokens"] for call in client.calls] == [256, 64, 256, 128]
    assert all(call["temperature"] == 0 and call["top_p"] == 1 for call in client.calls)
    assert result["provenance"]["backend_generation_settings"].startswith("author_per_stage")


def test_base_shares_original_author_answer_prompt_and_parser(upstream):
    adapter, client, retrieval = make_adapter(upstream, [ANSWER])
    result = adapter.answer_once("Q", "[Northbridge] Founded in 1920.", "base-q")
    assert result["answer"] == "northbridge" and result["api_calls"] == 1
    assert not retrieval
    assert client.calls[0]["messages"][0]["content"] == adapter.scope["force_answer_prompt"]
    assert (
        "Retrieved Document: [Northbridge] Founded in 1920."
        in (client.calls[0]["messages"][1]["content"])
    )
    assert "main_batch" not in result["executed_author_functions"]


def test_bridge_does_not_import_heavy_dependencies(upstream, monkeypatch):
    real_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "torch",
            "transformers",
            "peft",
            "pandas",
            "openai",
            "pyserini",
            "pysbd",
        }
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    adapter, _, _ = make_adapter(upstream, [FALSE, PICK, TRUE, ANSWER])
    assert adapter.run("Q", "q")["answer"] == "northbridge"


def test_changed_source_is_refused_before_execution(tmp_path):
    path = tmp_path / "inference" / "inference_bm25.py"
    path.parent.mkdir()
    path.write_text("raise AssertionError('must not execute')")
    with pytest.raises(ValueError, match="fingerprint changed"):
        bridge.load_author_scope(tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_turns": 0},
        {"max_turns": 5},
        {"max_turns": True},
        {"top_docs": 0},
        {"top_docs": 7},
        {"top_docs": True},
        {"gap_profile": "invented"},
        {"dataset_name": "made-up"},
        {"remove_repeat_docs": "yes"},
    ],
)
def test_invalid_config_fails_before_loading_source(tmp_path, kwargs):
    with pytest.raises(ValueError):
        bridge.S2GAuthorAPI(tmp_path, None, None, **kwargs)


@pytest.mark.parametrize(
    "values", [("", "T", "S"), ("ID", "", "S"), ("ID", "T", ""), (None, "T", "S")]
)
def test_document_contract(values):
    with pytest.raises(ValueError):
        bridge.AuthorDocument(*values)


def test_changing_document_identity_or_duplicate_ids_fails(upstream):
    corpus = {}
    search = bridge._CallbackSearcher(lambda *_: [DOCS[0]], corpus)
    search.search("Q", 10)
    search.retrieve = lambda *_: [bridge.AuthorDocument("doc-a", "New", "Different.")]
    with pytest.raises(ValueError, match="changed content"):
        search.search("Q2", 10)
    search.retrieve = lambda *_: [DOCS[0], DOCS[0]]
    with pytest.raises(ValueError, match="duplicate"):
        search.search("Q2", 10)


def test_sequential_questions_do_not_share_evidence_or_return_mutable_old_logs(upstream):
    adapter, _, _ = make_adapter(upstream, [FALSE, PICK, TRUE, ANSWER] * 2)
    first = adapter.run("Q1", "q1")
    first_snapshot = json.dumps(first, sort_keys=True)
    second = adapter.run("Q2", "q2")
    assert first_snapshot == json.dumps(first, sort_keys=True)
    assert all(event["question_id"] == "q2" for event in second["events"])
    judge = next(event for event in second["events"] if event["kind"] == "judge")
    assert judge["evidence_contexts"] == [""]
