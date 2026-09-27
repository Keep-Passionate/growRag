"""Offline ReFormeR bridge checks: pinned author methods, never paid APIs.

Author sources remain ignored local inputs. These tests intentionally do not
redistribute the pattern library or prompt text in the public test suite.
"""

import ast
import builtins
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from growrag.experiments import reformer_api as bridge
from growrag.experiments.api_client import APIRequestError, ChatResponse
from growrag.experiments.s2g_author_api import AuthorDocument

UPSTREAM = Path("external/reformer_author_snapshot/aminbigdeli-ReFormeR-72e5245")
READER = Path("external/s2g_author_snapshot/nianaaa-S2G-RAG-5d842a6")
ANSWER = "Answer: Northbridge\nRationale: The supplied sentence states this."
QUESTION = "Which synthetic town is older?"
REWRITE = "Northbridge Eastbridge foundation dates"
DOCS = tuple(
    AuthorDocument(f"doc-{i}", f"Title {i}", f"Unique retrieved text marker {i}.") for i in range(6)
)


class FakeClient:
    """Uses the real transport response contract without opening a connection."""

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []
        self.block_reason = None

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": copy.deepcopy(messages), **kwargs})
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return ChatResponse(
            content=output,
            requested_model="test-fake-model",
            returned_model="test-fake-model",
            response_id="synthetic-response",
            request_id="synthetic-request",
            input_tokens=20,
            output_tokens=10,
            elapsed_seconds=0.0,
            audit_path=Path("synthetic-no-network"),
            transport_source="test_fake",
        )


@pytest.fixture
def upstream():
    if not UPSTREAM.exists() or not READER.exists():
        pytest.skip("pinned author snapshots are intentionally not redistributed")
    return UPSTREAM


def make_adapter(upstream, outputs, *, docs=DOCS, event_callback=None):
    client = FakeClient(outputs)
    calls = []

    def retrieve(query, k):
        calls.append((query, k))
        return docs[:k]

    adapter = bridge.ReFormeRAPI(upstream, READER, client, retrieve, event_callback=event_callback)
    return adapter, client, calls


def run_normal(upstream, *, event_callback=None):
    adapter, client, retrieval = make_adapter(upstream, [], event_callback=event_callback)
    selected = copy.deepcopy(adapter.patterns[2])
    client.outputs = [json.dumps(selected), REWRITE, ANSWER]
    return adapter, client, retrieval, selected, adapter.run(QUESTION, "train-q1")


def test_actual_author_methods_execute_and_same_reader_answers_original_question(upstream):
    adapter, client, retrieval, selected, result = run_normal(upstream)
    assert result["answer"] == "northbridge"
    assert result["question"] == QUESTION
    assert result["selected_pattern"] == selected
    assert result["canonical_library_match"] is True
    assert result["retrieval_query"] == QUESTION + " " + REWRITE
    assert retrieval == [(QUESTION, 3), (QUESTION + " " + REWRITE, 6)]
    assert result["retrieval_rounds"] == 2 and result["api_calls"] == 3
    assert len(client.calls) == 3
    assert len(result["initial_selector_documents"]) == 3
    assert len(result["retrieved_documents"]) == 6
    assert result["author_fallback_stages"] == []
    assert client.calls[2]["messages"][0]["content"] == adapter.reader.scope["force_answer_prompt"]
    assert QUESTION in client.calls[2]["messages"][1]["content"]
    assert REWRITE not in client.calls[2]["messages"][1]["content"]
    assert all(doc.text in client.calls[2]["messages"][1]["content"] for doc in DOCS)
    assert result["provenance"]["gold_used_by_method"] is False
    assert result["provenance"]["hotpot_memory_built"] is False
    assert result["provenance"]["trained_selector_reproduced"] is False
    assert result["provenance"]["pattern_library_frozen"] is True
    json.dumps(result)


def test_author_prompts_are_exact_and_documents_only_enter_selection(upstream):
    adapter, client, _, selected, _ = run_normal(upstream)
    prompts = json.loads((upstream / "reformer/prompts.json").read_bytes())
    expected_selection = prompts["user_prompts"]["pattern_selection"]["template"].format(
        query=QUESTION,
        documents_text="\n\n".join(doc.text for doc in DOCS[:3]),
        patterns_json=json.dumps(adapter.patterns, indent=2),
    )
    expected_rewrite = prompts["user_prompts"]["pattern_application"]["template"].format(
        query=QUESTION,
        transformation_rule=selected["transformation_rule"],
        examples_json=json.dumps(selected["examples"], indent=2),
    )
    for call, stage, user in zip(
        client.calls[:2],
        ("pattern_selection", "pattern_application"),
        (expected_selection, expected_rewrite),
        strict=True,
    ):
        assert call["messages"] == [
            {"role": "system", "content": prompts["system_prompts"][stage]},
            {"role": "user", "content": user},
        ]
    assert DOCS[3].text not in expected_selection
    assert all(doc.text not in expected_rewrite for doc in DOCS)
    assert [c["prompt_version"] for c in client.calls[:2]] == [
        bridge.PROMPT_VERSIONS["select"],
        bridge.PROMPT_VERSIONS["rewrite"],
    ]


def test_actual_selected_method_bytecode_is_unchanged(upstream):
    _, _, methods = bridge.load_author_components(upstream, SimpleNamespace(error=lambda _: None))
    path = upstream / "reformer/core/reformer.py"
    tree = ast.parse(path.read_bytes())
    original_class = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    for name, method in methods.items():
        node = next(
            n for n in original_class.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        scope = {"List": list, "Dict": dict}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        assert method.__code__.co_code == scope[name].__code__.co_code
        assert method.__code__.co_filename == str(path)


@pytest.mark.parametrize("content", ["not JSON", "```json\n{}\n```"])
def test_invalid_selection_json_preserves_explicit_author_first_pattern_fallback(upstream, content):
    adapter, client, retrieval = make_adapter(upstream, [content, REWRITE, ANSWER])
    result = adapter.run(QUESTION, "q1")
    assert result["selected_pattern"] == adapter.patterns[0]
    assert result["author_fallback_stages"] == ["select"]
    assert result["canonical_library_match"] is True
    assert any(e["kind"] == "author_fallback" and e["stage"] == "select" for e in result["events"])
    assert len(client.calls) == 3 and len(retrieval) == 2


def test_mutated_model_pattern_is_logged_not_silently_replaced(upstream):
    adapter, client, _ = make_adapter(upstream, [])
    selected = copy.deepcopy(adapter.patterns[0])
    selected["transformation_rule"] = "Synthetic changed rule to expose author behavior."
    client.outputs = [json.dumps(selected), REWRITE, ANSWER]
    result = adapter.run(QUESTION, "q1")
    assert result["selected_pattern"] == selected
    assert result["canonical_library_match"] is False
    assert selected["transformation_rule"] in client.calls[1]["messages"][1]["content"]
    assert adapter.patterns[0]["transformation_rule"] != selected["transformation_rule"]
    event = next(e for e in result["events"] if e["kind"] == "pattern_selection")
    assert event["canonical_library_match"] is False


def test_no_initial_documents_skips_selector_api_and_keeps_author_default(upstream):
    adapter, client, retrieval = make_adapter(upstream, [REWRITE, ANSWER], docs=())
    result = adapter.run(QUESTION, "q1")
    assert result["selected_pattern"] == adapter.patterns[0]
    assert result["api_calls"] == 2 and len(client.calls) == 2 and len(retrieval) == 2
    assert client.calls[0]["prompt_version"] == bridge.PROMPT_VERSIONS["rewrite"]
    assert any(e["kind"] == "author_no_documents_default" for e in result["events"])


@pytest.mark.parametrize("stage", ["select", "rewrite"])
@pytest.mark.parametrize("error_class", [APIRequestError, RuntimeError])
def test_api_or_budget_error_is_never_hidden_by_author_broad_except(upstream, stage, error_class):
    adapter, client, retrieval = make_adapter(upstream, [])
    outputs = [] if stage == "select" else [json.dumps(adapter.patterns[0])]
    client.outputs = outputs + [error_class("secret-error-must-not-leak"), ANSWER]
    with pytest.raises(RuntimeError, match="transport/budget failure"):
        adapter.run(QUESTION, "q1")
    assert len(client.calls) == (1 if stage == "select" else 2)
    assert len(retrieval) == 1
    assert "secret-error-must-not-leak" not in json.dumps(adapter.events)
    assert any(e["kind"] == "reformer_api_failure" and e["stage"] == stage for e in adapter.events)


def test_client_block_reason_stops_before_rewrite_even_with_returned_response(upstream):
    adapter, client, retrieval = make_adapter(upstream, [])
    client.outputs = [json.dumps(adapter.patterns[0]), REWRITE, ANSWER]
    complete = client.complete

    def blocked_complete(*args, **kwargs):
        result = complete(*args, **kwargs)
        client.block_reason = "budget_exhausted"
        return result

    client.complete = blocked_complete
    with pytest.raises(RuntimeError, match="transport/budget failure"):
        adapter.run(QUESTION, "q1")
    assert len(client.calls) == 1 and len(retrieval) == 1


@pytest.mark.parametrize("content", ["[]", "null", "{}", '{"pattern_name": 3}'])
def test_valid_json_with_invalid_shape_is_failure_not_unlogged_repair(upstream, content):
    adapter, client, retrieval = make_adapter(upstream, [content, REWRITE, ANSWER])
    with pytest.raises(ValueError, match="named pattern"):
        adapter.run(QUESTION, "q1")
    assert len(client.calls) == 1 and len(retrieval) == 1


def test_empty_rewrite_stops_before_second_search_or_answer(upstream):
    adapter, client, retrieval = make_adapter(upstream, [])
    client.outputs = [json.dumps(adapter.patterns[0]), "   ", ANSWER]
    with pytest.raises(ValueError, match="empty author rewrite"):
        adapter.run(QUESTION, "q1")
    assert len(client.calls) == 2 and len(retrieval) == 1


def test_sequential_questions_do_not_mutate_prior_trace_or_carry_evidence(upstream):
    adapter, client, _, _, result = run_normal(upstream)
    snapshot = json.dumps(result, sort_keys=True)
    client.outputs = [json.dumps(adapter.patterns[0]), REWRITE, ANSWER]
    second = adapter.run("Unrelated second question?", "q2")
    assert json.dumps(result, sort_keys=True) == snapshot
    assert all(e["question_id"].startswith("q2") for e in second["events"])
    assert QUESTION not in client.calls[3]["messages"][1]["content"]


def test_external_callback_cannot_corrupt_prompts_patterns_or_saved_events(upstream):
    seen = []

    def corrupting_callback(event):
        seen.append(copy.deepcopy(event))
        event["kind"] = "external corruption"
        if "messages" in event:
            event["messages"][0]["content"] = "corrupted prompt"
        if "selected_pattern" in event:
            event["selected_pattern"]["transformation_rule"] = "corrupted rule"

    adapter, client, _, selected, result = run_normal(upstream, event_callback=corrupting_callback)
    assert seen
    assert all(e["kind"] != "external corruption" for e in result["events"])
    assert result["selected_pattern"] == selected
    assert result["selected_pattern"] == adapter.patterns[2]
    assert all("corrupted" not in json.dumps(call["messages"]) for call in client.calls)


@pytest.mark.parametrize(
    "documents, error",
    [
        (DOCS[:4], "at most k"),
        ((DOCS[0], DOCS[0]), "duplicate"),
        (({"doc_id": "fake"},), "AuthorDocument"),
    ],
)
def test_invalid_retrieval_result_fails_before_selection(upstream, documents, error):
    adapter, client, _ = make_adapter(upstream, [])
    adapter.index = lambda *_: documents
    with pytest.raises(ValueError, match=error):
        adapter.run(QUESTION, "q1")
    assert not client.calls


def test_changed_document_identity_between_two_searches_stops_before_reader(upstream):
    adapter, client, _ = make_adapter(upstream, [])
    client.outputs = [json.dumps(adapter.patterns[0]), REWRITE, ANSWER]
    responses = [DOCS[:3], (AuthorDocument(DOCS[0].doc_id, "Changed", "Changed."),)]
    adapter.index = lambda *_: responses.pop(0)
    with pytest.raises(ValueError, match="identity changed"):
        adapter.run(QUESTION, "q1")
    assert len(client.calls) == 2


def test_no_heavy_dependency_imports_or_execution_of_upstream_main(upstream, monkeypatch):
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {
            "torch",
            "transformers",
            "vllm",
            "peft",
            "pandas",
            "pyserini",
            "openai",
        }
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    assert run_normal(upstream)[-1]["answer"] == "northbridge"


@pytest.mark.parametrize("relative", list(bridge.PINNED_HASHES))
def test_any_author_source_or_library_hash_drift_is_rejected(upstream, tmp_path, relative):
    for name in bridge.PINNED_HASHES:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((upstream / name).read_bytes())
    changed = tmp_path / relative
    changed.write_bytes(changed.read_bytes() + b"\n ")
    with pytest.raises(ValueError, match="source hash mismatch"):
        bridge.load_author_components(tmp_path, SimpleNamespace(error=lambda _: None))


@pytest.mark.parametrize("question", ["", "  ", None, 42])
def test_invalid_question_makes_no_model_or_retrieval_calls(upstream, question):
    adapter, client, retrieval = make_adapter(upstream, [])
    with pytest.raises(ValueError, match="nonempty question"):
        adapter.run(question, "q1")
    assert not client.calls and not retrieval


def test_public_execution_interface_accepts_no_gold_argument():
    assert list(inspect.signature(bridge.ReFormeRAPI.run).parameters) == [
        "self",
        "question",
        "trace_id",
    ]


@pytest.mark.parametrize(
    "source, error",
    [
        ("class Other:\n    pass\n", "class missing"),
        ("class Reviewed:\n    pass\n", "method missing"),
        ("class Reviewed:\n    def action(self):\n        import os\n", "imports forbidden"),
    ],
)
def test_ast_loader_refuses_unreviewed_structure(source, error):
    with pytest.raises(ValueError, match=error):
        bridge._methods(source, Path("synthetic.py"), "Reviewed", ("action",), {})


def test_ast_loader_never_executes_unselected_top_level_or_constructor():
    source = (
        "raise AssertionError('top level executed')\n"
        "class Reviewed:\n"
        "    def __init__(self):\n"
        "        raise AssertionError('constructor executed')\n"
        "    def action(self):\n"
        "        return 'safe'\n"
    )
    method = bridge._methods(source, Path("synthetic.py"), "Reviewed", ("action",), {})
    assert method["action"](None) == "safe"
