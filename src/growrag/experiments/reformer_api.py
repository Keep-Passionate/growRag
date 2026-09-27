"""SHA-locked ReFormeR public-method bridge, not a trained-selector reproduction.

中文：作者源码/提示/十模式只从 ignored external 读取，不公开再分发。
执行审阅过的选模式、应用模式方法；用普通 API 替换 vLLM。最后按论文
拼接原问和新问，在相同底座检索，用已有 S2G Answer 回答原问题。
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from .s2g_author_api import AuthorDocument, S2GAuthorAPI

UPSTREAM_COMMIT = "72e52450a922bc0051b5b39c3a6307186a555137"
BASELINE_ID = "REFORMER_PUBLIC_QWEN_HYBRID_V1"
PROMPT_VERSIONS = {
    stage: f"reformer-author-72e5245-{stage}-api-v1" for stage in ("select", "rewrite")
}
PINNED_HASHES = {
    "reformer/core/reformer.py": "1cacdc8dca84d863e4f559e2f078c777631e92f3b4cc7d6de6d3b76b31d592c5",
    "reformer/prompt_manager.py": (
        "7f154ca601470ce1918ebf2d72552fbef9d69bb21e78b6782fd94ac07e134d33"
    ),
    "reformer/prompts.json": "536bd3bd9596baf65c6b8f98f05e1320d59f221ae9b8d1799461374cd3fad652",
    "reformer/patterns/extracted_patterns.json": (
        "e2d0b1c7edc73c82776a1f01fc03f0ec5b06f286ac178ba5577cc0865d9bcaa5"
    ),
}


def _methods(source, path, class_name, allowed, scope):
    """Compile only fixed-hash, reviewed methods; never import/run upstream main."""
    tree = ast.parse(source, filename=str(path))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(classes) != 1:
        raise ValueError("reviewed upstream class missing")
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name in allowed]
    if {n.name for n in methods} != set(allowed):
        raise ValueError("reviewed upstream method missing")
    module = ast.Module(body=methods, type_ignores=[])
    if any(isinstance(n, ast.Import | ast.ImportFrom) for n in ast.walk(module)):
        raise ValueError("upstream imports forbidden")
    exec(compile(module, str(path), "exec"), scope)
    return {name: scope[name] for name in allowed}


def load_author_components(upstream: Path, logger):
    sources = {}
    for name, expected in PINNED_HASHES.items():
        raw = (upstream / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"ReFormeR source hash mismatch: {name}")
        sources[name] = raw
    scope = {"json": json, "List": list, "Dict": dict, "logging": logger}
    manager_methods = _methods(
        sources["reformer/prompt_manager.py"],
        upstream / "reformer/prompt_manager.py",
        "PromptManager",
        ("get_system_prompt", "get_user_prompt", "create_messages"),
        scope,
    )
    manager = SimpleNamespace(prompts=json.loads(sources["reformer/prompts.json"]))
    for name, method in manager_methods.items():
        setattr(manager, name, method.__get__(manager))
    methods = _methods(
        sources["reformer/core/reformer.py"],
        upstream / "reformer/core/reformer.py",
        "Reformulator",
        ("select_best_pattern", "apply_pattern"),
        scope,
    )
    patterns = json.loads(sources["reformer/patterns/extracted_patterns.json"])
    if not isinstance(patterns, list) or len(patterns) != 10:
        raise ValueError("expected fixed ten-pattern library")
    return manager, patterns, methods


class ReFormeRAPI:
    """Gold-free two-search arm; no history update or silent method improvement.

    Selector parse failure keeps the author's first-pattern fallback, explicitly
    logged. Transport/budget errors must still stop: upstream catches Exception,
    so the wrapper checks the failed-call flag before any next stage.
    """

    def __init__(self, reformer_snapshot, reader_snapshot, client, index, *, event_callback=None):
        self.client, self.index, self.event_callback = client, index, event_callback
        self.events, self.fallbacks = [], []
        self.seen_documents = {}
        self.stage, self.trace_id, self.backend_failed = "setup", "unstarted", False
        self.upstream = Path(reformer_snapshot).resolve(strict=True)
        manager, self.patterns, self.methods = load_author_components(
            self.upstream, SimpleNamespace(error=self._fallback)
        )
        self.author = SimpleNamespace(
            patterns=self.patterns,
            prompt_manager=manager,
            tokenizer=SimpleNamespace(apply_chat_template=self._chat_messages),
            llm=SimpleNamespace(generate=self._generate),
            sampling_params=SimpleNamespace(max_tokens=512, temperature=0),
        )
        self.reader = S2GAuthorAPI(
            reader_snapshot, client, index, event_callback=self._reader_event
        )

    @property
    def provenance(self):
        reader = self.reader.provenance
        return {
            "baseline_id": BASELINE_ID,
            "upstream_commit": UPSTREAM_COMMIT,
            "source_sha256": dict(PINNED_HASHES),
            "executed_author_methods": {
                name: method.__code__.co_filename for name, method in self.methods.items()
            },
            "pattern_count": len(self.patterns),
            "pattern_library_frozen": True,
            "trained_selector_reproduced": False,
            "hotpot_memory_built": False,
            "selector_context_top_k": 3,
            "rewrite_receives_documents": False,
            "retrieval_query_policy": "paper_hybrid_original_plus_rewrite",
            "public_core_emits": "rewrite_only; hybrid concatenation added from paper definition",
            "generation": {"max_tokens": 512, "temperature": 0, "top_p": 1},
            "upstream_code_defaults": {"max_tokens": 500, "temperature": 0.1},
            "paper_generation": {"max_tokens": 512, "temperature": 1.0},
            "reader": {
                "component": "answer_once_only; S2G Judge/Extractor/main_batch NOT executed",
                "upstream_commit": reader["upstream_commit"],
                "source_sha256": reader["source_sha256"],
                "author_prompt_sha256": reader["author_prompt_sha256"],
                "backend_max_output_tokens": 1024,
            },
            "gold_used_by_method": False,
            "licence_notice": "README MIT badge; full LICENSE missing; references not vendored",
        }

    def _emit(self, kind, **payload):
        event = {"kind": kind, "question_id": self.trace_id, **payload}
        self.events.append(copy.deepcopy(event))
        if self.event_callback:
            self.event_callback(copy.deepcopy(event))

    def _reader_event(self, event):
        self.events.append(copy.deepcopy(event))
        if self.event_callback:
            self.event_callback(copy.deepcopy(event))

    def _fallback(self, _message):
        # 上游任意异常文本不打印；日志记录发生在哪一步，原始模型输出另存。
        self.fallbacks.append(self.stage)
        self._emit("author_fallback", stage=self.stage)

    @staticmethod
    def _chat_messages(messages, *, tokenize, add_generation_prompt):
        if tokenize is not False or add_generation_prompt is not True:
            raise ValueError("unreviewed tokenizer invocation")
        # Preserve role/message boundaries; provider owns its actual chat template.
        return messages

    def _generate(self, *, prompts, sampling_params):
        if len(prompts) != 1 or sampling_params is not self.author.sampling_params:
            raise ValueError("serial frozen generation only")
        version = PROMPT_VERSIONS[self.stage]
        self._emit(
            "reformer_api_request", stage=self.stage, messages=prompts[0], prompt_version=version
        )
        try:
            response = self.client.complete(
                prompts[0], trace_id=f"{self.trace_id}/{self.stage}", prompt_version=version
            )
        except Exception:
            self.backend_failed = True
            self._emit("reformer_api_failure", stage=self.stage)
            raise
        self._emit(
            "reformer_api_response",
            stage=self.stage,
            content=response.content,
            returned_model=response.returned_model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            elapsed_seconds=response.elapsed_seconds,
            audit_path=str(response.audit_path),
            transport_source=response.transport_source,
        )
        return [SimpleNamespace(outputs=[SimpleNamespace(text=response.content)])]

    def _check_transport(self):
        if self.backend_failed or getattr(self.client, "block_reason", None):
            raise RuntimeError("ReFormeR transport/budget failure; no fallback continuation")

    def _retrieve(self, query, k, purpose, round_number):
        start = perf_counter()
        docs = tuple(self.index(query, k))
        if len(docs) > k or any(not isinstance(d, AuthorDocument) for d in docs):
            raise ValueError("retriever must return at most k AuthorDocument objects")
        if len({d.doc_id for d in docs}) != len(docs):
            raise ValueError("duplicate retrieved document identity")
        for doc in docs:
            if doc.doc_id in self.seen_documents and self.seen_documents[doc.doc_id] != doc:
                raise ValueError("document identity changed during one question")
            self.seen_documents[doc.doc_id] = doc
        elapsed = perf_counter() - start
        self._emit(
            "retrieval",
            round=round_number,
            query=query,
            purpose=purpose,
            documents=[asdict(d) for d in docs],
            elapsed_seconds=elapsed,
        )
        return docs, elapsed

    def run(self, question: str, trace_id: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("nonempty question required")
        self.events, self.fallbacks = [], []
        self.seen_documents = {}
        self.trace_id, self.backend_failed = trace_id, False
        initial, initial_time = self._retrieve(question, 3, "pattern_selection", 1)
        self.stage = "select"
        pattern = self.methods["select_best_pattern"](
            self.author, question, [d.text for d in initial]
        )
        self._check_transport()
        if not initial:
            self._emit("author_no_documents_default", stage="select")
        # 不用我们猜测的规则修复/覆盖作者返回内容；修改不匹配是诊断，不是新 gate。
        if not isinstance(pattern, dict) or not isinstance(pattern.get("pattern_name"), str):
            raise ValueError("author selection is not a named pattern object")
        canonical = next(
            (p for p in self.patterns if p["pattern_name"] == pattern["pattern_name"]), None
        )
        match = canonical == pattern
        self._emit("pattern_selection", selected_pattern=pattern, canonical_library_match=match)
        self.stage = "rewrite"
        rewrite = self.methods["apply_pattern"](self.author, question, pattern)
        self._check_transport()
        if not isinstance(rewrite, str) or not rewrite.strip():
            raise ValueError("empty author rewrite")
        final_query = question + " " + rewrite
        self._emit("query_rewrite", query=final_query, rewritten_query=rewrite)
        final, final_time = self._retrieve(final_query, 6, "answer_context", 2)
        context = self.reader.scope["concat_raw_retrieved_docs"](
            [d.title for d in final], [d.text for d in final]
        )
        result = self.reader.answer_once(question, context, trace_id + "/reader")
        result.update(
            question_id=trace_id,
            initial_selector_documents=[asdict(d) for d in initial],
            retrieved_documents=[asdict(d) for d in final],
            selected_pattern=pattern,
            canonical_library_match=match,
            rewritten_query=rewrite,
            retrieval_query=final_query,
            retrieval_rounds=2,
            retrieval_seconds=initial_time + final_time,
            author_fallback_stages=list(self.fallbacks),
            stop_reason="fixed_pattern_rewrite_then_answer",
            events=list(self.events),
            api_calls=sum(
                e["kind"] in {"reformer_api_request", "api_request"} for e in self.events
            ),
            provenance=self.provenance,
        )
        return result
