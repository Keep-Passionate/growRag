"""Pinned author S2G control loop with an explicitly substituted API transport.

中文：不把无许可证的作者源码复制进本仓库。先核验四个文件，再仅加载已审阅的
函数与提示。原 main_batch 真正执行；Judge 的张量生成段改成 API 调用，其他
控制、解析、原句指针恢复与查询拼接继续执行作者代码。不是已训练 Judge 复现。
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import string
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

from .s2g_upstream_check import SOURCE_HASHES, UPSTREAM_COMMIT

BASELINE_ID = "S2G_AUTHOR_CONTROL_API_MIGRATION_V1"
PROMPT_VERSIONS = {
    stage: f"s2g-author-5d842a6-{stage}-api-v1" for stage in ("judge", "extract", "answer")
}
PINNED_HASHES = {
    **SOURCE_HASHES,
    "utils/text_processing.py": (
        "721abfcdbd7bab252b22c02522b718dd8dec9c87e32fe00fbaf857a4f16bf967"
    ),
    "utils/evaluation.py": ("9bab7d6e3ad5d1bcafcdd1119ffcc2d39ec91da3c37616c3145bbefd966ef80b"),
}
INFERENCE_FUNCTIONS = frozenset(
    {
        "get_selector_system_prompt",
        "get_answer_system_prompt",
        "safe_json_load",
        "split_wiki_sentences",
        "concat_raw_retrieved_docs",
        "bm25_search_batch",
        "build_suff_user_prompt",
        "call_suff_gate_batch",
        "build_query_from_missing",
        "should_force_first_retrieval",
        "format_missing_facts_for_selector",
        "concat_and_pick_sentences_batch",
        "merge_evidence_only",
        "append_evidence_context",
        "update_task_with_evidence",
        "main_batch",
    }
)
TEXT_FUNCTIONS = frozenset(
    {
        "_normalize_json_key",
        "get_json_value",
        "extract_gap_items",
        "extract_evidence_global_ids",
        "extract_final_answer_and_rationale",
        "extract_number",
    }
)
PROMPT_CONSTANTS = frozenset(
    {
        "formatting2",
        "force_answer_prompt",
        "SUFF_SYSTEM_PROMPT",
        "SELECTOR_SYSTEM_PROMPT",
        "TRIVIAQA_SELECTOR_SYSTEM_PROMPT",
        "TRIVIAQA_FORCE_ANSWER_PROMPT",
    }
)
TEXT_CONSTANTS = frozenset(
    {"NUMBER_AFTER_ANSWER_RE", "ANY_NUMBER_RE", "FINAL_ANSWER_AND_RATIONALE_RE"}
)


@dataclass(frozen=True, slots=True)
class AuthorDocument:
    """One source document, not a sentence pretending to be a document."""

    doc_id: str
    title: str
    text: str

    def __post_init__(self):
        for name in ("doc_id", "title", "text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"document {name} must be a nonempty string")


def _assigned_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _substitute_gate_transport(node):
    """Keep author prompt construction and parsing; replace only tensor generation.

    中文：这是显式 AST 迁移，不假装原 LoRA 运行。若作者结构变动则立即停止。
    """
    node = copy.deepcopy(node)
    starts = [i for i, item in enumerate(node.body) if _assigned_name(item) == "prompts"]
    ends = [i for i, item in enumerate(node.body) if _assigned_name(item) == "texts"]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError("author gate transport changed; manual re-audit required")
    replacement = ast.parse("texts = _api_gate_backend(messages_batch, max_new_tokens)").body[0]
    ast.copy_location(replacement, node.body[starts[0]])
    node.body[starts[0] : ends[0] + 1] = [replacement]
    return ast.fix_missing_locations(node)


def load_author_scope(upstream: Path) -> dict:
    """No upstream imports/top-level runtime: only SHA-locked allowlisted definitions."""
    sources = {}
    for relative, expected in PINNED_HASHES.items():
        raw = (upstream / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"author fingerprint changed: {relative}; execution refused")
        sources[relative] = raw
    scope = {
        "re": re,
        "json": json,
        "string": string,
        "_JSON_RE": re.compile(r"\{.*\}", re.S),
        "_CJK_RE": re.compile(r"[\u4e00-\u9fff]"),
        # 作者自带的无 pysbd 回退分支；不隐式安装或导入重型依赖。
        "_PYSBD_OK": False,
        "_PYSBD_SEGMENTER": None,
        "gate_model": None,
        "gate_tokenizer": None,
    }
    specifications = (
        ("utils/prompt_template.py", frozenset(), PROMPT_CONSTANTS),
        ("utils/text_processing.py", TEXT_FUNCTIONS, TEXT_CONSTANTS),
        ("utils/evaluation.py", {"normalize_answer", "exact_match_multiple"}, frozenset()),
        ("inference/inference_bm25.py", INFERENCE_FUNCTIONS, frozenset()),
    )
    for relative, functions, constants in specifications:
        path = upstream / relative
        tree = ast.parse(sources[relative], filename=str(path))
        selected, found = [], set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                found.add(node.name)
                if node.name == "call_suff_gate_batch":
                    node = _substitute_gate_transport(node)
                selected.append(node)
            elif _assigned_name(node) in constants:
                found.add(_assigned_name(node))
                selected.append(node)
        if found != functions | constants:
            raise ValueError(f"missing reviewed author definitions: {relative}")
        module = ast.Module(body=selected, type_ignores=[])
        if any(isinstance(n, ast.Import | ast.ImportFrom) for n in ast.walk(module)):
            raise ValueError("imports in selected author code are forbidden")
        exec(compile(module, str(path), "exec"), scope)
    return scope


class _CallbackSearcher:
    """Adapt ranked documents to the original author's searcher.search contract."""

    def __init__(self, retrieve, corpus):
        self.retrieve, self.corpus = retrieve, corpus

    def search(self, query, k):
        docs = tuple(self.retrieve(query, k))
        if len(docs) > k or any(not isinstance(doc, AuthorDocument) for doc in docs):
            raise ValueError("retriever must return at most k AuthorDocument objects")
        if len({doc.doc_id for doc in docs}) != len(docs):
            raise ValueError("retriever returned duplicate document IDs")
        for doc in docs:
            value = {"title": doc.title, "text": doc.text}
            if doc.doc_id in self.corpus and self.corpus[doc.doc_id] != value:
                raise ValueError("same document ID changed content within one query")
            self.corpus[doc.doc_id] = value
        return [SimpleNamespace(docid=doc.doc_id) for doc in docs]


class S2GAuthorAPI:
    """Serial, auditable API migration executing the pinned author's real loop.

    client.complete(messages, trace_id=..., prompt_version=...) must not retry.
    A budgeted client is the caller's responsibility; no keys are read here.
    This class deliberately accepts no gold answer or support annotations.
    """

    def __init__(
        self,
        upstream: Path,
        client,
        retrieve: Callable[[str, int], Sequence[AuthorDocument]],
        *,
        max_turns: int = 4,
        top_docs: int = 6,
        gap_profile: str = "code_default",
        remove_repeat_docs: bool = False,
        dataset_name: str = "hotpotqa",
        event_callback: Callable[[dict], None] | None = None,
    ):
        if type(max_turns) is not int or not 1 <= max_turns <= 4:
            raise ValueError("max_turns must be within [1, 4]")
        if type(top_docs) is not int or not 1 <= top_docs <= 6:
            raise ValueError("top_docs must be within [1, 6]")
        if gap_profile not in {"code_default", "paper_k1"}:
            raise ValueError("gap_profile must be code_default or paper_k1")
        if dataset_name not in {"hotpotqa", "triviaqa", "2wikimultihopqa"}:
            raise ValueError("unknown author dataset")
        if type(remove_repeat_docs) is not bool:
            raise ValueError("remove_repeat_docs must be boolean")
        self.upstream = Path(upstream).resolve(strict=True)
        self.client, self.retrieve = client, retrieve
        self.max_turns, self.top_docs = max_turns, top_docs
        self.gap_profile, self.dataset_name = gap_profile, dataset_name
        self.remove_repeat_docs, self.event_callback = remove_repeat_docs, event_callback
        self.scope = load_author_scope(self.upstream)
        self.author_functions = {
            name: value
            for name, value in self.scope.items()
            if callable(value) and name not in {"re"}
        }
        self._install_bridges()
        self._reset("unstarted")

    @property
    def provenance(self):
        return {
            "baseline_id": BASELINE_ID,
            "upstream_commit": UPSTREAM_COMMIT,
            "source_sha256": dict(PINNED_HASHES),
            "actual_main_batch_filename": self.author_functions["main_batch"].__code__.co_filename,
            "author_prompt_sha256": {
                key: hashlib.sha256(self.scope[key].encode()).hexdigest()
                for key in sorted(PROMPT_CONSTANTS)
            },
            "max_retrieval_rounds": self.max_turns,
            "max_model_calls": 2 * self.max_turns + 2,
            "top_documents": self.top_docs,
            "gap_profile": self.gap_profile,
            "remove_repeat_docs": self.remove_repeat_docs,
            "dedup_key": "document_id" if self.remove_repeat_docs else "disabled",
            "sentence_splitter": "author_regex_fallback",
            "trained_author_judge_used": False,
            "retrieval_backend": "injected_document_callback_not_author_lucene",
            "backend_generation_settings": (
                getattr(
                    self.client,
                    "generation_profile",
                    "author_per_stage_limits_via_complete_author",
                )
                if callable(getattr(self.client, "complete_author", None))
                else "frozen_client_config_not_per_stage_author_limits"
            ),
            "preserved_behaviors": [
                "Judge before first retrieval; a false verdict may rewrite the first query.",
                "Empty first context cannot stop when Judge says sufficient.",
                "Final judge at T; forced final answer even when evidence is insufficient.",
                "Original bool coercion, JSON salvage and sentence pointer parsing.",
                "No new stagnation/repetition/no-evidence stop rules added.",
            ],
            "gold_provided_to_author_loop": False,
        }

    def _reset(self, question_id):
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("nonempty question_id required")
        self.question_id = question_id
        self.events, self.sources, self.retrieved_documents = [], [], []
        self.executed_functions: set[str] = set()
        self.calls = 0
        self.retrieval_rounds = 0
        self._last_doc_ids = []
        self._raw_answer = ""
        self.scope["corpus"] = {}
        self.scope["searcher"] = _CallbackSearcher(self.retrieve, self.scope["corpus"])

    def _emit(self, kind, **payload):
        event = {
            "event_index": len(self.events),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "question_id": self.question_id,
            "kind": kind,
            **payload,
        }
        event = json.loads(json.dumps(event, ensure_ascii=False))
        self.events.append(event)
        if self.event_callback:
            self.event_callback(copy.deepcopy(event))

    def _api(self, messages, stage, author_generation):
        self.calls += 1
        if self.calls > 2 * self.max_turns + 2:
            raise RuntimeError("author API call ceiling exceeded")
        trace_id = f"{self.question_id}/s2g-author/{self.calls:02d}-{stage}"
        cap_resolver = getattr(self.client, "author_output_cap", None)
        actual_cap = (
            cap_resolver(PROMPT_VERSIONS[stage], author_generation["max_new_tokens"])
            if callable(cap_resolver)
            else None
        )
        self._emit(
            "api_request",
            stage=stage,
            trace_id=trace_id,
            prompt_version=PROMPT_VERSIONS[stage],
            messages=messages,
            author_generation_requested=author_generation,
            backend_max_output_tokens=actual_cap,
        )
        try:
            complete_author = getattr(self.client, "complete_author", None)
            if callable(complete_author):
                response = complete_author(
                    messages,
                    trace_id=trace_id,
                    prompt_version=PROMPT_VERSIONS[stage],
                    max_output_tokens=author_generation["max_new_tokens"],
                    temperature=author_generation.get("temperature", 0.0),
                    top_p=author_generation.get("top_p", 1.0),
                )
            else:
                response = self.client.complete(
                    messages, trace_id=trace_id, prompt_version=PROMPT_VERSIONS[stage]
                )
        except Exception as exc:
            # 不把 provider 异常的任意文本落盘，避免异常中回显密钥。
            self._emit("api_failure", stage=stage, trace_id=trace_id, error_type=type(exc).__name__)
            raise
        metadata = {
            name: getattr(response, name, None)
            for name in (
                "requested_model",
                "returned_model",
                "response_id",
                "request_id",
                "input_tokens",
                "output_tokens",
                "elapsed_seconds",
                "transport_source",
            )
        }
        metadata["audit_path"] = str(getattr(response, "audit_path", ""))
        self._emit(
            "api_response", stage=stage, trace_id=trace_id, content=response.content, **metadata
        )
        if stage == "answer":
            self._raw_answer = response.content
        return response.content

    def _install_bridges(self):
        # 给所有实际运行的作者函数留执行名单，而非只声明“借鉴”。
        for name, function in self.author_functions.items():

            @wraps(function)
            def tracked(*args, __name=name, __function=function, **kwargs):
                self.executed_functions.add(__name)
                return __function(*args, **kwargs)

            self.scope[name] = tracked

        def gate_backend(messages_batch, max_new_tokens):
            return [
                self._api(messages, "judge", {"max_new_tokens": max_new_tokens, "do_sample": False})
                for messages in messages_batch
            ]

        self.scope["_api_gate_backend"] = gate_backend
        original_gate = self.scope["call_suff_gate_batch"]

        def gate(*args, **kwargs):
            verdicts, gaps = original_gate(*args, **kwargs)
            self._emit("judge", evidence_contexts=args[1], verdicts=verdicts, gap_items=gaps)
            return verdicts, gaps

        self.scope["call_suff_gate_batch"] = gate

        def reasoner(system_prefixes, task_contents, **kwargs):
            prefixes = (
                [system_prefixes] * len(task_contents)
                if isinstance(system_prefixes, str)
                else system_prefixes
            )
            outputs = []
            for system, task in zip(prefixes, task_contents, strict=True):
                stage = (
                    "extract"
                    if system
                    in {
                        self.scope["SELECTOR_SYSTEM_PROMPT"],
                        self.scope["TRIVIAQA_SELECTOR_SYSTEM_PROMPT"],
                    }
                    else "answer"
                )
                outputs.append(
                    self._api(
                        [{"role": "system", "content": system}, {"role": "user", "content": task}],
                        stage,
                        {key: value for key, value in kwargs.items() if key != "gpt"},
                    )
                )
            return outputs

        self.scope["call_reasoner_batch"] = reasoner
        original_query = self.scope["build_query_from_missing"]

        def query(*args, **kwargs):
            if self.gap_profile == "paper_k1":
                kwargs["max_facts"] = 1
            value = original_query(*args, **kwargs)
            self._emit("query", query=value, gap_items=args[1], gap_profile=self.gap_profile)
            return value

        self.scope["build_query_from_missing"] = query
        original_search = self.scope["bm25_search_batch"]

        def search(*args, **kwargs):
            titles, texts, ids = original_search(*args, **kwargs)
            self.retrieval_rounds += 1
            self._last_doc_ids = ids
            docs = [
                [
                    asdict(AuthorDocument(doc_id, title, text))
                    for doc_id, title, text in zip(
                        batch_ids, batch_titles, batch_texts, strict=True
                    )
                    if doc_id
                ]
                for batch_ids, batch_titles, batch_texts in zip(ids, titles, texts, strict=True)
            ]
            self.retrieved_documents.extend(doc for batch in docs for doc in batch)
            self._emit("retrieval", round=self.retrieval_rounds, queries=args[0], documents=docs)
            return titles, texts, ids

        self.scope["bm25_search_batch"] = search
        original_extract = self.scope["concat_and_pick_sentences_batch"]

        def extract(**kwargs):
            pointers = original_extract(**kwargs)
            selected = []
            for batch_index, per_doc in enumerate(pointers):
                for doc_index, sentence_ids in enumerate(per_doc):
                    title = kwargs["titles_batch"][batch_index][doc_index]
                    text = kwargs["texts_batch"][batch_index][doc_index]
                    sentences = self.scope["split_wiki_sentences"](text)
                    for sid in sentence_ids:
                        selected.append(
                            {
                                "doc_id": self._last_doc_ids[batch_index][doc_index],
                                "title": title,
                                "sentence_id": sid,
                                "text": sentences[sid - 1],
                            }
                        )
            self.sources.extend(selected)
            self._emit(
                "extraction", round=self.retrieval_rounds, pointers=pointers, sources=selected
            )
            return pointers

        self.scope["concat_and_pick_sentences_batch"] = extract

    def run(self, question: str, question_id: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("nonempty question required")
        self._reset(question_id)
        self._emit("start", question=question, provenance=self.provenance)
        rows = self.scope["main_batch"](
            task_contents=[f"Question: {question}\n"],
            question_type="OEQ",
            ids=[question_id],
            # 原主循环的评分字段是死侧路：传空列表，无 gold 标签能进入模型。
            batch_gold_answers=[[]],
            gold_retrieved_docs=[[]],
            checkpoint_file=None,
            max_turns=self.max_turns,
            dataset_name=self.dataset_name,
            gpt=True,
            top_docs=self.top_docs,
            remove_repeat_docs=self.remove_repeat_docs,
            batch_questions=[question],
            write_csv=False,
        )
        final = rows[-1]
        stop = "sufficient" if final["Gate Output"] else "max_turns"
        # 空 gold 导致作者评分栏无意义，必须删掉，不能被误当成真实评测结果。
        rows = [
            {
                k: v
                for k, v in row.items()
                if k
                not in {"Gold Answer", "Correct Answer", "Gold Retrieved Docs", "Correct Retrieval"}
            }
            for row in rows
        ]
        self._emit(
            "finish",
            answer=final["Reasoner Answer"],
            stop_reason=stop,
            retrieval_rounds=self.retrieval_rounds,
            api_calls=self.calls,
        )
        return {
            "question_id": question_id,
            "question": question,
            "answer": final["Reasoner Answer"],
            "raw_answer": self._raw_answer,
            "stop_reason": stop,
            "retrieval_rounds": self.retrieval_rounds,
            "api_calls": self.calls,
            "evidence_context": final["Evidence Context So Far"],
            "sources": self.sources,
            "retrieved_documents": self.retrieved_documents,
            "author_rows": rows,
            "events": self.events,
            "executed_author_functions": sorted(self.executed_functions),
            "provenance": self.provenance,
        }

    def answer_once(self, question: str, evidence_context: str, question_id: str) -> dict:
        """One-call BASE with exactly the author's answer prompt and answer parser."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("nonempty question required")
        if not isinstance(evidence_context, str):
            raise ValueError("evidence_context must be text")
        self._reset(question_id)
        task = self.scope["update_task_with_evidence"](
            f"Question: {question}\n", question, evidence_context
        )
        outputs = self.scope["call_reasoner_batch"](
            system_prefixes=self.scope["get_answer_system_prompt"](self.dataset_name),
            task_contents=[task],
            gpt=True,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=128,
            do_sample=False,
        )
        answer, _ = self.scope["extract_final_answer_and_rationale"](outputs[0], "OEQ")
        answer = self.scope["normalize_answer"](answer)
        self._emit("finish", answer=answer, stop_reason="base_single_answer", api_calls=self.calls)
        return {
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "raw_answer": outputs[0],
            "stop_reason": "base_single_answer",
            "api_calls": self.calls,
            "evidence_context": evidence_context,
            "events": self.events,
            "executed_author_functions": sorted(self.executed_functions),
            "provenance": self.provenance,
        }
