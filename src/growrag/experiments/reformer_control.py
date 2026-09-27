"""Isolated, shared-selection control for ReFormeR's example field.

中文：本实验不是增加历史记忆。先从同一冻结模式库选一次 ID，再把完全
相同的规则分别交给两个执行臂，唯一人为开关是是否附带该模式的原始示例。
原基线文件和作者 apply_pattern 都不修改；选择器的输出合同单独版本化。
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict

from .reformer_api import ReFormeRAPI
from .s2g_author_api import AuthorDocument

CONTROL_ID = "REFORMER_SHARED_ID_EXAMPLES_CONTROL_V1"
SELECT_PROMPT_VERSION = "reformer-author-72e5245-select-id-control-v1"
_AUTHOR_OUTPUT_INSTRUCTION = "Return only the JSON of the selected pattern."
_ID_OUTPUT_INSTRUCTION = (
    'Return only one JSON object with exactly one key: {"pattern_id": N}. '
    "N must be an integer from 0 through 9, identifying the selected pattern's "
    "zero-based position in Available Patterns. Do not return the pattern object, "
    "any other keys, markdown, or explanatory text."
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate selection JSON key")
        result[key] = value
    return result


def parse_pattern_id(content: str) -> int:
    """No JSON salvage/default: malformed IDs must remain visible failures."""
    if not isinstance(content, str):
        raise ValueError("selection must be strict pattern_id JSON")
    try:
        parsed = json.loads(content, object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise ValueError("selection must be strict pattern_id JSON") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"pattern_id"}:
        raise ValueError("selection must contain only pattern_id")
    pattern_id = parsed["pattern_id"]
    # bool 是 int 的子类，故必须使用 type(...) is int，避免 True 被当作 ID 1。
    if type(pattern_id) is not int or not 0 <= pattern_id < 10:
        raise ValueError("pattern_id must be an integer from 0 through 9")
    return pattern_id


def selection_digest(selection: dict) -> str:
    """Accidental-mutation check, not a cryptographic provider attestation."""
    payload = {key: value for key, value in selection.items() if key != "selection_sha256"}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ReFormeRControlAPI(ReFormeRAPI):
    """One shared selection, two independently traced rewrite/answer executions.

    Constructor inherits SHA-locked load_author_components: no heavyweight
    upstream imports. Returned data and callback events never alias internal
    canonical patterns. Resources in run_selected are arm-exclusive; initial
    retrieval/selection are charged once in select, not once per experimental arm.
    """

    def __init__(self, reformer_snapshot, reader_snapshot, client, index, *, event_callback=None):
        super().__init__(
            reformer_snapshot, reader_snapshot, client, index, event_callback=event_callback
        )
        self._canonical_patterns = copy.deepcopy(self.patterns)

    @property
    def provenance(self):
        result = super().provenance
        result.update(
            baseline_id=CONTROL_ID,
            selection_method="strict_id_from_frozen_zero_based_library",
            selection_prompt_version=SELECT_PROMPT_VERSION,
            author_selection_executed=False,
            executed_author_methods={
                "apply_pattern": self.methods["apply_pattern"].__code__.co_filename
            },
            experimental_contrast="same_selected_rule_with_vs_without_original_examples",
            examples_are_new_memory=False,
            selection_shared_across_arms=True,
            selection_failure_policy="stop_without_fallback",
            arm_resource_policy="exclusive_rewrite_answer_and_final_retrieval_only",
        )
        return result

    def _reset_control(self, question, trace_id):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("nonempty question required")
        if not isinstance(trace_id, str) or not trace_id.strip():
            raise ValueError("nonempty trace_id required")
        self.events, self.fallbacks, self.seen_documents = [], [], {}
        self.trace_id, self.backend_failed, self.stage = trace_id, False, "setup"
        self.reader._reset(trace_id + "/reader")
        self._check_transport()

    def _selection_messages(self, question, documents):
        messages = self.author.prompt_manager.create_messages(
            prompt_type="pattern_selection",
            system_prompt_type="pattern_selection",
            query=question,
            documents_text="\n\n".join(doc.text for doc in documents),
            patterns_json=json.dumps(self._canonical_patterns, indent=2),
        )
        # 保留冻结作者模板的任务、问题、文档和完整模式库，仅替换输出合同。
        # 必须只命中模板结尾一次；不对 query/document 中同名文本做替换。
        if not messages[-1]["content"].endswith(_AUTHOR_OUTPUT_INSTRUCTION):
            raise ValueError("pinned selection output instruction changed")
        messages[-1]["content"] = (
            messages[-1]["content"][: -len(_AUTHOR_OUTPUT_INSTRUCTION)] + _ID_OUTPUT_INSTRUCTION
        )
        return messages

    def _selection_response(self, messages):
        self.stage = "select_id"
        self._emit(
            "reformer_api_request",
            stage=self.stage,
            messages=messages,
            prompt_version=SELECT_PROMPT_VERSION,
        )
        try:
            response = self.client.complete(
                messages,
                trace_id=f"{self.trace_id}/select_id",
                prompt_version=SELECT_PROMPT_VERSION,
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
        self._check_transport()
        return response.content

    def select(self, question: str, trace_id: str) -> dict:
        self._reset_control(question, trace_id)
        initial, initial_time = self._retrieve(question, 3, "shared_pattern_selection", 1)
        content = self._selection_response(self._selection_messages(question, initial))
        try:
            pattern_id = parse_pattern_id(content)
        except ValueError:
            self._emit("selection_validation_failure", stage=self.stage)
            raise
        pattern = copy.deepcopy(self._canonical_patterns[pattern_id])
        self._emit(
            "pattern_selection",
            pattern_id=pattern_id,
            selected_pattern=pattern,
            canonical_library_match=True,
        )
        provenance = self.provenance
        provenance["executed_author_methods"] = {}
        result = {
            "question": question,
            "question_id": trace_id,
            "pattern_id": pattern_id,
            "selected_pattern": pattern,
            "initial_selector_documents": [asdict(doc) for doc in initial],
            "retrieval_seconds": initial_time,
            "retrieval_rounds": 1,
            "api_calls": 1,
            "events": copy.deepcopy(self.events),
            "provenance": provenance,
        }
        result["selection_sha256"] = selection_digest(result)
        return copy.deepcopy(result)

    def _validate_selection(self, question, selection):
        if not isinstance(selection, dict) or selection.get("question") != question:
            raise ValueError("shared selection must belong to the same question")
        if selection.get("selection_sha256") != selection_digest(selection):
            raise ValueError("shared selection snapshot was modified")
        pattern_id = selection.get("pattern_id")
        if type(pattern_id) is not int or not 0 <= pattern_id < 10:
            raise ValueError("invalid shared selection pattern_id")
        if selection.get("selected_pattern") != self._canonical_patterns[pattern_id]:
            raise ValueError("shared selection is not the canonical frozen pattern")
        provenance = self.provenance
        provenance["executed_author_methods"] = {}
        if selection.get("provenance") != provenance:
            raise ValueError("shared selection provenance mismatch")
        if (
            not isinstance(selection.get("question_id"), str)
            or not selection["question_id"].strip()
        ):
            raise ValueError("shared selection requires a trace identity")
        raw_docs = selection.get("initial_selector_documents")
        if not isinstance(raw_docs, list) or len(raw_docs) > 3:
            raise ValueError("shared selection must contain at most three documents")
        documents = []
        for raw in raw_docs:
            if not isinstance(raw, dict) or set(raw) != {"doc_id", "title", "text"}:
                raise ValueError("invalid shared selection document")
            documents.append(AuthorDocument(**raw))
        if len({doc.doc_id for doc in documents}) != len(documents):
            raise ValueError("duplicate shared selection document identity")
        self.seen_documents = {doc.doc_id: doc for doc in documents}
        return pattern_id, copy.deepcopy(self._canonical_patterns[pattern_id]), documents

    def run_selected(
        self, question: str, trace_id: str, selection: dict, *, include_examples: bool
    ) -> dict:
        self._reset_control(question, trace_id)
        if type(include_examples) is not bool:
            raise ValueError("include_examples must be boolean")
        pattern_id, canonical, initial = self._validate_selection(question, selection)
        pattern = copy.deepcopy(canonical)
        # 消融只清空 examples；名称、描述、规则及被选中的 ID 全部不变。
        if not include_examples:
            pattern["examples"] = []
        self._emit(
            "shared_selection_loaded",
            shared_selection_question_id=selection["question_id"],
            shared_selection_sha256=selection["selection_sha256"],
            pattern_id=pattern_id,
            selected_pattern=pattern,
            include_examples=include_examples,
            example_count=len(pattern["examples"]),
        )
        self.stage = "rewrite"
        rewrite = self.methods["apply_pattern"](self.author, question, pattern)
        self._check_transport()
        if self.fallbacks:
            raise RuntimeError("author rewrite fallback; control arm cannot continue")
        if not isinstance(rewrite, str) or not rewrite.strip():
            raise ValueError("empty author rewrite")
        final_query = question + " " + rewrite
        self._emit("query_rewrite", query=final_query, rewritten_query=rewrite)
        final, final_time = self._retrieve(final_query, 6, "answer_context", 2)
        context = self.reader.scope["concat_raw_retrieved_docs"](
            [doc.title for doc in final], [doc.text for doc in final]
        )
        result = self.reader.answer_once(question, context, trace_id + "/reader")
        result.update(
            question_id=trace_id,
            initial_selector_documents=[asdict(doc) for doc in initial],
            retrieved_documents=[asdict(doc) for doc in final],
            pattern_id=pattern_id,
            selected_pattern_id=pattern_id,
            selected_pattern=pattern,
            canonical_library_match=pattern == canonical,
            canonical_action_match=True,
            include_examples=include_examples,
            example_count=len(pattern["examples"]),
            shared_selection_question_id=selection["question_id"],
            shared_selection_sha256=selection["selection_sha256"],
            rewritten_query=rewrite,
            retrieval_query=final_query,
            retrieval_rounds=1,
            logical_retrieval_rounds=2,
            retrieval_seconds=final_time,
            author_fallback_stages=list(self.fallbacks),
            stop_reason="shared_pattern_rewrite_then_answer",
            events=copy.deepcopy(self.events),
            api_calls=sum(
                event["kind"] in {"reformer_api_request", "api_request"} for event in self.events
            ),
            provenance=self.provenance,
        )
        return copy.deepcopy(result)
