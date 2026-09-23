"""Same-question execution sharing for paired experiments, not learned memory.

中文：相同 Reader / judge 输入只执行一次，避免把模型随机差异算成经验收益。
命中缓存是本地重放：新增 API 用量为零，原响应的来源和用量另存供审计及
独立运行成本估计。缓存仅存在于当前题的内存中，不跨题、不保存失败响应。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

from . import controller
from .controller import APIEvidenceAssessor, EvidenceAssessment
from .experiments.protocol import (
    Answer,
    CallResult,
    Evidence,
    ExecutionKind,
    RuntimeQuestion,
    Usage,
)
from .outer_loop import Feedback, LoopState, RagReply

CACHE_VERSION = "growrag-paired-execution-v1"


def _configuration(delegate) -> dict:
    """Hash response-affecting options only; never inspect keys or environment."""
    client = delegate.client
    config = client.config
    fields = (
        "base_url",
        "model",
        "max_output_tokens",
        "output_limit_parameter",
        "enable_thinking",
        "temperature",
        "json_object_mode",
    )
    result = {name: getattr(config, name, None) for name in fields}
    if any(not isinstance(result[name], str) or not result[name] for name in ("base_url", "model")):
        raise ValueError("paired execution requires an explicit endpoint and model")
    result["transport_source"] = client.transport_source
    result["adapter"] = f"{type(delegate).__module__}.{type(delegate).__qualname__}"
    return result


def _hash(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _metadata(result: CallResult) -> dict:
    return {
        "usage": asdict(result.usage),
        "provider": result.provider,
        "model": result.model,
        "request_id": result.request_id,
        "audit_path": result.audit_path,
        "transport_source": result.transport_source,
    }


def _replay(result: CallResult) -> CallResult:
    # The source audit identifies the original response, not a newly paid request.
    return replace(
        result,
        usage=Usage(0, 0, 0),
        provider="local-paired-replay",
        transport_source="local_compute",
    )


@dataclass(frozen=True)
class _Entry:
    result: CallResult
    source_record_index: int
    assessment: EvidenceAssessment | None = None
    assessor_record: dict | None = None


class PairedExecutionCache:
    """One cache per exact RuntimeQuestion; sequential paired arms share it.

    This is an experimental variance control, not a claim that deployed API
    outputs are deterministic. It is deliberately neither persistent nor thread
    safe. ``records`` contains JSON-safe provenance for BOTH misses and hits.
    Sum actual client calls for billed cost; add a hit's ``source_usage`` only
    when estimating what that arm would cost if independently executed.
    """

    def __init__(self, question: RuntimeQuestion) -> None:
        if not isinstance(question, RuntimeQuestion):
            raise TypeError("paired execution accepts only a gold-free RuntimeQuestion")
        self.question = question
        self.records: list[dict] = []
        self._entries: dict[str, _Entry] = {}

    def _check_question(self, question: RuntimeQuestion) -> None:
        if not isinstance(question, RuntimeQuestion) or question != self.question:
            raise ValueError("paired execution cannot share across different questions")

    def _key(self, component: str, delegate, payload: dict, prompt: dict) -> str:
        return _hash(
            {
                "schema": CACHE_VERSION,
                "question": asdict(self.question),
                "component": component,
                "configuration": _configuration(delegate),
                "prompt": prompt,
                "payload": payload,
            }
        )

    def _record(self, component: str, key: str, entry: _Entry, *, hit: bool, cacheable: bool):
        source = entry.result
        self.records.append(
            {
                "schema": CACHE_VERSION,
                "component": component,
                "cache_key": key,
                "cache_hit": hit,
                "cacheable": cacheable,
                "status": "cache_replay" if hit else "original_execution",
                "source_record_index": entry.source_record_index,
                "source_usage": asdict(source.usage),
                "source_audit_path": source.audit_path,
                "source_request_id": source.request_id,
                "source_provider": source.provider,
                "source_model": source.model,
                "source_transport_source": source.transport_source,
                "usage": asdict(Usage(0, 0, 0) if hit else source.usage),
                "transport_source": "local_compute" if hit else source.transport_source,
            }
        )

    def _store(self, component, key, result, delegate, *, assessment=None, assessor_record=None):
        if not isinstance(result, CallResult) or not isinstance(result.usage, Usage):
            raise TypeError("paired execution requires a typed, audited CallResult")
        expected = "live_api" if delegate.execution_kind is ExecutionKind.REAL else "mock"
        if result.transport_source != expected:
            raise ValueError("paired execution source provenance mismatch")
        # Unknown paid usage/model identity must not be hidden by a cache hit.
        # Mock outputs may lack token counts; they still stay labelled mock.
        cacheable = expected == "mock" or (
            result.model == delegate.client.config.model
            and result.audit_path is not None
            and result.usage.api_requests == 1
            and result.usage.input_tokens is not None
            and result.usage.output_tokens is not None
        )
        entry = _Entry(result, len(self.records), assessment, deepcopy(assessor_record))
        if cacheable:
            self._entries[key] = entry
        self._record(component, key, entry, hit=False, cacheable=cacheable)

    def reader(self, delegate, *, prompt_version: str, prompt_text: str):
        """Wrap an API Reader with its actual prompt text and version explicitly."""
        if any(not isinstance(value, str) or not value for value in (prompt_version, prompt_text)):
            raise ValueError("Reader prompt text and version must be explicit")
        _configuration(delegate)
        return _SharedReader(self, delegate, prompt_version, prompt_text)

    def assessor(self, delegate: APIEvidenceAssessor) -> APIEvidenceAssessor:
        # The key exactly mirrors this implementation's payload and parser.
        # An unknown subclass may have different hidden inputs: reject it.
        if type(delegate) is not APIEvidenceAssessor:
            raise TypeError("paired assessor requires the standard APIEvidenceAssessor")
        return _SharedAssessor(self, delegate)


class _SharedReader:
    def __init__(self, cache, delegate, version, prompt):
        self.cache, self.delegate = cache, delegate
        self.execution_kind = delegate.execution_kind
        self.prompt = {"version": version, "text": prompt}

    def answer(
        self, question: RuntimeQuestion, evidence: tuple[Evidence, ...]
    ) -> CallResult[Answer]:
        self.cache._check_question(question)
        if not isinstance(evidence, tuple) or not all(
            isinstance(item, Evidence) for item in evidence
        ):
            raise TypeError("Reader evidence must be an immutable tuple")
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise ValueError("duplicate Reader evidence ID")
        payload = {"original_question": question.text, "evidence": [asdict(e) for e in evidence]}
        key = self.cache._key("reader", self.delegate, payload, self.prompt)
        if entry := self.cache._entries.get(key):
            self.cache._record("reader", key, entry, hit=True, cacheable=True)
            return _replay(entry.result)
        result = self.delegate.answer(question, evidence)
        if not isinstance(result, CallResult) or not isinstance(result.value, Answer):
            raise TypeError("Reader must return a typed Answer in a CallResult")
        if not set(result.value.cited_evidence_ids) <= {item.evidence_id for item in evidence}:
            raise ValueError("Reader result cites evidence outside its context")
        self.cache._store("reader", key, result, self.delegate)
        return result


class _SharedAssessor(APIEvidenceAssessor):
    """Subclass preserves the strict controller/generator assessor type contract."""

    def __init__(self, cache: PairedExecutionCache, delegate: APIEvidenceAssessor):
        super().__init__(delegate.client)
        self.cache, self.delegate = cache, delegate
        self.records = delegate.records
        self.latest, self.latest_question = delegate.latest, delegate.latest_question

    def seed(self, question: RuntimeQuestion, assessment: EvidenceAssessment) -> None:
        self.cache._check_question(question)
        super().seed(question, assessment)
        self.delegate.seed(question, assessment)

    def __call__(self, state: LoopState, reply: RagReply) -> CallResult[Feedback]:
        if not isinstance(state, LoopState) or not isinstance(reply, RagReply):
            raise TypeError("assessment accepts typed runtime state and reply")
        self.cache._check_question(state.question)
        self.latest, self.latest_question = None, state.question
        payload = {
            "original_question": state.question.text,
            "reader_context": [asdict(e) for e in reply.evidence or ()],
            "current_answer": asdict(reply.answer),
            "previous_evidence_ids": [e.evidence_id for e in state.observed_evidence],
            "has_previous_round": bool(state.rounds),
        }
        prompt = {
            "version": controller.ASSESS_PROMPT_VERSION,
            "text": controller.ASSESS_PROMPT,
            "parser_version": controller.ASSESS_PARSER_VERSION,
        }
        key = self.cache._key("assessor", self.delegate, payload, prompt)
        if entry := self.cache._entries.get(key):
            self.seed(state.question, entry.assessment)
            result = _replay(entry.result)
            record = deepcopy(entry.assessor_record)
            record.update(
                status="cache_replay",
                metadata=_metadata(result),
                replay={
                    "schema": CACHE_VERSION,
                    "cache_key": key,
                    "source_record_index": entry.source_record_index,
                    "source_metadata": _metadata(entry.result),
                    "response_origin": "original_response_not_a_new_api_call",
                },
            )
            self.records.append(record)
            self.cache._record("assessor", key, entry, hit=True, cacheable=True)
            return result
        result = self.delegate(state, reply)
        if (
            not isinstance(result, CallResult)
            or not isinstance(result.value, Feedback)
            or not isinstance(self.delegate.latest, EvidenceAssessment)
            or self.delegate.latest_question != state.question
            or not self.records
            or self.records[-1].get("status") != "ok"
        ):
            raise ValueError("only a validated assessment may enter the paired cache")
        self.latest = self.delegate.latest
        self.cache._store(
            "assessor",
            key,
            result,
            self.delegate,
            assessment=self.latest,
            assessor_record=self.records[-1],
        )
        return result
