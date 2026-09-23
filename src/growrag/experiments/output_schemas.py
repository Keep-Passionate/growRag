"""Reviewed output schemas keyed by EXACT prompt versions; no fallback.

中文：JSON Object 只约束合法 JSON；本模块进一步约束字段和类型。
它不能证明证据充分或回答正确，原有解析器的引用、长度和语义检查继续保留。
Only basic types, arrays, enums and closed objects from the documented provider
interface are used. Local parsers enforce bounds and cross-field constraints.

Provider reference (reviewed 2026-09-23):
https://help.aliyun.com/zh/model-studio/qwen-structured-output
No controller imports: the transport imports this module, so importing the
controller here would create a circular dependency. Tests check version alignment.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

REGISTRY_VERSION = "growrag-output-schemas-v1"


def _object(properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items: dict) -> dict:
    return {"type": "array", "items": items}


def _text() -> dict:
    return {"type": "string"}


_ASSESSMENT = _object(
    {
        "requirements": _array(
            _object(
                {
                    "description": _text(),
                    "status": {
                        "type": "string",
                        "enum": ["supported", "missing", "conflicted", "unknown"],
                    },
                    "evidence_ids": _array(_text()),
                }
            )
        ),
        "sufficient": {"type": "boolean"},
        "useful_gain": {"type": ["boolean", "null"]},
        "gap": _text(),
        "next_intent": _text(),
        "reason": _text(),
    }
)
_ROUTE = _object(
    {
        "action": {"type": "string", "enum": ["BASE", "FRESH", "REUSE"]},
        "memory_id": {"type": ["string", "null"]},
        "reason": _text(),
        "condition_checks": _array(
            _object(
                {
                    "condition": _text(),
                    "status": {
                        "type": "string",
                        "enum": ["supported", "conflicted", "unknown"],
                    },
                    "critical": {"type": "boolean"},
                    "reason": _text(),
                }
            )
        ),
    }
)
_QUERY = _object({"query": _text()})
_ANSWER = _object({"answer": _text(), "cited_evidence_ids": _array(_text())})

_REGISTRY = {
    "growrag-evidence-requirements-v2": ("growrag_evidence_assessment_v2", _ASSESSMENT),
    "growrag-three-way-route-v2": ("growrag_three_way_route_v2", _ROUTE),
    "growrag-gap-query-v2": ("growrag_gap_query_v2", _QUERY),
    "growrag-short-supported-answer-v2": ("growrag_short_supported_answer_v2", _ANSWER),
}


def response_format_for(prompt_version: str) -> dict:
    """Return an isolated strict provider format; unknown versions fail closed."""
    if not isinstance(prompt_version, str) or prompt_version not in _REGISTRY:
        raise ValueError("strict JSON Schema has no reviewed schema for this prompt version")
    name, schema = _REGISTRY[prompt_version]
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": deepcopy(schema)},
    }


def schema_fingerprint(prompt_version: str) -> str:
    """Hash the full wire-format schema, including name and strictness."""
    serialized = json.dumps(
        response_format_for(prompt_version),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def registry_manifest() -> dict:
    """Safe frozen-plan projection; returned dictionaries do not mutate registry."""
    return {
        "registry_version": REGISTRY_VERSION,
        "schemas": {
            version: {
                "sha256": schema_fingerprint(version),
                "response_format": response_format_for(version),
            }
            for version in sorted(_REGISTRY)
        },
    }
