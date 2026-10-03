"""Static strict output shapes for the unchanged A1 locate/judge prompts.

只约束输出外壳，不导入传输、读取题目、复制expected或发API请求。locations
与checks必须区分；P0仍完全本地计算。这里只用简单type/array/enum/closed
object，避免把问题依赖的检查清单或答案写入schema。

仍由旧本地解析器验证：当前检查集合/顺序身份、kind/status/reason组合、引用
白名单及依赖、同一输入/窗口与审计绑定。正确字段和真实引用不证明语义蕴含，
更不证明动作可带来额外收益；这些仍需独立语义校准及后续结果实验。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .a1_conditions import _REASONS
from .a1_two_stage_observer import JUDGE_PROMPT_VERSION, LOCATOR_PROMPT_VERSION

SCHEMA_VERSION = "growrag-a1-two-stage-structured-output-v1"
MODEL_KINDS = ("T", "R", "P1")
STATUSES = tuple(dict.fromkeys(status for kind in MODEL_KINDS for status in _REASONS[kind]))
REASON_CODES = tuple(
    sorted(
        {
            reason
            for kind in MODEL_KINDS
            for reasons in _REASONS[kind].values()
            for reason in reasons
        }
    )
)


def _object(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items):
    return {"type": "array", "items": items}


def _string():
    return {"type": "string"}


def _enum(values):
    return {"type": "string", "enum": list(values)}


_LOCATOR = _object(
    {
        "locations": _array(
            _object(
                {
                    "kind": _enum(MODEL_KINDS),
                    "slot": _string(),
                    "candidate_ref_ids": _array(_string()),
                }
            )
        )
    }
)
_JUDGE = _object(
    {
        "checks": _array(
            _object(
                {
                    "kind": _enum(MODEL_KINDS),
                    "slot": _string(),
                    "status": _enum(STATUSES),
                    # Union不编码交叉字段条件：旧解析器仍检查各kind/status对应reason。
                    "reason": _enum(REASON_CODES),
                    "ref_ids": _array(_string()),
                }
            )
        )
    }
)
_REGISTRY = {
    LOCATOR_PROMPT_VERSION: ("growrag_a1_two_stage_locator_v1", _LOCATOR),
    JUDGE_PROMPT_VERSION: ("growrag_a1_two_stage_judge_v1", _JUDGE),
}


def two_stage_schema_registry():
    """Return isolated (wire name, schema) pairs for only the exact two versions."""
    return deepcopy(_REGISTRY)


def response_format_for(prompt_version):
    """No aliases or JSON-object fallback; unknown versions fail before transport."""
    if type(prompt_version) is not str or prompt_version not in _REGISTRY:
        raise ValueError("strict A1 schema has no reviewed exact prompt version")
    name, schema = _REGISTRY[prompt_version]
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": deepcopy(schema)},
    }


def schema_fingerprint(prompt_version):
    """Hash the complete provider response_format including name and strict flag."""
    serialized = json.dumps(
        response_format_for(prompt_version),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def schema_manifest():
    """Safe freeze metadata, not a provider-compatibility or semantic certificate."""
    return {
        "schema_version": SCHEMA_VERSION,
        "schemas": {
            version: {
                "sha256": schema_fingerprint(version),
                "response_format": response_format_for(version),
            }
            for version in sorted(_REGISTRY)
        },
    }
