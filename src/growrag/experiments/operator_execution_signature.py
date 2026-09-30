"""Pin the actual operator method independently of resumable runner plumbing.

中文：只改变断点续跑的外壳不能偷偷改变方法；提示、模型、检索、窗口、
算子语法、预算、生成参数一并固定。签名不含密钥或任何数据标签。
"""

from pathlib import Path

from .fresh_dev_manifest import _sha
from .representation_runner import fingerprint
from .run_pilot import PILOT_MODEL

SCHEMA = "growrag-operator-execution-signature-v1"
METHOD_FILES = (
    "src/growrag/macro_operators.py",
    "src/growrag/operator_bank.py",
    "src/growrag/operator_loop.py",
    "src/growrag/experiments/operator_model.py",
    "src/growrag/experiments/protocol.py",
    "src/growrag/experiments/shared_s2g_corpus.py",
    "src/growrag/experiments/lexical_retriever.py",
    "src/growrag/experiments/api_client.py",
    "src/growrag/experiments/budget.py",
)


def execution_configuration():
    """A fresh dict: callers cannot mutate a shared experiment default."""
    return {
        "model": PILOT_MODEL,
        "max_output_tokens": 2048,
        "output_limit_parameter": "max_tokens",
        "enable_thinking": False,
        "temperature": 0,
        "top_p": 1,
        "json_object_mode": True,
        "retrieval_budget": 3,
        "base_retrieval_budget": 1,
        "max_decisions": 2,
        "top_k": 6,
        "max_prompt_bytes": 30000,
        "bm25_k1": 1.2,
        "bm25_b": 0.75,
        "runtime_memory_updates": False,
    }


def execution_signature(project_root):
    root = Path(project_root).resolve(strict=True)
    files = {}
    for name in METHOD_FILES:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("method source escapes project")
        files[name] = _sha(path)
    value = {"schema": SCHEMA, "configuration": execution_configuration(), "files": files}
    return {**value, "sha256": fingerprint(value)}


def validate_execution_signature(value):
    """Validate a historical recorded signature, not today's possibly changed code."""
    if type(value) is not dict or set(value) != {"schema", "configuration", "files", "sha256"}:
        raise ValueError("missing execution signature")
    body = {key: value[key] for key in ("schema", "configuration", "files")}
    if (
        value["schema"] != SCHEMA
        or type(value["configuration"]) is not dict
        or value["configuration"] != execution_configuration()
        or type(value["files"]) is not dict
        or set(value["files"]) != set(METHOD_FILES)
        or any(
            type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            for digest in value["files"].values()
        )
        or value["sha256"] != fingerprint(body)
    ):
        raise ValueError("invalid execution signature")
    return value["sha256"]
