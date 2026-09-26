"""Offline preflight of pinned author code, not an S2G model reproduction.

中文：只执行已读过且 SHA 固定的四个纯函数，不导入作者整套推理脚本，
不下载模型/数据，不读密钥，也不运行训练。作者文件保留在 ignored external。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from .pre_pilot import write_json

UPSTREAM_COMMIT = "5d842a67a0a99a7b545bbad0dc402ceaae0e5eff"
SOURCE_HASHES = {
    "inference/inference_bm25.py": (
        "29da5a909f26e7bdc09e37b3760941a0e4cac27d56494c8c298537445fee104e"
    ),
    "utils/prompt_template.py": "a642f36fa25db2597b9b072fdbb02dc5ec3925b8d149dfa8c2cfaac138341482",
}
UTILITY_NAMES = {
    "build_query_from_missing",
    "should_force_first_retrieval",
    "append_evidence_context",
    "build_suff_user_prompt",
}


def load_pinned_utilities(upstream: Path) -> dict:
    """Execute only reviewed, fingerprint-locked pure utility definitions."""
    for relative, expected in SOURCE_HASHES.items():
        source = (upstream / relative).read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise ValueError("upstream fingerprint changed; re-audit before execution")
    path = upstream / "inference/inference_bm25.py"
    tree = ast.parse(path.read_bytes(), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in UTILITY_NAMES]
    if {n.name for n in nodes} != UTILITY_NAMES:
        raise ValueError("missing audited utility")
    scope: dict = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return {name: scope[name] for name in UTILITY_NAMES}


def check_memory_source(*, official_split, candidate_ids, allowed_train_ids, protected_ids):
    """Fail closed for future memory builders; ID sets must come from audited manifests.

    This helper alone is not a project-wide firewall. The future author-code runner
    must call it before teacher labeling, card building, or router training.
    """
    if official_split != "train":
        raise ValueError("author evaluation/dev/test/unknown split cannot build memory")
    ids = tuple(candidate_ids)
    if not ids or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("explicit nonempty question IDs required")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source question IDs")
    if not set(ids) <= set(allowed_train_ids) or set(ids) & set(protected_ids):
        raise ValueError("not in audited train allowlist or overlaps protected evaluation")


def audit(upstream: Path) -> dict:
    functions = load_pinned_utilities(upstream)
    builder = functions["build_query_from_missing"]
    gaps = [
        {"target": "Northbridge", "slot": "founding year", "description": "unused"},
        {"target": "Eastbridge", "slot": "founding year", "description": "unused"},
    ]
    checks = {
        "empty_gap_preserves_question": builder("Q", [], dataset_name="hotpotqa") == "Q",
        "hotpot_code_default_uses_multiple_gaps": builder("Q", gaps, dataset_name="hotpotqa")
        == ("Q Northbridge founding year Eastbridge founding year"),
        "explicit_paper_k1": builder("Q", gaps, max_facts=1) == "Q Northbridge founding year",
        "empty_context_guard": functions["should_force_first_retrieval"](0, "", True),
        "duplicate_context_not_appended": functions["append_evidence_context"]("A", "A") == "A",
        "judge_input_question_and_context": "QUESTION:\nQ"
        in functions["build_suff_user_prompt"]("Q", "Evidence"),
    }
    if not all(checks.values()):
        raise ValueError("author utility contract failed; stop before model calls")
    sources = {}
    for path in sorted(upstream.rglob("*.py")):
        raw = path.read_bytes()
        compile(raw, str(path), "exec")  # syntax only: no imports, execution or pycache
        sources[path.relative_to(upstream).as_posix()] = hashlib.sha256(raw).hexdigest()
    dependencies = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "torch",
            "transformers",
            "peft",
            "accelerate",
            "datasets",
            "pyserini",
            "openai",
            "pysbd",
            "trl",
        )
    }
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_path": str(upstream),
        "file_sha256": sources,
        "python_files_syntax_checked": len(sources),
        "pure_function_checks": checks,
        "dependencies_present": dependencies,
        "java_found": shutil.which("java") is not None,
        "execution_kind": "local_author_utility_check",
        "model_api_requests": 0,
        "datasets_loaded": False,
        "training_performed": False,
        "full_s2g_inference_completed": False,
        "blocking_items": [
            "No verified released S2G-Judge LoRA checkpoint or cleaned training data.",
            "Need isolated inference dependencies, base model and retrieval resources.",
            "Qwen teacher/API variant requires a separately declared backend adaptation.",
        ],
        "scope_notice": "Passing pure utilities/syntax is not evidence of QA quality or readiness.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("preserve existing audit")
    result = audit(args.upstream.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "syntax_checked": result["python_files_syntax_checked"],
                "utility_checks_passed": len(result["pure_function_checks"]),
                "model_calls": 0,
                "full_inference": False,
                "report": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
