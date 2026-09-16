"""Read-only prompt registry and zero-API paper-appendix export.

中文导读：这里记录实际代码使用的 system prompt，而不是事后重新措辞的版本。
哈希用于发现提示词变化，不代表提示词有效。原始 RRR 与 MINIMAL 是提示词消融；
原始 RRR 与 ANCHOR 是拼接方式消融，不能混成同一个实验变量。

Run ``python -m growrag.experiments.prompt_registry --output NEW_DIRECTORY``.
This command reads no API configuration, calls no model, accepts no dataset and
never overwrites an earlier export. Model settings still belong in each run's
manifest; a prompt hash alone is insufficient to reproduce an experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

from growrag.query_actions import ACTION_PROMPT

from .fresh_baselines import (
    ALL_BASELINE_SPECS,
    BASELINE_SPECS,
    QUERY2DOC_PROMPT,
    RRR_MINIMAL_PROMPT,
    RRR_PROMPT,
)
from .llm_adapters import READER_PROMPT, READER_PROMPT_VERSION

REGISTRY_VERSION = "growrag-prompt-registry-v1"

# The historical paired adapter assembles this suffix inline. Keep the old code
# untouched; a transport-capture contract test fails if this export ever drifts.
_PAIRED_PROMPT = ACTION_PROMPT + (
    "\nIf an optional historical card is supplied, its conditions are not "
    "pre-verified for the current question. Use its procedure only when "
    "compatible with current inputs; otherwise ignore it. Whether or not a "
    "card is supplied, you may return the original question unchanged.\n"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """Immutable execution description; all collection fields are tuples."""

    variant_id: str
    stage: str
    prompt_version: str
    system_prompt: str
    user_payload_template: str
    runtime_input_fields: tuple[str, ...]
    forbidden_input_fields: tuple[str, ...]
    output_contract: str
    postprocessing: str
    source_urls: tuple[str, ...]
    adaptation_notes: tuple[str, ...]

    @property
    def system_prompt_sha256(self) -> str:
        """Hash exact UTF-8 bytes, including whitespace and terminal newline."""
        return _sha256(self.system_prompt)

    @property
    def execution_signature_sha256(self) -> str:
        """Identity includes input/output handling, not merely prompt text.

        RRR_QUERY_ANCHOR intentionally shares the RRR system-prompt hash while
        having a different execution signature. Provider settings and source
        code commit remain separate required provenance in the run manifest.
        """
        execution = {
            "stage": self.stage,
            "prompt_version": self.prompt_version,
            "system_prompt_sha256": self.system_prompt_sha256,
            "user_payload_template": self.user_payload_template,
            "runtime_input_fields": self.runtime_input_fields,
            "forbidden_input_fields": self.forbidden_input_fields,
            "output_contract": self.output_contract,
            "postprocessing": self.postprocessing,
        }
        return _sha256(json.dumps(execution, ensure_ascii=False, sort_keys=True))

    def identity(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "prompt_version": self.prompt_version,
            "system_prompt_sha256": self.system_prompt_sha256,
            "execution_signature_sha256": self.execution_signature_sha256,
        }

    def to_dict(self) -> dict:
        """Return an independent copy, never a mutable view of the registry."""
        return {**asdict(self), **self.identity()}


_FORBIDDEN_PRE_INPUTS = (
    "gold_answer",
    "gold_supporting_facts",
    "question_type_or_difficulty_annotation",
    "current_retrieved_evidence",
    "previous_queries",
    "historical_memory",
)


def _baseline_record(variant_id: str) -> PromptRecord:
    spec = ALL_BASELINE_SPECS[variant_id]
    prompts = {
        "SIMPLE_PARAPHRASE": _PAIRED_PROMPT,
        "RRR_KEYWORDS": RRR_PROMPT,
        "RRR_MINIMAL": RRR_MINIMAL_PROMPT,
        "RRR_QUERY_ANCHOR": RRR_PROMPT,
        "QUERY2DOC": QUERY2DOC_PROMPT,
    }
    payload: dict = {"original_question": "<original question text>"}
    if variant_id == "SIMPLE_PARAPHRASE":
        payload.update(
            form=spec.form.value,
            intent=spec.intent,
            current_evidence=[],
            previous_queries=[],
        )
    is_passage = variant_id == "QUERY2DOC"
    anchored = variant_id in {"QUERY2DOC", "RRR_QUERY_ANCHOR"}
    return PromptRecord(
        variant_id=variant_id,
        stage="PRE: before any document retrieval",
        prompt_version=spec.prompt_version,
        system_prompt=prompts[variant_id],
        user_payload_template=json.dumps(payload, ensure_ascii=False, indent=2),
        runtime_input_fields=("original_question",),
        forbidden_input_fields=_FORBIDDEN_PRE_INPUTS,
        output_contract=(
            'Exactly {"passage": nonempty string, at most 1000 characters}'
            if is_passage
            else 'Exactly {"query": nonempty string, at most 2000 characters}'
        ),
        postprocessing=(
            "Original question + newline + stripped model text; combined maximum 2000 "
            "characters, reject overlength without truncation. Search text only, never Evidence."
            if anchored
            else "Strip model query; use only as retrieval query. Never replace Reader question."
        ),
        source_urls=(spec.source_url,) if spec.source_url else (),
        adaptation_notes=spec.adaptation_notes,
    )


PROMPT_REGISTRY: Mapping[str, PromptRecord] = MappingProxyType(
    {
        **{variant: _baseline_record(variant) for variant in ALL_BASELINE_SPECS},
        "READER": PromptRecord(
            variant_id="READER",
            stage="POST: after actual document retrieval",
            prompt_version=READER_PROMPT_VERSION,
            system_prompt=READER_PROMPT,
            user_payload_template=json.dumps(
                {
                    "original_question": "<original question text, NOT the rewritten query>",
                    "evidence": [
                        {
                            "evidence_id": "<retrieved evidence id>",
                            "title": "<document title>",
                            "sentence_id": "<integer sentence index>",
                            "text": "<actual retrieved text>",
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            runtime_input_fields=("original_question", "evidence"),
            forbidden_input_fields=(
                "gold_answer",
                "gold_supporting_facts",
                "historical_answers_or_trajectories",
                "hypothetical_passage_as_evidence",
            ),
            output_contract=(
                'Exactly {"answer": string, "cited_evidence_ids": list of supplied IDs}; '
                "empty answer requires empty citations, nonempty answer requires citations."
            ),
            postprocessing=(
                "Preserve answer text and deduplicate citations; never silently repair JSON."
            ),
            source_urls=(),
            adaptation_notes=(
                "Existing GrowRAG evidence-bound Reader, not a paper reproduction.",
                "Citation validity checks IDs, NOT whether cited text entails the answer.",
                "Keep Reader prompt/model identical across query-prompt comparisons.",
            ),
        ),
    }
)


def prompt_identity(variant_id: str) -> dict:
    """Return hash provenance for a query variant or READER; BASE uses no prompt."""
    if variant_id == "BASE":
        return {
            "variant_id": "BASE",
            "prompt_version": None,
            "system_prompt_sha256": None,
            "execution_signature_sha256": None,
            "note": "No query-generation call; the shared READER prompt still applies.",
        }
    try:
        return PROMPT_REGISTRY[variant_id].identity()
    except KeyError:
        raise ValueError("unknown prompt registry variant") from None


def registry_document() -> dict:
    """Static descriptions only: no runtime questions, responses or credentials."""
    return {
        "schema_version": REGISTRY_VERSION,
        "purpose": "Reproducible prompt appendix, not model-quality evidence.",
        "default_query_variants": list(BASELINE_SPECS),
        "additional_required_run_metadata": [
            "code commit",
            "requested and returned model/version",
            "temperature, thinking setting, output limit, provider endpoint region",
            "dataset split and selection hash",
            "retriever/index fingerprint and top_k",
            "actual API usage, errors and latency",
        ],
        "comparison_boundaries": [
            "RRR_MINIMAL versus RRR_KEYWORDS: explicit semantic-invariant prompt reminders.",
            "RRR_QUERY_ANCHOR versus RRR_KEYWORDS: post-generation query composition only.",
            "SIMPLE, RRR and QUERY2DOC: method comparison, not a one-factor prompt ablation.",
            "Do not select prompts using final test outcomes or hide failed API calls.",
            "No parameter training, paid execution or quality improvement is implied by export.",
        ],
        "prompts": [record.to_dict() for record in PROMPT_REGISTRY.values()],
    }


def registry_markdown() -> str:
    lines = [
        "# GrowRAG system prompt appendix",
        "",
        "最需要记住：这里是实际执行提示词的可复现清单，不是优劣结论。",
        "同一方法比较不同 prompt 时，模型、数据、检索器、Reader 和预算应保持一致。",
        "prompt 哈希不包含 API 设置；仍必须保存运行清单。BASE 没有查询生成提示词。",
        "",
        "RRR_MINIMAL / RRR_KEYWORDS：比较显式语义约束提醒。",
        "RRR_QUERY_ANCHOR / RRR_KEYWORDS：比较是否程序拼回原问题，system prompt 相同。",
        "不要根据最终测试集结果挑选 prompt，也不要把异常请求当作零成本。",
        "",
    ]
    for record in PROMPT_REGISTRY.values():
        lines.extend(
            [
                f"## {record.variant_id}",
                "",
                f"- Version: `{record.prompt_version}`",
                f"- Stage: {record.stage}",
                f"- Exact UTF-8 system prompt SHA-256: `{record.system_prompt_sha256}`",
                f"- Execution signature SHA-256: `{record.execution_signature_sha256}`",
                f"- Runtime input: {', '.join(record.runtime_input_fields)}",
                f"- Not supplied: {', '.join(record.forbidden_input_fields)}",
                "",
                "### Exact system message",
                "",
                "```text",
                record.system_prompt.rstrip("\n"),
                "```",
                "",
                "Exact terminal whitespace is retained in JSON; Markdown fences are display-only.",
                "",
                "### User payload template (placeholders, not experiment data)",
                "",
                "```json",
                record.user_payload_template,
                "```",
                "",
                f"Output: {record.output_contract}",
                "",
                f"Postprocessing: {record.postprocessing}",
                "",
                "### Sources and adaptation limits",
                "",
                *[f"- {url}" for url in record.source_urls],
                *[f"- {note}" for note in record.adaptation_notes],
                "",
            ]
        )
    return "\n".join(lines)


def export_registry(output: Path) -> tuple[Path, Path]:
    """Create a new export directory. No external service/configuration is read."""
    output.mkdir(parents=True, exist_ok=False)
    json_path, markdown_path = output / "prompts.json", output / "prompts.md"
    with json_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(registry_document(), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with markdown_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(registry_markdown())
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new export directory")
    args = parser.parse_args(argv)
    paths = export_registry(args.output)
    print(json.dumps({"api_calls": 0, "paths": [str(path) for path in paths]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
