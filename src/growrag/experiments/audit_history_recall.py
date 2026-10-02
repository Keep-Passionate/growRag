"""A0: fixed-library ranking audit on exposed development questions, without gold/API.

旧25题只用于定位工程瓶颈，不重新执行RAG或报告新准确率。保留全部44个名次，
不把“来源模板更多/排得更前”当相关性、适用性或收益标签。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from growrag.history_library import FrozenHistoryLibrary

from .fresh_dev_manifest import _sha
from .history_calibration import LIBRARY_FINGERPRINT, LIBRARY_SHA256, MANIFEST_SHA256
from .history_rank_diagnostics import POLICIES, action_text, rank_cards
from .pre_pilot import write_json
from .run_operator_study import artifact, load_inputs

OUTPUT = "runs/history_recall_audit_20261002_v1"


def compare_rankings(question, library):
    """Pure descriptive comparison; no labels and no proposed new action."""
    kinds = {r.card.card_id: r.source_kind for r in library.records if r.status == "published"}
    rankings = {policy: rank_cards(question.text, library, policy) for policy in POLICIES}
    baseline = [r["card_id"] for r in rankings[POLICIES[0]][:3]]
    return {
        "question_id": question.question_id,
        "question": question.text,
        "rankings": rankings,
        "top3": {
            policy: {
                "ids": [r["card_id"] for r in ranking[:3]],
                "source_template_count": sum(kinds[r["card_id"]] == "learned" for r in ranking[:3]),
                "changed_from_legacy": [r["card_id"] for r in ranking[:3]] != baseline,
            }
            for policy, ranking in rankings.items()
        },
    }


def audit(project):
    project = Path(project).resolve(strict=True)
    manifest_path = project / "data/hotpotqa/operator_scale_action_v3/manifest.json"
    library_path = project / "runs/history_foundation_20261001_v1/combined.json"
    if _sha(manifest_path) != MANIFEST_SHA256 or _sha(library_path) != LIBRARY_SHA256:
        raise ValueError("frozen inputs changed")
    manifest, questions = load_inputs(manifest_path, "calibration")
    library = FrozenHistoryLibrary.from_json(library_path.read_text(encoding="utf-8"))
    if library.fingerprint != LIBRARY_FINGERPRINT or len(library.published_cards) != 44:
        raise ValueError("library fingerprint or action count changed")
    chosen = questions[65:90]
    rows = [compare_rankings(question, library) for question in chosen]
    inventory = [
        {
            "card_id": r.card.card_id,
            "source_kind": r.source_kind,
            "action_kind": r.card.action_kind,
            "action_text": action_text(r.card),
            "conditions_count": len(r.card.conditions),
            "step_count": len(r.card.operator_spec.steps) if r.card.operator_spec else None,
            "gap_fields": [f.name for f in r.card.operator_spec.gap_schema]
            if r.card.operator_spec
            else [],
        }
        for r in library.records
        if r.status == "published"
    ]
    runtime = artifact(manifest_path.parent, manifest, "calibration_runtime_questions.jsonl")
    summary = {
        "protocol": "growrag-history-recall-audit-v1",
        "questions": len(rows),
        "actions": len(inventory),
        "api_requests": 0,
        "gold_loaded": False,
        "memory_updated": False,
        "question_scope": "exposed official-train development calibration[65:90]",
        "policies": {
            policy: {
                "changed_top3_questions": sum(
                    r["top3"][policy]["changed_from_legacy"] for r in rows
                ),
                "questions_with_source_template": sum(
                    r["top3"][policy]["source_template_count"] > 0 for r in rows
                ),
                "source_template_positions": sum(
                    r["top3"][policy]["source_template_count"] for r in rows
                ),
            }
            for policy in POLICIES
        },
        "inputs": {str(p): _sha(p) for p in (manifest_path, library_path, runtime)},
        "notice": "Ranking movement is not relevance/benefit. "
        "All methods remain lexical, not semantic/QPP.",
    }
    return summary, rows, inventory


def run(project, *, write=False):
    summary, rows, inventory = audit(project)
    if not write:
        return summary
    target = Path(project).resolve(strict=True) / OUTPUT
    if target.exists():
        raise FileExistsError("audit output already exists; never overwrite")
    target.mkdir(exist_ok=False)
    for name, value in (
        ("SUMMARY.json", summary),
        ("rankings.json", rows),
        ("cards.json", inventory),
    ):
        write_json(target / name, value)
    write_json(target / "seal.json", {p.name: _sha(p) for p in target.iterdir()})
    return {**summary, "output": str(target)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(Path.cwd(), write=args.write), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
