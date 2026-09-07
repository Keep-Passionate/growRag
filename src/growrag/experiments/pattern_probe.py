"""Offline pattern candidate diagnostics. No reader, rewrite, model, or paid call.

An explicit CLI flag declares a provisional AGE scope for one compiled view;
it does not confer human approval or prove the card's free-text conditions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from growrag.experience.cards import ActivationStage
from growrag.experience.pattern_matcher import (
    MATCHER_VERSION,
    ReviewedScope,
    match_reviewed_candidates,
    view_fingerprint,
)
from growrag.experience.query_views import CardMemoryView
from growrag.query_operators import RewriteForm

from .pre_manifest import load_pre_examples
from .pre_pilot import write_json
from .protocol import RuntimeQuestion

SYNTHETIC_QUERIES = (
    "Between two people Clara Vale and Martin Reed, who is older?",
    "Who is younger, Iris Moon or Oscar Vale?",
    "Was Noah Reed born earlier than Eva Stone?",
    "What profession do Clara Vale and Martin Reed have in common?",
    "Between Clara Vale, Martin Reed and Nina Lake, who is older?",
    "Between two people Clara Vale and Martin Reed, who is not older?",
    "Between two people Clara Vale and Martin Reed, who is older and taller?",
    "Which portrait is older: Clara Vale or Martin Reed?",
    "Who is older, he or she?",
)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate library JSON field")
        value[key] = item
    return value


def load_views(path: Path) -> tuple[tuple[CardMemoryView, ...], str]:
    """Validate compiled-view shape, NOT re-certify original source admissions."""
    raw = path.read_bytes()
    library = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(library, dict) or library.get("schema_version") != (
        "growrag-pre-frozen-library-v1"
    ):
        raise ValueError("a frozen PRE candidate library is required")
    if not isinstance(library.get("views"), list):
        raise ValueError("compiled views must be a list")
    views = []
    for data in library["views"]:
        if not isinstance(data, dict) or any(
            not isinstance(data.get(key), list)
            for key in ("source_query_ids", "source_question_hashes")
        ):
            raise ValueError("invalid compiled view structure")
        views.append(
            CardMemoryView(
                **{
                    **data,
                    "source_query_ids": tuple(data["source_query_ids"]),
                    "source_question_hashes": tuple(data["source_question_hashes"]),
                    "stage": ActivationStage(data["stage"]),
                    "form": RewriteForm(data["form"]),
                }
            )
        )
    if len({view.memory_id for view in views}) != len(views):
        raise ValueError("duplicate compiled memory identity")
    return tuple(views), hashlib.sha256(raw).hexdigest()


def diagnose(questions, candidates, *, origin: str) -> dict:
    rows = []
    for question in questions:
        decision = match_reviewed_candidates(question, candidates)
        rows.append(
            {
                "question": asdict(question),
                "route": decision.route,
                "pattern": asdict(decision.pattern),
                "candidate_ids": [view.memory_id for view in decision.candidates],
                "audits": [asdict(audit) for audit in decision.audits],
            }
        )
    return {
        "origin": origin,
        "question_count": len(rows),
        "candidate_count": sum(row["route"] == "CANDIDATE" for row in rows),
        "base_count": sum(row["route"] == "BASE" for row in rows),
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--declare-age-scope", metavar="MEMORY_ID", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("diagnostic output already exists; no overwrite")
    views, library_hash = load_views(args.library)
    bound = next((view for view in views if view.memory_id == args.declare_age_scope), None)
    if bound is None:
        raise ValueError("the explicitly selected memory is not in this library")
    scope = ReviewedScope(
        view_fingerprint(bound),
        reviewer_note=(
            "Developer-provisional diagnostic scope: explicit two-name age comparison only. "
            "Not human approval, not height or all comparative facts; card efficacy unknown."
        ),
    )
    candidates = tuple((view, scope if view is bound else None) for view in views)
    synthetic = tuple(
        RuntimeQuestion(f"pattern-fixture-{i}", text, "handwritten-pattern-fixture")
        for i, text in enumerate(SYNTHETIC_QUERIES)
    )
    groups = [diagnose(synthetic, candidates, origin="handwritten_routing_smoke_not_benchmark")]
    manifest = None
    if args.manifest is not None:
        _, targets, manifest = load_pre_examples(args.manifest)
        groups.append(
            diagnose(
                tuple(example.question for example in targets),
                candidates,
                origin="previously_exposed_train_calibration_query_only_posthoc",
            )
        )
    report = {
        "schema_version": "growrag-pattern-probe-v1",
        "matcher_version": MATCHER_VERSION,
        "library_file_sha256": library_hash,
        "scope": asdict(scope),
        "manifest": manifest,
        "groups": groups,
        "api_requests": 0,
        "rag_calls": 0,
        "answer_effect": "unknown",
        "notice": "Only rule decisions are executed. Synthetic and exposed real questions "
        "are separate; no answer/gold evaluation, semantic certification or source revalidation. "
        "CANDIDATE is not permission to serve REUSE or to promote the frozen card.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "api_requests": 0,
                "output": str(args.output),
                "groups": [
                    {key: value for key, value in group.items() if key != "rows"}
                    for group in groups
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
