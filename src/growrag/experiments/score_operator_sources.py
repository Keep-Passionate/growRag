"""Offline source-only scoring, gated on complete frozen prediction coverage.

中文：必须先收齐全部来源题的终态预测，再打开来源标签。校准/评测标签从不读取。
失败与坏标注是未知值，不伪造零分；段落级ID不能被误当作第0句支持证据。
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from .fresh_dev_manifest import _sha
from .hotpot import _answer_f1, _normalize
from .pre_pilot import write_json
from .representation_runner import fingerprint
from .run_operator_study import PREFIX, PROTOCOL, load_inputs
from .shared_hotpot_dev import document_id

ARMS = ("base", "fresh", "static")
SCHEMA = "growrag-operator-source-feedback-v1"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(root: Path, path: Path) -> Path:
    path = path.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("audit input escapes its declared root")
    return path


def collect_frozen_sources(manifest: dict, manifest_sha: str, runs_root: Path) -> tuple:
    """No labels are accessed here. Missing/unstarted source IDs fail closed."""
    runs_root = Path(runs_root).resolve(strict=True)
    sources = manifest["roles"]["source"]
    wanted = set(sources)
    records, models, inputs = {}, set(), []
    for launch in sorted(runs_root.glob(f"{PREFIX}*/launch_plan.json")):
        launch = _inside(runs_root, launch)
        plan = _read(launch)
        if plan.get("protocol") != PROTOCOL or plan.get("phase") != "source":
            continue
        if (
            plan.get("manifest_sha256") != manifest_sha
            or not isinstance(plan.get("model"), str)
            or not plan["model"].strip()
            or plan.get("gold_loaded") is not False
            or plan.get("memory_updates") is not False
        ):
            raise ValueError("source launch manifest/model/gold policy mismatch")
        planned_ids, planned_arms = plan.get("question_ids"), plan.get("arms")
        if (
            type(planned_ids) is not list
            or not planned_ids
            or len(set(planned_ids)) != len(planned_ids)
            or not set(planned_ids) <= wanted
            or type(planned_arms) is not list
            or not planned_arms
            or len(set(planned_arms)) != len(planned_arms)
            or not set(planned_arms) <= set(ARMS)
        ):
            raise ValueError("source launch has invalid source IDs/arms")
        predictions_path = _inside(runs_root, launch.parent / "predictions.json")
        frozen_path = _inside(runs_root, launch.parent / "predictions_frozen.json")
        reports, frozen = _read(predictions_path), _read(frozen_path)
        if (
            type(reports) is not list
            or frozen.get("sha256") != fingerprint(reports)
            or frozen.get("phase") != "source"
            or frozen.get("gold_loaded") is not False
            or frozen.get("status") not in {"completed", "failed"}
            or frozen.get("cleanup_errors") != []
        ):
            raise ValueError("source predictions missing valid freeze certificate")
        actual_ids = [item.get("question_id") for item in reports]
        if (
            actual_ids != frozen.get("question_ids")
            or len(set(actual_ids)) != len(actual_ids)
            or actual_ids != planned_ids[: len(actual_ids)]
        ):
            raise ValueError("frozen source prediction order/coverage mismatch")
        if frozen["status"] == "completed" and actual_ids != planned_ids:
            raise ValueError("completed launch has unstarted questions")
        for item in reports:
            qid, arms = item["question_id"], item.get("arms")
            if type(arms) is not dict or not arms or not set(arms) <= set(planned_arms):
                raise ValueError("source report has invalid arms")
            merged = records.setdefault(
                qid, {"question_id": qid, "phase": "source", "protocol": PROTOCOL, "arms": {}}
            )
            for arm, report in arms.items():
                if arm in merged["arms"]:
                    raise ValueError("duplicate source question/arm; no cherry-picked retries")
                if (
                    type(report) is not dict
                    or report.get("status") not in {"completed", "failed"}
                    or report.get("question_id", qid) != qid
                    or report.get("arm", arm) != arm
                    or report.get("feedback") is not None
                    or report.get("memory_updated", False)
                ):
                    raise ValueError("invalid source terminal report or preloaded feedback")
                if report["status"] == "completed" and (
                    type(report.get("reader")) is not dict
                    or not isinstance(report["reader"].get("answer"), str)
                    or type(report.get("episode")) is not dict
                    or report["episode"].get("question_id") != qid
                    or type(report["episode"].get("evidence")) is not list
                ):
                    raise ValueError("completed source arm lacks reader/episode")
                merged["arms"][arm] = deepcopy(report)
        models.add(plan["model"])
        inputs.extend(
            {"path": str(path), "sha256": _sha(path)}
            for path in (launch, predictions_path, frozen_path)
        )
    if len(models) != 1:
        raise ValueError("source launches must exist and share one fixed model")
    if set(records) != wanted:
        raise ValueError("not all source IDs have terminal reports; labels remain sealed")
    for item in records.values():
        if set(item["arms"]) != set(ARMS) and not any(
            report["status"] == "failed" for report in item["arms"].values()
        ):
            raise ValueError("source question has unstarted arms without a recorded failure")
    return [records[qid] for qid in sources], next(iter(models)), inputs


def _load_source_gold(project_root: Path, manifest: dict) -> tuple[dict, list]:
    """Only invoked after ALL source terminal reports are verified. No dev files."""
    import pyarrow.dataset as ds

    provenance_rel = "data/hotpotqa/official_train_v1_1/mirror_provenance.json"
    recorded = {item["path"].replace("\\", "/"): item["sha256"] for item in manifest["input_files"]}
    provenance_path = _inside(project_root, project_root / provenance_rel)
    if recorded.get(provenance_rel) != _sha(provenance_path):
        raise ValueError("train provenance differs from frozen data manifest")
    provenance = _read(provenance_path)
    if provenance.get("official_split") != "train":
        raise ValueError("source labels must come from official train")
    paths, audit = [], []
    for shard in provenance["shards"]:
        path = _inside(project_root, provenance_path.parent / shard["file"])
        relative = path.relative_to(project_root).as_posix()
        digest = _sha(path)
        if digest != shard["sha256"] or recorded.get(relative) != digest:
            raise ValueError("source label parquet differs from frozen manifest")
        paths.append(str(path))
        audit.append({"path": str(path), "sha256": digest})
    if not paths:
        raise ValueError("missing pinned source parquet")
    source_ids = manifest["roles"]["source"]
    rows = (
        ds.dataset(paths, format="parquet")
        .to_table(
            columns=["id", "answer", "supporting_facts", "context"],
            filter=ds.field("id").isin(source_ids),
        )
        .to_pylist()
    )
    if len(rows) != len(source_ids) or {row["id"] for row in rows} != set(source_ids):
        raise ValueError("frozen source label projection missing/duplicate IDs; never replace")
    return {row["id"]: row for row in rows}, audit


def _parse_gold(row: dict) -> tuple:
    answer = row.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("invalid_answer")
    context, support = row.get("context"), row.get("supporting_facts")
    if type(context) is not dict or type(support) is not dict:
        raise ValueError("invalid_annotation_structure")
    titles, paragraphs = context.get("title"), context.get("sentences")
    support_titles, indices = support.get("title"), support.get("sent_id")
    if (
        any(type(value) is not list for value in (titles, paragraphs, support_titles, indices))
        or len(titles) != len(paragraphs)
        or len(support_titles) != len(indices)
        or not support_titles
        or any(not isinstance(title, str) or not title.strip() for title in titles)
        or len(set(titles)) != len(titles)
    ):
        raise ValueError("invalid_or_ambiguous_support")
    documents = {}
    by_title = {}
    for title, sentences in zip(titles, paragraphs, strict=True):
        if type(sentences) is not list or any(not isinstance(s, str) for s in sentences):
            raise ValueError("invalid_context_sentences")
        doc_id = document_id(title, sentences)
        documents[doc_id] = (title, sentences)
        by_title[title] = doc_id
    targets = set()
    for title, index in zip(support_titles, indices, strict=True):
        if (
            not isinstance(title, str)
            or title not in by_title
            or type(index) is not int
            or index < 0
            or index >= len(documents[by_title[title]][1])
            or not documents[by_title[title]][1][index].strip()
        ):
            raise ValueError("invalid_support_reference")
        targets.add((by_title[title], index))
    return answer, documents, targets


def _support_coverage(report: dict, documents: dict, targets: set) -> tuple:
    evidence = report["episode"]["evidence"]
    known, seen = {}, set()
    for item in evidence:
        if type(item) is not dict or not isinstance(item.get("evidence_id"), str):
            raise ValueError("invalid_evidence")
        identity = item["evidence_id"]
        if identity in seen:
            raise ValueError("duplicate_evidence_id")
        seen.add(identity)
        if identity in documents:
            title, sentences = documents[identity]
            if item.get("title") != title or item.get("text") != "\n".join(sentences):
                raise ValueError("evidence_content_id_mismatch")
            known[identity] = item
    raw = sum(identity in known for identity, _ in targets) / len(targets)
    reader = report["reader"]
    windows, visible_ids = reader.get("evidence_windows"), reader.get("visible_evidence_ids")
    if type(windows) is not list or type(visible_ids) is not list:
        return raw, None, "visible_windows_not_recorded"
    if (
        any(type(window) is not dict for window in windows)
        or [window.get("evidence_id") for window in windows] != visible_ids
        or len(set(visible_ids)) != len(visible_ids)
        or not set(visible_ids) <= seen
    ):
        return raw, None, "invalid_visible_windows"
    ranges = {}
    for window in windows:
        identity = window["evidence_id"]
        if identity not in known:
            continue
        start, end = window.get("text_start"), window.get("text_end")
        length = len(known[identity]["text"])
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= length
            or window.get("original_text_chars") != length
        ):
            return raw, None, "invalid_visible_offsets"
        ranges[identity] = (start, end)
    covered = 0
    for identity, index in targets:
        if identity not in ranges:
            continue
        sentences = documents[identity][1]
        sentence_start = sum(len(sentence) + 1 for sentence in sentences[:index])
        sentence_end = sentence_start + len(sentences[index])
        start, end = ranges[identity]
        covered += int(start <= sentence_start and sentence_end <= end)
    return raw, covered / len(targets), None


def score_source_report(report: dict | None, gold_row: dict) -> dict:
    """Pure per-arm scoring; missing/failed/invalid states stay explicitly unknown."""
    result = {
        "answer_em": None,
        "answer_f1": None,
        "raw_support_recall": None,
        "visible_support_recall": None,
        "annotation_status": "unchecked",
        "status": report.get("status") if report else "not_attempted",
        "feedback_source": "gold",
        "issue": None,
    }
    try:
        answer, documents, targets = _parse_gold(gold_row)
        result["annotation_status"] = "valid"
    except (TypeError, ValueError, KeyError) as error:
        result.update(annotation_status="invalid", issue=str(error))
        return result
    if report is None or report.get("status") != "completed":
        result["issue"] = "missing_or_failed_execution"
        return result
    prediction = report["reader"]["answer"]
    result["answer_em"] = float(_normalize(prediction) == _normalize(answer))
    result["answer_f1"] = _answer_f1(prediction, answer)
    try:
        raw, visible, issue = _support_coverage(report, documents, targets)
        result.update(raw_support_recall=raw, visible_support_recall=visible, issue=issue)
    except (TypeError, ValueError, KeyError) as error:
        result["issue"] = str(error)
    return result


def score_operator_sources(
    project_root: Path,
    manifest_path: Path,
    runs_root: Path,
    output_dir: Path,
    *,
    write: bool = False,
) -> dict:
    project_root = Path(project_root).resolve(strict=True)
    manifest_path = _inside(project_root, Path(manifest_path))
    runs_root = Path(runs_root).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if (
        not runs_root.is_relative_to(project_root)
        or not output_dir.is_relative_to(project_root)
        or output_dir == project_root
        or output_dir == runs_root
    ):
        raise ValueError("runs/output must stay inside project and not replace a root")
    if write and output_dir.exists():
        raise FileExistsError("feedback output exists; never overwrite")
    manifest, _ = load_inputs(manifest_path, "source")
    manifest_sha = _sha(manifest_path)
    reports, model, audited = collect_frozen_sources(manifest, manifest_sha, runs_root)
    audit = {
        "schema_version": SCHEMA,
        "protocol": PROTOCOL,
        "phase": "source",
        "model": model,
        "manifest_sha256": manifest_sha,
        "source_ids": manifest["roles"]["source"],
        "source_count": len(reports),
        "prediction_inputs": audited,
        "gold_loaded": False,
        "api_calls": 0,
        "calibration_evaluation_gold_loaded": False,
        "raw_predictions_modified": False,
        "coverage_notice": "Exact annotated support coverage is not textual entailment. "
        "Paragraph Evidence.sentence_id is never interpreted as a raw sentence reference.",
    }
    if not write:
        return audit
    # This is the first point at which ANY labels may enter memory.
    gold, gold_inputs = _load_source_gold(project_root, manifest)
    scoring_audits = {
        item["question_id"]: {
            arm: score_source_report(item["arms"].get(arm), gold[item["question_id"]])
            for arm in ARMS
        }
        for item in reports
    }
    # The bank builder accepts only scalar feedback, not annotation text or audits.
    feedback = {
        qid: {
            arm: {key: metrics[key] for key in ("answer_em", "answer_f1", "raw_support_recall")}
            for arm, metrics in arms.items()
        }
        for qid, arms in scoring_audits.items()
    }
    for item in audited + gold_inputs:
        if _sha(Path(item["path"])) != item["sha256"]:
            raise ValueError("frozen input changed during offline scoring")
    if _sha(manifest_path) != manifest_sha:
        raise ValueError("manifest changed during offline scoring")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "feedback.json", feedback)
    write_json(output_dir / "source_reports.json", reports)
    write_json(output_dir / "scoring_audits.json", scoring_audits)
    audit.update(
        gold_loaded=True,
        gold_inputs=gold_inputs,
        feedback_sha256=fingerprint(feedback),
        source_reports_sha256=fingerprint(reports),
        artifacts={
            name: {"sha256": _sha(output_dir / name)}
            for name in ("feedback.json", "source_reports.json", "scoring_audits.json")
        },
    )
    write_json(output_dir / "audit.json", audit)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-new", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve(strict=True)
    result = score_operator_sources(
        root, root / args.manifest, root / args.runs, root / args.output_dir, write=args.score_new
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "schema_version",
                    "protocol",
                    "phase",
                    "source_count",
                    "model",
                    "gold_loaded",
                    "api_calls",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
