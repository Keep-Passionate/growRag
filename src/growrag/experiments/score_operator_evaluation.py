"""Offline sealed-evaluation scoring; no memory writes, API calls or resampling.

默认仅预检：全部固定500题均有终态记录后才允许显式--score-new打开dev标签。
题目终态不等于七臂都已生成预测；因失败未执行的臂必须单独报告。
失败之后尚未启动的臂记作failure_induced_unstarted，指标为未知，绝不补零。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

from growrag.operator_bank import FrozenOperatorBank, operator_to_dict

from .build_operator_banks import _read
from .fresh_dev_manifest import _sha
from .operator_execution_signature import validate_execution_signature
from .pre_pilot import write_json
from .representation_runner import fingerprint
from .run_operator_study import ARMS, PREFIX, PROTOCOL
from .score_operator_sources import score_source_report

SCHEMA = "growrag-operator-evaluation-feedback-v1"
_EVALUATION_COUNT = 500


def _inside(root: Path, value: str | Path, *, directory=False) -> Path:
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root) or (not path.is_dir() if directory else not path.is_file()):
        raise ValueError("evaluation audit path escapes declared root")
    return path


def _validate_freeze(root, path, expected):
    # Local import avoids a scorer->freeze->runner import cycle during integration.
    from .operator_evaluation_freeze import validate_evaluation_freeze

    return validate_evaluation_freeze(root, path, expected_certificate_sha256=expected)


def _check_executed_reuse(outcome: dict, arm: str, published: dict) -> None:
    """Audit actual retrieval actions, not rejected/unexecuted proposals.

    Reader failure can retain a complete episode, whose already executed reuse
    still needs provenance checking. Missing episodes remain unknown.
    """
    episode = outcome.get("episode")
    if episode is None:
        return
    if type(episode) is not dict:
        raise ValueError("recorded episode must be an object or unknown")
    if episode.get("question_id") != outcome.get("question_id"):
        raise ValueError("recorded episode question differs from its arm")
    searches, proposals = episode.get("searches"), episode.get("proposals")
    if (
        type(searches) is not list
        or type(proposals) is not list
        or any(type(proposal) is not dict for proposal in proposals)
    ):
        raise ValueError("recorded episode lacks searches/proposals")
    if arm not in published and any(proposal.get("origin") == "reuse" for proposal in proposals):
        raise ValueError("non-memory arm cannot claim historical reuse")
    for event in searches:
        if (
            type(event) is not dict
            or type(event.get("step")) is not int
            or not 0 <= event["step"] <= len(proposals)
        ):
            raise ValueError("search references an absent proposal")
        step = event["step"]
        if step and proposals[step - 1].get("origin") == "reuse":
            spec = proposals[step - 1].get("spec")
            if type(spec) is not dict or fingerprint(spec) not in {
                fingerprint(item) for item in published.get(arm, [])
            }:
                raise ValueError("executed reuse spec differs from frozen published bank")


def collect_frozen_evaluation(
    root: Path,
    manifest: dict,
    freeze: dict,
    runs_root: Path,
    certificate_path: Path,
    certificate_sha: str,
) -> tuple[list, list]:
    """Read only predictions/provenance. Do not open runtime question text or labels."""
    root = Path(root).resolve(strict=True)
    runs_root = _inside(root, runs_root, directory=True)
    certificate_path = _inside(root, certificate_path)
    expected_ids = freeze["evaluation_ids"]
    roles = manifest["roles"]
    if (
        len(expected_ids) != _EVALUATION_COUNT
        or len(set(expected_ids)) != len(expected_ids)
        or roles["evaluation"] != expected_ids
        or set(expected_ids) & (set(roles["source"]) | set(roles["calibration"]))
        or freeze["evaluation_order_sha256"] != fingerprint(expected_ids)
    ):
        raise ValueError("independent fixed evaluation500/order required")
    by_id, audited = {}, {}
    indexes = {qid: index for index, qid in enumerate(expected_ids)}

    def read(path):
        path = _inside(root, path)
        audited[str(path)] = _sha(path)
        return _read(path)

    expected_banks = {f"memory{size}": row["fingerprint"] for size, row in freeze["banks"].items()}
    expected_bank_files = {
        f"memory{size}": row["file_sha256"] for size, row in freeze["banks"].items()
    }
    bank_root = _inside(root, Path(freeze["paths"]["banks"]), directory=True)
    published = {}
    for size, entry in freeze["banks"].items():
        path = _inside(root, bank_root / f"bank_{size}.json")
        bank = FrozenOperatorBank.from_dict(read(path))
        if (
            audited[str(path)] != entry["file_sha256"]
            or bank.fingerprint != entry["fingerprint"]
            or bank.protocol_id != PROTOCOL
            or set(bank.allowed_source_ids) != set(manifest["nested_source_ids"][size])
        ):
            raise ValueError("evaluation bank changed or contains non-source records")
        published[f"memory{size}"] = [operator_to_dict(spec) for spec in bank.published_specs]
    for launch_path in sorted(runs_root.glob(f"{PREFIX}*/launch_plan.json")):
        launch_path = _inside(runs_root, launch_path)
        plan = _read(launch_path)
        if plan.get("protocol") != PROTOCOL or plan.get("phase") != "evaluation":
            continue
        read(launch_path)
        folder = launch_path.parent
        if (
            plan.get("evaluation_freeze_sha256") != certificate_sha
            or _inside(root, Path(plan.get("evaluation_freeze_path", ""))) != certificate_path
            or plan.get("evaluation_order_sha256") != freeze["evaluation_order_sha256"]
            or plan.get("manifest_sha256") != freeze["expected"]["manifest"]
            or plan.get("model") != freeze["execution_signature"]["configuration"]["model"]
            or plan.get("execution_signature") != freeze["execution_signature"]
            or validate_execution_signature(plan.get("execution_signature"))
            != freeze["expected"]["execution"]
            or plan.get("arms") != list(ARMS)
            or plan.get("bank_sha256") != expected_banks
            or plan.get("bank_file_sha256") != expected_bank_files
            or plan.get("gold_loaded") is not False
            or plan.get("memory_updates") is not False
        ):
            raise ValueError("evaluation launch differs from frozen protocol/model/banks")
        for key, value in freeze["execution_signature"]["configuration"].items():
            if plan.get(key) != value:
                raise ValueError("evaluation runtime configuration changed")
        if read(folder / "evaluation_freeze_verified.json") != freeze:
            raise ValueError("launch verified a different evaluation freeze")
        snapshot = read(folder / "source_snapshot.json")
        files = snapshot.get("files")
        if (
            type(files) is not dict
            or snapshot.get("sha256") != fingerprint(files)
            or plan.get("source_sha256") != snapshot.get("sha256")
        ):
            raise ValueError("source snapshot fingerprint mismatch")
        runner = files.get("src/growrag/experiments/run_operator_study.py", {})
        if (
            runner.get("sha256") != freeze["runner_sha256"]
            or not isinstance(runner.get("text"), str)
            or hashlib.sha256(runner["text"].encode()).hexdigest() != freeze["runner_sha256"]
        ):
            raise ValueError("evaluation runner version differs from frozen wrapper")
        for name, digest in freeze["execution_signature"]["files"].items():
            entry = files.get(name, {})
            if (
                entry.get("sha256") != digest
                or not isinstance(entry.get("text"), str)
                or hashlib.sha256(entry["text"].encode()).hexdigest() != digest
            ):
                raise ValueError("evaluation snapshot core method differs from freeze")
        planned = plan.get("question_ids")
        if (
            type(planned) is not list
            or not 1 <= len(planned) <= 25
            or any(qid not in indexes for qid in planned)
            or planned != expected_ids[indexes[planned[0]] : indexes[planned[0]] + len(planned)]
        ):
            raise ValueError("batch must contain 1..25 questions in a fixed contiguous subsequence")
        reports, seal = read(folder / "predictions.json"), read(folder / "predictions_frozen.json")
        if (
            type(reports) is not list
            or any(type(item) is not dict for item in reports)
            or seal.get("sha256") != fingerprint(reports)
            or seal.get("phase") != "evaluation"
            or seal.get("gold_loaded") is not False
            or seal.get("status") not in {"completed", "failed"}
            or seal.get("cleanup_errors") != []
        ):
            raise ValueError("evaluation predictions lack a valid terminal seal")
        started = [item.get("question_id") for item in reports]
        if (
            started != planned[: len(started)]
            or started != seal.get("question_ids")
            or seal["status"] == "completed"
            and started != planned
        ):
            raise ValueError("evaluation batch terminal order/coverage mismatch")
        for number, item in enumerate(reports):
            qid, arms = item["question_id"], item.get("arms")
            if item.get("feedback") is not None or item.get("memory_updated", False):
                raise ValueError("evaluation question was scored or used for memory updates")
            if qid in by_id:
                raise ValueError("evaluation question repeated; no best-of-retry selection")
            if type(arms) is not dict or not arms or list(arms) != list(ARMS[: len(arms)]):
                raise ValueError("evaluation arm order is not the declared prefix")
            if read(folder / f"checkpoint_{number:04d}.json") != item:
                raise ValueError("question checkpoint differs from final predictions")
            failures = []
            for arm, outcome in arms.items():
                if (
                    type(outcome) is not dict
                    or outcome.get("status") not in {"completed", "failed"}
                    or outcome.get("question_id") != qid
                    or outcome.get("arm") != arm
                    or outcome.get("feedback") is not None
                    or outcome.get("memory_updated") is not False
                    or type(outcome.get("calls")) is not list
                    or read(folder / f"{qid}_{arm}.json") != outcome
                ):
                    raise ValueError("invalid/updated/scored evaluation arm or report mismatch")
                if outcome["status"] == "failed":
                    failures.append(arm)
                elif (
                    type(outcome.get("episode")) is not dict
                    or outcome["episode"].get("question_id") != qid
                    or type(outcome["episode"].get("evidence")) is not list
                    or type(outcome.get("reader")) is not dict
                    or not isinstance(outcome["reader"].get("answer"), str)
                ):
                    raise ValueError("completed evaluation arm lacks prediction/evidence")
                _check_executed_reuse(outcome, arm, published)
            if (
                len(failures) > 1
                or failures
                and failures[0] != list(arms)[-1]
                or len(arms) != len(ARMS)
                and not failures
                or failures
                and (seal["status"] != "failed" or number != len(reports) - 1)
            ):
                raise ValueError("partial evaluation arms require one recorded final failure")
            result = {
                "question_id": qid,
                "phase": "evaluation",
                "protocol": PROTOCOL,
                "arms": deepcopy(arms),
            }
            for arm in ARMS[len(arms) :]:
                failed_arm = failures[0]
                result["arms"][arm] = {
                    "question_id": qid,
                    "arm": arm,
                    "status": "failure_induced_unstarted",
                    "caused_by_arm": failed_arm,
                    "error_type": arms[failed_arm].get("error_type"),
                    "episode": None,
                    "reader": None,
                    "calls": [],
                    "feedback": None,
                    "memory_updated": False,
                }
            by_id[qid] = result
        # An unassigned failure call must not disappear from per-arm cost reports.
        ledger = read(folder / "final_budget.json")
        owned_calls = [
            call
            for item in reports
            for outcome in item["arms"].values()
            for call in outcome["calls"]
        ]
        if (
            ledger.get("calls") != owned_calls
            or type(ledger.get("api_requests")) is not int
            or ledger["api_requests"] < 0
            or any(type(call) is not dict for call in owned_calls)
        ):
            raise ValueError("final ledger and per-arm owned calls differ")
        traces = [call.get("trace_id") for call in owned_calls]
        if any(type(trace) is not str or not trace for trace in traces) or len(set(traces)) != len(
            traces
        ):
            raise ValueError("call trace must have exactly one arm owner")
        counts = [call.get("api_requests") for call in owned_calls]
        if any(count is not None and (type(count) is not int or count < 0) for count in counts):
            raise ValueError("invalid recorded API request count")
        if all(count is not None for count in counts) and sum(counts) != ledger["api_requests"]:
            raise ValueError("ledger request count differs from recorded calls")
        events_path = _inside(root, folder / "events.jsonl")
        audited[str(events_path)] = _sha(events_path)
        events = [
            json.loads(line) for line in events_path.read_bytes().splitlines() if line.strip()
        ]
        if (
            not events
            or any(type(event) is not dict for event in events)
            or events[-1].get("kind") != "exit"
            or events[-1].get("status") != seal["status"]
            or events[-1].get("requests") != ledger["api_requests"]
            or any(event.get("kind") == "exit" for event in events[:-1])
        ):
            raise ValueError("terminal exit differs from seal or final request ledger")
    if set(by_id) != set(expected_ids):
        raise ValueError("all fixed500 evaluation questions need terminal reports before labels")
    return [by_id[qid] for qid in expected_ids], [
        {"path": path, "sha256": digest} for path, digest in sorted(audited.items())
    ]


def _load_dev_gold(root: Path, manifest: dict) -> tuple[dict, list]:
    """Called only after all questions are terminal, not necessarily all arms complete."""
    import pyarrow.dataset as ds

    relative = "data/hotpotqa/official_dev_v1/mirror_provenance.json"
    declared = {item["path"].replace("\\", "/"): item["sha256"] for item in manifest["input_files"]}
    provenance_path = _inside(root, Path(relative))
    if declared.get(relative) != _sha(provenance_path):
        raise ValueError("dev provenance differs from frozen manifest")
    provenance = _read(provenance_path)
    if provenance.get("official_split") != "dev":
        raise ValueError("evaluation labels must be official dev, never train")
    paths, audit = [], [{"path": str(provenance_path), "sha256": _sha(provenance_path)}]
    for shard in provenance["shards"]:
        path = _inside(root, provenance_path.parent / shard["file"])
        if path.parent != provenance_path.parent:
            raise ValueError("dev shard escapes pinned provenance directory")
        digest = _sha(path)
        if digest != shard["sha256"] or declared.get(path.relative_to(root).as_posix()) != digest:
            raise ValueError("dev label shard differs from frozen manifest")
        paths.append(str(path))
        audit.append({"path": str(path), "sha256": digest})
    if not paths:
        raise ValueError("dev provenance lacks shards")
    wanted = manifest["roles"]["evaluation"]
    rows = (
        ds.dataset(paths, format="parquet")
        .to_table(
            columns=["id", "answer", "supporting_facts", "context"],
            filter=ds.field("id").isin(wanted),
        )
        .to_pylist()
    )
    if len(rows) != len(wanted) or {row["id"] for row in rows} != set(wanted):
        raise ValueError("missing/duplicate frozen evaluation annotations; never resample")
    return {row["id"]: row for row in rows}, audit


def score_operator_evaluation(
    root: Path,
    certificate_path: Path,
    runs_root: Path,
    output_dir: Path,
    *,
    expected_certificate_sha256: str,
    write: bool = False,
) -> dict:
    from . import hotpot, operator_evaluation_summary, score_operator_sources, shared_hotpot_dev

    implementation_inputs = [
        {"path": str(path), "sha256": _sha(path)}
        for path in (
            Path(__file__).resolve(),
            Path(operator_evaluation_summary.__file__).resolve(),
            Path(score_operator_sources.__file__).resolve(),
            Path(hotpot.__file__).resolve(),
            Path(shared_hotpot_dev.__file__).resolve(),
        )
    ]
    root = Path(root).resolve(strict=True)
    certificate_path = _inside(root, certificate_path)
    runs_root = _inside(root, runs_root, directory=True)
    output_dir = (root / output_dir).resolve()
    if not output_dir.is_relative_to(root) or output_dir in {root, runs_root}:
        raise ValueError("evaluation feedback output must be a new project directory")
    if write and output_dir.exists():
        raise FileExistsError("evaluation feedback exists; never overwrite or select a rerun")
    freeze = _validate_freeze(root, certificate_path, expected_certificate_sha256)
    if _inside(root, Path(freeze["paths"]["runs"]), directory=True) != runs_root:
        raise ValueError("runs root differs from evaluation freeze")
    manifest_path = _inside(root, Path(freeze["paths"]["manifest"]))
    if _sha(manifest_path) != freeze["expected"]["manifest"]:
        raise ValueError("evaluation manifest changed")
    manifest = _read(manifest_path)
    reports, inputs = collect_frozen_evaluation(
        root, manifest, freeze, runs_root, certificate_path, expected_certificate_sha256
    )
    statuses = ("completed", "failed", "failure_induced_unstarted")
    execution_counts = {}
    for arm in ARMS:
        counts = Counter(item["arms"][arm]["status"] for item in reports)
        execution_counts[arm] = {status: counts[status] for status in statuses}
    audit = {
        "schema_version": SCHEMA,
        "phase": "evaluation",
        "protocol": PROTOCOL,
        "certificate_sha256": expected_certificate_sha256,
        "manifest_sha256": freeze["expected"]["manifest"],
        "execution_sha256": freeze["expected"]["execution"],
        "runner_sha256": freeze["runner_sha256"],
        "banks": freeze["banks"],
        "evaluation_order_sha256": freeze["evaluation_order_sha256"],
        "question_count": len(reports),
        "arm_count": len(ARMS),
        "all_questions_terminal": True,
        "all_arm_predictions_completed": all(
            counts["completed"] == len(reports) for counts in execution_counts.values()
        ),
        "execution_counts": execution_counts,
        "scoring_implementation": implementation_inputs,
        "prediction_inputs": inputs,
        "gold_loaded": False,
        "api_calls": 0,
        "memory_updated": False,
        "raw_predictions_modified": False,
        "failure_policy": (
            "failed and failure_induced_unstarted metrics remain unknown; no resampling"
        ),
    }
    if not write:
        return audit
    gold, gold_inputs = _load_dev_gold(root, manifest)
    feedback = {
        item["question_id"]: {
            arm: score_source_report(report, gold[item["question_id"]])
            for arm, report in item["arms"].items()
        }
        for item in reports
    }
    for item in reports:
        for arm, report in item["arms"].items():
            if report["status"] != "completed":
                feedback[item["question_id"]][arm].update(
                    execution_error_type=report.get("error_type"),
                    caused_by_arm=report.get("caused_by_arm"),
                )
    summary = operator_evaluation_summary.summarize_evaluation(reports, feedback)
    for entry in inputs + gold_inputs:
        if _sha(_inside(root, Path(entry["path"]))) != entry["sha256"]:
            raise ValueError("evaluation input changed during offline scoring")
    if _validate_freeze(root, certificate_path, expected_certificate_sha256) != freeze:
        raise ValueError("evaluation bank/method/freeze changed while scoring")
    if any(_sha(Path(entry["path"])) != entry["sha256"] for entry in implementation_inputs):
        raise ValueError("scoring implementation changed during scoring")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "feedback.json", feedback)
    write_json(output_dir / "evaluation_reports.json", reports)
    write_json(output_dir / "summary.json", summary)
    audit.update(
        gold_loaded=True,
        gold_inputs=gold_inputs,
        feedback_sha256=fingerprint(feedback),
        reports_sha256=fingerprint(reports),
        artifacts={
            name: {"sha256": _sha(output_dir / name)}
            for name in ("feedback.json", "evaluation_reports.json", "summary.json")
        },
    )
    write_json(output_dir / "audit.json", audit)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--expected-certificate-sha256", required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-new", action="store_true")
    args = parser.parse_args(argv)
    result = score_operator_evaluation(
        args.root,
        args.certificate,
        args.runs,
        args.output_dir,
        expected_certificate_sha256=args.expected_certificate_sha256,
        write=args.score_new,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "schema_version",
                    "phase",
                    "question_count",
                    "arm_count",
                    "all_questions_terminal",
                    "all_arm_predictions_completed",
                    "gold_loaded",
                    "api_calls",
                    "memory_updated",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
