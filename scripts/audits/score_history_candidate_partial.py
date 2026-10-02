"""Externally amended, fixed45 paired development analysis; never a full50 score.

原50题评分器保持关闭；本脚本只在无标签terminal审计与事前分析修订均封存后，
投影原顺序前45题标签。失败1份BASE及24条未启动路径仍单列，不补齐、不重跑。
同Reader输入采用原固定method顺序首次回答；原预测、全部实发费用始终保留。
默认只读无标签预检，只有--score才读取固定45题标签并创建独占反馈目录。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

from growrag.experiments import score_history_candidates as frozen
from growrag.history_library import FrozenHistoryLibrary

_TERMINAL_PATH = Path(__file__).with_name("audit_history_candidate_terminal.py")
_SPEC = importlib.util.spec_from_file_location("candidate_terminal_for_partial", _TERMINAL_PATH)
terminal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(terminal)

AMENDMENT = "knowledge/experiments/2026-10-02_A路线_45题分析修订.md"
AMENDMENT_SHA256 = "788df571ef4fb3cf670f7b83d5bc0be477db796550c9bce6020763b7ea331057"
TERMINAL_SEAL_SHA256 = "02ab9559f1fab5a883f2a00750c5c340cd8a765e1fe8ff1b75f3752f17ec5a84"
PREFIX_SHA256 = "4b24918763e708403d350bcc56b9da901ad633fa063ecd090d934cbbce131296"
OUTPUT = "runs/history_candidates45_feedback_v1"
SCHEMA = "growrag-history-candidates45-partial-feedback-v1"
PAIRS = (("history_legacy3", "history_body3"), ("history_body3", "history_body8"))


def _check(condition, message):
    frozen._require(condition, message)


def _rehash(inputs):
    _check(
        all(frozen._sha(Path(path)) == digest for path, digest in inputs.items()),
        "sealed inputs changed before/while partial analysis",
    )


def collect(project):
    """Audit only: fixed45 prefix, original50 missing states and all costs remain."""
    project = Path(project).resolve(strict=True)
    amendment = frozen._inside(project, project / AMENDMENT)
    _check(
        len(AMENDMENT_SHA256) == 64 and frozen._sha(amendment) == AMENDMENT_SHA256,
        "pre-label amendment is not the hard-pinned version",
    )
    folder = (project / terminal.OUTPUT).resolve(strict=True)
    _check(
        folder.is_relative_to(project / "runs") and folder.is_dir(), "terminal folder escapes runs"
    )
    seal_path = frozen._inside(project, folder / "terminal_audit_frozen.json")
    _check(frozen._sha(seal_path) == TERMINAL_SEAL_SHA256, "terminal seal bytes drift")
    seal = frozen._read(seal_path)
    names = {"SUMMARY.json", "per_question_technical.json", "audit_inputs.json"}
    _check(
        seal.get("gold_loaded") is False
        and seal.get("api_calls") == 0
        and set(seal.get("files", {})) == names
        and {p.name for p in folder.iterdir()} == names | {seal_path.name},
        "terminal audit file coverage/role mismatch",
    )
    seal_inputs = {str(seal_path): TERMINAL_SEAL_SHA256, str(amendment): AMENDMENT_SHA256}
    for name in names:
        path = frozen._inside(project, folder / name)
        _check(frozen._sha(path) == seal["files"][name], "terminal output bytes drift")
        seal_inputs[str(path)] = seal["files"][name]
    saved_inputs = frozen._read(folder / "audit_inputs.json")["inputs"]
    _rehash(saved_inputs)
    public, technical, audited_inputs = terminal.collect(project)
    expected_inputs = {
        **audited_inputs,
        str(Path(terminal.__file__).resolve()): frozen._sha(Path(terminal.__file__)),
    }
    _check(
        saved_inputs == expected_inputs
        and frozen._read(folder / "SUMMARY.json") == public
        and frozen._read(folder / "per_question_technical.json") == technical,
        "re-audited terminal state differs from immutable terminal seal",
    )
    _check(
        public["complete_paired_questions"] == 45
        and public["planned_questions"] == 50
        and public["completed_method_reports"] == 225
        and public["failed_method_reports"] == 1
        and public["not_attempted_method_paths"] == 24
        and public["gold_loaded"] is False
        and public["source_sha256"] == terminal.SOURCE_SHA256
        and public["summary_sha256_pins"] == list(terminal.SUMMARY_PINS),
        "partial scope/source differs from pre-label terminal protocol",
    )
    manifest, questions, _ = frozen.load_bundle(project, expected_sha=frozen.study.BUNDLE_SHA256)
    ids = manifest["question_ids"][:45]
    _check(
        len(ids) == len(set(ids)) == 45
        and frozen.fingerprint(ids) == PREFIX_SHA256
        and [r["question_id"] for r in technical] == ids
        and [q.question_id for q in questions] == manifest["question_ids"],
        "paired prefix order differs from original fixed50",
    )
    library = FrozenHistoryLibrary.from_json(
        (project / manifest["library"]["path"]).read_text(encoding="utf-8")
    )
    inputs = {**saved_inputs, **seal_inputs}
    records, seen, all_calls = [], set(), []
    for (start, count), completed in zip(
        frozen.study.BATCHES, terminal.COMPLETE_BY_BATCH, strict=True
    ):
        directory = project / "runs" / frozen.study.run_id(start, count)
        plan = frozen._read(directory / "launch_plan.json")
        events = [
            json.loads(line)
            for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for qid in manifest["question_ids"][start : start + completed]:
            row = {"question_id": qid, "methods": {}, "usage": {}}
            for method in frozen.study.METHODS:
                report = frozen._read(directory / f"{qid}_{method}.json")
                row["methods"][method] = report
                row["usage"][method] = frozen._audit_method(
                    report, directory, plan, events, library, inputs, seen
                )
                all_calls.extend(report["calls"])
            row["question"] = row["methods"]["base"]["question"]["text"]
            records.append(row)
    _check([r["question_id"] for r in records] == ids, "fixed45 paired records incomplete")
    paired_totals = frozen._totals(all_calls)
    for key, value in paired_totals.items():
        _check(
            math.isclose(
                value + public["failed_reader"][key],
                public["totals_including_failed_http"][key],
                abs_tol=1e-10,
            ),
            "failed Reader cost omitted or duplicated",
        )
    missing = []
    for qid in manifest["question_ids"][45:]:
        states = {
            method: "failed" if qid == terminal.FAILED_ID and method == "base" else "not_attempted"
            for method in frozen.study.METHODS
        }
        missing.append({"question_id": qid, "methods": states, "labels_loaded": False})
    _rehash(inputs)
    return {
        "manifest": manifest,
        "ids": ids,
        "records": records,
        "missing": missing,
        "inputs": inputs,
        "terminal": public,
        "paired_totals": paired_totals,
    }


def _load_gold(project, manifest, ids):
    """Strict projection of45 prefix only. No labels for the five missing IDs."""
    import pyarrow.dataset as ds

    project = Path(project).resolve(strict=True)
    _check(
        len(manifest["question_ids"]) == 50
        and ids == manifest["question_ids"][:45]
        and len(ids) == len(set(ids)) == 45
        and manifest["question_ids"][45] == terminal.FAILED_ID
        and manifest["official_split"] == "train",
        "gold projection must be the fixed train45 prefix, not a chosen subset",
    )
    entry = manifest["source_provenance"]
    path = frozen._inside(project, project / entry["path"])
    _check(frozen._sha(path) == entry["sha256"], "train provenance hash mismatch")
    provenance = frozen._read(path)
    _check(provenance.get("official_split") == "train", "gold is not official train")
    recorded = {r["path"].replace("\\", "/"): r["sha256"] for r in manifest["parquet_inputs"]}
    _check(len(recorded) == len(manifest["parquet_inputs"]), "duplicate declared train shard")
    paths, inputs = [], {str(path): frozen._sha(path)}
    for shard in provenance["shards"]:
        part = frozen._inside(project, path.parent / shard["file"])
        digest = frozen._sha(part)
        _check(
            part.parent == path.parent
            and digest == shard["sha256"] == recorded.get(part.relative_to(project).as_posix()),
            "train shard/provenance/manifest mismatch",
        )
        paths.append(str(part))
        inputs[str(part)] = digest
    _check(
        paths and len(paths) == len(set(paths)) == len(recorded), "missing/duplicate train shards"
    )
    rows = (
        ds.dataset(paths, format="parquet")
        .to_table(
            columns=["id", "answer", "context", "supporting_facts"],
            filter=ds.field("id").isin(ids),
        )
        .to_pylist()
    )
    _check(
        len(rows) == 45 and {r["id"] for r in rows} == set(ids),
        "gold ID coverage includes missing/duplicate/unapproved labels",
    )
    return {r["id"]: r for r in rows}, inputs


def _extra_analysis(rows, summary):
    """Post-label descriptive contrasts, never selection labels or causal proof."""
    contrasts = {}
    opportunities = {label: {} for label in ("raw", frozen.PRIMARY)}
    for baseline, treatment in PAIRS:
        name = baseline + "_to_" + treatment
        values = []
        for row in rows:
            comparison = {
                "baseline": baseline,
                "treatment": treatment,
                "reader_input_identical": row["usage"][baseline]["reader_input_sha256"]
                == row["usage"][treatment]["reader_input_sha256"],
                "extra_api_requests": row["usage"][treatment]["api_requests"]
                - row["usage"][baseline]["api_requests"],
                "extra_retrieval_calls": row["usage"][treatment]["retrieval_calls"]
                - row["usage"][baseline]["retrieval_calls"],
                "extra_actual_estimated_cny": row["usage"][treatment]["estimated_actual_cny"]
                - row["usage"][baseline]["estimated_actual_cny"],
            }
            for label, key in (("raw", "raw_scores"), (frozen.PRIMARY, "controlled_reader_scores")):
                before, after = row[key][baseline], row[key][treatment]
                comparison[label] = {
                    "repaired": (before["answer_em"], after["answer_em"]) == (0.0, 1.0),
                    "harmed": (before["answer_em"], after["answer_em"]) == (1.0, 0.0),
                    "delta": {
                        metric: None
                        if before[metric] is None or after[metric] is None
                        else after[metric] - before[metric]
                        for metric in frozen.METRICS
                    },
                }
            row.setdefault("candidate_policy_contrasts", {})[name] = comparison
            values.append(comparison)
        contrasts[name] = {
            "baseline": baseline,
            "treatment": treatment,
            "paired_questions": len(rows),
            "changed_reader_input": sum(not v["reader_input_identical"] for v in values),
            "extra_api_requests": sum(v["extra_api_requests"] for v in values),
            "extra_retrieval_calls": sum(v["extra_retrieval_calls"] for v in values),
            "extra_actual_estimated_cny": sum(v["extra_actual_estimated_cny"] for v in values),
            **{
                label: {
                    "repaired": sum(v[label]["repaired"] for v in values),
                    "harmed": sum(v[label]["harmed"] for v in values),
                    "delta": {
                        metric: frozen._mean([v[label]["delta"][metric] for v in values])
                        for metric in frozen.METRICS
                    },
                }
                for label in ("raw", frozen.PRIMARY)
            },
        }
    for label, key in (("raw", "raw_scores"), (frozen.PRIMARY, "controlled_reader_scores")):
        for method in frozen.CANDIDATE_METHODS:
            qids = [
                row["question_id"]
                for row in rows
                if row[key]["base"]["answer_em"] == row[key]["fresh"]["answer_em"] == 0.0
                and row[key][method]["answer_em"] == 1.0
            ]
            opportunities[label][method] = {"count": len(qids), "question_ids": qids}
    summary["candidate_policy_contrasts"] = contrasts
    summary["base_fresh_wrong_history_correct"] = opportunities
    summary["descriptive_analysis_notice"] = (
        "Post-label diagnosis of the fixed completed45 only; an EM repair is not proof of "
        "causal memory benefit or correct applicability. No model/memory update or rerun."
    )


def _missing_bounds(rows):
    """Arithmetic uncertainty bounds only; no imputation or statistical CI."""
    result = {}
    comparisons = PAIRS + tuple(
        (other, method) for method in frozen.CANDIDATE_METHODS for other in ("base", "fresh")
    )
    for label, key in (("raw", "raw_scores"), (frozen.PRIMARY, "controlled_reader_scores")):
        methods, paired = {}, {}
        for method in frozen.study.METHODS:
            values = [row[key][method]["answer_em"] for row in rows]
            known = [v for v in values if v is not None]
            count = sum(v == 1.0 for v in known)
            methods[method] = {
                "observed_correct": count,
                "observed_valid_n": len(known),
                "worst_full50": count / 50 if len(known) == 45 else None,
                "best_full50": (count + 5) / 50 if len(known) == 45 else None,
            }
        for baseline, treatment in comparisons:
            values = [
                row[key][treatment]["answer_em"] - row[key][baseline]["answer_em"]
                for row in rows
                if row[key][treatment]["answer_em"] is not None
                and row[key][baseline]["answer_em"] is not None
            ]
            delta = sum(values)
            paired[baseline + "_to_" + treatment] = {
                "observed_delta_sum": delta,
                "observed_valid_n": len(values),
                "worst_full50_delta": (delta - 5) / 50 if len(values) == 45 else None,
                "best_full50_delta": (delta + 5) / 50 if len(values) == 45 else None,
            }
        result[label] = {"methods": methods, "paired": paired}
    return {
        "notice": (
            "Pure missing-result bounds, not actual50 accuracy, CI, imputation or significance."
        ),
        "planned_n": 50,
        "paired_observed_n": 45,
        "missing_n": 5,
        "by_metric_policy": result,
    }


def _markdown(rows, missing):
    lines = [
        "# A路线：中断后固定45题开发反馈（原计划50题）",
        "",
        "只分析原顺序前45道完整五路题；缺失5道单列，未打开其标签。不是官方测试成绩或完整50题结果。",
        "原始与同输入首次Reader规范化分数并列；规范化仅后处理，所有原调用成本保留，不是缓存省钱。",
        "",
    ]
    for row in rows:
        lines += [
            f"## {row['question_id']}",
            "",
            row["question"],
            "",
            f"参考答案：{row['reference_answer']}",
            "",
            "| method | 原EM/F1 | 控制EM/F1 | 可见证据覆盖 | API/检索 | 实发估算元 |",
            "|---|---|---|---|---|---|",
        ]
        for method in frozen.study.METHODS:
            raw = row["raw_scores"][method]
            controlled = row["controlled_reader_scores"][method]
            usage = row["usage"][method]
            lines.append(
                f"| {method} | {raw['answer_em']}/{raw['answer_f1']} | "
                f"{controlled['answer_em']}/{controlled['answer_f1']} | "
                f"{raw['visible_support_recall']} | "
                f"{usage['api_requests']}/{usage['retrieval_calls']} | "
                f"{usage['estimated_actual_cny']:.6f} |"
            )
        lines += [
            "",
            "```json",
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "answers",
                        "canonical_reader_method",
                        "usage",
                        "history_vs",
                        "candidate_policy_contrasts",
                    )
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    lines += ["## 原计划中未纳入准确率分析的5道题", "", "仅状态记录，无答案、无标签、无分数。", ""]
    for item in missing:
        lines.append(f"- {item['question_id']}：{json.dumps(item['methods'], ensure_ascii=False)}")
    return "\n".join(lines)


def run(project, *, score=False):
    project = Path(project).resolve(strict=True)
    output = (project / OUTPUT).resolve()
    _check(output.is_relative_to(project / "runs"), "feedback output escapes runs")
    if score and output.exists():
        raise FileExistsError("partial feedback exists; no overwrite or rescoring")
    checked = collect(project)
    public = {
        "schema_version": SCHEMA,
        "status": "amended_partial_preflight_passed",
        "planned_questions": 50,
        "complete_paired_questions": 45,
        "completed_method_reports": 225,
        "failed_method_reports": 1,
        "not_attempted_method_paths": 24,
        "gold_loaded": False,
        "api_calls": 0,
        "memory_updated": False,
        "prediction_modified": False,
        "full50_score_gate_amended": False,
        "prelabel_amendment_sha256": AMENDMENT_SHA256,
        "terminal_seal_sha256": TERMINAL_SEAL_SHA256,
        "source_sha256": terminal.SOURCE_SHA256,
        "paired_scored_totals": checked["paired_totals"],
        "campaign_totals_including_failed_http": checked["terminal"][
            "totals_including_failed_http"
        ],
    }
    if not score:
        return public
    _rehash(checked["inputs"])
    gold, gold_inputs = _load_gold(project, checked["manifest"], checked["ids"])
    _check(set(gold) == set(checked["ids"]), "label loader returned unapproved labels")
    rows, summary = frozen.summarize(checked["records"], gold)
    _extra_analysis(rows, summary)
    inputs = {
        **checked["inputs"],
        **gold_inputs,
        str(Path(__file__).resolve()): frozen._sha(Path(__file__)),
    }
    _rehash(inputs)
    summary.update(
        **{key: value for key, value in public.items() if key not in {"status", "gold_loaded"}},
        status="partial_development_scored_after_prelabel_amendment",
        gold_loaded=True,
        gold_projection_question_ids=checked["ids"],
        excluded_question_ids=checked["manifest"]["question_ids"][45:],
        labels_opened_after_partial_terminal_seal_and_prelabel_amendment=True,
        protocol=frozen.study.PROTOCOL,
        model=frozen.study.CONFIGURATION["model"],
        split="official_train_independent_candidate_development_partial45_of_planned50",
        missing_data_notice=(
            "45 complete pairs are a sequential completed prefix, not a randomized omission "
            "or full50 benchmark. Failure and unattempted paths are not imputed or scored."
        ),
        failed_reader_costs=checked["terminal"]["failed_reader"],
        full50_missing_result_bounds=_missing_bounds(rows),
        continuation_authorized=False,
    )
    output.mkdir(exist_ok=False)
    for name, value in (
        ("per_question.json", rows),
        ("missing_questions.json", checked["missing"]),
        ("SUMMARY.json", summary),
        ("audit.json", {"inputs": inputs, "prelabel_amendment_sha256": AMENDMENT_SHA256}),
    ):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    with (output / "per_question.md").open("x", encoding="utf-8") as handle:
        handle.write(_markdown(rows, checked["missing"]))
    with (output / "feedback_frozen.json").open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "files": {
                    p.name: frozen._sha(p)
                    for p in output.iterdir()
                    if p.name != "feedback_frozen.json"
                },
                "api_calls": 0,
                "primary_metric_policy": frozen.PRIMARY,
                "planned_questions": 50,
                "complete_paired_questions": 45,
                "gold_projection_question_ids": checked["ids"],
                "prelabel_amendment_sha256": AMENDMENT_SHA256,
                "terminal_seal_sha256": TERMINAL_SEAL_SHA256,
            },
            handle,
            indent=2,
        )
    return {**public, "status": summary["status"], "gold_loaded": True, "output": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score", action="store_true", help="project only fixed45 after audited amendment"
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(Path.cwd(), score=args.score), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
