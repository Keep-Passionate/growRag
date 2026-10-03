"""Controlled lexical retrieval ablation with frozen, already-produced slot values.

中文：固定旧历史选择与填参，只比较query文本。槽值并非免费得到，控制也
不是可部署的无记忆方法。旧结果只读，新检索先封存、后投影旧许可标签。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from string import Formatter
from time import perf_counter

from . import first_repair_replay as source
from .a0_v2_feedback import load_gold
from .lexical_retriever import _tokens
from .research_diagnostic import FEEDBACK, HISTORY, _inside, _read, _sha, _write
from .score_operator_sources import _parse_gold
from .shared_s2g_corpus import SharedBM25Index

OUTPUT = "runs/slot_query_ablation_20261003_v1"
PROTOCOL = "knowledge/experiments/2026-10-03_固定槽值查询消融_执行前协议.md"
H_ACTUAL = "history_actual"
SLOTS = "slots_only"
Q_SLOTS = "query_plus_slots"
VARIANTS = (H_ACTUAL, SLOTS, Q_SLOTS)
GROUP_COUNTS = {"slots_with_fixed_words": 9, "question_plus_slots": 4, "slot_passthrough": 3}


def query_variants(question, proposal, saved_query):
    """Remove literal words without changing slot values or generating new text.

    只取模板实际引用的非original_question槽，不把字典里其他字段拼进来。
    Formatter语法拒绝属性访问、索引、格式化、转换，避免混进额外变换。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("nonempty original question required")
    if type(proposal) is not dict or type(proposal.get("goal")) is not dict:
        raise ValueError("proposal and goal must be objects")
    if proposal["goal"].get("original_question") != question or proposal.get("origin") != "reuse":
        raise ValueError("proposal goal or origin differs")
    if proposal["goal"].get("constraints") != [] or proposal.get("bindings") != []:
        raise ValueError("registered ablation requires empty goal constraints and bindings")
    if type(proposal.get("spec")) is not dict or type(proposal["spec"].get("steps")) is not list:
        raise ValueError("template steps must be an array")
    steps = proposal["spec"]["steps"]
    if (
        len(steps) != 1
        or type(steps[0]) is not dict
        or any(
            type(steps[0].get(k)) is not list or steps[0][k] for k in ("when", "requires_bindings")
        )
    ):
        raise ValueError("ablation requires one unconditional template step")
    gap = proposal.get("gap")
    if type(gap) is not dict:
        raise ValueError("gap must be an object")
    if "original_question" in gap:
        raise ValueError("gap cannot replace original_question")
    template = steps[0].get("template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("nonempty template required")
    names, literals, uses_question = [], [], False
    try:
        for literal, name, format_spec, conversion in Formatter().parse(template):
            literals.append(literal)
            if name is None:
                continue
            if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", name) or format_spec or conversion:
                raise ValueError("only plain template slots permitted")
            if name == "original_question":
                uses_question = True
            elif name not in names:
                names.append(name)
        values = [gap[name] for name in names]
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("used gap values must be nonempty text")
        schema = proposal["spec"].get("gap_schema")
        if type(schema) is not list or any(type(field) is not dict for field in schema):
            raise ValueError("declared gap schema required")
        by_name = {field.get("name"): field for field in schema}
        if len(by_name) != len(schema) or any(
            name not in by_name
            or by_name[name].get("kind") != "text"
            or by_name[name].get("required") is not True
            for name in names
        ):
            raise ValueError("referenced slots must be declared required text")
        rebuilt = template.format_map({**gap, "original_question": question}).strip()
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("invalid template or missing gap value") from error
    if rebuilt != saved_query:
        raise ValueError("saved query does not equal frozen template compilation")
    slots_query = " ".join(values).strip()
    if not any(c.isalnum() for c in slots_query):
        raise ValueError("slot query has no searchable tokens")
    literal_text = " ".join(literals)
    group = (
        "slots_with_fixed_words"
        if any(c.isalnum() for c in literal_text)
        else "question_plus_slots"
        if uses_question
        else "slot_passthrough"
    )
    return {
        "queries": {
            H_ACTUAL: saved_query,
            SLOTS: slots_query,
            Q_SLOTS: (question + " " + slots_query).strip(),
        },
        "slot_names": names,
        "slot_values": values,
        "uses_original_question": uses_question,
        "template": template,
        "operator_id": proposal["spec"]["operator_id"],
        "literal_text": literal_text,
        "structure_group": group,
    }


def load_source(project):
    source.verify(project)  # Old outputs and original inputs must still be sealed.
    rows, _, manifest, inputs = source.load_inputs(project)
    folder = project / source.OUTPUT
    scored = _read(folder / "per_question.json")
    traces = [
        json.loads(s) for s in (folder / "REPLAY.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [r["question_id"] for r in scored] != [r["question_id"] for r in rows] or (
        [r["question_id"] for r in traces] != [r["question_id"] for r in rows]
    ):
        raise ValueError("source replay cohort differs")
    trace_map = {r["question_id"]: r for r in traces}
    chosen = [r for r in scored if r["methods"][HISTORY]["action_executed"]]
    if len(chosen) != 16:
        raise ValueError("registered all-executed historical cohort differs")
    plans = []
    for row in chosen:
        historical = row["methods"][HISTORY]
        proposals = historical["source_proposals"]
        if len(proposals) != 1:
            raise ValueError("first proposal identity ambiguous")
        variants = query_variants(row["question"], proposals[0], historical["query"])
        group, queries = variants["structure_group"], variants["queries"]
        if (group == "question_plus_slots" and queries[Q_SLOTS] != queries[H_ACTUAL]) or (
            group == "slot_passthrough" and queries[SLOTS] != queries[H_ACTUAL]
        ):
            raise ValueError("registered structural negative control differs")
        plans.append({"question_id": row["question_id"], "question": row["question"], **variants})
    if dict(Counter(p["structure_group"] for p in plans)) != GROUP_COUNTS:
        raise ValueError("registered structural grouping differs")
    for path in folder.iterdir():
        if path.is_file():
            inputs[path.relative_to(project).as_posix()] = _sha(path)
    for relative in (
        PROTOCOL,
        "src/growrag/experiments/slot_query_ablation.py",
        "src/growrag/experiments/shared_hotpot_dev.py",
    ):
        inputs[relative] = _sha(_inside(project, project / relative))
    return plans, trace_map, manifest, inputs


def retrieve(index, query):
    """Fetch exact source sentences, including legal zero-match results."""
    started = perf_counter()
    returned = index(query, 6)
    docs = tuple(index.document(d.doc_id) for d in returned)
    if (
        len(docs) > 6
        or len({d.doc_id for d in docs}) != len(docs)
        or any(
            (d.doc_id, d.title, d.text) != (r.doc_id, r.title, r.text)
            for d, r in zip(docs, returned, strict=True)
        )
    ):
        raise ValueError("retrieval callback/source contract differs")
    return docs, perf_counter() - started


def replay_plan(index, plan, previous):
    initial, elapsed = retrieve(index, plan["question"])
    old_initial = source._documents(previous["initial"]["documents"])
    if initial != old_initial:
        raise ValueError("initial exact source ranking differs")
    result = {
        "question_id": plan["question_id"],
        "initial": [asdict(d) for d in initial],
        "variants": {},
        "local_retrieval_calls": 1,
        "local_retrieval_seconds": elapsed,
    }
    token_results = {tuple(sorted(set(_tokens(plan["question"])))): initial}
    for name in VARIANTS:
        query = plan["queries"][name]
        docs, elapsed = retrieve(index, query)
        if name == H_ACTUAL and docs != source._documents(
            previous["methods"][HISTORY]["repair_documents"]
        ):
            raise ValueError("historical exact source ranking differs")
        terms = tuple(sorted(set(_tokens(query))))
        if terms in token_results and docs != token_results[terms]:
            raise ValueError("same normalized BM25 term set returned different documents")
        token_results[terms] = docs
        union = source.merge_documents(initial, docs)
        result["variants"][name] = {
            "query": query,
            "repair_documents": [asdict(d) for d in docs],
            "documents": [asdict(d) for d in union],
            "replay_seconds": elapsed,
            "new_document_count": len(union) - len(initial),
            "normalized_unique_terms": list(terms),
        }
        result["local_retrieval_calls"] += 1
        result["local_retrieval_seconds"] += elapsed
    return result


def score_plan(plan, replay, gold, original):
    _, gold_docs, targets = _parse_gold(gold)
    initial = source._documents(replay["initial"])
    base = source.coverage(initial, gold_docs, targets)
    result = {**plan, "methods": {"base": {"coverage": base}}, "controls": {}}
    for name in VARIANTS:
        r = replay["variants"][name]
        cov = source.coverage(source._documents(r["documents"]), gold_docs, targets)
        if cov["raw_supported"] < base["raw_supported"]:
            raise ValueError("append-only raw support decreased")
        result["methods"][name] = {
            "coverage": cov,
            "repair_only_coverage": source.coverage(
                source._documents(r["repair_documents"]), gold_docs, targets
            ),
            "new_document_count": r["new_document_count"],
        }
    if base != original["methods"]["base"]["coverage"] or (
        result["methods"][H_ACTUAL]["coverage"] != original["methods"][HISTORY]["coverage"]
    ):
        raise ValueError("actual-query coverage differs from previous sealed replay")
    actual = replay["variants"][H_ACTUAL]
    for name in (SLOTS, Q_SLOTS):
        control = replay["variants"][name]
        result["controls"][name] = {
            "same_query": actual["query"] == control["query"],
            "same_term_set": actual["normalized_unique_terms"]
            == control["normalized_unique_terms"],
            "same_ranked_documents": actual["repair_documents"] == control["repair_documents"],
            "raw_difference_actual_minus_control": result["methods"][H_ACTUAL]["coverage"][
                "raw_support_recall"
            ]
            - result["methods"][name]["coverage"]["raw_support_recall"],
        }
    return result


def summarize(rows, replay):
    summary = {
        "n": len(rows),
        "api_calls": 0,
        "new_answers_generated": False,
        "memory_updated": False,
        "new_gold_opened": False,
        "local_retrieval_calls": sum(r["local_retrieval_calls"] for r in replay),
        "local_retrieval_seconds": sum(r["local_retrieval_seconds"] for r in replay),
        "groups": {},
        "notice": "Retrospective selected-cohort query expression ablation; "
        "slot values already produced with history, not a deployable memory-free baseline.",
    }
    for group in ("all_selected_16", *GROUP_COUNTS):
        selected = (
            rows
            if group == "all_selected_16"
            else [r for r in rows if r["structure_group"] == group]
        )
        if not selected:
            raise ValueError("empty structural group")
        stats = {"n": len(selected), "methods": {}, "actual_minus_control": {}}
        for method in ("base", *VARIANTS):
            stats["methods"][method] = {
                metric: mean(r["methods"][method]["coverage"][metric] for r in selected)
                for metric in ("raw_support_recall", "visible_support_recall")
            }
        for control in (SLOTS, Q_SLOTS):
            deltas = [
                r["controls"][control]["raw_difference_actual_minus_control"] for r in selected
            ]
            stats["actual_minus_control"][control] = {
                "positive_n": sum(d > 0 for d in deltas),
                "negative_n": sum(d < 0 for d in deltas),
                "tied_n": sum(d == 0 for d in deltas),
                "net": mean(deltas),
                "same_ranked_documents_n": sum(
                    r["controls"][control]["same_ranked_documents"] for r in selected
                ),
                "same_query_n": sum(r["controls"][control]["same_query"] for r in selected),
                "same_term_set_n": sum(r["controls"][control]["same_term_set"] for r in selected),
                "positive_ids": [
                    r["question_id"] for r, d in zip(selected, deltas, strict=True) if d > 0
                ],
                "negative_ids": [
                    r["question_id"] for r, d in zip(selected, deltas, strict=True) if d < 0
                ],
            }
        summary["groups"][group] = stats
    return summary


def run(project, output=OUTPUT):
    project = Path(project).resolve(strict=True)
    target = (project / output).resolve()
    if target.parent != project / "runs":
        raise ValueError("output must be a direct runs child")
    if target.exists():
        raise FileExistsError("ablation exists; never overwrite")
    plans, previous, manifest, inputs = load_source(project)
    ref = manifest["corpus_ref"]
    index = SharedBM25Index(
        project / ref["index_path"], project / ref["path"], ref["sha256"], ref["rows"]
    )
    target.mkdir(exist_ok=False)
    stage, replays = "query_plans", []
    try:
        _write(target / "PLANS.json", plans)
        _write(target / "plans_frozen.json", {"files": {"PLANS.json": _sha(target / "PLANS.json")}})
        stage = "retrieval"
        with (target / "REPLAY.jsonl").open("x", encoding="utf-8") as handle:
            for n, plan in enumerate(plans, 1):
                item = replay_plan(index, plan, previous[plan["question_id"]])
                replays.append(item)
                handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                print(f"ablated {n}/{len(plans)} {plan['question_id']}", flush=True)
        stage = "replay_seal"
        _write(
            target / "replay_frozen.json",
            {"files": {"REPLAY.jsonl": _sha(target / "REPLAY.jsonl")}},
        )
        if any(_sha(project / p) != digest for p, digest in inputs.items()):
            raise ValueError("input changed during ablation")
        stage = "old_gold_projection"
        # Only source IDs already opened under the successful old audit are allowed.
        ids = [p["question_id"] for p in plans]
        old_rows = _read(project / FEEDBACK / "per_question.json")
        if not set(ids) <= {r["question_id"] for r in old_rows}:
            raise ValueError("new/unopened gold ID requested")
        gold, gold_inputs = load_gold(project, manifest, ids)
        audit_inputs = _read(project / FEEDBACK / "audit.json")["inputs"]
        for path, digest in gold_inputs.items():
            if audit_inputs.get(path) != digest:
                raise ValueError("gold input not in successful old audit")
            inputs[Path(path).relative_to(project).as_posix()] = digest
        originals = {
            r["question_id"]: r for r in _read(project / source.OUTPUT / "per_question.json")
        }
        stage = "evidence_scoring"
        rows = [
            score_plan(p, r, gold[p["question_id"]], originals[p["question_id"]])
            for p, r in zip(plans, replays, strict=True)
        ]
        summary = summarize(rows, replays)
        _write(target / "per_question.json", rows)
        _write(target / "SUMMARY.json", summary)
        _write(target / "INPUTS.json", inputs)
        with (target / "per_question.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            fields = (
                "question_id",
                "structure_group",
                "method",
                "query",
                "raw_support_recall",
                "visible_support_recall",
                "same_ranked_as_actual",
                "actual_minus_control",
            )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                for name, item in row["methods"].items():
                    control = row["controls"].get(name, {})
                    writer.writerow(
                        {
                            "question_id": row["question_id"],
                            "structure_group": row["structure_group"],
                            "method": name,
                            "query": row["question"] if name == "base" else row["queries"][name],
                            "raw_support_recall": item["coverage"]["raw_support_recall"],
                            "visible_support_recall": item["coverage"]["visible_support_recall"],
                            "same_ranked_as_actual": control.get("same_ranked_documents"),
                            "actual_minus_control": control.get(
                                "raw_difference_actual_minus_control"
                            ),
                        }
                    )
        if any(_sha(project / p) != digest for p, digest in inputs.items()):
            raise ValueError("input changed during scoring")
        _write(target / "TERMINAL.json", {"status": "completed", "n": len(rows), "api_calls": 0})
        _write(target / "frozen.json", {"files": {p.name: _sha(p) for p in target.iterdir()}})
        return summary
    except BaseException as error:
        _write(
            target / "FAILED.json",
            {"stage": stage, "error_type": type(error).__name__, "error": str(error)},
        )
        raise
    finally:
        index.close()


def verify(project, output=OUTPUT):
    project = Path(project).resolve(strict=True)
    target = _inside(project, project / output)
    if target.parent != project / "runs":
        raise ValueError("output must be a direct runs child")
    seal = _read(target / "frozen.json")
    required = {
        "PLANS.json",
        "plans_frozen.json",
        "REPLAY.jsonl",
        "replay_frozen.json",
        "per_question.json",
        "SUMMARY.json",
        "INPUTS.json",
        "per_question.csv",
        "TERMINAL.json",
    }
    if set(seal["files"]) != required:
        raise ValueError("ablation seal coverage differs")
    for name, digest in seal["files"].items():
        if Path(name).name != name or _sha(_inside(project, target / name)) != digest:
            raise ValueError("ablation output SHA differs")
    for name in ("plans", "replay"):
        embedded = _read(target / (name + "_frozen.json"))["files"]
        expected = "PLANS.json" if name == "plans" else "REPLAY.jsonl"
        if embedded != {expected: _sha(target / expected)}:
            raise ValueError("inner ablation seal differs")
    for relative, digest in _read(target / "INPUTS.json").items():
        if _sha(_inside(project, project / relative)) != digest:
            raise ValueError("ablation input SHA differs")
    if _read(target / "TERMINAL.json")["status"] != "completed":
        raise ValueError("ablation terminal not completed")
    plans = _read(target / "PLANS.json")
    rows = _read(target / "per_question.json")
    replays = [
        json.loads(line)
        for line in (target / "REPLAY.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if [p["question_id"] for p in plans] != [r["question_id"] for r in rows] or (
        [p["question_id"] for p in plans] != [r["question_id"] for r in replays]
    ):
        raise ValueError("ablation ID pairing differs")
    if _read(target / "SUMMARY.json") != summarize(rows, replays):
        raise ValueError("ablation summary differs")
    return {
        "status": "verified",
        "n": len(rows),
        "api_calls": 0,
        "local_retrieval_calls": sum(r["local_retrieval_calls"] for r in replays),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            (verify if args.verify else run)(args.project_root, args.output),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
