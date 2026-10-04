"""Export a small, local-only view of the sealed S2G development trajectories.

This is an archive reader: it imports no GrowRAG runtime, calls no API, reads no
gold, and never chooses examples by their answers. The first N manifest IDs are
selected before events are inspected. Request/response events are skipped before
their payload is parsed; exported fields have explicit allowlists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SERIES = "2026-09-27_s2g_shared500_v1"
ARM = "S2G_AUTHOR_API4"
MANIFEST = "data/hotpotqa/shared500_sep27_v1/manifest.json"
INDEX = "data/indexes/shared500_offline_bm25_v1.sqlite"
KINDS = {
    "arm_start",
    "start",
    "judge",
    "query",
    "retrieval",
    "extraction",
    "finish",
    "question_complete",
}
HEADER = {"logged_at_utc", "arm", "event_index", "timestamp_utc", "question_id", "kind"}
FIELDS = {
    "arm_start": (),
    "start": ("question",),
    "judge": ("evidence_contexts", "verdicts", "gap_items"),
    "query": ("query", "gap_items", "gap_profile"),
    "retrieval": ("round", "queries", "documents"),
    "extraction": ("round", "pointers", "sources"),
    "finish": ("stop_reason", "api_calls"),
    "question_complete": ("status", "requests"),
}
CONFIG_FIELDS = (
    "protocol",
    "series",
    "git",
    "model",
    "backend_output_caps",
    "generation_profile",
    "answer_length_policy",
    "max_retrieval_rounds",
    "top_docs",
    "gap_profile",
    "arms",
    "no_training_no_memory_updates",
    "official_dev_test_used",
    "runtime_metadata",
)
AUTHOR_FIELDS = (
    "baseline_id",
    "upstream_commit",
    "source_sha256",
    "author_prompt_sha256",
    "max_retrieval_rounds",
    "max_model_calls",
    "top_documents",
    "gap_profile",
    "remove_repeat_docs",
    "dedup_key",
    "sentence_splitter",
    "trained_author_judge_used",
    "retrieval_backend",
    "backend_generation_settings",
    "preserved_behaviors",
    "gold_provided_to_author_loop",
)
MODULES = (
    "src/growrag/experiments/run_shared_s2g.py",
    "src/growrag/experiments/s2g_adapter.py",
    "src/growrag/experiments/s2g_author_api.py",
    "src/growrag/experiments/shared_s2g_corpus.py",
    "src/growrag/experiments/lexical_retriever.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def safe_file(project, relative):
    path = project / relative
    resolved = path.resolve(strict=True)
    require(resolved == path.absolute() and resolved.is_relative_to(project), "redirected input")
    require(resolved.is_file(), "input is not a file")
    return resolved


def metadata(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def read_json(path):
    return json.loads(path.read_bytes())


def event_kind(line):
    """Read only the short top-level header, stopping before API payloads.

    A regex over the whole line could mistake quoted model text for metadata.
    This parser requires kind to occur in the known header and fails closed on
    an unexpected field before it. It never decodes messages/content/answers.
    """
    decoder = json.JSONDecoder()
    offset = len(line) - len(line.lstrip())
    require(line[offset : offset + 1] == "{", "event must be an object")
    offset += 1
    while True:
        while offset < len(line) and line[offset].isspace():
            offset += 1
        key, offset = decoder.raw_decode(line, offset)
        require(key in HEADER, "unexpected field before event kind")
        while offset < len(line) and line[offset].isspace():
            offset += 1
        require(line[offset : offset + 1] == ":", "bad event header")
        offset += 1
        while offset < len(line) and line[offset].isspace():
            offset += 1
        value, offset = decoder.raw_decode(line, offset)
        if key == "kind":
            require(isinstance(value, str), "invalid event kind")
            return value
        while offset < len(line) and line[offset].isspace():
            offset += 1
        require(line[offset : offset + 1] == ",", "missing event kind")
        offset += 1


def project_object(line, allowed):
    """Decode allowlisted top-level values only, lexically skip the others.

    In particular, a finish event's answer is traversed as opaque characters but
    is never decoded to a Python value. This is not a general JSON validator.
    """
    decoder = json.JSONDecoder()
    offset = line.index("{") + 1
    result, seen = {}, set()
    while True:
        while line[offset].isspace():
            offset += 1
        if line[offset] == "}":
            return result
        key, offset = decoder.raw_decode(line, offset)
        require(isinstance(key, str) and key not in seen, "duplicate or invalid event key")
        seen.add(key)
        while line[offset].isspace():
            offset += 1
        require(line[offset] == ":", "bad event field")
        offset += 1
        start, depth, quoted, escaped = offset, 0, False, False
        while offset < len(line):
            char = line[offset]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
            elif char in "]}":
                if depth == 0:
                    break
                depth -= 1
            elif char == "," and depth == 0:
                break
            offset += 1
        require(offset < len(line) and not quoted and depth == 0, "unclosed event value")
        if key in allowed:
            result[key] = json.loads(line[start:offset])
        if line[offset] == "}":
            return result
        offset += 1


def selected_event(line):
    kind = event_kind(line)
    if kind not in KINDS:
        return None
    event = project_object(line, HEADER | set(FIELDS[kind]))
    if event.get("arm") != ARM:
        return None
    # finish.answer and other nonallowlisted top-level values were not decoded.
    fields = {key: event[key] for key in (*HEADER, *FIELDS[kind]) if key in event}
    if "gap_items" in fields:

        def gap(value):
            if isinstance(value, list):
                return [gap(item) for item in value]
            require(isinstance(value, dict), "invalid gap item")
            return {
                key: value[key]
                for key in ("category", "target", "slot", "description")
                if key in value
            }

        fields["gap_items"] = gap(fields["gap_items"])
    if "documents" in fields:
        fields["documents"] = [
            [{key: doc[key] for key in ("doc_id", "title", "text")} for doc in group]
            for group in fields["documents"]
        ]
    if "sources" in fields:
        fields["sources"] = [
            {key: source[key] for key in ("doc_id", "title", "sentence_id", "text")}
            for source in fields["sources"]
        ]
    return fields


def normalize_qid(event, run_id, whitelist):
    raw = event["question_id"]
    if event["kind"] in {"arm_start", "question_complete"}:
        qid = raw
    else:
        prefix, suffix = run_id + "/", "/" + ARM
        require(raw.startswith(prefix) and raw.endswith(suffix), "unexpected trace ID")
        qid = raw[len(prefix) : -len(suffix)]
    require(qid in whitelist, "event question outside exact launch whitelist")
    return qid


def verify_snapshot(snapshot, expected):
    files = snapshot["files"]
    require(fingerprint(files) == snapshot["sha256"] == expected, "snapshot aggregate mismatch")
    for name, entry in files.items():
        actual = hashlib.sha256(entry["text"].encode("utf-8")).hexdigest()
        require(actual == entry["sha256"], "snapshot module mismatch: " + name)
    require(all(name in files for name in MODULES), "missing necessary snapshot module")
    return {name: files[name]["sha256"] for name in MODULES}


def build_package(project, sample_count=10):
    """Preflight everything in memory; no output directory is written on failure."""
    project = project.resolve(strict=True)
    manifest_path = safe_file(project, MANIFEST)
    manifest = read_json(manifest_path)
    require(
        manifest["official_split"] == "train" and manifest["role"] == "development",
        "not train/development",
    )
    require(
        manifest.get("official_dev_test_used") is False, "official dev/test exclusion unverified"
    )
    ids = manifest["question_ids"]
    require(len(ids) == 500 and len(set(ids)) == 500, "expected 500 unique manifest IDs")
    require(type(sample_count) is int and 1 <= sample_count <= len(ids), "invalid sample count")
    selected = set(ids[:sample_count])
    manifest_sha = digest(manifest_path)
    resources = {"manifest": metadata(manifest_path)}
    for name, key in (("corpus.jsonl", "corpus"), ("runtime_questions.jsonl", "runtime_questions")):
        artifact = manifest["artifacts"][name]
        require(
            artifact["runtime_safe"] is True and artifact["contains_gold"] is False,
            "unsafe runtime artifact",
        )
        require(name in manifest["runtime_input_allowlist"], "artifact outside runtime allowlist")
        path = safe_file(project, str(Path(MANIFEST).parent / name))
        info = metadata(path)
        require(
            info["sha256"] == artifact["sha256"] and info["bytes"] == artifact["bytes"],
            "runtime artifact mismatch",
        )
        resources[key] = info
    resources["index"] = metadata(safe_file(project, INDEX))
    batches, traces = [], []
    inventory = {
        qid: {"manifest_index": i, "question_id": qid, "attempts": []} for i, qid in enumerate(ids)
    }
    paths = sorted((project / "runs").glob(SERIES + "_*"))
    paths = [p for p in paths if p.is_dir() and re.fullmatch(SERIES + r"_\d{4}_\d{4}", p.name)]
    require(paths, "no matching batches")
    planned = set()
    for batch in paths:
        files = {
            name: safe_file(project, f"runs/{batch.name}/{name}.json")
            for name in ("launch_plan", "source_snapshot")
        }
        files["events"] = safe_file(project, f"runs/{batch.name}/events.jsonl")
        plan = read_json(files["launch_plan"])
        require(plan["run_id"] == batch.name, "run ID mismatch")
        require(Path(plan["manifest_path"]).resolve() == manifest_path, "unexpected manifest path")
        require(plan["manifest_sha256"] == manifest_sha, "manifest hash mismatch")
        start, count = plan["start"], plan["count"]
        require(
            type(start) is int and type(count) is int and 0 <= start < 500 and count > 0,
            "invalid slice",
        )
        whitelist = ids[start : start + count]
        require(
            len(whitelist) == count and plan["question_ids"] == whitelist,
            "launch IDs differ from manifest slice",
        )
        require(
            Path(plan["index_path"]).resolve() == Path(resources["index"]["path"]),
            "unexpected index path",
        )
        runtime = plan["runtime_metadata"]
        require(
            runtime["manifest_sha256"] == manifest_sha
            and runtime["corpus_sha256"] == resources["corpus"]["sha256"],
            "runtime pin mismatch",
        )
        module_hashes = verify_snapshot(read_json(files["source_snapshot"]), plan["source_sha256"])
        planned.update(whitelist)
        config = {key: plan[key] for key in CONFIG_FIELDS if key in plan}
        config["author"] = {
            key: plan["author"][key] for key in AUTHOR_FIELDS if key in plan["author"]
        }
        config["index_path"] = plan["index_path"]
        batch_record = {
            "run_id": batch.name,
            "start": start,
            "count": count,
            "paths": {key: str(value) for key, value in files.items()},
            "launch_plan_sha256": digest(files["launch_plan"]),
            "source_snapshot_file_sha256": digest(files["source_snapshot"]),
            "events_sha256": digest(files["events"]),
            "source_aggregate_sha256": plan["source_sha256"],
            "source_modules_sha256": module_hashes,
            "configuration": config,
        }
        batches.append(batch_record)
        attempts, selected_rows = {}, {}
        for qid in whitelist:
            attempt = {
                "run_id": batch.name,
                "counts": {},
                "status": "not_started",
                "stop_reason": None,
            }
            inventory[qid]["attempts"].append(attempt)
            attempts[qid] = attempt
            if qid in selected:
                selected_rows[qid] = {
                    "manifest_index": ids.index(qid),
                    "question_id": qid,
                    "run_id": batch.name,
                    "question": None,
                    "events": [],
                }
        prior = {qid: [] for qid in whitelist}
        ordinal = Counter()
        with files["events"].open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                event = selected_event(line)
                if event is None:
                    continue
                qid = normalize_qid(event, batch.name, whitelist)
                kind, attempt = event["kind"], attempts[qid]
                attempt["counts"][kind] = attempt["counts"].get(kind, 0) + 1
                if kind in {"arm_start", "start"}:
                    attempt["status"] = "started"
                elif kind == "question_complete":
                    attempt["status"] = event["status"]
                elif kind == "finish":
                    attempt["stop_reason"] = event["stop_reason"]
                clean = {
                    key: value
                    for key, value in event.items()
                    if key not in {"question_id", "arm", "logged_at_utc"}
                }
                clean["source_line"] = line_number
                clean["prior_retrieved_doc_ids"] = list(prior[qid])
                if kind == "judge":
                    ordinal[qid] += 1
                    clean["judge_ordinal"] = ordinal[qid]
                if qid in selected:
                    if kind == "start":
                        selected_rows[qid]["question"] = event["question"]
                        selected_rows[qid]["question_text_sha256"] = hashlib.sha256(
                            event["question"].encode("utf-8")
                        ).hexdigest()
                    selected_rows[qid]["events"].append(clean)
                # Only earlier completed retrieval events enter the next event's history.
                if kind == "retrieval":
                    for group in event["documents"]:
                        for document in group:
                            if document["doc_id"] not in prior[qid]:
                                prior[qid].append(document["doc_id"])
        traces.extend(selected_rows.values())
    require(planned == set(ids), "launch plans do not cover complete manifest")
    counts = Counter()
    for row in inventory.values():
        actual = [a for a in row["attempts"] if a["counts"].get("start")]
        row["started_attempts"] = len(actual)
        row["complete_trajectory_present"] = any(a["counts"].get("finish") for a in actual)
        counts["started_questions"] += bool(actual)
        counts["finished_questions"] += row["complete_trajectory_present"]
        counts["failed_questions"] += any(a["status"] == "failed" for a in actual)
        for attempt in row["attempts"]:
            counts.update({"event_" + key: value for key, value in attempt["counts"].items()})
    counts["not_started_questions"] = len(ids) - counts["started_questions"]
    traces.sort(key=lambda row: (row["manifest_index"], row["run_id"]))
    summary = {
        "schema_version": "growrag-s2g-local-handoff-v1",
        "official_split": "train",
        "role": "development",
        "selection": (
            "first N manifest IDs, all planned attempts, without answers or outcome filtering"
        ),
        "selected_question_ids": ids[:sample_count],
        "sample_count": sample_count,
        "manifest_question_count": len(ids),
        "batch_count": len(batches),
        "counts": dict(counts),
        "resources": resources,
        "batches": batches,
        "api_calls_made": 0,
        "limitations": [
            "This package is local only; do not upload raw trajectories to the public repository.",
            "Judge evidence comes only from its own evidence_contexts, before later retrieval.",
            "Retrieval events contain postfilter documents; raw top-50 is not logged here.",
            "No answer labels were read; completeness does not establish correctness.",
            "Index hash is measured now; the original plan pins corpus/config, not SQLite bytes.",
            "Query/retrieval replay is unverified until a separate offline audit runs.",
            "API payloads are skipped before parsing; events files are hashed as opaque bytes.",
            "question_complete.requests is cumulative batch count, not per-question API cost.",
        ],
    }
    return summary, list(inventory.values()), traces


def export_package(project, output, sample_count=10):
    project = project.resolve(strict=True)
    output = output.absolute()
    resolved = output.resolve()
    require(
        resolved == output
        and resolved.is_relative_to(project / "runs")
        and resolved != project / "runs",
        "output must be a new local runs subdirectory",
    )
    require(not output.exists(), "output already exists")
    summary, inventory, traces = build_package(project, sample_count)
    output.mkdir(parents=False, exist_ok=False)
    for name, value in (("inventory.json", inventory),):
        with (output / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    with (output / "selected_traces.jsonl").open("x", encoding="utf-8") as handle:
        for value in traces:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    summary["artifacts"] = {path.name: metadata(path) for path in sorted(output.iterdir())}
    with (output / "package_manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {
        "output": str(output),
        "batch_count": summary["batch_count"],
        "sample_count": sample_count,
        "counts": summary["counts"],
        "files": {path.name: metadata(path) for path in sorted(output.iterdir())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            export_package(args.project, args.output, args.sample_count),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
