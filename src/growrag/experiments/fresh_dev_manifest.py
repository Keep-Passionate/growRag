"""Freeze NEW Hotpot train development questions without opening old check labels.

中文：先只按 ID/type 选 6+6 题，再解析这 12 题。旧池只读清单，
规范化重复检查复用同源 SHA 的既有全库审计，不投影旧 24check 文本。
This is development, never an official test split. No API or download exists here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import re
from collections import Counter
from pathlib import Path

from .data_protocol import normalize_question
from .hotpot import HotpotExample, parse_hotpot_example

SCHEMA = "growrag-fresh-dev-manifest-v1"
SEED = 20260926
SOURCE_SHA = "041237a12614e591efb7ed595a1f626f9ad5631ab38a5a3bb1252cfc0c66d4c5"
HISTORY_CUTOFF = "2026-09-24"
KINDS = ("bridge", "comparison")
META_NAMES = {"plan.json", "launch_plan.json", "batch_manifest.json", "source_pool_manifest.json"}
EARLY_PROBES = (
    "2026-09-07_pattern_probe_v1/report.json",
    "2026-09-07_pre_source_retrieval_probe_v1/probe.json",
)
_STRING = re.compile(rb'"(?:[^"\\]|\\.)*"', re.DOTALL)
_TOKEN = re.compile(rb'"(?:[^"\\]|\\.)*"|[{}\[\]]', re.DOTALL)
_QID = re.compile(r"[0-9a-f]{24}")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _digest(values) -> str:
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(path: Path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("metadata/subset unexpectedly large")
    return json.loads(path.read_bytes(), object_pairs_hook=_unique)


def _ids(value) -> set[str]:
    """Only known ID fields are interpreted; prompts/questions/gold are ignored."""
    found: set[str] = set()

    def strings(node):
        if isinstance(node, str):
            if not _QID.fullmatch(node):
                raise ValueError("invalid question ID in exclusion metadata")
            found.add(node)
        elif isinstance(node, list):
            for child in node:
                strings(child)
        elif isinstance(node, dict):
            for child in node.values():
                strings(child)
        else:
            raise ValueError("invalid exclusion ID collection")

    def visit(node):
        if isinstance(node, dict):
            for key, child in node.items():
                if (
                    key == "question_id"
                    or key.endswith("question_ids")
                    or key in {"selected", "roles", "source_expansion_order"}
                ):
                    strings(child)
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return found


def _history_paths(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        raise ValueError("history root missing")
    paths = []
    for folder in sorted(runs_root.iterdir()):
        if folder.is_dir() and re.match(r"2026-\d\d-\d\d_", folder.name):
            if folder.name[:10] <= HISTORY_CUTOFF:
                paths.extend(
                    p for p in sorted(folder.iterdir()) if p.name in META_NAMES and p.is_file()
                )
    # These two early diagnostic outputs expose only old preview questions.
    # Their IDs are collected, but no question/answer fields are exported.
    for name in EARLY_PROBES:
        path = runs_root / name
        if not path.is_file():
            raise ValueError("required early exposure metadata missing")
        paths.append(path)
    if not paths:
        raise ValueError("no historical exclusion provenance")
    return sorted(set(paths))


def _exclusions(representation: Path, preview: Path, runs_root: Path, source_sha: str):
    old, initial = _read_json(representation), _read_json(preview)
    if (
        old.get("schema_version") != "growrag-representation-manifest-v1"
        or old.get("official_split") != "train"
        or old.get("source_sha256") != source_sha
        or old.get("complete_train_declared") is not True
        or old.get("checks", {}).get("normalized_duplicates_handled") is not True
        or old.get("exclusions", {}).get("duplicate_policy") != "exclude_all"
    ):
        raise ValueError("same-source complete duplicate/exposure audit is required")
    expansion = old.get("source_expansion_order", [])
    targets = old.get("selected", {}).get("target", [])
    checks = old.get("selected", {}).get("check", [])
    if len(set(expansion)) != 256 or len(set(targets)) != 32 or len(set(checks)) != 24:
        raise ValueError("complete reserved 256+32 pool and 24 check IDs required")
    if not set(checks) <= set(targets) or set(expansion) & set(targets):
        raise ValueError("invalid reserved pool relationships")
    if initial.get("record_count") != 200 or initial.get("official_split") != "train":
        raise ValueError("complete old 200-question preview manifest required")
    if len(_ids(initial)) != 200:
        raise ValueError("old preview coverage incomplete")
    previous = old["exclusions"]
    groups = previous.get("duplicate_normalized_groups")
    if not isinstance(groups, list):
        raise ValueError("duplicate audit groups missing")
    duplicate_ids = {qid for group in groups for qid in group["question_ids"]}
    if len(duplicate_ids) != previous.get("duplicate_normalized_member_count"):
        raise ValueError("duplicate audit count mismatch")
    if not duplicate_ids <= set(previous.get("excluded_union_question_ids", [])):
        raise ValueError("normalized duplicates were not all excluded")
    hashes = set(previous.get("requested_normalized_question_sha256s", []))
    hashes.update(group["normalized_question_sha256"] for group in groups)
    if len(previous.get("requested_normalized_question_sha256s", [])) != 200:
        raise ValueError("old exposed normalized question hashes missing")
    if any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes):
        raise ValueError("invalid normalized question hash")
    paths = [representation, preview, *_history_paths(runs_root)]
    all_ids, provenance = set(), []
    for path in paths:
        ids = _ids(_read_json(path))
        if not ids:
            raise ValueError(f"exclusion source has no recognized IDs: {path.name}")
        all_ids.update(ids)
        provenance.append(
            {
                "path": str(path),
                "sha256": _sha(path),
                "id_count": len(ids),
                "ids_sha256": _digest(ids),
            }
        )
    if not (set(expansion) | set(targets) | _ids(initial)) <= all_ids:
        raise ValueError("prior pools incompletely excluded")
    return all_ids, hashes, provenance, set(checks)


def _skip_value(raw: bytes, start: int) -> int:
    """Lexically skip unrequested JSON values, without decoding their content."""
    if raw[start : start + 1] == b'"':
        match = _STRING.match(raw, start)
        if match is None:
            raise ValueError("unterminated JSON string")
        return match.end()
    if raw[start : start + 1] in (b"{", b"["):
        stack = []
        for token in _TOKEN.finditer(raw, start):
            symbol = token.group()
            if symbol in (b"{", b"["):
                stack.append(symbol)
            elif symbol in (b"}", b"]"):
                if not stack or (stack.pop(), symbol) not in ((b"{", b"}"), (b"[", b"]")):
                    raise ValueError("unbalanced JSON")
                if not stack:
                    return token.end()
        raise ValueError("unterminated JSON structure")
    end = start
    while end < len(raw) and raw[end : end + 1] not in (b",", b"}", b"]"):
        end += 1
    return end


def _project(raw: bytes, wanted: set[str]) -> dict:
    """Decode top-level metadata only; never deserialize skipped gold/context."""
    position, result, seen = 1, {}, set()
    while position < len(raw):
        while raw[position : position + 1] in (b" ", b"\n", b"\r", b"\t"):
            position += 1
        if raw[position : position + 1] == b"}":
            return result
        key_match = _STRING.match(raw, position)
        if key_match is None:
            raise ValueError("invalid record key")
        key = json.loads(key_match.group())
        if key in seen:
            raise ValueError("duplicate record field")
        seen.add(key)
        position = key_match.end()
        while raw[position : position + 1].isspace():
            position += 1
        if raw[position : position + 1] != b":":
            raise ValueError("missing field delimiter")
        position += 1
        while raw[position : position + 1].isspace():
            position += 1
        end = _skip_value(raw, position)
        if key in wanted:
            result[key] = json.loads(raw[position:end])
        position = end
        while raw[position : position + 1].isspace():
            position += 1
        if raw[position : position + 1] == b",":
            position += 1
        elif raw[position : position + 1] != b"}":
            raise ValueError("invalid record separator")
    raise ValueError("unterminated record")


def _record_spans(mapped):
    depth, start, previous_end = 0, None, 0
    for token in _TOKEN.finditer(mapped):
        symbol = token.group()
        if symbol.startswith(b'"'):
            if depth < 2:
                raise ValueError("top-level array must contain objects")
            continue
        if symbol in (b"{", b"["):
            if depth == 0 and symbol != b"[":
                raise ValueError("source must be a JSON array")
            if depth == 1:
                if symbol != b"{" or mapped[previous_end : token.start()].strip() not in (
                    b"",
                    b",",
                ):
                    raise ValueError("invalid top-level record separator")
                start = token.start()
            depth += 1
            if depth == 1:
                previous_end = token.end()
        else:
            depth -= 1
            if depth == 1:
                if symbol != b"}" or start is None:
                    raise ValueError("invalid top-level record")
                if token.end() - start > 4 * 1024 * 1024:
                    raise ValueError("unexpectedly large record")
                yield start, token.end()
                previous_end = token.end()
            elif depth == 0:
                if (
                    symbol != b"]"
                    or mapped[previous_end : token.start()].strip()
                    or mapped[token.end() :].strip()
                ):
                    raise ValueError("invalid array ending")
            elif depth < 0:
                raise ValueError("unbalanced source")
    if depth != 0:
        raise ValueError("truncated source")


def select_fresh_ids(metadata: dict[str, tuple[str, int, int]], excluded: set[str]) -> list[str]:
    """Selection depends ONLY on official type and seeded ID SHA, not outcomes."""
    if not excluded <= metadata.keys():
        raise ValueError("historical IDs missing from the pinned train source")
    pools = {}
    for kind in KINDS:
        pool = [
            qid for qid, (qtype, _, _) in metadata.items() if qtype == kind and qid not in excluded
        ]
        pool.sort(
            key=lambda qid: (hashlib.sha256(f"fresh-dev:{SEED}:{qid}".encode()).hexdigest(), qid)
        )
        if len(pool) < 6:
            raise ValueError("not enough unused questions for fixed 6+6 selection")
        pools[kind] = pool[:6]
    return [qid for pair in zip(pools["bridge"], pools["comparison"], strict=True) for qid in pair]


def _build(
    source: Path, source_provenance: Path, representation: Path, preview: Path, runs_root: Path
):
    source_sha = _sha(source)
    if source_sha != SOURCE_SHA:
        raise ValueError("raw source differs from the pinned complete train mirror")
    mirror = _read_json(source_provenance)
    if (
        mirror.get("official_split") != "train"
        or mirror.get("output_sha256") != source_sha
        or mirror.get("original_cmu_json_bytes") is not False
    ):
        raise ValueError("train mirror provenance mismatch")
    excluded, excluded_hashes, origins, protected = _exclusions(
        representation, preview, runs_root, source_sha
    )
    metadata = {}
    with (
        source.open("rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped,
    ):
        for start, end in _record_spans(mapped):
            row = _project(mapped[start:end], {"_id", "type"})
            qid, kind = row.get("_id"), row.get("type")
            if (
                not isinstance(qid, str)
                or not _QID.fullmatch(qid)
                or qid in metadata
                or kind not in KINDS
            ):
                raise ValueError("invalid/duplicate source ID or official type")
            metadata[qid] = (kind, start, end)
        if len(metadata) != mirror.get("row_count"):
            raise ValueError("source row count differs from provenance")
        selected = select_fresh_ids(metadata, excluded)
        # No other complete record is deserialized, especially no old check row.
        records = [
            json.loads(mapped[metadata[q][1] : metadata[q][2]], object_pairs_hook=_unique)
            for q in selected
        ]
    normalized = [
        hashlib.sha256(normalize_question(row["question"]).encode()).hexdigest() for row in records
    ]
    if len(set(normalized)) != 12 or set(normalized) & excluded_hashes:
        raise ValueError("selected normalized questions overlap: stop, do not resample")
    for record in records:
        example = parse_hotpot_example(record, dataset="hotpotqa-train-fresh-development-sep26")
        if example.gold is None:
            raise ValueError("selected train development example lacks offline feedback")
    if _sha(source) != source_sha or any(_sha(Path(p["path"])) != p["sha256"] for p in origins):
        raise ValueError("inputs changed during preparation")
    manifest = {
        "schema_version": SCHEMA,
        "seed": SEED,
        "dataset": "hotpotqa/hotpot_qa",
        "configuration": "distractor",
        "official_split": "train",
        "role": "development",
        "subset": "new12",
        "new12": True,
        "question_ids": selected,
        "question_type_counts": {"bridge": 6, "comparison": 6},
        "selection_rule": (
            "SHA256(fresh-dev:20260926:ID), independently per official type, "
            "interleaved bridge/comparison"
        ),
        "selection_uses_gold": False,
        "source_path": str(source),
        "source_sha256": source_sha,
        "source_bytes": source.stat().st_size,
        "source_record_count": len(metadata),
        "source_provenance_path": str(source_provenance),
        "source_provenance_sha256": _sha(source_provenance),
        "source_representation": "HF mirror of official train; not CMU original bytes",
        "original_cmu_json_bytes": False,
        "representation_manifest_path": str(representation),
        "preview_manifest_path": str(preview),
        "runs_root": str(runs_root),
        "history_cutoff": HISTORY_CUTOFF,
        "exclusion_sources": origins,
        "excluded_question_ids": sorted(excluded),
        "excluded_id_count": len(excluded),
        "excluded_ids_sha256": _digest(excluded),
        "excluded_normalized_question_sha256s": sorted(excluded_hashes),
        "normalized_duplicate_audit_reused": True,
        "normalized_duplicate_notice": (
            "Same-source full-corpus exclude_all audit; old check text/gold never projected"
        ),
        "protected_check_id_count": len(protected),
        "old_check_records_projected": False,
        "selected_normalized_question_sha256s": normalized,
        "eligible_type_counts": dict(
            Counter(kind for qid, (kind, _, _) in metadata.items() if qid not in excluded)
        ),
        "selected_records_file": "selected_records.json",
        "selected_records_count": 12,
        "selected_records_contains_gold": True,
        "selected_records_runtime_safe": False,
        "official_dev_test_used": False,
        "semantic_near_duplicates_checked": False,
        "document_disjointness_checked": False,
        "notice": (
            "New train development sample, not test; feedback remains separate from runtime "
            "prompts. Historical candidate pools excluded even if never executed."
        ),
    }
    return manifest, records


def prepare_fresh_dev(
    source: Path,
    source_provenance: Path,
    representation: Path,
    preview: Path,
    runs_root: Path,
    output_dir: Path,
) -> Path:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output directory already exists; never overwrite")
    paths = [
        Path(p).resolve(strict=True)
        for p in (source, source_provenance, representation, preview, runs_root)
    ]
    manifest, records = _build(*paths)
    data = (json.dumps(records, ensure_ascii=False, indent=2) + "\n").encode()
    manifest["selected_records_sha256"] = hashlib.sha256(data).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "selected_records.json").open("xb") as handle:
        handle.write(data)
    result = output_dir / "manifest.json"
    with result.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def load_fresh_dev(manifest_path: Path) -> tuple[tuple[HotpotExample, ...], dict]:
    """Revalidate every hash and seeded selection; parse only selected new rows."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA
        or manifest.get("selected_records_file") != "selected_records.json"
    ):
        raise ValueError("not the frozen fresh development protocol")
    paths = [
        Path(manifest[name]).resolve(strict=True)
        for name in (
            "source_path",
            "source_provenance_path",
            "representation_manifest_path",
            "preview_manifest_path",
            "runs_root",
        )
    ]
    expected, selected = _build(*paths)
    data_path = manifest_path.parent / "selected_records.json"
    expected["selected_records_sha256"] = _sha(data_path)
    if manifest != expected or _read_json(data_path) != selected:
        raise ValueError("manifest/subset changed or no longer matches the frozen protocol")
    examples = tuple(
        parse_hotpot_example(row, dataset="hotpotqa-train-fresh-development-sep26")
        for row in selected
    )
    provenance = {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha(manifest_path),
        "all_inputs_verified": True,
    }
    return examples, provenance


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument(
        "--source",
        type=Path,
        default=Path("data/hotpotqa/official_train_v1_1/hotpot_train_v1.1_hf_mirror.json"),
    )
    prepare.add_argument(
        "--source-provenance",
        type=Path,
        default=Path("data/hotpotqa/official_train_v1_1/mirror_provenance.json"),
    )
    prepare.add_argument(
        "--representation",
        type=Path,
        default=Path("data/hotpotqa/representation_sep11_v1/manifest.json"),
    )
    prepare.add_argument(
        "--preview", type=Path, default=Path("data/hotpotqa/train_preview_200_v1/manifest.json")
    )
    prepare.add_argument("--runs-root", type=Path, default=Path("runs"))
    prepare.add_argument("--output-dir", type=Path, required=True)
    load = sub.add_parser("load")
    load.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_fresh_dev(
            args.source,
            args.source_provenance,
            args.representation,
            args.preview,
            args.runs_root,
            args.output_dir,
        )
        print(
            json.dumps(
                {
                    "manifest_path": str(result),
                    "role": "development",
                    "official_split": "train",
                    "questions": 12,
                }
            )
        )
    else:
        examples, provenance = load_fresh_dev(args.manifest)
        print(
            json.dumps(
                {
                    "questions": len(examples),
                    "question_ids": provenance["question_ids"],
                    "role": provenance["role"],
                    "all_inputs_verified": True,
                }
            )
        )


if __name__ == "__main__":
    main()
