"""Build an offline frozen manifest from a complete, explicitly chosen train file.

No download or model calls. The selected raw records are a local labelled audit
artifact, not runtime input: loaders must keep their gold out of model prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data_protocol import normalize_question
from .hotpot import HotpotExample, parse_hotpot_example
from .representation_manifest import build_representation_manifest

EXPECTED_TRAIN_ROWS = 90_447
EXPECTED_PREVIEW_ROWS = 200
SOURCE_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
DATASET_LABEL = "hotpotqa-distractor-train-v1.1"


def _read_json_file(path: Path) -> tuple[list[dict], str]:
    # Hash exactly the bytes parsed, avoiding a file-change race between separate
    # checksum and parse passes. Do not retain the raw byte buffer afterwards.
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    records = json.loads(raw)
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"{path.name}: expected a JSON array of records")
    return records, digest


def _metadata_example(record: dict, *, dataset: str) -> HotpotExample:
    """Validate raw context shape but do not allocate millions of Evidence rows."""
    context = record.get("context")
    if not isinstance(context, list) or not context:
        raise ValueError("raw train record must contain a nonempty context array")
    titles = set()
    for paragraph in context:
        if (
            not isinstance(paragraph, list)
            or len(paragraph) != 2
            or not isinstance(paragraph[0], str)
            or not paragraph[0].strip()
            or not isinstance(paragraph[1], list)
            or not all(isinstance(sentence, str) for sentence in paragraph[1])
        ):
            raise ValueError("raw context must contain [nonempty title, sentence strings]")
        if paragraph[0] in titles:
            raise ValueError("duplicate raw context title makes sentence references ambiguous")
        titles.add(paragraph[0])
    projected = {
        name: record[name]
        for name in ("_id", "question", "answer", "supporting_facts", "type", "level")
        if name in record
    }
    projected["context"] = []
    return parse_hotpot_example(projected, dataset=dataset)


def _preview_exclusions(records: list[dict]) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    ids = []
    by_normalized = {}
    for record in records:
        qid, question = record.get("_id"), record.get("question")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError("preview exposure records require nonempty _id")
        normalized = normalize_question(question)
        ids.append(qid)
        # The filter treats equivalent exposed strings as one constraint, not a
        # selected training example. Record the collapse counts transparently.
        by_normalized.setdefault(normalized, question)
    unique_ids = sorted(set(ids))
    texts = tuple(by_normalized[key] for key in sorted(by_normalized))
    return (
        tuple(unique_ids),
        texts,
        {
            "policy": "Exclude every ID and normalized question from the whole old preview.",
            "preview_record_count": len(records),
            "unique_exposed_id_count": len(unique_ids),
            "unique_exposed_normalized_question_count": len(texts),
            "repeated_preview_id_count": len(ids) - len(unique_ids),
            "repeated_preview_normalized_question_count": len(records) - len(texts),
            "notice": (
                "Conservative expansion of prior-exposure exclusion; no split-role reassignment."
            ),
        },
    )


def _mirror_provenance(path: Path, source_sha256: str, row_count: int) -> dict:
    raw = path.read_bytes()
    metadata = json.loads(raw)
    if not isinstance(metadata, dict) or any(
        metadata.get(key) != value
        for key, value in {
            "schema_version": "growrag-hotpot-mirror-provenance-v1",
            "dataset": "hotpotqa/hotpot_qa",
            "configuration": "distractor",
            "official_split": "train",
            "row_count": row_count,
            "output_sha256": source_sha256,
            "original_cmu_json_bytes": False,
        }.items()
    ):
        raise ValueError("mirror provenance must match the complete train file and its exact SHA")
    if metadata.get("source_url") != (
        "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/train"
    ):
        raise ValueError("unsupported mirror source; inspect new provenance explicitly")
    return {
        "source_provenance_path": str(path.resolve()),
        "source_provenance_sha256": hashlib.sha256(raw).hexdigest(),
        "source_provenance": metadata,
        "source_url": metadata["source_url"],
        "original_cmu_json_bytes": False,
        "source_representation": "Hugging Face complete train Parquet-to-JSON mirror",
    }


def write_representation_manifest(
    data_path: Path,
    output_dir: Path,
    exclude_preview_path: Path,
    *,
    expected_row_count: int = EXPECTED_TRAIN_ROWS,
    synthetic: bool = False,
    source_count: int = 64,
    source_cap: int = 256,
    target_count: int = 32,
    debug_count: int = 8,
    source_provenance_path: Path | None = None,
) -> dict:
    """Write new manifest/selected records only after all preflight checks pass.

    Production requires the declared official v1.1 row count and all old 200
    preview rows. Small synthetic tests must explicitly opt in and are labelled
    as synthetic in outputs; changing a count cannot quietly disguise a subset.
    """
    if type(synthetic) is not bool:
        raise ValueError("synthetic must be a boolean")
    if synthetic and source_provenance_path is not None:
        raise ValueError("synthetic fixtures cannot claim public-mirror provenance")
    if type(expected_row_count) is not int or expected_row_count <= 0:
        raise ValueError("expected_row_count must be a positive integer")
    if not synthetic and expected_row_count != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            "production requires expected_row_count=90447; use explicit synthetic mode"
        )
    data_path = Path(data_path).resolve()
    exclude_preview_path = Path(exclude_preview_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists; refusing overwrite: {output_dir}")
    records, source_sha256 = _read_json_file(data_path)
    if len(records) != expected_row_count:
        raise ValueError(
            f"expected {expected_row_count} complete train records, found {len(records)}; "
            "do not substitute a partial file or dev/test"
        )
    provenance = (
        _mirror_provenance(Path(source_provenance_path), source_sha256, len(records))
        if source_provenance_path is not None
        else {}
    )
    preview, preview_sha256 = _read_json_file(exclude_preview_path)
    if not preview or (not synthetic and len(preview) != EXPECTED_PREVIEW_ROWS):
        raise ValueError("production requires all 200 old preview exposure records")
    excluded_ids, excluded_texts, exposure = _preview_exclusions(preview)
    label = "hotpotqa-synthetic-train" if synthetic else DATASET_LABEL
    examples = tuple(_metadata_example(record, dataset=label) for record in records)
    manifest = build_representation_manifest(
        examples,
        source_sha256=source_sha256,
        official_split="train",
        complete_train_declared=True,
        source_count=source_count,
        source_cap=source_cap,
        target_count=target_count,
        debug_count=debug_count,
        excluded_question_ids=excluded_ids,
        excluded_question_texts=excluded_texts,
        duplicate_policy="exclude_all",
    )
    by_id = {record["_id"]: record for record in records}
    selected_ids = manifest["source_expansion_order"] + manifest["selected"]["target"]
    assert len(set(selected_ids)) == len(selected_ids)
    selected = [by_id[qid] for qid in selected_ids]
    selected_bytes = (json.dumps(selected, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest.update(
        input_path=str(data_path),
        data_version="synthetic-test-only" if synthetic else "HotpotQA v1.1 train",
        source_url=None if synthetic else SOURCE_URL,
        source_url_is_provenance_declaration=True,
        official_authenticity_independently_verified=False,
        synthetic_data=synthetic,
        expected_row_count=expected_row_count,
        source_bytes_verified_by_builder=True,
        raw_context_shape_validated=True,
        metadata_only_projection=True,
        selected_records_file="selected_records.json",
        selected_records_sha256=hashlib.sha256(selected_bytes).hexdigest(),
        selected_records_count=len(selected),
        selected_records_scope="Frozen source-cap expansion order followed by calibration targets.",
        selected_records_contains_gold=True,
        selected_records_runtime_safe=False,
        selected_records_notice=(
            "LOCAL AUDIT DATA: raw questions, context and gold; never pass this file "
            "directly to a runtime component. Use only selected.source IDs to run sources."
        ),
        excluded_preview_path=str(exclude_preview_path),
        excluded_preview_sha256=preview_sha256,
        prior_exposure_exclusion=exposure,
        provenance_notice=(
            "CLI hashed the actual source bytes and checked the expected row count. "
            "The source URL/official identity are declarations, not cryptographic authenticity."
        ),
    )
    manifest["checks"]["expected_row_count_verified"] = True
    if provenance:
        manifest.update(provenance)
        manifest["provenance_notice"] = (
            "Complete train-split mirror converted from public Hugging Face Parquet. "
            "Source SHA identifies the converted JSON, NOT original CMU JSON bytes. "
            "The conversion provenance binds individual shard SHAs and row counts; "
            "it does not prove byte or semantic identity with the unavailable CMU file."
        )
    # Row count and structural validation do not prove the file's provenance.
    manifest["checks"]["full_file_completeness_independently_verified"] = False
    if synthetic:
        manifest["sampling_notice"] = "SYNTHETIC TEST ONLY; not an official dataset manifest."
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "selected_records.json").write_bytes(selected_bytes)
    (output_dir / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-preview", type=Path, required=True)
    parser.add_argument(
        "--synthetic", action="store_true", help="Explicitly mark test fixtures only"
    )
    parser.add_argument("--expected-row-count", type=int, default=EXPECTED_TRAIN_ROWS)
    parser.add_argument("--source-count", type=int, default=64)
    parser.add_argument("--source-cap", type=int, default=256)
    parser.add_argument("--target-count", type=int, default=32)
    parser.add_argument("--debug-count", type=int, default=8)
    parser.add_argument("--source-provenance", type=Path, default=None)
    args = parser.parse_args()
    try:
        manifest = write_representation_manifest(
            args.data,
            args.output,
            args.exclude_preview,
            expected_row_count=args.expected_row_count,
            synthetic=args.synthetic,
            source_count=args.source_count,
            source_cap=args.source_cap,
            target_count=args.target_count,
            debug_count=args.debug_count,
            source_provenance_path=args.source_provenance,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "manifest": str(args.output.resolve() / "manifest.json"),
                "synthetic_data": manifest["synthetic_data"],
                "selected_counts": manifest["selected_counts"],
                "excluded_union_count": manifest["exclusions"]["excluded_union_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
