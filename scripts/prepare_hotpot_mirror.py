"""Download a public full-train mirror and convert its Parquet records losslessly.

Requires pyarrow. This is a mirror conversion, NOT the original CMU JSON bytes.
Run explicitly; importing this module never downloads anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

API_URL = "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/train"
OFFICIAL_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
EXPECTED_ROWS = 90_447


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_shape(row: dict) -> dict:
    """Preserve all sentences and their indices, including empty sentences."""
    result = {name: row[name] for name in ("question", "answer", "type", "level")}
    result["_id"] = row["id"]
    result["context"] = [
        [title, sentences]
        for title, sentences in zip(
            row["context"]["title"], row["context"]["sentences"], strict=True
        )
    ]
    result["supporting_facts"] = [
        [title, sent_id]
        for title, sent_id in zip(
            row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"], strict=True
        )
    ]
    return result


def download(url: str, destination: Path) -> dict:
    if destination.exists() or destination.with_suffix(".part").exists():
        raise FileExistsError(f"refusing to overwrite existing download: {destination}")
    partial = destination.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("xb") as handle:
        # Redirects may contain temporary signed credentials. Do not persist
        # their query strings; reproducibility uses the public API URL and SHA.
        resolved = urlsplit(response.geturl())
        resolved_url = urlunsplit((resolved.scheme, resolved.netloc, resolved.path, "", ""))
        etag = response.headers.get("ETag")
        count = 0
        next_report = 32 * 1024 * 1024
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)
            count += len(chunk)
            if count >= next_report:
                print(f"{destination.name}: {count // (1024 * 1024)} MiB", flush=True)
                next_report += 32 * 1024 * 1024
    partial.rename(destination)
    return {
        "url": url,
        "resolved_url_without_query": resolved_url,
        "etag": etag,
        "file": destination.name,
        "bytes": count,
        "sha256": digest_file(destination),
    }


def prepare(output_dir: Path) -> dict:
    import pyarrow
    import pyarrow.parquet as parquet

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("output must be absent or empty; existing artifacts are preserved")
    output_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API_URL, timeout=60) as response:
        index_bytes = response.read()
    urls = json.loads(index_bytes)
    if (
        not isinstance(urls, list)
        or len(urls) != 2
        or any(url != f"{API_URL}/{index}.parquet" for index, url in enumerate(urls))
    ):
        raise ValueError("unexpected mirror shard list; inspect a changed source explicitly")
    (output_dir / "parquet_index.json").write_bytes(index_bytes)
    shards = []
    for index, url in enumerate(urls):
        print(f"Downloading public train shard {index + 1}/{len(urls)}", flush=True)
        shards.append(download(url, output_dir / f"train-{index}.parquet"))
    destination = output_dir / "hotpot_train_v1.1_hf_mirror.json"
    partial = output_dir / "hotpot_train_v1.1_hf_mirror.json.part"
    count, ids = 0, set()
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("[\n")
        for shard in shards:
            reader = parquet.ParquetFile(output_dir / shard["file"])
            shard["row_count"] = reader.metadata.num_rows
            for batch in reader.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    record = official_shape(row)
                    qid = record["_id"]
                    if not isinstance(qid, str) or not qid.strip() or qid in ids:
                        raise ValueError("invalid or duplicate question ID in public mirror")
                    ids.add(qid)
                    if count:
                        handle.write(",\n")
                    json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
                    count += 1
            print(f"Converted {count} records", flush=True)
        handle.write("\n]\n")
    if count != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} complete train rows, found {count}")
    partial.rename(destination)
    metadata = {
        "schema_version": "growrag-hotpot-mirror-provenance-v1",
        "dataset": "hotpotqa/hotpot_qa",
        "configuration": "distractor",
        "official_split": "train",
        "input_dataset_label": "hotpotqa-distractor-train-v1.1-hf-split-mirror",
        "data_version": "HotpotQA v1.1 train split, Hugging Face Parquet-to-JSON mirror",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "source_url": API_URL,
        "upstream_official_format_url": OFFICIAL_URL,
        "original_cmu_json_bytes": False,
        "row_count": count,
        "index_file": "parquet_index.json",
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "shards": shards,
        "output_file": destination.name,
        "output_bytes": destination.stat().st_size,
        "output_sha256": digest_file(destination),
        "pyarrow_version": pyarrow.__version__,
        "conversion_script_sha256": digest_file(Path(__file__)),
        "conversion": (
            "Read shards in API order and rows in Parquet order; map id to _id; "
            "zip context.title/sentences and supporting_facts.title/sent_id strictly; "
            "preserve sentence order, empty strings, question, answer, type and level."
        ),
        "authenticity_notice": (
            "Public HF mirror of the train split, validated by row count and IDs, "
            "not a byte-for-byte or semantic comparison against unavailable CMU JSON. "
            "SHA values identify the downloaded shards and converted file, not a CMU release hash."
        ),
    }
    (output_dir / "mirror_provenance.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps({key: result[key] for key in ("row_count", "output_file", "output_sha256")}))


if __name__ == "__main__":
    main()
