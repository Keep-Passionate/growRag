"""Download the public official-dev split mirror; preserve data without displaying labels.

独立下载器，不做抽题、训练、评分。原CMU链接目前无法访问，使用与train相同的
公开HF镜像，记录出处及SHA，不声称镜像与原CMU文件逐字节相同。
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from prepare_hotpot_mirror import digest_file, download, official_shape

API_URL = "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/distractor/validation"
OFFICIAL_URL = "https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
EXPECTED_ROWS = 7405


def prepare(output: Path) -> dict:
    import pyarrow.parquet as parquet

    output.mkdir(parents=True, exist_ok=False)
    with urllib.request.urlopen(API_URL, timeout=45) as response:
        urls = json.loads(response.read(100_000))
    if (
        not isinstance(urls, list)
        or not 1 <= len(urls) <= 3
        or any(url != f"{API_URL}/{i}.parquet" for i, url in enumerate(urls))
    ):
        raise ValueError("unexpected dev mirror shard list")
    shards = [download(url, output / f"dev-{i}.parquet") for i, url in enumerate(urls)]
    destination = output / "hotpot_dev_distractor_v1_hf_mirror.json"
    partial = destination.with_suffix(".json.part")
    ids, count = set(), 0
    with partial.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("[\n")
        for shard in shards:
            reader = parquet.ParquetFile(output / shard["file"])
            for batch in reader.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    record = official_shape(row)
                    qid = record["_id"]
                    if not isinstance(qid, str) or not qid or qid in ids:
                        raise ValueError("invalid or duplicate dev ID")
                    ids.add(qid)
                    if count:
                        handle.write(",\n")
                    json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
                    count += 1
        handle.write("\n]\n")
    if count != EXPECTED_ROWS:
        raise ValueError("unexpected official-dev mirror row count")
    partial.rename(destination)
    metadata = {
        "schema_version": "growrag-hotpot-dev-mirror-v1",
        "official_split": "dev",
        "hf_split": "validation",
        "source_url": API_URL,
        "upstream_official_format_url": OFFICIAL_URL,
        "original_cmu_json_bytes": False,
        "row_count": count,
        "shards": shards,
        "output_file": destination.name,
        "output_sha256": digest_file(destination),
        "created_utc": datetime.now(UTC).isoformat(),
        "script_sha256": digest_file(Path(__file__)),
        "labels_used_for_training_or_selection": False,
    }
    with (output / "mirror_provenance.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
