"""Frozen train-only HotpotQA development sample and role manifests.

The rows service is a public mirror, not an experiment model API. Taking its
first 200 training rows is a convenience sample for debugging, NOT a random
sample of the complete benchmark. Official validation/test are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from .hotpot import parse_hotpot_example

SPLIT_VERSION = "growrag-split-v1"
DATASET_ID = "hotpotqa/hotpot_qa"
ROLES = ("memory_seed", "selector_train", "calibration_dev")


def normalize_question(question: str) -> str:
    if not isinstance(question, str):
        raise ValueError("question must be text")
    value = unicodedata.normalize("NFKC", question).casefold()
    value = " ".join(re.findall(r"[^\W_]+", value, flags=re.UNICODE))
    if not value:
        raise ValueError("question must have alphanumeric content")
    return value


def role_for_question(question: str, *, seed: int = 42) -> str:
    """Same normalized question always has one role, even with another ID."""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    identity = f"{SPLIT_VERSION}:{seed}:{normalize_question(question)}"
    bucket = int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16) % 10000
    return ROLES[0] if bucket < 8000 else ROLES[1] if bucket < 9000 else ROLES[2]


def from_hf_row(row: dict) -> dict:
    """Convert columnar list fields without changing gold or sentence order."""
    if not isinstance(row, dict):
        raise ValueError("row must be an object")
    try:
        result = {
            "_id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
            "level": row["level"],
            "context": [
                list(pair)
                for pair in zip(row["context"]["title"], row["context"]["sentences"], strict=True)
            ],
            "supporting_facts": [
                list(pair)
                for pair in zip(
                    row["supporting_facts"]["title"],
                    row["supporting_facts"]["sent_id"],
                    strict=True,
                )
            ],
        }
        parse_hotpot_example(result, dataset="hotpotqa-distractor-train-preview")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid or inconsistent Hotpot mirror row") from error
    return result


def build_manifest(
    records: list[dict], *, seed: int = 42, source_count: int = 8, target_count: int = 8
) -> dict:
    if not records or len({r["_id"] for r in records}) != len(records):
        raise ValueError("records must be nonempty with unique question IDs")
    if any(type(n) is not int or n <= 0 for n in (source_count, target_count)):
        raise ValueError("source and target sizes must be positive integers")
    roles = {role: [] for role in ROLES}
    groups = {}
    for record in records:
        parse_hotpot_example(record, dataset="hotpotqa-distractor-train-preview")
        role = role_for_question(record["question"], seed=seed)
        roles[role].append(record["_id"])
        groups[record["_id"]] = normalize_question(record["question"])
    for values in roles.values():
        values.sort(key=lambda qid: hashlib.sha256(f"selection:{seed}:{qid}".encode()).hexdigest())
    selected = {}
    for role, count in ((ROLES[0], source_count), (ROLES[2], target_count)):
        selected[role], seen = [], set()
        for qid in roles[role]:
            if groups[qid] not in seen:
                selected[role].append(qid)
                seen.add(groups[qid])
            if len(selected[role]) == count:
                break
        if len(selected[role]) != count:
            raise ValueError("frozen sample lacks enough unique questions for requested role")
    return {
        "schema_version": "growrag-hotpot-manifest-v1",
        "dataset": DATASET_ID,
        "configuration": "distractor",
        "official_split": "train",
        "split_version": SPLIT_VERSION,
        "seed": seed,
        "role_weights": dict(zip(ROLES, (0.8, 0.1, 0.1), strict=True)),
        "role_counts": {role: len(values) for role, values in roles.items()},
        "roles": roles,
        "selected": selected,
        "record_count": len(records),
        "sampling_notice": "first 200 mirror train rows; convenience debug sample, not benchmark",
        "official_validation_used": False,
        "official_test_used": False,
    }


def download_preview(output_dir: Path) -> Path:
    """Download exactly two predeclared pages once; fail instead of replacing rows."""
    output_dir.mkdir(parents=True, exist_ok=False)
    records, provenance = [], []
    expected_total = None
    for offset in (0, 100):
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa"
            f"&config=distractor&split=train&offset={offset}&length=100"
        )
        with urllib.request.urlopen(url, timeout=45) as response:
            raw = response.read(16_000_001)
        if len(raw) > 16_000_000:
            raise ValueError("public dataset page exceeded size cap")
        path = output_dir / f"raw_rows_{offset}.json"
        path.write_bytes(raw)
        data = json.loads(raw)
        total = data["num_rows_total"]
        if expected_total is not None and total != expected_total:
            raise ValueError("dataset changed between pages")
        expected_total = total
        rows = data["rows"]
        if len(rows) != 100 or any(r["row_idx"] != offset + i for i, r in enumerate(rows)):
            raise ValueError("missing or unexpected public dataset rows")
        if any(r.get("truncated_cells") for r in rows):
            raise ValueError("truncated rows cannot be used for evaluation")
        records.extend(from_hf_row(r["row"]) for r in rows)
        provenance.append(
            {
                "url": url,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "retrieved_at": datetime.now(UTC).isoformat(),
                "row_count": len(rows),
            }
        )
        print(f"Downloaded public train rows {offset}-{offset + 99}", flush=True)
    encoded = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
    data_path = output_dir / "hotpot_train_preview_200.json"
    data_path.write_bytes(encoded)
    manifest = build_manifest(records)
    manifest.update(
        data_file=data_path.name,
        data_sha256=hashlib.sha256(encoded).hexdigest(),
        public_mirror_total_train_rows=expected_total,
        provenance=provenance,
        license="CC-BY-SA-4.0",
        source_project="https://github.com/hotpotqa/hotpot",
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"role_counts": manifest["role_counts"], "manifest": str(manifest_path)}))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_download:
        parser.error("explicit --allow-download required")
    download_preview(args.output)


if __name__ == "__main__":
    main()
