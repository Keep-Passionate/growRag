"""Write a Gate 1 offline candidate-set oracle report from a CSV file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from growrag.evaluation.oracle import analyze_oracle, read_oracle_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the offline candidate-set ceiling and pair-harm rate of historical "
            "query reuse. The result is not a deployable routing-label file."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input candidate-pair CSV")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    parser.add_argument(
        "--good-threshold",
        type=float,
        default=0.5,
        help="Score threshold defining good/bad for harm statistics (default: 0.5)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_oracle_csv(args.input)
    analysis = analyze_oracle(rows, good_threshold=args.good_threshold)
    payload = analysis.to_dict()
    payload["source_csv"] = str(args.input.resolve())

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Offline candidate-set oracle report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
