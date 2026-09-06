"""Explicit mock-only CLI. A missing real backend never activates this demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from growrag.experiments.mock import run_demo
from growrag.experiments.paired_runner import load_run, write_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="NEW directory for a hand-written mock run")
    mode.add_argument("--replay", type=Path, help="Read saved run.json; no components are executed")
    args = parser.parse_args()
    try:
        if args.replay:
            report = load_run(args.replay)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            destination = write_run(run_demo(), args.output)
            print("MOCK ONLY: synthetic fixture; no LLM, no API, no HotpotQA score.")
            print(f"Result saved to: {destination.resolve()}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Experiment error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
