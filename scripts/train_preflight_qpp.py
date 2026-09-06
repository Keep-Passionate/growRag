"""Run the fixed tiny ORIGINAL-query QPP diagnostic locally, without any API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from growrag.preflight_qpp import train_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train_from_manifest(args.manifest, args.output)
    print(
        json.dumps(
            {"notice": report["notice"], "metrics": report["metrics"], "output": str(args.output)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
