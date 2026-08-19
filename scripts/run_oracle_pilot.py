"""Compatibility wrapper for the packaged GrowRAG oracle command."""

from __future__ import annotations

import sys

from growrag.cli import main as growrag_main


def main() -> int:
    return growrag_main(["oracle", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
