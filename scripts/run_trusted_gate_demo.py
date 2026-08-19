"""Compatibility wrapper for the packaged GrowRAG gate demonstration."""

from __future__ import annotations

from growrag.cli import main as growrag_main


def main() -> int:
    return growrag_main(["demo"])


if __name__ == "__main__":
    raise SystemExit(main())
