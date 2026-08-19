"""Allow ``python -m growrag`` to invoke the command-line interface."""

from growrag.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
