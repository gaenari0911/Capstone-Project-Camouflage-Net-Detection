from __future__ import annotations

import argparse
import sys
from pathlib import Path

from capstone_lab.errors import CapstoneLabError

from .dualhead import run_explicit_worker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m capstone_lab.training.worker")
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run_explicit_worker(args.spec.resolve(strict=True))
    except (CapstoneLabError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
