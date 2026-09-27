from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from capstone_lab.config import sha256_file
from .contracts import code_records, output_path
from .io import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    output = output_path(root, args.output)
    output.mkdir(parents=True, exist_ok=False)
    before = code_records(root)
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    with (output / "tests.log").open("xb") as log:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, shell=False)
    log_text = (output / "tests.log").read_text(encoding="utf-8", errors="replace")
    count = re.search(r"Ran (\d+) tests? in", log_text)
    after = code_records(root)
    summary = {"status": "VERIFIED" if result.returncode == 0 and count and before == after else "FAILED",
               "command": command, "return_code": result.returncode,
               "tests": int(count[1]) if count else None,
               "log_sha256": sha256_file(output / "tests.log"),
               "code_before": before, "code_after": after}
    atomic_json(output / "summary.json", summary, immutable=True)
    print(summary["status"], summary["tests"])
    return 0 if summary["status"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
