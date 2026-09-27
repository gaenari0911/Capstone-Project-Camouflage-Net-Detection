from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internal S1 fake job")
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--input-hash", required=True)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args(argv)

    time.sleep(args.duration)
    if args.fail:
        print(f"intentional failure: {args.job} attempt {args.attempt}", flush=True)
        return 23
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    payload = {
        "schema_version": 1,
        "campaign": args.campaign,
        "job_id": args.job,
        "attempt": args.attempt,
        "config_hash": args.config_hash,
        "input_hash": args.input_hash,
        "pid": os.getpid(),
        "kind": "fake",
        "validated": True,
    }
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, args.output)
    print(f"completed {args.job} attempt {args.attempt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
