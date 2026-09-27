from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import contract, output_path, readiness_plan, read_json
from .io import atomic_json
from .supervisor import launch, snapshot, supervise


def main():
    parser = argparse.ArgumentParser(prog="python -m capstone_lab.campaign")
    parser.add_argument("action", choices=("plan", "run", "resume", "serve", "status", "logs", "pause", "report"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--token")
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.action == "plan":
        result = readiness_plan(root)
    else:
        if args.config is None:
            parser.error("--config is required")
        config_path = (root / args.config).resolve()
        if args.action in {"run", "resume"}:
            result = launch(root, config_path)
        elif args.action == "serve":
            if not args.token:
                parser.error("serve requires internal --token")
            result = supervise(root, config_path, args.token)
        else:
            config = read_json(config_path)
            run = output_path(root, config["output"])
            if args.action == "pause":
                token = read_json(run / "token.json")["token"]
                atomic_json(run / "control.json", {"token": token, "action": "pause"})
                result = {"status": "PAUSE_REQUESTED_EPOCH_BOUNDARY_FOR_TRAINING_OTHER_JOBS_FINISH", "run": str(run)}
            else:
                result = snapshot(run)
                if args.action == "logs":
                    result = {"run": str(run), "log_files": [str(p) for p in sorted(run.rglob("*.log"))]}
                elif args.action == "report":
                    # Read-only report command: supervisor alone writes report.json.
                    result["report_path"] = str(run / "report.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
