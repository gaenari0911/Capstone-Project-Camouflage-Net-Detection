from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from capstone_lab.config import canonical_json
from capstone_lab.errors import ConfigError
from capstone_lab.synthesis.smoke import SmokeConfig, run_actual_smoke_campaign


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    spec = json.loads(parser.parse_args(argv).spec.read_text(encoding="utf-8"))
    output = Path(spec["output_dir"]).resolve(strict=False)
    allowed = Path(spec["allowed_output_root"]).resolve(strict=True)
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise ConfigError("CPU profile output is outside the S6 artifact root") from exc
    started = time.time()
    summary = run_actual_smoke_campaign(
        project_root=Path(spec["project_root"]),
        source_manifest_path=Path(spec["source_manifest"]),
        resolution_manifest_path=Path(spec["resolution_manifest"]),
        output_dir=output / "synthesis",
        workers=int(spec["workers"]),
        config=SmokeConfig(**spec["smoke_config"]),
        enforce_smoke_minimum=False,
    )
    result = {
        "schema_version": 1,
        "status": "SUCCEEDED" if summary["status"] == "VERIFIED" else "FAILED",
        "profile_scope": "actual_S4_synthesis_smoke",
        "job_id": spec["job_id"], "pid": os.getpid(),
        "started_unix": started, "ended_unix": time.time(),
        "contract": spec["contract"],
        "s4_deterministic_sha256": summary["deterministic_sha256"],
        "s4_summary": summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(canonical_json(result) + "\n", encoding="utf-8")
    return 0 if result["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
