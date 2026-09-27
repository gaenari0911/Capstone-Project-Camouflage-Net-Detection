from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import CampaignConfig, load_config
from .doctor import inspect_environment
from .environment_resolution import resolve_object_environments_from_disk
from .errors import CapstoneLabError
from .manifests import build_s2_manifests
from .scheduler import Scheduler
from .state import StateStore
from .synthesis.executor import run_fixture_campaign
from .synthesis.smoke import (
    SmokeConfig,
    build_smoke_source_manifest,
    run_actual_smoke_campaign,
)


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _db_path(config: CampaignConfig) -> Path:
    return config.artifact_root / "_state" / "campaigns.sqlite3"


def _plan(config: CampaignConfig) -> dict[str, Any]:
    return {
        "campaign": config.campaign_name,
        "status": config.status,
        "runnable": config.status == "frozen" and config.unresolved_field is None,
        "unresolved_field": config.unresolved_field,
        "config_hash": config.config_hash,
        "input_hash": config.input_hash,
        "limits": {"cpu": config.limits.cpu, "gpu_mib": config.limits.gpu_mib},
        "jobs": [
            {
                "job_id": job.job_id,
                "dependencies": list(job.dependencies),
                "resources": {"cpu": job.resources.cpu, "gpu_mib": job.resources.gpu_mib},
                "max_attempts": job.max_attempts,
            }
            for job in config.jobs
        ],
    }


def _status(config: CampaignConfig) -> dict[str, Any]:
    path = _db_path(config)
    if not path.exists():
        return {"campaign": config.campaign_name, "status": "NOT_STARTED", "jobs": []}
    with StateStore(path) as store:
        rows = store.rows(config.campaign_name)
        return {
            "campaign": config.campaign_name,
            "jobs": [
                {
                    "job_id": row["job_id"],
                    "state": row["state"],
                    "attempt_count": row["attempt_count"],
                    "last_error": row["last_error"],
                    "artifact_path": row["artifact_path"],
                }
                for row in rows
            ],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m capstone_lab")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="read-only environment and input inspection")
    doctor.add_argument("--config", type=Path)
    plan = commands.add_parser("plan", help="validate and display a campaign without writing state")
    plan.add_argument("--config", type=Path, required=True)
    manifest = commands.add_parser("manifest", help="deterministic, read-only source manifest tools")
    manifest_commands = manifest.add_subparsers(dest="manifest_command", required=True)
    low187 = manifest_commands.add_parser("build-low187")
    low187.add_argument("--project-root", type=Path, default=Path("."))
    low187.add_argument(
        "--inventory",
        type=Path,
        default=Path("docs/recovery_audit/current_split_inventory.csv"),
    )
    low187.add_argument(
        "--verification",
        type=Path,
        default=Path("docs/recovery_audit/verification.json"),
    )
    low187.add_argument("--output-dir", type=Path, default=Path("manifests"))
    low187.add_argument("--seed", type=int, default=42)
    low187.add_argument("--workers", type=int, default=4)
    resolve_environments = manifest_commands.add_parser("resolve-environments")
    resolve_environments.add_argument("--project-root", type=Path, default=Path("."))
    resolve_environments.add_argument(
        "--review",
        type=Path,
        default=Path("manifests/object_environment_review_v1.jsonl"),
    )
    resolve_environments.add_argument(
        "--object-pool",
        type=Path,
        default=Path("Dataset/object_pool"),
    )
    resolve_environments.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/object_environment_resolution_v1.jsonl"),
    )
    synthesis = commands.add_parser("synthesis", help="S3/S4 synthesis tools")
    synthesis_commands = synthesis.add_subparsers(
        dest="synthesis_command", required=True
    )
    fixture = synthesis_commands.add_parser("fixture")
    fixture.add_argument("--project-root", type=Path, default=Path("."))
    fixture.add_argument(
        "--fixture", type=Path, default=Path("tests/fixtures/s3_fixture.json")
    )
    fixture.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/s3_fixture")
    )
    fixture.add_argument("--workers", type=int, choices=(1, 2, 6), required=True)
    fixture.add_argument(
        "--delay-order", choices=("none", "forward", "reverse"), default="none"
    )
    smoke_manifest = synthesis_commands.add_parser("build-smoke-manifest")
    smoke_manifest.add_argument("--project-root", type=Path, default=Path("."))
    smoke_manifest.add_argument(
        "--resolution-manifest",
        type=Path,
        default=Path("manifests/object_environment_resolution_v1.jsonl"),
    )
    smoke_manifest.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/s4_smoke/source_manifest_v1.json"),
    )
    smoke_manifest.add_argument("--max-foregrounds-per-environment", type=int, default=48)
    smoke = synthesis_commands.add_parser("smoke")
    smoke.add_argument("--project-root", type=Path, default=Path("."))
    smoke.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("artifacts/s4_smoke/source_manifest_v1.json"),
    )
    smoke.add_argument(
        "--resolution-manifest",
        type=Path,
        default=Path("manifests/object_environment_resolution_v1.jsonl"),
    )
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--workers", type=int, choices=(1, 2, 6), required=True)
    smoke.add_argument("--sample-count", type=int, choices=range(20, 51), default=20)
    smoke.add_argument("--target-long-side", type=int, default=384)
    training = commands.add_parser("training", help="strict model training/evaluation adapters")
    training_commands = training.add_subparsers(dest="training_command", required=True)
    training_smoke = training_commands.add_parser(
        "smoke", help="run the explicit S5 two-job GPU smoke"
    )
    training_smoke.add_argument("--project-root", type=Path, default=Path("."))
    training_smoke.add_argument(
        "--config",
        type=Path,
        default=Path("configs/campaigns/s5_dualhead_smoke.json"),
    )
    for name in ("run", "resume", "status", "logs"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            config = load_config(args.config) if args.config else None
            _print(inspect_environment(config))
            return 0
        if args.command == "plan":
            _print(_plan(load_config(args.config)))
            return 0
        if args.command == "manifest" and args.manifest_command == "build-low187":
            _print(
                build_s2_manifests(
                    project_root=args.project_root,
                    inventory_path=args.inventory,
                    verification_path=args.verification,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    workers=args.workers,
                )
            )
            return 0
        if args.command == "manifest" and args.manifest_command == "resolve-environments":
            _print(
                resolve_object_environments_from_disk(
                    project_root=args.project_root,
                    review_path=args.review,
                    object_pool_root=args.object_pool,
                    output_path=args.output,
                )
            )
            return 0
        if args.command == "synthesis" and args.synthesis_command == "fixture":
            root = args.project_root.resolve(strict=True)
            _print(
                run_fixture_campaign(
                    fixture_path=args.fixture,
                    legacy_source_path=root / "build_demo_synthetic_segmentation.py",
                    output_dir=args.output_dir,
                    allowed_root=root,
                    protected_paths=(root / "Dataset",),
                    workers=args.workers,
                    delay_order=args.delay_order,
                )
            )
            return 0
        if (
            args.command == "synthesis"
            and args.synthesis_command == "build-smoke-manifest"
        ):
            _print(
                build_smoke_source_manifest(
                    project_root=args.project_root,
                    output_path=args.output,
                    resolution_manifest_path=args.resolution_manifest,
                    max_foregrounds_per_environment=args.max_foregrounds_per_environment,
                )
            )
            return 0
        if args.command == "synthesis" and args.synthesis_command == "smoke":
            result = run_actual_smoke_campaign(
                project_root=args.project_root,
                source_manifest_path=args.source_manifest,
                resolution_manifest_path=args.resolution_manifest,
                output_dir=args.output_dir,
                workers=args.workers,
                config=SmokeConfig(
                    sample_count=args.sample_count,
                    target_long_side=args.target_long_side,
                ),
            )
            _print(result)
            return 0 if result["status"] == "VERIFIED" else 1
        if args.command == "training" and args.training_command == "smoke":
            from .training.scheduler import run_s5_smoke_campaign

            result = run_s5_smoke_campaign(
                config_path=args.config,
                project_root=args.project_root,
            )
            _print(result)
            return 0 if result["status"] == "VERIFIED" else 1
        if args.command in {"run", "resume"}:
            config = load_config(args.config, require_runnable=True)
            with StateStore(_db_path(config)) as store:
                result = Scheduler(config, store).run()
            _print(result)
            return 0 if result["status"] == "SUCCEEDED" else 1
        config = load_config(args.config)
        if args.command == "status":
            _print(_status(config))
            return 0
        if args.command == "logs":
            path = _db_path(config)
            if not path.exists():
                _print([])
            else:
                with StateStore(path) as store:
                    _print(store.events(config.campaign_name))
            return 0
    except (CapstoneLabError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2
