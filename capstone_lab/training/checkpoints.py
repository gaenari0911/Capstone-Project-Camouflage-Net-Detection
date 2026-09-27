from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ArtifactError


CHECKPOINT_SCHEMA_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    contract: dict[str, str],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "rng": capture_rng_state(),
        "contract": dict(contract),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
    digest = sha256_file(path)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint": path.name,
        "checkpoint_sha256": digest,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "contract": dict(contract),
    }
    manifest_path = path.with_suffix(path.suffix + ".json")
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return digest


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    expected_contract: dict[str, str],
) -> dict[str, int | str]:
    manifest_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not manifest_path.is_file():
        raise ArtifactError(f"checkpoint or manifest is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_digest = sha256_file(path)
    if manifest.get("checkpoint_sha256") != actual_digest:
        raise ArtifactError("checkpoint hash does not match its manifest")
    if manifest.get("contract") != expected_contract:
        raise ArtifactError("checkpoint manifest contract changed; resume refused")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ArtifactError("unsupported checkpoint schema")
    if payload.get("contract") != expected_contract:
        raise ArtifactError("checkpoint payload contract changed; resume refused")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng"])
    return {
        "epoch": int(payload["epoch"]),
        "global_step": int(payload["global_step"]),
        "checkpoint_sha256": actual_digest,
    }
