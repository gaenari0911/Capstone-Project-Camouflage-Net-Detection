from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ArtifactError, ConfigError
from capstone_lab.training.dualhead import (
    _resolved_model_yaml,
    set_reproducible_seed,
    tensor_state_hash,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _batch(step: int, *, batch_size: int, imgsz: int, device: torch.device) -> dict[str, torch.Tensor]:
    from ultralytics.data.dataset import generate_boundary_and_distance_map

    generator = torch.Generator(device="cpu")
    generator.manual_seed(840_000 + step)
    images = torch.rand((batch_size, 3, imgsz, imgsz), generator=generator).to(device)
    masks = torch.zeros((batch_size, imgsz, imgsz), dtype=torch.float32)
    for index in range(batch_size):
        margin = max(4, imgsz // 5 + index % max(1, imgsz // 20))
        masks[index, margin : imgsz - margin, margin : imgsz - margin] = 1.0
    boundaries, distance_maps = generate_boundary_and_distance_map(masks)
    return {
        "img": images,
        "batch_idx": torch.arange(batch_size, dtype=torch.long),
        "cls": torch.zeros((batch_size, 1), dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32).repeat(batch_size, 1),
        "masks": masks,
        "boundaries": boundaries,
        "distance_maps": distance_maps,
    }


def _initial_hashes(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> dict[str, Any]:
    initialized = model.state_dict()
    changed = sorted(name for name, value in initialized.items() if not torch.equal(value, before[name]))
    boundary = sorted(name for name in initialized if ".cv5." in name)
    if not boundary or any(name in changed for name in boundary):
        raise ArtifactError("boundary head must remain independently seed-initialized")
    return {
        "full_state_sha256": tensor_state_hash(initialized),
        "common_state_sha256": tensor_state_hash(initialized, include=lambda name: ".cv5." not in name),
        "boundary_head_sha256": tensor_state_hash(initialized, include=lambda name: ".cv5." in name),
        "changed_by_pretrained_sha256": hashlib.sha256(canonical_json(changed).encode("utf-8")).hexdigest(),
        "changed_by_pretrained_count": len(changed),
        "state_tensor_count": len(initialized),
        "boundary_tensor_count": len(boundary),
    }


def run(spec_path: Path) -> int:
    from ultralytics import YOLO, __version__ as ultralytics_version
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.modules.head import DualHeadSegment

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = Path(spec["output_dir"]).resolve(strict=False)
    allowed = Path(spec["allowed_output_root"]).resolve(strict=True)
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise ConfigError("GPU profile output is outside the S6 artifact root") from exc
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    started = time.time()
    seed = int(spec["seed"])
    batch_size = int(spec["batch"])
    effective = spec["effective_args"]
    contract = dict(spec["contract"])
    model_yaml = Path(spec["model_yaml"]).resolve(strict=True)
    pretrained = Path(spec["pretrained"]).resolve(strict=True)
    if sha256_file(model_yaml) != contract["model_source_sha256"]:
        raise ArtifactError("custom model YAML changed")
    if sha256_file(pretrained) != contract["pretrained_sha256"]:
        raise ArtifactError("base pretrained changed")
    try:
        if not torch.cuda.is_available():
            raise ConfigError("S6 GPU profile requires CUDA")
        device = torch.device(spec["device"])
        set_reproducible_seed(seed)
        resolved = output / "yolov8n-seg-dualhead-s6-profile.yaml"
        resolved_hash = _resolved_model_yaml(model_yaml, resolved, spec["scale"], int(spec["num_classes"]))
        wrapper = YOLO(str(resolved))
        before = {name: value.detach().clone() for name, value in wrapper.model.state_dict().items()}
        wrapper.load(str(pretrained))
        model = wrapper.model
        if not isinstance(model.model[-1], DualHeadSegment):
            raise ArtifactError("profile model does not end with DualHeadSegment")
        hashes = _initial_hashes(model, before)
        if hashes["full_state_sha256"] != spec["expected_initial_state_sha256"]:
            raise ArtifactError("profile initial state differs from verified S5")
        del before, wrapper
        model.to(device).train()
        model.args = get_cfg(overrides={
            "overlap_mask": effective["overlap_mask"], "box": effective["box"],
            "cls": effective["cls"], "dfl": effective["dfl"],
        })
        optimizer = torch.optim.AdamW(model.parameters(), lr=effective["lr0"], weight_decay=effective["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=bool(effective["amp"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        latencies: list[float] = []
        losses: list[float] = []
        grad_norms: list[float] = []
        overflow_count = 0
        total_steps = int(spec["warmup_steps"]) + int(spec["measure_steps"])
        for step in range(total_steps):
            batch = _batch(step, batch_size=batch_size, imgsz=int(effective["imgsz"]), device=device)
            torch.cuda.synchronize(device)
            tick = time.perf_counter()
            retries = 0
            while True:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=bool(effective["amp"])):
                    total_loss, _ = model.loss(batch)
                    scalar = total_loss.sum()
                if not bool(torch.isfinite(scalar)):
                    raise ArtifactError("non-finite profile loss")
                scaler.scale(scalar).backward()
                scaler.unscale_(optimizer)
                boundary_grads = [p.grad for name, p in model.named_parameters() if ".cv5." in name and p.grad is not None]
                if not boundary_grads:
                    raise ArtifactError("boundary head gradient is absent")
                finite = all(bool(torch.isfinite(value).all()) for value in boundary_grads)
                grad_norm = float(torch.sqrt(sum(torch.sum(value.detach().float() ** 2) for value in boundary_grads)).cpu())
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                if float(scaler.get_scale()) < previous_scale:
                    retries += 1
                    overflow_count += 1
                    if retries > 16:
                        raise ArtifactError("AMP overflow did not stabilize within 16 retries")
                    continue
                if not finite:
                    raise ArtifactError("non-finite boundary gradient without AMP backoff")
                break
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - tick
            if step >= int(spec["warmup_steps"]):
                latencies.append(elapsed)
                losses.append(float(scalar.detach().cpu()))
                grad_norms.append(grad_norm)
            del batch, scalar, total_loss, boundary_grads
        train_allocated = int(torch.cuda.max_memory_allocated(device) / 2**20)
        train_reserved = int(torch.cuda.max_memory_reserved(device) / 2**20)

        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        validation_batch = _batch(99_999, batch_size=batch_size, imgsz=int(effective["imgsz"]), device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=bool(effective["amp"])):
            validation_loss, _ = model.loss(validation_batch)
            validation_scalar = validation_loss.sum()
        torch.cuda.synchronize(device)
        validation_allocated = int(torch.cuda.max_memory_allocated(device) / 2**20)
        validation_reserved = int(torch.cuda.max_memory_reserved(device) / 2**20)
        if not bool(torch.isfinite(validation_scalar)):
            raise ArtifactError("non-finite validation fixture loss")
        mean_latency = sum(latencies) / len(latencies)
        result = {
            "schema_version": 1, "status": "SUCCEEDED", "validated": True,
            "profile_scope": "fixed_train_tensor_not_model_performance_not_checkpoint",
            "job_id": spec["job_id"], "pid": os.getpid(), "batch": batch_size,
            "warmup_steps": int(spec["warmup_steps"]), "measured_steps": len(latencies),
            "optimizer_steps": total_steps, "amp_overflow_backoffs": overflow_count,
            "step_latency_seconds": {"values": latencies, "mean": mean_latency, "min": min(latencies), "max": max(latencies)},
            "samples_per_second": batch_size / mean_latency,
            "loss": {"values": losses, "all_finite": True, "boundary_gradient_norms": grad_norms, "boundary_gradients_finite": True},
            "validation": {"scope": "minimal_validation_loss_fixture_not_Test", "loss": float(validation_scalar.cpu()), "mask_metric_name": "mask_fixture_only", "box_metric": None},
            "memory": {
                "torch_peak_allocated_mib": train_allocated,
                "torch_peak_reserved_mib": train_reserved,
                "validation_peak_allocated_mib": validation_allocated,
                "validation_peak_reserved_mib": validation_reserved,
            },
            "device": {"requested": spec["device"], "name": torch.cuda.get_device_name(device), "capability": list(torch.cuda.get_device_capability(device))},
            "versions": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "ultralytics": ultralytics_version},
            "model": {"class": type(model.model[-1]).__name__, "scale": model.yaml.get("scale"), "num_classes": int(model.yaml.get("nc")), "parameters": sum(p.numel() for p in model.parameters()), "initial_hashes": hashes},
            "effective_args": effective,
            "contract": {**contract, "resolved_model_sha256": resolved_hash, "initial_state_sha256": hashes["full_state_sha256"]},
            "started_unix": started, "ended_unix": time.time(),
        }
        _write(result_path, result)
        del validation_batch, validation_loss, validation_scalar, optimizer, scaler, model
        torch.cuda.empty_cache()
        return 0
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).casefold()
        if not is_oom:
            raise
        _write(result_path, {
            "schema_version": 1, "status": "OOM", "validated": False,
            "profile_scope": "fixed_train_tensor_not_model_performance_not_checkpoint",
            "job_id": spec["job_id"], "pid": os.getpid(), "batch": batch_size,
            "error": f"{type(exc).__name__}: {exc}", "contract": contract,
            "memory": {
                "torch_peak_allocated_mib": int(torch.cuda.max_memory_allocated() / 2**20),
                "torch_peak_reserved_mib": int(torch.cuda.max_memory_reserved() / 2**20),
            },
            "started_unix": started, "ended_unix": time.time(),
        })
        torch.cuda.empty_cache()
        return 42


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    return run(parser.parse_args(argv).spec)


if __name__ == "__main__":
    raise SystemExit(main())
