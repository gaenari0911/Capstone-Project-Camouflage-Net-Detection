from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ArtifactError, ConfigError

from .checkpoints import load_training_checkpoint, save_training_checkpoint
from .metrics import evaluate_paired_masks


def tensor_state_hash(
    state: dict[str, torch.Tensor], *, include: Callable[[str], bool] | None = None
) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if include is not None and not include(name):
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(canonical_json(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _resolved_model_yaml(source: Path, output: Path, scale: str, num_classes: int) -> str:
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read custom model YAML: {exc}") from exc
    if not isinstance(payload, dict) or scale not in payload.get("scales", {}):
        raise ConfigError(f"custom model YAML does not define scale {scale!r}")
    head_rows = payload.get("head", [])
    if not any(
        isinstance(row, list) and len(row) >= 3 and row[2] == "DualHeadSegment"
        for row in head_rows
    ):
        raise ConfigError("custom model YAML does not contain DualHeadSegment")
    payload["scale"] = scale
    payload["nc"] = num_classes
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    output.write_text(rendered, encoding="utf-8")
    return sha256_file(output)


def _fixture_metrics(confidence: float, mask_iou: float) -> dict[str, Any]:
    targets = [[1, 1, 1, 1] for _ in range(4)]
    predictions = [
        [1, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
    return evaluate_paired_masks(
        predictions,
        targets,
        [0.90, 0.80, 0.70, 0.60],
        operating_confidence=confidence,
        operating_iou=mask_iou,
    )


def _synthetic_batch(step: int, *, device: torch.device, imgsz: int) -> dict[str, torch.Tensor]:
    from ultralytics.data.dataset import generate_boundary_and_distance_map

    generator = torch.Generator(device="cpu")
    generator.manual_seed(840_000 + step)
    image = torch.rand((1, 3, imgsz, imgsz), generator=generator).to(device)
    masks = torch.zeros((1, imgsz, imgsz), dtype=torch.float32)
    margin = max(4, imgsz // 4)
    masks[:, margin : imgsz - margin, margin : imgsz - margin] = 1.0
    boundaries, distance_maps = generate_boundary_and_distance_map(masks)
    return {
        "img": image,
        "batch_idx": torch.tensor([0], dtype=torch.long),
        "cls": torch.tensor([[0.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32),
        "masks": masks,
        "boundaries": boundaries,
        "distance_maps": distance_maps,
    }


def run_explicit_worker(spec_path: Path) -> int:
    from ultralytics import YOLO, __version__ as ultralytics_version
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.modules.head import DualHeadSegment

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output_dir = Path(spec["output_dir"]).resolve(strict=False)
    allowed_root = Path(spec["allowed_output_root"]).resolve(strict=True)
    try:
        output_dir.relative_to(allowed_root)
    except ValueError as exc:
        raise ConfigError("worker output is outside the S5 artifact root") from exc
    output_dir.mkdir(parents=True, exist_ok=True)

    model_yaml = Path(spec["model_yaml"]).resolve(strict=True)
    pretrained = Path(spec["pretrained"]).resolve(strict=True)
    expected = spec["hashes"]
    if sha256_file(model_yaml) != expected["model_source_sha256"]:
        raise ArtifactError("custom model YAML changed")
    if sha256_file(pretrained) != expected["pretrained_sha256"]:
        raise ArtifactError("base pretrained changed")

    seed = int(spec["seed"])
    set_reproducible_seed(seed)
    resolved_yaml = output_dir / f"yolov8{spec['scale']}-seg-dualhead-s5.yaml"
    resolved_yaml_hash = _resolved_model_yaml(
        model_yaml, resolved_yaml, spec["scale"], int(spec["num_classes"])
    )
    wrapper = YOLO(str(resolved_yaml))
    before = {name: value.detach().clone() for name, value in wrapper.model.state_dict().items()}
    wrapper.load(str(pretrained))
    model = wrapper.model
    if not isinstance(model.model[-1], DualHeadSegment):
        raise ArtifactError("resolved model does not end with DualHeadSegment")
    if model.yaml.get("scale") != spec["scale"]:
        raise ArtifactError("resolved model scale does not match the explicit scale")

    initialized = model.state_dict()
    changed_by_pretrained = sorted(
        name for name, value in initialized.items() if not torch.equal(value, before[name])
    )
    boundary_keys = sorted(name for name in initialized if ".cv5." in name)
    if not boundary_keys or any(name in changed_by_pretrained for name in boundary_keys):
        raise ArtifactError("boundary head must remain seed-initialized, not pretrained")
    initial_hashes = {
        "full_state_sha256": tensor_state_hash(initialized),
        "common_state_sha256": tensor_state_hash(
            initialized, include=lambda name: ".cv5." not in name
        ),
        "boundary_head_sha256": tensor_state_hash(
            initialized, include=lambda name: ".cv5." in name
        ),
        "changed_by_pretrained_sha256": hashlib.sha256(
            canonical_json(changed_by_pretrained).encode("utf-8")
        ).hexdigest(),
        "changed_by_pretrained_count": len(changed_by_pretrained),
        "state_tensor_count": len(initialized),
        "boundary_tensor_count": len(boundary_keys),
    }

    device = torch.device(spec["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ConfigError("S5 smoke requires the approved CUDA device")
    model.to(device).train()
    effective = spec["effective_args"]
    model.args = get_cfg(
        overrides={
            "overlap_mask": effective["overlap_mask"],
            "box": effective["box"],
            "cls": effective["cls"],
            "dfl": effective["dfl"],
        }
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective["lr0"],
        weight_decay=effective["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(spec["steps"])),
        eta_min=effective["lr0"] * effective["lrf"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(effective["amp"]))
    contract = {
        "config_sha256": expected["config_sha256"],
        "input_sha256": expected["input_sha256"],
        "code_sha256": expected["code_sha256"],
        "library_sha256": expected["library_sha256"],
        "model_source_sha256": expected["model_source_sha256"],
        "pretrained_sha256": expected["pretrained_sha256"],
        "source_zip_sha256": expected["source_zip_sha256"],
        "train_manifest_sha256": expected["train_manifest_sha256"],
        "val_manifest_sha256": expected["val_manifest_sha256"],
        "resolved_model_sha256": resolved_yaml_hash,
        "initial_state_sha256": initial_hashes["full_state_sha256"],
    }
    checkpoint_path = output_dir / "checkpoint.pt"
    start_step = 0
    resume_record: dict[str, Any] | None = None
    if checkpoint_path.exists():
        resume_record = load_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            expected_contract=contract,
        )
        start_step = int(resume_record["global_step"])
    if start_step > int(spec["steps"]):
        raise ArtifactError("checkpoint global step exceeds requested steps")

    losses: list[dict[str, Any]] = []
    checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path.exists() else None
    for step in range(start_step, int(spec["steps"])):
        overflow_retries = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            batch = _synthetic_batch(step, device=device, imgsz=int(effective["imgsz"]))
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(effective["amp"])):
                total_loss, parts = model.loss(batch)
                scalar_loss = total_loss.sum()
            if not bool(torch.isfinite(scalar_loss)):
                raise ArtifactError("non-finite smoke loss")
            scaler.scale(scalar_loss).backward()
            scaler.unscale_(optimizer)
            boundary_gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if ".cv5." in name and parameter.grad is not None
            ]
            if not boundary_gradients:
                raise ArtifactError("boundary head gradient is absent")
            boundary_finite = all(
                bool(torch.isfinite(value).all()) for value in boundary_gradients
            )
            previous_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            current_scale = float(scaler.get_scale())
            if current_scale < previous_scale:
                overflow_retries += 1
                if overflow_retries > 16:
                    raise ArtifactError("AMP overflow did not stabilize within 16 retries")
                continue
            if not boundary_finite:
                raise ArtifactError("boundary head gradient remained non-finite without AMP backoff")
            break
        scheduler.step()
        losses.append(
            {
                "global_step": step + 1,
                "total": float(scalar_loss.detach().cpu()),
                "parts": [float(value) for value in parts.detach().cpu()],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "amp_overflow_retries": overflow_retries,
                "amp_scale": float(scaler.get_scale()),
            }
        )
        checkpoint_hash = save_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=0,
            global_step=step + 1,
            contract=contract,
        )
        if (
            int(spec.get("inject_failure_after_step", 0)) == step + 1
            and int(spec["attempt"]) == 1
        ):
            partial = {
                "schema_version": 1,
                "status": "INJECTED_FAILURE_AFTER_CHECKPOINT",
                "job_id": spec["job_id"],
                "attempt": spec["attempt"],
                "global_step": step + 1,
                "checkpoint_sha256": checkpoint_hash,
                "contract": contract,
                "initial_hashes": initial_hashes,
            }
            (output_dir / "partial_result.json").write_text(
                canonical_json(partial) + "\n", encoding="utf-8"
            )
            return 75

    metrics = _fixture_metrics(
        float(spec["metrics"]["confidence"]), float(spec["metrics"]["mask_iou"])
    )
    result = {
        "schema_version": 1,
        "status": "SUCCEEDED",
        "validated": True,
        "job_id": spec["job_id"],
        "attempt": int(spec["attempt"]),
        "seed": seed,
        "device": {
            "requested": spec["device"],
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "versions": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "ultralytics": ultralytics_version,
        },
        "model": {
            "class": type(model.model[-1]).__name__,
            "scale": model.yaml.get("scale"),
            "num_classes": int(model.yaml.get("nc")),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "initial_hashes": initial_hashes,
            "initialization_policy": "independent base pretrained plus seed-initialized cv5 boundary head",
        },
        "effective_args": effective,
        "contract": contract,
        "resumed": resume_record is not None,
        "resume_from_global_step": start_step,
        "losses_this_attempt": losses,
        "final_global_step": int(spec["steps"]),
        "checkpoint_sha256": checkpoint_hash,
        "metrics": metrics,
    }
    result_path = output_dir / "result.json"
    result_path.write_text(canonical_json(result) + "\n", encoding="utf-8")
    return 0
