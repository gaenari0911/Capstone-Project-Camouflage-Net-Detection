from __future__ import annotations

import copy
import gc
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Segment
from ultralytics.models.yolo.segment.val import SegmentationValidator

from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError
from capstone_lab.training.dualhead import _resolved_model_yaml, set_reproducible_seed, tensor_state_hash
from .contracts import approved_plan, code_records, digest, read_json
from .data import ReadOnlySegDataset, effective_args, make_loader, manifest_records, model_batch
from .io import atomic_json


def create_model(root, output, seed, variant):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise ArtifactError("CUBLAS_WORKSPACE_CONFIG is incompatible with strict determinism")
    torch.use_deterministic_algorithms(True)
    source = read_json(root / "configs/campaigns/s5_dualhead_smoke.json")
    pretrained = root / source["paths"]["pretrained"]
    approved = read_json(root / "configs/campaigns/s7_plan.json")["planning"]
    if sha256_file(pretrained) != approved["pretrained_sha256"]:
        raise ArtifactError("pretrained hash changed")
    set_reproducible_seed(seed)
    path = output / f"yolov8n-s8-{variant}.yaml"
    _resolved_model_yaml(root / source["paths"]["model_yaml"], path, "n", 1)
    model = YOLO(str(path)).load(str(pretrained)).model
    if variant == "H0":
        # Initialize the SAME full model first, then remove only the training branch.
        # This preserves common tensors including RNG-dependent Segment initialization.
        del model.model[-1].cv5
        model.model[-1].__class__ = Segment
        model.yaml["head"][-1][2] = "Segment"
    elif variant not in {"H1", "H2"}:
        raise ArtifactError("invalid variant")
    initial = {"common": tensor_state_hash(model.state_dict(), include=lambda n: ".cv5." not in n),
               "boundary": tensor_state_hash(model.state_dict(), include=lambda n: ".cv5." in n),
               "full": tensor_state_hash(model.state_dict()),
               "parameters": sum(p.numel() for p in model.parameters())}
    model.args = effective_args(root)
    return model, initial


@torch.inference_mode()
def validate_masks(model, loader, output, args):
    """Use fork-native mask matching/AP, with separate fixed-confidence mask P/R/F1."""
    device = next(model.parameters()).device
    validator = SegmentationValidator(dataloader=loader, save_dir=output,
                                     args=vars(copy.deepcopy(args)))
    validator.device = device
    validator.training = True
    validator.data = {"val": "frozen-val100", "names": {0: "camouflage"}}
    validator.init_metrics(model)
    model.eval()
    boundary_calls = []
    hook = model.model[-1].cv5.register_forward_hook(lambda *_: boundary_calls.append(True)) if hasattr(model.model[-1], "cv5") else None
    tp = fp = fn = 0
    start = time.perf_counter()
    try:
        for raw in loader:
            batch = validator.preprocess(raw)
            predictions = validator.postprocess(model(batch["img"]))
            validator.update_metrics(predictions, batch)
            for index, pred in enumerate(predictions):
                target = validator._prepare_batch(index, batch)
                selected = pred["conf"] >= .25
                fixed = {k: v[selected] for k, v in pred.items()}
                matched = validator._process_batch(fixed, target)["tp_m"][:, 0]
                correct = int(matched.sum())
                tp += correct
                fp += len(matched) - correct
                fn += len(target["cls"]) - correct
        result = validator.get_stats()
    finally:
        if hook:
            hook.remove()
    if boundary_calls:
        raise ArtifactError("boundary branch executed during inference")
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return {"library_metrics": {k: float(v) for k, v in result.items()},
            "mask_precision": precision, "mask_recall": recall,
            "mask_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "mask_tp": tp, "mask_fp": fp, "mask_fn": fn, "images": validator.seen,
            "operating_confidence": .25, "operating_mask_iou": .5, "ap_min_confidence": .001,
            "nms_iou": .7, "max_det": 300, "mask_processing": "fork_process_mask_default",
            "inference_boundary_calls": len(boundary_calls), "wall_seconds": time.perf_counter() - start}


def real_preflight(root, output):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise ArtifactError("approved CUDA device unavailable")
    _, _, source_records = approved_plan(root)
    output.mkdir(parents=True, exist_ok=True)
    args = effective_args(root)
    atomic_json(output / "effective_args.json", vars(args), immutable=True)
    import ultralytics
    library = Path(ultralytics.__file__).parent
    library_hashes = {p.relative_to(library).as_posix(): sha256_file(p) for p in sorted(library.rglob("*.py"))}
    atomic_json(output / "provenance.json",
                {"sources": source_records, "adapter_code": code_records(root),
                 "library_root": str(library), "library_code": library_hashes,
                 "torch_version": torch.__version__, "effective_args_sha256": digest(vars(args)),
                 "deterministic_algorithms": True, "cublas_workspace": ":4096:8 unless valid :16:8 explicitly provided"}, immutable=True)
    records = manifest_records(root, "low187")
    validation = manifest_records(root, "val100")
    train = ReadOnlySegDataset(root, records, args, augment=True)
    val = ReadOnlySegDataset(root, validation, args, augment=False)
    # A real augmented 16-image batch; all instances survive the fork's collate contract.
    set_reproducible_seed(920000)
    batch = next(iter(make_loader(train, seed=0, workers=0, shuffle=True)))
    atomic_json(output / "batch.json",
                {"images": list(batch["img"].shape), "instances": len(batch["cls"]),
                 "masks": list(batch["masks"].shape), "paths": list(batch["im_file"])}, immutable=True)
    results = {}
    for variant in ("H0", "H1", "H2"):
        destination = output / variant
        destination.mkdir()
        model, initial = create_model(root, destination, 0, variant)
        model.to("cuda:0").train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0005)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        torch.cuda.reset_peak_memory_stats()
        prepared = model_batch(batch, variant, "cuda:0")
        losses = []
        for step in range(2):
            for retry in range(17):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss, components = model.loss(prepared)
                    scalar = loss.sum()
                if not torch.isfinite(scalar):
                    raise ArtifactError("non-finite real batch loss")
                scaler.scale(scalar).backward()
                scaler.unscale_(optimizer)
                scale = scaler.get_scale()
                # Full/common gradients checked before optimizer changes/clears them.
                gradients = [p.grad for n,p in model.named_parameters() if p.grad is not None and ".cv5." in n]
                finite = all(bool(torch.isfinite(g).all()) for g in gradients)
                nonzero = any(bool(g.abs().sum() > 0) for g in gradients)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale:
                    continue
                if variant != "H0" and (not gradients or not finite or not nonzero):
                    raise ArtifactError("boundary gradient absent/non-finite/zero")
                losses.append({"step": step + 1, "loss": float(scalar.detach()),
                               "components": components.detach().cpu().tolist(), "amp_retries": retry,
                               "boundary_gradient_checked": variant != "H0"})
                break
            else:
                raise ArtifactError("AMP failed to stabilize")
        train_peak = torch.cuda.max_memory_reserved() // 2**20
        del prepared, optimizer, scaler, loss, scalar, components, gradients
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        metrics = validate_masks(model, make_loader(val, seed=0), destination / "validation", args)
        results[variant] = {"initial": initial, "losses": losses, "train_peak_reserved_mib": train_peak,
                            "val_peak_reserved_mib": torch.cuda.max_memory_reserved() // 2**20, "validation": metrics}
        atomic_json(destination / "result.json", results[variant], immutable=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if len({r["initial"]["common"] for r in results.values()}) != 1 or results["H1"]["initial"]["boundary"] != results["H2"]["initial"]["boundary"]:
        raise ArtifactError("common/boundary initialization mismatch")
    summary = {"status": "VERIFIED_REAL_BATCH_AND_VAL_ONLY", "batch": 16, "imgsz": 640, "results": results,
               "formal_training_completed": False, "resume_verified": False, "concurrency2_real_loader_verified": False}
    atomic_json(output / "summary.json", summary, immutable=True)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from .contracts import output_path
    root = Path.cwd()
    result = real_preflight(root, output_path(root, args.output))
    print(result["status"])


if __name__ == "__main__":
    main()
