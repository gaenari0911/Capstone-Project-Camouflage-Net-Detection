from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .contracts import approved_plan, code_records, digest, output_path
from .data import ReadOnlySegDataset, effective_args, make_loader, manifest_records, model_batch
from .io import atomic_json
from .models import create_model
from .resume_preflight import update


def stress_batch(root, output):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _, _, sources = approved_plan(root)
    records = manifest_records(root, "real748")
    records.sort(key=lambda r: (-len((root/r["label"]).read_text().strip().splitlines()), r["image"]))
    selected = records[:16]
    args = effective_args(root)
    # Disable augmentation for this memory stress test only: preserve all annotated instances.
    dataset = ReadOnlySegDataset(root, selected, args, augment=False)
    batch = next(iter(make_loader(dataset, seed=0)))
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "provenance.json", {"sources": sources, "code": code_records(root),
                "selection": "16 train images with highest polygon instance count",
                "augmentation": False, "formal_training": False}, immutable=True)
    model, initial = create_model(root, output, 0, "H2")
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory // 2**20
    cap = min(int(total*.85), total-2048)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0005)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        prepared = model_batch(batch, "H2", device)
        loss = update(model, prepared, optimizer, scaler)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved() // 2**20
        reservation = peak + 512
        result = {"status": "VERIFIED_SINGLE_JOB_STRESS", "batch": 16, "imgsz": 640,
                  "instances": len(batch["cls"]), "peak_reserved_mib": peak,
                  "reservation_with_margin_mib": reservation, "device_cap_mib": cap,
                  "two_reservations_fit_before_external_usage": 2 * reservation <= cap,
                  "loss": loss, "wall_seconds": time.perf_counter()-start,
                  "concurrency2_verified": False}
    except torch.cuda.OutOfMemoryError as exc:
        result = {"status":"SINGLE_JOB_OOM_NO_BATCH_SHRINK", "batch":16, "imgsz":640,
                  "instances":len(batch["cls"]), "error":str(exc), "concurrency2_verified":False}
    atomic_json(output / "summary.json", result, immutable=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    import json
    print(json.dumps(stress_batch(root, output_path(root, args.output)), indent=2))


if __name__ == "__main__":
    main()
