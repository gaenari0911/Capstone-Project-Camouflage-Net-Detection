from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.ops import segments2boxes

from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError, ConfigError
from .contracts import inside, read_json
from .sources import jsonl


def effective_args(root):
    recipe = read_json(root / "configs/campaigns/s7_plan.json")["planning"]["training_contract"]
    overrides = {k: recipe[k] for k in ("imgsz", "optimizer", "lr0", "weight_decay",
                                      "warmup_epochs", "lrf", "amp", "compile", "mask_ratio", "overlap_mask")}
    overrides.update(recipe["augmentation"])
    overrides.update(batch=16, epochs=150, patience=0, cos_lr=True, device="0",
                     workers=0, cache=False, rect=False, seed=0, deterministic=True,
                     save_json=False, save_txt=False, plots=False, task="segment",
                     conf=.001, iou=.7, max_det=300, half=False, nbs=16)
    args = get_cfg(overrides=overrides)
    return args


def manifest_records(root, subset):
    if subset not in {"real748", "low187", "val100"}:
        raise ConfigError("development loader only permits explicit train/val subsets; Test denied")
    path = root / ("manifests/low187_v1.jsonl" if subset == "low187" else "manifests/real_split_observed_v1.jsonl")
    split = "val" if subset == "val100" else "train"
    records = [r for r in jsonl(path) if r["original_split"] == split]
    expected = {"real748": 748, "low187": 187, "val100": 100}[subset]
    if len(records) != expected or len({r["image"] for r in records}) != expected:
        raise ArtifactError("manifest count/identity mismatch")
    for row in records:
        if not row["image"].startswith(f"Dataset/images/{split}/") or not row["label"].startswith(f"Dataset/labels/{split}/"):
            raise ArtifactError("split/path mismatch")
        for key in ("image", "label"):
            if sha256_file(inside(root, row[key])) != row[key + "_sha256"]:
                raise ArtifactError(f"manifest source changed: {row[key]}")
    return records


class ReadOnlySegDataset(YOLODataset):
    """Reuse the fork's augmentation/Format/collate without cache or JPEG repair writes."""
    def __init__(self, root, records, args, *, augment):
        self.root = Path(root)
        self.records = copy.deepcopy(records)
        super().__init__(img_path="manifest-only", data={"names": {0: "camouflage"}, "nc": 1, "channels": 3},
                         task="segment", imgsz=args.imgsz, cache=False, augment=augment,
                         hyp=copy.deepcopy(args), rect=False, batch_size=args.batch, stride=32, pad=0.)
        self.max_buffer_length = min(8, self.ni)
        # Never reuse an adjacent, potentially stale or user-owned .npy cache.
        self.npy_files = [self.root / "artifacts/s8_no_source_cache" / (str(i) + ".npy") for i in range(self.ni)]

    def get_img_files(self, _):
        return [str(inside(self.root, r["image"])) for r in self.records]

    def get_labels(self):
        labels = []
        self.label_files = []
        for row in self.records:
            image = inside(self.root, row["image"])
            label = inside(self.root, row["label"])
            self.label_files.append(str(label))
            with Image.open(image) as decoded:
                width, height = decoded.size
                decoded.verify()  # No library verify_image_label: it can repair JPEGs in place.
            segments, classes = [], []
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                values = np.asarray([float(x) for x in line.split()], dtype=np.float32)
                if len(values) < 7 or len(values) % 2 != 1 or not np.isfinite(values).all() or values[0] != 0 or np.any(values[1:] < 0) or np.any(values[1:] > 1):
                    raise ArtifactError(f"invalid segmentation polygon: {label}")
                polygon = values[1:].reshape(-1, 2)
                if cv2.contourArea(polygon) <= 0:
                    raise ArtifactError(f"empty polygon area: {label}")
                segments.append(polygon)
                classes.append([0.])
            if not segments:
                raise ArtifactError(f"positive Real source has missing/empty label: {label}")
            labels.append(dict(im_file=str(image), shape=(height, width), cls=np.asarray(classes, dtype=np.float32),
                               bboxes=segments2boxes(segments), segments=segments, keypoints=None,
                               normalized=True, bbox_format="xywh"))
        return labels


def worker_seed(_):
    seed = torch.initial_seed() % (2**32)
    import random
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)


def make_loader(dataset, *, seed, workers=0, shuffle=False):
    generator = torch.Generator().manual_seed(900000 + seed)
    return DataLoader(dataset, batch_size=16, shuffle=shuffle, num_workers=workers,
                      collate_fn=dataset.collate_fn, generator=generator,
                      worker_init_fn=worker_seed, pin_memory=True, drop_last=False)


def model_batch(batch, variant, device):
    result = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    result["img"] = result["img"].float() / 255.
    if variant in {"H1", "H2"}:
        for key in ("boundaries", "distance_maps"):
            if key not in result or result[key].shape != result["masks"].shape:
                raise ArtifactError("boundary target/collate missing or misaligned")
        if variant == "H1":
            result["distance_maps"] = torch.ones_like(result["distance_maps"])
    elif variant != "H0":
        raise ConfigError("unknown head/loss variant")
    return result
