"""
Build train/val/test datasets using revised hard positive selection.

Workflow:
1. Copy all Training images and labels into Dataset/images/train and Dataset/labels/train
2. Copy revised hard positive Testing samples into reserve pools
3. Exclude those hard positive stems, then split remaining Testing samples into val/test

Example:
    python Build_Splits_From_Revised_Selected.py --dry-run
    python Build_Splits_From_Revised_Selected.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SNOW_BRIGHTNESS_THRESHOLD = 160.0
SNOW_SATURATION_THRESHOLD = 60.0
RESIZE_MAX_SIDE = 256


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Copy train set, reserve revised hard positives, and split remaining Testing data."
    )
    parser.add_argument(
        "--revised-selected-csv",
        type=Path,
        default=Path("revised_hard_positive_selected.csv"),
        help="CSV containing final hard positive filenames.",
    )
    parser.add_argument(
        "--training-images-dir",
        type=Path,
        default=Path("Dataset") / "masks_raw" / "Training" / "images",
        help="Source Training image directory.",
    )
    parser.add_argument(
        "--training-labels-dir",
        type=Path,
        default=Path("converted") / "labels" / "Training",
        help="Source Training label directory.",
    )
    parser.add_argument(
        "--testing-images-dir",
        type=Path,
        default=Path("Dataset") / "masks_raw" / "Testing" / "images",
        help="Source Testing image directory.",
    )
    parser.add_argument(
        "--testing-labels-dir",
        type=Path,
        default=Path("converted") / "labels" / "Testing",
        help="Source Testing label directory.",
    )
    parser.add_argument(
        "--train-images-out",
        type=Path,
        default=Path("Dataset") / "images" / "train",
        help="Destination train image directory.",
    )
    parser.add_argument(
        "--train-labels-out",
        type=Path,
        default=Path("Dataset") / "labels" / "train",
        help="Destination train label directory.",
    )
    parser.add_argument(
        "--val-images-out",
        type=Path,
        default=Path("Dataset") / "images" / "val",
        help="Destination val image directory.",
    )
    parser.add_argument(
        "--val-labels-out",
        type=Path,
        default=Path("Dataset") / "labels" / "val",
        help="Destination val label directory.",
    )
    parser.add_argument(
        "--test-images-out",
        type=Path,
        default=Path("Dataset") / "images" / "test",
        help="Destination test image directory.",
    )
    parser.add_argument(
        "--test-labels-out",
        type=Path,
        default=Path("Dataset") / "labels" / "test",
        help="Destination test label directory.",
    )
    parser.add_argument(
        "--reserve-images-out",
        type=Path,
        default=Path("Dataset") / "reserve" / "hard_positive_pool",
        help="Destination reserve image directory.",
    )
    parser.add_argument(
        "--reserve-labels-out",
        type=Path,
        default=Path("Dataset") / "reserve_labels" / "hard_positive_pool",
        help="Destination reserve label directory.",
    )
    parser.add_argument(
        "--split-log-csv",
        type=Path,
        default=Path("split_log.csv"),
        help="CSV path for split log output.",
    )
    parser.add_argument("--val-count", type=int, default=100, help="Total val sample count.")
    parser.add_argument("--test-count", type=int, default=200, help="Total test sample count.")
    parser.add_argument("--val-non-snow", type=int, default=70, help="Val non_snow target.")
    parser.add_argument("--val-snow", type=int, default=30, help="Val snow target.")
    parser.add_argument("--test-non-snow", type=int, default=140, help="Test non_snow target.")
    parser.add_argument("--test-snow", type=int, default=60, help="Test snow target.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copy/move actions without modifying files.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_stem(filename: str) -> str:
    """Normalize a filename or identifier into a lowercase stem."""
    return Path(str(filename)).stem.lower()


def load_selected_csv(csv_path: Path) -> tuple[list[str], set[str]]:
    """Load revised selected CSV and return filenames and normalized stems."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Revised selected CSV not found: {csv_path}")

    filenames: list[str] = []
    stems: set[str] = set()

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if "filename" not in (reader.fieldnames or []):
            raise ValueError("'filename' column missing in revised selected CSV.")

        for row in reader:
            filename = row.get("filename", "").strip()
            if not filename:
                continue
            filenames.append(filename)
            stems.add(normalize_stem(filename))

    return filenames, stems


def collect_image_files(images_dir: Path) -> list[Path]:
    """Collect direct image files from a directory."""
    if not images_dir.exists():
        return []

    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def collect_training_pairs(images_dir: Path, labels_dir: Path) -> list[dict[str, object]]:
    """Collect Training image/label pairs by stem."""
    pairs: list[dict[str, object]] = []
    for image_path in collect_image_files(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        pairs.append(
            {
                "filename": image_path.name,
                "stem": image_path.stem.lower(),
                "image_path": image_path,
                "label_path": label_path,
                "image_found": True,
                "label_found": label_path.exists(),
            }
        )
    return pairs


def collect_testing_candidates(
    images_dir: Path, labels_dir: Path, excluded_stems: set[str]
) -> list[dict[str, object]]:
    """Collect Testing candidates excluding revised hard positive stems."""
    candidates: list[dict[str, object]] = []
    for image_path in collect_image_files(images_dir):
        stem = image_path.stem.lower()
        if stem in excluded_stems:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        candidates.append(
            {
                "filename": image_path.name,
                "stem": stem,
                "image_path": image_path,
                "label_path": label_path,
                "image_found": True,
                "label_found": label_path.exists(),
            }
        )
    return candidates


def classify_environment(image_path: Path) -> str:
    """
    Classify a scene as snow or non_snow using image brightness/saturation.

    This classification is heuristic and may require manual correction.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    height, width = image.shape[:2]
    scale = min(1.0, RESIZE_MAX_SIDE / max(height, width))
    if scale < 1.0:
        resized = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        resized = image

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mean_brightness = float(np.mean(hsv[:, :, 2]))
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    if (
        mean_brightness >= SNOW_BRIGHTNESS_THRESHOLD
        and mean_saturation <= SNOW_SATURATION_THRESHOLD
    ):
        return "snow"
    return "non_snow"


def split_testing_set(
    candidates: list[dict[str, object]],
    val_non_snow: int,
    val_snow: int,
    test_non_snow: int,
    test_snow: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[str]]:
    """
    Split remaining Testing candidates into val/test.

    The script first tries to satisfy the requested snow/non_snow ratio.
    If one environment is insufficient, it fills the remaining test slots with
    any leftover candidates so the final test count can still reach the target.
    """
    warnings: list[str] = []
    snow_items: list[dict[str, object]] = []
    non_snow_items: list[dict[str, object]] = []

    for item in candidates:
        if not item["label_found"]:
            warnings.append(f"Missing label for split candidate: {item['filename']}")
            continue
        try:
            environment = classify_environment(item["image_path"])
            item["environment"] = environment
            if environment == "snow":
                snow_items.append(item)
            else:
                non_snow_items.append(item)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{item['filename']}: {exc}")

    rng = random.Random(seed)
    rng.shuffle(snow_items)
    rng.shuffle(non_snow_items)

    requested_non_snow = val_non_snow + test_non_snow
    requested_snow = val_snow + test_snow
    if len(non_snow_items) < requested_non_snow:
        warnings.append(
            f"Insufficient non_snow candidates: required={requested_non_snow}, available={len(non_snow_items)}"
        )
    if len(snow_items) < requested_snow:
        warnings.append(
            f"Insufficient snow candidates: required={requested_snow}, available={len(snow_items)}"
        )

    total_available = len(non_snow_items) + len(snow_items)
    target_val_total = val_non_snow + val_snow
    target_test_total = test_non_snow + test_snow

    if total_available < target_val_total + target_test_total:
        warnings.append(
            f"Insufficient total candidates: required={target_val_total + target_test_total}, available={total_available}"
        )

    # If the requested 7:3 ratio cannot be satisfied, rebalance val/test to
    # follow the actual remaining environment distribution as closely as possible.
    if len(non_snow_items) < requested_non_snow or len(snow_items) < requested_snow:
        total_non_snow = len(non_snow_items)
        total_snow = len(snow_items)
        available_total = total_non_snow + total_snow

        if available_total == 0:
            return [], [], warnings

        val_snow = int(round(target_val_total * total_snow / available_total))
        val_snow = max(0, min(val_snow, total_snow, target_val_total))
        val_non_snow = target_val_total - val_snow
        if val_non_snow > total_non_snow:
            val_non_snow = total_non_snow
            val_snow = target_val_total - val_non_snow

        remaining_non_snow_capacity = max(0, total_non_snow - val_non_snow)
        remaining_snow_capacity = max(0, total_snow - val_snow)
        test_non_snow = min(target_test_total - 0, remaining_non_snow_capacity)
        test_snow = min(target_test_total - test_non_snow, remaining_snow_capacity)

        # Recompute test split to use all remaining quota with the observed ratio.
        test_snow = int(round(target_test_total * total_snow / available_total))
        test_snow = max(0, min(test_snow, remaining_snow_capacity, target_test_total))
        test_non_snow = target_test_total - test_snow
        if test_non_snow > remaining_non_snow_capacity:
            test_non_snow = remaining_non_snow_capacity
            test_snow = min(target_test_total - test_non_snow, remaining_snow_capacity)

        warnings.append(
            f"Rebalanced val/test to follow available distribution: "
            f"val non_snow/snow={val_non_snow}/{val_snow}, "
            f"test non_snow/snow={test_non_snow}/{test_snow}"
        )

    val_non_snow_items = non_snow_items[:val_non_snow]
    val_snow_items = snow_items[:val_snow]
    val_items = val_non_snow_items + val_snow_items

    remaining_non_snow = non_snow_items[val_non_snow:]
    remaining_snow = snow_items[val_snow:]

    test_non_snow_items = remaining_non_snow[:test_non_snow]
    test_snow_items = remaining_snow[:test_snow]
    test_items = test_non_snow_items + test_snow_items

    leftover_non_snow = remaining_non_snow[len(test_non_snow_items) :]
    leftover_snow = remaining_snow[len(test_snow_items) :]

    if len(test_items) < target_test_total:
        fallback_pool = leftover_non_snow + leftover_snow
        rng.shuffle(fallback_pool)
        needed = target_test_total - len(test_items)
        fallback_items = fallback_pool[:needed]
        test_items.extend(fallback_items)

        if fallback_items:
            warnings.append(
                f"Filled {len(fallback_items)} additional test slots with leftover candidates due to environment shortage."
            )

    if len(test_items) < target_test_total:
        warnings.append(
            f"Unable to fully fill test split: required={target_test_total}, actual={len(test_items)}"
        )

    return val_items, test_items, warnings


def copy_training_set(
    pairs: list[dict[str, object]],
    train_images_out: Path,
    train_labels_out: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[list[dict[str, object]], dict[str, int], list[str]]:
    """Copy all valid Training image/label pairs into train directories."""
    logs: list[dict[str, object]] = []
    warnings: list[str] = []
    stats = {"images": 0, "labels": 0, "skipped": 0}

    ensure_dir(train_images_out)
    ensure_dir(train_labels_out)

    for item in pairs:
        image_path = item["image_path"]
        label_path = item["label_path"]
        reason = ""

        if not item["label_found"]:
            reason = "training_label_missing"
            warnings.append(f"Training mismatch: missing label for {image_path.name}")
            stats["skipped"] += 1
            logs.append(
                {
                    "filename": image_path.name,
                    "stem": item["stem"],
                    "action": "skipped",
                    "environment": "train",
                    "image_found": True,
                    "label_found": False,
                    "reason": reason,
                }
            )
            continue

        dst_image = train_images_out / image_path.name
        dst_label = train_labels_out / label_path.name

        if (dst_image.exists() or dst_label.exists()) and not overwrite:
            reason = "destination_exists"
            stats["skipped"] += 1
            logs.append(
                {
                    "filename": image_path.name,
                    "stem": item["stem"],
                    "action": "skipped",
                    "environment": "train",
                    "image_found": True,
                    "label_found": True,
                    "reason": reason,
                }
            )
            continue

        if dry_run:
            print(f"[DRY-RUN][TRAIN] {image_path.name} -> {dst_image}")
            print(f"[DRY-RUN][TRAIN] {label_path.name} -> {dst_label}")
        else:
            shutil.copy2(str(image_path), str(dst_image))
            shutil.copy2(str(label_path), str(dst_label))

        stats["images"] += 1
        stats["labels"] += 1
        logs.append(
            {
                "filename": image_path.name,
                "stem": item["stem"],
                "action": "train_copy",
                "environment": "train",
                "image_found": True,
                "label_found": True,
                "reason": "copied_to_train",
            }
        )

    return logs, stats, warnings


def copy_reserve_set(
    selected_filenames: list[str],
    testing_images_dir: Path,
    testing_labels_dir: Path,
    reserve_images_out: Path,
    reserve_labels_out: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[list[dict[str, object]], dict[str, int], set[str], list[str]]:
    """Copy revised hard positive files into reserve pools."""
    logs: list[dict[str, object]] = []
    warnings: list[str] = []
    stats = {"images": 0, "labels": 0, "copied": 0, "skipped": 0}
    excluded_stems: set[str] = set()

    ensure_dir(reserve_images_out)
    ensure_dir(reserve_labels_out)

    for filename in selected_filenames:
        stem = normalize_stem(filename)
        image_path = testing_images_dir / filename
        label_path = testing_labels_dir / f"{Path(filename).stem}.txt"
        image_found = image_path.exists()
        label_found = label_path.exists()
        excluded_stems.add(stem)

        if not image_found or not label_found:
            reason = "reserve_source_missing"
            warnings.append(
                f"Reserve missing for {filename}: image_found={image_found}, label_found={label_found}"
            )
            stats["skipped"] += 1
            logs.append(
                {
                    "filename": filename,
                    "stem": stem,
                    "action": "skipped",
                    "environment": "reserve",
                    "image_found": image_found,
                    "label_found": label_found,
                    "reason": reason,
                }
            )
            continue

        dst_image = reserve_images_out / image_path.name
        dst_label = reserve_labels_out / label_path.name

        if (dst_image.exists() or dst_label.exists()) and not overwrite:
            stats["skipped"] += 1
            logs.append(
                {
                    "filename": filename,
                    "stem": stem,
                    "action": "skipped",
                    "environment": "reserve",
                    "image_found": True,
                    "label_found": True,
                    "reason": "destination_exists",
                }
            )
            continue

        if dry_run:
            print(f"[DRY-RUN][RESERVE] {image_path.name} -> {dst_image}")
            print(f"[DRY-RUN][RESERVE] {label_path.name} -> {dst_label}")
        else:
            shutil.copy2(str(image_path), str(dst_image))
            shutil.copy2(str(label_path), str(dst_label))

        stats["images"] += 1
        stats["labels"] += 1
        stats["copied"] += 1
        logs.append(
            {
                "filename": filename,
                "stem": stem,
                "action": "reserve_copy",
                "environment": "reserve",
                "image_found": True,
                "label_found": True,
                "reason": "copied_to_reserve",
            }
        )

    return logs, stats, excluded_stems, warnings


def move_or_copy_split_set(
    items: list[dict[str, object]],
    split_name: str,
    images_out: Path,
    labels_out: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[list[dict[str, object]], dict[str, int], list[str]]:
    """
    Move images and copy labels into val/test split directories.

    Labels are copied with copy2 instead of moved so the converted label source
    remains intact as a reusable canonical annotation store.
    """
    logs: list[dict[str, object]] = []
    warnings: list[str] = []
    stats = {"images": 0, "labels": 0, "snow": 0, "non_snow": 0, "skipped": 0}

    ensure_dir(images_out)
    ensure_dir(labels_out)

    for item in items:
        image_path = item["image_path"]
        label_path = item["label_path"]
        environment = item.get("environment", "unknown")
        image_found = image_path.exists()
        label_found = label_path.exists()

        if not image_found or not label_found:
            reason = "split_source_missing"
            warnings.append(
                f"{split_name} missing for {item['filename']}: image_found={image_found}, label_found={label_found}"
            )
            stats["skipped"] += 1
            logs.append(
                {
                    "filename": item["filename"],
                    "stem": item["stem"],
                    "action": "skipped",
                    "environment": environment,
                    "image_found": image_found,
                    "label_found": label_found,
                    "reason": reason,
                }
            )
            continue

        dst_image = images_out / image_path.name
        dst_label = labels_out / label_path.name

        if (dst_image.exists() or dst_label.exists()) and not overwrite:
            stats["skipped"] += 1
            logs.append(
                {
                    "filename": item["filename"],
                    "stem": item["stem"],
                    "action": "skipped",
                    "environment": environment,
                    "image_found": True,
                    "label_found": True,
                    "reason": "destination_exists",
                }
            )
            continue

        if dry_run:
            print(f"[DRY-RUN][{split_name.upper()}] move image {image_path.name} -> {dst_image}")
            print(f"[DRY-RUN][{split_name.upper()}] copy label {label_path.name} -> {dst_label}")
        else:
            if dst_image.exists() and overwrite:
                dst_image.unlink()
            if dst_label.exists() and overwrite:
                dst_label.unlink()
            shutil.move(str(image_path), str(dst_image))
            shutil.copy2(str(label_path), str(dst_label))

        stats["images"] += 1
        stats["labels"] += 1
        if environment == "snow":
            stats["snow"] += 1
        elif environment == "non_snow":
            stats["non_snow"] += 1

        logs.append(
            {
                "filename": item["filename"],
                "stem": item["stem"],
                "action": f"{split_name}_move",
                "environment": environment,
                "image_found": True,
                "label_found": True,
                "reason": f"assigned_to_{split_name}",
            }
        )

    return logs, stats, warnings


def save_split_log(log_rows: list[dict[str, object]], csv_path: Path) -> None:
    """Save split operation logs to CSV."""
    ensure_dir(csv_path.parent if str(csv_path.parent) not in ("", ".") else Path("."))
    fieldnames = [
        "filename",
        "stem",
        "action",
        "environment",
        "image_found",
        "label_found",
        "reason",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)


def summarize_results(
    train_stats: dict[str, int],
    reserve_stats: dict[str, int],
    val_stats: dict[str, int],
    test_stats: dict[str, int],
    excluded_count: int,
    target_val_count: int,
    target_test_count: int,
    warnings: list[str],
) -> None:
    """Print concise summary of the pipeline results."""
    print("\n[SUMMARY]")
    print(f"train images/labels        : {train_stats['images']} / {train_stats['labels']}")
    print(f"reserve images/labels      : {reserve_stats['images']} / {reserve_stats['labels']}")
    print(f"val images/labels          : {val_stats['images']} / {val_stats['labels']}")
    print(f"test images/labels         : {test_stats['images']} / {test_stats['labels']}")
    print(f"val non_snow / snow        : {val_stats['non_snow']} / {val_stats['snow']}")
    print(f"test non_snow / snow       : {test_stats['non_snow']} / {test_stats['snow']}")
    print(f"reserve copied count       : {reserve_stats['copied']}")
    print(f"excluded hard positive     : {excluded_count}")
    print(
        f"mismatch/missing count     : {len(warnings)}"
    )
    if val_stats["images"] != target_val_count or test_stats["images"] != target_test_count:
        print(
            f"[WARN] split count mismatch: "
            f"val={val_stats['images']}/{target_val_count}, "
            f"test={test_stats['images']}/{target_test_count}"
        )
    if warnings:
        print("[WARNINGS]")
        for warning in warnings:
            print(f"- {warning}")


def main() -> None:
    """Run the full dataset build pipeline."""
    args = parse_args()

    if args.val_non_snow + args.val_snow != args.val_count:
        raise ValueError("val_non_snow + val_snow must equal val_count.")
    if args.test_non_snow + args.test_snow != args.test_count:
        raise ValueError("test_non_snow + test_snow must equal test_count.")

    all_logs: list[dict[str, object]] = []
    all_warnings: list[str] = []

    selected_filenames, selected_stems = load_selected_csv(args.revised_selected_csv)

    training_pairs = collect_training_pairs(args.training_images_dir, args.training_labels_dir)
    train_logs, train_stats, train_warnings = copy_training_set(
        pairs=training_pairs,
        train_images_out=args.train_images_out,
        train_labels_out=args.train_labels_out,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    all_logs.extend(train_logs)
    all_warnings.extend(train_warnings)

    reserve_logs, reserve_stats, excluded_stems, reserve_warnings = copy_reserve_set(
        selected_filenames=selected_filenames,
        testing_images_dir=args.testing_images_dir,
        testing_labels_dir=args.testing_labels_dir,
        reserve_images_out=args.reserve_images_out,
        reserve_labels_out=args.reserve_labels_out,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    all_logs.extend(reserve_logs)
    all_warnings.extend(reserve_warnings)

    if excluded_stems != selected_stems:
        missing_exclusions = selected_stems - excluded_stems
        for stem in sorted(missing_exclusions):
            all_warnings.append(f"Selected stem missing from reserve exclusion set: {stem}")

    testing_candidates = collect_testing_candidates(
        images_dir=args.testing_images_dir,
        labels_dir=args.testing_labels_dir,
        excluded_stems=selected_stems,
    )

    val_items, test_items, split_warnings = split_testing_set(
        candidates=testing_candidates,
        val_non_snow=args.val_non_snow,
        val_snow=args.val_snow,
        test_non_snow=args.test_non_snow,
        test_snow=args.test_snow,
        seed=args.seed,
    )
    all_warnings.extend(split_warnings)

    val_logs, val_stats, val_warnings = move_or_copy_split_set(
        items=val_items,
        split_name="val",
        images_out=args.val_images_out,
        labels_out=args.val_labels_out,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    all_logs.extend(val_logs)
    all_warnings.extend(val_warnings)

    test_logs, test_stats, test_warnings = move_or_copy_split_set(
        items=test_items,
        split_name="test",
        images_out=args.test_images_out,
        labels_out=args.test_labels_out,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    all_logs.extend(test_logs)
    all_warnings.extend(test_warnings)

    save_split_log(all_logs, args.split_log_csv)
    summarize_results(
        train_stats=train_stats,
        reserve_stats=reserve_stats,
        val_stats=val_stats,
        test_stats=test_stats,
        excluded_count=len(selected_stems),
        target_val_count=args.val_count,
        target_test_count=args.test_count,
        warnings=all_warnings,
    )

    print(f"\n[INFO] Split log: {args.split_log_csv}")
    print("\n[EXAMPLE]")
    print(r"python Build_Splits_From_Revised_Selected.py --dry-run")
    print(r"python Build_Splits_From_Revised_Selected.py --overwrite")


if __name__ == "__main__":
    main()
