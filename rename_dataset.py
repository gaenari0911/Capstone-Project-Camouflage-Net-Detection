"""
Rename dataset images and YOLO txt labels while preserving image-label pairing.

Targets:
1. Dataset/reserve/hard_positive_pool + Dataset/reserve_labels/hard_positive_pool
2. Dataset/images/train|val|test + Dataset/labels/train|val|test

Example:
    python rename_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Safely rename dataset images and matching YOLO txt labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset"),
        help="Root directory containing dataset subfolders.",
    )
    return parser.parse_args()


def collect_pairs(images_dir: Path, labels_dir: Path) -> tuple[list[dict[str, Path]], int]:
    """
    Collect image/txt pairs by shared stem.

    Returns:
        pairs: matched image-label pairs
        skipped_count: unmatched files that were skipped
    """
    pairs: list[dict[str, Path]] = []
    skipped_count = 0

    if not images_dir.exists():
        print(f"[WARN] Image directory not found: {images_dir}")
        return pairs, skipped_count

    if not labels_dir.exists():
        print(f"[WARN] Label directory not found: {labels_dir}")
        return pairs, skipped_count

    image_files = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )

    image_stems = {path.stem for path in image_files}
    label_files = sorted(
        path for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    )
    label_stems = {path.stem for path in label_files}

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARN] Missing label for image, skipping: {image_path.name}")
            skipped_count += 1
            continue

        pairs.append(
            {
                "stem": Path(image_path.stem),
                "image_path": image_path,
                "label_path": label_path,
            }
        )

    for label_path in label_files:
        if label_path.stem not in image_stems:
            print(f"[WARN] Missing image for label, skipping: {label_path.name}")
            skipped_count += 1

    # `label_stems` is computed above intentionally so both sides are checked.
    _ = label_stems

    return pairs, skipped_count


def sort_pairs(pairs: list[dict[str, Path]]) -> list[dict[str, Path]]:
    """Sort matched pairs by filename ascending."""
    return sorted(pairs, key=lambda item: item["image_path"].name.lower())


def rename_with_temp(
    pairs: list[dict[str, Path]], prefix: str
) -> list[dict[str, Path | str]]:
    """
    Rename all matched pairs to temporary names first to avoid collisions.

    Returns records with temp paths for final rename.
    """
    temp_records: list[dict[str, Path | str]] = []

    for index, pair in enumerate(pairs, start=1):
        image_path = pair["image_path"]
        label_path = pair["label_path"]

        temp_image = image_path.with_name(f"__tmp__{prefix}_{index}{image_path.suffix.lower()}")
        temp_label = label_path.with_name(f"__tmp__{prefix}_{index}.txt")

        if temp_image.exists() or temp_label.exists():
            raise FileExistsError(
                f"Temporary rename target already exists: {temp_image.name} or {temp_label.name}"
            )

        image_path.rename(temp_image)
        label_path.rename(temp_label)

        temp_records.append(
            {
                "temp_image": temp_image,
                "temp_label": temp_label,
                "original_image": image_path,
                "original_label": label_path,
            }
        )

    return temp_records


def rename_final(
    temp_records: list[dict[str, Path | str]],
    images_dir: Path,
    labels_dir: Path,
    final_prefix: str,
) -> int:
    """
    Rename temporary files to final dataset names.

    Returns:
        rename_count: number of image-label pairs renamed
    """
    for index, record in enumerate(temp_records, start=1):
        temp_image = record["temp_image"]
        temp_label = record["temp_label"]

        assert isinstance(temp_image, Path)
        assert isinstance(temp_label, Path)

        final_image = images_dir / f"{final_prefix}{index}{temp_image.suffix.lower()}"
        final_label = labels_dir / f"{final_prefix}{index}.txt"

        if final_image.exists() or final_label.exists():
            raise FileExistsError(
                f"Final rename target already exists: {final_image.name} or {final_label.name}"
            )

    for index, record in enumerate(temp_records, start=1):
        temp_image = record["temp_image"]
        temp_label = record["temp_label"]

        assert isinstance(temp_image, Path)
        assert isinstance(temp_label, Path)

        final_image = images_dir / f"{final_prefix}{index}{temp_image.suffix.lower()}"
        final_label = labels_dir / f"{final_prefix}{index}.txt"

        temp_image.rename(final_image)
        temp_label.rename(final_label)

    return len(temp_records)


def process_directory_pair(
    images_dir: Path,
    labels_dir: Path,
    final_prefix: str,
) -> tuple[int, int]:
    """
    Process one directory pair using two-phase rename.

    Returns:
        renamed_pairs: number of renamed image-label pairs
        skipped_count: number of skipped unmatched files
    """
    pairs, skipped_count = collect_pairs(images_dir, labels_dir)
    sorted_pairs = sort_pairs(pairs)

    if not sorted_pairs:
        print(f"[INFO] No matched pairs to rename in: {images_dir}")
        return 0, skipped_count

    temp_records = rename_with_temp(sorted_pairs, final_prefix)
    renamed_pairs = rename_final(temp_records, images_dir, labels_dir, final_prefix)

    print(
        f"[INFO] Processed {images_dir} <-> {labels_dir}: "
        f"renamed_pairs={renamed_pairs}, skipped={skipped_count}"
    )
    return renamed_pairs, skipped_count


def main() -> None:
    """Run rename jobs for reserve, train, val, and test."""
    args = parse_args()
    dataset_root = args.dataset_root

    jobs = [
        (
            dataset_root / "reserve" / "hard_positive_pool",
            dataset_root / "reserve_labels" / "hard_positive_pool",
            "FN",
        ),
        (
            dataset_root / "images" / "train",
            dataset_root / "labels" / "train",
            "train",
        ),
        (
            dataset_root / "images" / "val",
            dataset_root / "labels" / "val",
            "val",
        ),
        (
            dataset_root / "images" / "test",
            dataset_root / "labels" / "test",
            "test",
        ),
    ]

    total_renamed = 0
    total_skipped = 0

    for images_dir, labels_dir, final_prefix in jobs:
        renamed_pairs, skipped_count = process_directory_pair(
            images_dir=images_dir,
            labels_dir=labels_dir,
            final_prefix=final_prefix,
        )
        total_renamed += renamed_pairs
        total_skipped += skipped_count

    print("\n[SUMMARY]")
    print(f"Total renamed pairs: {total_renamed}")
    print(f"Total skipped files: {total_skipped}")

    print("\n[EXAMPLE]")
    print("python rename_dataset.py")


if __name__ == "__main__":
    main()
