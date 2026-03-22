"""
Convert binary-like mask images into YOLOv8 segmentation label files.

This script only processes the following two directories:
- Dataset/masks_raw/Training/GT
- Dataset/masks_raw/Testing/GT

The generated labels are saved to:
- converted/labels/Training
- converted/labels/Testing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
THRESHOLD_VALUE = 128
CLASS_ID = 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for mask-to-label conversion."""
    parser = argparse.ArgumentParser(
        description="Convert mask images to YOLOv8 segmentation label files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Dataset") / "masks_raw",
        help="Root directory that contains Training/GT and Testing/GT.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("converted") / "labels",
        help="Directory where converted label files will be saved.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=10.0,
        help="Minimum contour area to keep.",
    )
    parser.add_argument(
        "--approx-epsilon",
        type=float,
        default=0.0,
        help=(
            "Polygon simplification epsilon for cv2.approxPolyDP. "
            "If 0, the original contour points are used."
        ),
    )
    return parser.parse_args()


def get_mask_files(gt_dir: Path) -> list[Path]:
    """Return mask files in a single GT directory without recursive traversal."""
    if not gt_dir.exists():
        return []

    return sorted(
        path
        for path in gt_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def load_mask(mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a mask image in grayscale and threshold it into a binary image.

    Returns:
        gray_mask: Original grayscale mask.
        binary_mask: Thresholded binary mask with values 0 or 255.
    """
    gray_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gray_mask is None:
        raise ValueError("Failed to read image as grayscale.")

    _, binary_mask = cv2.threshold(
        gray_mask, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY
    )
    return gray_mask, binary_mask


def mask_to_contours(
    binary_mask: np.ndarray, min_area: float, approx_epsilon: float
) -> list[np.ndarray]:
    """
    Extract external contours from a binary mask and optionally simplify them.

    Small contours are filtered out by area. OpenCV contours are returned in
    Nx1x2 integer format.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    filtered_contours: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        processed_contour = contour
        if approx_epsilon > 0:
            processed_contour = cv2.approxPolyDP(contour, approx_epsilon, True)

        # YOLO segmentation polygons require at least 3 points.
        if len(processed_contour) < 3:
            continue

        filtered_contours.append(processed_contour)

    return filtered_contours


def contour_to_yolo_polygon(
    contour: np.ndarray, image_width: int, image_height: int
) -> str:
    """
    Convert a contour into one YOLO segmentation label line.

    Output format:
        class_id x1 y1 x2 y2 x3 y3 ...
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")

    points = contour.reshape(-1, 2).astype(np.float32)

    normalized_coords: list[str] = []
    for x, y in points:
        x_norm = np.clip(x / image_width, 0.0, 1.0)
        y_norm = np.clip(y / image_height, 0.0, 1.0)
        normalized_coords.append(f"{x_norm:.6f}")
        normalized_coords.append(f"{y_norm:.6f}")

    return f"{CLASS_ID} " + " ".join(normalized_coords)


def save_label_file(output_path: Path, lines: list[str]) -> None:
    """Save YOLO label lines to a txt file. Creates parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    output_path.write_text(content, encoding="utf-8")


def process_split(
    split_name: str,
    gt_dir: Path,
    output_dir: Path,
    min_area: float,
    approx_epsilon: float,
) -> dict[str, int]:
    """
    Process one split directory and convert all supported mask files.

    Returns a summary dictionary for this split.
    """
    stats = {
        "processed": 0,
        "success": 0,
        "no_contour": 0,
        "error": 0,
    }

    mask_files = get_mask_files(gt_dir)

    if not gt_dir.exists():
        print(f"[WARN] Input directory does not exist: {gt_dir}")
        return stats

    print(f"\n[INFO] Processing split: {split_name}")
    print(f"[INFO] Input:  {gt_dir}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Found {len(mask_files)} mask files")

    for mask_path in mask_files:
        stats["processed"] += 1
        output_path = output_dir / f"{mask_path.stem}.txt"

        try:
            gray_mask, binary_mask = load_mask(mask_path)
            image_height, image_width = gray_mask.shape

            contours = mask_to_contours(
                binary_mask=binary_mask,
                min_area=min_area,
                approx_epsilon=approx_epsilon,
            )

            if not contours:
                # Create an empty txt file when no valid contour exists.
                save_label_file(output_path, [])
                stats["no_contour"] += 1
                stats["success"] += 1
                continue

            lines = [
                contour_to_yolo_polygon(
                    contour=contour,
                    image_width=image_width,
                    image_height=image_height,
                )
                for contour in contours
            ]
            save_label_file(output_path, lines)
            stats["success"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["error"] += 1
            print(f"[ERROR] {mask_path.name}: {exc}")

    return stats


def main() -> None:
    """Entry point for converting Training/Testing GT masks into YOLO labels."""
    args = parse_args()

    if args.min_area < 0:
        raise ValueError("--min-area must be 0 or greater.")

    if args.approx_epsilon < 0:
        raise ValueError("--approx-epsilon must be 0 or greater.")

    dataset_root = args.dataset_root
    output_root = args.output_root

    splits = {
        "Training": dataset_root / "Training" / "GT",
        "Testing": dataset_root / "Testing" / "GT",
    }

    total_stats = {
        "processed": 0,
        "success": 0,
        "no_contour": 0,
        "error": 0,
    }

    for split_name, gt_dir in splits.items():
        split_stats = process_split(
            split_name=split_name,
            gt_dir=gt_dir,
            output_dir=output_root / split_name,
            min_area=args.min_area,
            approx_epsilon=args.approx_epsilon,
        )

        for key in total_stats:
            total_stats[key] += split_stats[key]

    print("\n[SUMMARY]")
    print(f"Processed files : {total_stats['processed']}")
    print(f"Success files   : {total_stats['success']}")
    print(f"No contour files: {total_stats['no_contour']}")
    print(f"Error files     : {total_stats['error']}")

    print("\n[EXAMPLE]")
    print("python Mask_To_txt.py --min-area 20 --approx-epsilon 2.0")


if __name__ == "__main__":
    main()
