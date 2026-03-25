"""
Build an object pool from train images and YOLO segmentation txt labels,
classifying each extracted object itself as snow / non_snow.

읽는 순서:
1. train 이미지와 YOLO segmentation txt를 짝지어 읽습니다.
2. 가장 큰 polygon을 골라 binary mask로 만듭니다.
3. 이미지와 마스크에서 객체 영역을 타이트하게 crop 합니다.
4. crop된 객체를 heuristic 기반으로 snow / non_snow로 분류합니다.
5. 이후 파이프라인에서 재사용할 object pool과 CSV 메타데이터를 저장합니다.

Example:
    python Build_Object_Pool_From_Txt_ObjectBased.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RESIZE_MAX_SIDE = 256

# 이 스크립트는 학습된 분류기가 아니라 HSV/밝기 기반 휴리스틱을 사용합니다.
# 객체 환경 분류가 어색해 보이면 아래 threshold를 조정하는 방식으로 보정하면 됩니다.
# Heuristic thresholds for object-based environment classification.
# This classification is heuristic and may require manual correction.
SNOW_BRIGHTNESS_THRESHOLD = 165.0
SNOW_SATURATION_THRESHOLD = 55.0
SNOW_WHITE_RATIO_THRESHOLD = 0.35
WHITE_VALUE_THRESHOLD = 180
WHITE_SATURATION_THRESHOLD = 50


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build an object pool from train images and YOLO segmentation txt labels using object-based classification."
    )
    parser.add_argument(
        "--train-images-dir",
        type=Path,
        default=Path("Dataset") / "images" / "train",
        help="Train image directory.",
    )
    parser.add_argument(
        "--train-labels-dir",
        type=Path,
        default=Path("Dataset") / "labels" / "train",
        help="Train YOLO segmentation label directory.",
    )
    parser.add_argument(
        "--object-pool-dir",
        type=Path,
        default=Path("Dataset") / "object_pool",
        help="Output object pool directory.",
    )
    parser.add_argument(
        "--environment-csv",
        type=Path,
        default=Path("train_object_environment_labels.csv"),
        help="CSV output for object-based environment labels.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("object_pool_metadata.csv"),
        help="CSV output for object metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output object files.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def collect_pairs(
    images_dir: Path, labels_dir: Path
) -> tuple[list[dict[str, Path]], list[str]]:
    """Collect image/txt pairs by shared stem."""
    if not images_dir.exists():
        raise FileNotFoundError(f"Train image directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Train label directory not found: {labels_dir}")

    warnings: list[str] = []
    pairs: list[dict[str, Path]] = []

    image_files = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )
    label_files = sorted(
        path for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    )

    label_by_stem = {path.stem.lower(): path for path in label_files}
    image_stems = {path.stem.lower() for path in image_files}

    for image_path in image_files:
        stem = image_path.stem.lower()
        label_path = label_by_stem.get(stem)
        if label_path is None:
            warnings.append(f"Missing label for image: {image_path.name}")
            continue
        pairs.append(
            {
                "stem": Path(stem),
                "image_path": image_path,
                "label_path": label_path,
            }
        )

    for label_path in label_files:
        if label_path.stem.lower() not in image_stems:
            warnings.append(f"Missing image for label: {label_path.name}")

    return pairs, warnings


def load_polygon_from_txt(
    label_path: Path, image_width: int, image_height: int
) -> np.ndarray | None:
    """
    Load polygons from YOLO segmentation txt and return the largest polygon.

    If multiple polygons exist, the largest area polygon is used.
    """
    lines = label_path.read_text(encoding="utf-8").splitlines()
    polygons: list[np.ndarray] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 7:
            continue

        coords = parts[1:]
        if len(coords) % 2 != 0:
            continue

        points: list[list[int]] = []
        for index in range(0, len(coords), 2):
            x_norm = float(coords[index])
            y_norm = float(coords[index + 1])
            x = int(round(np.clip(x_norm, 0.0, 1.0) * (image_width - 1)))
            y = int(round(np.clip(y_norm, 0.0, 1.0) * (image_height - 1)))
            points.append([x, y])

        if len(points) < 3:
            continue

        polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        polygons.append(polygon)

    if not polygons:
        return None

    polygons.sort(key=cv2.contourArea, reverse=True)
    return polygons[0]


def polygon_to_mask(polygon: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Convert a polygon into a binary mask."""
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def extract_object(image: np.ndarray, mask: np.ndarray) -> dict[str, object] | None:
    """
    Extract one object crop using the full mask region.

    The object is defined by the nonzero region of the binary mask.
    """
    foreground_points = cv2.findNonZero(mask)
    if foreground_points is None:
        return None

    object_image = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(foreground_points)

    crop_image = object_image[y : y + h, x : x + w].copy()
    crop_mask = mask[y : y + h, x : x + w].copy()
    object_area = float(np.count_nonzero(crop_mask))

    return {
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
        "object_area": object_area,
        "crop_image": crop_image,
        "crop_mask": crop_mask,
    }


def classify_environment(
    object_image: np.ndarray, object_mask: np.ndarray
) -> tuple[str, float, float, float]:
    """
    Classify the extracted object itself as snow or non_snow.

    Only pixels inside the object mask are used for statistics.
    This classification is heuristic and may require manual correction.
    """
    height, width = object_image.shape[:2]
    scale = min(1.0, RESIZE_MAX_SIDE / max(height, width))
    if scale < 1.0:
        resized_image = cv2.resize(
            object_image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        resized_mask = cv2.resize(
            object_mask,
            (resized_image.shape[1], resized_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        resized_image = object_image
        resized_mask = object_mask

    object_pixels = resized_mask > 0
    if not np.any(object_pixels):
        raise ValueError("Object mask has no foreground pixels.")

    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    brightness_values = hsv[:, :, 2][object_pixels]
    saturation_values = hsv[:, :, 1][object_pixels]

    brightness_mean = float(np.mean(brightness_values))
    saturation_mean = float(np.mean(saturation_values))

    white_pixels = (
        (hsv[:, :, 2] >= WHITE_VALUE_THRESHOLD)
        & (hsv[:, :, 1] <= WHITE_SATURATION_THRESHOLD)
        & object_pixels
    )
    white_ratio = float(np.count_nonzero(white_pixels)) / float(np.count_nonzero(object_pixels))

    if (
        brightness_mean >= SNOW_BRIGHTNESS_THRESHOLD
        and saturation_mean <= SNOW_SATURATION_THRESHOLD
        and white_ratio >= SNOW_WHITE_RATIO_THRESHOLD
    ):
        return "snow", brightness_mean, saturation_mean, white_ratio
    return "non_snow", brightness_mean, saturation_mean, white_ratio


def save_object(
    object_entry: dict[str, object],
    source_stem: str,
    environment: str,
    object_pool_dir: Path,
    overwrite: bool,
) -> str:
    """Save one object image and mask into the environment-specific pool."""
    env_root = object_pool_dir / environment
    images_dir = env_root / "images"
    masks_dir = env_root / "masks"
    ensure_dir(images_dir)
    ensure_dir(masks_dir)

    object_name = f"{source_stem}_obj1"
    image_path = images_dir / f"{object_name}.jpg"
    mask_path = masks_dir / f"{object_name}.png"

    if (image_path.exists() or mask_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Output already exists: {image_path.name} or {mask_path.name}"
        )

    crop_image = object_entry["crop_image"]
    crop_mask = object_entry["crop_mask"]
    assert isinstance(crop_image, np.ndarray)
    assert isinstance(crop_mask, np.ndarray)

    if not cv2.imwrite(str(image_path), crop_image):
        raise ValueError(f"Failed to save object image: {image_path}")
    if not cv2.imwrite(str(mask_path), crop_mask):
        raise ValueError(f"Failed to save object mask: {mask_path}")

    return object_name


def save_csv(rows: list[dict[str, object]], csv_path: Path, fieldnames: list[str]) -> None:
    """Save rows to a CSV file."""
    ensure_dir(csv_path.parent if str(csv_path.parent) not in ("", ".") else Path("."))
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_tree(root: Path) -> None:
    """Print a compact directory tree for object_pool."""
    max_files_to_show = 8

    def walk(path: Path, prefix: str = "") -> None:
        entries = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        dirs = [entry for entry in entries if entry.is_dir()]
        files = [entry for entry in entries if entry.is_file()]

        for index, directory in enumerate(dirs):
            is_last = index == len(dirs) - 1 and not files
            branch = "└─ " if is_last else "├─ "
            file_count = sum(1 for item in directory.iterdir() if item.is_file())
            print(f"{prefix}{branch}{directory.name}/ ({file_count} files)")
            extension = "   " if is_last else "│  "
            walk(directory, prefix + extension)

        if files:
            branch = "└─ "
            if len(files) <= max_files_to_show:
                print(f"{prefix}{branch}files ({len(files)}): " + ", ".join(file.name for file in files))
            else:
                print(f"{prefix}{branch}files ({len(files)})")

    print(f"\n{root.as_posix()}/")
    if root.exists():
        walk(root)
    else:
        print("(directory does not exist)")


def main() -> None:
    """Run object-pool generation from train images and txt polygons."""
    args = parse_args()

    pairs, mismatch_warnings = collect_pairs(args.train_images_dir, args.train_labels_dir)

    total_images = len(
        [
            path
            for path in args.train_images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
    )

    environment_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    skipped: list[str] = list(mismatch_warnings)

    snow_objects = 0
    non_snow_objects = 0
    object_count = 0

    for pair in pairs:
        image_path = pair["image_path"]
        label_path = pair["label_path"]
        source_stem = image_path.stem

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            image_height, image_width = image.shape[:2]
            polygon = load_polygon_from_txt(label_path, image_width, image_height)
            if polygon is None:
                skipped.append(f"No valid polygon in label: {label_path.name}")
                continue

            mask = polygon_to_mask(polygon, image_width, image_height)
            object_entry = extract_object(image, mask)
            if object_entry is None:
                skipped.append(f"No foreground from polygon mask: {image_path.name}")
                continue

            crop_image = object_entry["crop_image"]
            crop_mask = object_entry["crop_mask"]
            assert isinstance(crop_image, np.ndarray)
            assert isinstance(crop_mask, np.ndarray)

            environment, brightness_mean, saturation_mean, white_ratio = classify_environment(
                crop_image, crop_mask
            )

            object_name = save_object(
                object_entry=object_entry,
                source_stem=source_stem,
                environment=environment,
                object_pool_dir=args.object_pool_dir,
                overwrite=args.overwrite,
            )

            environment_rows.append(
                {
                    "source_image": image_path.name,
                    "object_name": object_name,
                    "environment": environment,
                    "brightness_mean": f"{brightness_mean:.6f}",
                    "saturation_mean": f"{saturation_mean:.6f}",
                    "white_ratio": f"{white_ratio:.6f}",
                }
            )
            metadata_rows.append(
                {
                    "source_image": image_path.name,
                    "object_name": object_name,
                    "environment": environment,
                    "bbox_x": int(object_entry["bbox_x"]),
                    "bbox_y": int(object_entry["bbox_y"]),
                    "bbox_w": int(object_entry["bbox_w"]),
                    "bbox_h": int(object_entry["bbox_h"]),
                    "object_area": float(object_entry["object_area"]),
                }
            )

            object_count += 1
            if environment == "snow":
                snow_objects += 1
            else:
                non_snow_objects += 1
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{image_path.name}: {exc}")

    save_csv(
        environment_rows,
        args.environment_csv,
        [
            "source_image",
            "object_name",
            "environment",
            "brightness_mean",
            "saturation_mean",
            "white_ratio",
        ],
    )
    save_csv(
        metadata_rows,
        args.metadata_csv,
        [
            "source_image",
            "object_name",
            "environment",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "object_area",
        ],
    )

    print("\n[SUMMARY]")
    print(f"Total images        : {total_images}")
    print(f"Matched pairs       : {len(pairs)}")
    print(f"Matching failures   : {len(mismatch_warnings)}")
    print(f"Snow objects        : {snow_objects}")
    print(f"Non-snow objects    : {non_snow_objects}")
    print(f"Generated objects   : {object_count}")
    print(f"Skipped items       : {len(skipped)}")
    if skipped:
        print("[SKIPPED]")
        for item in skipped:
            print(f"- {item}")

    print(f"\n[INFO] Environment CSV : {args.environment_csv}")
    print(f"[INFO] Metadata CSV    : {args.metadata_csv}")
    print("\n[OBJECT POOL TREE]")
    print_tree(args.object_pool_dir)

    print("\n[EXAMPLE]")
    print(r"python Build_Object_Pool_From_Txt_ObjectBased.py")


if __name__ == "__main__":
    main()
