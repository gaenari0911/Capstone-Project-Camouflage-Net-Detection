"""
Visualize YOLO segmentation polygons by drawing red outlines on an image.

Expected project structure:
- Dataset/masks_raw/<split>/images/<image_name>.jpg
- converted/labels/<split>/<image_name>.txt

Example:
    python Test_RedLine.py --split Training --name image14
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    """Parse arguments for polygon visualization."""
    parser = argparse.ArgumentParser(
        description="Draw YOLO segmentation polygons as red lines on an image."
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("Training", "Testing"),
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Image base name without extension. Example: image14",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("Dataset") / "masks_raw",
        help="Root directory that contains split image folders.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("converted") / "labels",
        help="Root directory that contains converted YOLO label txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("converted") / "visualizations",
        help="Directory where visualization images are saved.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Polygon line thickness.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a window to display the result.",
    )
    return parser.parse_args()


def find_image_file(images_dir: Path, image_stem: str) -> Path:
    """Find the image file by trying supported extensions."""
    for extension in VALID_IMAGE_EXTENSIONS:
        candidate = images_dir / f"{image_stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Image file not found for '{image_stem}' in {images_dir} "
        f"with extensions {VALID_IMAGE_EXTENSIONS}"
    )


def load_label_polygons(label_path: Path, image_width: int, image_height: int) -> list[np.ndarray]:
    """
    Load YOLO segmentation polygons and convert normalized coordinates to pixels.

    Each returned polygon is shaped for cv2.polylines as Nx1x2 int32.
    """
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    polygons: list[np.ndarray] = []
    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_index, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7:
            raise ValueError(
                f"Invalid polygon format at line {line_index} in {label_path.name}"
            )

        coord_values = parts[1:]
        if len(coord_values) % 2 != 0:
            raise ValueError(
                f"Coordinate count must be even at line {line_index} in {label_path.name}"
            )

        points: list[list[int]] = []
        for index in range(0, len(coord_values), 2):
            x_norm = float(coord_values[index])
            y_norm = float(coord_values[index + 1])

            x_pixel = int(round(np.clip(x_norm, 0.0, 1.0) * (image_width - 1)))
            y_pixel = int(round(np.clip(y_norm, 0.0, 1.0) * (image_height - 1)))
            points.append([x_pixel, y_pixel])

        polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        polygons.append(polygon)

    return polygons


def main() -> None:
    """Load one image, draw red polygon lines, and save the result."""
    args = parse_args()

    images_dir = args.images_root / args.split / "images"
    labels_dir = args.labels_root / args.split

    image_path = find_image_file(images_dir, args.name)
    label_path = labels_dir / f"{args.name}.txt"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image_height, image_width = image.shape[:2]
    polygons = load_label_polygons(label_path, image_width, image_height)

    visualized = image.copy()

    if polygons:
        # OpenCV uses BGR, so (0, 0, 255) is red.
        cv2.polylines(
            visualized,
            polygons,
            isClosed=True,
            color=(0, 0, 255),
            thickness=args.thickness,
            lineType=cv2.LINE_AA,
        )
    else:
        print(f"[INFO] Label file is empty: {label_path.name}")

    output_dir = args.output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.name}_redline.jpg"

    if not cv2.imwrite(str(output_path), visualized):
        raise ValueError(f"Failed to save visualization image: {output_path}")

    print(f"[INFO] Image:  {image_path}")
    print(f"[INFO] Label:  {label_path}")
    print(f"[INFO] Saved:  {output_path}")
    print(f"[INFO] Polygons drawn: {len(polygons)}")

    if args.show:
        cv2.imshow("YOLO Polygon Visualization", visualized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
