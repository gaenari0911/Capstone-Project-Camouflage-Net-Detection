from pathlib import Path

import cv2
import numpy as np


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MASK_THRESHOLD = 128

IMAGE_DIR = Path("Dataset") / "masks_raw" / "Training" / "images"
MASK_DIR = Path("Dataset") / "masks_raw" / "Training" / "GT"


def collect_image_files(directory: Path) -> list[Path]:
    """Return sorted image files from the given directory."""
    if not directory.exists():
        return []
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )


def build_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Match training images with GT masks using the file stem."""
    image_files = collect_image_files(image_dir)
    mask_files = collect_image_files(mask_dir)
    mask_by_stem = {path.stem.lower(): path for path in mask_files}

    pairs: list[tuple[Path, Path]] = []
    for image_path in image_files:
        mask_path = mask_by_stem.get(image_path.stem.lower())
        if mask_path is None:
            print(f"[SKIP] matching GT not found: {image_path.name}")
            continue
        pairs.append((image_path, mask_path))
    return pairs


def load_mask(mask_path: Path) -> np.ndarray | None:
    """Load GT mask as grayscale."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[SKIP] failed to load GT: {mask_path}")
    return mask


def compute_mask_coverage(mask: np.ndarray) -> float:
    """
    Compute the foreground area ratio in the mask.
    Bright pixels are treated as segmentation foreground.
    """
    _, binary = cv2.threshold(mask, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    foreground_pixels = int(np.count_nonzero(binary))
    total_pixels = int(binary.shape[0] * binary.shape[1])

    if total_pixels <= 0:
        raise ValueError("Mask has invalid size.")

    return foreground_pixels / float(total_pixels)


def format_percent(value: float) -> str:
    """Format a 0~1 ratio into a percentage string."""
    return f"{value * 100:.4f}%"


def print_summary_statistics(coverage_array: np.ndarray) -> None:
    """Print the main descriptive statistics."""
    mean_value = float(np.mean(coverage_array))
    median_value = float(np.median(coverage_array))
    lower_quartile = float(np.percentile(coverage_array, 25))
    upper_quartile = float(np.percentile(coverage_array, 75))
    min_value = float(np.min(coverage_array))
    max_value = float(np.max(coverage_array))

    print("[Mask Coverage Statistics]")
    print(f"Valid pairs   : {len(coverage_array)}")
    print(f"Mean          : {format_percent(mean_value)}")
    print(f"Median        : {format_percent(median_value)}")
    print(f"Lower 25%     : {format_percent(lower_quartile)}")
    print(f"Upper 25%     : {format_percent(upper_quartile)}")
    print(f"Minimum       : {format_percent(min_value)}")
    print(f"Maximum       : {format_percent(max_value)}")


def print_coverage_bins(coverage_array: np.ndarray) -> None:
    """Print how many masks fall into each 5% coverage interval."""
    print("\n[Coverage Distribution]")

    first_count = int(np.sum(coverage_array < 0.05))
    print(f"{'5% 미만':<24}: {first_count}개")

    for start in range(5, 80, 5):
        end = start + 5
        start_ratio = start / 100.0
        end_ratio = end / 100.0
        count = int(np.sum((coverage_array >= start_ratio) & (coverage_array < end_ratio)))
        label = f"{start}% 이상 {end}% 미만"
        print(f"{label:<24}: {count}개")

    over_count = int(np.sum(coverage_array >= 0.80))
    print(f"{'80% 이상':<24}: {over_count}개")


def main() -> None:
    if not IMAGE_DIR.exists():
        print(f"[ERROR] image directory not found: {IMAGE_DIR}")
        return

    if not MASK_DIR.exists():
        print(f"[ERROR] GT directory not found: {MASK_DIR}")
        return

    pairs = build_pairs(IMAGE_DIR, MASK_DIR)
    if not pairs:
        print("[ERROR] no valid image-GT pairs found.")
        return

    coverage_values: list[float] = []

    for _, mask_path in pairs:
        try:
            mask = load_mask(mask_path)
            if mask is None:
                continue
            coverage_values.append(compute_mask_coverage(mask))
        except Exception as exc:
            print(f"[SKIP] {mask_path.name}: {exc}")

    if not coverage_values:
        print("[ERROR] no valid mask coverage values computed.")
        return

    coverage_array = np.array(coverage_values, dtype=np.float64)
    print_summary_statistics(coverage_array)
    print_coverage_bins(coverage_array)


if __name__ == "__main__":
    main()
