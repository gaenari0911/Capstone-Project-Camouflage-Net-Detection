"""
Integrated hard-positive pipeline.

읽는 순서:
1. testing 이미지/GT 쌍을 모읍니다.
2. 후보마다 handcrafted difficulty 지표를 계산합니다.
3. 이 지표들을 hard-positive score로 변환합니다.
4. 환경 비율을 고려해 1차 선택을 수행합니다.
5. similarity group을 이용해 너무 비슷한 샘플이 몰리지 않게 보정합니다.
6. 최종 CSV 로그를 저장하고 필요하면 파일도 이동합니다.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MASK_THRESHOLD = 128
SNOW_BRIGHTNESS_THRESHOLD = 160.0
SNOW_SATURATION_THRESHOLD = 60.0

WEIGHT_SMALL_OBJECT = 0.30
WEIGHT_LOW_COLOR_DISTANCE = 0.25
WEIGHT_LOW_CONTRAST = 0.20
WEIGHT_BLUR = 0.20
WEIGHT_EDGE_COMPLEXITY = 0.05

# 여기서는 score가 높을수록 더 어려운 hard positive 샘플이라는 뜻입니다.
# 아래 weight는 어떤 시각적 특성을 더 중요하게 볼지 정의합니다.

SIMILARITY_GROUPS_RAW = {
    "non_snow": [
        ["image741", "image742"],
        ["image743", "image745", "image746", "image747", "image748", "image750", "image752"],
    ],
    "snow": [
        ["image973", "image974", "image975", "image977"],
        ["image983", "image984", "image985", "image986", "image987","image981","image982", "image979"],
    ],
}


@dataclass
class CandidateRecord:
    filename: str
    stem: str
    image_path: Path
    mask_path: Path
    environment: str
    object_area_ratio: float
    fg_bg_color_distance: float
    contrast_score: float
    blur_score: float
    edge_complexity: float
    hard_positive_score: float = 0.0
    selected: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score, revise, and move hard positive candidates."
    )
    parser.add_argument(
        "--testing-images-dir",
        type=Path,
        default=Path("Dataset") / "masks_raw" / "Testing" / "images",
    )
    parser.add_argument(
        "--testing-masks-dir",
        type=Path,
        default=Path("Dataset") / "masks_raw" / "Testing" / "GT",
    )
    parser.add_argument(
        "--test-images-dir",
        type=Path,
        default=Path("Dataset") / "images" / "test",
    )
    parser.add_argument(
        "--test-labels-dir",
        type=Path,
        default=Path("Dataset") / "labels" / "test",
    )
    parser.add_argument(
        "--reserve-images-dir",
        type=Path,
        default=Path("Dataset") / "reserve" / "hard_positive_pool",
    )
    parser.add_argument(
        "--reserve-labels-dir",
        type=Path,
        default=Path("Dataset") / "reserve_labels" / "hard_positive_pool",
    )
    parser.add_argument("--scores-csv", type=Path, default=Path("hard_positive_scores.csv"))
    parser.add_argument("--selected-csv", type=Path, default=Path("hard_positive_selected.csv"))
    parser.add_argument(
        "--revised-selected-csv",
        type=Path,
        default=Path("revised_hard_positive_selected.csv"),
    )
    parser.add_argument(
        "--change-log-csv",
        type=Path,
        default=Path("revised_hard_positive_change_log.csv"),
    )
    parser.add_argument("--target-total", type=int, default=30)
    parser.add_argument("--non-snow-count", type=int, default=21)
    parser.add_argument("--snow-count", type=int, default=9)
    parser.add_argument("--revise-existing-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_stem(filename: str) -> str:
    return Path(str(filename)).stem.lower()


def build_similarity_groups() -> tuple[dict[str, list[str]], dict[str, str]]:
    groups_by_id: dict[str, list[str]] = {}
    stem_to_group: dict[str, str] = {}

    for environment, groups in SIMILARITY_GROUPS_RAW.items():
        for index, group in enumerate(groups, start=1):
            group_id = f"{environment}_group_{index}"
            members = [normalize_stem(item) for item in group]
            groups_by_id[group_id] = members
            for stem in members:
                stem_to_group[stem] = group_id

    return groups_by_id, stem_to_group


def collect_image_mask_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []

    if not images_dir.exists():
        print(f"[ERROR] Testing image directory does not exist: {images_dir}")
        return pairs

    if not masks_dir.exists():
        print(f"[ERROR] Testing mask directory does not exist: {masks_dir}")
        return pairs

    image_files = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )

    for image_path in image_files:
        mask_path = None
        for extension in VALID_IMAGE_EXTENSIONS:
            candidate = masks_dir / f"{image_path.stem}{extension}"
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            print(f"[WARN] Matching mask not found for image: {image_path.name}")
            continue

        pairs.append((image_path, mask_path))

    return pairs


def load_mask(mask_path: Path) -> np.ndarray:
    gray_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gray_mask is None:
        raise ValueError("Failed to read GT mask image.")

    _, binary_mask = cv2.threshold(gray_mask, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary_mask


def classify_environment(image: np.ndarray) -> str:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv_image[:, :, 1]))
    mean_brightness = float(np.mean(hsv_image[:, :, 2]))

    if (
        mean_brightness >= SNOW_BRIGHTNESS_THRESHOLD
        and mean_saturation <= SNOW_SATURATION_THRESHOLD
    ):
        return "snow"
    return "non_snow"


def compute_object_area_ratio(binary_mask: np.ndarray) -> float:
    object_pixels = float(np.count_nonzero(binary_mask))
    total_pixels = float(binary_mask.size)
    return object_pixels / total_pixels


def compute_fg_bg_color_distance(image: np.ndarray, binary_mask: np.ndarray) -> float:
    foreground_mask = binary_mask > 0
    background_mask = ~foreground_mask

    if not np.any(foreground_mask) or not np.any(background_mask):
        raise ValueError("Foreground or background region is empty.")

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    fg_mean = np.mean(lab_image[foreground_mask], axis=0)
    bg_mean = np.mean(lab_image[background_mask], axis=0)
    return float(np.linalg.norm(fg_mean - bg_mean))


def compute_contrast_score(image: np.ndarray, binary_mask: np.ndarray) -> float:
    foreground_mask = binary_mask > 0
    background_mask = ~foreground_mask

    if not np.any(foreground_mask) or not np.any(background_mask):
        raise ValueError("Foreground or background region is empty.")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fg_mean = float(np.mean(gray_image[foreground_mask]))
    bg_mean = float(np.mean(gray_image[background_mask]))
    return abs(fg_mean - bg_mean)


def compute_blur_score(image: np.ndarray) -> float:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def compute_edge_complexity(binary_mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("No contour found in GT mask.")

    total_perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    object_pixels = float(np.count_nonzero(binary_mask))
    if object_pixels <= 0:
        raise ValueError("Object area is zero.")

    return total_perimeter / np.sqrt(object_pixels)


def normalize_scores(values: list[float], invert: bool = False) -> list[float]:
    if not values:
        return []

    values_array = np.array(values, dtype=np.float32)
    min_value = float(np.min(values_array))
    max_value = float(np.max(values_array))

    if max_value == min_value:
        normalized = np.full_like(values_array, 0.5, dtype=np.float32)
    else:
        normalized = (values_array - min_value) / (max_value - min_value)

    if invert:
        normalized = 1.0 - normalized

    return [float(value) for value in normalized]


def compute_hard_positive_score(records: list[CandidateRecord]) -> None:
    area_scores = normalize_scores(
        [record.object_area_ratio for record in records], invert=True
    )
    color_scores = normalize_scores(
        [record.fg_bg_color_distance for record in records], invert=True
    )
    contrast_scores = normalize_scores(
        [record.contrast_score for record in records], invert=True
    )
    blur_scores = normalize_scores([record.blur_score for record in records], invert=True)
    edge_scores = normalize_scores([record.edge_complexity for record in records])

    for index, record in enumerate(records):
        record.hard_positive_score = (
            WEIGHT_SMALL_OBJECT * area_scores[index]
            + WEIGHT_LOW_COLOR_DISTANCE * color_scores[index]
            + WEIGHT_LOW_CONTRAST * contrast_scores[index]
            + WEIGHT_BLUR * blur_scores[index]
            + WEIGHT_EDGE_COMPLEXITY * edge_scores[index]
        )


def score_testing_pairs(pairs: list[tuple[Path, Path]]) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []

    for image_path, mask_path in pairs:
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Failed to read testing image.")

            binary_mask = load_mask(mask_path)
            if binary_mask.shape[:2] != image.shape[:2]:
                print(
                    f"[WARN] Size mismatch, resizing mask: {mask_path.name} "
                    f"{binary_mask.shape[:2]} -> {image.shape[:2]}"
                )
                binary_mask = cv2.resize(
                    binary_mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            object_area_ratio = compute_object_area_ratio(binary_mask)
            if object_area_ratio <= 0:
                print(f"[WARN] Empty object mask, skipping: {mask_path.name}")
                continue

            records.append(
                CandidateRecord(
                    filename=image_path.name,
                    stem=image_path.stem.lower(),
                    image_path=image_path,
                    mask_path=mask_path,
                    environment=classify_environment(image),
                    object_area_ratio=object_area_ratio,
                    fg_bg_color_distance=compute_fg_bg_color_distance(image, binary_mask),
                    contrast_score=compute_contrast_score(image, binary_mask),
                    blur_score=compute_blur_score(image),
                    edge_complexity=compute_edge_complexity(binary_mask),
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {image_path.name}: {exc}")

    if records:
        compute_hard_positive_score(records)

    return records


def select_candidates_by_ratio(
    records: list[CandidateRecord], non_snow_count: int, snow_count: int
) -> list[dict[str, object]]:
    snow_records = sorted(
        (record for record in records if record.environment == "snow"),
        key=lambda item: (-item.hard_positive_score, item.filename.lower()),
    )
    non_snow_records = sorted(
        (record for record in records if record.environment == "non_snow"),
        key=lambda item: (-item.hard_positive_score, item.filename.lower()),
    )

    selected_non_snow = non_snow_records[:non_snow_count]
    selected_snow = snow_records[:snow_count]

    if len(selected_non_snow) < non_snow_count:
        print(
            f"[WARN] non_snow candidates 부족: requested={non_snow_count}, "
            f"available={len(selected_non_snow)}"
        )

    if len(selected_snow) < snow_count:
        print(
            f"[WARN] snow candidates 부족: requested={snow_count}, "
            f"available={len(selected_snow)}"
        )

    selected_rows: list[dict[str, object]] = []
    for rank, record in enumerate(selected_non_snow, start=1):
        record.selected = True
        selected_rows.append(
            {
                "filename": record.filename,
                "environment": record.environment,
                "hard_positive_score": record.hard_positive_score,
                "selected_reason": "initial_top_by_score",
                "rank_in_group": rank,
                "stem": record.stem,
                "was_original_selected": True,
            }
        )

    for rank, record in enumerate(selected_snow, start=1):
        record.selected = True
        selected_rows.append(
            {
                "filename": record.filename,
                "environment": record.environment,
                "hard_positive_score": record.hard_positive_score,
                "selected_reason": "initial_top_by_score",
                "rank_in_group": rank,
                "stem": record.stem,
                "was_original_selected": True,
            }
        )

    return selected_rows


def records_to_score_rows(records: list[CandidateRecord]) -> list[dict[str, object]]:
    return [
        {
            "filename": record.filename,
            "environment": record.environment,
            "object_area_ratio": record.object_area_ratio,
            "fg_bg_color_distance": record.fg_bg_color_distance,
            "contrast_score": record.contrast_score,
            "blur_score": record.blur_score,
            "edge_complexity": record.edge_complexity,
            "hard_positive_score": record.hard_positive_score,
            "selected": record.selected,
            "stem": record.stem,
        }
        for record in records
    ]


def load_scores_csv(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Scores CSV not found: {csv_path}")

    rows: list[dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row.get("filename"):
                continue
            row["stem"] = normalize_stem(row["filename"])
            row["hard_positive_score"] = float(row["hard_positive_score"])
            row["selected"] = str(row.get("selected", "False")).lower() == "true"
            rows.append(row)

    return rows


def load_selected_csv(csv_path: Path) -> list[dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Selected CSV not found: {csv_path}")

    rows: list[dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row.get("filename"):
                continue
            row["stem"] = normalize_stem(row["filename"])
            row["hard_positive_score"] = float(row["hard_positive_score"])
            row["rank_in_group"] = int(row.get("rank_in_group", 0) or 0)
            row["selected_reason"] = row.get("selected_reason", "initial_top_by_score")
            row["was_original_selected"] = True
            rows.append(row)

    return rows


def mark_best_in_each_group(
    selected_rows: list[dict[str, object]], stem_to_group: dict[str, str]
) -> dict[str, str]:
    grouped_rows: dict[str, list[dict[str, object]]] = {}
    kept_reason_by_filename: dict[str, str] = {}

    for row in selected_rows:
        group_id = stem_to_group.get(str(row["stem"]))
        if group_id is None:
            continue
        grouped_rows.setdefault(group_id, []).append(row)

    for rows in grouped_rows.values():
        rows_sorted = sorted(
            rows,
            key=lambda item: (
                -float(item["hard_positive_score"]),
                0 if bool(item.get("was_original_selected", True)) else 1,
                str(item["filename"]).lower(),
            ),
        )
        kept_reason_by_filename[str(rows_sorted[0]["filename"])] = "kept_best_in_similar_group"

    return kept_reason_by_filename


def remove_similar_duplicates(
    selected_rows: list[dict[str, object]], stem_to_group: dict[str, str]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    kept_reason_by_filename = mark_best_in_each_group(selected_rows, stem_to_group)

    kept_rows: list[dict[str, object]] = []
    removed_rows: list[dict[str, object]] = []

    for row in selected_rows:
        row_copy = dict(row)
        filename = str(row_copy["filename"])
        stem = str(row_copy["stem"])
        group_id = stem_to_group.get(stem)

        if group_id is None:
            row_copy["selected_reason"] = "kept_original"
            kept_rows.append(row_copy)
            continue

        if filename in kept_reason_by_filename:
            row_copy["selected_reason"] = kept_reason_by_filename[filename]
            kept_rows.append(row_copy)
        else:
            row_copy["change_reason"] = "removed_duplicate_in_similar_group"
            removed_rows.append(row_copy)

    return kept_rows, removed_rows


def promote_replacements(
    score_rows: list[dict[str, object]],
    current_selected_rows: list[dict[str, object]],
    removed_rows: list[dict[str, object]],
    stem_to_group: dict[str, str],
) -> list[dict[str, object]]:
    final_selected_rows = [dict(row) for row in current_selected_rows]

    removed_non_snow = sum(row["environment"] == "non_snow" for row in removed_rows)
    removed_snow = sum(row["environment"] == "snow" for row in removed_rows)

    selected_stems = {str(row["stem"]) for row in final_selected_rows}
    occupied_group_ids = {
        stem_to_group[stem] for stem in selected_stems if stem in stem_to_group
    }

    for environment, needed_count in [("non_snow", removed_non_snow), ("snow", removed_snow)]:
        if needed_count <= 0:
            continue

        env_candidates = sorted(
            (row for row in score_rows if row["environment"] == environment),
            key=lambda item: (-float(item["hard_positive_score"]), str(item["filename"]).lower()),
        )

        promoted_count = 0
        for row in env_candidates:
            if promoted_count >= needed_count:
                break

            stem = str(row["stem"])
            group_id = stem_to_group.get(stem)
            if stem in selected_stems:
                continue
            if group_id is not None and group_id in occupied_group_ids:
                continue

            final_selected_rows.append(
                {
                    "filename": row["filename"],
                    "environment": row["environment"],
                    "hard_positive_score": float(row["hard_positive_score"]),
                    "selected_reason": "promoted_from_scores",
                    "rank_in_group": 0,
                    "stem": stem,
                    "was_original_selected": False,
                }
            )
            selected_stems.add(stem)
            if group_id is not None:
                occupied_group_ids.add(group_id)
            promoted_count += 1

        if promoted_count < needed_count:
            print(
                f"[WARN] Not enough promotable {environment} candidates: "
                f"needed={needed_count}, promoted={promoted_count}"
            )

    return final_selected_rows


def recompute_ranks(final_selected_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ranked_rows: list[dict[str, object]] = []

    for environment in ["non_snow", "snow"]:
        env_rows = sorted(
            (dict(row) for row in final_selected_rows if row["environment"] == environment),
            key=lambda item: (-float(item["hard_positive_score"]), str(item["filename"]).lower()),
        )
        for rank, row in enumerate(env_rows, start=1):
            row["rank_in_group"] = rank
            ranked_rows.append(row)

    return ranked_rows


def revise_selected_candidates(
    score_rows: list[dict[str, object]],
    selected_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    _, stem_to_group = build_similarity_groups()
    kept_rows, removed_rows = remove_similar_duplicates(selected_rows, stem_to_group)
    final_selected_rows = promote_replacements(
        score_rows=score_rows,
        current_selected_rows=kept_rows,
        removed_rows=removed_rows,
        stem_to_group=stem_to_group,
    )
    final_selected_rows = recompute_ranks(final_selected_rows)

    change_log_rows: list[dict[str, object]] = []
    for row in final_selected_rows:
        action = "promoted" if row["selected_reason"] == "promoted_from_scores" else "kept"
        change_log_rows.append(
            {
                "filename": row["filename"],
                "action": action,
                "reason": row["selected_reason"],
                "environment": row["environment"],
                "hard_positive_score": float(row["hard_positive_score"]),
            }
        )

    for row in removed_rows:
        change_log_rows.append(
            {
                "filename": row["filename"],
                "action": "removed",
                "reason": row["change_reason"],
                "environment": row["environment"],
                "hard_positive_score": float(row["hard_positive_score"]),
            }
        )

    change_log_rows.sort(
        key=lambda item: (
            str(item["action"]).lower(),
            str(item["environment"]).lower(),
            -float(item["hard_positive_score"]),
            str(item["filename"]).lower(),
        )
    )

    return final_selected_rows, change_log_rows


def move_selected_files(
    selected_rows: list[dict[str, object]],
    test_images_dir: Path,
    test_labels_dir: Path,
    reserve_images_dir: Path,
    reserve_labels_dir: Path,
    dry_run: bool,
) -> tuple[int, int, int]:
    ensure_dir(reserve_images_dir)
    ensure_dir(reserve_labels_dir)

    moved_count = 0
    missing_count = 0
    skipped_count = 0

    for row in selected_rows:
        stem = str(row["stem"])
        source_image_path = None
        for extension in VALID_IMAGE_EXTENSIONS:
            candidate = test_images_dir / f"{stem}{extension}"
            if candidate.exists():
                source_image_path = candidate
                break

        source_label_path = test_labels_dir / f"{stem}.txt"
        if source_image_path is None:
            print(f"[WARN] Test image not found for move: {stem}")
            missing_count += 1
            continue

        if not source_label_path.exists():
            print(f"[WARN] Test label not found for move: {stem}.txt")
            missing_count += 1
            continue

        destination_image_path = reserve_images_dir / source_image_path.name
        destination_label_path = reserve_labels_dir / source_label_path.name

        if destination_image_path.exists() or destination_label_path.exists():
            print(f"[WARN] Destination already exists, skipping move: {stem}")
            skipped_count += 1
            continue

        if dry_run:
            print(
                f"[DRY-RUN] Would move image: {source_image_path} -> {destination_image_path}"
            )
            print(
                f"[DRY-RUN] Would move label: {source_label_path} -> {destination_label_path}"
            )
            continue

        shutil.move(str(source_image_path), str(destination_image_path))
        shutil.move(str(source_label_path), str(destination_label_path))
        moved_count += 1

    return moved_count, missing_count, skipped_count


def save_scores_csv(score_rows: list[dict[str, object]], csv_path: Path) -> None:
    ensure_dir(csv_path.parent if str(csv_path.parent) not in ("", ".") else Path("."))

    fieldnames = [
        "filename",
        "environment",
        "object_area_ratio",
        "fg_bg_color_distance",
        "contrast_score",
        "blur_score",
        "edge_complexity",
        "hard_positive_score",
        "selected",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(score_rows, key=lambda item: str(item["filename"]).lower()):
            writer.writerow(
                {
                    "filename": row["filename"],
                    "environment": row["environment"],
                    "object_area_ratio": f"{float(row['object_area_ratio']):.8f}",
                    "fg_bg_color_distance": f"{float(row['fg_bg_color_distance']):.8f}",
                    "contrast_score": f"{float(row['contrast_score']):.8f}",
                    "blur_score": f"{float(row['blur_score']):.8f}",
                    "edge_complexity": f"{float(row['edge_complexity']):.8f}",
                    "hard_positive_score": f"{float(row['hard_positive_score']):.8f}",
                    "selected": bool(row["selected"]),
                }
            )


def save_selected_csv(selected_rows: list[dict[str, object]], csv_path: Path) -> None:
    ensure_dir(csv_path.parent if str(csv_path.parent) not in ("", ".") else Path("."))

    fieldnames = ["filename", "environment", "hard_positive_score", "rank_in_group"]
    ordered_rows = sorted(
        selected_rows,
        key=lambda item: (str(item["environment"]).lower(), int(item["rank_in_group"])),
    )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "environment": row["environment"],
                    "hard_positive_score": f"{float(row['hard_positive_score']):.8f}",
                    "rank_in_group": int(row["rank_in_group"]),
                }
            )


def save_revised_selected(final_selected_rows: list[dict[str, object]], csv_path: Path) -> None:
    ensure_dir(csv_path.parent if str(csv_path.parent) not in ("", ".") else Path("."))

    fieldnames = [
        "filename",
        "environment",
        "hard_positive_score",
        "selected_reason",
        "rank_in_group",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_selected_rows:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "environment": row["environment"],
                    "hard_positive_score": f"{float(row['hard_positive_score']):.8f}",
                    "selected_reason": row["selected_reason"],
                    "rank_in_group": int(row["rank_in_group"]),
                }
            )


def save_change_log(change_log_rows: list[dict[str, object]], csv_path: Path) -> None:
    ensure_dir(csv_path.parent if str(csv_path.parent) not in ("", ".") else Path("."))

    fieldnames = ["filename", "action", "reason", "environment", "hard_positive_score"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in change_log_rows:
            writer.writerow(
                {
                    "filename": row["filename"],
                    "action": row["action"],
                    "reason": row["reason"],
                    "environment": row["environment"],
                    "hard_positive_score": f"{float(row['hard_positive_score']):.8f}",
                }
            )


def summarize_selection(
    total_pairs: int,
    valid_records: int,
    initial_selected_rows: list[dict[str, object]],
    removed_count: int,
    promoted_count: int,
    final_selected_rows: list[dict[str, object]],
    moved_count: int,
    missing_count: int,
    skipped_count: int,
) -> None:
    final_non_snow = sum(row["environment"] == "non_snow" for row in final_selected_rows)
    final_snow = sum(row["environment"] == "snow" for row in final_selected_rows)

    print("\n[SUMMARY]")
    print(f"Collected pairs            : {total_pairs}")
    print(f"Valid scored records       : {valid_records}")
    print(f"Initial selected count     : {len(initial_selected_rows)}")
    print(f"Removed by similarity      : {removed_count}")
    print(f"Promoted replacements      : {promoted_count}")
    print(f"Final selected count       : {len(final_selected_rows)}")
    print(f"Final non_snow             : {final_non_snow}")
    print(f"Final snow                 : {final_snow}")
    print(f"Moved image/txt pairs      : {moved_count}")
    print(f"Missing move source pairs  : {missing_count}")
    print(f"Skipped existing dest pair : {skipped_count}")


def main() -> None:
    args = parse_args()

    if args.non_snow_count + args.snow_count != args.target_total:
        print(
            "[WARN] non_snow_count + snow_count does not match target_total. "
            "The script will use group counts as the actual selection target."
        )

    if args.revise_existing_only:
        score_rows = load_scores_csv(args.scores_csv)
        initial_selected_rows = load_selected_csv(args.selected_csv)
        total_pairs = 0
        valid_records = len(score_rows)
    else:
        pairs = collect_image_mask_pairs(args.testing_images_dir, args.testing_masks_dir)
        records = score_testing_pairs(pairs)
        if not records:
            print("[ERROR] No valid records were produced. Nothing to save or move.")
            return

        initial_selected_rows = select_candidates_by_ratio(
            records=records,
            non_snow_count=args.non_snow_count,
            snow_count=args.snow_count,
        )
        score_rows = records_to_score_rows(records)
        save_scores_csv(score_rows, args.scores_csv)
        save_selected_csv(initial_selected_rows, args.selected_csv)
        total_pairs = len(pairs)
        valid_records = len(records)

    final_selected_rows, change_log_rows = revise_selected_candidates(
        score_rows=score_rows,
        selected_rows=initial_selected_rows,
    )

    removed_count = sum(row["action"] == "removed" for row in change_log_rows)
    promoted_count = sum(row["action"] == "promoted" for row in change_log_rows)

    final_non_snow = sum(row["environment"] == "non_snow" for row in final_selected_rows)
    final_snow = sum(row["environment"] == "snow" for row in final_selected_rows)
    if final_non_snow != args.non_snow_count or final_snow != args.snow_count:
        print(
            f"[WARN] Final environment ratio mismatch: "
            f"non_snow={final_non_snow}/{args.non_snow_count}, "
            f"snow={final_snow}/{args.snow_count}"
        )

    save_revised_selected(final_selected_rows, args.revised_selected_csv)
    save_change_log(change_log_rows, args.change_log_csv)

    moved_count, missing_count, skipped_count = move_selected_files(
        selected_rows=final_selected_rows,
        test_images_dir=args.test_images_dir,
        test_labels_dir=args.test_labels_dir,
        reserve_images_dir=args.reserve_images_dir,
        reserve_labels_dir=args.reserve_labels_dir,
        dry_run=args.dry_run,
    )

    summarize_selection(
        total_pairs=total_pairs,
        valid_records=valid_records,
        initial_selected_rows=initial_selected_rows,
        removed_count=removed_count,
        promoted_count=promoted_count,
        final_selected_rows=final_selected_rows,
        moved_count=moved_count,
        missing_count=missing_count,
        skipped_count=skipped_count,
    )

    print(f"\n[INFO] Scores CSV          : {args.scores_csv}")
    print(f"[INFO] Initial selected    : {args.selected_csv}")
    print(f"[INFO] Revised selected    : {args.revised_selected_csv}")
    print(f"[INFO] Change log          : {args.change_log_csv}")
    print("\n[EXAMPLE]")
    print(r"python Select_Hard_Positive.py --dry-run")
    print(r"python Select_Hard_Positive.py --revise-existing-only --dry-run")
    print(r"python Select_Hard_Positive.py")


if __name__ == "__main__":
    main()
