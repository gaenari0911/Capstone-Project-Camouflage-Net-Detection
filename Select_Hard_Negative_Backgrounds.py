"""
Heuristic hard-negative background selector.

목적:
1. FULL background root 아래의 도메인별 배경 이미지를 수집합니다.
2. "말하지 않으면 위장망으로 헷갈릴 수 있는" 배경을 휴리스틱 점수로 평가합니다.
3. 상위 30장을 domain 비율을 크게 깨지 않도록 선별합니다.
4. 선별된 이미지를 Dataset/reserve/hard_negative_pool 로 이동합니다.
5. 대응하는 빈 txt 라벨을 Dataset/reserve_labels/hard_negative_pool 에 생성합니다.

핵심 철학:
- 실제 위장 객체가 없는 배경만 대상으로 삼습니다.
- 점수는 low saturation + dense texture + edge complexity + local self-similarity를 함께 봅니다.
- 완전 동일하거나 매우 비슷한 배경이 몰리지 않도록 간단한 시그니처 기반 중복 억제를 수행합니다.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"}


@dataclass
class BackgroundRecord:
    domain: str
    image_path: Path
    stem: str
    mean_saturation: float
    edge_density: float
    local_contrast: float
    texture_entropy: float
    self_similarity: float
    clutter_score: float
    hard_negative_score: float = 0.0
    signature: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select camouflage-like hard negative backgrounds.")
    parser.add_argument(
        "--background-root",
        type=Path,
        default=Path(r"G:\내 드라이브\데이터셋"),
        help="Root directory that contains domain folders such as snow / forest_dense / grass_field.",
    )
    parser.add_argument(
        "--reserve-images-dir",
        type=Path,
        default=Path("Dataset") / "reserve" / "hard_negative_pool",
        help="Destination directory for selected hard-negative background images.",
    )
    parser.add_argument(
        "--reserve-labels-dir",
        type=Path,
        default=Path("Dataset") / "reserve_labels" / "hard_negative_pool",
        help="Destination directory for empty txt labels paired with hard-negative backgrounds.",
    )
    parser.add_argument(
        "--scores-csv",
        type=Path,
        default=Path("hard_negative_background_scores.csv"),
        help="CSV output for all scored candidate backgrounds.",
    )
    parser.add_argument(
        "--selected-csv",
        type=Path,
        default=Path("hard_negative_background_selected.csv"),
        help="CSV output for final selected backgrounds.",
    )
    parser.add_argument("--target-total", type=int, default=30, help="How many hard negatives to select in total.")
    parser.add_argument("--dry-run", action="store_true", help="Score and print planned moves without changing files.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]


def collect_domain_directories(background_root: Path) -> list[Path]:
    if not background_root.exists():
        return []
    return [path for path in sorted(background_root.iterdir()) if path.is_dir()]


def load_image(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def compute_mean_saturation(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


def compute_edge_density(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    return float(np.mean((grad > 28.0).astype(np.float32)))


def compute_local_contrast(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=7.0)
    contrast_map = np.abs(gray - blurred)
    return float(np.mean(contrast_map))


def compute_texture_entropy(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten().astype(np.float32)
    hist /= max(float(np.sum(hist)), 1e-6)
    entropy = -np.sum(hist * np.log2(np.clip(hist, 1e-8, 1.0)))
    return float(entropy)


def compute_patch_self_similarity(image: np.ndarray) -> float:
    """
    배경 내부 패치끼리 색/밝기 통계가 비슷한지를 보며,
    반복적이고 위장망 같은 질감이면 점수를 조금 올립니다.
    """
    h, w = image.shape[:2]
    grid_rows = 3
    grid_cols = 3
    patch_vectors: list[np.ndarray] = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = int(round(h * row / grid_rows))
            y1 = int(round(h * (row + 1) / grid_rows))
            x0 = int(round(w * col / grid_cols))
            x1 = int(round(w * (col + 1) / grid_cols))
            patch = image[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.float32)
            vec = np.array(
                [
                    float(np.mean(hsv[:, :, 0])),
                    float(np.mean(hsv[:, :, 1])),
                    float(np.mean(hsv[:, :, 2])),
                    float(np.std(hsv[:, :, 2])),
                ],
                dtype=np.float32,
            )
            patch_vectors.append(vec)

    if len(patch_vectors) < 2:
        return 0.0

    distances = []
    for idx in range(len(patch_vectors)):
        for jdx in range(idx + 1, len(patch_vectors)):
            distances.append(float(np.linalg.norm(patch_vectors[idx] - patch_vectors[jdx])))

    mean_distance = float(np.mean(distances))
    similarity = 1.0 - min(1.0, mean_distance / 55.0)
    return float(np.clip(similarity, 0.0, 1.0))


def compute_clutter_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0.0
    component_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    medium_components = np.mean((component_areas >= 6.0) & (component_areas <= 180.0))
    edge_fill = float(np.mean(edges > 0))
    clutter = 0.65 * edge_fill + 0.35 * float(medium_components)
    return float(np.clip(clutter, 0.0, 1.0))


def build_signature(image: np.ndarray) -> str:
    thumb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA)
    mean_value = float(np.mean(thumb))
    bits = (thumb > mean_value).astype(np.uint8).flatten()
    return "".join(str(int(bit)) for bit in bits[:128])


def hamming_distance(sig_a: str, sig_b: str) -> int:
    length = min(len(sig_a), len(sig_b))
    distance = 0
    for idx in range(length):
        if sig_a[idx] != sig_b[idx]:
            distance += 1
    distance += abs(len(sig_a) - len(sig_b))
    return distance


def score_record(record: BackgroundRecord) -> float:
    """
    위장망처럼 착시될 가능성이 있는 배경에 높은 점수를 주는 휴리스틱.

    직관:
    - 채도가 너무 높으면 화려해서 위장 느낌이 약해집니다.
    - 엣지가 적당히 많고 질감 엔트로피가 높으면 복잡한 camouflage texture에 가깝습니다.
    - local self-similarity가 있으면 망사/겹침 패턴처럼 보일 가능성이 있습니다.
    """
    low_saturation_score = 1.0 - min(1.0, record.mean_saturation / 120.0)
    edge_score = min(1.0, record.edge_density / 0.22)
    contrast_score = min(1.0, record.local_contrast / 32.0)
    entropy_score = min(1.0, record.texture_entropy / 5.4)
    similarity_score = record.self_similarity
    clutter_score = record.clutter_score

    total = (
        0.22 * low_saturation_score
        + 0.22 * edge_score
        + 0.15 * contrast_score
        + 0.18 * entropy_score
        + 0.13 * similarity_score
        + 0.10 * clutter_score
    )
    return float(total)


def collect_records(background_root: Path) -> tuple[list[BackgroundRecord], dict[str, int]]:
    records: list[BackgroundRecord] = []
    domain_counts: dict[str, int] = {}
    domain_dirs = collect_domain_directories(background_root)

    if not domain_dirs:
        print(f"[ERROR] No first-level directories found under background root: {background_root}")
        return records, domain_counts

    for domain_dir in domain_dirs:
        domain = domain_dir.name
        image_files = collect_image_files(domain_dir)
        domain_counts[domain] = len(image_files)
        for image_path in image_files:
            image = load_image(image_path)
            if image is None:
                print(f"[WARN] Failed to read image: {image_path}")
                continue

            record = BackgroundRecord(
                domain=domain,
                image_path=image_path,
                stem=image_path.stem.lower(),
                mean_saturation=compute_mean_saturation(image),
                edge_density=compute_edge_density(image),
                local_contrast=compute_local_contrast(image),
                texture_entropy=compute_texture_entropy(image),
                self_similarity=compute_patch_self_similarity(image),
                clutter_score=compute_clutter_score(image),
            )
            record.hard_negative_score = score_record(record)
            record.signature = build_signature(image)
            records.append(record)
    return records, domain_counts


def allocate_domain_targets(total_count: int, domain_counts: dict[str, int]) -> dict[str, int]:
    total_weight = float(sum(domain_counts.values()))
    raw = {}
    allocated = {}
    assigned = 0
    remainder = []
    for domain, count in domain_counts.items():
        value = total_count * (float(count) / total_weight)
        floor_value = int(np.floor(value))
        raw[domain] = value
        allocated[domain] = floor_value
        assigned += floor_value
        remainder.append((value - floor_value, domain))

    remainder.sort(reverse=True)
    index = 0
    while assigned < total_count and remainder:
        _, domain = remainder[index % len(remainder)]
        allocated[domain] += 1
        assigned += 1
        index += 1

    return allocated


def select_records(records: list[BackgroundRecord], domain_counts: dict[str, int], target_total: int) -> list[BackgroundRecord]:
    by_domain: dict[str, list[BackgroundRecord]] = {domain: [] for domain in domain_counts.keys()}
    for record in records:
        by_domain[record.domain].append(record)

    for domain in by_domain:
        by_domain[domain].sort(key=lambda item: (-item.hard_negative_score, item.image_path.name.lower()))

    domain_targets = allocate_domain_targets(target_total, domain_counts)
    selected: list[BackgroundRecord] = []
    used_signatures: list[str] = []

    for domain, target_count in domain_targets.items():
        for record in by_domain.get(domain, []):
            if len([item for item in selected if item.domain == domain]) >= target_count:
                break
            if any(hamming_distance(record.signature, prev_sig) <= 10 for prev_sig in used_signatures):
                continue
            selected.append(record)
            used_signatures.append(record.signature)

    if len(selected) < target_total:
        remaining = sorted(records, key=lambda item: (-item.hard_negative_score, item.image_path.name.lower()))
        for record in remaining:
            if record in selected:
                continue
            if any(hamming_distance(record.signature, prev_sig) <= 8 for prev_sig in used_signatures):
                continue
            selected.append(record)
            used_signatures.append(record.signature)
            if len(selected) >= target_total:
                break

    return selected[:target_total]


def save_scores_csv(records: list[BackgroundRecord], csv_path: Path) -> None:
    rows = []
    for record in records:
        rows.append(
            {
                "domain": record.domain,
                "filename": record.image_path.name,
                "image_path": str(record.image_path),
                "mean_saturation": f"{record.mean_saturation:.6f}",
                "edge_density": f"{record.edge_density:.6f}",
                "local_contrast": f"{record.local_contrast:.6f}",
                "texture_entropy": f"{record.texture_entropy:.6f}",
                "self_similarity": f"{record.self_similarity:.6f}",
                "clutter_score": f"{record.clutter_score:.6f}",
                "hard_negative_score": f"{record.hard_negative_score:.6f}",
            }
        )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["domain", "filename"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def save_selected_csv(records: list[BackgroundRecord], csv_path: Path) -> None:
    rows = []
    for rank, record in enumerate(records, start=1):
        rows.append(
            {
                "rank": rank,
                "domain": record.domain,
                "filename": record.image_path.name,
                "image_path": str(record.image_path),
                "hard_negative_score": f"{record.hard_negative_score:.6f}",
            }
        )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["rank", "domain", "filename"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def move_selected_records(
    selected: list[BackgroundRecord],
    reserve_images_dir: Path,
    reserve_labels_dir: Path,
    dry_run: bool,
) -> None:
    ensure_dir(reserve_images_dir)
    ensure_dir(reserve_labels_dir)

    for record in selected:
        dst_image = reserve_images_dir / record.image_path.name
        dst_label = reserve_labels_dir / f"{record.image_path.stem}.txt"

        if dry_run:
            print(f"[DRY-RUN][MOVE] {record.image_path} -> {dst_image}")
            print(f"[DRY-RUN][LABEL] {dst_label} (empty txt)")
            continue

        shutil.move(str(record.image_path), str(dst_image))
        dst_label.write_text("", encoding="utf-8")


def print_summary(selected: list[BackgroundRecord]) -> None:
    counts: dict[str, int] = {}
    for record in selected:
        counts[record.domain] = counts.get(record.domain, 0) + 1

    print("\n[SUMMARY] Selected hard-negative backgrounds")
    print(f"  total: {len(selected)}")
    for domain in sorted(counts.keys()):
        print(f"  {domain}: {counts.get(domain, 0)}")


def main() -> None:
    args = parse_args()
    records, domain_counts = collect_records(args.background_root)
    if not records:
        print("[ERROR] No background candidates were collected.")
        return

    selected = select_records(records, domain_counts, args.target_total)
    save_scores_csv(records, args.scores_csv)
    save_selected_csv(selected, args.selected_csv)
    move_selected_records(
        selected,
        args.reserve_images_dir,
        args.reserve_labels_dir,
        args.dry_run,
    )
    print_summary(selected)
    print(f"[INFO] Scores CSV   : {args.scores_csv}")
    print(f"[INFO] Selected CSV : {args.selected_csv}")
    print(f"[INFO] Reserve imgs : {args.reserve_images_dir}")
    print(f"[INFO] Reserve txts : {args.reserve_labels_dir}")


if __name__ == "__main__":
    main()
