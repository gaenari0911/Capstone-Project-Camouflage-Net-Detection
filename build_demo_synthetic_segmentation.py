"""
Synthetic segmentation generator used to build demo/full synthetic data.

읽는 순서:
1. 전역 상수에서 경로, mode, scale bucket, quota 설정을 먼저 봅니다.
2. helper 함수들이 배경/객체를 로드하고 crop/resize 하는 역할을 맡습니다.
3. 파이프라인은 scale을 먼저 고정한 뒤, object 후보를 평가하고,
   각 object마다 여러 위치 후보를 다시 평가합니다.
4. 점수는 tone, texture, edge compatibility, placement, usage penalty를 함께 반영해
   특정 객체/배경 조합이 과도하게 반복되지 않도록 합니다.
5. 최종 선택 결과를 배경에 합성하고, sharp binary GT mask와 YOLO polygon txt를 저장합니다.
"""

from pathlib import Path
import random
import cv2
import numpy as np


OBJECT_POOL_ROOT = Path("Dataset") / "object_pool"
BACKGROUND_ROOT = Path(r"G:\내 드라이브\데이터셋")
OUTPUT_IMAGE_ROOT = Path(r"G:\내 드라이브\합성_데이터셋\images")
OUTPUT_MASK_ROOT = Path(r"G:\내 드라이브\합성_데이터셋\masks")
OUTPUT_TXT_ROOT = Path(r"G:\내 드라이브\합성_데이터셋\txt")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
TARGET_BACKGROUND_LONG_SIDE = 1280
MASK_THRESHOLD = 127
OUTER_MARGIN_RATIO = 0.05
MIN_OBJECT_PIXEL_SIZE = 20
MAX_OBJECT_CANDIDATES = 8
MAX_POSITION_CANDIDATES = 16
RANDOM_SEED = 7
EXECUTION_MODE = "DEMO"  # "DEMO" or "FULL"
SAVE_MASKS = True
CLASS_ID = 0
MAX_OBJECT_USAGE = 5
RECENT_HISTORY_SIZE = 4
MAX_BACKGROUND_REPEAT_PER_OBJECT = 1
DOMAIN_PENALTY_WEIGHT = 0.08
BACKGROUND_PENALTY_WEIGHT = 0.18
RECENT_PENALTY_WEIGHT = 0.10
QUOTA_BONUS_WEIGHT = 0.06

# 이 프로젝트는 domain 구조가 비대칭입니다.
# - snow 객체는 자연스럽게 연결되는 coarse domain이 1개뿐이고
# - non-snow 객체는 여러 background domain으로 퍼질 수 있습니다.
# 아래 quota/penalty 로직은 이 비대칭을 조금 더 공정하게 다루기 위한 장치입니다.

ALLOWED_DOMAINS = {
    "snow": ["snow"],
    "non_snow": ["forest", "grass_field", "leaf_ground", "rocky", "mixed"],
}

MODE_CONFIG = {
    "natural": {
        "brightness_alpha": (0.20, 0.35),
        "contrast_alpha": (0.20, 0.35),
        "saturation_alpha": (0.05, 0.12),
        "feather_kernel": (3, 7),
        "weights": {"tone": 0.35, "texture": 0.20, "edge": 0.30, "placement": 0.15},
        "top_k": 1,
        "strictness": "strict",
        "domain_same_ratio": 1.00,
    },
    "semi": {
        "brightness_alpha": (0.08, 0.20),
        "contrast_alpha": (0.08, 0.18),
        "saturation_alpha": (0.03, 0.10),
        "feather_kernel": (0, 5),
        "weights": {"tone": 0.30, "texture": 0.25, "edge": 0.25, "placement": 0.20},
        "top_k": 3,
        "strictness": "medium",
        "domain_same_ratio": 0.85,
    },
    "hard": {
        "brightness_alpha": (0.00, 0.10),
        "contrast_alpha": (0.00, 0.10),
        "saturation_alpha": (0.00, 0.05),
        "feather_kernel": (0, 0),
        "weights": {"tone": 0.20, "texture": 0.25, "edge": 0.20, "placement": 0.35},
        "top_k": 5,
        "strictness": "loose",
        "domain_same_ratio": 0.70,
    },
    "minimal": {
        "brightness_alpha": (0.0, 0.0),
        "contrast_alpha": (0.0, 0.0),
        "saturation_alpha": (0.0, 0.0),
        "feather_kernel": (0, 0),
        "weights": {"tone": 0.15, "texture": 0.20, "edge": 0.15, "placement": 0.50},
        "top_k": 7,
        "strictness": "minimal",
        "domain_same_ratio": 0.50,
    },
}

SCALE_DISTRIBUTION = [
    {"name": "00to05", "min_ratio": 0.00, "max_ratio": 0.05, "weight": 10},
    {"name": "05to10", "min_ratio": 0.05, "max_ratio": 0.10, "weight": 20},
    {"name": "10to15", "min_ratio": 0.10, "max_ratio": 0.15, "weight": 15},
    {"name": "15to20", "min_ratio": 0.15, "max_ratio": 0.20, "weight": 15},
    {"name": "20to25", "min_ratio": 0.20, "max_ratio": 0.25, "weight": 15},
    {"name": "25to30", "min_ratio": 0.25, "max_ratio": 0.30, "weight": 10},
    {"name": "30to40", "min_ratio": 0.30, "max_ratio": 0.40, "weight": 10},
    {"name": "40to60", "min_ratio": 0.40, "max_ratio": 0.60, "weight": 5},
]

DOMAIN_TARGETS = {
    "snow": 600,
    "forest_dense": 600,
    "grass_field": 600,
    "leaf_ground": 450,
    "rocky": 300,
    "mixed": 450,
}

MODE_TARGETS = {
    "natural": {"snow": 240, "forest_dense": 240, "grass_field": 240, "leaf_ground": 180, "rocky": 120, "mixed": 180},
    "semi": {"snow": 180, "forest_dense": 180, "grass_field": 180, "leaf_ground": 135, "rocky": 90, "mixed": 135},
    "hard": {"snow": 120, "forest_dense": 120, "grass_field": 120, "leaf_ground": 90, "rocky": 60, "mixed": 90},
    "minimal": {"snow": 60, "forest_dense": 60, "grass_field": 60, "leaf_ground": 45, "rocky": 30, "mixed": 45},
}

DEMO_MODE_TARGETS = {
    "natural": {"forest_dense": 8, "grass_field": 6, "mixed": 6},
    "semi": {"forest_dense": 5, "grass_field": 5, "mixed": 5},
    "hard": {"forest_dense": 5, "grass_field": 5, "mixed": 5},
    "minimal": {"forest_dense": 4, "grass_field": 3, "mixed": 3},
}

DEMO_BACKGROUND_DIR_MAP = {
    "forest_dense": BACKGROUND_ROOT / "forest_dense",
    "grass_field": BACKGROUND_ROOT / "grass_field",
    "leaf_ground": BACKGROUND_ROOT / "leaf_ground",
}

FULL_BACKGROUND_DIR_MAP = {
    "snow": BACKGROUND_ROOT / "snow",
    "forest_dense": BACKGROUND_ROOT / "forest_dense",
    "grass_field": BACKGROUND_ROOT / "grass_field",
    "leaf_ground": BACKGROUND_ROOT / "leaf_ground",
    "rocky": BACKGROUND_ROOT / "rocky",
    "mixed": BACKGROUND_ROOT / "mixed",
}

GLOBAL_STATE = {
    "rng": random.Random(RANDOM_SEED),
    "background_state": {},
    "background_rotation": {},
    "background_usage": {},
    "usage_state": {},
    "object_pools": {},
    "recent_objects": [],
    "scale_plan": [],
    "scale_hist": {},
    "saved_images": [],
    "saved_masks": [],
    "saved_txts": [],
}


def _get_object_domain_for_target_domain(domain):
    return "snow" if domain == "snow" else "non_snow"


def _get_allowed_domains(object_domain):
    return ALLOWED_DOMAINS.get(object_domain, [])


def _coarse_domain_penalty(object_domain, target_domain, mode):
    allowed_domains = _get_allowed_domains(object_domain)
    if target_domain in allowed_domains:
        return 0.0
    if mode in ("hard", "minimal"):
        return 0.30
    return 1.0


def _build_object_quota(total_needed, object_ids, object_domain):
    """
    각 객체가 최대 몇 번까지 등장할 수 있는지 미리 계산합니다.

    전략:
    - 가능하면 모든 객체에 최소 1번씩 먼저 기회를 줍니다.
    - 남은 횟수는 round-robin으로 분배합니다.
    - 기본 최대 사용 횟수는 5이고, 꼭 필요할 때만 완화합니다.
    """
    quotas = {}
    shuffled_ids = list(object_ids)
    GLOBAL_STATE["rng"].shuffle(shuffled_ids)
    for object_id in shuffled_ids:
        quotas[object_id] = 0

    max_usage = MAX_OBJECT_USAGE
    capacity = len(shuffled_ids) * max_usage
    while total_needed > capacity and max_usage < 7:
        new_limit = max_usage + 1
        print(f"[INFO] Usage limit relaxed for {object_domain}: {max_usage} -> {new_limit}")
        max_usage = new_limit
        capacity = len(shuffled_ids) * max_usage

    if total_needed <= 0 or not shuffled_ids:
        return quotas, max_usage

    if total_needed < len(shuffled_ids):
        print(
            f"[WARN] Cannot use every {object_domain} object at least once: "
            f"targets={total_needed}, objects={len(shuffled_ids)}"
        )
        for object_id in shuffled_ids[:total_needed]:
            quotas[object_id] = 1
        return quotas, max_usage

    for object_id in shuffled_ids:
        quotas[object_id] = 1

    remaining = total_needed - len(shuffled_ids)
    round_robin_ids = list(shuffled_ids)
    index = 0
    while remaining > 0 and round_robin_ids:
        object_id = round_robin_ids[index % len(round_robin_ids)]
        if quotas[object_id] < max_usage:
            quotas[object_id] += 1
            remaining -= 1
        index += 1
        if index > len(round_robin_ids) * max_usage * 2:
            break

    if remaining > 0:
        print(f"[WARN] Quota allocation incomplete for {object_domain}: remaining={remaining}")

    return quotas, max_usage


def _initialize_usage_state(schedule):
    """합성 루프가 시작되기 전에 quota/usage 테이블을 준비합니다."""
    targets_by_object_domain = {"snow": 0, "non_snow": 0}
    for item in schedule:
        object_domain = _get_object_domain_for_target_domain(item["domain"])
        targets_by_object_domain[object_domain] += 1

    for object_domain, pool in GLOBAL_STATE["object_pools"].items():
        object_ids = [item["id"] for item in pool]
        quotas, max_usage = _build_object_quota(
            targets_by_object_domain.get(object_domain, 0),
            object_ids,
            object_domain,
        )
        used_total = {}
        used_by_domain = {}
        used_by_background = {}
        for object_id in object_ids:
            used_total[object_id] = 0
            used_by_domain[object_id] = {}
            for allowed_domain in _get_allowed_domains(object_domain):
                used_by_domain[object_id][allowed_domain] = 0
            used_by_background[object_id] = {}

        GLOBAL_STATE["usage_state"][object_domain] = {
            "remaining_quota": quotas,
            "used_total": used_total,
            "used_by_domain": used_by_domain,
            "used_by_background": used_by_background,
            "max_usage": max_usage,
            "allowed_domains": list(_get_allowed_domains(object_domain)),
        }


def _compute_usage_guidance(object_id, object_domain, target_domain, background_key, mode):
    state = GLOBAL_STATE["usage_state"][object_domain]
    allowed_domains = state["allowed_domains"]
    allowed_count = max(1, len(allowed_domains))
    remaining_quota = state["remaining_quota"].get(object_id, 0)
    used_total = state["used_total"].get(object_id, 0)
    used_by_domain = state["used_by_domain"].get(object_id, {}).get(target_domain, 0)
    used_by_background = state["used_by_background"].get(object_id, {}).get(background_key, 0)

    domain_repeat_penalty = 0.0 if object_domain == "snow" else used_by_domain / float(allowed_count)
    background_repeat_penalty = float(used_by_background)
    recent_history_penalty = 1.0 if object_id in set(GLOBAL_STATE["recent_objects"]) else 0.0
    coarse_domain_penalty = _coarse_domain_penalty(object_domain, target_domain, mode)
    quota_bonus = remaining_quota / float(max(1, state["max_usage"]))
    diversity_bonus = 1.0 / float(1 + used_total)

    return {
        "remaining_quota": remaining_quota,
        "domain_repeat_penalty": domain_repeat_penalty,
        "background_repeat_penalty": background_repeat_penalty,
        "recent_history_penalty": recent_history_penalty,
        "coarse_domain_penalty": coarse_domain_penalty,
        "quota_bonus": quota_bonus,
        "diversity_bonus": diversity_bonus,
    }


def collect_image_files(directory):
    if directory is None or not Path(directory).exists():
        return []
    files = []
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            files.append(path)
    return files


def build_object_pool(image_dir, mask_dir):
    image_files = collect_image_files(image_dir)
    mask_lookup = {}
    for mask_path in collect_image_files(mask_dir):
        mask_lookup[mask_path.stem] = mask_path

    pool = []
    for image_path in image_files:
        mask_path = mask_lookup.get(image_path.stem)
        if mask_path is None:
            continue
        pool.append(
            {
                "id": image_path.stem,
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )
    return pool


def load_image(path, flags):
    if path is None:
        return None
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, flags)
    return image


def resize_background_long_side(image, target_long_side):
    if image is None:
        return None
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= 0:
        return None
    scale = float(target_long_side) / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def binarize_mask(mask):
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def crop_to_mask_bbox(image, mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None, None, None
    x, y, w, h = cv2.boundingRect(coords)
    if w < 1 or h < 1:
        return None, None, None
    cropped_image = image[y:y + h, x:x + w].copy()
    cropped_mask = mask[y:y + h, x:x + w].copy()
    return cropped_image, cropped_mask, (x, y, w, h)


def sample_scale_bucket(mode):
    if GLOBAL_STATE["scale_plan"]:
        return GLOBAL_STATE["scale_plan"].pop(0)
    weights = [item["weight"] for item in SCALE_DISTRIBUTION]
    return GLOBAL_STATE["rng"].choices(SCALE_DISTRIBUTION, weights=weights, k=1)[0]


def compute_target_area_ratio(scale_bucket):
    low = scale_bucket["min_ratio"]
    high = scale_bucket["max_ratio"]
    return GLOBAL_STATE["rng"].uniform(low, high)


def resize_object_to_target_scale(image, mask, bg_h, bg_w, target_ratio):
    bbox_area = float(mask.shape[0] * mask.shape[1])
    bg_area = float(bg_h * bg_w)
    if bbox_area <= 0 or bg_area <= 0 or target_ratio <= 0:
        return None, None

    desired_bbox_area = max(1.0, bg_area * target_ratio)
    resize_scale = np.sqrt(desired_bbox_area / bbox_area)
    new_w = max(MIN_OBJECT_PIXEL_SIZE, int(round(image.shape[1] * resize_scale)))
    new_h = max(MIN_OBJECT_PIXEL_SIZE, int(round(image.shape[0] * resize_scale)))
    if new_w >= bg_w or new_h >= bg_h:
        fit_scale = min((bg_w - 2) / float(max(1, image.shape[1])), (bg_h - 2) / float(max(1, image.shape[0])))
        if fit_scale <= 0:
            return None, None
        new_w = max(MIN_OBJECT_PIXEL_SIZE, int(round(image.shape[1] * fit_scale)))
        new_h = max(MIN_OBJECT_PIXEL_SIZE, int(round(image.shape[0] * fit_scale)))

    if new_w < MIN_OBJECT_PIXEL_SIZE or new_h < MIN_OBJECT_PIXEL_SIZE:
        return None, None

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    resized_mask = binarize_mask(resized_mask)
    return resized_image, resized_mask


def choose_target_schedule():
    schedule = []
    if EXECUTION_MODE == "DEMO":
        targets = DEMO_MODE_TARGETS
    else:
        targets = MODE_TARGETS

    for mode_name in ["natural", "semi", "hard", "minimal"]:
        mode_targets = targets.get(mode_name, {})
        for domain_name in sorted(mode_targets.keys()):
            count = mode_targets[domain_name]
            for _ in range(count):
                schedule.append({"mode": mode_name, "domain": domain_name})

    GLOBAL_STATE["rng"].shuffle(schedule)
    return schedule


def sample_background_for_domain(domain, mode):
    """
    요청된 domain에 맞는 배경 이미지 하나를 선택합니다.

    Natural/Semi에서는 사용 횟수가 적은 배경을 조금 더 우선해서
    몇 장의 배경만 과도하게 반복되는 현상을 줄입니다.
    """
    state = GLOBAL_STATE["background_state"].get(domain, {})
    paths = state.get("paths", [])
    if not paths:
        return None

    rotation = GLOBAL_STATE["background_rotation"].setdefault(domain, list(range(len(paths))))
    usage = GLOBAL_STATE["background_usage"].setdefault(domain, {})
    if not rotation:
        rotation.extend(list(range(len(paths))))
        GLOBAL_STATE["rng"].shuffle(rotation)

    if mode in ("natural", "semi") and GLOBAL_STATE["rng"].random() < MODE_CONFIG[mode]["domain_same_ratio"]:
        rotation.sort(key=lambda idx: usage.get(str(paths[idx]), 0))

    selected_index = rotation.pop(0)
    selected_path = paths[selected_index]
    usage[str(selected_path)] = usage.get(str(selected_path), 0) + 1
    return selected_path


def sample_object_candidates(domain, background_path, k, usage_state, mode):
    """
    비용이 큰 위치 평가 전에 object 후보를 먼저 걸러내고 정렬합니다.

    이 단계에서 이미 quota, domain 반복, background 반복, recent history를 반영해서
    과도하게 많이 쓰인 객체가 초반부터 밀려나도록 합니다.
    """
    object_domain = _get_object_domain_for_target_domain(domain)
    pool = GLOBAL_STATE["object_pools"].get(object_domain, [])
    if not pool:
        return []

    state = usage_state.get(object_domain, {})
    background_key = background_path.stem if background_path is not None else "unknown_bg"
    ordered = []
    for item in pool:
        object_id = item["id"]
        guidance = _compute_usage_guidance(object_id, object_domain, domain, background_key, mode)
        if guidance["remaining_quota"] <= 0:
            continue
        if guidance["coarse_domain_penalty"] >= 1.0:
            continue
        if guidance["background_repeat_penalty"] >= MAX_BACKGROUND_REPEAT_PER_OBJECT:
            continue

        priority_score = (
            3.0 * guidance["quota_bonus"]
            + 1.2 * guidance["diversity_bonus"]
            + 1.0 * (1.0 - guidance["recent_history_penalty"])
            - 1.3 * guidance["domain_repeat_penalty"]
            - 2.2 * guidance["background_repeat_penalty"]
            - 3.0 * guidance["coarse_domain_penalty"]
            + GLOBAL_STATE["rng"].random() * 0.01
        )
        ordered.append((priority_score, item))

    ordered.sort(key=lambda x: x[0], reverse=True)
    candidates = []
    for _, item in ordered:
        candidates.append(item)
        if len(candidates) >= k:
            break

    if len(candidates) < k and ordered:
        relaxed_candidates = []
        for item in pool:
            object_id = item["id"]
            guidance = _compute_usage_guidance(object_id, object_domain, domain, background_key, mode)
            if guidance["remaining_quota"] <= 0:
                continue
            if guidance["coarse_domain_penalty"] >= 1.0:
                continue
            priority_score = (
                2.5 * guidance["quota_bonus"]
                + 1.0 * guidance["diversity_bonus"]
                - 1.0 * guidance["domain_repeat_penalty"]
                - 1.5 * guidance["background_repeat_penalty"]
                - 1.5 * guidance["coarse_domain_penalty"]
                - 0.5 * guidance["recent_history_penalty"]
                + GLOBAL_STATE["rng"].random() * 0.01
            )
            relaxed_candidates.append((priority_score, item))

        relaxed_candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = []
        for _, item in relaxed_candidates:
            if item not in candidates:
                candidates.append(item)
            if len(candidates) >= k:
                break

    return candidates


def sample_candidate_positions(bg_h, bg_w, obj_h, obj_w, num_candidates):
    positions = []
    if obj_h >= bg_h or obj_w >= bg_w:
        return positions

    margin_x = int(round(bg_w * OUTER_MARGIN_RATIO))
    margin_y = int(round(bg_h * OUTER_MARGIN_RATIO))
    max_x = max(0, bg_w - obj_w)
    max_y = max(0, bg_h - obj_h)
    for _ in range(num_candidates):
        x = GLOBAL_STATE["rng"].randint(0, max_x)
        y = GLOBAL_STATE["rng"].randint(0, max_y)
        positions.append(
            {
                "x": x,
                "y": y,
                "center_x_ratio": (x + obj_w * 0.5) / float(bg_w),
                "center_y_ratio": (y + obj_h * 0.5) / float(bg_h),
                "touches_margin": x < margin_x or y < margin_y or x + obj_w > bg_w - margin_x or y + obj_h > bg_h - margin_y,
            }
        )
    return positions


def extract_background_patch(background, x, y, obj_w, obj_h):
    bg_h, bg_w = background.shape[:2]
    if x < 0 or y < 0 or x + obj_w > bg_w or y + obj_h > bg_h:
        return None
    return background[y:y + obj_h, x:x + obj_w].copy()


def compute_patch_texture_score(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_mean = float(np.mean(grad))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    edge_density = float(np.mean((grad > 24.0).astype(np.float32)))
    return {"grad_mean": grad_mean, "lap_var": lap_var, "edge_density": edge_density}


def _masked_hsv_stats(image_bgr, mask):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask_bool = mask > 0
    if np.count_nonzero(mask_bool) == 0:
        return {"brightness": 0.0, "contrast": 0.0, "saturation": 0.0}
    v = hsv[:, :, 2][mask_bool]
    s = hsv[:, :, 1][mask_bool]
    return {
        "brightness": float(np.mean(v)),
        "contrast": float(np.std(v)),
        "saturation": float(np.mean(s)),
    }


def _build_boundary_band(mask, band_width=5):
    kernel = np.ones((band_width, band_width), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    band = cv2.subtract(dilated, eroded)
    return binarize_mask(band)


def _score_from_difference(diff_value, scale):
    return max(0.0, 1.0 - (diff_value / float(max(scale, 1e-6))))


def compute_object_background_match_score(object_bgr, object_mask, patch_bgr, y_ratio, mode):
    """
    resize된 object 하나가 background patch 하나에 얼마나 자연스럽게 붙는지 점수화합니다.

    점수는 여러 축을 함께 봅니다.
    - tone 유사도
    - texture 유사도
    - edge / boundary 접합감
    - placement 선호도
    """
    object_stats = _masked_hsv_stats(object_bgr, object_mask)
    bg_full_mask = np.ones((patch_bgr.shape[0], patch_bgr.shape[1]), dtype=np.uint8) * 255
    patch_stats = _masked_hsv_stats(patch_bgr, bg_full_mask)

    brightness_diff = abs(object_stats["brightness"] - patch_stats["brightness"])
    contrast_diff = abs(object_stats["contrast"] - patch_stats["contrast"])
    saturation_diff = abs(object_stats["saturation"] - patch_stats["saturation"])

    tone_score = (
        _score_from_difference(brightness_diff, 120.0) * 0.4
        + _score_from_difference(contrast_diff, 80.0) * 0.3
        + _score_from_difference(saturation_diff, 120.0) * 0.3
    )

    object_texture = compute_patch_texture_score(object_bgr)
    patch_texture = compute_patch_texture_score(patch_bgr)
    texture_score = (
        _score_from_difference(abs(object_texture["grad_mean"] - patch_texture["grad_mean"]), 80.0) * 0.4
        + _score_from_difference(abs(object_texture["lap_var"] - patch_texture["lap_var"]), 800.0) * 0.3
        + _score_from_difference(abs(object_texture["edge_density"] - patch_texture["edge_density"]), 1.0) * 0.3
    )

    band = _build_boundary_band(object_mask, 5)
    band_pixels = np.count_nonzero(band)
    if band_pixels < 20:
        return None
    obj_band_stats = _masked_hsv_stats(object_bgr, band)
    bg_band_stats = _masked_hsv_stats(patch_bgr, band)
    edge_score = (
        _score_from_difference(abs(obj_band_stats["brightness"] - bg_band_stats["brightness"]), 100.0) * 0.45
        + _score_from_difference(abs(obj_band_stats["contrast"] - bg_band_stats["contrast"]), 70.0) * 0.25
        + _score_from_difference(abs(obj_band_stats["saturation"] - bg_band_stats["saturation"]), 100.0) * 0.15
        + _score_from_difference(abs(object_texture["grad_mean"] - patch_texture["grad_mean"]), 90.0) * 0.15
    )

    low_texture_penalty = 0.35 if patch_texture["edge_density"] < 0.02 else 0.0
    vertical_bonus = 1.0 - abs(y_ratio - 0.60)
    placement_score = max(0.0, min(1.0, vertical_bonus - low_texture_penalty))

    weights = MODE_CONFIG[mode]["weights"]
    total_score = (
        weights["tone"] * tone_score
        + weights["texture"] * texture_score
        + weights["edge"] * edge_score
        + weights["placement"] * placement_score
    )

    return {
        "total_score": float(total_score),
        "tone_score": float(tone_score),
        "texture_score": float(texture_score),
        "edge_score": float(edge_score),
        "placement_score": float(placement_score),
        "brightness_diff": float(brightness_diff),
        "contrast_diff": float(contrast_diff),
        "saturation_diff": float(saturation_diff),
    }


def select_best_object_and_position(background, background_path, object_candidates, mode, target_domain):
    """
    하나의 고정된 scale에 대해 object-position 조합을 전부 평가합니다.

    이 함수에 들어오기 전에 scale은 이미 정해져 있으므로,
    여기서는 object identity와 placement만 탐색합니다.
    """
    bg_h, bg_w = background.shape[:2]
    accepted = []
    strictness = MODE_CONFIG[mode]["strictness"]
    background_key = background_path.stem if background_path is not None else "unknown_bg"

    for object_item in object_candidates:
        object_bgr = load_image(object_item["image_path"], cv2.IMREAD_COLOR)
        object_mask = load_image(object_item["mask_path"], cv2.IMREAD_GRAYSCALE)
        if object_bgr is None or object_mask is None:
            continue

        object_domain = _get_object_domain_for_target_domain(target_domain)
        usage_guidance = _compute_usage_guidance(
            object_item["id"],
            object_domain,
            target_domain,
            background_key,
            mode,
        )
        if usage_guidance["remaining_quota"] <= 0:
            continue
        if usage_guidance["coarse_domain_penalty"] >= 1.0:
            continue

        object_mask = binarize_mask(object_mask)
        cropped_bgr, cropped_mask, _ = crop_to_mask_bbox(object_bgr, object_mask)
        if cropped_bgr is None or cropped_mask is None:
            continue

        target_ratio = object_item["target_ratio"]
        resized_bgr, resized_mask = resize_object_to_target_scale(cropped_bgr, cropped_mask, bg_h, bg_w, target_ratio)
        if resized_bgr is None or resized_mask is None:
            continue

        obj_h, obj_w = resized_mask.shape[:2]
        positions = sample_candidate_positions(bg_h, bg_w, obj_h, obj_w, object_item["num_position_candidates"])
        if not positions:
            continue

        for pos in positions:
            if strictness == "strict" and pos["touches_margin"]:
                continue

            patch = extract_background_patch(background, pos["x"], pos["y"], obj_w, obj_h)
            if patch is None:
                continue

            texture = compute_patch_texture_score(patch)
            if strictness in ("strict", "medium") and texture["edge_density"] < 0.01:
                continue

            if np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]) < 18 and np.mean(patch[:, :, 2]) > 190 and strictness == "strict":
                continue

            score = compute_object_background_match_score(
                resized_bgr,
                resized_mask,
                patch,
                pos["center_y_ratio"],
                mode,
            )
            if score is None:
                continue

            placement_adjust = -0.15 if pos["touches_margin"] else 0.05
            score["placement_score"] = max(0.0, min(1.0, score["placement_score"] + placement_adjust))
            weights = MODE_CONFIG[mode]["weights"]
            score["total_score"] = (
                weights["tone"] * score["tone_score"]
                + weights["texture"] * score["texture_score"]
                + weights["edge"] * score["edge_score"]
                + weights["placement"] * score["placement_score"]
            )
            score["domain_repeat_penalty"] = usage_guidance["domain_repeat_penalty"]
            score["background_repeat_penalty"] = usage_guidance["background_repeat_penalty"]
            score["recent_history_penalty"] = usage_guidance["recent_history_penalty"]
            score["quota_bonus"] = usage_guidance["quota_bonus"]
            usage_adjustment = (
                QUOTA_BONUS_WEIGHT * usage_guidance["quota_bonus"]
                - DOMAIN_PENALTY_WEIGHT * usage_guidance["domain_repeat_penalty"]
                - BACKGROUND_PENALTY_WEIGHT * usage_guidance["background_repeat_penalty"]
                - RECENT_PENALTY_WEIGHT * usage_guidance["recent_history_penalty"]
                - 0.30 * usage_guidance["coarse_domain_penalty"]
            )
            score["usage_adjustment"] = float(usage_adjustment)
            score["total_score"] += float(usage_adjustment)

            accepted.append(
                {
                    "object_item": object_item,
                    "object_bgr": resized_bgr,
                    "object_mask": resized_mask,
                    "x": pos["x"],
                    "y": pos["y"],
                    "score": score,
                    "patch": patch,
                    "background_key": background_key,
                }
            )

    if not accepted:
        return None

    accepted.sort(key=lambda item: item["score"]["total_score"], reverse=True)
    top_k = min(MODE_CONFIG[mode]["top_k"], len(accepted))

    if mode == "natural":
        return accepted[0]
    if mode == "semi":
        return accepted[GLOBAL_STATE["rng"].randint(0, max(0, min(3, top_k) - 1))]
    if mode == "hard":
        return accepted[GLOBAL_STATE["rng"].randint(0, max(0, min(5, top_k) - 1))]

    pool_size = min(max(3, top_k), len(accepted))
    weighted_pool = accepted[:pool_size]
    return weighted_pool[GLOBAL_STATE["rng"].randint(0, pool_size - 1)]


def apply_tone_matching(object_bgr, object_mask, patch_bgr, mode):
    """
    약한 global tone adjustment와 더 강한 local edge-band matching을 적용합니다.

    binary GT mask는 의도적으로 sharp하게 유지하고,
    실제로 시각적 보정은 합성 이미지 쪽에만 적용합니다.
    """
    config = MODE_CONFIG[mode]
    brightness_alpha = GLOBAL_STATE["rng"].uniform(config["brightness_alpha"][0], config["brightness_alpha"][1])
    contrast_alpha = GLOBAL_STATE["rng"].uniform(config["contrast_alpha"][0], config["contrast_alpha"][1])
    saturation_alpha = GLOBAL_STATE["rng"].uniform(config["saturation_alpha"][0], config["saturation_alpha"][1])

    result = object_bgr.astype(np.float32).copy()
    obj_stats = _masked_hsv_stats(object_bgr, object_mask)
    patch_mask = np.ones((patch_bgr.shape[0], patch_bgr.shape[1]), dtype=np.uint8) * 255
    bg_stats = _masked_hsv_stats(patch_bgr, patch_mask)

    mask_bool = object_mask > 0
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2][mask_bool] = np.clip(
        hsv[:, :, 2][mask_bool] + (bg_stats["brightness"] - obj_stats["brightness"]) * brightness_alpha,
        0,
        255,
    )
    if obj_stats["contrast"] > 1e-6:
        v = hsv[:, :, 2]
        mean_v = np.mean(v[mask_bool])
        target_scale = 1.0 + ((bg_stats["contrast"] / obj_stats["contrast"]) - 1.0) * contrast_alpha
        v[mask_bool] = np.clip((v[mask_bool] - mean_v) * target_scale + mean_v, 0, 255)
        hsv[:, :, 2] = v
    hsv[:, :, 1][mask_bool] = np.clip(
        hsv[:, :, 1][mask_bool] + (bg_stats["saturation"] - obj_stats["saturation"]) * saturation_alpha,
        0,
        255,
    )

    matched = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    band = _build_boundary_band(object_mask, band_width=5)
    band_bool = band > 0
    if np.count_nonzero(band_bool) > 0:
        local_alpha = min(1.0, brightness_alpha + 0.10)
        matched_hsv = cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        bg_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        band_v = matched_hsv[:, :, 2]
        band_s = matched_hsv[:, :, 1]
        band_v[band_bool] = np.clip(band_v[band_bool] * (1.0 - local_alpha) + bg_hsv[:, :, 2][band_bool] * local_alpha, 0, 255)
        band_s[band_bool] = np.clip(band_s[band_bool] * (1.0 - saturation_alpha - 0.05) + bg_hsv[:, :, 1][band_bool] * (saturation_alpha + 0.05), 0, 255)
        matched_hsv[:, :, 2] = band_v
        matched_hsv[:, :, 1] = band_s
        matched = cv2.cvtColor(matched_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return matched.astype(np.uint8)


def build_alpha_from_mask(mask, mode):
    alpha = mask.astype(np.float32) / 255.0
    feather_low, feather_high = MODE_CONFIG[mode]["feather_kernel"]
    if feather_high <= 0:
        return alpha
    kernel_size = GLOBAL_STATE["rng"].randint(feather_low, feather_high)
    if kernel_size <= 1:
        return alpha
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)
    return np.clip(blurred, 0.0, 1.0)


def blend_object(background, object_bgr, alpha, x, y):
    result = background.copy().astype(np.float32)
    obj_h, obj_w = object_bgr.shape[:2]
    roi = result[y:y + obj_h, x:x + obj_w]
    alpha_3 = np.repeat(alpha[:, :, None], 3, axis=2)
    blended = object_bgr.astype(np.float32) * alpha_3 + roi * (1.0 - alpha_3)
    result[y:y + obj_h, x:x + obj_w] = blended
    return np.clip(result, 0, 255).astype(np.uint8)


def build_transformed_binary_mask(bg_h, bg_w, object_mask, x, y):
    canvas = np.zeros((bg_h, bg_w), dtype=np.uint8)
    obj_h, obj_w = object_mask.shape[:2]
    canvas[y:y + obj_h, x:x + obj_w] = np.maximum(canvas[y:y + obj_h, x:x + obj_w], object_mask)
    return canvas


def mask_to_polygon_lines(mask, class_id):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    lines = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8.0:
            continue
        epsilon = max(1.0, 0.003 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        coords = []
        points = approx.reshape(-1, 2).astype(np.float32)
        for point in points:
            x_norm = np.clip(point[0] / float(w), 0.0, 1.0)
            y_norm = np.clip(point[1] / float(h), 0.0, 1.0)
            coords.append(f"{x_norm:.6f}")
            coords.append(f"{y_norm:.6f}")
        lines.append(str(class_id) + " " + " ".join(coords))
    return lines


def save_polygon_txt(txt_path, polygon_lines):
    try:
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text("\n".join(polygon_lines), encoding="utf-8")
        return True
    except OSError as exc:
        print(f"[WARN] Failed to save txt: {txt_path} ({exc})")
        return False

def save_image_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True

def save_result_image(path, image):
    ok = save_image_unicode(path, image)
    print(f"[DEBUG] save image: {path} -> {ok}")
    return ok

def save_result_mask(path, mask):
    ok = save_image_unicode(path, mask)
    print(f"[DEBUG] save mask: {path} -> {ok}")
    return ok


def _initialize_object_pools():
    GLOBAL_STATE["object_pools"] = {
        "snow": build_object_pool(OBJECT_POOL_ROOT / "snow" / "images", OBJECT_POOL_ROOT / "snow" / "masks"),
        "non_snow": build_object_pool(OBJECT_POOL_ROOT / "non_snow" / "images", OBJECT_POOL_ROOT / "non_snow" / "masks"),
    }
    for key in GLOBAL_STATE["object_pools"]:
        GLOBAL_STATE["rng"].shuffle(GLOBAL_STATE["object_pools"][key])


def _initialize_background_state():
    background_state = {}
    if EXECUTION_MODE == "DEMO":
        for domain, directory in DEMO_BACKGROUND_DIR_MAP.items():
            files = collect_image_files(directory)
            if domain == "forest_dense":
                files = files[:1]
            elif domain == "grass_field":
                files = files[:1]
            elif domain == "mixed":
                files = files[:2]
            background_state[domain] = {"paths": files}
    else:
        for domain, directory in FULL_BACKGROUND_DIR_MAP.items():
            background_state[domain] = {"paths": collect_image_files(directory)}
    GLOBAL_STATE["background_state"] = background_state


def _build_scale_plan(sample_count):
    weighted_buckets = []
    total_weight = 0
    for bucket in SCALE_DISTRIBUTION:
        total_weight += bucket["weight"]

    for bucket in SCALE_DISTRIBUTION:
        amount = int(round(sample_count * bucket["weight"] / float(total_weight)))
        for _ in range(amount):
            weighted_buckets.append(bucket)

    if weighted_buckets:
        weighted_buckets[0] = SCALE_DISTRIBUTION[0]
        if len(weighted_buckets) > 1:
            weighted_buckets[1] = SCALE_DISTRIBUTION[-1]

    while len(weighted_buckets) < sample_count:
        weighted_buckets.append(GLOBAL_STATE["rng"].choices(SCALE_DISTRIBUTION, weights=[b["weight"] for b in SCALE_DISTRIBUTION], k=1)[0])
    weighted_buckets = weighted_buckets[:sample_count]
    GLOBAL_STATE["rng"].shuffle(weighted_buckets)
    GLOBAL_STATE["scale_plan"] = list(weighted_buckets)


def _sanitize_stem(text):
    text = str(text).lower()
    cleaned = []
    for char in text:
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    result = "".join(cleaned)
    while "__" in result:
        result = result.replace("__", "_")
    return result.strip("_")


def _register_object_usage(object_id, object_domain, target_domain, background_key):
    state = GLOBAL_STATE["usage_state"][object_domain]
    state["remaining_quota"][object_id] = max(0, state["remaining_quota"].get(object_id, 0) - 1)
    state["used_total"][object_id] = state["used_total"].get(object_id, 0) + 1
    state["used_by_domain"].setdefault(object_id, {})
    state["used_by_domain"][object_id][target_domain] = state["used_by_domain"][object_id].get(target_domain, 0) + 1
    state["used_by_background"].setdefault(object_id, {})
    state["used_by_background"][object_id][background_key] = state["used_by_background"][object_id].get(background_key, 0) + 1
    GLOBAL_STATE["recent_objects"].append(object_id)
    if len(GLOBAL_STATE["recent_objects"]) > RECENT_HISTORY_SIZE:
        GLOBAL_STATE["recent_objects"] = GLOBAL_STATE["recent_objects"][-RECENT_HISTORY_SIZE:]


def _print_sample_debug(sample_idx, sample_info):
    print(
        "[SAMPLE] "
        f"idx={sample_idx:04d} "
        f"mode={sample_info['mode']} "
        f"domain={sample_info['domain']} "
        f"background={sample_info['background_name']} "
        f"object={sample_info['object_name']} "
        f"scale={sample_info['scale_bucket']} "
        f"position=({sample_info['x']},{sample_info['y']}) "
        f"score={sample_info['score']['total_score']:.4f} "
        f"tone={sample_info['score']['tone_score']:.4f} "
        f"texture={sample_info['score']['texture_score']:.4f} "
        f"edge={sample_info['score']['edge_score']:.4f} "
        f"placement={sample_info['score']['placement_score']:.4f} "
        f"quota_bonus={sample_info['score'].get('quota_bonus', 0.0):.4f} "
        f"domain_penalty={sample_info['score'].get('domain_repeat_penalty', 0.0):.4f} "
        f"background_penalty={sample_info['score'].get('background_repeat_penalty', 0.0):.4f} "
        f"recent_penalty={sample_info['score'].get('recent_history_penalty', 0.0):.4f} "
        f"usage_adjustment={sample_info['score'].get('usage_adjustment', 0.0):.4f} "
        f"brightness_diff={sample_info['score']['brightness_diff']:.2f} "
        f"contrast_diff={sample_info['score']['contrast_diff']:.2f} "
        f"saturation_diff={sample_info['score']['saturation_diff']:.2f}"
    )
    print(f"[SAVE] image={sample_info['image_path']}")
    print(f"[SAVE] txt={sample_info['txt_path']}")
    if sample_info["mask_path"] is not None:
        print(f"[SAVE] mask={sample_info['mask_path']}")


def _print_scale_histogram():
    print("\n[SUMMARY] Scale histogram")
    total = max(1, sum(GLOBAL_STATE["scale_hist"].values()))
    for bucket in SCALE_DISTRIBUTION:
        name = bucket["name"]
        count = GLOBAL_STATE["scale_hist"].get(name, 0)
        ratio = 100.0 * count / float(total)
        print(f"  {name}: {count} ({ratio:.2f}%)")


def _print_output_summary():
    print("\n[SUMMARY] Output locations")
    print(f"  image_root: {OUTPUT_IMAGE_ROOT}")
    print(f"  mask_root: {OUTPUT_MASK_ROOT}")
    print(f"  txt_root: {OUTPUT_TXT_ROOT}")
    print(f"  saved_images: {len(GLOBAL_STATE['saved_images'])}")
    print(f"  saved_masks: {len(GLOBAL_STATE['saved_masks'])}")
    print(f"  saved_txts: {len(GLOBAL_STATE['saved_txts'])}")
    if GLOBAL_STATE["saved_images"]:
        print(f"  first_image: {GLOBAL_STATE['saved_images'][0]}")
        print(f"  last_image: {GLOBAL_STATE['saved_images'][-1]}")
    if GLOBAL_STATE["saved_txts"]:
        print(f"  first_txt: {GLOBAL_STATE['saved_txts'][0]}")
        print(f"  last_txt: {GLOBAL_STATE['saved_txts'][-1]}")


def main():
    """
    synthetic data 생성 루프를 처음부터 끝까지 실행합니다.

    전체 순서:
    1. pool / schedule 초기화
    2. domain + scale + background 선택
    3. object / position 후보 평가
    4. 합성 및 저장
    5. usage 통계 갱신
    """
    _initialize_object_pools()
    schedule = choose_target_schedule()
    _initialize_usage_state(schedule)
    _initialize_background_state()
    _build_scale_plan(len(schedule))

    OUTPUT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASK_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT_ROOT.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    max_attempts = max(100, len(schedule) * 8)
    attempt = 0

    while schedule and attempt < max_attempts:
        attempt += 1
        target = schedule.pop(0)
        mode = target["mode"]
        domain = target["domain"]
        scale_bucket = sample_scale_bucket(mode)
        target_ratio = compute_target_area_ratio(scale_bucket)

        background_path = sample_background_for_domain(domain, mode)
        if background_path is None:
            print(f"[WARN] No background available for domain={domain}, skipping")
            skipped += 1
            continue

        background = load_image(background_path, cv2.IMREAD_COLOR)
        if background is None:
            print(f"[WARN] Failed to load background: {background_path}")
            skipped += 1
            continue
        background = resize_background_long_side(background, TARGET_BACKGROUND_LONG_SIDE)
        if background is None:
            skipped += 1
            continue

        candidate_count = MAX_OBJECT_CANDIDATES if EXECUTION_MODE == "FULL" else min(MAX_OBJECT_CANDIDATES, 6)
        position_count = MAX_POSITION_CANDIDATES if EXECUTION_MODE == "FULL" else min(MAX_POSITION_CANDIDATES, 12)
        object_candidates = sample_object_candidates(domain, background_path, candidate_count, GLOBAL_STATE["usage_state"], mode)
        if not object_candidates:
            print(f"[WARN] No object candidates available for domain={domain}")
            skipped += 1
            continue

        staged_candidates = []
        for item in object_candidates:
            staged = dict(item)
            staged["target_ratio"] = target_ratio
            staged["num_position_candidates"] = position_count
            staged_candidates.append(staged)

        best = select_best_object_and_position(background, background_path, staged_candidates, mode, domain)
        if best is None:
            print(f"[WARN] No valid object-position match for domain={domain}, mode={mode}")
            skipped += 1
            continue

        matched_object = apply_tone_matching(best["object_bgr"], best["object_mask"], best["patch"], mode)
        alpha = build_alpha_from_mask(best["object_mask"], mode)
        synthetic = blend_object(background, matched_object, alpha, best["x"], best["y"])
        transformed_mask = build_transformed_binary_mask(background.shape[0], background.shape[1], best["object_mask"], best["x"], best["y"])
        polygon_lines = mask_to_polygon_lines(transformed_mask, CLASS_ID)
        if not polygon_lines:
            print("[WARN] Polygon generation failed, skipping")
            skipped += 1
            continue

        stem = f"bg_{mode}_{created + 1}"
        image_path = OUTPUT_IMAGE_ROOT / f"{stem}.jpg"
        mask_path = OUTPUT_MASK_ROOT / f"{stem}.png" if SAVE_MASKS else None
        txt_path = OUTPUT_TXT_ROOT / f"{stem}.txt"

        image_saved = save_result_image(image_path, synthetic)
        txt_saved = save_polygon_txt(txt_path, polygon_lines)
        mask_saved = True
        if mask_path is not None:
            mask_saved = save_result_mask(mask_path, transformed_mask)

        if not image_saved or not txt_saved or not mask_saved:
            print(
                "[WARN] Save failed, sample will not be counted: "
                f"image={image_saved}, txt={txt_saved}, mask={mask_saved}"
            )
            skipped += 1
            continue

        object_domain = _get_object_domain_for_target_domain(domain)
        _register_object_usage(best["object_item"]["id"], object_domain, domain, best["background_key"])
        GLOBAL_STATE["scale_hist"][scale_bucket["name"]] = GLOBAL_STATE["scale_hist"].get(scale_bucket["name"], 0) + 1
        GLOBAL_STATE["saved_images"].append(str(image_path))
        GLOBAL_STATE["saved_txts"].append(str(txt_path))
        if mask_path is not None:
            GLOBAL_STATE["saved_masks"].append(str(mask_path))

        created += 1
        _print_sample_debug(
            created,
            {
                "mode": mode,
                "domain": domain,
                "background_name": background_path.name,
                "object_name": best["object_item"]["id"],
                "scale_bucket": scale_bucket["name"],
                "x": best["x"],
                "y": best["y"],
                "score": best["score"],
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else None,
                "txt_path": str(txt_path),
            },
        )

    print(f"\n[SUMMARY] created={created} skipped={skipped} remaining_schedule={len(schedule)} attempts={attempt}")
    if EXECUTION_MODE == "DEMO":
        print("[SUMMARY] Demo backgrounds were auto-limited to 4 images total:")
        for domain in sorted(GLOBAL_STATE["background_state"].keys()):
            paths = GLOBAL_STATE["background_state"][domain].get("paths", [])
            print(f"  {domain}: {len(paths)} file(s)")
            for path in paths:
                print(f"    - {path}")
    _print_scale_histogram()
    _print_output_summary()


if __name__ == "__main__":
    main()
