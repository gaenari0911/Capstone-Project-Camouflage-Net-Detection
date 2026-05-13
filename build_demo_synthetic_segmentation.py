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
import csv
import cv2
import numpy as np


OBJECT_POOL_ROOT = Path("Dataset") / "object_pool"
DEMO_BACKGROUND_ROOT = Path(r"G:\내 드라이브\데이터셋\snow")
FULL_BACKGROUND_ROOT = Path(r"G:\내 드라이브\데이터셋")
OUTPUT_IMAGE_ROOT = Path(r"G:\내 드라이브\합성_데이터셋\images")
OUTPUT_MASK_ROOT = Path(r"G:\내 드라이브\합성_데이터셋\masks")
OUTPUT_TXT_ROOT = Path(r"G:\내 드라이브\합성_데이터셋\txt")
OUTPUT_DEBUG_ROOT = Path(r"G:\내 드라이브\합성_데이터셋\debug")
OUTPUT_METADATA_CSV = Path(r"G:\내 드라이브\합성_데이터셋\synthesis_metadata.csv")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"}
TARGET_BACKGROUND_LONG_SIDE = 1280
MASK_THRESHOLD = 127
OUTER_MARGIN_RATIO = 0.05
MIN_OBJECT_PIXEL_SIZE = 20
MAX_OBJECT_CANDIDATES = 8
MAX_POSITION_CANDIDATES = 16
RANDOM_SEED = 7
EXECUTION_MODE = "FULL"  # "DEMO" or "FULL"
SYNTHESIS_SPLIT_NAME = "train" # 생성한 합성데이터가 어디 소속인지 metadata에 기록되는 값
DIFFICULTY_MODE = "hard"  # "easy", "medium", "hard", "extreme"
DEBUG_VISUALIZE = True
SAVE_MASKS = True
CLASS_ID = 0
MAX_OBJECT_USAGE = 5
RECENT_HISTORY_SIZE = 4
MAX_BACKGROUND_REPEAT_PER_OBJECT = 1
DOMAIN_PENALTY_WEIGHT = 0.08 #같은 객체가 반복되면 감점
BACKGROUND_PENALTY_WEIGHT = 0.18 #같은 배경 반복되면 감점
RECENT_PENALTY_WEIGHT = 0.10 #최근 쓴 객체면 감점
QUOTA_BONUS_WEIGHT = 0.06 #아직 덜 쓴 객체면 보너스
ARTIFACT_PENALTY_WEIGHT = 0.12 #경계 artifact가 있으면 감점

FULL_BACKGROUND_COUNTS = {
    "grass_field": 570,
    "forest_dense": 414,
    "leaf_ground": 258,
    "rocky": 305,
    "snow": 424,
    "mixed": 559,
}

FULL_MODE_TOTALS = {
    "natural": 1200,
    "semi": 900,
    "hard": 600,
    "minimal": 300,
}

FULL_TOTAL_IMAGES = 100
DIFFICULTY_SAMPLING_MODE = "mixed"  # "fixed" or "mixed" fixed는 difficulty mix 안하는거
DIFFICULTY_RATIOS = {
    "easy": 0.05,
    "medium": 0.25,
    "hard": 0.50,
    "extreme": 0.20,
}

DIFFICULTY_CONFIG = {
    "easy": {
        "scale_multiplier": 1.18,
        "bg_aug_strength": 0.60,
        "object_aug_strength": 0.55,
        "camouflage_pull": 0.10,
        "boundary_softness": 0.95,
        "clutter_strength": 0.05,
    },
    "medium": {
        "scale_multiplier": 1.00,
        "bg_aug_strength": 0.85,
        "object_aug_strength": 0.75,
        "camouflage_pull": 0.16,
        "boundary_softness": 1.00,
        "clutter_strength": 0.10,
    },
    "hard": {
        "scale_multiplier": 0.86,
        "bg_aug_strength": 1.00,
        "object_aug_strength": 0.92,
        "camouflage_pull": 0.22,
        "boundary_softness": 1.06,
        "clutter_strength": 0.16,
    },
    "extreme": {
        "scale_multiplier": 0.72,
        "bg_aug_strength": 1.12,
        "object_aug_strength": 1.00,
        "camouflage_pull": 0.28,
        "boundary_softness": 1.14,
        "clutter_strength": 0.22,
    },
}
ARTIFACT_PENALTY_WEIGHT = 0.12 #경계 artifact가 크면 감점

# 이 프로젝트는 domain 구조가 비대칭입니다.
# - snow 객체는 자연스럽게 연결되는 coarse domain이 1개뿐이고
# - non-snow 객체는 여러 background domain으로 퍼질 수 있습니다.
# 아래 quota/penalty 로직은 이 비대칭을 조금 더 공정하게 다루기 위한 장치입니다.

ALLOWED_DOMAINS = {
    "snow": ["snow"],
    "non_snow": ["forest_dense", "grass_field", "leaf_ground", "rocky", "mixed"],
}

DOMAIN_SPATIAL_POLICY = {
    "snow": {
        "perspective_strength": "strong", #원근감 적용 강도
        "use_vertical_prior": True, #y축 기반 prior = True : 아래쪽에 더 많이 배치
        "preferred_center_y": 0.72, # 이상적인 y좌표 / 0:화면 맨 위 / 1:화면 맨 아래
        "margin_penalty_strength": 0.65, # 가장자리 패널티 강도 / 값이 클 수록 점수 크게 깎여서
        #중앙에 위치하게 됨
    },
    "forest_dense": {
        "perspective_strength": "medium",
        "use_vertical_prior": True,
        "preferred_center_y": 0.68,
        "margin_penalty_strength": 0.55,
    },
    "grass_field": {
        "perspective_strength": "medium",
        "use_vertical_prior": True,
        "preferred_center_y": 0.70,
        "margin_penalty_strength": 0.50,
    },
    "mixed": {
        "perspective_strength": "medium",
        "use_vertical_prior": True,
        "preferred_center_y": 0.67,
        "margin_penalty_strength": 0.50,
    },
    "leaf_ground": {
        "perspective_strength": "weak",
        "use_vertical_prior": False,
        "preferred_center_y": 0.62,
        "margin_penalty_strength": 0.35,
    },
    "rocky": {
        "perspective_strength": "weak",
        "use_vertical_prior": False,
        "preferred_center_y": 0.60,
        "margin_penalty_strength": 0.35,
    },
}

SCALE_POSITION_POLICY = { #객체 위치 분포
    "00to05": {"min_center_y": 0.18, "max_center_y": 0.90}, # 작은 객체니까 y좌표 어디든 가능
    "05to10": {"min_center_y": 0.22, "max_center_y": 0.92},
    "10to15": {"min_center_y": 0.28, "max_center_y": 0.92},
    "15to20": {"min_center_y": 0.35, "max_center_y": 0.95},
    "20to25": {"min_center_y": 0.42, "max_center_y": 0.96},
    "25to30": {"min_center_y": 0.48, "max_center_y": 0.97},
    "30to40": {"min_center_y": 0.56, "max_center_y": 0.98},
    "40to60": {"min_center_y": 0.62, "max_center_y": 0.98}, # 큰 객체일 수록 아래쪽에 배치
}

MODE_CONFIG = {
    "natural": {
        "brightness_alpha": (0.20, 0.35), #배경 밝기에 맞게, 객체 밝기를 얼마나 조절할거냐(현재는 차이를 20~35프로 보정)
        "contrast_alpha": (0.20, 0.35), #대비 보정 강도
        "saturation_alpha": (0.05, 0.12), #채도 보정 강도
        "feather_kernel": (3, 7), # 커널 크기 3~7
        "weights": {"tone": 0.35, "texture": 0.20, "edge": 0.30, "placement": 0.15},
        # 최종 점수 비율: 색상/텍스쳐/경계/배치 위치
        "top_k": 1, # 점수 상위 몇 개의 후보 중에서 선택할 지 (후보 개수)
        "strictness": "strict", # 전체 합성 제약의 강도
        "domain_same_ratio": 1.00, #객체 도메인과 배경 도메인을 같은 계열로 유지할 확률
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

SCALE_DISTRIBUTION = [ #객체 크기 분포
    {"name": "00to05", "min_ratio": 0.00, "max_ratio": 0.05, "weight": 10},
    # 객체 크기가 배경의 10~15프로가 되게끔
    {"name": "05to10", "min_ratio": 0.05, "max_ratio": 0.10, "weight": 20},
    {"name": "10to15", "min_ratio": 0.10, "max_ratio": 0.15, "weight": 15},
    {"name": "15to20", "min_ratio": 0.15, "max_ratio": 0.20, "weight": 15},
    {"name": "20to25", "min_ratio": 0.20, "max_ratio": 0.25, "weight": 15},
    {"name": "25to30", "min_ratio": 0.25, "max_ratio": 0.30, "weight": 10},
    {"name": "30to40", "min_ratio": 0.30, "max_ratio": 0.40, "weight": 10},
    {"name": "40to60", "min_ratio": 0.40, "max_ratio": 0.60, "weight": 5},
]

DOMAIN_TARGETS = { #합성데이터 생성 목표 개수
    "snow": 300,
    "forest_dense": 900,
    "grass_field": 600,
    "leaf_ground": 450,
    "rocky": 300,
    "mixed": 450,
}

MODE_TARGETS = { # 합성데이터 목표를, 모드 별로 분할한거임
    "natural": {"snow": 120, "forest_dense": 360, "grass_field": 240, "leaf_ground": 180, "rocky": 120, "mixed": 180},
    "semi": {"snow": 90, "forest_dense": 270, "grass_field": 180, "leaf_ground": 135, "rocky": 90, "mixed": 135},
    "hard": {"snow": 60, "forest_dense": 180, "grass_field": 120, "leaf_ground": 90, "rocky": 60, "mixed": 90},
    "minimal": {"snow": 30, "forest_dense": 90, "grass_field": 60, "leaf_ground": 45, "rocky": 30, "mixed": 45},
}

DEMO_MODE_TARGETS = { # 데모용
    "natural": {"snow": 120},
    "semi": {"snow": 90},
    "hard": {"snow": 60},
    "minimal": {"snow": 30},
}

DEMO_BACKGROUND_DIR_MAP = {
    "snow": DEMO_BACKGROUND_ROOT,
}

FULL_BACKGROUND_DIR_MAP = {
    "snow": FULL_BACKGROUND_ROOT / "snow",
    "forest_dense": FULL_BACKGROUND_ROOT / "forest_dense",
    "grass_field": FULL_BACKGROUND_ROOT / "grass_field",
    "leaf_ground": FULL_BACKGROUND_ROOT / "leaf_ground",
    "rocky": FULL_BACKGROUND_ROOT / "rocky",
    "mixed": FULL_BACKGROUND_ROOT / "mixed",
}

GLOBAL_STATE = {
    "rng": random.Random(RANDOM_SEED), #랜덤 생성기(재현가능하도록)
    "background_state": {}, #도메인별 배경 이미지 리스트 저장
    "background_rotation": {}, #배경 중복 사용 방지용 큐
    "background_usage": {}, #각 배경이 몇 번 쓰였는지 기록 => 덜 쓴 배경 먼저 쓰기 위함
    "usage_state": {}, #객체 최대 몇 번 쓸 수 있는지,
    #얼마나 썼는지, 특정 도메인에서 얼마나 썼는지, 같은 배경에 몇 번 붙였는지
    #같은 객체 반복 사용을 방지하기 위함
    "object_pools": {}, # 사용할 객체 리스트 저장
    "recent_objects": [], # 최근 사용 객체면 패널티
    "scale_plan": [], # 미리 만들어둔 스케일 계획 리스트
    "scale_hist": {}, # 실제 사용된 스케일 기록 (계획과 비교)
    "saved_images": [], # 저장 로그
    "saved_masks": [], # 저장 로그
    "saved_txts": [], # 저장 로그
    "metadata_rows": [],
    "background_split_registry": {},
    "background_variant_signatures": {},
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


def get_scale_position_policy(scale_bucket_name):
    """
    scale bucket별 기본 center_y 허용 범위를 반환합니다.

    여기서의 범위는 hard constraint의 출발점이고,
    이후 domain별 spatial policy로 약간 완화/강화됩니다.
    """
    return SCALE_POSITION_POLICY.get(
        scale_bucket_name,
        {"min_center_y": 0.20, "max_center_y": 0.95},
    )


def get_domain_spatial_policy(domain):
    """
    background domain별 spatial prior 설정을 반환합니다.

    weak/medium/strong는 후보 생성 단계에서 hard range를 얼마나
    완화할지 결정하고, preferred_center_y는 soft prior 계산 시 참고합니다.
    """
    return DOMAIN_SPATIAL_POLICY.get(
        domain,
        {
            "perspective_strength": "medium",
            "use_vertical_prior": True,
            "preferred_center_y": 0.66,
            "margin_penalty_strength": 0.45,
        },
    )


def get_effective_center_y_range(scale_bucket_name, domain):
    """
    scale policy와 domain policy를 합쳐 실제 후보 생성용 center_y 범위를 만듭니다.

    - strong: 기본 범위 유지
    - medium: 위/아래로 소폭 완화
    - weak: 좀 더 넓게 완화
    """
    scale_policy = get_scale_position_policy(scale_bucket_name)
    domain_policy = get_domain_spatial_policy(domain)

    min_center_y = scale_policy["min_center_y"]
    max_center_y = scale_policy["max_center_y"]

    if not domain_policy["use_vertical_prior"]:
        relax = 0.10
    else:
        strength = domain_policy["perspective_strength"]
        if strength == "strong":
            relax = 0.00
        elif strength == "medium":
            relax = 0.04
        else:
            relax = 0.08

    min_center_y = np.clip(min_center_y - relax, 0.05, 0.98)
    max_center_y = np.clip(max_center_y + relax, 0.10, 0.99)
    return float(min_center_y), float(max_center_y)


def compute_soft_perspective_score(center_y_ratio, allowed_min_center_y, allowed_max_center_y):
    """
    허용 band의 중앙에 가까울수록 높은 soft perspective score를 반환합니다.

    후보 생성 단계에서 이미 hard filtering을 했기 때문에,
    여기서는 가능/불가능 판정보다 "더 자연스러운 정도"만 약하게 반영합니다.
    """
    expected_center_y = (allowed_min_center_y + allowed_max_center_y) * 0.5
    band_half = max(0.05, (allowed_max_center_y - allowed_min_center_y) * 0.5)
    distance = abs(center_y_ratio - expected_center_y)
    normalized_distance = min(1.0, distance / band_half)
    score = 1.0 - normalized_distance
    return float(np.clip(score, 0.0, 1.0)), float(expected_center_y), float(distance)


def compute_margin_score(x, y, obj_w, obj_h, bg_w, bg_h):
    """
    물체가 배경 가장자리에 너무 붙었는지를 0~1 점수로 계산합니다.

    중앙이라고 무조건 최고점을 주기보다는,
    경계에서 일정 거리 이상만 확보되면 점수가 포화되도록 설계합니다.
    """
    left_margin = float(x)
    top_margin = float(y)
    right_margin = float(bg_w - (x + obj_w))
    bottom_margin = float(bg_h - (y + obj_h))
    min_margin = min(left_margin, top_margin, right_margin, bottom_margin)
    reference_margin = max(12.0, min(bg_w, bg_h) * OUTER_MARGIN_RATIO)
    margin_ratio = float(np.clip(min_margin / reference_margin, 0.0, 1.0))
    score = float(np.sqrt(margin_ratio))
    return score, margin_ratio


def deduplicate_positions(positions, min_center_dist_px):
    """
    중심점이 너무 가까운 후보를 제거해 비슷한 위치가 과하게 누적되지 않게 합니다.
    """
    deduped = []
    min_center_dist_sq = float(min_center_dist_px * min_center_dist_px)
    for position in positions:
        center_x = position["x"] + position["obj_w"] * 0.5
        center_y = position["y"] + position["obj_h"] * 0.5
        duplicated = False
        for existing in deduped:
            existing_center_x = existing["x"] + existing["obj_w"] * 0.5
            existing_center_y = existing["y"] + existing["obj_h"] * 0.5
            dx = center_x - existing_center_x
            dy = center_y - existing_center_y
            if dx * dx + dy * dy < min_center_dist_sq:
                duplicated = True
                break
        if not duplicated:
            deduped.append(position)
    return deduped


def get_edge_blend_params(mask, mode):
    """
    GT용 binary mask와 별개로, blending용 경계 완화 파라미터를 계산합니다.

    hard/minimal에서도 feather 0은 금지하고 최소 anti-alias를 유지합니다.
    내부는 최대한 보존하고, 경계 band만 부드럽게 전이되도록 설계합니다.
    """
    area = max(1.0, float(np.count_nonzero(mask)))
    size_hint = np.sqrt(area)
    config = {
        "natural": {"ratio": 0.030, "min_bw": 3, "max_bw": 6, "min_blur": 3, "max_blur": 7},
        "semi": {"ratio": 0.024, "min_bw": 2, "max_bw": 5, "min_blur": 3, "max_blur": 5},
        "hard": {"ratio": 0.016, "min_bw": 1, "max_bw": 3, "min_blur": 3, "max_blur": 3},
        "minimal": {"ratio": 0.012, "min_bw": 1, "max_bw": 2, "min_blur": 3, "max_blur": 3},
    }.get(
        mode,
        {"ratio": 0.020, "min_bw": 2, "max_bw": 4, "min_blur": 3, "max_blur": 5},
    )

    boundary_width_px = int(round(size_hint * config["ratio"]))
    boundary_width_px = int(np.clip(boundary_width_px, config["min_bw"], config["max_bw"]))

    blur_kernel_px = boundary_width_px * 2 + 1
    blur_kernel_px = int(np.clip(blur_kernel_px, config["min_blur"], config["max_blur"]))
    if blur_kernel_px % 2 == 0:
        blur_kernel_px += 1

    return {
        "boundary_width_px": boundary_width_px,
        "blur_kernel_px": blur_kernel_px,
        "interior_preserve_ratio": 1.0,
    }


def shrink_mask_for_blending(mask, mode, object_bgr=None, patch_bgr=None):
    """
    GT mask와 별개로 blending에 사용할 source mask를 아주 약하게 안쪽으로 줄입니다.

    흰 fringe가 alpha에 직접 참여하지 않도록 1~2px 수준의 미세 erosion만 허용합니다.
    """
    binary_mask = (mask > 0).astype(np.uint8) * 255
    area = max(1.0, float(np.count_nonzero(binary_mask)))
    size_hint = np.sqrt(area)

    if mode in ("hard", "minimal"):
        shrink_px = 1
    elif mode == "semi":
        shrink_px = 2 if size_hint >= 140 else 1
    else:
        shrink_px = 2 if size_hint >= 110 else 1

    if object_bgr is not None and patch_bgr is not None and np.count_nonzero(binary_mask) > 0:
        metrics = compute_boundary_artifact_metrics(object_bgr, binary_mask, patch_bgr)
        if metrics is not None and size_hint >= 42:
            extra_shrink = 0
            if metrics["white_fringe_risk"] > 0.18 or metrics["white_matte_fraction"] > 0.24:
                extra_shrink += 1
            if metrics["white_fringe_risk"] > 0.32 and metrics["halo_risk_score"] > 22.0 and size_hint >= 90:
                extra_shrink += 1
            max_shrink = 2 if mode in ("hard", "minimal") else 3
            shrink_px = int(np.clip(shrink_px + extra_shrink, 1, max_shrink))

    kernel_size = shrink_px * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    shrunk = cv2.erode(binary_mask, kernel, iterations=1)

    original_pixels = np.count_nonzero(binary_mask)
    shrunk_pixels = np.count_nonzero(shrunk)
    if shrunk_pixels <= 0:
        return binary_mask

    # 작은 객체가 과하게 얇아지는 것을 방지합니다.
    if shrunk_pixels < original_pixels * 0.65:
        fallback_kernel = np.ones((3, 3), np.uint8)
        shrunk = cv2.erode(binary_mask, fallback_kernel, iterations=1)
        if np.count_nonzero(shrunk) <= 0:
            return binary_mask

    return shrunk


def build_boundary_only_alpha(mask, boundary_width_px, blur_kernel_px):
    """
    내부는 alpha=1을 유지하고, 경계 band에서만 soft transition을 만듭니다.

    GT mask 자체는 blur하지 않고 그대로 유지하며,
    blending용 alpha만 부드럽게 생성합니다.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    if np.count_nonzero(binary_mask) == 0:
        return binary_mask.astype(np.float32)

    mask_255 = binary_mask * 255
    dist_inside = cv2.distanceTransform(mask_255, cv2.DIST_L2, 5)
    dist_outside = cv2.distanceTransform(255 - mask_255, cv2.DIST_L2, 5)

    boundary_width_px = max(1, int(boundary_width_px))
    alpha = np.clip((dist_inside + 0.25) / float(boundary_width_px + 0.25), 0.0, 1.0)
    alpha[binary_mask == 0] = 0.0
    alpha[dist_inside >= float(boundary_width_px)] = 1.0

    if blur_kernel_px > 1:
        blurred = cv2.GaussianBlur(alpha.astype(np.float32), (blur_kernel_px, blur_kernel_px), 0)
        band_region = (dist_inside <= float(boundary_width_px * 1.5)) | (dist_outside <= float(boundary_width_px))
        alpha = np.where(band_region, blurred, alpha)
        alpha[binary_mask == 0] = np.where(dist_outside[binary_mask == 0] <= float(boundary_width_px), alpha[binary_mask == 0], 0.0)
        alpha[dist_inside >= float(boundary_width_px)] = 1.0

    return np.clip(alpha.astype(np.float32), 0.0, 1.0)


def decontaminate_object_edge(object_bgr, object_mask, patch_bgr, mode):
    """
    객체 외곽의 흰 matte fringe를 줄이기 위해 boundary RGB를 정리합니다.

    interior color를 우선 참조하고, patch color는 약하게만 섞어 halo를 완화합니다.
    """
    binary_mask = (object_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(binary_mask) == 0:
        return object_bgr

    inner_kernel = np.ones((5, 5), np.uint8)
    inner_mask = cv2.erode(binary_mask, inner_kernel, iterations=1)
    if np.count_nonzero(inner_mask) < 12:
        inner_mask = cv2.erode(binary_mask, np.ones((3, 3), np.uint8), iterations=1)
    if np.count_nonzero(inner_mask) < 12:
        inner_mask = binary_mask.copy()

    boundary_band = cv2.subtract(binary_mask, inner_mask)
    boundary_band = binarize_mask(boundary_band)
    if np.count_nonzero(boundary_band) < 8:
        return object_bgr

    result = object_bgr.astype(np.float32).copy()
    inner_bool = inner_mask > 0
    boundary_bool = boundary_band > 0

    inner_mask_float = (inner_mask > 0).astype(np.float32)
    blur_kernel = {
        "natural": 15,
        "semi": 13,
        "hard": 11,
        "minimal": 9,
    }.get(mode, 11)
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    interior_weight = cv2.GaussianBlur(inner_mask_float, (blur_kernel, blur_kernel), 0)
    local_interior = np.zeros_like(result, dtype=np.float32)
    for channel_idx in range(3):
        channel = result[:, :, channel_idx] * inner_mask_float
        blurred_channel = cv2.GaussianBlur(channel, (blur_kernel, blur_kernel), 0)
        local_interior[:, :, channel_idx] = blurred_channel / np.maximum(interior_weight, 1e-4)

    if np.count_nonzero(inner_bool) > 0:
        fallback_interior = np.mean(result[inner_bool], axis=0).astype(np.float32)
        weak_local = interior_weight < 1e-3
        if np.any(weak_local):
            local_interior[weak_local] = fallback_interior

    self_weight, interior_weight_base, patch_weight = {
        "natural": (0.35, 0.50, 0.15),
        "semi": (0.40, 0.48, 0.12),
        "hard": (0.45, 0.45, 0.10),
        "minimal": (0.50, 0.42, 0.08),
    }.get(mode, (0.42, 0.46, 0.10))

    object_hsv = cv2.cvtColor(object_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    boundary_v = object_hsv[:, :, 2][boundary_bool]
    boundary_s = object_hsv[:, :, 1][boundary_bool]
    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    patch_v = patch_hsv[:, :, 2][boundary_bool]

    white_like = np.clip((boundary_v - 170.0) / 70.0, 0.0, 1.0) * np.clip((70.0 - boundary_s) / 70.0, 0.0, 1.0)
    bright_over_patch = np.clip((boundary_v - patch_v - 8.0) / 40.0, 0.0, 1.0)
    matte_strength = np.clip(white_like * bright_over_patch, 0.0, 1.0)
    white_like = white_like.astype(np.float32)

    boundary_pixels = result[boundary_bool]
    patch_pixels = patch_bgr.astype(np.float32)[boundary_bool]
    local_interior_pixels = local_interior[boundary_bool]

    total_interior_weight = interior_weight_base + (0.15 + 0.10 * (mode == "natural")) * matte_strength[:, None]
    total_patch_weight = patch_weight + 0.04 * matte_strength[:, None]
    total_self_weight = np.clip(self_weight - 0.20 * matte_strength[:, None], 0.12, 1.0)

    weight_sum = np.clip(total_self_weight + total_interior_weight + total_patch_weight, 1e-4, None)
    total_self_weight = total_self_weight / weight_sum
    total_interior_weight = total_interior_weight / weight_sum
    total_patch_weight = total_patch_weight / weight_sum

    cleaned = (
        boundary_pixels * total_self_weight
        + local_interior_pixels * total_interior_weight
        + patch_pixels * total_patch_weight
    )
    result[boundary_bool] = cleaned

    return np.clip(result, 0, 255).astype(np.uint8)


def compute_boundary_artifact_metrics(object_bgr, object_mask, patch_bgr):
    """
    경계 접합부에서 잘라붙인 티가 얼마나 나는지 정량화합니다.

    edge_score와 별도로 brightness jump, saturation jump, gradient discontinuity,
    halo risk를 직접 보고 약한 penalty로 사용합니다.
    """
    binary_mask = (object_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(binary_mask) == 0:
        return None

    inner_band = _build_boundary_band(binary_mask, band_width=3)
    outer_band = cv2.subtract(cv2.dilate(binary_mask, np.ones((5, 5), np.uint8), iterations=1), binary_mask)
    outer_band = binarize_mask(outer_band)

    if np.count_nonzero(inner_band) < 12 or np.count_nonzero(outer_band) < 12:
        return None

    object_hsv = cv2.cvtColor(object_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    inner_bool = inner_band > 0
    outer_bool = outer_band > 0

    boundary_brightness_jump = abs(
        float(np.mean(object_hsv[:, :, 2][inner_bool])) - float(np.mean(patch_hsv[:, :, 2][outer_bool]))
    )
    boundary_saturation_jump = abs(
        float(np.mean(object_hsv[:, :, 1][inner_bool])) - float(np.mean(patch_hsv[:, :, 1][outer_bool]))
    )

    object_gray = cv2.cvtColor(object_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    patch_gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    object_grad = cv2.magnitude(
        cv2.Sobel(object_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(object_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    patch_grad = cv2.magnitude(
        cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    boundary_gradient_jump = abs(
        float(np.mean(object_grad[inner_bool])) - float(np.mean(patch_grad[outer_bool]))
    )

    inner_value = float(np.mean(object_hsv[:, :, 2][inner_bool]))
    outer_value = float(np.mean(patch_hsv[:, :, 2][outer_bool]))
    halo_risk_score = abs(inner_value - outer_value)

    boundary_v = object_hsv[:, :, 2][inner_bool]
    boundary_s = object_hsv[:, :, 1][inner_bool]
    patch_boundary_v_mean = float(np.mean(patch_hsv[:, :, 2][outer_bool]))
    white_like_score = np.clip((boundary_v - 175.0) / 65.0, 0.0, 1.0) * np.clip((65.0 - boundary_s) / 65.0, 0.0, 1.0)
    bright_over_patch = np.clip((boundary_v - patch_boundary_v_mean - 10.0) / 45.0, 0.0, 1.0)
    white_matte_pixels = ((boundary_v > 205.0) & (boundary_s < 42.0)).astype(np.float32)
    white_matte_fraction = float(np.mean(white_matte_pixels)) if white_matte_pixels.size > 0 else 0.0
    white_fringe_risk = float(np.mean(white_like_score * bright_over_patch)) if white_like_score.size > 0 else 0.0
    white_fringe_risk = float(np.clip(0.75 * white_fringe_risk + 0.25 * white_matte_fraction, 0.0, 1.0))

    artifact_penalty = (
        np.clip(boundary_brightness_jump / 90.0, 0.0, 1.0) * 0.35
        + np.clip(boundary_saturation_jump / 90.0, 0.0, 1.0) * 0.25
        + np.clip(boundary_gradient_jump / 70.0, 0.0, 1.0) * 0.30
        + np.clip(halo_risk_score / 110.0, 0.0, 1.0) * 0.10
        + np.clip(white_fringe_risk, 0.0, 1.0) * 0.12
        + np.clip(white_matte_fraction, 0.0, 1.0) * 0.08
    )

    return {
        "artifact_penalty": float(np.clip(artifact_penalty, 0.0, 1.0)),
        "boundary_brightness_jump": float(boundary_brightness_jump),
        "boundary_saturation_jump": float(boundary_saturation_jump),
        "boundary_gradient_jump": float(boundary_gradient_jump),
        "halo_risk_score": float(halo_risk_score),
        "white_fringe_risk": float(np.clip(white_fringe_risk, 0.0, 1.0)),
        "white_matte_fraction": float(np.clip(white_matte_fraction, 0.0, 1.0)),
    }


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


def _allocate_counts_by_weights(total_count, weight_map):
    """
    총 샘플 수를 weight 비율에 맞춰 정수 count로 나눕니다.

    FULL 모드에서는 실제 확보된 background 분포를 그대로 따라가도록
    domain별 목표 개수를 동적으로 계산하는 데 사용합니다.
    """
    if total_count <= 0:
        return {key: 0 for key in weight_map}

    total_weight = float(sum(weight_map.values()))
    raw_counts = {}
    allocated = {}
    remainders = []
    assigned = 0
    for key, value in weight_map.items():
        raw = total_count * (float(value) / max(total_weight, 1e-6))
        count = int(np.floor(raw))
        raw_counts[key] = raw
        allocated[key] = count
        remainders.append((raw - count, key))
        assigned += count

    remainders.sort(reverse=True)
    idx = 0
    while assigned < total_count and remainders:
        _, key = remainders[idx % len(remainders)]
        allocated[key] += 1
        assigned += 1
        idx += 1
    return allocated


def build_full_mode_targets_from_background_distribution():
    """
    FULL 모드의 mode/domain 목표 수를 실제 background 확보 비율 기반으로 만듭니다.
    """
    mode_ratios = {}
    total_mode_weight = float(sum(FULL_MODE_TOTALS.values()))
    for mode_name, count in FULL_MODE_TOTALS.items():
        mode_ratios[mode_name] = float(count) / max(total_mode_weight, 1e-6)

    mode_totals = _allocate_counts_by_weights(FULL_TOTAL_IMAGES, mode_ratios)
    weighted_targets = {}
    for mode_name, total_count in mode_totals.items():
        weighted_targets[mode_name] = _allocate_counts_by_weights(total_count, FULL_BACKGROUND_COUNTS)
    return weighted_targets


def get_difficulty_config(difficulty_mode):
    return DIFFICULTY_CONFIG.get(difficulty_mode, DIFFICULTY_CONFIG["hard"])


def adjust_target_area_ratio_for_difficulty(target_ratio, difficulty_mode):
    """
    difficulty가 높을수록 객체 면적이 조금 더 작아지게 조정합니다.
    """
    cfg = get_difficulty_config(difficulty_mode)
    adjusted = target_ratio * cfg["scale_multiplier"]
    return float(np.clip(adjusted, 0.008, 0.60))


def _sample_uniform(low, high):
    return GLOBAL_STATE["rng"].uniform(low, high)


def _sample_int(low, high):
    return GLOBAL_STATE["rng"].randint(low, high)


def _apply_gamma(image, gamma):
    image_f = image.astype(np.float32) / 255.0
    corrected = np.power(np.clip(image_f, 0.0, 1.0), gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def _apply_motion_blur(image, kernel_size, horizontal=True):
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    if horizontal:
        kernel[kernel_size // 2, :] = 1.0
    else:
        kernel[:, kernel_size // 2] = 1.0
    kernel /= np.sum(kernel)
    return cv2.filter2D(image, -1, kernel)


def _apply_directional_blur(image, kernel_size, angle_deg):
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rot_mat = cv2.getRotationMatrix2D((kernel_size / 2.0 - 0.5, kernel_size / 2.0 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))
    kernel /= max(np.sum(kernel), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def _apply_local_brightness_variation(image, amplitude):
    h, w = image.shape[:2]
    noise = GLOBAL_STATE["rng"].random()
    grid_h = max(8, h // 8)
    grid_w = max(8, w // 8)
    coarse = np.random.default_rng(int(RANDOM_SEED + noise * 100000)).normal(0.0, amplitude, size=(grid_h, grid_w)).astype(np.float32)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + coarse), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_shadow_overlay(image, strength):
    h, w = image.shape[:2]
    center_x = _sample_uniform(0.2, 0.8) * w
    center_y = _sample_uniform(0.2, 0.8) * h
    axis_x = _sample_uniform(0.18, 0.40) * w
    axis_y = _sample_uniform(0.12, 0.35) * h
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (((xx - center_x) / max(axis_x, 1.0)) ** 2 + ((yy - center_y) / max(axis_y, 1.0)) ** 2)
    mask = np.exp(-mask).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(9.0, min(h, w) * 0.06))
    factor = 1.0 - strength * mask[:, :, None]
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _apply_fog_effect(image, strength):
    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=6.0 + 10.0 * strength)
    lifted = cv2.addWeighted(image, 1.0 - 0.20 * strength, blur, 0.12 * strength, 255.0 * 0.04 * strength)
    return np.clip(lifted, 0, 255).astype(np.uint8)


def _apply_patch_color_perturbation(image, strength):
    result = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    patch_count = 2 if strength < 0.9 else 3
    for _ in range(patch_count):
        pw = _sample_int(max(24, w // 10), max(32, w // 4))
        ph = _sample_int(max(24, h // 10), max(32, h // 4))
        x = _sample_int(0, max(0, w - pw))
        y = _sample_int(0, max(0, h - ph))
        delta = np.array([
            _sample_uniform(-16.0, 16.0) * strength,
            _sample_uniform(-12.0, 12.0) * strength,
            _sample_uniform(-18.0, 18.0) * strength,
        ], dtype=np.float32)
        result[y:y + ph, x:x + pw] = np.clip(result[y:y + ph, x:x + pw] + delta[None, None, :], 0, 255)
    return result.astype(np.uint8)


def _apply_partial_blur_region(image, strength):
    h, w = image.shape[:2]
    pw = _sample_int(max(20, w // 8), max(28, w // 3))
    ph = _sample_int(max(20, h // 8), max(28, h // 3))
    x = _sample_int(0, max(0, w - pw))
    y = _sample_int(0, max(0, h - ph))
    sigma = 1.0 + 2.0 * strength
    blurred_patch = cv2.GaussianBlur(image[y:y + ph, x:x + pw], (0, 0), sigmaX=sigma)
    result = image.copy()
    result[y:y + ph, x:x + pw] = blurred_patch
    return result


def _apply_edge_noise(image, strength):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    noise = np.random.default_rng(_sample_int(0, 10**6)).normal(0.0, 10.0 * strength, size=image.shape).astype(np.float32)
    result = image.astype(np.float32)
    edge_mask = (edges > 0)[:, :, None].astype(np.float32)
    result = np.clip(result + noise * edge_mask, 0, 255)
    return result.astype(np.uint8)


def _simulate_jpeg_artifact(image, strength):
    quality = int(np.clip(95 - 18 * strength, 68, 95))
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image


def apply_domain_aware_background_augmentation(background, domain, difficulty_mode):
    """
    domain 특성에 맞는 augmentation을 background에 적용합니다.
    """
    strength = get_difficulty_config(difficulty_mode)["bg_aug_strength"]
    image = background.copy()
    records = []

    if domain == "forest_dense":
        image = _apply_shadow_overlay(image, 0.12 * strength)
        image = _apply_local_brightness_variation(image, 0.05 * strength)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + 4.0 * strength, 180.0)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        image = _apply_motion_blur(image, 3 + int(round(2 * strength)), horizontal=False)
        records.extend(["shadow", "green_hue", "local_darkness", "motion_blur"])
    elif domain == "grass_field":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + _sample_uniform(-0.10, 0.18) * strength), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + _sample_uniform(-0.08, 0.10) * strength), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        image = _apply_directional_blur(image, 5, angle_deg=_sample_uniform(-20.0, 20.0))
        image = _apply_patch_color_perturbation(image, 0.8 * strength)
        records.extend(["sat_jitter", "brightness_fluctuation", "directional_blur", "texture_noise"])
    elif domain == "leaf_ground":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + _sample_uniform(-6.0, 10.0) * strength, 180.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 - 0.08 * strength), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 - 0.05 * strength), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        image = cv2.addWeighted(image, 0.92, cv2.GaussianBlur(image, (0, 0), sigmaX=4.0), 0.08, 0.0)
        image = _apply_patch_color_perturbation(image, 0.75 * strength)
        records.extend(["brown_yellow_shift", "contrast_reduction", "debris_overlay"])
    elif domain == "rocky":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        image = cv2.addWeighted(image, 0.75, gray_3, 0.25, 0.0)
        sharp = cv2.addWeighted(image, 1.18, cv2.GaussianBlur(image, (0, 0), sigmaX=1.1), -0.18, 0.0)
        image = _apply_edge_noise(np.clip(sharp, 0, 255).astype(np.uint8), 0.70 * strength)
        records.extend(["grayscale_perturb", "rough_sharpen", "edge_noise"])
    elif domain == "snow":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 - 0.18 * strength), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + 0.08 * strength), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        image = _apply_fog_effect(image, 0.55 * strength)
        wb_shift = np.array([1.0 - 0.03 * strength, 1.0, 1.0 + 0.03 * strength], dtype=np.float32)
        image = np.clip(image.astype(np.float32) * wb_shift[None, None, :], 0, 255).astype(np.uint8)
        records.extend(["overexposure", "low_saturation", "white_balance", "fog"])
    else:
        image = _apply_patch_color_perturbation(image, 0.55 * strength)
        image = _apply_local_brightness_variation(image, 0.04 * strength)
        if GLOBAL_STATE["rng"].random() < 0.5:
            image = _apply_motion_blur(image, 3, horizontal=GLOBAL_STATE["rng"].random() < 0.5)
            records.append("mixed_blur")
        records.extend(["mixed_patch_color", "mixed_local_brightness"])

    return np.clip(image, 0, 255).astype(np.uint8), records


def apply_anti_memorization_background_transform(background, domain, difficulty_mode):
    """
    동일 background 원본을 써도 appearance가 충분히 달라지도록 변형합니다.
    """
    image = background.copy()
    h, w = image.shape[:2]
    strength = get_difficulty_config(difficulty_mode)["bg_aug_strength"]
    records = []

    crop_ratio = _sample_uniform(0.84, 0.98 if difficulty_mode == "easy" else 0.94)
    crop_w = max(64, int(round(w * crop_ratio)))
    crop_h = max(64, int(round(h * crop_ratio)))
    crop_x = _sample_int(0, max(0, w - crop_w))
    crop_y = _sample_int(0, max(0, h - crop_h))
    image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    records.append("random_crop_resize")

    angle = _sample_uniform(-4.0, 4.0) * strength
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, _sample_uniform(0.98, 1.02))
    image = cv2.warpAffine(image, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    records.append("rotation")

    perspective_mag = 0.018 * strength
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst = src.copy()
    for idx in range(4):
        dst[idx, 0] += _sample_uniform(-perspective_mag, perspective_mag) * w
        dst[idx, 1] += _sample_uniform(-perspective_mag, perspective_mag) * h
    persp = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(image, persp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    records.append("perspective")

    image = _apply_local_brightness_variation(image, 0.03 + 0.03 * strength)
    image = _apply_patch_color_perturbation(image, 0.55 + 0.20 * strength)
    image = _apply_partial_blur_region(image, 0.45 + 0.30 * strength)
    image = _apply_gamma(image, _sample_uniform(0.90, 1.12))
    records.extend(["local_brightness", "patch_color", "partial_blur", "gamma"])

    return image, records


def apply_object_texture_diversity(object_bgr, object_mask, difficulty_mode, domain):
    """
    foreground 텍스처에 약한 다양성을 주되 camouflage realism은 유지합니다.
    """
    image = object_bgr.copy()
    records = []
    strength = get_difficulty_config(difficulty_mode)["object_aug_strength"]
    mask_bool = object_mask > 0
    if np.count_nonzero(mask_bool) == 0:
        return image, records

    if GLOBAL_STATE["rng"].random() < 0.60:
        sigma = _sample_uniform(0.4, 1.0) * strength
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
        image[mask_bool] = blurred[mask_bool]
        records.append("local_blur")

    if GLOBAL_STATE["rng"].random() < 0.55:
        sharpened = cv2.addWeighted(image, 1.0 + 0.14 * strength, cv2.GaussianBlur(image, (0, 0), sigmaX=1.0), -0.14 * strength, 0.0)
        image[mask_bool] = np.clip(sharpened, 0, 255).astype(np.uint8)[mask_bool]
        records.append("sharpen")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2][mask_bool] = np.clip(hsv[:, :, 2][mask_bool] * _sample_uniform(0.92, 1.08), 0, 255)
    hsv[:, :, 1][mask_bool] = np.clip(hsv[:, :, 1][mask_bool] * _sample_uniform(0.94, 1.08), 0, 255)
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    records.extend(["contrast_perturb", "gamma_perturb"])

    if GLOBAL_STATE["rng"].random() < 0.45:
        image = _simulate_jpeg_artifact(image, 0.5 * strength)
        records.append("jpeg_artifact")

    return image, records


def compute_background_variant_signature(image):
    thumb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA)
    mean_value = float(np.mean(thumb))
    bits = (thumb > mean_value).astype(np.uint8).flatten()
    return "".join(str(int(bit)) for bit in bits[:128])


def load_existing_metadata_registry():
    registry = {}
    if not OUTPUT_METADATA_CSV.exists():
        return registry
    try:
        with OUTPUT_METADATA_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                bg_origin = row.get("background_origin", "")
                split_name = row.get("split_name", "")
                if bg_origin and split_name:
                    registry[bg_origin] = split_name
    except OSError:
        return registry
    return registry


def is_background_split_compatible(background_origin, split_name):
    registered = GLOBAL_STATE["background_split_registry"].get(background_origin)
    return registered is None or registered == split_name


def register_background_split(background_origin, split_name):
    GLOBAL_STATE["background_split_registry"][background_origin] = split_name


def is_background_variant_duplicate(background_origin, signature):
    seen = GLOBAL_STATE["background_variant_signatures"].setdefault(background_origin, set())
    if signature in seen:
        return True
    seen.add(signature)
    return False


def save_synthesis_metadata_csv():
    if not GLOBAL_STATE["metadata_rows"]:
        return
    fieldnames = sorted({key for row in GLOBAL_STATE["metadata_rows"] for key in row.keys()})
    OUTPUT_METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_METADATA_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(GLOBAL_STATE["metadata_rows"])


def save_debug_visualization(stem, background, alpha, matched_object, transformed_mask, x, y):
    if not DEBUG_VISUALIZE:
        return
    debug_dir = OUTPUT_DEBUG_ROOT / stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    alpha_vis = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    placed_object = np.zeros_like(background)
    obj_h, obj_w = matched_object.shape[:2]
    placed_object[y:y + obj_h, x:x + obj_w] = matched_object
    save_image_unicode(debug_dir / "background.png", background)
    save_image_unicode(debug_dir / "alpha.png", alpha_vis)
    save_image_unicode(debug_dir / "object.png", matched_object)
    save_image_unicode(debug_dir / "placed_object.png", placed_object)
    save_image_unicode(debug_dir / "mask.png", transformed_mask)


def resize_object_to_target_scale(image, mask, bg_h, bg_w, target_ratio):
    """
    resize 단계에서는 GT용 binary mask만 관리합니다.

    blending alpha는 별도 경계 완화 로직(build_alpha_from_mask)에서 만들고,
    여기서 반환하는 mask는 sharp binary 상태를 유지합니다.
    """
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
        targets = build_full_mode_targets_from_background_distribution()

    total_samples = 0
    for mode_targets in targets.values():
        total_samples += int(sum(mode_targets.values()))

    difficulty_plan = []
    if DIFFICULTY_SAMPLING_MODE == "mixed":
        difficulty_targets = _allocate_counts_by_weights(total_samples, DIFFICULTY_RATIOS)
        for difficulty_name, count in difficulty_targets.items():
            for _ in range(int(count)):
                difficulty_plan.append(difficulty_name)
        GLOBAL_STATE["rng"].shuffle(difficulty_plan)

    for mode_name in ["natural", "semi", "hard", "minimal"]:
        mode_targets = targets.get(mode_name, {})
        for domain_name in sorted(mode_targets.keys()):
            count = mode_targets[domain_name]
            for _ in range(count):
                if DIFFICULTY_SAMPLING_MODE == "mixed" and difficulty_plan:
                    difficulty_name = difficulty_plan.pop()
                else:
                    difficulty_name = DIFFICULTY_MODE
                schedule.append({"mode": mode_name, "domain": domain_name, "difficulty": difficulty_name})

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


def sample_candidate_positions(
    bg_h,
    bg_w,
    obj_h,
    obj_w,
    num_candidates,
    scale_bucket_name,
    domain,
    mode,
):
    """
    위치 후보 생성 단계에서는 hard constraint를 먼저 적용합니다.

    핵심 원칙:
    - 후보 생성 단계: size/domain 기반 위치 제약을 사용
    - 후보 평가 단계: 더 자연스러운 후보를 soft score로 비교

    따라서 여기서는 center_y band를 기준으로 후보를 우선 생성하되,
    후보가 너무 적으면 fallback 랜덤 샘플도 일부 허용합니다.
    """
    positions = []
    if obj_h >= bg_h or obj_w >= bg_w:
        return positions

    strictness = MODE_CONFIG[mode]["strictness"]
    allowed_min_center_y, allowed_max_center_y = get_effective_center_y_range(scale_bucket_name, domain)
    domain_policy = get_domain_spatial_policy(domain)

    max_x = max(0, bg_w - obj_w)
    max_y = max(0, bg_h - obj_h)
    margin_x = int(round(bg_w * OUTER_MARGIN_RATIO))
    margin_y = int(round(bg_h * OUTER_MARGIN_RATIO))

    if strictness == "strict":
        primary_attempts = num_candidates * 5
        fallback_attempts = max(2, num_candidates // 4)
    elif strictness == "medium":
        primary_attempts = num_candidates * 4
        fallback_attempts = max(3, num_candidates // 3)
    else:
        primary_attempts = num_candidates * 3
        fallback_attempts = max(4, num_candidates // 2)

    # 1차 후보: spatial policy가 허용하는 center_y band 안에서 우선 샘플링
    for _ in range(primary_attempts):
        center_x_ratio = GLOBAL_STATE["rng"].uniform(0.10, 0.90)
        center_y_ratio = GLOBAL_STATE["rng"].uniform(allowed_min_center_y, allowed_max_center_y)

        x = int(round(center_x_ratio * bg_w - obj_w * 0.5))
        y = int(round(center_y_ratio * bg_h - obj_h * 0.5))
        x = int(np.clip(x, 0, max_x))
        y = int(np.clip(y, 0, max_y))

        actual_center_x_ratio = (x + obj_w * 0.5) / float(bg_w)
        actual_center_y_ratio = (y + obj_h * 0.5) / float(bg_h)
        perspective_valid = allowed_min_center_y <= actual_center_y_ratio <= allowed_max_center_y
        band_center = (allowed_min_center_y + allowed_max_center_y) * 0.5
        center_distance = abs(actual_center_y_ratio - band_center)
        margin_score, margin_distance_ratio = compute_margin_score(x, y, obj_w, obj_h, bg_w, bg_h)

        positions.append(
            {
                "x": x,
                "y": y,
                "obj_w": obj_w,
                "obj_h": obj_h,
                "center_x_ratio": actual_center_x_ratio,
                "center_y_ratio": actual_center_y_ratio,
                "touches_margin": x < margin_x or y < margin_y or x + obj_w > bg_w - margin_x or y + obj_h > bg_h - margin_y,
                "allowed_min_center_y": allowed_min_center_y,
                "allowed_max_center_y": allowed_max_center_y,
                "perspective_valid": perspective_valid,
                "center_distance_from_allowed_band_center": center_distance,
                "margin_distance_ratio": margin_distance_ratio,
                "margin_score": margin_score,
                "domain_margin_penalty_strength": domain_policy["margin_penalty_strength"],
            }
        )

    # 2차 후보: spatial 제약 때문에 후보가 부족하면 기존 전역 랜덤 샘플링을 일부 허용
    # 단, fallback 후보는 perspective_valid=False로 표시해 나중에 로그에서 구분 가능하게 둡니다.
    for _ in range(fallback_attempts):
        if len(positions) >= primary_attempts + num_candidates:
            break
        x = GLOBAL_STATE["rng"].randint(0, max_x)
        y = GLOBAL_STATE["rng"].randint(0, max_y)
        actual_center_x_ratio = (x + obj_w * 0.5) / float(bg_w)
        actual_center_y_ratio = (y + obj_h * 0.5) / float(bg_h)
        band_center = (allowed_min_center_y + allowed_max_center_y) * 0.5
        center_distance = abs(actual_center_y_ratio - band_center)
        margin_score, margin_distance_ratio = compute_margin_score(x, y, obj_w, obj_h, bg_w, bg_h)

        positions.append(
            {
                "x": x,
                "y": y,
                "obj_w": obj_w,
                "obj_h": obj_h,
                "center_x_ratio": actual_center_x_ratio,
                "center_y_ratio": actual_center_y_ratio,
                "touches_margin": x < margin_x or y < margin_y or x + obj_w > bg_w - margin_x or y + obj_h > bg_h - margin_y,
                "allowed_min_center_y": allowed_min_center_y,
                "allowed_max_center_y": allowed_max_center_y,
                "perspective_valid": False,
                "center_distance_from_allowed_band_center": center_distance,
                "margin_distance_ratio": margin_distance_ratio,
                "margin_score": margin_score,
                "domain_margin_penalty_strength": domain_policy["margin_penalty_strength"],
            }
        )

    min_center_dist_px = max(12, int(round(min(obj_h, obj_w) * 0.20)))
    positions = deduplicate_positions(positions, min_center_dist_px)

    primary_positions = [item for item in positions if item["perspective_valid"]]
    fallback_positions = [item for item in positions if not item["perspective_valid"]]
    primary_positions.sort(
        key=lambda item: (
            item["touches_margin"],
            item["center_distance_from_allowed_band_center"],
            -item["margin_distance_ratio"],
        )
    )
    fallback_positions.sort(
        key=lambda item: (
            item["touches_margin"],
            item["center_distance_from_allowed_band_center"],
            -item["margin_distance_ratio"],
        )
    )

    ordered = primary_positions + fallback_positions
    if len(ordered) > num_candidates:
        ordered = ordered[:num_candidates]
    return ordered


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


def compute_object_background_match_score(
    object_bgr,
    object_mask,
    patch_bgr,
    y_ratio,
    mode,
    domain,
    scale_bucket_name,
    position_info,
):
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

    # placement_score는 hard constraint를 다시 강하게 반복하지 않도록
    # "조금 더 자연스러운 후보를 선호"하는 약한 prior만 사용합니다.
    edge_density = patch_texture["edge_density"]
    texture_suitability_score = np.clip((edge_density - 0.005) / 0.055, 0.0, 1.0)

    allowed_min_center_y = position_info["allowed_min_center_y"]
    allowed_max_center_y = position_info["allowed_max_center_y"]
    perspective_soft_score, expected_center_y, center_distance_from_expected = compute_soft_perspective_score(
        y_ratio,
        allowed_min_center_y,
        allowed_max_center_y,
    )

    margin_score = position_info.get("margin_score", 0.0)
    placement_score = (
        0.45 * texture_suitability_score
        + 0.35 * perspective_soft_score
        + 0.20 * margin_score
    )
    placement_score = float(np.clip(placement_score, 0.0, 1.0))

    weights = MODE_CONFIG[mode]["weights"]
    base_score = (
        weights["tone"] * tone_score
        + weights["texture"] * texture_score
        + weights["edge"] * edge_score
        + weights["placement"] * placement_score
    )

    artifact_metrics = compute_boundary_artifact_metrics(object_bgr, object_mask, patch_bgr)
    if artifact_metrics is None:
        artifact_metrics = {
            "artifact_penalty": 0.0,
            "boundary_brightness_jump": 0.0,
            "boundary_saturation_jump": 0.0,
            "boundary_gradient_jump": 0.0,
            "halo_risk_score": 0.0,
            "white_fringe_risk": 0.0,
            "white_matte_fraction": 0.0,
        }
    total_score = base_score - ARTIFACT_PENALTY_WEIGHT * artifact_metrics["artifact_penalty"]

    return {
        "total_score": float(total_score),
        "base_score": float(base_score),
        "tone_score": float(tone_score),
        "texture_score": float(texture_score),
        "edge_score": float(edge_score),
        "placement_score": float(placement_score),
        "perspective_soft_score": float(perspective_soft_score),
        "texture_suitability_score": float(texture_suitability_score),
        "margin_score": float(margin_score),
        "expected_center_y": float(expected_center_y),
        "allowed_min_center_y": float(allowed_min_center_y),
        "allowed_max_center_y": float(allowed_max_center_y),
        "center_y_ratio": float(y_ratio),
        "center_distance_from_expected": float(center_distance_from_expected),
        "scale_bucket_name": scale_bucket_name,
        "domain": domain,
        "brightness_diff": float(brightness_diff),
        "contrast_diff": float(contrast_diff),
        "saturation_diff": float(saturation_diff),
        "artifact_penalty": float(artifact_metrics["artifact_penalty"]),
        "boundary_brightness_jump": float(artifact_metrics["boundary_brightness_jump"]),
        "boundary_saturation_jump": float(artifact_metrics["boundary_saturation_jump"]),
        "boundary_gradient_jump": float(artifact_metrics["boundary_gradient_jump"]),
        "halo_risk_score": float(artifact_metrics["halo_risk_score"]),
        "white_fringe_risk": float(artifact_metrics["white_fringe_risk"]),
        "white_matte_fraction": float(artifact_metrics["white_matte_fraction"]),
    }


def select_best_object_and_position(background, background_path, object_candidates, mode, target_domain, difficulty_mode="hard"):
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

        augmented_bgr, object_aug_records = apply_object_texture_diversity(
            resized_bgr,
            resized_mask,
            difficulty_mode,
            target_domain,
        )

        obj_h, obj_w = resized_mask.shape[:2]
        positions = sample_candidate_positions(
            bg_h,
            bg_w,
            obj_h,
            obj_w,
            object_item["num_position_candidates"],
            object_item["scale_bucket_name"],
            target_domain,
            mode,
        )
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
                augmented_bgr,
                resized_mask,
                patch,
                pos["center_y_ratio"],
                mode,
                target_domain,
                object_item["scale_bucket_name"],
                pos,
            )
            if score is None:
                continue

            placement_adjust = -0.04 if pos["touches_margin"] else 0.01
            score["placement_score"] = max(0.0, min(1.0, score["placement_score"] + placement_adjust))
            weights = MODE_CONFIG[mode]["weights"]
            score["base_score"] = (
                weights["tone"] * score["tone_score"]
                + weights["texture"] * score["texture_score"]
                + weights["edge"] * score["edge_score"]
                + weights["placement"] * score["placement_score"]
            )
            score["total_score"] = (
                score["base_score"]
                - ARTIFACT_PENALTY_WEIGHT * score.get("artifact_penalty", 0.0)
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
                    "object_bgr": augmented_bgr,
                    "object_mask": resized_mask,
                    "x": pos["x"],
                    "y": pos["y"],
                    "score": score,
                    "patch": patch,
                    "background_key": background_key,
                    "position_info": pos,
                    "object_aug_records": list(object_aug_records),
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


def apply_difficulty_camouflage_pull(object_bgr, object_mask, patch_bgr, difficulty_mode):
    """
    difficulty가 높을수록 foreground를 background 쪽으로 조금 더 끌어당깁니다.
    """
    strength = get_difficulty_config(difficulty_mode)["camouflage_pull"]
    if strength <= 0.0:
        return object_bgr

    result = object_bgr.astype(np.float32).copy()
    obj_hsv = cv2.cvtColor(object_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask_bool = object_mask > 0
    if np.count_nonzero(mask_bool) == 0:
        return object_bgr

    obj_hsv[:, :, 2][mask_bool] = np.clip(
        obj_hsv[:, :, 2][mask_bool] * (1.0 - strength) + patch_hsv[:, :, 2][mask_bool] * strength,
        0,
        255,
    )
    obj_hsv[:, :, 1][mask_bool] = np.clip(
        obj_hsv[:, :, 1][mask_bool] * (1.0 - 0.8 * strength) + patch_hsv[:, :, 1][mask_bool] * (0.8 * strength),
        0,
        255,
    )
    result = cv2.cvtColor(obj_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result


def apply_blur_consistency_matching(object_bgr, object_mask, patch_bgr, difficulty_mode):
    """
    foreground의 경계 blur 수준을 patch의 blur 수준과 조금 더 가깝게 맞춥니다.
    """
    object_texture = compute_patch_texture_score(object_bgr)
    patch_texture = compute_patch_texture_score(patch_bgr)
    grad_gap = object_texture["grad_mean"] - patch_texture["grad_mean"]
    if grad_gap <= 6.0:
        return object_bgr

    strength = get_difficulty_config(difficulty_mode)["boundary_softness"]
    sigma = np.clip((grad_gap / 55.0) * 0.8 * strength, 0.0, 1.6)
    if sigma <= 0.05:
        return object_bgr

    softened = cv2.GaussianBlur(object_bgr, (0, 0), sigmaX=sigma)
    result = object_bgr.copy()
    edge_band = _build_boundary_band(object_mask, band_width=max(3, int(round(3 * strength))))
    edge_bool = edge_band > 0
    result[edge_bool] = softened[edge_bool]
    return result


def apply_post_blend_edge_smoothing(background, blended, alpha, x, y, difficulty_mode):
    """
    합성 후 경계 band만 아주 약하게 smoothing해 경계 단차를 줄입니다.
    """
    obj_h, obj_w = alpha.shape[:2]
    result = blended.copy()
    roi = result[y:y + obj_h, x:x + obj_w]
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    band = _build_boundary_band(alpha_u8, band_width=max(3, int(round(3 * get_difficulty_config(difficulty_mode)["boundary_softness"]))))
    if np.count_nonzero(band) == 0:
        return blended
    smooth_roi = cv2.GaussianBlur(roi, (0, 0), sigmaX=0.8)
    band_bool = band > 0
    roi[band_bool] = smooth_roi[band_bool]
    result[y:y + obj_h, x:x + obj_w] = roi
    return result


def apply_tone_matching(object_bgr, object_mask, patch_bgr, mode, difficulty_mode="hard"):
    """
    약한 global tone adjustment와 더 강한 local edge-band matching을 적용합니다.

    binary GT mask는 의도적으로 sharp하게 유지하고,
    실제로 시각적 보정은 합성 이미지 쪽에만 적용합니다.
    """
    config = MODE_CONFIG[mode]
    decontaminated_object = decontaminate_object_edge(object_bgr, object_mask, patch_bgr, mode)
    decontaminated_object = apply_difficulty_camouflage_pull(decontaminated_object, object_mask, patch_bgr, difficulty_mode)
    global_strength_scale = {
        "natural": 0.55,
        "semi": 0.70,
        "hard": 0.85,
        "minimal": 0.90,
    }.get(mode, 0.70)
    global_strength_scale *= (0.92 + 0.12 * get_difficulty_config(difficulty_mode)["camouflage_pull"] / max(0.10, DIFFICULTY_CONFIG["easy"]["camouflage_pull"]))
    brightness_alpha = GLOBAL_STATE["rng"].uniform(config["brightness_alpha"][0], config["brightness_alpha"][1]) * global_strength_scale
    contrast_alpha = GLOBAL_STATE["rng"].uniform(config["contrast_alpha"][0], config["contrast_alpha"][1]) * global_strength_scale
    saturation_alpha = GLOBAL_STATE["rng"].uniform(config["saturation_alpha"][0], config["saturation_alpha"][1]) * global_strength_scale

    result = decontaminated_object.astype(np.float32).copy()
    obj_stats = _masked_hsv_stats(decontaminated_object, object_mask)
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
    edge_params = get_edge_blend_params(object_mask, mode)
    boundary_band_width = max(2, edge_params["boundary_width_px"])
    band = _build_boundary_band(object_mask, band_width=boundary_band_width)
    band_bool = band > 0
    if np.count_nonzero(band_bool) > 0:
        local_alpha = {
            "natural": 0.22,
            "semi": 0.18,
            "hard": 0.14,
            "minimal": 0.10,
        }.get(mode, 0.16)
        matched_hsv = cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        bg_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        band_v = matched_hsv[:, :, 2]
        band_s = matched_hsv[:, :, 1]
        band_v[band_bool] = np.clip(band_v[band_bool] * (1.0 - local_alpha) + bg_hsv[:, :, 2][band_bool] * local_alpha, 0, 255)
        local_s_alpha = min(0.22, saturation_alpha + 0.03)
        band_s[band_bool] = np.clip(band_s[band_bool] * (1.0 - local_s_alpha) + bg_hsv[:, :, 1][band_bool] * local_s_alpha, 0, 255)
        matched_hsv[:, :, 2] = band_v
        matched_hsv[:, :, 1] = band_s
        matched = cv2.cvtColor(matched_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    return apply_blur_consistency_matching(matched.astype(np.uint8), object_mask, patch_bgr, difficulty_mode)


def build_alpha_from_mask(mask, mode, object_bgr=None, patch_bgr=None, difficulty_mode="hard"):
    blend_source_mask = shrink_mask_for_blending(mask, mode, object_bgr=object_bgr, patch_bgr=patch_bgr)
    params = get_edge_blend_params(blend_source_mask, mode)
    alpha = build_boundary_only_alpha(
        blend_source_mask,
        params["boundary_width_px"],
        params["blur_kernel_px"],
    )
    if object_bgr is not None and patch_bgr is not None:
        metrics = compute_boundary_artifact_metrics(object_bgr, blend_source_mask, patch_bgr)
        if metrics is not None:
            suppression = (
                0.06 * np.clip(metrics["white_fringe_risk"], 0.0, 1.0)
                + 0.04 * np.clip(metrics["white_matte_fraction"], 0.0, 1.0)
            )
            suppression *= get_difficulty_config(difficulty_mode)["boundary_softness"]
            suppression = float(np.clip(suppression, 0.0, 0.12))
            if suppression > 0.0:
                boundary_band = _build_boundary_band(blend_source_mask, band_width=max(2, params["boundary_width_px"]))
                boundary_bool = boundary_band > 0
                alpha[boundary_bool] *= (1.0 - suppression)
                interior_bool = cv2.erode(blend_source_mask, np.ones((3, 3), np.uint8), iterations=1) > 0
                alpha[interior_bool] = 1.0
                alpha[blend_source_mask == 0] = 0.0
    return np.clip(alpha, 0.0, 1.0)


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
        snow_files = collect_image_files(DEMO_BACKGROUND_ROOT)
        background_state["snow"] = {"paths": snow_files}
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
        f"split={sample_info.get('split_name', SYNTHESIS_SPLIT_NAME)} "
        f"difficulty={sample_info.get('difficulty_mode', DIFFICULTY_MODE)} "
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
        f"center_y={sample_info['score'].get('center_y_ratio', 0.0):.4f} "
        f"expected_center_y={sample_info['score'].get('expected_center_y', 0.0):.4f} "
        f"allowed_center_y=({sample_info['score'].get('allowed_min_center_y', 0.0):.4f},{sample_info['score'].get('allowed_max_center_y', 0.0):.4f}) "
        f"perspective_soft={sample_info['score'].get('perspective_soft_score', 0.0):.4f} "
        f"texture_suitability={sample_info['score'].get('texture_suitability_score', 0.0):.4f} "
        f"margin_score={sample_info['score'].get('margin_score', 0.0):.4f} "
        f"artifact_penalty={sample_info['score'].get('artifact_penalty', 0.0):.4f} "
        f"halo_risk_score={sample_info['score'].get('halo_risk_score', 0.0):.4f} "
        f"white_fringe_risk={sample_info['score'].get('white_fringe_risk', 0.0):.4f} "
        f"white_matte_fraction={sample_info['score'].get('white_matte_fraction', 0.0):.4f} "
        f"boundary_brightness_jump={sample_info['score'].get('boundary_brightness_jump', 0.0):.2f} "
        f"boundary_saturation_jump={sample_info['score'].get('boundary_saturation_jump', 0.0):.2f} "
        f"boundary_gradient_jump={sample_info['score'].get('boundary_gradient_jump', 0.0):.2f} "
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
    print(f"  metadata_csv: {OUTPUT_METADATA_CSV}")
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
    GLOBAL_STATE["background_split_registry"] = load_existing_metadata_registry()
    GLOBAL_STATE["metadata_rows"] = []

    OUTPUT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASK_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXT_ROOT.mkdir(parents=True, exist_ok=True)
    if DEBUG_VISUALIZE:
        OUTPUT_DEBUG_ROOT.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    max_attempts = max(100, len(schedule) * 8)
    attempt = 0

    while schedule and attempt < max_attempts:
        attempt += 1
        target = schedule.pop(0)
        mode = target["mode"]
        domain = target["domain"]
        difficulty_mode = target.get("difficulty", DIFFICULTY_MODE)
        scale_bucket = sample_scale_bucket(mode)
        target_ratio = adjust_target_area_ratio_for_difficulty(
            compute_target_area_ratio(scale_bucket),
            difficulty_mode,
        )

        background_path = sample_background_for_domain(domain, mode)
        if background_path is None:
            print(f"[WARN] No background available for domain={domain}, skipping")
            skipped += 1
            continue
        background_origin = background_path.stem
        if not is_background_split_compatible(background_origin, SYNTHESIS_SPLIT_NAME):
            print(f"[WARN] Background split conflict: origin={background_origin} existing_split={GLOBAL_STATE['background_split_registry'].get(background_origin)}")
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
        background, anti_mem_records = apply_anti_memorization_background_transform(background, domain, difficulty_mode)
        background, domain_aug_records = apply_domain_aware_background_augmentation(background, domain, difficulty_mode)
        background_signature = compute_background_variant_signature(background)
        if is_background_variant_duplicate(background_origin, background_signature):
            print(f"[WARN] Near-duplicate background variant skipped: origin={background_origin}")
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
            staged["scale_bucket_name"] = scale_bucket["name"]
            staged_candidates.append(staged)

        best = select_best_object_and_position(
            background,
            background_path,
            staged_candidates,
            mode,
            domain,
            difficulty_mode=difficulty_mode,
        )
        if best is None:
            print(f"[WARN] No valid object-position match for domain={domain}, mode={mode}")
            skipped += 1
            continue

        blend_source_mask = shrink_mask_for_blending(
            best["object_mask"],
            mode,
            object_bgr=best["object_bgr"],
            patch_bgr=best["patch"],
        )
        matched_object = apply_tone_matching(
            best["object_bgr"],
            blend_source_mask,
            best["patch"],
            mode,
            difficulty_mode=difficulty_mode,
        )
        alpha = build_alpha_from_mask(
            best["object_mask"],
            mode,
            object_bgr=best["object_bgr"],
            patch_bgr=best["patch"],
            difficulty_mode=difficulty_mode,
        )
        synthetic = blend_object(background, matched_object, alpha, best["x"], best["y"])
        synthetic = apply_post_blend_edge_smoothing(background, synthetic, alpha, best["x"], best["y"], difficulty_mode)
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
        register_background_split(background_origin, SYNTHESIS_SPLIT_NAME)
        GLOBAL_STATE["scale_hist"][scale_bucket["name"]] = GLOBAL_STATE["scale_hist"].get(scale_bucket["name"], 0) + 1
        GLOBAL_STATE["saved_images"].append(str(image_path))
        GLOBAL_STATE["saved_txts"].append(str(txt_path))
        if mask_path is not None:
            GLOBAL_STATE["saved_masks"].append(str(mask_path))

        GLOBAL_STATE["metadata_rows"].append(
            {
                "sample_stem": stem,
                "split_name": SYNTHESIS_SPLIT_NAME,
                "execution_mode": EXECUTION_MODE,
                "difficulty_mode": difficulty_mode,
                "mode": mode,
                "domain": domain,
                "background_origin": background_origin,
                "background_path": str(background_path),
                "background_signature": background_signature,
                "object_id": best["object_item"]["id"],
                "object_domain": object_domain,
                "scale_bucket": scale_bucket["name"],
                "target_ratio": f"{target_ratio:.6f}",
                "x": best["x"],
                "y": best["y"],
                "score_total": f"{best['score']['total_score']:.6f}",
                "score_tone": f"{best['score']['tone_score']:.6f}",
                "score_texture": f"{best['score']['texture_score']:.6f}",
                "score_edge": f"{best['score']['edge_score']:.6f}",
                "score_placement": f"{best['score']['placement_score']:.6f}",
                "artifact_penalty": f"{best['score'].get('artifact_penalty', 0.0):.6f}",
                "halo_risk_score": f"{best['score'].get('halo_risk_score', 0.0):.6f}",
                "white_fringe_risk": f"{best['score'].get('white_fringe_risk', 0.0):.6f}",
                "white_matte_fraction": f"{best['score'].get('white_matte_fraction', 0.0):.6f}",
                "background_augments": "|".join(anti_mem_records + domain_aug_records),
                "object_augments": "|".join(best.get("object_aug_records", [])),
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "txt_path": str(txt_path),
            }
        )

        created += 1
        _print_sample_debug(
            created,
            {
                "mode": mode,
                "domain": domain,
                "split_name": SYNTHESIS_SPLIT_NAME,
                "difficulty_mode": difficulty_mode,
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
        save_debug_visualization(stem, background, alpha, matched_object, transformed_mask, best["x"], best["y"])

    print(f"\n[SUMMARY] created={created} skipped={skipped} remaining_schedule={len(schedule)} attempts={attempt}")
    if EXECUTION_MODE == "DEMO":
        print("[SUMMARY] Demo mode uses snow-only backgrounds:")
        snow_paths = GLOBAL_STATE["background_state"].get("snow", {}).get("paths", [])
        print(f"  snow: {len(snow_paths)} file(s)")
        if snow_paths:
            print(f"  first: {snow_paths[0]}")
            print(f"  last: {snow_paths[-1]}")
    save_synthesis_metadata_csv()
    _print_scale_histogram()
    _print_output_summary()


if __name__ == "__main__":
    main()
