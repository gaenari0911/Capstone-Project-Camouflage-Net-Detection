import argparse
from pathlib import Path
import random
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns

# OpenCV가 한글 경로를 읽을 때 내는 경고를 억제합니다.
try:
    if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    elif hasattr(cv2, 'setLogLevel'):
        cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except Exception:
    pass

# 시스템에서 사용할 수 있는 한글 폰트 탐색
def _get_available_korean_font():
    candidates = [
        "Malgun Gothic",
        "맑은 고딕",
        "NanumGothic",
        "Nanum Gothic",
        "NanumGothicCoding",
        "AppleGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Gulim",
        "Dotum",
        "Batang",
    ]
    installed = {f.name: f.fname for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            return name
    for f in font_manager.fontManager.ttflist:
        lname = f.name.lower()
        if any(prefix in lname for prefix in ["malgun", "nanum", "gothic", "noto", "gulim", "dotum", "batang"]):
            return f.name
    return None

font_name = _get_available_korean_font()
if font_name:
    matplotlib.rcParams['font.family'] = font_name
    matplotlib.rcParams['font.sans-serif'] = [font_name]
else:
    matplotlib.rcParams['font.family'] = ['sans-serif']
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    print("경고: 한글 폰트를 찾을 수 없습니다. Malgun Gothic 또는 NanumGothic을 설치해 주세요.")
matplotlib.rcParams['axes.unicode_minus'] = False


# ----------------------------- 설정값 -----------------------------
OUTPUT_IMAGE_ROOT = Path(r"G:\내 드라이브\합성_데이터셋_2차\images")
REAL_IMAGE_ROOT = Path(r"D:\졸업작품\Share_git\Capstone-Project-Camouflage-Net-Detection\Dataset\images\train")
REAL_MASK_ROOT = Path(r"D:\졸업작품\Share_git\Capstone-Project-Camouflage-Net-Detection\Dataset\labels\train")
SYNTH_MASK_ROOT = Path(r"G:\내 드라이브\합성_데이터셋_2차\masks")
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp"]
MASK_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]
RANDOM_SEED = 42
FIGURE_DPI = 300
OUTPUT_FIG_DIR = Path("evaluation_outputs_2차")
DEFAULT_MAX_IMAGE_SIZE = 1024
MAX_TSNE_POINTS = 500
MAX_FFT_SAMPLES = 200
MAX_EDGE_GRADIENT_SAMPLES = 200000


def get_image_paths(root_path: Path):
    """지정된 루트 디렉토리에서 유효한 이미지 파일 경로를 모두 수집합니다."""
    image_paths = []
    for ext in VALID_EXTENSIONS:
        image_paths.extend(sorted(root_path.rglob(f"*{ext}")))
    return [p for p in image_paths if p.is_file()]


def sample_balanced_pairs(real_paths, synth_paths):
    """두 집합의 개수를 맞추기 위해 작은 쪽 N에 맞춰 랜덤 샘플링합니다."""
    n_real = len(real_paths)
    n_synth = len(synth_paths)
    if n_real == 0 or n_synth == 0:
        raise ValueError("실제 이미지 또는 합성 이미지 경로에 이미지가 하나도 없습니다.")
    if n_real == n_synth:
        return real_paths, synth_paths

    random.seed(RANDOM_SEED)
    if n_real < n_synth:
        synth_paths = random.sample(synth_paths, n_real)
    else:
        real_paths = random.sample(real_paths, n_synth)
    return real_paths, synth_paths


def prepare_output_directory(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_gray_image(path: Path, max_size: int = None):
    # 먼저 일반적인 cv2.imread 시도
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    # Windows에서 경로에 한글(유니코드) 등이 포함되면 OpenCV의 imread가 None을 반환할 수 있음.
    # 이 경우 numpy.fromfile + cv2.imdecode를 사용하여 바이너리로 읽어들임.
    if image is None:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size:
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            image = None
    if image is None:
        raise RuntimeError(f"이미지를 로드할 수 없습니다: {path}")
    if max_size is not None:
        h, w = image.shape[:2]
        scale = min(1.0, max_size / max(h, w))
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_gradient_magnitude(gray_image):
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    return magnitude


def resolve_mask_path(image_path: Path, mask_root: Path):
    """이미지 파일 이름과 동일한 스템을 가진 마스크 경로를 찾습니다."""
    candidate_stem = image_path.stem
    for ext in MASK_EXTENSIONS:
        candidate_path = mask_root / f"{candidate_stem}{ext}"
        if candidate_path.exists():
            return candidate_path
    return None


def load_mask_image(mask_path: Path):
    """마스크 이미지를 흑백으로 로드하고 이진화합니다."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        try:
            data = np.fromfile(str(mask_path), dtype=np.uint8)
            if data.size:
                mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        except Exception:
            mask = None
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def get_boundary_mask_from_mask(binary_mask):
    """마스크 테두리 주변 픽셀을 추출하기 위한 경계 마스크를 생성합니다."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    boundary = cv2.morphologyEx(binary_mask, cv2.MORPH_GRADIENT, kernel)
    if np.count_nonzero(boundary) < 100:
        boundary = cv2.Canny(binary_mask, 50, 150)
    return (boundary > 0).astype(np.uint8)


def _resize_mask_to_image_shape(mask, image_shape):
    if mask.shape[:2] != image_shape[:2]:
        return cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def get_synthetic_boundary_mask(gray_image, image_path: Path):
    """합성 이미지의 경계부를 마스크를 이용해 추출합니다."""
    mask_path = resolve_mask_path(image_path, SYNTH_MASK_ROOT)
    if mask_path is not None:
        binary_mask = load_mask_image(mask_path)
        if binary_mask is not None:
            binary_mask = _resize_mask_to_image_shape(binary_mask, gray_image.shape)
            return get_boundary_mask_from_mask(binary_mask)

    # 마스크가 없거나 로드할 수 없을 때 fallback
    blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)
    diff = cv2.absdiff(gray_image, blurred)
    _, strong_diff = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(strong_diff, cv2.MORPH_CLOSE, kernel, iterations=2)
    gradient = cv2.morphologyEx(closed, cv2.MORPH_GRADIENT, kernel)
    if np.count_nonzero(gradient) < 100:
        gradient = cv2.Canny(gray_image, 50, 150)
    return gradient // 255


def get_real_boundary_mask(gray_image, image_path: Path):
    """실제 이미지에서 마스크 기반 경계부를 추출합니다."""
    mask_path = resolve_mask_path(image_path, REAL_MASK_ROOT)
    if mask_path is not None:
        binary_mask = load_mask_image(mask_path)
        if binary_mask is not None:
            binary_mask = _resize_mask_to_image_shape(binary_mask, gray_image.shape)
            return get_boundary_mask_from_mask(binary_mask)

    # 마스크가 없으면 Canny 기반 fallback
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny = cv2.Canny(blurred, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(canny, kernel, iterations=1)
    return dilated // 255


def _sample_values(values, max_samples=MAX_EDGE_GRADIENT_SAMPLES):
    if values.size <= max_samples:
        return values
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.choice(values.size, size=max_samples, replace=False)
    return values[indices]


def extract_boundary_gradient_values(gray_image, boundary_mask):
    magnitude = compute_gradient_magnitude(gray_image)
    values = magnitude[boundary_mask.astype(bool)]
    if values.size == 0:
        values = magnitude.flatten()
    return values


def plot_edge_gradient_distribution(real_values, synth_values, output_dir: Path):
    real_values = _sample_values(real_values)
    synth_values = _sample_values(synth_values)
    plt.figure(figsize=(11, 6), dpi=FIGURE_DPI)
    sns.set_style("whitegrid")
    sns.histplot(real_values, stat="density", bins=100, color="#1f77b4", alpha=0.35, label="Real Images")
    sns.histplot(synth_values, stat="density", bins=100, color="#d62728", alpha=0.35, label="Synthetic Images")
    plt.title("경계부 그라디언트 강도 분포 비교 (Edge Gradient Analysis)", fontsize=18)
    plt.xlabel("Gradient Magnitude", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    save_path = output_dir / "edge_gradient_distribution.png"
    plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.show()
    plt.close()
    return save_path


def get_image_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_inception_feature_extractor(device):
    try:
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights, aux_logits=True)
    except AttributeError:
        model = models.inception_v3(pretrained=True, aux_logits=True)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model


def extract_inception_features(image_paths, model, device, batch_size=16):
    transform = get_image_transform()
    features = []
    skipped_count = 0
    
    with torch.no_grad():
        batch_images = []
        for idx, path in enumerate(image_paths, start=1):
            # cv2.imread 시도, 실패 시 fallback
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                try:
                    data = np.fromfile(str(path), dtype=np.uint8)
                    if data.size:
                        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
                except Exception:
                    image = None
            
            if image is None:
                skipped_count += 1
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image)
            batch_images.append(image_tensor)
            
            # 배치가 가득 찼거나 마지막 이미지이면 처리
            if len(batch_images) == batch_size or idx == len(image_paths):
                if batch_images:  # 배치가 비어있지 않으면
                    batch_tensor = torch.stack(batch_images).to(device)
                    batch_features = model(batch_tensor)
                    features.append(batch_features.cpu().numpy())
                    batch_images = []
    
    if not features:
        raise RuntimeError(f"로드된 이미지가 없습니다. {skipped_count}/{len(image_paths)}개 건너뜀. 이미지 경로를 확인하세요.")
    
    if skipped_count > 0:
        print(f"경고: {skipped_count}/{len(image_paths)}개 이미지를 로드할 수 없었습니다.")
    
    return np.vstack(features)


def calculate_fid_score(features_real, features_synth):
    mu_real = np.mean(features_real, axis=0)
    mu_synth = np.mean(features_synth, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_synth = np.cov(features_synth, rowvar=False)
    diff = mu_real - mu_synth
    covmean = compute_covariance_sqrt(sigma_real.dot(sigma_synth))
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_synth - 2 * covmean)
    return float(np.real(fid))


def compute_covariance_sqrt(matrix):
    """대칭 행렬의 행렬제곱근을 고유값 분해로 계산합니다."""
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    return (eigenvectors * sqrt_eigenvalues).dot(eigenvectors.T)


def plot_tsne(features_real, features_synth, fid_score, output_dir: Path):
    n_real = features_real.shape[0]
    n_synth = features_synth.shape[0]
    labels = np.array([0] * n_real + [1] * n_synth)
    features = np.vstack([features_real, features_synth])

    if features.shape[0] > 2 * MAX_TSNE_POINTS:
        real_idx = np.random.choice(n_real, MAX_TSNE_POINTS, replace=False)
        synth_idx = np.random.choice(n_synth, MAX_TSNE_POINTS, replace=False)
        features = np.vstack([features_real[real_idx], features_synth[synth_idx]])
        labels = np.array([0] * MAX_TSNE_POINTS + [1] * MAX_TSNE_POINTS)

    tsne = TSNE(n_components=2, init="pca", random_state=RANDOM_SEED, learning_rate="auto", metric="cosine")
    embedded = tsne.fit_transform(features)
    plt.figure(figsize=(12, 9), dpi=FIGURE_DPI)
    plt.scatter(embedded[labels == 0, 0], embedded[labels == 0, 1], c="#1f77b4", label="Real Images", alpha=0.45, s=28)
    plt.scatter(embedded[labels == 1, 0], embedded[labels == 1, 1], c="#d62728", label="Synthetic Images", alpha=0.45, s=28)
    plt.title(f"t-SNE Feature Space (FID: {fid_score:.2f})", fontsize=18)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_path = output_dir / "tsne_feature_space_fid.png"
    plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.show()
    plt.close()
    return save_path


def fft_power_spectrum(gray_image):
    f = np.fft.fft2(gray_image.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift) ** 2
    power = np.log1p(magnitude)
    return power


def radial_profile(power, center=None, num_bins=100):
    y, x = np.indices(power.shape)
    if center is None:
        center = np.array([(power.shape[0] - 1) / 2.0, (power.shape[1] - 1) / 2.0])
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    r_flat = r.flatten()
    power_flat = power.flatten()
    max_radius = np.max(r_flat)
    bin_edges = np.linspace(0.0, max_radius, num_bins + 1)
    radial_mean = np.zeros(num_bins, dtype=np.float64)
    for i in range(num_bins):
        mask = (r_flat >= bin_edges[i]) & (r_flat < bin_edges[i + 1])
        if np.any(mask):
            radial_mean[i] = np.mean(power_flat[mask])
        else:
            radial_mean[i] = 0.0
    return radial_mean, bin_edges[1:]


def plot_frequency_analysis(real_spectrum, synth_spectrum, real_profile, synth_profile, output_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=FIGURE_DPI)
    sns.set_style("darkgrid")

    vmax = max(real_spectrum.max(), synth_spectrum.max())
    axes[0, 0].imshow(real_spectrum, cmap="gray", vmin=0, vmax=vmax)
    axes[0, 0].set_title("Real Image Average Power Spectrum", fontsize=16)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(synth_spectrum, cmap="gray", vmin=0, vmax=vmax)
    axes[0, 1].set_title("Synthetic Image Average Power Spectrum", fontsize=16)
    axes[0, 1].axis("off")

    axes[1, 0].plot(real_profile, label="Real Images", color="#1f77b4", linewidth=2.5)
    axes[1, 0].plot(synth_profile, label="Synthetic Images", color="#d62728", linewidth=2.5)
    axes[1, 0].set_title("Radial Frequency Profile (Center -> High Frequency)", fontsize=16)
    axes[1, 0].set_xlabel("Radial Frequency Bin", fontsize=14)
    axes[1, 0].set_ylabel("Average Power (log scale)", fontsize=14)
    axes[1, 0].legend(fontsize=12)
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)

    axes[1, 1].axis("off")
    fig.suptitle("주파수 도메인 스펙트럼 비교", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = output_dir / "frequency_spectrum_analysis.png"
    plt.savefig(save_path, dpi=FIGURE_DPI)
    plt.show()
    plt.close()
    return save_path


def compute_average_spectrum_and_profile(image_paths, max_samples=None, fixed_size=512):
    """모든 이미지를 같은 크기로 리사이즈한 후 평균 스펙트럼과 레이디얼 프로필을 계산합니다."""
    if max_samples is not None and len(image_paths) > max_samples:
        image_paths = random.sample(image_paths, max_samples)
    
    spectrum_sum = None
    profile_list = []
    
    for path in image_paths:
        gray = load_gray_image(path, max_size=fixed_size)
        # 모든 이미지를 고정 크기(fixed_size x fixed_size)로 리사이즈하여 스펙트럼 크기를 통일
        gray_resized = cv2.resize(gray, (fixed_size, fixed_size), interpolation=cv2.INTER_AREA)
        spectrum = fft_power_spectrum(gray_resized)
        
        if spectrum_sum is None:
            spectrum_sum = np.zeros_like(spectrum, dtype=np.float64)
        spectrum_sum += spectrum
        
        profile, _ = radial_profile(spectrum, center=None, num_bins=120)
        profile_list.append(profile)
    
    spectrum_avg = spectrum_sum / len(image_paths)
    profile_avg = np.mean(np.vstack(profile_list), axis=0)
    return spectrum_avg, profile_avg


def run_edge_gradient_analysis(real_paths, synth_paths, max_size=DEFAULT_MAX_IMAGE_SIZE):
    real_values = []
    synth_values = []

    for idx, path in enumerate(real_paths, start=1):
        gray = load_gray_image(path, max_size=max_size)
        mask = get_real_boundary_mask(gray, path)
        values = extract_boundary_gradient_values(gray, mask)
        real_values.append(values)
        if idx % 100 == 0:
            print(f"  [진행] 실제 이미지 {idx}/{len(real_paths)} 처리 중...")

    for idx, path in enumerate(synth_paths, start=1):
        gray = load_gray_image(path, max_size=max_size)
        mask = get_synthetic_boundary_mask(gray, path)
        values = extract_boundary_gradient_values(gray, mask)
        synth_values.append(values)
        if idx % 100 == 0:
            print(f"  [진행] 합성 이미지 {idx}/{len(synth_paths)} 처리 중...")

    print("  [완료] 모든 경계부 gradient 값을 수집했습니다. 시각화로 넘어갑니다...")
    real_values = np.concatenate([v.flatten() for v in real_values])
    synth_values = np.concatenate([v.flatten() for v in synth_values])
    return real_values, synth_values


def run_fid_and_tsne(real_paths, synth_paths, device):
    feature_extractor = build_inception_feature_extractor(device)
    features_real = extract_inception_features(real_paths, feature_extractor, device)
    features_synth = extract_inception_features(synth_paths, feature_extractor, device)
    fid_score = calculate_fid_score(features_real, features_synth)
    return features_real, features_synth, fid_score


def run_frequency_analysis(real_paths, synth_paths):
    real_spectrum, real_profile = compute_average_spectrum_and_profile(real_paths, max_samples=MAX_FFT_SAMPLES)
    synth_spectrum, synth_profile = compute_average_spectrum_and_profile(synth_paths, max_samples=MAX_FFT_SAMPLES)
    return real_spectrum, synth_spectrum, real_profile, synth_profile


def main():
    parser = argparse.ArgumentParser(description="Synthetic vs Real Domain Validation Script")
    parser.add_argument("--real_root", type=str, default=str(REAL_IMAGE_ROOT), help="Real image root folder")
    parser.add_argument("--synth_root", type=str, default=str(OUTPUT_IMAGE_ROOT), help="Synthetic image root folder")
    parser.add_argument("--save_dir", type=str, default=str(OUTPUT_FIG_DIR), help="저장할 출력 디렉토리")
    parser.add_argument(
        "--analysis",
        type=str,
        default="all",
        choices=["all", "edge", "fid_tsne", "frequency"],
        help="실행할 분석 선택: all(모두), edge(경계부 그라디언트), fid_tsne(FID & t-SNE), frequency(주파수 스펙트럼)"
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIZE,
        help="경계부 및 주파수 분석에서 사용할 최대 이미지 긴 변 길이 (기본값 1024)"
    )
    args = parser.parse_args()

    real_root = Path(args.real_root)
    synth_root = Path(args.synth_root)
    output_dir = prepare_output_directory(Path(args.save_dir))
    analysis_choice = args.analysis.lower()
    max_size = args.max_size

    print("[1/4] 이미지 경로 수집 중...")
    real_paths = get_image_paths(real_root)
    synth_paths = get_image_paths(synth_root)
    print(f"실제 이미지 개수: {len(real_paths)}")
    print(f"합성 이미지 개수: {len(synth_paths)}")

    real_paths, synth_paths = sample_balanced_pairs(real_paths, synth_paths)
    print(f"균형 샘플링 후 데이터 개수: {len(real_paths)} (한쪽 기준)")

    if analysis_choice in ["all", "edge"]:
        print("[2/4] 경계부 그라디언트 분석 실행 중...")
        real_values, synth_values = run_edge_gradient_analysis(real_paths, synth_paths, max_size=max_size)
        edge_plot_path = plot_edge_gradient_distribution(real_values, synth_values, output_dir)
        print(f"경계부 분석 결과 저장: {edge_plot_path}")

    if analysis_choice in ["all", "fid_tsne"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[3/4] FID 및 t-SNE 피처 분석 실행 중... (Device: {device})")
        features_real, features_synth, fid_score = run_fid_and_tsne(real_paths, synth_paths, device)
        tsne_plot_path = plot_tsne(features_real, features_synth, fid_score, output_dir)
        print(f"t-SNE 및 FID 시각화 저장: {tsne_plot_path}")

    if analysis_choice in ["all", "frequency"]:
        print("[4/4] 주파수 스펙트럼 분석 실행 중...")
        real_spectrum, synth_spectrum, real_profile, synth_profile = run_frequency_analysis(real_paths, synth_paths)
        freq_plot_path = plot_frequency_analysis(real_spectrum, synth_spectrum, real_profile, synth_profile, output_dir)
        print(f"주파수 분석 결과 저장: {freq_plot_path}")

    print("\n===== 평가 완료 =====")

    print(f"산출물 디렉토리: {output_dir.absolute()}")
    print("- edge_gradient_distribution.png")
    print("- tsne_feature_space_fid.png")
    print("- frequency_spectrum_analysis.png")


if __name__ == "__main__":
    main()
