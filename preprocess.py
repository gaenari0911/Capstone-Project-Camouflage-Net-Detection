import cv2
import os
import glob
from pathlib import Path

def convert_mask_to_yolo(mask_dir, output_dir, class_id=0):
    """
    Binary Mask 이미지를 YOLO Polygon 포맷(.txt)으로 변환합니다. (GT to label)
    """
    os.makedirs(output_dir, exist_ok=True)
    mask_paths = glob.glob(os.path.join(mask_dir, '*.*')) # .png, .jpg 등
    
    for mask_path in mask_paths:
        # 1. 흑백 이미지로 읽기
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # 2. 이진화 (안전장치: 픽셀값이 127보다 크면 255(흰색), 아니면 0(검은색))
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 3. 윤곽선(Contours) 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        H, W = mask.shape
        polygons = []
        
        for contour in contours:
            # 너무 작은 노이즈 영역은 무시 (면적 10 미만)
            if cv2.contourArea(contour) < 10:
                continue
                
            # 4. 좌표를 1차원 배열로 펴고 0~1 사이로 정규화
            contour = contour.flatten().tolist()
            polygon = []
            for i in range(0, len(contour), 2):
                x = contour[i] / W
                y = contour[i + 1] / H
                polygon.extend([f"{x:.6f}", f"{y:.6f}"])
                
            # 점이 3개(좌표 6개) 이상이어야 유효한 다각형이 됨
            if len(polygon) >= 6:
                line = f"{class_id} " + " ".join(polygon)
                polygons.append(line)
                
        # 5. .txt 파일로 저장 (이미지 파일명과 동일하게)
        file_stem = Path(mask_path).stem
        txt_path = os.path.join(output_dir, f"{file_stem}.txt")
        
        with open(txt_path, 'w') as f:
            f.write('\n'.join(polygons))

    print(f"변환 완료 저장된 폴더: {output_dir}")

# --- 실행부 ---
# 훈련 데이터 변환
convert_mask_to_yolo(
    mask_dir='./datasets/acd1k/train/label_images', # 기존 마스크 이미지 폴더
    output_dir='./datasets/acd1k/train/labels',     # 생성될 txt 파일 폴더
    class_id=0 # 위장 객체 클래스 번호
)

# 검증 데이터 변환
convert_mask_to_yolo('./datasets/acd1k/val/label_images', 
                     './datasets/acd1k/val/labels', 
                     0
)