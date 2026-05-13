import cv2
from ultralytics import YOLO

def test_and_infer():
    # 1. 학습된 최고 성능의 가중치 불러오기
    # 학습 코드에서 설정한 project와 name 경로를 따라갑니다.
    weight_path = '../result/dual_head_run/weights/best.pt'
    
    print(f"[{weight_path}] 가중치를 불러옵니다...")
    model = YOLO(weight_path)

    # 2. 정량적 평가 (Validation)
    print("\n--- 검증 데이터셋(Test/Val) 평가 시작 ---")
    metrics = model.val(
        data='./data.yaml',
        split='val',   # acd1k.yaml에 정의된 val 또는 test 데이터
        imgsz=640,
        device=0,
        batch=16
    )
    
    # 평가 지표 출력
    print(f"Box mAP50-95: {metrics.box.map:.4f}")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")

    # 3. 실제 이미지 추론 (Inference)
    print("\n--- 실제 이미지 추론 및 시각화 ---")
    
    # 추론해볼 테스트 이미지 경로 지정
    test_image_path = '../datasets/acd1k/val/images/image501.jpg' 
    
    results = model.predict(
        source=test_image_path,
        conf=0.25,        # 위장 객체 특성상 Confidence 임계값을 약간 낮춰서 숨은 객체 탐지
        iou=0.45,         # NMS 임계값
        retina_masks=True,# 고해상도 마스크 렌더링
        save=True,        # 결과 이미지를 runs/predict 폴더에 저장
        show=True         # 실행 시 화면에 팝업으로 결과 출력
    )
    
    # 결과가 저장된 경로 안내
    for result in results:
        print(f"결과 이미지가 저장되었습니다: {result.save_dir}")

if __name__ == '__main__':
    test_and_infer()
    