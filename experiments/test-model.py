import torch
from ultralytics import YOLO

def test_custom_model():
    print("모델 초기화 중...")
    model = YOLO("yolov8-seg-custom.yaml")
    
    model.model.train()
    
    dummy_input = torch.randn(2, 3, 640, 640)
    
    print("가짜 데이터 모델 통과 중...")
    preds = model.model(dummy_input)
    
    print("\n모델 통과 성공")
    
    # preds가 딕셔너리인지 튜플인지에 따라 다르게 처리
    if isinstance(preds, dict):
        print(f"반환된 데이터 타입: Dictionary")
        print(f"딕셔너리 키 목록: {list(preds.keys())}")
        
        # 우리가 추가했던 'boundary' 키로 예측값을 꺼냅니다.
        pb = preds['boundary'] 
    else:
        p, pb = preds
        print(f"Mask & Box 예측(p) 개수: {len(p)}")
        
    print(f"\nBoundary 예측(pb) 개수: {len(pb)}")
    for i, b_pred in enumerate(pb):
        print(f"Boundary 피처맵 {i}의 형태(shape): {b_pred.shape}")

if __name__ == "__main__":
    test_custom_model()