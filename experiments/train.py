import os
from ultralytics import YOLO

def train_model():
    print("--- Dual-Head YOLOv8-seg 학습 시작 ---")
    
    # 1. 모델 초기화
    # 커스텀 YAML로 뼈대를 만들고, 사전 학습된 가중치(Backbone 용)를 덮어씌웁니다.
    model = YOLO('./yolov8-seg-custom.yaml').load('yolov8m-seg.pt')

    # 모델 구조가 우리가 설계한 대로 잡혔는지 확인 (맨 마지막에 DualHeadSegment가 보여야 함)
    model.info()

    # 2. 학습 파라미터 설정 및 실행
    # Windows 환경에서 DataLoader 병렬 처리를 위해 workers 조절이 필요할 수 있습니다.
    results = model.train(
        data='./data.yaml', # 데이터셋 설정 파일 경로
        project='../../../result', # 결과가 저장될 최상위 폴더
        name='cod_train',          # 이번 실험의 폴더명
        
        # --- 주요 하이퍼파라미터 ---
        epochs=150,                    # 학습 반복 횟수
        imgsz=640,                     # 이미지 입력 해상도 (위장 객체는 해상도가 높을수록 유리함)
        batch=16,                      # 12GB VRAM 환경에 맞춰 안정적으로 설정 (Out of Memory 발생 시 8로 하향)
        device=0,                      # 첫 번째 GPU 사용
        workers=4,                     # 데이터 로딩 스레드 수
        
        # --- 위장 객체 탐지(COD) 특화 설정 ---
        optimizer='AdamW',             # 복잡한 Loss(Boundary+Mask) 최적화에 유리
        lr0=0.001,                     # 초기 학습률 (기본값보다 약간 낮춰 안정성 확보)
        mask_ratio=1,                  # 마스크 다운샘플링 비율 (1=원본 해상도 유지하여 경계선 보존)
        overlap_mask=False,            # 인스턴스가 겹칠 때 마스크를 덮어쓰지 않음
        
        # --- Augmentation (데이터 증강) ---
        mosaic=0.5,                    # 모자이크 증강 (위장 객체의 다양한 배경 학습)
        degrees=10.0,                  # 약간의 회전
        hsv_s=0.2,                     # 채도 변환 (위장 객체는 색상에 민감하므로 과도한 변환 주의)
    )
    
    print(f"--- 학습 완료 ---")

def resume_train():
    print("--- 완료된 모델에 추가 학습 시작 ---")
    
    # 1. 학습 완료된 가중치 로드 (best.pt 또는 last.pt)
    weights_path = '../ultralytics_lib/runs/result/capston/weights/last.pt'
    model = YOLO(weights_path) # yaml 파일 없이 가중치만 바로 로드합니다.

    # 2. 학습 파라미터 설정 및 실행
    results = model.train(
        data='./data.yaml', 
        project='../../../result', 
        name='cod_train_continue',  # 덮어쓰기 방지를 위해 폴더명을 다르게 주는 것을 권장합니다.
        
        epochs=50,                  # 추가로 진행할 에폭 수 설정
        imgsz=640,                     
        batch=16,                      
        device=0,                      
        workers=4,                     
        
        # 이전과 동일한 나머지 파라미터들 유지
        optimizer='AdamW',             
        lr0=0.001,                     
        mask_ratio=1,                  
        overlap_mask=False,            
        mosaic=0.5,                    
        degrees=10.0,                  
        hsv_s=0.2,                     
    )
    
    print(f"--- 추가 학습 완료 ---")

if __name__ == '__main__':
    # Windows/PyTorch 환경에서 다중 프로세싱 에러를 방지하기 위해 필수
    #train_model()
    resume_train()