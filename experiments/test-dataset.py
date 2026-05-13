import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Step 3에서 작성하신 함수를 그대로 복사해 옵니다.
def generate_boundary_and_distance_map(masks_tensor, distance_clip=50):
    masks_np = masks_tensor.cpu().numpy().astype(np.uint8)
    N, H, W = masks_np.shape
    boundaries_np = np.zeros((N, H, W), dtype=np.uint8)
    distance_maps_np = np.zeros((N, H, W), dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for i in range(N):
        mask = masks_np[i]
        if mask.sum() == 0: continue
            
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = dilated - eroded
        boundaries_np[i] = boundary
        
        inv_boundary = 1 - boundary
        dist_map = cv2.distanceTransform(inv_boundary, cv2.DIST_L2, 5)
        dist_map = np.clip(dist_map, 0, distance_clip)
        
        weight_map = (distance_clip - dist_map) / distance_clip
        weight_map = 1.0 + weight_map 
        distance_maps_np[i] = weight_map

    boundaries_tensor = torch.from_numpy(boundaries_np).to(masks_tensor.device)
    distance_maps_tensor = torch.from_numpy(distance_maps_np).to(masks_tensor.device)
    return boundaries_tensor, distance_maps_tensor

# --- 테스트 실행부 ---
if __name__ == "__main__":
    # 1. 가상의 마스크 데이터 만들기 (배경 0, 객체 1)
    # 위장망이나 위장 객체라고 가정하고 임의의 도형을 그립니다.
    dummy_mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(dummy_mask, (100, 100), 50, 1, -1) 
    cv2.rectangle(dummy_mask, (40, 40), (80, 80), 1, -1)

    # 2. 파이토치 텐서로 변환 (N, H, W 형태)
    mask_tensor = torch.tensor(dummy_mask).unsqueeze(0)

    # 3. 함수 통과
    boundaries, distance_maps = generate_boundary_and_distance_map(mask_tensor, distance_clip=30)

    # 4. 시각화하여 확인하기
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("1. Original Mask (GT)")
    plt.imshow(mask_tensor[0].numpy(), cmap='gray')
    
    plt.subplot(1, 3, 2)
    plt.title("2. Boundary Mask")
    plt.imshow(boundaries[0].numpy(), cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("3. Distance Weight Map")
    # 거리가 가까울수록 값이 높아야(빨간색) 합니다.
    plt.imshow(distance_maps[0].numpy(), cmap='jet') 
    plt.colorbar(label='Weight')
    
    plt.tight_layout()
    plt.show()