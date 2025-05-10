import numpy as np
import torch

def transform_volume(volume):
    """
    Args:
        volume: numpy array, shape (T, H, W), dtype=uint8

    Returns:
        torch.Tensor, shape (1, T, H, W), dtype=torch.float32
    """
    num_slices = 32
    total_slices = volume.shape[0]  # 예: 224

    # 1. 중앙 32 슬라이스 추출 (Temporal crop)
    start = (total_slices - num_slices) // 2
    end = start + num_slices
    cropped = volume[start:end, :, :]  # (32, H, W)

    # 2. float32 변환 + [0, 1] 정규화
    vol_float = cropped.astype(np.float32) / 255.0  # (32, H, W)

    # 3. mean, std 정규화 (r3d_18 grayscale 기준)
    mean = 0.43216
    std = 0.22803
    vol_normalized = (vol_float - mean) / std  # (32, H, W)

    # 4. (T, H, W) -> (1, T, H, W) -> torch.Tensor 변환
    vol_tensor = torch.from_numpy(vol_normalized).unsqueeze(0)  # 채널 차원 추가

    return vol_tensor  # shape: (1, 32, H, W)
