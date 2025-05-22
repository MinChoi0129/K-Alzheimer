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

    stride = 2
    span = (num_slices - 1) * stride
    if span >= total_slices:
        raise ValueError("Impossible")

    start = (total_slices - span) // 2
    indices = start + np.arange(num_slices) * stride
    cropped = volume[indices, :, :]

    # 2. float32 변환 + [0, 1] 정규화
    vol_float = cropped.astype(np.float32) / 255.0  # (32, H, W)

    # 3. mean, std 정규화 (r3d_18 grayscale 기준)
    mean = 0.45
    std = 0.225
    vol_normalized = (vol_float - mean) / std  # (32, H, W)

    # 4. (T, H, W) -> (1, T, H, W) -> torch.Tensor 변환
    vol_tensor = torch.from_numpy(vol_normalized).unsqueeze(0)  # 채널 차원 추가

    return vol_tensor  # shape: (1, 32, H, W)
