"""공유 패치 좌표와 실제 변경 면적. 탐지기나 그림 정의는 포함하지 않는다."""
import numpy as np


def patch_region_index(block=7, grid=14):
    if not isinstance(block, int) or block <= 0 or grid % block:
        raise ValueError('block은 grid의 양의 약수여야 합니다.')
    row, col = np.divmod(np.arange(grid * grid), grid)
    return (row // block) * (grid // block) + col // block


def patch_to_region(patch, block=7, grid=14):
    patch = np.asarray(patch)
    if np.any((patch < 0) | (patch >= grid * grid)):
        raise ValueError('patch 인덱스가 격자 밖입니다.')
    return (patch // grid // block) * (grid // block) + patch % grid // block


def coverage_from_images(clean, attacked, patch_size=16, threshold=1e-6):
    """NumPy 또는 torch BCHW 입력 -> (B, patches) NumPy 실제 변경 비율.

    실제 픽셀 차이이며 공격에 허용한 mask와는 다르다.
    """
    if clean.shape != attacked.shape or len(clean.shape) != 4:
        raise ValueError('동일한 BCHW 이미지 쌍이 필요합니다.')
    batch, _, height, width = clean.shape
    if patch_size <= 0 or height % patch_size or width % patch_size:
        raise ValueError('이미지 크기는 patch_size의 배수여야 합니다.')
    if isinstance(clean, np.ndarray):
        mask = np.abs(attacked - clean).sum(axis=1) > threshold
        cells = mask.reshape(batch, height//patch_size, patch_size, width//patch_size, patch_size)
        return cells.mean(axis=(2, 4)).reshape(batch, -1)
    mask = (attacked-clean).abs().sum(dim=1) > threshold
    cells = mask.float().unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    return cells.mean(dim=(-1, -2)).flatten(1).detach().cpu().numpy()
