"""Reusable attention-map scaling and patch annotations. No experiment runner."""
import numpy as np
from src.visualization.boxes import draw_patch_box

def draw_gt(axis, coverage, side, color):
    """부분적으로라도 변경된 토큰에 테두리를 표시한다. 이 맵은 imshow 기본 extent(칸 중심이
    정수 좌표)라 patch_size=1, offset=-.5로 반 칸 보정한다."""
    for patch_i in np.flatnonzero(coverage > 0):
        draw_patch_box(axis, int(patch_i), color, ppl=side, patch_size=1, lw=1.2, offset=-.5)


def robust_scale(values, percentile=99):
    """단일 극단값이 색 범위를 지배하지 않도록 백분위수로 vmax를 잡는다.
    이보다 큰 값은 색이 잘리며(colorbar의 extend로 표시), 실제 값이 바뀌지는 않는다."""
    finite = np.abs(values[np.isfinite(values)])
    return float(np.percentile(finite, percentile)) if finite.size else 1e-8


def map_scales(metrics, outcome):
    """맵 색 범위(퍼센트 포인트 단위). 같은 공격의 모든 clean 정답 표본·전 레이어에서 잡는다."""
    valid = outcome >= 0
    mass_max = robust_scale(100 * np.concatenate([metrics['clean'][valid, :, 1:],
                                                    metrics['attacked'][valid, :, 1:]]),
                            percentile=99.5)
    delta_max = robust_scale(100 * metrics['delta'][valid, :, 1:], percentile=99.5)
    return max(mass_max, 1e-8), max(delta_max, 1e-8)
