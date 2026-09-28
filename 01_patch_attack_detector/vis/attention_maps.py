"""vis/attention_maps.py — clean/attack/차이 attention 맵 3열짜리 그림. 원래
experiments/characterization/02_compare_attention_success_failure.py 안에 있던 그리기 함수들이고,
03_patchfool_success_split.py가 이걸 다시 쓰려고 02를 통째로 importlib로 불러오고 있었다 — 여기로
옮겨서 둘 다(그리고 앞으로 만들 탐지기 시각화도) 정상적으로 import해서 쓴다.

data/manifest/metrics의 형태는 characterization의 02_compare_attention_success_failure.py의
load_paired_run() / compute_metrics()가 만드는 것과 같다(이 모듈 자체는 그 두 함수에 의존하지 않는다 —
같은 모양의 dict만 주면 된다).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boxes import draw_patch_box, CLEAN_BOX_COLOR, ATTACK_BOX_COLOR   # noqa: E402
from colors import SEQUENTIAL, DIVERGING                              # noqa: E402


def save_figure(figure, path):
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)


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


def draw_maps(data, manifest, attack, sample_i, metrics, layers, scales, status_label, output):
    """clean/attack 사진 + GT coverage를 첫 행에, 이후 지정한 레이어만 clean/attack/차이로 그린다.
    모든 행(사진 포함)이 같은 3열 grid를 쓰고, 컬러바는 열 전체(사진 행 포함)에서 폭을
    빌려 배치한다 -- 그래야 컬러바가 붙는 마지막 레이어 행만 폭이 줄어 다른 행과 어긋나
    보이는 문제가 생기지 않는다."""
    sample_id = int(data['sample_index'][sample_i])
    coverage = data[attack + '__gt_coverage'][sample_i]
    side = int(np.sqrt(len(coverage)))
    if side * side != len(coverage):
        raise ValueError('현재 맵은 정사각 patch 격자를 지원합니다.')
    n_rows = len(layers) + 1
    # 열 순서: Clean | Attack | raw 컬러바 | 차이 | 차이 컬러바. 컬러바를 이미지 열에서 폭을 빌려
    # 붙이면 열마다 폭이 달라져 어긋나므로, 컬러바는 전용 열에 두고 같은 길이·같은 높이로 놓는다.
    figure = plt.figure(figsize=(12.6, 2.9 * n_rows), layout='constrained')
    grid = figure.add_gridspec(n_rows, 5, width_ratios=[1, 1, .3, 1, .3])
    image_grid_col = [0, 1, 3]
    axes = np.array([[figure.add_subplot(grid[r, image_grid_col[c]]) for c in range(3)] for r in range(n_rows)])
    mean = np.asarray(manifest['mean'])[:, None, None]
    std = np.asarray(manifest['std'])[:, None, None]
    extent = [-.5, side - .5, side - .5, -.5]

    clean_img = np.clip((data['clean__images'][sample_i] * std + mean).transpose(1, 2, 0), 0, 1)
    attack_img = np.clip((data[attack + '__images'][sample_i] * std + mean).transpose(1, 2, 0), 0, 1)
    top_panels = [(0, clean_img, 'Clean', CLEAN_BOX_COLOR), (1, attack_img, attack, ATTACK_BOX_COLOR)]
    for column, image, title, box_color in top_panels:
        axis = axes[0, column]
        axis.imshow(image, extent=extent)
        draw_gt(axis, coverage, side, box_color)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    # 차이 열의 첫 행은 비워 둔다(칸은 남겨야 컬러바가 열 전체에서 같은 폭을 빌려 행이 어긋나지 않는다).
    axes[0, 2].set_axis_off()

    mass_max, delta_max = scales
    mass_norm = Normalize(vmin=0, vmax=mass_max)
    delta_norm = Normalize(vmin=-delta_max, vmax=delta_max)
    column_image = {}
    for row, layer_i in enumerate(layers, 1):
        panels = [('clean', 'Clean'), ('attacked', attack), ('delta', '(Attack − clean)')]
        for column, (key, title) in enumerate(panels):
            axis = axes[row, column]
            values = metrics[key][sample_i, layer_i, 1:].reshape(side, side) * 100
            if key == 'delta':
                image = axis.imshow(values, cmap=DIVERGING, norm=delta_norm, interpolation='nearest')
            else:
                image = axis.imshow(values, cmap=SEQUENTIAL, norm=mass_norm, interpolation='nearest')
            column_image[column] = image
            draw_gt(axis, coverage, side, CLEAN_BOX_COLOR if key == 'clean' else ATTACK_BOX_COLOR)
            axis.set_title(f'L{layer_i + 1} · {title}')
            axis.set_xticks([])
            axis.set_yticks([])
            if key == 'delta':
                gt_gain = metrics['gt_delta'][sample_i, layer_i] * 100
                cls_gain = metrics['cls_delta'][sample_i, layer_i] * 100
                axis.set_xlabel(f'GT {gt_gain:+.2f} pp | CLS {cls_gain:+.2f} pp', fontsize=8)
    # raw(파랑)는 Clean/Attack이 같은 스케일이라 하나만 두고, 차이 컬러바와 길이·위치를 맞춘다.
    # 12개 레이어 그림은 짧게(그림 가운데쯤), 레이어 몇 개만 그린 그림은 행 전체 높이로 길게 둔다.
    bar_box = [.1, .064, .35, .928] if len(layers) <= 4 else [.1, .42, .35, .22]
    for grid_col, image, label, extend in [(2, column_image[0], 'Raw attention (%)', 'max'),
                                             (4, column_image[2], '(Attack − clean) (pp)', 'both')]:
        holder = figure.add_subplot(grid[1:, grid_col])
        holder.set_axis_off()
        figure.colorbar(image, cax=holder.inset_axes(bar_box), extend=extend, label=label)
    label_value = int(data['label'][sample_i])
    clean_pred = int(data['clean__prediction'][sample_i])
    attacked_pred = int(data[attack + '__prediction'][sample_i])
    figure.suptitle(f'{attack} · sample {sample_id} · {status_label}\n'
                   f'label {label_value} | prediction {clean_pred} → {attacked_pred}', fontsize=13)
    save_figure(figure, output)
