"""Attention comparison fig layout; output is owned by this experiment."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from src.visualization.colors import CLEAN_BOX_COLOR, attack_color, SEQUENTIAL, DIVERGING
from vis.attention_maps import draw_gt, map_scales
from src.visualization.figs import save_fig, unnormalize

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
    fig = plt.figure(figsize=(12.6, 2.9 * n_rows), layout='constrained')
    grid = fig.add_gridspec(n_rows, 5, width_ratios=[1, 1, .3, 1, .3])
    image_grid_col = [0, 1, 3]
    axes = np.array([[fig.add_subplot(grid[r, image_grid_col[c]]) for c in range(3)] for r in range(n_rows)])
    extent = [-.5, side - .5, side - .5, -.5]

    clean_img = unnormalize(data['clean__images'][sample_i], manifest['mean'], manifest['std'])
    attack_img = unnormalize(data[attack + '__images'][sample_i], manifest['mean'], manifest['std'])
    top_panels = [(0, clean_img, 'Clean', CLEAN_BOX_COLOR), (1, attack_img, attack, attack_color(attack))]
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
            draw_gt(axis, coverage, side, CLEAN_BOX_COLOR if key == 'clean' else attack_color(attack))
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
        holder = fig.add_subplot(grid[1:, grid_col])
        holder.set_axis_off()
        fig.colorbar(image, cax=holder.inset_axes(bar_box), extend=extend, label=label)
    label_value = int(data['label'][sample_i])
    clean_pred = int(data['clean__prediction'][sample_i])
    attacked_pred = int(data[attack + '__prediction'][sample_i])
    fig.suptitle(f'{attack} · sample {sample_id} · {status_label}\n'
                   f'label {label_value} | prediction {clean_pred} → {attacked_pred}', fontsize=13)
    save_fig(fig, output)
