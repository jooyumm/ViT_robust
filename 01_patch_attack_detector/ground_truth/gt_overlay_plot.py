"""
ground_truth/gt_overlay_plot.py — GT(gt_lib.gt_coverage, 픽셀 diff)와 관측된 attention
argmax를 같은 이미지 위에 겹쳐 그리는 그림 함수. check_attention_vs_gt.py 전용.

experiments/characterization/plotting.py의 GROUP_COLORS/GROUP_LABELS를 그대로 다시
쓰지 않고 여기 따로 둔다 — ground_truth/는 characterization 하나의 부산물이 아니라
탐지기 테스트 등 다른 곳에서도 재사용할 독립 모듈이라, characterization 내부 파일에
import 의존을 만들지 않는 게 낫다고 판단했다(두 줄짜리 dict라 중복 비용도 작다).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from gt_lib import gt_patch_summary, PATCH_SIZE, PPL

GROUP_LABELS = {'clean': 'Clean', 'patchfool': 'PatchFool', 'lavan': 'LaVAN'}


def _unnormalize(img):
    """(3,H,W) 정규화된 텐서 -> (H,W,3) [0,1] ndarray. dataset.py의 mean=std=0.5 규약."""
    disp = img * 0.5 + 0.5
    return np.clip(disp.transpose(1, 2, 0), 0, 1)


def _draw_patch_grid(ax, ppl=PPL, patch_size=PATCH_SIZE):
    for i in range(1, ppl):
        ax.axhline(i * patch_size, color='white', lw=0.3, alpha=0.4)
        ax.axvline(i * patch_size, color='white', lw=0.3, alpha=0.4)


def _draw_patch_box(ax, patch_idx, color, ppl=PPL, patch_size=PATCH_SIZE, lw=2.0):
    r, c = patch_idx // ppl, patch_idx % ppl
    ax.add_patch(Rectangle((c * patch_size, r * patch_size), patch_size, patch_size,
                            fill=False, edgecolor=color, lw=lw))


def plot_gt_overlay(repr_by_group, all_metrics, img_idx, group, layer, out_path,
                     gt_min_frac=0.01):
    """"공격이 실제로 어디를 건드렸는가(GT, 노란 박스)"와 "레이어 L에서 attention이 실제로
    가장 몰리는 곳(argmax_col_avg, 하늘색 박스)"을 같은 이미지 위에 그려서 눈으로 직접
    겹치는지 확인한다. clean/adv 원본 이미지와 diff 히트맵을 나란히 보여준다. GT는
    gt_lib.gt_coverage(픽셀 diff)로 계산한 것이지 attention으로 추정한 게 아니다 — 그래서
    이 그림이 "진짜 공격 위치 vs 관측된 attention 위치"의 직접 비교가 된다.
    repr_by_group[group]['images'/'gt_coverage']가 없으면(구버전 npz) 조용히 건너뛴다."""
    if 'images' not in repr_by_group.get('clean', {}) or \
       'images' not in repr_by_group.get(group, {}) or \
       'gt_coverage' not in repr_by_group.get(group, {}):
        return False

    clean_img = repr_by_group['clean']['images'][img_idx]
    adv_img = repr_by_group[group]['images'][img_idx]
    gt_cov = repr_by_group[group]['gt_coverage'][img_idx]
    gt_patches = gt_patch_summary(gt_cov, min_frac=gt_min_frac)
    gt_top = gt_patches[0][0] if gt_patches else None
    observed = int(all_metrics[group]['argmax_col_avg'][img_idx, layer - 1])
    match = (observed == gt_top)

    diff = np.abs(_unnormalize(adv_img) - _unnormalize(clean_img)).sum(axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, img, title in [(axes[0], _unnormalize(clean_img), 'Clean'),
                            (axes[1], _unnormalize(adv_img), f'{GROUP_LABELS[group]}'),
                            (axes[2], diff, 'diff (bright = pixels changed)')]:
        cmap = None if img.ndim == 3 else 'hot'
        ax.imshow(img, cmap=cmap)
        _draw_patch_grid(ax)
        for p, frac in gt_patches:
            _draw_patch_box(ax, p, 'yellow')
        _draw_patch_box(ax, observed, 'cyan')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)

    gt_str = ', '.join(f'{p}({frac:.0%})' for p, frac in gt_patches) or 'none'
    fig.suptitle(f'{GROUP_LABELS[group]} image {img_idx}, L={layer} — '
                 f'GT patch(es) [yellow]: {gt_str} | attention argmax [cyan]: {observed} | '
                 f'{"MATCH" if match else "no match"}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True
