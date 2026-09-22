"""
ground_truth/gt_overlay_plot.py — GT(gt_lib.gt_coverage, 픽셀 diff로 계산한 "공격이 실제로
건드린 patch")를 원본/공격 이미지 위에 그려서 보여주는 그림 함수. attention이나 탐지기
출력은 여기서 다루지 않는다 — 이 폴더의 역할은 정답 위치를 보여주는 것뿐이다.
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


def _draw_patch_box(ax, patch_idx, color, label=None, ppl=PPL, patch_size=PATCH_SIZE, lw=2.0):
    r, c = patch_idx // ppl, patch_idx % ppl
    ax.add_patch(Rectangle((c * patch_size, r * patch_size), patch_size, patch_size,
                            fill=False, edgecolor=color, lw=lw))
    if label:
        ax.text(c * patch_size + 1, r * patch_size + patch_size - 2, label,
                color=color, fontsize=7, fontweight='bold', va='bottom')


def plot_gt_locations(repr_by_group, img_idx, group, out_path, gt_min_frac=0.01):
    """공격이 실제로 건드린 patch(들)을 [clean | adv | diff 히트맵] 세 장에 노란 박스로
    표시하고, 박스마다 겹침 비율(%)을 적는다(100%든 아니든 전부 — PatchFool처럼 patch
    하나가 100% 나올 때만 숨기면, LaVAN처럼 여러 patch가 섞인 그림에서 제일 많이 겹친
    가운데 patch만 라벨이 비어 보여 오히려 헷갈린다). GT는 순전히 gt_lib.gt_coverage
    (픽셀 diff)로 계산한 것이고, attention이나 탐지기 결과는 이 그림에 전혀 들어가지
    않는다. repr_by_group[group]['images'/'gt_coverage']가 없으면(구버전 npz) 조용히
    건너뛴다."""
    if 'images' not in repr_by_group.get('clean', {}) or \
       'images' not in repr_by_group.get(group, {}) or \
       'gt_coverage' not in repr_by_group.get(group, {}):
        return False

    clean_img = repr_by_group['clean']['images'][img_idx]
    adv_img = repr_by_group[group]['images'][img_idx]
    gt_cov = repr_by_group[group]['gt_coverage'][img_idx]
    gt_patches = gt_patch_summary(gt_cov, min_frac=gt_min_frac)

    diff = np.abs(_unnormalize(adv_img) - _unnormalize(clean_img)).sum(axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, img, title in [(axes[0], _unnormalize(clean_img), 'Clean'),
                            (axes[1], _unnormalize(adv_img), f'{GROUP_LABELS[group]}'),
                            (axes[2], diff, 'diff (bright = pixels changed)')]:
        cmap = None if img.ndim == 3 else 'hot'
        ax.imshow(img, cmap=cmap)
        _draw_patch_grid(ax)
        for p, frac in gt_patches:
            _draw_patch_box(ax, p, 'yellow', label=f'{frac:.0%}')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)

    gt_str = ', '.join(f'{p}({frac:.0%})' for p, frac in gt_patches) or 'none'
    fig.suptitle(f'{GROUP_LABELS[group]} image {img_idx} — GT attacked patch(es): {gt_str}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True
