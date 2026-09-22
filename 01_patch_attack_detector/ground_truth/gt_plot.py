"""
ground_truth/gt_plot.py — GT(gt_lib.gt_coverage, 픽셀 diff로 계산한 "공격이 실제로
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
# clean 패널은 파랑, 공격(patchfool/lavan) 패널은 전부 노랑 — clean도 박스를 지우지
# 않는다(지우면 adv 패널과 같은 위치인지 비교하기 어려워진다), 색만 달라서 clean인지
# 공격인지 한눈에 구분된다.
CLEAN_BOX_COLOR = 'blue'
ATTACK_BOX_COLOR = 'yellow'


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


def _draw_bbox(ax, gt_patches, color, ppl=PPL, patch_size=PATCH_SIZE, lw=2.0):
    """gt_patches(여러 patch에 걸친 GT)를 patch별로 따로따로가 아니라, 전부를 감싸는
    사각형 하나로 그린다 — LaVAN처럼 겹친 patch가 여러 개면 patch 경계마다 박스를 치는
    게 오히려 3x3 격자처럼 보여서 attack이 안 보인다는 피드백 반영."""
    rows = [p // ppl for p, _ in gt_patches]
    cols = [p % ppl for p, _ in gt_patches]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    ax.add_patch(Rectangle((c0 * patch_size, r0 * patch_size),
                            (c1 - c0 + 1) * patch_size, (r1 - r0 + 1) * patch_size,
                            fill=False, edgecolor=color, lw=lw))


def plot_gt_locations(repr_by_group, img_idx, group, out_path, gt_min_frac=0.01):
    """공격이 실제로 건드린 patch(들)을 [clean | adv | diff 히트맵] 세 장에 표시한다.
    clean/adv 두 패널은 배경 patch 격자는 그대로 두되(patch 경계 참고용), GT 표시는
    patch별로 따로 박스를 치지 않고 **전부를 감싸는 사각형 하나**로만 그린다 — LaVAN처럼
    여러 patch에 걸쳐 있을 때 patch마다 박스를 치면 그 자체로 3x3 격자처럼 보여서 오히려
    공격이 잘 안 보인다는 피드백 반영. patch별 겹침 비율(%)까지 자세히 보여주는 건 diff
    패널 하나뿐 — 거기서만 patch별 박스+라벨을 그린다. clean은 파랑, 공격(adv/diff)은
    노랑으로 색만 달리해서 clean인지 공격인지 구분한다(clean도 박스를 지우지 않아야 adv
    패널과 같은 위치인지 바로 비교된다). PatchFool은 항상 patch 정확히 1개뿐이라 bbox와
    개별 patch 박스가 어차피 같은 모양이 되고, 라벨도 필요 없어 안 붙인다(title에 patch
    번호는 그대로 나온다). GT는 순전히 gt_lib.gt_coverage(픽셀 diff)로 계산한 것이고,
    attention이나 탐지기 결과는 이 그림에 전혀 들어가지 않는다.
    repr_by_group[group]['images'/'gt_coverage']가 없으면(구버전 npz) 조용히 건너뛴다."""
    if 'images' not in repr_by_group.get('clean', {}) or \
       'images' not in repr_by_group.get(group, {}) or \
       'gt_coverage' not in repr_by_group.get(group, {}):
        return False

    clean_img = repr_by_group['clean']['images'][img_idx]
    adv_img = repr_by_group[group]['images'][img_idx]
    gt_cov = repr_by_group[group]['gt_coverage'][img_idx]
    gt_patches = gt_patch_summary(gt_cov, min_frac=gt_min_frac)
    show_pct = (group != 'patchfool')

    diff = np.abs(_unnormalize(adv_img) - _unnormalize(clean_img)).sum(axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    panels = [(axes[0], _unnormalize(clean_img), 'Clean', CLEAN_BOX_COLOR),
              (axes[1], _unnormalize(adv_img), f'{GROUP_LABELS[group]}', ATTACK_BOX_COLOR),
              (axes[2], diff, 'diff (bright = pixels changed)', ATTACK_BOX_COLOR)]
    for i, (ax, img, title, box_color) in enumerate(panels):
        cmap = None if img.ndim == 3 else 'hot'
        ax.imshow(img, cmap=cmap)
        _draw_patch_grid(ax)
        if gt_patches:
            if i == 2:   # diff: patch별 상세(경계 + 겹침 %)
                for p, frac in gt_patches:
                    label = f'{frac:.0%}' if show_pct else None
                    _draw_patch_box(ax, p, box_color, label=label)
            else:        # clean/adv: 감싸는 사각형 하나만
                _draw_bbox(ax, gt_patches, box_color)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)

    if show_pct:
        gt_str = ', '.join(f'{p}({frac:.0%})' for p, frac in gt_patches) or 'none'
    else:
        gt_str = ', '.join(str(p) for p, _ in gt_patches) or 'none'
    fig.suptitle(f'{GROUP_LABELS[group]} image {img_idx} — GT attacked patch(es): {gt_str}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True
