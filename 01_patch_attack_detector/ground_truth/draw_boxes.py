"""
ground_truth/draw_boxes.py — GT(coverage.gt_coverage, 픽셀 diff로 계산한 "공격이 실제로
건드린 patch")를 원본/공격 이미지 위에 박스로 그려서 보여주는 그림 함수. attention이나
탐지기 출력은 여기서 다루지 않는다 — 이 폴더의 역할은 정답 위치를 보여주는 것뿐이다.
박스 그리는 원시 함수(격자·박스·색)는 vis/boxes.py에 있다 — 여기서는 GT 표시 방식(감싸는
사각형 vs patch별 박스+퍼센트 라벨)만 결정한다.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coverage import gt_patch_summary

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vis'))
from boxes import draw_patch_box, draw_bbox, draw_patch_grid, CLEAN_BOX_COLOR, ATTACK_BOX_COLOR  # noqa: E402

GROUP_LABELS = {'clean': 'Clean', 'patchfool': 'PatchFool', 'lavan': 'LaVAN'}


def _unnormalize(img):
    """(3,H,W) 정규화된 텐서 -> (H,W,3) [0,1] ndarray. dataset.py의 mean=std=0.5 규약."""
    disp = img * 0.5 + 0.5
    return np.clip(disp.transpose(1, 2, 0), 0, 1)


def plot_gt_locations(clean_img, adv_img, gt_cov, group, out_path, label=None, gt_min_frac=0.01):
    """공격이 실제로 건드린 patch(들)을 [clean | adv | diff 히트맵] 세 장에 표시한다.
    clean_img/adv_img: (3,H,W) 텐서 한 장. gt_cov: (196,) coverage.gt_coverage 출력.
    label: 제목에 덧붙일 설명(예: "image 4", "still correct (true=670)") — 없으면 생략.

    clean/adv 두 패널은 배경 patch 격자는 그대로 두되(patch 경계 참고용), GT 표시는
    patch별로 따로 박스를 치지 않고 **전부를 감싸는 사각형 하나**로만 그린다 — LaVAN처럼
    여러 patch에 걸쳐 있을 때 patch마다 박스를 치면 그 자체로 3x3 격자처럼 보여서 오히려
    공격이 잘 안 보인다는 피드백 반영. patch별 겹침 비율(%)까지 자세히 보여주는 건 diff
    패널 하나뿐 — 거기서만 patch별 박스+라벨을 그린다. clean은 파랑, 공격(adv/diff)은
    노랑으로 색만 달리해서 clean인지 공격인지 구분한다(clean도 박스를 지우지 않아야 adv
    패널과 같은 위치인지 바로 비교된다). PatchFool은 항상 patch 정확히 1개뿐이라 bbox와
    개별 patch 박스가 어차피 같은 모양이 되고, 라벨도 필요 없어 안 붙인다(title에 patch
    번호는 그대로 나온다). GT는 순전히 coverage.gt_coverage(픽셀 diff)로 계산한 것이고,
    attention이나 탐지기 결과는 이 그림에 전혀 들어가지 않는다."""
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
        draw_patch_grid(ax)
        if gt_patches:
            if i == 2:   # diff: patch별 상세(경계 + 겹침 %)
                for p, frac in gt_patches:
                    patch_label = f'{frac:.0%}' if show_pct else None
                    draw_patch_box(ax, p, box_color, label=patch_label)
            else:        # clean/adv: 감싸는 사각형 하나만
                draw_bbox(ax, gt_patches, box_color)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=10)

    if show_pct:
        gt_str = ', '.join(f'{p}({frac:.0%})' for p, frac in gt_patches) or 'none'
    else:
        gt_str = ', '.join(str(p) for p, _ in gt_patches) or 'none'
    title = f'{GROUP_LABELS[group]}' + (f' — {label}' if label else '')
    fig.suptitle(f'{title} — GT attacked patch(es): {gt_str}')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True
