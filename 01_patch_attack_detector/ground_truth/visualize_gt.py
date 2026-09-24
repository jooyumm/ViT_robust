"""
ground_truth/visualize_gt.py — PatchFool/LaVAN 각각 예시 이미지 한 장으로 "공격이 실제로
어느 patch를 건드렸는가"(coverage.gt_coverage로 픽셀 diff에서 계산)를 그림으로 보여준다.
공격이 어떻게 생겼는지 예시를 보여주는 용도 — attention/탐지기는 다루지 않는다.

과거 NPZ에 저장된 gt_coverage/repr_images를 읽는 독립적인 예시 도구이다.
현재 characterization의 추출·분석 코드에는 의존하지 않는다.

사용법:
  python visualize_gt.py --num_samples 100
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_ROOT = os.path.dirname(HERE)   # 01_patch_attack_detector
sys.path.insert(0, HERE)       # for `import coverage/draw_boxes`

import numpy as np
from draw_boxes import plot_gt_locations

ATTACKED_GROUPS = ('patchfool', 'lavan')
EXAMPLE_IMG_IDX = 4   # 예시로 보여줄 대표 이미지 인덱스 (0~4 중 아무거나, 4로 고정)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    npz_path = os.path.join(DETECT_ROOT, 'results', 'characterization',
                             f'00_attention_data_n{args.num_samples}.npz')
    if not os.path.exists(npz_path):
        raise SystemExit(f"과거 GT 예시 데이터가 없습니다: {npz_path}")
    repr_by_group = {}
    with np.load(npz_path, allow_pickle=False) as archive:
        for group in ('clean', *ATTACKED_GROUPS):
            repr_by_group[group] = {'images': archive[group + '__repr_images']}
            if group != 'clean':
                repr_by_group[group]['gt_coverage'] = archive[group + '__gt_coverage_repr']

    out_dir = os.path.join(HERE, 'results')
    os.makedirs(out_dir, exist_ok=True)
    clean_img = repr_by_group['clean']['images'][EXAMPLE_IMG_IDX]
    for g in ATTACKED_GROUPS:
        adv_img = repr_by_group[g]['images'][EXAMPLE_IMG_IDX]
        gt_cov = repr_by_group[g]['gt_coverage'][EXAMPLE_IMG_IDX]
        p = os.path.join(out_dir, f'{g}.png')
        plot_gt_locations(clean_img, adv_img, gt_cov, g, p, label=f'image {EXAMPLE_IMG_IDX}')
        print(f"Saved: {p}")


if __name__ == '__main__':
    main()
