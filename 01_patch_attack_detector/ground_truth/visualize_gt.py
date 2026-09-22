"""
ground_truth/visualize_gt.py — PatchFool/LaVAN 각각 예시 이미지 한 장으로 "공격이 실제로
어느 patch를 건드렸는가"(coverage.gt_coverage로 픽셀 diff에서 계산)를 그림으로 보여준다.
공격이 어떻게 생겼는지 예시를 보여주는 용도 — attention/탐지기는 다루지 않는다.

GT 자체(gt_coverage/repr_images)는 characterization의 00_extract_attention.py가 이미
npz에 저장해 뒀다(GPU 필요한 단계는 거기뿐) — 여기서는 그 npz를 읽기만 한다.

사용법:
  python visualize_gt.py --num_samples 100
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_ROOT = os.path.dirname(HERE)   # 01_patch_attack_detector
CHAR_DIR = os.path.join(DETECT_ROOT, 'experiments', 'characterization')
sys.path.insert(0, HERE)       # for `import coverage/draw_boxes`
sys.path.insert(0, CHAR_DIR)   # for `import data_io` (attention npz 로더)

from data_io import load_attention_data
from draw_boxes import plot_gt_locations

ATTACKED_GROUPS = ('patchfool', 'lavan')
EXAMPLE_IMG_IDX = 4   # 예시로 보여줄 대표 이미지 인덱스 (0~4 중 아무거나, 4로 고정)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    npz_path = os.path.join(DETECT_ROOT, 'results', 'characterization',
                             f'00_attention_data_n{args.num_samples}.npz')
    assert os.path.exists(npz_path), \
        f"{npz_path} 없음 — 먼저 characterization/00_extract_attention.py " \
        f"--num_samples {args.num_samples} 실행할 것"
    _, repr_by_group, _ = load_attention_data(npz_path)

    for g in ATTACKED_GROUPS:
        if 'gt_coverage' not in repr_by_group.get(g, {}):
            raise SystemExit(
                f"이 npz는 {g}__gt_coverage_repr 필드가 없는 구버전입니다 — "
                "00_extract_attention.py를 다시 실행해서 npz를 새로 만드세요.")

    out_dir = os.path.join(HERE, 'results')
    os.makedirs(out_dir, exist_ok=True)
    for g in ATTACKED_GROUPS:
        p = os.path.join(out_dir, f'{g}.png')
        plot_gt_locations(repr_by_group, EXAMPLE_IMG_IDX, g, p)
        print(f"Saved: {p}")


if __name__ == '__main__':
    main()
