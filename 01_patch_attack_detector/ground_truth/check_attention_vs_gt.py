"""
ground_truth/check_attention_vs_gt.py — "attention이 정말 공격당한 자리로 몰리는가"를 한
장씩 눈으로/일회성 스크립트로 확인하지 않고, ground truth와 자동으로 대조하는 시스템.

이 폴더(`ground_truth/`)는 원래 experiments/characterization/ 안에 있었는데, 탐지기
테스트(`01_patch_attack_detector/detector/`)에도 같은 방식(공격이 실제로 어디 있는지 vs
탐지/attention이 어디를 가리키는지 대조)을 재사용할 수 있어서 characterization 전용이
아닌 01_patch_attack_detector 최상위로 옮겼다.

방법론: characterization의 04_column_check.py에서 image 0/L=12에 patch 17 attention
sink를 찾았는데, 그게 진짜 공격이 들어간 자리인지는 clean/adv 이미지를 직접 diff해서
검증했다(추측 아님). 이 스크립트는 그 diff 검증을 매번 새로 손으로 하는 대신,
`../experiments/characterization/00_extract_attention.py`가 대표 이미지 n_repr장(기본
5장)에 대해 저장해 둔 gt_coverage(gt_lib.py로 만든 "실제로 어느 patch가 얼마나
바뀌었는지")를 관측된 attention(argmax_col_avg)과 자동으로 비교한다.

PatchFool: 정확히 patch 1개만 건드리므로 "GT patch == 레이어 L의 attention argmax"가
정확히 맞아야 sink가 진짜라고 볼 수 있다.
LaVAN: 16px 그리드에 정렬 안 된 위치에 놓이므로 여러 patch에 걸쳐 부분적으로 겹친다 —
"attention argmax가 GT가 조금이라도 겹친 patch 중 하나인가"로 판정한다(정확히 1개 일치를
요구하면 애초에 안 맞을 수 있는 구조).

GPU 필요 없음 — characterization의 00_extract_attention.py가 저장한 npz만 읽는다.

사용법:
  python check_attention_vs_gt.py --num_samples 100 --layer 12
  (../experiments/characterization/results/00_attention_data_n100.npz에 gt_coverage_repr/
   repr_images 필드가 있어야 함 — 없으면 00_extract_attention.py를 다시 실행할 것)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_ROOT = os.path.dirname(HERE)   # 01_patch_attack_detector
CHAR_DIR = os.path.join(DETECT_ROOT, 'experiments', 'characterization')
sys.path.insert(0, HERE)       # for `import gt_lib/gt_overlay_plot`
sys.path.insert(0, CHAR_DIR)   # for `import data_io` (attention npz 로더)

from data_io import load_attention_data
from gt_lib import gt_patch_summary
from gt_overlay_plot import plot_gt_overlay

ATTACKED_GROUPS = ('patchfool', 'lavan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--layer', type=int, default=12, help='관측 attention을 볼 레이어 (1-indexed)')
    args = parser.parse_args()

    npz_path = os.path.join(DETECT_ROOT, 'results', 'characterization',
                             f'00_attention_data_n{args.num_samples}.npz')
    assert os.path.exists(npz_path), \
        f"{npz_path} 없음 — 먼저 characterization/00_extract_attention.py " \
        f"--num_samples {args.num_samples} 실행할 것"
    all_metrics, repr_by_group, n_layers = load_attention_data(npz_path)

    for g in ATTACKED_GROUPS:
        if 'gt_coverage' not in repr_by_group.get(g, {}):
            raise SystemExit(
                f"이 npz는 {g}__gt_coverage_repr 필드가 없는 구버전입니다 — "
                "00_extract_attention.py를 다시 실행해서 npz를 새로 만드세요.")

    out_dir = os.path.join(HERE, 'results')
    n_repr = repr_by_group['patchfool']['gt_coverage'].shape[0]
    rows = []
    for g in ATTACKED_GROUPS:
        group_dir = os.path.join(out_dir, g)
        os.makedirs(group_dir, exist_ok=True)
        n_match = 0
        for img_idx in range(n_repr):
            gt_cov = repr_by_group[g]['gt_coverage'][img_idx]
            gt_patches = gt_patch_summary(gt_cov)
            observed = int(all_metrics[g]['argmax_col_avg'][img_idx, args.layer - 1])
            gt_set = {p for p, _ in gt_patches}
            match = observed in gt_set
            n_match += match
            gt_str = ', '.join(f'{p}({frac:.0%})' for p, frac in gt_patches) or 'none'
            rows.append((g, img_idx, gt_str, observed, match))

            p = os.path.join(group_dir, f'img{img_idx}_L{args.layer}.png')
            plot_gt_overlay(repr_by_group, all_metrics, img_idx, g, args.layer, p)

        print(f"[{g}] L={args.layer}: attention argmax가 GT patch에 포함된 이미지 "
              f"{n_match}/{n_repr}")

    md_path = os.path.join(out_dir, f'gt_check_L{args.layer}_n{args.num_samples}.md')
    with open(md_path, 'w') as f:
        f.write(f"# GT vs 관측된 attention (L={args.layer}, n_repr={n_repr})\n\n")
        f.write("공격이 실제로 건드린 patch(GT, `gt_lib.gt_coverage`로 픽셀 diff에서 계산)와, "
                "그 레이어에서 attention이 실제로 가장 몰리는 patch(`argmax_col_avg`)를 "
                "대조한다. PatchFool은 정확히 1개 patch만 공격하므로 완전 일치를 기대하고, "
                "LaVAN은 여러 patch에 걸쳐 있으므로 '겹친 patch 중 하나에 attention이 "
                "몰리는가'로 판정한다.\n\n")
        f.write("| group | image | GT patch(es) (겹침 %) | attention argmax | match |\n")
        f.write("|---|---|---|---|---|\n")
        for g, img_idx, gt_str, observed, match in rows:
            f.write(f"| {g} | {img_idx} | {gt_str} | {observed} | {'O' if match else 'X'} |\n")
    print(f"Saved: {md_path}")
    print(f"Saved: results/{{patchfool,lavan}}/img*_L{args.layer}.png "
          f"({len(ATTACKED_GROUPS) * n_repr} files)")


if __name__ == '__main__':
    main()
