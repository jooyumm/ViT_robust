"""
experiments/characterization/05_gt_check.py — "attention이 정말 공격당한 자리로 몰리는가"를
더 이상 한 장씩 눈으로/스크립트로 확인하지 않고, ground truth와 자동으로 대조하는 시스템.

방법론: 04_column_check.py를 만들 때 image 0/L=12에서 patch 17이 attention sink인 걸
확인했는데, 그게 진짜 공격이 들어간 자리인지는 별도로 clean/adv 이미지를 직접 diff해서
검증했다(추측 아님). 이 스크립트는 그 diff 검증을 매번 새로 손으로 하는 대신, 00_extract_
attention.py가 대표 이미지 n_repr장(기본 5장)에 대해 저장해 둔 gt_coverage(gt_lib.py로 만든
"실제로 어느 patch가 얼마나 바뀌었는지")를 관측된 attention(argmax_col_avg)과 자동으로
비교한다.

PatchFool: 정확히 patch 1개만 건드리므로 "GT patch == 레이어 L의 attention argmax"가
정확히 맞아야 sink가 진짜라고 볼 수 있다.
LaVAN: 16px 그리드에 정렬 안 된 위치에 놓이므로 여러 patch에 걸쳐 부분적으로 겹친다 —
"attention argmax가 GT가 조금이라도 겹친 patch 중 하나인가"로 판정한다(정확히 1개 일치를
요구하면 애초에 안 맞을 수 있는 구조).

GPU 필요 없음 — 00_extract_attention.py가 저장한 npz만 읽는다.

사용법:
  python 05_gt_check.py --num_samples 100 --layer 12
  (같은 폴더에 00_extract_attention.py로 만든, gt_coverage_repr/repr_images 필드가 있는
   버전의 00_attention_data_n100.npz가 있어야 함 — 없으면 00_extract_attention.py를 다시
   실행할 것)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # for `import data_io/plotting/gt_lib`

from data_io import load_attention_data
from gt_lib import gt_patch_summary
from plotting import plot_gt_overlay

ATTACKED_GROUPS = ('patchfool', 'lavan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--layer', type=int, default=12, help='관측 attention을 볼 레이어 (1-indexed)')
    args = parser.parse_args()

    out_dir = HERE.replace('/experiments/', '/results/', 1)
    npz_path = os.path.join(out_dir, f'00_attention_data_n{args.num_samples}.npz')
    assert os.path.exists(npz_path), \
        f"{npz_path} 없음 — 먼저 00_extract_attention.py --num_samples {args.num_samples} 실행할 것"
    all_metrics, repr_by_group, n_layers = load_attention_data(npz_path)

    for g in ATTACKED_GROUPS:
        if 'gt_coverage' not in repr_by_group.get(g, {}):
            raise SystemExit(
                f"이 npz는 {g}__gt_coverage_repr 필드가 없는 구버전입니다 — "
                "00_extract_attention.py를 다시 실행해서 npz를 새로 만드세요.")

    n_repr = repr_by_group['patchfool']['gt_coverage'].shape[0]
    rows = []
    for g in ATTACKED_GROUPS:
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

            p = os.path.join(
                out_dir, f'05_gt_overlay_{g}_img{img_idx}_L{args.layer}.png')
            plot_gt_overlay(repr_by_group, all_metrics, img_idx, g, args.layer, p)

        print(f"[{g}] L={args.layer}: attention argmax가 GT patch에 포함된 이미지 "
              f"{n_match}/{n_repr}")

    md_path = os.path.join(out_dir, f'05_gt_check_L{args.layer}_n{args.num_samples}.md')
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
    print(f"Saved: 05_gt_overlay_*_L{args.layer}.png ({len(ATTACKED_GROUPS) * n_repr} files)")


if __name__ == '__main__':
    main()
