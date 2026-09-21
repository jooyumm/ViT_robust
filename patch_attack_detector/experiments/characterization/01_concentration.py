"""
experiments/characterization/01_concentration.py — "attention이 얼마나 몰리는가, 그리고
공격에 따라 달라지는가"만 다룬다(위치가 어디인지는 02_sink_position.py 담당). GPU 필요
없음 — 00_extract_attention.py가 저장한 npz만 읽는다.

산출물 (results/characterization/, 전부 01_ 접두어로 이 스크립트가 만든 것임을 표시):
  - 01_hist_top1_row_by_layer.png     레이어별 top-1 mass(CLS row, 헤드평균) 히스토그램(3그룹)
  - 01_hist_top1_col_by_layer.png     동일, column(전체 쿼리 집계) 기준
  - 01_hist_entropy_row_by_layer.png  동일, 정규화 entropy 기준
  - 01_head_variability_L5_L12.png    L=5/L=12에서 헤드별 top-1 mass (헤드평균에서도 살아남는가)
  - 01_layer_comparison_top1_row.png  레이어별 중앙값+IQR (PatchFool vs LaVAN vs Clean)
  - 01_concentration_summary.md       레이어별 중앙값 표 + "5x 균등분포 초과 비율"(설명용, 임계값 아님)

범위 제한: 탐지 임계값/AUROC/flag 없음.

사용법:
  python 01_concentration.py --num_samples 100
  (같은 폴더에 00_extract_attention.py로 만든 00_attention_data_n100.npz가 있어야 함)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # for `import attention_lib/plotting/data_io`

from attention_lib import UNIFORM
from data_io import load_attention_data, GROUPS
from plotting import plot_histograms, plot_head_variability, plot_layer_comparison

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()

    out_dir = HERE.replace('/experiments/', '/results/', 1)
    npz_path = os.path.join(out_dir, f'00_attention_data_n{args.num_samples}.npz')
    assert os.path.exists(npz_path), \
        f"{npz_path} 없음 — 먼저 00_extract_attention.py --num_samples {args.num_samples} 실행할 것"
    all_metrics, repr_by_group, n_layers = load_attention_data(npz_path)

    print(f"{'L':>3} | {'clean':>8} | {'PatchFool':>10} | {'LaVAN':>8} | "
          f"{'PF-clean':>9} | {'LaVAN-clean':>11}")
    print("-" * 65)
    rows = []
    for L in range(n_layers):
        med_c = np.median(all_metrics['clean']['top1_row_avg'][:, L])
        med_p = np.median(all_metrics['patchfool']['top1_row_avg'][:, L])
        med_v = np.median(all_metrics['lavan']['top1_row_avg'][:, L])
        frac_c = (all_metrics['clean']['top1_row_avg'][:, L] > 5 * UNIFORM).mean()
        frac_p = (all_metrics['patchfool']['top1_row_avg'][:, L] > 5 * UNIFORM).mean()
        frac_v = (all_metrics['lavan']['top1_row_avg'][:, L] > 5 * UNIFORM).mean()
        print(f"{L+1:>3} | {med_c:>8.4f} | {med_p:>10.4f} | {med_v:>8.4f} | "
              f"{med_p - med_c:>+9.4f} | {med_v - med_c:>+11.4f}")
        rows.append(dict(layer=L + 1, med_c=med_c, med_p=med_p, med_v=med_v,
                          frac_c=frac_c, frac_p=frac_p, frac_v=frac_v))

    table_path = os.path.join(out_dir, f'01_concentration_summary_n{args.num_samples}.md')
    with open(table_path, 'w') as f:
        f.write(f"# Attention concentration — Clean vs PatchFool vs LaVAN "
                f"(n={args.num_samples})\n\n")
        f.write("CLS-row, head-averaged top-1 mass 기준. \"5x 균등분포 초과 비율\"은 "
                "**설명용 기준선이지 탐지 임계값이 아니다**(균등분포 1/196의 5배 = "
                f"{5*UNIFORM:.4f}).\n\n")
        f.write("| L | median(clean) | median(PatchFool) | median(LaVAN) | "
                "frac>5x unif (clean) | frac>5x unif (PF) | frac>5x unif (LaVAN) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['layer']} | {r['med_c']:.4f} | {r['med_p']:.4f} | {r['med_v']:.4f} | "
                    f"{r['frac_c']:.2f} | {r['frac_p']:.2f} | {r['frac_v']:.2f} |\n")
    print(f"\nSaved: {table_path}")

    for metric_key, fname, title in [
        ('top1_row_avg', '01_hist_top1_row_by_layer.png',
         'Top-1 mass (CLS row, head-averaged) by layer'),
        ('top1_col_avg', '01_hist_top1_col_by_layer.png',
         'Top-1 mass (column aggregate, head-averaged) by layer'),
        ('ent_row_avg', '01_hist_entropy_row_by_layer.png',
         'Normalized entropy (CLS row, head-averaged) by layer'),
    ]:
        p = os.path.join(out_dir, fname)
        plot_histograms(all_metrics, metric_key, n_layers, title, p)
        print(f"Saved: {p}")

    p = os.path.join(out_dir, '01_head_variability_L5_L12.png')
    plot_head_variability(repr_by_group, 0, [5, 12], p)
    print(f"Saved: {p}")

    p = os.path.join(out_dir, '01_layer_comparison_top1_row.png')
    plot_layer_comparison(all_metrics, 'top1_row_avg', n_layers,
                           'top-1 mass (CLS row, head-averaged)',
                           'PatchFool vs LaVAN vs Clean — median + IQR by layer', p)
    print(f"Saved: {p}")


if __name__ == '__main__':
    main()
