"""공격이 하나의 토큰으로 쏠리는지, 그게 모든 레이어에서 그런지 확인.
top-1 mass(CLS row, 헤드평균 — 가장 많이 보는 토큰 하나가 가져가는 비율)를 레이어별로
집계한다. GPU 필요 없음, 00_extract_attention.py의 npz만 읽는다.

산출물:
  01_hist_top1_row_by_layer.png   레이어별 top-1 mass 히스토그램(3그룹)
  01_layer_comparison_top1_row.png  레이어별 중앙값+IQR
  01_concentration_summary_n{N}.md  레이어별 중앙값 표
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_io import load_attention_data
from plotting import plot_histograms, plot_layer_comparison

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

    print(f"{'L':>3} | {'clean':>8} | {'PatchFool':>10} | {'LaVAN':>8} | {'PF-clean':>9}")
    print("-" * 55)
    rows = []
    for L in range(n_layers):
        med_c = np.median(all_metrics['clean']['top1_row_avg'][:, L])
        med_p = np.median(all_metrics['patchfool']['top1_row_avg'][:, L])
        med_v = np.median(all_metrics['lavan']['top1_row_avg'][:, L])
        print(f"{L+1:>3} | {med_c:>8.4f} | {med_p:>10.4f} | {med_v:>8.4f} | {med_p - med_c:>+9.4f}")
        rows.append((L + 1, med_c, med_p, med_v))

    table_path = os.path.join(out_dir, f'01_concentration_summary_n{args.num_samples}.md')
    with open(table_path, 'w') as f:
        f.write(f"# top-1 mass by layer (n={args.num_samples})\n\n")
        f.write("| L | clean | PatchFool | LaVAN |\n|---|---|---|---|\n")
        for L, c, p, v in rows:
            f.write(f"| {L} | {c:.4f} | {p:.4f} | {v:.4f} |\n")
    print(f"Saved: {table_path}")

    p = os.path.join(out_dir, '01_hist_top1_row_by_layer.png')
    plot_histograms(all_metrics, 'top1_row_avg', n_layers,
                     'Top-1 mass (CLS row, head-averaged) by layer', p)
    print(f"Saved: {p}")

    p = os.path.join(out_dir, '01_layer_comparison_top1_row.png')
    plot_layer_comparison(all_metrics, 'top1_row_avg', n_layers,
                          'top-1 mass (CLS row, head-averaged)',
                          'PatchFool vs LaVAN vs Clean — median + IQR by layer', p)
    print(f"Saved: {p}")


if __name__ == '__main__':
    main()
