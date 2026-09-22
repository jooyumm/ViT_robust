"""
experiments/characterization/02_sink_position.py — "attention이 어디로 몰리는가, 그
위치가 고정돼 있는가, 공격이 그 위치를 바꾸는가"만 다룬다(얼마나 몰리는지의 크기는
01_concentration.py 담당). GPU 필요 없음 — 00_extract_attention.py가 저장한 npz만 읽는다.

산출물 (results/characterization/, 전부 02_ 접두어로 이 스크립트가 만든 것임을 표시):
  - 02_sink_position_concentration.md   레이어별 top-3 최빈 위치가 전체 이미지의 몇 %를 차지하는가
  - 02_sink_position_match_rate.md      같은 이미지에서 clean→공격 후 argmax 위치 일치율
  - 02_attention_heatmap_grid_img{0,1}.png  clean/PatchFool/LaVAN x 12레이어 CLS-row 히트맵
  - 02_sorted_mass_by_layer_img0.png        "슬라이드 막대그래프"(정상=분산/공격=집중) 재현

범위 제한: 탐지 임계값/AUROC/flag 없음. "top-3 초과" 등은 전부 설명용 관찰이지 판정 기준이
아니다.

사용법:
  python 02_sink_position.py --num_samples 100
  (같은 폴더에 00_extract_attention.py로 만든 00_attention_data_n100.npz가 있어야 함)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # for `import data_io/plotting`

from data_io import load_attention_data, GROUPS
from plotting import plot_heatmap_grid, plot_sorted_mass_by_layer

import numpy as np


def position_concentration_table(all_metrics, n_layers, top_k_modes=3):
    """레이어x그룹별: 그 레이어에서 가장 흔한 top_k_modes개 argmax 위치가 전체 이미지의
    몇 %를 차지하는지("고정 위치"에 몰리는 정도). 탐지 임계값이 아니라 순수 관찰."""
    rows = []
    for L in range(n_layers):
        row = {'layer': L + 1}
        for g in GROUPS:
            idxs = all_metrics[g]['argmax_row_avg'][:, L].astype(int)
            vals, counts = np.unique(idxs, return_counts=True)
            order = np.argsort(-counts)
            top_positions = vals[order[:top_k_modes]]
            frac = np.isin(idxs, top_positions).mean()
            row[f'{g}_frac'] = frac
            row[f'{g}_positions'] = top_positions.tolist()
        rows.append(row)
    return rows


def position_match_table(all_metrics, n_layers):
    """레이어별: 같은 이미지에서 clean의 argmax 위치와 PatchFool/LaVAN 적용 후 argmax 위치가
    일치하는 비율(이미지 단위 매칭, 집계 평균 아님)."""
    rows = []
    for L in range(n_layers):
        clean_idx = all_metrics['clean']['argmax_row_avg'][:, L].astype(int)
        pf_idx = all_metrics['patchfool']['argmax_row_avg'][:, L].astype(int)
        lavan_idx = all_metrics['lavan']['argmax_row_avg'][:, L].astype(int)
        rows.append(dict(
            layer=L + 1,
            match_pf=(clean_idx == pf_idx).mean(),
            match_lavan=(clean_idx == lavan_idx).mean(),
        ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--top_k_modes', type=int, default=3)
    args = parser.parse_args()

    out_dir = HERE.replace('/experiments/', '/results/', 1)
    npz_path = os.path.join(out_dir, f'00_attention_data_n{args.num_samples}.npz')
    assert os.path.exists(npz_path), \
        f"{npz_path} 없음 — 먼저 00_extract_attention.py --num_samples {args.num_samples} 실행할 것"
    all_metrics, repr_by_group, n_layers = load_attention_data(npz_path)

    conc_rows = position_concentration_table(all_metrics, n_layers, top_k_modes=args.top_k_modes)
    conc_path = os.path.join(out_dir, f'02_sink_position_concentration_n{args.num_samples}.md')
    with open(conc_path, 'w') as f:
        f.write(f"# Sink position concentration — top-{args.top_k_modes} most common argmax "
                f"positions' share of all images (n={args.num_samples})\n\n")
        f.write("CLS-row, head-averaged argmax(196 patch index) 기준. "
                "\"top-3 위치가 몇 %를 차지하는가\"로 \"고정 위치\" 주장을 정량화 — "
                "탐지 임계값이 아니라 순수 관찰.\n\n")
        f.write("| L | clean frac | clean top-3 idx | PatchFool frac | PF top-3 idx | "
                "LaVAN frac | LaVAN top-3 idx |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in conc_rows:
            f.write(f"| {r['layer']} | {r['clean_frac']:.2f} | {r['clean_positions']} | "
                    f"{r['patchfool_frac']:.2f} | {r['patchfool_positions']} | "
                    f"{r['lavan_frac']:.2f} | {r['lavan_positions']} |\n")
    print(f"Saved: {conc_path}")

    match_rows = position_match_table(all_metrics, n_layers)
    match_path = os.path.join(out_dir, f'02_sink_position_match_rate_n{args.num_samples}.md')
    with open(match_path, 'w') as f:
        f.write(f"# Clean -> attacked argmax position match rate, per image "
                f"(n={args.num_samples})\n\n")
        f.write("같은 이미지에서 clean일 때의 top-1 위치와 공격 적용 후 top-1 위치가 "
                "같은 patch인지 이미지별로 비교한 비율. 높으면 \"기존 sink 증폭\", "
                "낮으면 \"공격이 새 위치를 만듦\"에 가깝다.\n\n")
        f.write("| L | match rate (clean vs PatchFool) | match rate (clean vs LaVAN) |\n")
        f.write("|---|---|---|\n")
        for r in match_rows:
            f.write(f"| {r['layer']} | {r['match_pf']:.2f} | {r['match_lavan']:.2f} |\n")
    print(f"Saved: {match_path}")

    print(f"\n{'L':>3} | {'top3-conc clean':>16} | {'top3-conc PF':>13} | {'top3-conc LaVAN':>16} | "
          f"{'match PF':>9} | {'match LaVAN':>11}")
    print("-" * 85)
    for cr, mr in zip(conc_rows, match_rows):
        print(f"{cr['layer']:>3} | {cr['clean_frac']:>16.2f} | {cr['patchfool_frac']:>13.2f} | "
              f"{cr['lavan_frac']:>16.2f} | {mr['match_pf']:>9.2f} | {mr['match_lavan']:>11.2f}")

    for img_idx in range(min(repr_by_group['clean']['row_avg'].shape[0], 2)):
        p = os.path.join(out_dir, f'02_attention_heatmap_grid_img{img_idx}.png')
        plot_heatmap_grid(repr_by_group, img_idx, n_layers, p)
        print(f"Saved: {p}")

    p = os.path.join(out_dir, '02_sorted_mass_by_layer_img0.png')
    plot_sorted_mass_by_layer(repr_by_group, 0, n_layers, p)
    print(f"Saved: {p}")


if __name__ == '__main__':
    main()
