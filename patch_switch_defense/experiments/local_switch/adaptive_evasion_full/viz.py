"""
18_local_swap_adaptive_evasion_full/viz.py — [탐색적, 롤백 가능] §18 결과(두 target_bound)를
§17(회피 없음)·§14(전체 재분류의 완전판 adaptive, 참고용 상수)와 나란히 비교.
GPU/재실험 불필요.

사용법:
  python viz.py [--n 50] [--seed 123]
"""
import argparse
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = HERE

RED, BLUE, GREEN, GRAY, DARK = '#E94B3C', '#3B82F6', '#22A559', '#6B7280', '#1F2937'

S17_NO_EVASION = 0.167          # §17: local-swap joint attack, 탐지 회피 없음
S14_WORST_CASE = 0.158          # §14: 전체 재분류판, 완전판 adaptive worst-case (참고)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    if args.n is not None:
        npz_path = os.path.join(RESULTS, f'18_local_swap_adaptive_evasion_full_n{args.n}_seed{args.seed}.npz')
    else:
        candidates = glob.glob(os.path.join(RESULTS, '18_local_swap_adaptive_evasion_full_n*.npz'))
        assert candidates, f"결과 npz를 못 찾음: {RESULTS}"
        npz_path = max(candidates, key=os.path.getmtime)
    d = np.load(npz_path)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.suptitle('§18. Full adaptive evasion on local-swap vs §14 (full reclassification)',
                 fontsize=14, fontweight='bold', y=1.0)

    labels = ['S17\nno evasion', 'S18\nclean_max bound', 'S18\ncalibrated_threshold bound',
              'S14 (reference)\nfull reclass. worst-case']
    heights = [S17_NO_EVASION * 100,
               float(d['clean_max_n_complete_defeat']) / float(d['clean_max_n_base']) * 100,
               float(d['calibrated_threshold_n_complete_defeat']) / float(d['calibrated_threshold_n_base']) * 100,
               S14_WORST_CASE * 100]
    colors = [GRAY, BLUE, BLUE, GREEN]
    xs = list(range(4))

    ax.bar(xs, heights, color=colors, edgecolor='white', linewidth=1.3, width=0.6)
    for xi, h in zip(xs, heights):
        ax.text(xi, h + 1.5, f'{h:.1f}%', ha='center', fontsize=13, fontweight='bold', color=DARK)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel('Complete defeat + undetected (%)', fontsize=12)
    ax.set_ylim(0, 40)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.5, 33, '95% CIs overlap heavily at n=42 (Wilson) -\nevasion constraint doesn\'t clearly move the needle,\nsame pattern §14 found for the full-reclassification design',
           fontsize=9, color=GRAY, style='italic')

    out = os.path.join(RESULTS, '18_local_swap_adaptive_evasion_full_viz.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
