"""
17_local_swap_joint_attack/viz.py — [탐색적, 롤백 가능] §16(local-swap)에 대한 joint attack
결과를, §7(P16+P8 전체)·§8(정렬된 P16-A/B)의 완전 무력화율과 나란히 비교하는 그림.
GPU/재실험 불필요 (기존 npz + §7/§8 확정 수치를 상수로 인용).

사용법:
  python viz.py [--n 50]
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

# §7/§8에서 이미 확정된 숫자 (npz를 다시 열지 않고 상수로 인용 — 프로젝트 관행)
S7_JOINT_DEFEAT = 0.184        # P16+P8 전체 joint attack, 완전 무력화율
S8_ALIGNED_DEFEAT = 0.744      # 정렬된 P16-A/B joint attack, 완전 무력화율 (페어링)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    if args.n is not None:
        npz_path = os.path.join(RESULTS, f'17_local_swap_joint_attack_n{args.n}.npz')
    else:
        candidates = glob.glob(os.path.join(RESULTS, '17_local_swap_joint_attack_n*.npz'))
        assert candidates, f"결과 npz를 못 찾음: {RESULTS}"
        npz_path = max(candidates, key=os.path.getmtime)
    d = np.load(npz_path)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle('§17. Does local-swap fall into the "aligned representation" trap (§8)?',
                 fontsize=14.5, fontweight='bold', y=1.0)

    labels = ['S8: aligned\n(P16-A vs P16-B)', 'S7: diverse\n(P16 vs full P8)',
              'S17: local-swap\n(P16 vs local P8-bridge)']
    heights = [S8_ALIGNED_DEFEAT * 100, S7_JOINT_DEFEAT * 100,
               float(d['joint_defeat_rate']) * 100]
    colors = [RED, GREEN, BLUE]
    xs = list(range(3))

    ax.bar(xs, heights, color=colors, edgecolor='white', linewidth=1.3, width=0.55)
    for xi, h in zip(xs, heights):
        ax.text(xi, h + 2, f'{h:.1f}%', ha='center', fontsize=15, fontweight='bold', color=DARK)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylabel('Joint attack complete-defeat rate (%)', fontsize=12)
    ax.set_ylim(0, 90)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.text(2, heights[2] + 10, 'close to S7,\nfar from S8 → no trap',
           ha='center', fontsize=10, color=BLUE, style='italic')

    out = os.path.join(RESULTS, '17_local_swap_joint_attack_viz.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
