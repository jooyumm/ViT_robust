"""
experiments/system_comparison/viz.py — system_comparison_test.py가 저장한 npz를 읽어 all_switch
vs local_switch 비교 막대그래프를 다시 그린다 (GPU/재실험 불필요).

사용법:
  python viz.py  (experiments/system_comparison/ 안에서)
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = HERE

ALL_COLOR, LOCAL_COLOR = '#E94B3C', '#3B82F6'  # cost_comparison/paper_summary와 동일 배색


def main():
    candidates = glob.glob(os.path.join(RESULTS, 'system_comparison_n*.npz'))
    assert candidates, f"npz 없음: {RESULTS}에서 system_comparison_test.py를 먼저 돌릴 것"
    npz_path = max(candidates, key=os.path.getmtime)
    d = np.load(npz_path)
    print(f"Loaded: {npz_path}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # (a) 무조건 복원율
    ax = axes[0]
    vals = [float(d['rate_all']), float(d['rate_local'])]
    cis = [d['ci_all'], d['ci_local']]
    bars = ax.bar(['all_switch', 'local_switch'], vals, color=[ALL_COLOR, LOCAL_COLOR], alpha=0.85)
    for i, (v, ci) in enumerate(zip(vals, cis)):
        ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt='none', ecolor='black', capsize=5)
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_title(f'(a) Recovery rate (unconditional, n_attacked={int(d["n_attacked"])})')
    ax.grid(axis='y', alpha=0.3)

    # (b) 시스템 정확도
    ax = axes[1]
    labels = ['P16 alone', 'all_switch\nsystem', 'local_switch\nsystem']
    vals = [float(d['p16_only_acc']), float(d['sys_acc_all']), float(d['sys_acc_local'])]
    colors = ['#6B7280', ALL_COLOR, LOCAL_COLOR]
    ax.bar(labels, vals, color=colors, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_title(f'(b) System accuracy (n_eval={int(d["n_eval"])}, attacked)')
    ax.grid(axis='y', alpha=0.3)

    # (c) clean 오탐 비용
    ax = axes[2]
    labels = ['P16 clean', 'all_switch\napplied', 'local_switch\napplied']
    vals = [float(d['clean_p16_acc']), float(d['clean_all_acc']), float(d['clean_local_acc'])]
    ax.bar(labels, vals, color=colors, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_title('(c) Clean false-positive cost (unconditional)')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(
        f'System comparison: all_switch vs local_switch (same images, same detection outcome)\n'
        f'threshold={float(d["threshold"]):.4f}  FPR={float(d["fpr"]):.3f}  recall={float(d["recall"]):.3f}',
        fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(RESULTS, 'system_comparison_viz.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
