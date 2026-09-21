"""
experiments/cost_comparison/viz.py — cost_comparison_test.py가 저장한 npz를 읽어
(a) baseline/all_switch/local_switch latency 막대그래프, (b) E[cost(pi)] 곡선(둘 다)을
다시 그린다 (GPU/재실험 불필요).

사용법:
  python viz.py  (experiments/cost_comparison/ 안에서)
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = HERE

RED, GRAY, BLUE = '#E94B3C', '#6B7280', '#3B82F6'


def main():
    candidates = glob.glob(os.path.join(RESULTS, 'cost_comparison.npz'))
    assert candidates, f"npz 없음: {RESULTS}에서 cost_comparison_test.py를 먼저 돌릴 것"
    d = np.load(candidates[0])
    print(f"Loaded: {candidates[0]}  (device_type={d['device_type']})")

    has_pi = 'pi_pis' in d.files
    fig, axes = plt.subplots(1, 2 if has_pi else 1, figsize=(14 if has_pi else 7, 5.5))
    if not has_pi:
        axes = [axes]

    # (a) latency bar
    ax = axes[0]
    labels = ['baseline\n(P16+detect+localize)', '+ all_switch\nextra', '+ local_switch\nextra']
    means = [float(d['base_mean_ms']), float(d['all_extra_mean_ms']), float(d['local_extra_mean_ms'])]
    stds = [float(d['base_std_ms']), float(d['all_extra_std_ms']), float(d['local_extra_std_ms'])]
    colors = [GRAY, RED, BLUE]
    ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    for i, v in enumerate(means):
        ax.text(i, v + max(means) * 0.03, f'{v:.2f}ms', ha='center', fontweight='bold')
    ax.set_ylabel('latency (ms, batch=1)')
    ax.set_title('Pipeline cost breakdown (batch=1)')
    ax.grid(axis='y', alpha=0.3)

    if has_pi:
        ax = axes[1]
        pis = d['pi_pis'] * 100
        ax.plot(pis, d['pi_exp_all_ms'], color=RED, linewidth=2.5, label='all_switch: E[cost(π)]')
        ax.plot(pis, d['pi_exp_local_ms'], color=BLUE, linewidth=2.5, label='local_switch: E[cost(π)]')
        ax.axhline(float(d['base_mean_ms']), color=GRAY, linestyle=':', linewidth=1.5,
                    label='baseline (P16 alone)')
        ax.axhline(float(d['pi_naive_ms']), color='black', linestyle='--', linewidth=1.2,
                    label='naive: always escalate (all_switch)')
        ax.set_xlabel('attack prevalence π (%)')
        ax.set_ylabel('expected latency (ms)')
        ax.set_title(f"Expected cost vs. prevalence (FPR={float(d['pi_fpr']):.3f}, "
                      f"recall={float(d['pi_recall']):.3f})")
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(RESULTS, 'cost_comparison_viz.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
