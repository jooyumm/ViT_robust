"""
joint_attack/viz.py — [탐색적, 롤백 가능] 07_joint_attack_test_n50.npz를
읽어서 그림만 다시 그린다 (GPU/재실험 불필요).

사용법:
  python viz.py  (experiments/all_switch/joint_attack/ 안에서)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = HERE

RED, BLUE, GRAY, DARK = '#E94B3C', '#2563EB', '#6B7280', '#1F2937'
FS_TITLE, FS_SUB, FS_TICK, FS_VAL = 14, 12, 10.5, 11


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)


def joint_stats(pred16_clean, pred8_clean, pred16_x, pred8_x, labels):
    both_ok = (pred16_clean == labels) & (pred8_clean == labels)
    n_base = both_ok.sum()
    fool16 = both_ok & (pred16_x != labels)
    fool8 = both_ok & (pred8_x != labels)
    fool_both = (fool16 & fool8).sum()
    return int(fool_both), int(n_base)


def main():
    d = np.load(os.path.join(RESULTS, '07_joint_attack_test_n50.npz'))
    both_ok = (d['pred16_clean'] == d['labels']) & (d['pred8_clean'] == d['labels'])
    fool16_single = both_ok & (d['pred16_single'] != d['labels'])
    n_transfer_denom = int(fool16_single.sum())
    n_single, _ = joint_stats(d['pred16_clean'], d['pred8_clean'],
                               d['pred16_single'], d['pred8_single'], d['labels'])
    transfer_rate = n_single / max(n_transfer_denom, 1) * 100

    n_joint, n_base2 = joint_stats(d['pred16_clean'], d['pred8_clean'],
                                    d['pred16_joint'], d['pred8_joint'], d['labels'])
    joint_rate = n_joint / n_base2 * 100

    flagged = (d['score_joint'] > d['threshold'])
    flag_rate = flagged.mean() * 100
    both_fooled_joint = both_ok & (d['pred16_joint'] != d['labels']) & (d['pred8_joint'] != d['labels'])
    n_both_fooled = int(both_fooled_joint.sum())
    flag_among_fooled = (flagged & both_fooled_joint).sum() / max(n_both_fooled, 1) * 100

    lo_t, hi_t = wilson_ci(n_single, max(n_transfer_denom, 1))
    lo_j, hi_j = wilson_ci(n_joint, n_base2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('§7. Joint attack — an attacker who knows the P16+P8 defense (n=50, seed=123)',
                 fontsize=FS_TITLE, fontweight='bold')

    xs = [0, 1]
    heights = [transfer_rate, joint_rate]
    yerr = [[transfer_rate - lo_t * 100, joint_rate - lo_j * 100],
            [hi_t * 100 - transfer_rate, hi_j * 100 - joint_rate]]
    ax1.bar(xs, heights, color=[GRAY, RED], width=0.55, edgecolor='white',
            linewidth=1.2, yerr=yerr, capsize=5)
    for xi, h in zip(xs, heights):
        ax1.text(xi, h + 3, f'{h:.1f}%', ha='center', fontsize=FS_VAL, fontweight='bold', color=DARK)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(['P16-only attack\n(accidental transfer)', 'P16+P8\njoint optimization'])
    ax1.set_ylabel('both models fooled (%)'); ax1.set_ylim(0, 45)
    ax1.set_title('How much more dangerous is knowing the defense?', fontsize=FS_SUB)
    _style_ax(ax1)

    xs2 = [0, 1]
    h2 = [flag_rate, flag_among_fooled]
    ax2.bar(xs2, h2, color=BLUE, edgecolor='white', linewidth=1.2, width=0.6)
    for xi, h in zip(xs2, h2):
        ax2.text(xi, h + 2, f'{h:.1f}', ha='center', fontsize=FS_VAL, fontweight='bold', color=DARK)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(['flagged, all 50', 'flagged, among the\n"both fooled" cases'])
    ax2.set_ylabel('%'); ax2.set_ylim(0, 100)
    ax2.set_title('Detector also weakens under joint attack', fontsize=FS_SUB)
    _style_ax(ax2)

    out = os.path.join(RESULTS, '07_joint_attack_test_viz.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"[참고, §8 viz.py에서 재사용] joint_rate={joint_rate:.4f} n_base={n_base2} "
          f"lo={lo_j:.4f} hi={hi_j:.4f}")


if __name__ == '__main__':
    main()
