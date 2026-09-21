"""
evasion_robustness/viz.py — [탐색적, 롤백 가능] 03_evasion_test_P16.npz +
03_range_evasion_P16.npz를 읽어서 회피 시도 3종 비교 그림을 다시 그린다 (GPU/재실험 불필요).

사용법:
  python viz.py  (experiments/detection_localization/evasion_robustness/ 안에서)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = HERE

RED, BLUE, DARK = '#E94B3C', '#2563EB', '#1F2937'
FS_TITLE, FS_SUB, FS_TICK, FS_VAL = 14, 12, 10.5, 11


def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)


def recall_at(sorted_idx, true_idx, K):
    hit = (sorted_idx[:, :K] == true_idx[:, None]).any(axis=1)
    return hit.mean() * 100


def main():
    d1 = np.load(os.path.join(RESULTS, '03_evasion_test_P16.npz'))
    d2 = np.load(os.path.join(RESULTS, '03_range_evasion_P16.npz'))

    groups = [
        ('Raw detector\n(lowest-saliency attack)', d1['sorted_raw'], d1['true_idx']),
        ('Normalized detector B\n(same attack, failed)', d1['sorted_norm'], d1['true_idx']),
        ('Raw detector\n(re-targeted to measured range)', d2['sorted_raw'], d2['true_idx']),
    ]
    Ks = [1, 4]
    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.suptitle('§3. Three evasion attempts — recall@K (raw detector holds, normalized one fails)',
                 fontsize=FS_TITLE, fontweight='bold')
    for i, K in enumerate(Ks):
        vals = [recall_at(s, t, K) for _, s, t in groups]
        offset = (i - 0.5) * width
        color = RED if K == 1 else BLUE
        ax.bar(x + offset, vals, width=width, color=color, edgecolor='white',
               linewidth=1.2, label=f'recall@{K}')
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 1.5, f'{v:.1f}', ha='center', va='bottom', fontsize=FS_VAL,
                    fontweight='bold', color=DARK)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=FS_TICK)
    ax.set_ylabel('%'); ax.set_ylim(0, 110)
    ax.legend(fontsize=FS_SUB, frameon=False)
    _style_ax(ax)

    out = os.path.join(RESULTS, '03_evasion_attempts_summary.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
