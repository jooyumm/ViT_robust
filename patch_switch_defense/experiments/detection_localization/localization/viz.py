"""
localization/viz.py — [탐색적, 롤백 가능] 02_localization_P16.npz를
읽어서 그림만 다시 그린다 (GPU/재실험 불필요). 원자료는 localization.py가 만든 것.

사용법:
  python viz.py  (experiments/detection_localization/localization/ 안에서)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = HERE

RED = '#E94B3C'
DARK = '#1F2937'
FS_TITLE, FS_SUB, FS_TICK, FS_VAL = 14, 12, 10.5, 11


def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)


def main():
    d = np.load(os.path.join(RESULTS, '02_localization_P16.npz'))
    recalls = [d['hit_at_1'].mean() * 100, d['hit_at_4'].mean() * 100, d['hit_at_8'].mean() * 100]
    cheby = d['cheby']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('§2. Localization — does L=12 raw attention find the actually attacked token? (n=30)',
                 fontsize=FS_TITLE, fontweight='bold')

    ax1.bar([0, 1, 2], recalls, color=RED, edgecolor='white', linewidth=1.2, width=0.6)
    for xi, h in zip([0, 1, 2], recalls):
        ax1.text(xi, h + 2, f'{h:.1f}', ha='center', fontsize=FS_VAL, fontweight='bold', color=DARK)
    ax1.set_xticks([0, 1, 2]); ax1.set_xticklabels(['recall@1', 'recall@4', 'recall@8'])
    ax1.set_ylabel('%'); ax1.set_ylim(0, 105)
    ax1.set_title('Fraction where true attacked token is in top-K', fontsize=FS_SUB)
    _style_ax(ax1)

    bins = np.arange(0, cheby.max() + 2) - 0.5
    ax2.hist(cheby, bins=bins, color=RED, edgecolor='white')
    ax2.set_xlabel('grid distance (Chebyshev, top-1 vs true location)')
    ax2.set_ylabel('count')
    ax2.set_title(f'mean={cheby.mean():.2f}, exact match={float((cheby==0).mean()):.1%}',
                  fontsize=FS_SUB)
    _style_ax(ax2)

    out = os.path.join(RESULTS, '02_localization_viz.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
