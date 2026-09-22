"""
09_protocol_c/viz.py — [탐색적, 롤백 가능] 09_protocol_c_tokenmatch_np4.npz를
읽어서 그림만 다시 그린다 (GPU/재실험 불필요).

사용법:
  python 09_protocol_c/viz.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')

RED, BLUE, DARK = '#E94B3C', '#2563EB', '#1F2937'
FS_TITLE, FS_SUB, FS_VAL = 14, 12, 11


def _style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)


def _bars(ax, x, heights, color):
    ax.bar(x, heights, color=color, edgecolor='white', linewidth=1.2, width=0.6)
    for xi, h in zip(x, heights):
        ax.text(xi, h + 2, f'{h:.1f}', ha='center', fontsize=FS_VAL, fontweight='bold', color=DARK)


def main():
    d = np.load(os.path.join(RESULTS, '09_protocol_c_tokenmatch_np4.npz'))
    P, RA, ASR = d['P'], d['RA'], d['ASR']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle('§9. Protocol C — attacked TOKEN COUNT fixed across P8/16/32 (num_patch=4, n=50)',
                 fontsize=FS_TITLE, fontweight='bold')
    x = np.arange(len(P))
    _bars(ax1, x, RA, RED)
    ax1.set_xticks(x); ax1.set_xticklabels([f'P={p}' for p in P])
    ax1.set_ylabel('Robust Accuracy %'); ax1.set_ylim(0, 100)
    ax1.set_title('RA (higher = more robust)', fontsize=FS_SUB); _style_ax(ax1)

    _bars(ax2, x, ASR, BLUE)
    ax2.set_xticks(x); ax2.set_xticklabels([f'P={p}' for p in P])
    ax2.set_ylabel('Attack Success Rate %'); ax2.set_ylim(0, 105)
    ax2.set_title('ASR (lower = more robust)', fontsize=FS_SUB); _style_ax(ax2)

    out = os.path.join(RESULTS, '09_protocol_c_viz.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
