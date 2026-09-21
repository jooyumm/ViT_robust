"""
diversity_diagnostic/viz.py — [탐색적, 롤백 가능] 08_diversity_test_n50_seed123.npz를
읽어서 §7 대비 헤드라인 비교 그림을 다시 그린다 (GPU/재실험 불필요).

§7(다른 patch size: P16 vs P8) 참고값은 이 폴더가 자기 완결적이도록(다른 실험 폴더의 .npz를
안 열고) 07_joint_attack_test_n50.npz에서 이미 확정된 숫자를 상수로 직접 박아넣었다 —
diversity_test.py 자신도 원래 이렇게 print 문에 §7 수치를 참고용 상수로 인용하던
관례를 그대로 따름. 원본 값 재확인하려면:
  experiments/all_switch/joint_attack/07_joint_attack_test_n50.npz

사용법:
  python viz.py  (experiments/diversity_diagnostic/ 안에서)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = HERE

RED, GRAY, DARK = '#E94B3C', '#6B7280', '#1F2937'
FS_TITLE, FS_SUB = 14, 12

# §7 참고값 (07_joint_attack_test_n50.npz에서 도출된 확정 수치, 하드코딩 출처 명시)
JOINT_RATE, JOINT_N_BASE, JOINT_LO, JOINT_HI = 18.4, 38, 9.2, 33.4


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
    ax.tick_params(labelsize=10.5)


def main():
    d = np.load(os.path.join(RESULTS, '08_diversity_test_n50_seed123.npz'))
    both_ok = (d['predA_clean'] == d['labels']) & (d['predB_clean'] == d['labels'])
    n_base_div = int(both_ok.sum())
    fool_both = both_ok & (d['predA_joint'] != d['labels']) & (d['predB_joint'] != d['labels'])
    n_div = int(fool_both.sum())
    div_rate = n_div / n_base_div * 100
    lo_d, hi_d = wilson_ci(n_div, n_base_div)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    fig.suptitle('Key finding: robustness comes from "different patch size", not "different model"\n'
                '(same 50 images, seed=123, paired)', fontsize=FS_TITLE, fontweight='bold')

    xs = [0, 1]
    heights = [JOINT_RATE, div_rate]
    labels_x = [f'Different patch size\n(P16 vs P8)\nn={JOINT_N_BASE}',
                f'Same patch size,\ndifferent training (A vs B)\nn={n_base_div}']
    yerr = [[JOINT_RATE - JOINT_LO, div_rate - lo_d * 100],
            [JOINT_HI - JOINT_RATE, hi_d * 100 - div_rate]]
    ax.bar(xs, heights, color=[RED, GRAY], width=0.5, edgecolor='white', linewidth=1.4,
           yerr=yerr, capsize=6)
    for xi, h in zip(xs, heights):
        ax.text(xi, h + 4, f'{h:.1f}%', ha='center', fontsize=15, fontweight='bold', color=DARK)
    ax.set_xticks(xs); ax.set_xticklabels(labels_x, fontsize=FS_SUB)
    ax.set_ylabel('Joint attack complete-defeat rate (%, 95% Wilson CI)')
    ax.set_ylim(0, 100)
    _style_ax(ax)

    out = os.path.join(RESULTS, '08_diversity_diagnostic_headline.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
