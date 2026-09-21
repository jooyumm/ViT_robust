"""
adaptive_evasion_full/viz.py — [탐색적, 롤백 가능] 14_adaptive_evasion_full_n50_seed123.npz를
읽어서 그림만 다시 그린다 (GPU/재실험 불필요).

§7(회피 없는 joint attack) 참고값은 이 폴더가 자기 완결적이도록 07_joint_attack의 확정된
결과를 상수로 직접 박아넣었다(adaptive_evasion_full_test.py 자신도 PREV_FOOL_BOTH
등으로 이렇게 참조하던 관례를 그대로 따름). 원본 재확인:
  results/all_switch/joint_attack/07_joint_attack_test_n50.npz

사용법:
  python viz.py  (experiments/all_switch/adaptive_evasion_full/ 안에서)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = HERE

RED, BLUE, DARK = '#E94B3C', '#2563EB', '#1F2937'

# §7 참고값 (07_joint_attack_test_n50.npz + README §7에서 도출된 확정 수치)
PREV_FOOL_BOTH, PREV_N_BASE = 7, 38
PREV_FLAG = 0.20
PREV_WORST = 6  # "둘 다 속은 사례 중 탐지되는 것 1/7" -> 미탐지 6/7


def main():
    d = np.load(os.path.join(RESULTS, '14_adaptive_evasion_full_n50_seed123.npz'))

    conditions = ['§7\n(no evasion)', 'clean_max\nconstraint', 'calibrated_threshold\nconstraint']
    fool_both = [PREV_FOOL_BOTH / PREV_N_BASE * 100,
                 float(d['clean_max_rate_both']) * 100,
                 float(d['calibrated_threshold_rate_both']) * 100]
    flag_rate = [PREV_FLAG * 100,
                 d['clean_max_flagged_by_threshold'].mean() * 100,
                 d['calibrated_threshold_flagged_by_threshold'].mean() * 100]
    worst = [PREV_WORST / PREV_N_BASE * 100,
             float(d['clean_max_n_complete_defeat']) / float(d['clean_max_n_base']) * 100,
             float(d['calibrated_threshold_n_complete_defeat']) / float(d['calibrated_threshold_n_base']) * 100]

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle('§14. Full adaptive attack — explicit evasion constraint barely changes §7',
                 fontsize=14, fontweight='bold')
    x = np.arange(len(conditions))
    width = 0.25
    ax.bar(x - width, fool_both, width, color=RED, edgecolor='white', linewidth=1.2, label='both fooled (complete defeat)')
    ax.bar(x, flag_rate, width, color=BLUE, edgecolor='white', linewidth=1.2, label='flagged by deployed threshold')
    ax.bar(x + width, worst, width, color=DARK, edgecolor='white', linewidth=1.2, label='true worst-case (defeat + undetected)')
    for xs, vals in [(x - width, fool_both), (x, flag_rate), (x + width, worst)]:
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 1, f'{v:.1f}', ha='center', fontsize=10, fontweight='bold', color=DARK)
    ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=10.5)
    ax.set_ylabel('%'); ax.set_ylim(0, 35)
    ax.legend(fontsize=12, frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    out = os.path.join(RESULTS, '14_adaptive_evasion_full_viz.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
