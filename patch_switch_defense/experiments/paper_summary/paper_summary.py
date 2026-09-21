"""
experiments/paper_summary/paper_summary.py — all_switch vs local_switch 비교의 모든 근거를
하나의 표 + 하나의 그림으로 합친다. 새 GPU 실험이 아니라 기존 npz 5개를 읽어서 계산만 한다:

  - experiments/system_comparison/system_comparison_n*.npz   (복원율/시스템 정확도/clean 비용)
  - experiments/cost_comparison/cost_comparison.npz            (파이프라인 비용)
  - experiments/all_switch/joint_attack/07_joint_attack_test_n50.npz            (naive joint attack)
  - experiments/local_switch/joint_attack/17_local_swap_joint_attack_n50.npz     (naive joint attack)
  - experiments/all_switch/adaptive_evasion_full/14_..._n50_seed123.npz          (완전판 adaptive evasion)
  - experiments/local_switch/adaptive_evasion_full/18_..._n50_seed123.npz        (완전판 adaptive evasion)

§18(local_switch adaptive evasion)의 worst-case는 calibrated_threshold 기준(공격자가 실배포
임계값을 직접 알고 최적화하는, 더 보수적인 위협 모델)을 대표 숫자로 쓴다 — clean_max 기준은
참고치로만 표에 같이 싣는다(NARRATIVE_16_17_18.md §3 참고).

사용법:
  python paper_summary.py
  (system_comparison_test.py, cost_comparison_test.py를 먼저 돌려서 npz를 만들어둘 것)
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = HERE
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'experiments')):
    ROOT = os.path.dirname(ROOT)

RED, BLUE, GRAY = '#E94B3C', '#3B82F6', '#6B7280'


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def both_correct_mask(pred_a, pred_b, labels):
    return (pred_a == labels) & (pred_b == labels)


def joint_defeat_from_raw(npz_path):
    """all_switch의 07 npz는 원시 예측만 저장 — §8/§17과 같은 방식으로 직접 계산."""
    d = np.load(npz_path)
    labels = d['labels']
    base_mask = both_correct_mask(d['pred16_clean'], d['pred8_clean'], labels)
    fool16 = base_mask & (d['pred16_joint'] != labels)
    fool8 = base_mask & (d['pred8_joint'] != labels)
    n_base = int(base_mask.sum())
    n_fool_both = int((fool16 & fool8).sum())
    return n_fool_both, n_base


def main():
    sys_cand = glob.glob(os.path.join(ROOT, 'experiments/system_comparison/system_comparison_n*.npz'))
    cost_cand = glob.glob(os.path.join(ROOT, 'experiments/cost_comparison/cost_comparison.npz'))
    assert sys_cand, "experiments/system_comparison/에 npz 없음 — system_comparison_test.py 먼저 실행"
    assert cost_cand, "experiments/cost_comparison/에 npz 없음 — cost_comparison_test.py 먼저 실행"
    sysd = np.load(max(sys_cand, key=os.path.getmtime))
    costd = np.load(cost_cand[0])

    n_all_joint, n_all_base = joint_defeat_from_raw(
        os.path.join(ROOT, 'experiments/all_switch/joint_attack/07_joint_attack_test_n50.npz'))
    all_joint_rate = n_all_joint / n_all_base
    all_joint_ci = wilson_ci(n_all_joint, n_all_base)

    loc_joint_d = np.load(os.path.join(
        ROOT, 'experiments/local_switch/joint_attack/17_local_swap_joint_attack_n50.npz'))
    loc_joint_rate = float(loc_joint_d['joint_defeat_rate'])
    loc_joint_ci = wilson_ci(int(loc_joint_d['n_fool_both_joint']), int(loc_joint_d['n_base']))

    all_evasion_cand = glob.glob(os.path.join(
        ROOT, 'experiments/all_switch/adaptive_evasion_full/14_*_n50_seed123.npz'))
    loc_evasion_cand = glob.glob(os.path.join(
        ROOT, 'experiments/local_switch/adaptive_evasion_full/18_*_n50_seed123.npz'))
    assert all_evasion_cand and loc_evasion_cand, "§14/§18 npz가 없음 — 삭제되지 않았는지 확인"
    all_ev = np.load(all_evasion_cand[0])
    loc_ev = np.load(loc_evasion_cand[0])
    # §14: 두 target_bound 결과가 동일(15.8%) — calibrated_threshold를 대표로 씀(§18과 통일)
    all_worst_n = int(all_ev['calibrated_threshold_n_complete_defeat'])
    all_worst_base = int(all_ev['calibrated_threshold_n_base'])
    all_worst_rate = all_worst_n / all_worst_base
    all_worst_ci = wilson_ci(all_worst_n, all_worst_base)
    # §18: calibrated_threshold가 대표(21.4%), clean_max는 참고(11.9%) — NARRATIVE_16_17_18.md §3
    loc_worst_n = int(loc_ev['calibrated_threshold_n_complete_defeat'])
    loc_worst_base = int(loc_ev['calibrated_threshold_n_base'])
    loc_worst_rate = loc_worst_n / loc_worst_base
    loc_worst_ci = wilson_ci(loc_worst_n, loc_worst_base)
    loc_ref_n = int(loc_ev['clean_max_n_complete_defeat'])
    loc_ref_base = int(loc_ev['clean_max_n_base'])
    loc_ref_rate = loc_ref_n / loc_ref_base

    rows = [
        ('Recovery rate (unconditional, same images/detection)',
         f"{float(sysd['rate_all']):.1%}", f"{float(sysd['rate_local']):.1%}"),
        ('System accuracy (attacked eval set)',
         f"{float(sysd['sys_acc_all']):.1%}", f"{float(sysd['sys_acc_local']):.1%}"),
        ('Clean false-positive cost (delta vs P16 clean)',
         f"{float(sysd['clean_all_acc']) - float(sysd['clean_p16_acc']):+.1%}",
         f"{float(sysd['clean_local_acc']) - float(sysd['clean_p16_acc']):+.1%}"),
        ('Naive joint attack complete-defeat rate',
         f"{all_joint_rate:.1%} [{all_joint_ci[0]:.1%},{all_joint_ci[1]:.1%}]",
         f"{loc_joint_rate:.1%} [{loc_joint_ci[0]:.1%},{loc_joint_ci[1]:.1%}]"),
        ('Adaptive evasion worst-case (calibrated_threshold, representative)',
         f"{all_worst_rate:.1%} [{all_worst_ci[0]:.1%},{all_worst_ci[1]:.1%}]",
         f"{loc_worst_rate:.1%} [{loc_worst_ci[0]:.1%},{loc_worst_ci[1]:.1%}] (ref. clean_max {loc_ref_rate:.1%})"),
        ('Pipeline extra cost (escalate 시, batch=1)',
         f"{float(costd['all_extra_mean_ms']):.2f} ms",
         f"{float(costd['local_extra_mean_ms']):.2f} ms"),
    ]

    print(f"\n{'Metric':<58}{'all_switch':>28}{'local_switch':>40}")
    print("-" * 126)
    for name, a, l in rows:
        print(f"{name:<58}{a:>28}{l:>40}")

    md_path = os.path.join(RESULTS, 'paper_summary_table.md')
    os.makedirs(RESULTS, exist_ok=True)
    with open(md_path, 'w') as f:
        f.write("| Metric | all_switch | local_switch |\n|---|---|---|\n")
        for name, a, l in rows:
            f.write(f"| {name} | {a} | {l} |\n")
    print(f"\nSaved: {md_path}")

    # ── 4패널 그림: 복원율 | 시스템 정확도 | adaptive worst-case | 비용 ──────
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))

    ax = axes[0]
    vals = [float(sysd['rate_all']), float(sysd['rate_local'])]
    ax.bar(['all_switch', 'local_switch'], vals, color=[RED, BLUE], alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontweight='bold')
    ax.set_ylim(0, 1.1); ax.set_title('Recovery rate'); ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    vals = [float(sysd['p16_only_acc']), float(sysd['sys_acc_all']), float(sysd['sys_acc_local'])]
    ax.bar(['P16 alone', 'all_switch', 'local_switch'], vals, color=[GRAY, RED, BLUE], alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontweight='bold')
    ax.set_ylim(0, 1.1); ax.set_title('System accuracy'); ax.grid(axis='y', alpha=0.3)

    ax = axes[2]
    vals = [all_worst_rate, loc_worst_rate]
    cis = [all_worst_ci, loc_worst_ci]
    ax.bar(['all_switch\n(§14)', 'local_switch\n(§18, calibrated_threshold)'], vals,
           color=[RED, BLUE], alpha=0.85)
    for i, (v, ci) in enumerate(zip(vals, cis)):
        ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt='none', ecolor='black', capsize=5)
        ax.text(i, v + 0.03, f'{v:.2f}', ha='center', fontweight='bold')
    ax.set_ylim(0, 1.1); ax.set_title('Adaptive evasion worst-case'); ax.grid(axis='y', alpha=0.3)

    ax = axes[3]
    base = float(costd['base_mean_ms'])
    vals = [base, base + float(costd['all_extra_mean_ms']), base + float(costd['local_extra_mean_ms'])]
    ax.bar(['baseline\n(P16+detect)', 'all_switch\n(escalated)', 'local_switch\n(escalated)'],
           vals, color=[GRAY, RED, BLUE], alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f'{v:.2f}ms', ha='center', fontweight='bold')
    ax.set_title('Pipeline latency (batch=1)'); ax.grid(axis='y', alpha=0.3)

    fig.suptitle('all_switch vs local_switch — paper headline comparison', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = os.path.join(RESULTS, 'paper_summary_headline.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")


if __name__ == '__main__':
    main()
