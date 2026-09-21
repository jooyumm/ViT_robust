"""
experiments/attack_type_comparison/attack_type_comparison_viz.py — 공격 유형별(PatchFool vs
LaVAN) 탐지 성능과 방어 효과를 한 그림에 비교한다. 새 GPU 실험이 아니라 기존 npz 2개를 읽어서
계산만 하는 분석 스크립트다(paper_summary.py와 동일한 성격):

  - experiments/detection_localization/signature/01_layer_sweep_P16_raw.npz
      (§1 원자료: clean/LaVAN/PatchFool 각 30장의 레이어별 top4_mass raw score)
      여기서 L=12 raw score로 AUROC(PatchFool vs Clean), AUROC(LaVAN vs Clean)를 직접
      재계산한다(README에 이미 인용된 0.879/0.363을 하드코딩하지 않고 원자료에서 재현 — 재현
      확인 결과 소수점까지 일치함).
  - experiments/system_comparison/system_comparison_n250.npz
      (PatchFool에 대한 무방어/all_switch/local_switch 시스템 RA — 8.7%/66.0%/65.3%. RA(Robust
      Accuracy) = adversarial 이미지 기준 정확도, patch_size_tradeoff/metrics.py 정의와 동일한
      용어 — CA(clean accuracy)가 아님, eval_labels 대비 전부 공격받은 이미지에서만 계산됨)

LaVAN에 대해서는 이 프로젝트에서 "방어 적용 후" 정확도를 측정한 적이 없다(system_comparison은
PatchFool만 공격으로 씀) — AUROC가 chance 이하(0.363)라 탐지기가 원리적으로 LaVAN을 못 잡고,
그 결과 escalate 로직이 사실상 무작위로만 발동해 방어 효과가 없다는 것이 핵심 주장이므로, 없는
수치를 만들어내지 않고 "정량 측정 안 됨 / 무방어와 동일" 텍스트로만 표시한다.

사용법:
  python attack_type_comparison_viz.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'experiments')):
    ROOT = os.path.dirname(ROOT)
RESULTS = HERE

GREEN, RED_FAIL = '#16A34A', '#DC2626'
GRAY, RED_ALL, BLUE_LOCAL = '#6B7280', '#E94B3C', '#3B82F6'


def auroc(pos, neg):
    """Mann-Whitney U 기반 AUROC (sklearn 없이). pos/neg: 이상치 점수 1D 배열."""
    n1, n2 = len(pos), len(neg)
    all_scores = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(all_scores)) + 1
    rank_pos = ranks[:n1].sum()
    u = rank_pos - n1 * (n1 + 1) / 2
    return u / (n1 * n2)


def main():
    sig = np.load(os.path.join(ROOT, 'experiments/detection_localization/signature/01_layer_sweep_P16_raw.npz'))
    pf, clean, lavan = sig['12_raw_pf_top4'], sig['12_raw_clean_top4'], sig['12_raw_lavan_top4']
    auroc_pf = auroc(pf, clean)
    auroc_lavan = auroc(lavan, clean)
    print(f"AUROC PatchFool vs Clean (L=12, raw, n={len(pf)}/{len(clean)}): {auroc_pf:.4f}")
    print(f"AUROC LaVAN vs Clean     (L=12, raw, n={len(lavan)}/{len(clean)}): {auroc_lavan:.4f}")

    sysd = np.load(os.path.join(ROOT, 'experiments/system_comparison/system_comparison_n250.npz'))
    p16_acc = float(sysd['p16_only_acc']) * 100
    all_acc = float(sysd['sys_acc_all']) * 100
    local_acc = float(sysd['sys_acc_local']) * 100
    print(f"PatchFool system RA: P16 {p16_acc:.1f}% -> all_switch {all_acc:.1f}% / "
          f"local_switch {local_acc:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- (a) Detection: is the attack caught at all? (AUROC, threshold-free) ----
    ax = axes[0]
    cats = ['PatchFool', 'LaVAN']
    vals = [auroc_pf, auroc_lavan]
    colors = [GREEN, RED_FAIL]
    bars = ax.bar(cats, vals, color=colors, alpha=0.88, width=0.55)
    ax.axhline(0.5, color=RED_FAIL, linestyle='--', linewidth=1.5, zorder=0)
    ax.text(1.48, 0.53, 'chance', color=RED_FAIL, fontsize=9.5, ha='right', style='italic')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f'{v:.3f}', ha='center',
                fontweight='bold', fontsize=14, color=bar.get_facecolor())
    ax.text(1, auroc_lavan / 2, 'undetectable', ha='center', va='center',
            fontsize=10, color='white', fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('AUROC')
    ax.set_title('(a) Detected?', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25)

    # ---- (b) Given an attacked image: P16 alone vs. partial-P8 vs. full-P8 (PatchFool only) ----
    ax = axes[1]
    x_pf = np.array([0, 1, 2])
    pf_vals = [p16_acc, local_acc, all_acc]
    pf_colors = [GRAY, BLUE_LOCAL, RED_ALL]
    pf_labels = ['P16\n(no defense)', 'local_switch\n(partial P8)', 'all_switch\n(full P8)']
    bars = ax.bar(x_pf, pf_vals, color=pf_colors, alpha=0.88, width=0.6)
    for bar, v in zip(bars, pf_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f'{v:.1f}%', ha='center',
                fontweight='bold', fontsize=12, color=bar.get_facecolor())

    ax.set_xticks(x_pf)
    ax.set_xticklabels(pf_labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Robust Accuracy (RA, %)')
    ax.set_title('(b) Defense effect (PatchFool)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25)

    fig.suptitle('Detection & Defense by Attack Type', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()

    os.makedirs(RESULTS, exist_ok=True)
    out_png = os.path.join(RESULTS, 'attack_type_comparison_viz.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_png}")

    np.savez(os.path.join(RESULTS, 'attack_type_comparison.npz'),
             auroc_patchfool=auroc_pf, auroc_lavan=auroc_lavan,
             p16_only_acc=p16_acc, all_switch_acc=all_acc, local_switch_acc=local_acc)
    print(f"Saved: {os.path.join(RESULTS, 'attack_type_comparison.npz')}")


if __name__ == '__main__':
    main()
