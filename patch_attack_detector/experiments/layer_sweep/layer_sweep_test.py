"""
레이어 1~12 전체에서 Top-K Mass 탐지 신호의 AUROC/FPR/recall을 독립적으로 측정 
patch_switch_defense/system_comparison이 확정한 L=12 하나만 보는 대신, "L=12까지 봐야 신호가 확실히 분리된다"는 주장 자체를 전체 곡선으로 뒷받침/반박).

배경
----
지금까지는 L=6과 L=12 두 지점만 확인
두 점만으로는 그 사이·양옆에서 AUROC가 매끄럽게 오르는지, 특정 레이어에서 급격히 꺾이는지, L=12가 정말 최적점인지 알 수 없다

이 실험은 detector.py의 함수(collect_layer_attn/raw_at_layer/top4_mass)를 레이어 인덱스만
바꿔가며 그대로 재사용해서 그 곡선 전체를 채운다 — 새 탐지 로직을 만들지 않는다.

방법
----
1) patch_switch_defense/experiments/system_comparison과 완전히 동일한 표본·공격 설정을 쓴다
   (n=250, seed=42, calibration 100 / eval 150, PatchFool attn_layer_idx=4, chunk=20)
   — 그래야 이 실험의 L=12 행이 system_comparison이 이미 발표한 수치
   (threshold=0.5116, FPR=13.3%, recall=72.7%)와 직접 대조되는 내장 sanity check가 된다.
2) collect_layer_attn(model16, images)를 clean/adv 각각 **한 번씩만** 호출한다 — 이 함수가
   forward 1회로 12개 block 전부의 attention을 이미 hook으로 수집하므로, 레이어마다 다시
   forward를 돌릴 필요가 없다(요청사항: "데이터를 새로 뽑을 필요 없이 저장된 attention
   텐서에서 레이어 인덱스만 바꿔 재계산").
3) L=1..12 각각에 대해 raw_at_layer(., L) + top4_mass(.)로 점수를 뽑고:
   - AUROC: eval 150개(held-out)만으로 계산 — calibration은 임계값 전용이라는 이 프로젝트의
     순환평가 방지 원칙(patch_switch_defense README "샘플링 감사")을 그대로 따른다.
   - threshold: calibration 100개로만 Youden's J 최적화(system_comparison과 동일 함수).
   - FPR/recall: 그 threshold로 eval 150개를 판정.
4) 얕은 레이어(L=1~3)의 "인접 토큰에 국소적으로 집중된다"는 정성적 주장을 뒷받침하기 위해,
   clean 이미지 2장의 CLS→patch attention을 14×14 그리드 히트맵으로 그린다(이미 1)에서
   확보한 clean attention을 재사용 — 별도 forward 없음).

프로젝트 경계에 대한 메모
------------------------
표본/공격 생성 코드(models.py/dataset.py/attacks/patch_fool.py)는 `ViT_robust/src/` —
`patch_size_tradeoff`/`patch_switch_defense`/`patch_attack_detector` 세 프로젝트가 각자 사본을 갖던 걸 2026-09-21에
공용으로 합친 것(`ViT_robust/README.md` 참고) — 에서 가져온다. `detector/topk_mass_v1.py`는
이 프로젝트 안에 있으므로 그냥 import(cross-project 아님).

calibration_eval_split/youden_threshold/wilson_ci는
patch_switch_defense/experiments/system_comparison/eval_utils.py에서 로직 변경 없이 그대로
복사했다(이 세 함수는 "공유 인프라"로 지정된 적이 없는 experiment-local 유틸이라 복사
원칙을 따름 — detector.py처럼 진짜 공유 인프라인 것과는 다르게 취급).

사용법:
  python layer_sweep_test.py --num_samples 250 --cal_frac 0.4 --seed 42 --chunk 20
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_ROOT = HERE
while not os.path.isdir(os.path.join(DETECT_ROOT, 'experiments')):
    DETECT_ROOT = os.path.dirname(DETECT_ROOT)
SHARED_SRC_ROOT = os.path.dirname(DETECT_ROOT)  # ViT_robust -- shared src/ (models/dataset/attacks)
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, DETECT_ROOT)

import numpy as np
import torch
import timm
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import get_device, load_vit_model, MODEL_NAMES
from src.dataset import get_dataloader
from src.attacks.patch_fool import patch_fool_attack
from detector.topk_mass_v1 import collect_layer_attn, raw_at_layer, top4_mass


# ── eval_utils.py(patch_switch_defense/experiments/system_comparison)에서 로직 변경 없이 복사 ──

def calibration_eval_split(num_total, frac=0.5):
    assert 0.0 < frac < 1.0
    n_cal = int(round(num_total * frac))
    n_eval = num_total - n_cal
    assert n_cal > 0 and n_eval > 0
    return slice(0, n_cal), slice(n_cal, n_cal + n_eval)


def youden_threshold(pos_scores, neg_scores):
    candidates = np.unique(np.concatenate([pos_scores, neg_scores]))
    best_j, best_t = -1.0, candidates[0]
    for t in candidates:
        tpr = (pos_scores > t).mean()
        fpr = (neg_scores > t).mean()
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def chunked_patch_fool(model16, images, labels, device, attn_layer_idx, chunk):
    adv_chunks = []
    for s in range(0, images.shape[0], chunk):
        e = min(s + chunk, images.shape[0])
        adv_chunk, _ = patch_fool_attack(
            model16, images[s:e], labels[s:e], device, patch_size_model=16,
            attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn',
            attn_layer_idx=attn_layer_idx)
        adv_chunks.append(adv_chunk)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return torch.cat(adv_chunks, dim=0)


def plot_shallow_heatmaps(lw_clean, images, out_path, layers=(1, 2, 3), n_imgs=2, ppl=14):
    """clean 이미지 n_imgs장 x (원본 + layers) 그리드. lw_clean은 이미 수집된 attention을
    재사용(추가 forward 없음)."""
    cfg = timm.get_pretrained_cfg(MODEL_NAMES[16])
    mean = torch.tensor(cfg.mean).view(3, 1, 1)
    std = torch.tensor(cfg.std).view(3, 1, 1)

    fig, axes = plt.subplots(n_imgs, len(layers) + 1,
                              figsize=(3 * (len(layers) + 1), 3 * n_imgs))
    for i in range(n_imgs):
        img = (images[i].cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'image {i} (original)')
        axes[i, 0].axis('off')
        for j, L in enumerate(layers):
            cls_to_patch = lw_clean[L - 1][i, 0, 1:].cpu().numpy()
            grid = cls_to_patch.reshape(ppl, ppl)
            ax = axes[i, j + 1]
            im = ax.imshow(grid, cmap='viridis')
            ax.set_title(f'L={L} CLS->patch attn')
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=250)
    parser.add_argument('--cal_frac', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cal, ev = calibration_eval_split(args.num_samples, frac=args.cal_frac)
    n_cal, n_eval = cal.stop - cal.start, ev.stop - ev.start
    print(f"calibration: {n_cal}개, evaluation: {n_eval}개")

    device = get_device()
    model16 = load_vit_model(16, device)
    model16.eval()

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples,
                                seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    print(f"\n[PatchFool on P16] 공격 생성 중 (총 {args.num_samples}개, "
          f"{args.chunk}개씩 나눠서, system_comparison과 동일 설정)...")
    adv = chunked_patch_fool(model16, images, labels, device, args.attn_layer_idx, args.chunk)

    print("\n[전 레이어 attention 수집] clean/adv 각 1회 forward...")
    lw_clean = collect_layer_attn(model16, images)
    lw_adv = collect_layer_attn(model16, adv)
    n_layers = len(lw_clean)
    assert n_layers == 12

    rows = []
    print(f"\n{'L':>3} | {'AUROC':>7} | {'thr':>8} {'J':>6} | "
          f"{'FPR':>16} | {'recall':>16}")
    print("-" * 70)
    for L in range(1, n_layers + 1):
        score_clean = top4_mass(raw_at_layer(lw_clean, L)).cpu().numpy()
        score_adv = top4_mass(raw_at_layer(lw_adv, L)).cpu().numpy()

        # AUROC: held-out eval만 사용 (calibration은 임계값 전용 — 순환평가 방지)
        y_true = np.concatenate([np.ones(n_eval), np.zeros(n_eval)])
        y_score = np.concatenate([score_adv[ev], score_clean[ev]])
        auroc = roc_auc_score(y_true, y_score)

        thr, j = youden_threshold(score_adv[cal], score_clean[cal])
        flagged_clean_ev = score_clean[ev] > thr
        flagged_adv_ev = score_adv[ev] > thr
        n_fp, n_tp = int(flagged_clean_ev.sum()), int(flagged_adv_ev.sum())
        fpr, recall = flagged_clean_ev.mean(), flagged_adv_ev.mean()
        fpr_ci = wilson_ci(n_fp, n_eval)
        recall_ci = wilson_ci(n_tp, n_eval)

        rows.append(dict(layer=L, auroc=auroc, threshold=float(thr), j=j,
                          fpr=fpr, fpr_ci=fpr_ci, n_fp=n_fp,
                          recall=recall, recall_ci=recall_ci, n_tp=n_tp))
        print(f"{L:>3} | {auroc:>7.3f} | {thr:>8.4f} {j:>6.3f} | "
              f"{fpr:.3f} ({n_fp}/{n_eval}) [{fpr_ci[0]:.3f},{fpr_ci[1]:.3f}] | "
              f"{recall:.3f} ({n_tp}/{n_eval}) [{recall_ci[0]:.3f},{recall_ci[1]:.3f}]")

    l12 = rows[11]
    print(f"\n[sanity] L=12: threshold={l12['threshold']:.4f}, FPR={l12['fpr']:.3f}, "
          f"recall={l12['recall']:.3f} — system_comparison 발표치(threshold=0.5116, "
          f"FPR=0.133, recall=0.727)와 비교할 것.")

    out_dir = HERE.replace('/experiments/', '/results/', 1)
    os.makedirs(out_dir, exist_ok=True)

    npz_path = os.path.join(out_dir, f'auroc_fpr_recall_by_layer_n{args.num_samples}.npz')
    np.savez(
        npz_path,
        layers=np.array([r['layer'] for r in rows]),
        auroc=np.array([r['auroc'] for r in rows]),
        threshold=np.array([r['threshold'] for r in rows]),
        youden_j=np.array([r['j'] for r in rows]),
        fpr=np.array([r['fpr'] for r in rows]),
        fpr_ci_lo=np.array([r['fpr_ci'][0] for r in rows]),
        fpr_ci_hi=np.array([r['fpr_ci'][1] for r in rows]),
        n_fp=np.array([r['n_fp'] for r in rows]),
        recall=np.array([r['recall'] for r in rows]),
        recall_ci_lo=np.array([r['recall_ci'][0] for r in rows]),
        recall_ci_hi=np.array([r['recall_ci'][1] for r in rows]),
        n_tp=np.array([r['n_tp'] for r in rows]),
        n_cal=n_cal, n_eval=n_eval,
    )

    # AUROC vs Layer
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r['layer'] for r in rows], [r['auroc'] for r in rows], marker='o')
    ax.axhline(0.5, color='gray', ls='--', lw=1, label='chance (0.5)')
    ax.set_xlabel('Layer L')
    ax.set_ylabel('AUROC (top-4 mass, PatchFool vs Clean, held-out eval)')
    ax.set_xticks(range(1, n_layers + 1))
    ax.set_ylim(0.3, 1.0)
    ax.set_title(f'Top-4 mass AUROC vs Layer (PatchFool, n={args.num_samples}, seed={args.seed})')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    auroc_path = os.path.join(out_dir, f'auroc_vs_layer_n{args.num_samples}.png')
    plt.savefig(auroc_path, dpi=150)
    plt.close(fig)

    # FPR/recall 표 (markdown)
    table_path = os.path.join(out_dir, f'auroc_fpr_recall_table_n{args.num_samples}.md')
    with open(table_path, 'w') as f:
        f.write(f"# Layer sweep — Top-4 mass (PatchFool, n={args.num_samples}, "
                f"seed={args.seed}, cal={n_cal}/eval={n_eval})\n\n")
        f.write("| L | AUROC | threshold | Youden J | FPR (n) | 95% CI | recall (n) | 95% CI |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['layer']} | {r['auroc']:.3f} | {r['threshold']:.4f} | {r['j']:.3f} | "
                f"{r['fpr']:.3f} ({r['n_fp']}/{n_eval}) | "
                f"[{r['fpr_ci'][0]:.3f}, {r['fpr_ci'][1]:.3f}] | "
                f"{r['recall']:.3f} ({r['n_tp']}/{n_eval}) | "
                f"[{r['recall_ci'][0]:.3f}, {r['recall_ci'][1]:.3f}] |\n")

    # 얕은 레이어(L=1~3) 정성적 히트맵 (clean 이미지 2장, 추가 forward 없음)
    heatmap_path = os.path.join(out_dir, f'shallow_layer_heatmaps_n{args.num_samples}.png')
    plot_shallow_heatmaps(lw_clean, images, heatmap_path, layers=(1, 2, 3), n_imgs=2)

    print(f"\nSaved: {npz_path}")
    print(f"Saved: {auroc_path}")
    print(f"Saved: {table_path}")
    print(f"Saved: {heatmap_path}")


if __name__ == '__main__':
    main()
