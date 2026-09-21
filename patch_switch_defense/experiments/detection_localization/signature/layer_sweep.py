"""
probes/layer_sweep.py — [탐색적 검증, 롤백 가능] 레이어별 · 방식별(rollout vs raw attention)
집중도 신호와 AUROC 비교.

배경
----
signature.py(12레이어 전체 rollout)의 결과가 애매했다 — PatchFool의 top-4 질량비율이
평균적으로는 높았지만(0.060 vs Clean/LaVAN 0.042) 개별 샘플 단위로는 범위가 겹쳤다. 원인 후보:
PatchFool 공격 자체가 attn_layer_idx=4(4번째 레이어)를 특정해서 그 레이어의 attention만 최적화
하는데, 12개 레이어를 전부 곱해버려서 4번째 레이어의 강한 신호가 나머지 레이어와 섞여 희석됐을
가능성.

근데 레이어 4에서'만' 신호가 잘 잡힌다면, 그건 "attention 집중이라는 일반적 방어 신호"가 아니라
"지금 우리가 쓰는 PatchFool 구현이 layer 4를 타겟하도록 설정돼 있어서 생긴 우연/과적합"일 수 있다.
그래서 레이어를 좁히기 전에, 여러 레이어(1,2,4,6,8,10,12)에서 신호가 폭넓게 존재하는지부터 본다.

방법
----
- clean / LaVAN / PatchFool 세 조건 각각, 모델 forward 1회로 12개 레이어의 attention을 전부 수집
  (attack 생성은 재사용 불가 — 재실행하되 seed 고정이라 signature.py와 같은 샘플/공격이 나옴)
- 레이어 L마다 두 가지 집중도 벡터 계산:
    (a) rollout: 1번째 레이어부터 L번째까지 누적곱 (residual-aware, ViTGuard 정의)
    (b) raw: L번째 레이어 attention만 단독 사용 (rollout 없음)
- 각 (L, 방식)에서 top-4 질량비율 / entropy 계산
- "PatchFool vs Clean", "PatchFool vs LaVAN" 이진 분류 AUROC를 sklearn으로 계산
  (range overlap을 눈으로 보는 대신 정직한 분리도 지표)

주의: 원본 signature.py와 src/, experiments/ 코드는 전혀 건드리지 않음.
     이 파일도 필요없어지면 지우기만 하면 됨.

사용법:
  python probes/layer_sweep.py --patch_size 16 --num_samples 30 --seed 42
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(ROOT, 'experiments')):
    ROOT = os.path.dirname(ROOT)
SHARED_SRC_ROOT = os.path.dirname(ROOT)  # ViT_robust -- shared src/ (models/dataset/attacks) across sibling projects
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.lavan import lavan_attack
from src.attacks.patch_fool import patch_fool_attack

CANDIDATE_LAYERS = [1, 2, 4, 6, 8, 10, 12]


def _attn_hook(weights_list):
    def hook(module, input, output):
        with torch.no_grad():
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(
                B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            weights_list.append(attn.mean(dim=1).detach())  # (B, N, N), head 평균
    return hook


def collect_layer_attn(model, images):
    """레이어 순서대로 (B,N,N) attention 리스트 반환 (head 평균, 총 len(model.blocks)개)."""
    weights = []
    hooks = [blk.attn.register_forward_hook(_attn_hook(weights)) for blk in model.blocks]
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()
    return weights


def rollout_upto(layer_weights, L):
    """1번째 레이어부터 L번째까지 residual-aware rollout 누적곱 -> CLS->patch 벡터 (정규화)."""
    attn0 = layer_weights[0]
    B, N, _ = attn0.shape
    eye = torch.eye(N, device=attn0.device).unsqueeze(0)
    rollout = eye.expand(B, N, N).clone()
    for attn in layer_weights[:L]:
        a_hat = 0.5 * attn + 0.5 * eye
        rollout = a_hat @ rollout
    cls_to_patch = rollout[:, 0, 1:]
    return cls_to_patch / cls_to_patch.sum(dim=1, keepdim=True)


def raw_at_layer(layer_weights, L):
    """L번째 레이어 attention 단독 (rollout 없이) -> CLS->patch 벡터 (정규화)."""
    attn = layer_weights[L - 1]
    cls_to_patch = attn[:, 0, 1:]
    return cls_to_patch / cls_to_patch.sum(dim=1, keepdim=True)


def concentration_stats(rollout_vec):
    top4 = rollout_vec.topk(4, dim=1).values.sum(dim=1)
    entropy = -(rollout_vec * rollout_vec.clamp(min=1e-12).log()).sum(dim=1)
    norm_entropy = entropy / np.log(rollout_vec.shape[1])
    return top4.cpu().numpy(), norm_entropy.cpu().numpy()


def safe_auroc(pos_scores, neg_scores):
    """pos=PatchFool(1), neg=비교대상(0). 점수가 클수록 '더 PatchFool 같다'는 방향으로 넣을 것."""
    y_true  = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])
    return roc_auc_score(y_true, y_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    model = load_vit_model(args.patch_size, device)
    model.eval()

    # [샘플링 감사 통과] get_dataloader를 레이어 스윕 루프 밖에서 1회만 호출 -> 모든 레이어가
    # 동일한 고정 이미지 집합으로 비교됨 (patch_switch_defense/README.md "샘플링 감사" 참고)
    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    print("[Clean] attention 수집 중...")
    lw_clean = collect_layer_attn(model, images)

    print("[LaVAN] 공격 생성 및 attention 수집 중...")
    adv_lavan = lavan_attack(model, images, labels, device, patch_ratio=0.02, steps=40, alpha=2/255)
    lw_lavan = collect_layer_attn(model, adv_lavan)

    print("[PatchFool] 공격 생성 및 attention 수집 중...")
    adv_pf, _ = patch_fool_attack(
        model, images, labels, device, patch_size_model=args.patch_size,
        attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn')
    lw_pf = collect_layer_attn(model, adv_pf)

    n_layers = len(lw_clean)
    layers = [L for L in CANDIDATE_LAYERS if L <= n_layers]

    results = {}  # (L, method, metric) -> dict(clean, lavan, pf arrays)
    rows = []
    for L in layers:
        for method_name, fn in [('rollout', rollout_upto), ('raw', raw_at_layer)]:
            v_clean = fn(lw_clean, L)
            v_lavan = fn(lw_lavan, L)
            v_pf    = fn(lw_pf, L)

            top4_c, ent_c = concentration_stats(v_clean)
            top4_l, ent_l = concentration_stats(v_lavan)
            top4_p, ent_p = concentration_stats(v_pf)

            results[(L, method_name)] = dict(
                top4=(top4_c, top4_l, top4_p), ent=(ent_c, ent_l, ent_p))

            # top4: 높을수록 PatchFool 같음 -> 그대로 score
            auroc_top4_vs_clean = safe_auroc(top4_p, top4_c)
            auroc_top4_vs_lavan = safe_auroc(top4_p, top4_l)
            # entropy: 낮을수록 PatchFool 같음 -> -entropy를 score로
            auroc_ent_vs_clean = safe_auroc(-ent_p, -ent_c)
            auroc_ent_vs_lavan = safe_auroc(-ent_p, -ent_l)

            rows.append(dict(
                layer=L, method=method_name,
                top4_pf=top4_p.mean(), top4_clean=top4_c.mean(), top4_lavan=top4_l.mean(),
                auroc_top4_vs_clean=auroc_top4_vs_clean, auroc_top4_vs_lavan=auroc_top4_vs_lavan,
                auroc_ent_vs_clean=auroc_ent_vs_clean, auroc_ent_vs_lavan=auroc_ent_vs_lavan,
            ))

    print(f"\n{'layer':>5} {'method':>8} | {'top4 PF':>8} {'top4 Cln':>9} {'top4 LaV':>9} | "
          f"{'AUROC(top4)':>18} | {'AUROC(entropy)':>18}")
    print(f"{'':>5} {'':>8} | {'':>8} {'':>9} {'':>9} | {'vs-Cln':>8} {'vs-LaV':>9} | "
          f"{'vs-Cln':>8} {'vs-LaV':>9}")
    print("-" * 90)
    for r in rows:
        print(f"{r['layer']:>5} {r['method']:>8} | "
              f"{r['top4_pf']:>8.3f} {r['top4_clean']:>9.3f} {r['top4_lavan']:>9.3f} | "
              f"{r['auroc_top4_vs_clean']:>8.3f} {r['auroc_top4_vs_lavan']:>9.3f} | "
              f"{r['auroc_ent_vs_clean']:>8.3f} {r['auroc_ent_vs_lavan']:>9.3f}")

    # 저장 (재실행 없이 재분석 가능하도록 raw 데이터도 npz로)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'01_layer_sweep_P{args.patch_size}_raw.npz'),
             **{f'{L}_{m}_{cond}_{stat}': arr
               for L in layers for m in ('rollout', 'raw')
               for stat in ('top4', 'ent')
               for cond, arr in zip(('clean', 'lavan', 'pf'), results[(L, m)][stat])})

    # AUROC vs layer 그래프 (rollout vs raw, PF-vs-Clean / PF-vs-LaVAN, top4 기준)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for method_name, marker in [('rollout', 'o'), ('raw', 's')]:
        sub = [r for r in rows if r['method'] == method_name]
        axes[0].plot([r['layer'] for r in sub], [r['auroc_top4_vs_clean'] for r in sub],
                    marker=marker, label=method_name)
        axes[1].plot([r['layer'] for r in sub], [r['auroc_top4_vs_lavan'] for r in sub],
                    marker=marker, label=method_name)
    for ax, title in zip(axes, ['PatchFool vs Clean', 'PatchFool vs LaVAN']):
        ax.axhline(0.5, color='gray', ls='--', lw=1, label='chance (0.5)')
        ax.set_xlabel('layer L')
        ax.set_ylabel('AUROC (top-4 mass fraction)')
        ax.set_title(title)
        ax.set_ylim(0.3, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'01_layer_sweep_P{args.patch_size}.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    print(f"Saved raw: {os.path.join(out_dir, f'01_layer_sweep_P{args.patch_size}_raw.npz')}")


if __name__ == '__main__':
    main()
