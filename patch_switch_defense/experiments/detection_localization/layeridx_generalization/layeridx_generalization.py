"""
probes/layeridx_generalization.py — [탐색적 검증, 롤백 가능] PatchFool의 토큰 선택
레이어(attn_layer_idx)를 바꿔가며, 고정된 탐지기(L=12 raw attention)의 localization recall이
attn_layer_idx=4(우리가 지금까지 써온 기본값)에 과적합된 결과였는지 확인.

배경
----
지금까지의 모든 결과(attention 기반/saliency 최저/saliency 구간 회피 시도, recall@1 85~97%)는
전부 PatchFool이 attn_layer_idx=4(기본값)로 토큰을 고른 공격에 대한 것이었다. 탐지기가 정말
견고한지, 아니면 attn_layer_idx=4라는 특정 설정에만 우연히 잘 맞았던 건지 구분하려면 다른
레이어를 타겟으로 한 공격에서도 똑같이 잘 잡히는지 봐야 한다.

방법
----
- attn_layer_idx in {1,2,4,6,8,10}마다: 원본 patch_fool_attack(patch_select='Attn',
  attack_mode='CE_loss')을 그 레이어로 실행 (원본 함수 그대로 사용, 수정 없음)
- 정답 토큰(그 레이어 기준 attention 최상위)은 clean 이미지 attention을 한 번만 수집해서
  각 레이어별로 _select_patch_attn을 재사용해 구한다 (원본 함수 재사용, 재계산 최소화)
- 고정된 탐지기(L=12 raw attention, attn_layer_idx와 무관하게 항상 마지막 레이어) 로
  recall/precision@K, 그리드 거리, 공격 성공률을 attn_layer_idx별로 비교

주의: src/attacks/patch_fool.py는 import만(수정 없음). 이 파일도 지우면 원상복구.

사용법:
  python probes/layeridx_generalization.py --patch_size 16 --num_samples 30 --seed 42
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.patch_fool import patch_fool_attack, _collect_attn, _select_patch_attn

LAYER_IDX_CANDIDATES = [1, 2, 4, 6, 8, 10]


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
            weights_list.append(attn.mean(dim=1).detach())
    return hook


def collect_layer_attn(model, images):
    weights = []
    hooks = [blk.attn.register_forward_hook(_attn_hook(weights)) for blk in model.blocks]
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()
    return weights


def raw_at_layer(layer_weights, L):
    attn = layer_weights[L - 1]
    cls_to_patch = attn[:, 0, 1:]
    return cls_to_patch / cls_to_patch.sum(dim=1, keepdim=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--detect_layer', type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    model = load_vit_model(args.patch_size, device)
    model.eval()
    ppl = 224 // args.patch_size

    # [샘플링 감사 통과] get_dataloader를 attn_layer_idx 루프 밖에서 1회만 호출 -> 모든
    # attn_layer_idx가 동일한 고정 이미지 집합으로 비교됨 (patch_switch_defense/README.md "샘플링 감사" 참고)
    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        pred_clean = model(images).argmax(dim=1)

    # clean attention은 한 번만 수집 (레이어별 정답 토큰은 여기서 재사용)
    attn_weights, hooks = _collect_attn(model)
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()

    rows = []
    for attn_layer_idx in LAYER_IDX_CANDIDATES:
        print(f"\n=== attn_layer_idx={attn_layer_idx} ===")
        true_idx = _select_patch_attn(attn_weights, attn_layer_idx, 1, device)[:, 0]

        print("  PatchFool 공격 생성 중...")
        adv_pf, _ = patch_fool_attack(
            model, images, labels, device, patch_size_model=args.patch_size,
            attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn',
            attn_layer_idx=attn_layer_idx)

        with torch.no_grad():
            pred_adv = model(adv_pf).argmax(dim=1)
        changed = (pred_adv != pred_clean)
        succ_rate = changed.float().mean().item()

        lw_adv = collect_layer_attn(model, adv_pf)
        v_raw = raw_at_layer(lw_adv, args.detect_layer)
        sorted_raw = v_raw.argsort(dim=1, descending=True)

        recalls = {}
        for K in (1, 4, 8):
            hit = (sorted_raw[:, :K] == true_idx.unsqueeze(1)).any(dim=1)
            recalls[K] = hit.float().mean().item()

        top1 = sorted_raw[:, 0]
        true_r, true_c = true_idx // ppl, true_idx % ppl
        top1_r, top1_c = top1 // ppl, top1 % ppl
        cheby = torch.maximum((true_r - top1_r).abs(), (true_c - top1_c).abs()).float()

        print(f"  공격 성공률={succ_rate:.3f}  recall@1={recalls[1]:.3f}  "
              f"recall@4={recalls[4]:.3f}  recall@8={recalls[8]:.3f}  "
              f"grid_dist_mean={cheby.mean().item():.2f}")

        rows.append(dict(attn_layer_idx=attn_layer_idx, succ_rate=succ_rate,
                         recall1=recalls[1], recall4=recalls[4], recall8=recalls[8],
                         cheby_mean=cheby.mean().item(), cheby_median=cheby.median().item()))

    print(f"\n{'attn_layer_idx':>15} {'succ_rate':>10} {'recall@1':>9} {'recall@4':>9} "
          f"{'recall@8':>9} {'grid_mean':>10} {'grid_med':>9}")
    print("-" * 75)
    for r in rows:
        print(f"{r['attn_layer_idx']:>15} {r['succ_rate']:>10.3f} {r['recall1']:>9.3f} "
              f"{r['recall4']:>9.3f} {r['recall8']:>9.3f} {r['cheby_mean']:>10.2f} {r['cheby_median']:>9.1f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xs = [r['attn_layer_idx'] for r in rows]
    axes[0].plot(xs, [r['recall1'] for r in rows], marker='o', label='recall@1')
    axes[0].plot(xs, [r['recall4'] for r in rows], marker='s', label='recall@4')
    axes[0].plot(xs, [r['succ_rate'] for r in rows], marker='^', ls='--', label='attack success rate')
    axes[0].set_xlabel('attn_layer_idx (PatchFool 토큰 선택 레이어)')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('Detector recall vs. attack target layer\n(detector fixed at L=12 raw)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(xs, [r['cheby_mean'] for r in rows], marker='o', color='#E94B3C')
    axes[1].set_xlabel('attn_layer_idx')
    axes[1].set_ylabel('mean grid distance (Chebyshev)')
    axes[1].set_title('Localization error vs. attack target layer')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'04_layeridx_generalization_P{args.patch_size}.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    raw_out_path = os.path.join(out_dir, f'04_layeridx_generalization_P{args.patch_size}.npz')
    np.savez(raw_out_path,
             layer_idx=np.array(xs),
             succ_rate=np.array([r['succ_rate'] for r in rows]),
             recall1=np.array([r['recall1'] for r in rows]),
             recall4=np.array([r['recall4'] for r in rows]),
             recall8=np.array([r['recall8'] for r in rows]),
             cheby_mean=np.array([r['cheby_mean'] for r in rows]))
    print(f"Saved raw: {raw_out_path}")


if __name__ == '__main__':
    main()
