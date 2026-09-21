"""
probes/localization.py — [탐색적 검증, 롤백 가능] top-K attention 토큰이 실제
PatchFool 공격 위치와 얼마나 겹치는지 (localization recall/precision).

배경
----
layer_sweep.py에서 "공격당한 이미지인지 아닌지" 탐지는 L=12 raw attention으로
잘 되는 걸 확인했다(AUROC~0.88). 근데 우리가 설계하는 방어("의심 영역만 P8로 국소 재분할")는
탐지뿐 아니라 *어디를* 재분할할지 위치까지 맞혀야 의미가 있다. top-K attention 토큰의 좌표가
실제 PatchFool이 공격한 토큰과 얼마나 겹치는지 확인한다.

방법
----
- PatchFool(num_patch=1, patch_select='Attn')이 실제로 어느 토큰을 공격했는지는 공격 시작 *전*
  clean 이미지의 attention에서 결정된다(공격 자체의 최적화 랜덤성과 무관, 결정적).
  src/attacks/patch_fool.py의 _collect_attn / _select_patch_attn을 그대로 import해서
  (원본 코드 수정 없이) 정답(ground truth) 토큰 인덱스를 그대로 재현한다.
- 탐지기 예측: L=12 raw attention(레이어 스윕에서 가장 좋았던 조합)으로 공격당한 이미지에서
  top-K 토큰 인덱스를 뽑는다 (K=1,4,8).
- recall@K = 정답 토큰이 top-K 예측 안에 있는 비율, precision@K = recall@K / K
  (정답이 이미지당 1개뿐이라 K개 중 최대 1개만 맞을 수 있음)
- 참고로 top-1 예측과 정답 사이의 패치 그리드 거리(Chebyshev, "몇 칸 떨어졌나")도 같이 본다
  — 완전히 못 맞혀도 "근처"면 국소 재분할 반경을 넉넉히 잡아서 커버할 여지가 있는지 보기 위함.

주의: src/attacks/patch_fool.py는 import만 하고 전혀 수정하지 않음. 이 파일도 지우면 그만.

사용법:
  python probes/localization.py --patch_size 16 --num_samples 30 --seed 42
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

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.patch_fool import patch_fool_attack, _collect_attn, _select_patch_attn


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
    parser.add_argument('--attn_layer_idx', type=int, default=4,
                        help='PatchFool이 토큰을 고를 때 쓰는 레이어 (patch_fool_attack 기본값)')
    parser.add_argument('--detect_layer', type=int, default=12,
                        help='탐지기가 쓰는 레이어 (레이어 스윕 결과 최적: 12, raw)')
    args = parser.parse_args()

    device = get_device()
    model = load_vit_model(args.patch_size, device)
    model.eval()
    ppl = 224 // args.patch_size  # patches per line

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    # 1) 정답(ground truth): PatchFool이 실제로 고르는 토큰. clean attention만으로 결정되므로
    #    (공격 최적화 이전 단계) 원본 함수를 그대로 재사용해서 재현한다.
    attn_weights, hooks = _collect_attn(model)
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()
    true_idx = _select_patch_attn(attn_weights, args.attn_layer_idx, 1, device)[:, 0]  # (B,)

    # 2) PatchFool 공격 생성 (원본 함수 그대로 호출)
    print("[PatchFool] 공격 생성 중...")
    adv_pf, _ = patch_fool_attack(
        model, images, labels, device, patch_size_model=args.patch_size,
        attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn',
        attn_layer_idx=args.attn_layer_idx)

    # 3) 탐지기 예측: L=detect_layer raw attention에서 top-K 토큰
    lw_pf = collect_layer_attn(model, adv_pf)
    v_pf = raw_at_layer(lw_pf, args.detect_layer)  # (B, num_patches)
    sorted_idx = v_pf.argsort(dim=1, descending=True)

    B = true_idx.shape[0]
    results_k = {}
    for K in (1, 4, 8):
        pred_topk = sorted_idx[:, :K]
        hit = (pred_topk == true_idx.unsqueeze(1)).any(dim=1)
        recall = hit.float().mean().item()
        precision = recall / K
        results_k[K] = (recall, precision, hit.cpu().numpy())

    top1 = sorted_idx[:, 0]
    true_r, true_c = true_idx // ppl, true_idx % ppl
    top1_r, top1_c = top1 // ppl, top1 % ppl
    cheby = torch.maximum((true_r - top1_r).abs(), (true_c - top1_c).abs()).float()

    print(f"\n표본 수: {B}, patch grid: {ppl}x{ppl} (P={args.patch_size}), "
          f"탐지기 레이어: {args.detect_layer} raw, 공격 타겟 레이어: {args.attn_layer_idx}")
    print("\n=== Localization: recall/precision@K (정답 토큰 1개가 top-K 예측 안에 있는가) ===")
    for K, (recall, precision, _) in results_k.items():
        print(f"  K={K:2d}: recall@K={recall:.3f}  precision@K={precision:.3f}")

    print("\n=== Top-1 예측 vs 정답의 패치 그리드 거리 (Chebyshev, 칸 단위) ===")
    print(f"  mean={cheby.mean().item():.2f}  median={cheby.median().item():.1f}  "
          f"exact-match(0칸)={float((cheby == 0).float().mean()):.3f}  "
          f"인접(<=1칸)={float((cheby <= 1).float().mean()):.3f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'02_localization_P{args.patch_size}.npz'),
             true_idx=true_idx.cpu().numpy(), top1=top1.cpu().numpy(),
             cheby=cheby.cpu().numpy(),
             **{f'hit_at_{K}': v[2] for K, v in results_k.items()})
    print(f"\nSaved: {os.path.join(out_dir, f'02_localization_P{args.patch_size}.npz')}")


if __name__ == '__main__':
    main()
