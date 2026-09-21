"""
probes/evasion_test.py — [탐색적 검증, 롤백 가능] PatchFool을 "이미지 saliency 최저
영역"으로 강제했을 때, L=12 raw-attention 탐지기를 일관되게 회피하는지 확인. 회피가 확인되면
attention을 이미지 saliency로 정규화한 점수가 이 약점을 줄이는지도 같이 본다.

배경
----
failure_case.py에서 발견한 4개 실패 샘플이 전부 "공격이 저saliency 배경에 떨어지고,
자연히 saliency 높은 다른 영역이 attention 경쟁에서 이겼다"는 공통 패턴을 보였다. 이게 우연한
실패가 아니라 재현 가능한 회피 전략이라면, 지금 탐지기 설계(raw attention 절대값 top-1)는
공격자가 저saliency 위치를 의도적으로 노리는 순간 뚫린다.

방법
----
1) probes/patch_fool_lowsal.py로 "이미지 saliency 최저 토큰"을 강제 공격 (attention 전혀 참고 안 함)
2) 공격이 여전히 성공하는지(라벨 변경) 확인 — 위치를 저saliency로 제한해도 공격력이 유지되는지
3) 탐지기 A: raw attention top-1 (기존 방식) — 회피되는지 확인
4) 탐지기 B: raw attention을 (같은 이미지의) patch saliency로 정규화한 점수의 top-1
   score = attn / (saliency + eps) — "그 위치가 원래 밋밋한데도 attention이 유난히 높다"를 잡아내려는 시도
5) A/B 각각 recall@1/4/8, 그리드 거리 비교

주의: src/attacks/patch_fool.py는 import만(수정 없음). 이 파일들 지우면 원상복구.

사용법:
  python probes/evasion_test.py --patch_size 16 --num_samples 30 --seed 42
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(ROOT, 'experiments')):
    ROOT = os.path.dirname(ROOT)
PROBES = os.path.dirname(os.path.abspath(__file__))
SHARED_SRC_ROOT = os.path.dirname(ROOT)  # ViT_robust -- shared src/ (models/dataset/attacks) across sibling projects
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, PROBES)

import numpy as np
import torch

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from patch_fool_lowsal import patch_fool_attack_lowsal, compute_patch_saliency


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


def localization_stats(true_idx, pred_sorted_idx, ppl, name):
    B = true_idx.shape[0]
    print(f"\n--- {name} ---")
    for K in (1, 4, 8):
        hit = (pred_sorted_idx[:, :K] == true_idx.unsqueeze(1)).any(dim=1)
        recall = hit.float().mean().item()
        print(f"  K={K:2d}: recall@K={recall:.3f}  precision@K={recall / K:.3f}")
    top1 = pred_sorted_idx[:, 0]
    true_r, true_c = true_idx // ppl, true_idx % ppl
    top1_r, top1_c = top1 // ppl, top1 % ppl
    cheby = torch.maximum((true_r - top1_r).abs(), (true_c - top1_c).abs()).float()
    print(f"  grid dist: mean={cheby.mean().item():.2f}  median={cheby.median().item():.1f}  "
          f"exact={float((cheby == 0).float().mean()):.3f}  adjacent<=1={float((cheby <= 1).float().mean()):.3f}")
    return cheby


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

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        pred_clean = model(images).argmax(dim=1)

    print("[PatchFool-lowsal] 공격 생성 중 (saliency 최저 토큰 강제)...")
    adv_lowsal, true_idx = patch_fool_attack_lowsal(
        model, images, labels, device, patch_size_model=args.patch_size,
        num_patch=1, train_attack_iters=250)
    true_idx = true_idx[:, 0]

    with torch.no_grad():
        pred_adv = model(adv_lowsal).argmax(dim=1)
    changed = (pred_adv != pred_clean)
    print(f"\n공격 성공(라벨 변경) 비율: {changed.float().mean().item():.3f} "
          f"({changed.sum().item()}/{args.num_samples})")
    print(f"(참고: attention 기반 위치 선택 baseline은 30/30 전부 라벨 변경됨 — failure_case.py 로그)")

    # 탐지기 A: raw attention top-1 (기존 방식)
    lw_adv = collect_layer_attn(model, adv_lowsal)
    v_raw = raw_at_layer(lw_adv, args.detect_layer)
    sorted_raw = v_raw.argsort(dim=1, descending=True)
    localization_stats(true_idx, sorted_raw, ppl, "탐지기 A: raw attention top-K (기존)")

    # 탐지기 B: attention / saliency 정규화
    sal_adv = compute_patch_saliency(adv_lowsal, args.patch_size).to(device)
    eps = 1e-6
    v_norm = v_raw / (sal_adv + eps)
    sorted_norm = v_norm.argsort(dim=1, descending=True)
    localization_stats(true_idx, sorted_norm, ppl, "탐지기 B: attention / saliency 정규화 top-K")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'03_evasion_test_P{args.patch_size}.npz'),
             true_idx=true_idx.cpu().numpy(),
             changed=changed.cpu().numpy(),
             sorted_raw=sorted_raw.cpu().numpy(),
             sorted_norm=sorted_norm.cpu().numpy())
    print(f"\nSaved: {os.path.join(out_dir, f'03_evasion_test_P{args.patch_size}.npz')}")


if __name__ == '__main__':
    main()
