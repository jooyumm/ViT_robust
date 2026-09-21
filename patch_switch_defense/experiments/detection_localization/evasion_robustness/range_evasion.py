"""
probes/range_evasion.py — [탐색적 검증, 롤백 가능] 지난 실패 4개 샘플(1,15,23,28)의
실제 saliency 값 범위를 실측하고, 그 구간을 의도적으로 노린 공격이 탐지기를 뚫는지 확인.

배경
----
evasion_test.py에서 "saliency 최저" 강제는 오히려 탐지가 잘 됐다(recall@1=0.90) —
근데 실패했던 4개 샘플(attention 기반 선택, failure_case.py)은 saliency 최저가
아니라 "애매하게 낮은" 어딘가였을 가능성이 있다. 막연히 "중간"을 시도하지 않고, 그 4개
샘플에서 실제로 공격당한 토큰의 saliency 값을 그대로 측정해서 범위를 정하고, 그 구간을
정확히 재현하는 공격을 만든다.

방법
----
1) attn_layer_idx=4 attention 기반 선택으로 정답 토큰을 재현(failure_case.py와 동일
   설정, torch.manual_seed 고정으로 같은 4개 샘플 1,15,23,28이 나오는 것을 전제)하고,
   그 4개 샘플의 clean 이미지에서 실제 공격 토큰 위치의 saliency 값을 측정한다.
2) [lo, hi] = [min, max] (4개 값 그대로, 패딩 없음)
3) patch_fool_lowsal.patch_fool_attack_rangesal로 30개 샘플 전체에 대해 saliency가 그
   구간 안(또는 구간에 가장 가까운 곳)인 토큰을 강제 공격
4) 공격 성공률(라벨 변경) 확인
5) L=12 raw attention 탐지기로 recall/precision@K, 그리드 거리 확인

주의: src/attacks/patch_fool.py는 import만(수정 없음). 이 파일들 지우면 원상복구.

사용법:
  python probes/range_evasion.py --patch_size 16 --num_samples 30 --seed 42
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
from src.attacks.patch_fool import _collect_attn, _select_patch_attn
from patch_fool_lowsal import compute_patch_saliency, patch_fool_attack_rangesal

KNOWN_FAILURE_SAMPLES = [1, 15, 23, 28]  # failure_case.py에서 확인된 실패 인덱스


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attn_layer_idx', type=int, default=4)
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

    # 1) attention 기반 선택으로 정답 토큰 재현 (failure_case.py와 동일 설정)
    attn_weights, hooks = _collect_attn(model)
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()
    orig_true_idx = _select_patch_attn(attn_weights, args.attn_layer_idx, 1, device)[:, 0]

    sal_clean = compute_patch_saliency(images, args.patch_size).to(device)  # (B, num_patches)

    fail_sal = []
    print("=== 실패 샘플 4개의 실제 saliency 값 ===")
    for i in KNOWN_FAILURE_SAMPLES:
        s = sal_clean[i, orig_true_idx[i]].item()
        fail_sal.append(s)
        r, c = orig_true_idx[i].item() // ppl, orig_true_idx[i].item() % ppl
        print(f"  샘플 {i}: 위치=({r},{c})  saliency={s:.4f}")

    lo, hi = min(fail_sal), max(fail_sal)
    print(f"\n실측 saliency 범위: [{lo:.4f}, {hi:.4f}]")

    # 참고: 전체 이미지의 saliency 분포에서 이 구간이 대략 몇 %ile인지
    all_sal = sal_clean.flatten().cpu().numpy()
    pct_lo = (all_sal < lo).mean() * 100
    pct_hi = (all_sal < hi).mean() * 100
    print(f"(참고: 전체 토큰 saliency 분포에서 대략 {pct_lo:.1f}%ile ~ {pct_hi:.1f}%ile 구간)")

    # 2) 이 구간을 노린 공격 생성 (30개 전체)
    print("\n[PatchFool-rangesal] 공격 생성 중 (실측 구간 강제)...")
    with torch.no_grad():
        pred_clean = model(images).argmax(dim=1)

    adv_range, true_idx = patch_fool_attack_rangesal(
        model, images, labels, device, lo=lo, hi=hi, patch_size_model=args.patch_size,
        num_patch=1, train_attack_iters=250)
    true_idx = true_idx[:, 0]

    with torch.no_grad():
        pred_adv = model(adv_range).argmax(dim=1)
    changed = (pred_adv != pred_clean)
    print(f"\n공격 성공(라벨 변경) 비율: {changed.float().mean().item():.3f} "
          f"({changed.sum().item()}/{args.num_samples})")
    print("(참고: attention기반=30/30, saliency최저강제=26/30)")

    # 실제로 구간 안에 위치했는지도 확인
    sal_range_actual = compute_patch_saliency(images, args.patch_size).to(device)
    sal_at_true = sal_range_actual.gather(1, true_idx.unsqueeze(1)).squeeze(1)
    in_range = ((sal_at_true >= lo) & (sal_at_true <= hi))
    print(f"실제 선택된 토큰이 목표 구간 안이었던 비율: {in_range.float().mean().item():.3f}")

    # 3) 탐지기: L=12 raw attention top-K
    lw_adv = collect_layer_attn(model, adv_range)
    v_raw = raw_at_layer(lw_adv, args.detect_layer)
    sorted_raw = v_raw.argsort(dim=1, descending=True)
    localization_stats(true_idx, sorted_raw, ppl, f"탐지기: raw attention top-K (구간 saliency 공격)")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'03_range_evasion_P{args.patch_size}.npz'),
             fail_sal=np.array(fail_sal), lo=lo, hi=hi,
             true_idx=true_idx.cpu().numpy(), changed=changed.cpu().numpy(),
             sorted_raw=sorted_raw.cpu().numpy(), in_range=in_range.cpu().numpy())
    print(f"\nSaved: {os.path.join(out_dir, f'03_range_evasion_P{args.patch_size}.npz')}")


if __name__ == '__main__':
    main()
