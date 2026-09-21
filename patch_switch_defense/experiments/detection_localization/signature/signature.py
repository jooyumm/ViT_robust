"""
probes/signature.py — [탐색적 검증, 롤백 가능] LaVAN vs PatchFool의 attention rollout
"시그니처"가 실제로 구분되는지 확인하는 최소 검증 스크립트.

배경/가설
---------
평소 P16으로 추론하다가, ViTGuard류 탐지기로 "공격이 소수 토큰에 집중된 의심 영역
(PatchFool 시그니처)"을 찾아서 그 부분만 P8로 국소 재분할하는 적응형 방어를 설계하려 한다.
이게 성립하려면 먼저 확인해야 할 전제: **PatchFool 공격과 LaVAN 공격이 attention rollout
패턴에서 실제로 구분되는가?**

- PatchFool은 공격 설계 자체가 CLS attention을 소수 토큰에 쏠리게 만드는 걸 목표로 함
  (attention-aware loss, 특정 토큰 집중 공격) → rollout이 "peaky"(소수 토큰 집중)할 것으로 예상
- LaVAN은 attention을 전혀 고려하지 않고 순수 픽셀 공간(고정 위치 박스)에서 최적화됨
  → rollout이 clean과 비슷하거나 적어도 PatchFool과는 다른 방식일 것으로 예상

방법
----
ViTGuard(arXiv:2409.13828)의 attention rollout 정의를 그대로 따른다:
  A_hat_l = 0.5 * attn_l + 0.5 * I   (residual connection 보정)
  Rollout = A_hat_L @ ... @ A_hat_1  (레이어 순서대로 누적곱)
CLS -> patch rollout 벡터에서 두 가지 "집중도" 통계를 낸다:
  - top-4 질량 비율 (상위 4개 토큰이 전체 rollout에서 차지하는 비율, 높을수록 집중)
  - 정규화 entropy (낮을수록 집중)
clean / LaVAN / PatchFool 세 조건에서 이 통계들을 비교해서 분리가 되는지 본다.

주의
----
이 파일은 src/, experiments/ 의 기존 코드를 전혀 수정하지 않고 import만 한다.
결과가 기대와 다르거나 방향을 접게 되면 probes/ 디렉토리를 통째로 지우면 원상복구된다
(다른 어떤 파일도 건드리지 않았음).

사용법:
  python probes/signature.py --patch_size 16 --num_samples 30 --seed 42
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
from src.attacks.lavan import lavan_attack
from src.attacks.patch_fool import patch_fool_attack


def _rollout_hook(weights_list):
    """timm ViT Attention 모듈 forward hook — head 평균 낸 attention (B,N,N)을 기록."""
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


def compute_rollout(model, images):
    """CLS -> patch rollout 벡터 (B, num_patches), 합이 1이 되도록 정규화해서 반환."""
    weights = []
    hooks = [blk.attn.register_forward_hook(_rollout_hook(weights)) for blk in model.blocks]
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()

    B, N, _ = weights[0].shape
    eye = torch.eye(N, device=images.device).unsqueeze(0)
    rollout = eye.expand(B, N, N).clone()
    for attn in weights:  # 레이어 순서대로(hook 등록 순서 = model.blocks 순서) 누적곱
        a_hat = 0.5 * attn + 0.5 * eye
        rollout = a_hat @ rollout

    cls_to_patch = rollout[:, 0, 1:]  # CLS 행, CLS 자기자신(0열) 제외 -> 패치 N-1개
    cls_to_patch = cls_to_patch / cls_to_patch.sum(dim=1, keepdim=True)
    return cls_to_patch


def concentration_stats(rollout_vec):
    """top-4 질량 비율, 정규화 entropy 반환 (둘 다 numpy 배열, 배치 크기만큼)."""
    top4 = rollout_vec.topk(4, dim=1).values.sum(dim=1)
    entropy = -(rollout_vec * rollout_vec.clamp(min=1e-12).log()).sum(dim=1)
    norm_entropy = entropy / np.log(rollout_vec.shape[1])
    return top4.cpu().numpy(), norm_entropy.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    model = load_vit_model(args.patch_size, device)
    model.eval()

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    print("[Clean] rollout 계산 중...")
    top4_clean, ent_clean = concentration_stats(compute_rollout(model, images))

    print("[LaVAN] 공격 생성 및 rollout 계산 중...")
    adv_lavan = lavan_attack(model, images, labels, device, patch_ratio=0.02, steps=40, alpha=2/255)
    top4_lavan, ent_lavan = concentration_stats(compute_rollout(model, adv_lavan))

    print("[PatchFool] 공격 생성 및 rollout 계산 중...")
    adv_pf, _ = patch_fool_attack(
        model, images, labels, device, patch_size_model=args.patch_size,
        attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn')
    top4_pf, ent_pf = concentration_stats(compute_rollout(model, adv_pf))

    print("\n=== Top-4 token mass fraction (높을수록 소수 토큰에 집중) ===")
    for name, arr in [('Clean', top4_clean), ('LaVAN', top4_lavan), ('PatchFool', top4_pf)]:
        print(f"  {name:10s}: mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")

    print("\n=== Normalized rollout entropy (낮을수록 소수 토큰에 집중) ===")
    for name, arr in [('Clean', ent_clean), ('LaVAN', ent_lavan), ('PatchFool', ent_pf)]:
        print(f"  {name:10s}: mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    labels_x = ['Clean', 'LaVAN', 'PatchFool']
    axes[0].boxplot([top4_clean, top4_lavan, top4_pf], tick_labels=labels_x)
    axes[0].set_title('Top-4 token mass fraction\n(higher = more concentrated)')
    axes[0].set_ylabel('fraction')
    axes[1].boxplot([ent_clean, ent_lavan, ent_pf], tick_labels=labels_x)
    axes[1].set_title('Normalized rollout entropy\n(lower = more concentrated)')
    fig.suptitle(f'Attention-rollout signature: Clean vs LaVAN vs PatchFool (P={args.patch_size})')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'01_signature_P{args.patch_size}.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
