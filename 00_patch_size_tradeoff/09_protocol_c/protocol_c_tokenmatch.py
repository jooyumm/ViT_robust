"""
09_protocol_c/protocol_c_tokenmatch.py - [탐색적 검증, 롤백 가능] Protocol C: pixel area 대신
attacked token 개수(num_patch)를 P8/P16/P32에서 동일하게 맞춘 PatchFool 실험.

배경
----
지금까지의 area-matched ablation(실험 4/5, 아주 예전 연구)은 "물리적 픽셀 면적"을 P16
기준으로 맞췄다(P8만 num_patch=4로 늘려서 P16의 1토큰 면적과 동일하게). 이건 그 반대 방향 -
"공격당하는 토큰 개수"를 P8/P16/P32 전부 동일(num_patch=4)하게 고정하고, 그 결과 생기는
물리적 면적 차이(P8:4*64=256px^2, P16:4*256=1024px^2, P32:4*1024=4096px^2)는 그대로 둔다.
baseline(num_patch=1)에서 본 "P8이 압도적으로 강건" 패턴이 토큰 개수를 4로 맞춰도 유지되는지
확인한다.

방법
----
- P in {8, 16, 32} 각각에 patch_fool_attack(patch_select='Attn', attack_mode='CE_loss',
  num_patch=4, attn_layer_idx=4)를 표준 그대로 적용 (원본 함수 그대로, 수정 없음)
- CA/RA/ASR은 metrics의 기존 함수 그대로 사용 (프로젝트 표준과 일치)

주의: src/attacks/patch_fool.py, metrics.py는 import만(수정 없음). 이 파일 지우면 원상복구.

사용법:
  python 09_protocol_c/protocol_c_tokenmatch.py --seed 789 --num_samples 50 --num_patch 4
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_SRC_ROOT = os.path.dirname(ROOT)  # ViT_robust -- shared src/ (models/dataset/attacks)
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, ROOT)

import numpy as np
import torch

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.patch_fool import patch_fool_attack
from metrics import compute_confusion_counts, compute_metrics_from_counts  # 00_patch_size_tradeoff-only


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=789)  # 새 seed
    parser.add_argument('--num_patch', type=int, default=4)
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=25)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    # [샘플링 감사 통과] get_dataloader를 P 루프 밖에서 1회만 호출 -> P=8/16/32 전부
    # 동일한 고정 이미지 집합(같은 순서)으로 평가됨 (../README.md "샘플링 감사" 참고)
    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    rows = []
    for P in (8, 16, 32):
        print(f"\n=== P={P}, num_patch={args.num_patch} (토큰 개수 고정, 면적={args.num_patch*P*P}px^2) ===")
        model = load_vit_model(P, device); model.eval()

        with torch.no_grad():
            preds_clean = model(images).argmax(dim=1).cpu().tolist()

        adv_chunks = []
        for s in range(0, args.num_samples, args.chunk):
            e = min(s + args.chunk, args.num_samples)
            c, _ = patch_fool_attack(
                model, images[s:e], labels[s:e], device, patch_size_model=P,
                attack_mode='CE_loss', train_attack_iters=250,
                num_patch=args.num_patch, patch_select='Attn', attn_layer_idx=args.attn_layer_idx)
            adv_chunks.append(c)
            torch.cuda.empty_cache()
        adv = torch.cat(adv_chunks, dim=0)

        with torch.no_grad():
            preds_adv = model(adv).argmax(dim=1).cpu().tolist()

        labels_list = labels.cpu().tolist()
        counts = compute_confusion_counts(preds_clean, preds_adv, labels_list)
        m = compute_metrics_from_counts(**counts)
        print(f"  CA={m['CA']:.2f}%  RA={m['RA']:.2f}%  ASR={m['ASR']:.2f}%  "
              f"[A={m['A']} B={m['B']} C={m['C']} D={m['D']}]")
        rows.append(dict(P=P, **m))

        del model
        torch.cuda.empty_cache()

    print(f"\n{'P':>4} {'면적(px^2)':>11} {'CA':>8} {'RA':>8} {'ASR':>8}")
    print("-" * 45)
    for r in rows:
        print(f"{r['P']:>4} {args.num_patch*r['P']*r['P']:>11} {r['CA']:>7.2f}% {r['RA']:>7.2f}% {r['ASR']:>7.2f}%")

    print(f"\n(참고: baseline num_patch=1 결과 - P8 RA~67.6%, P16 RA~9.1%, P32 RA~0.3%)")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'09_protocol_c_tokenmatch_np{args.num_patch}.npz'),
             P=np.array([r['P'] for r in rows]),
             CA=np.array([r['CA'] for r in rows]),
             RA=np.array([r['RA'] for r in rows]),
             ASR=np.array([r['ASR'] for r in rows]))
    print(f"\nSaved: {os.path.join(out_dir, f'09_protocol_c_tokenmatch_np{args.num_patch}.npz')}")


if __name__ == '__main__':
    main()
