"""
probes/diversity_test.py - [탐색적 검증, 롤백 가능] Diversity diagnostic:
같은 patch size(P16), 다른 체크포인트(다른 학습)로 만들어진 두 모델 사이에 joint attack을 걸어서,
P8/P16 joint attack에서 관찰된 "완전 무력화 18.4%"가 patch size 차이 때문인지, 그냥
"다른 모델이라서(diversity)"인지 분리한다.

배경
----
patch_fool_joint.py로 P16(augreg2)+P8을 joint attack했을 때 7/38(18.4%)이 완전히 뚫렸다.
이번엔 patch size를 완전히 고정(둘 다 P16)하고, 학습만 다른 두 체크포인트로 같은 joint
attack을 걸어서 비교한다. 만약 이것도 비슷하게 높은 비율로 뚫리면, patch size 차이의
방어 효과는 생각보다 작고 그냥 "다른 모델 두 개 쓰기"의 효과일 수 있다. 반대로 이게 훨씬
낮게 나오면 patch size 차이 자체가 유의미한 방어 효과를 준다는 뜻이 된다.

방법
----
- Model A: vit_base_patch16_224.augreg2_in21k_ft_in1k (기존에 쓰던 P16, src.models 그대로)
- Model B: vit_base_patch16_224.augreg_in21k_ft_in1k (같은 계열 "augreg", 버전만 다름 -
  timm에 이미 pretrained로 존재해서 새로 학습 안 해도 됨. "다른 모델" 대리로 사용)
- joint_patch_fool_attack(A, B, ...)로 patch_fool_joint.py의 공격 로직을 그대로 재사용
  (그 함수가 이미 patch_size=16을 하드코딩하고 있어서 두 모델 다 P16이면 수정 없이 그대로 씀)

주의: src/models.py, src/attacks/patch_fool.py는 import만(수정 없음). 이 파일들 지우면 원상복구.

사용법:
  python probes/diversity_test.py --seed 456 --num_samples 50
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
import timm

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from patch_fool_joint import joint_patch_fool_attack

MODEL_B_NAME = 'vit_base_patch16_224.augreg_in21k_ft_in1k'


def load_model_b(device):
    model = timm.create_model(MODEL_B_NAME, pretrained=True)
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=456)  # 지금까지 안 쓴 새 seed
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Model A: vit_base_patch16_224.augreg2_in21k_ft_in1k (기존)")
    modelA = load_vit_model(16, device); modelA.eval()
    print(f"Model B: {MODEL_B_NAME} (다른 학습, 같은 patch size)")
    modelB = load_model_b(device)

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        predA_clean = modelA(images).argmax(dim=1)
        predB_clean = modelB(images).argmax(dim=1)
    both_orig_correct = (predA_clean == labels) & (predB_clean == labels)
    n_base = both_orig_correct.sum().item()
    print(f"\n[참고] A clean acc: {(predA_clean==labels).float().mean().item():.3f}  "
          f"B clean acc: {(predB_clean==labels).float().mean().item():.3f}  "
          f"둘 다 원래 맞춘 이미지: {n_base}/{args.num_samples}")

    print(f"\n[Joint attack: P16-A + P16-B] 공격 생성 중 (총 {args.num_samples}개, {args.chunk}개씩)...")
    adv_chunks = []
    for s in range(0, args.num_samples, args.chunk):
        e = min(s + args.chunk, args.num_samples)
        print(f"  [{s}:{e}] 처리 중...")
        c, _ = joint_patch_fool_attack(
            modelA, modelB, images[s:e], labels[s:e], device,
            attn_layer_idx=args.attn_layer_idx, num_patch=1, train_attack_iters=250)
        adv_chunks.append(c)
        torch.cuda.empty_cache()
    adv_joint = torch.cat(adv_chunks, dim=0)

    with torch.no_grad():
        predA_joint = modelA(adv_joint).argmax(dim=1)
        predB_joint = modelB(adv_joint).argmax(dim=1)

    foolA = both_orig_correct & (predA_joint != labels)
    foolB = both_orig_correct & (predB_joint != labels)
    fool_both = foolA & foolB

    print(f"\n=== Joint attack 결과: P16-A vs P16-B (다른 patch size 아님, 같은 P16 다른 학습) ===")
    print(f"  A만 속음(B는 여전히 맞음): {(foolA & ~foolB).sum().item()}/{n_base}")
    print(f"  B만 속음(A는 여전히 맞음): {(foolB & ~foolA).sum().item()}/{n_base}")
    print(f"  둘 다 속음(완전 무력화)  : {fool_both.sum().item()}/{n_base} = "
          f"{fool_both.sum().item()/max(n_base,1):.3f}")
    print(f"\n  (참고: P16-augreg2 vs P8 joint attack에서는 7/38 = 0.184 였음 -"
          f" 이 값과 비교해서 patch size 차이의 기여를 가늠)")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    # 파일명에 seed 포함 (감사 중 발견: seed 없이 n만 쓰던 이전 버전은 seed=456 원본 실행 결과를
    # seed=123 페어링 재실행이 덮어써버림 -- 로그(results/job_logs/nohup_diversity_2143709.txt)는
    # 남아있어서 값 자체는 안 잃었지만, 재발 방지로 seed를 파일명에 넣음)
    save_path = os.path.join(out_dir, f'08_diversity_test_n{args.num_samples}_seed{args.seed}.npz')
    np.savez(save_path,
             predA_clean=predA_clean.cpu().numpy(), predB_clean=predB_clean.cpu().numpy(),
             predA_joint=predA_joint.cpu().numpy(), predB_joint=predB_joint.cpu().numpy(),
             labels=labels.cpu().numpy())
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()
