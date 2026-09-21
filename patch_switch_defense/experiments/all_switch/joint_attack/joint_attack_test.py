"""
probes/joint_attack_test.py — [탐색적 검증, 롤백 가능] joint attack(P16+P8 동시 공격)
stress test.

배경
----
지금까지의 모든 결과는 "P16만 아는 화이트박스 공격이 우연히 P8까지는 잘 안 뚫는다"는
것이었다(전이 저항성, recovery rate 97%). 근데 이건 공격자가 우리 방어 구조(P16 실패시
P8 전환)를 모른다는 전제에서만 성립한다. 방어를 아는 adaptive attacker라면 처음부터
P16과 P8을 동시에 속이도록 공격을 만들 수 있다 — patch_fool_joint.py의 joint_patch_fool_attack
이 그 공격이다. 이게 얼마나 잘 통하는지, 그리고 지금 탐지기(기존에 calibration으로 확정한
임계값 0.5567)로 여전히 잡히는지 확인한다.

방법
----
- calibration/evaluation에 안 쓰인 새 표본(다른 seed)으로 joint attack 생성
- P16/P8 각각 개별적으로/동시에 속는 비율 비교 (지금까지의 "P16만 공격" 결과와 대조)
- L=12 raw attention 탐지기(기존 calibration 임계값 재사용, 새로 계산 안 함)로 recall 확인
- localization(top-1이 실제 공격 토큰과 일치하는지)도 같이 확인

주의: src/attacks/patch_fool.py는 import만(수정 없음). 이 파일들 지우면 원상복구.

사용법:
  python probes/joint_attack_test.py --seed 123 --num_samples 50
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
from src.attacks.patch_fool import patch_fool_attack
from patch_fool_joint import joint_patch_fool_attack

# 기존 n=200 최종 검증(calibration 100개)에서 확정한 임계값 재사용 — 여기서 새로 안 정함
CALIBRATED_THRESHOLD = 0.5567


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


def top4_mass(v):
    return v.topk(4, dim=1).values.sum(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)  # calibration/evaluation(seed=42)과 다른 표본
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    parser.add_argument('--detect_layer', type=int, default=12)
    parser.add_argument('--chunk', type=int, default=20, help='joint attack은 모델 2개 동시 backward라 메모리 많이 씀')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()
    ppl = 224 // 16

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        pred16_clean = model16(images).argmax(dim=1)
        pred8_clean = model8(images).argmax(dim=1)
    both_orig_correct = (pred16_clean == labels) & (pred8_clean == labels)
    print(f"[참고] P16 clean acc: {(pred16_clean==labels).float().mean().item():.3f}  "
          f"P8 clean acc: {(pred8_clean==labels).float().mean().item():.3f}  "
          f"둘 다 원래 맞춘 이미지: {both_orig_correct.sum().item()}/{args.num_samples}")

    # ── (A) 참고용: P16만 공격(기존 방식)했을 때 P8까지 같이 속는 비율 ──────
    print(f"\n[참고 baseline] P16 단독 공격 생성 중 (전이 여부 확인용)...")
    adv_single_chunks = []
    for s in range(0, args.num_samples, args.chunk):
        e = min(s + args.chunk, args.num_samples)
        c, _ = patch_fool_attack(
            model16, images[s:e], labels[s:e], device, patch_size_model=16,
            attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn',
            attn_layer_idx=args.attn_layer_idx)
        adv_single_chunks.append(c)
        torch.cuda.empty_cache()
    adv_single = torch.cat(adv_single_chunks, dim=0)
    with torch.no_grad():
        pred16_single = model16(adv_single).argmax(dim=1)
        pred8_single = model8(adv_single).argmax(dim=1)
    fool16_single = both_orig_correct & (pred16_single != labels)
    fool_both_single = fool16_single & (pred8_single != labels)
    print(f"  P16만 공격: P16 속음 {fool16_single.sum().item()}/{both_orig_correct.sum().item()}, "
          f"그 중 P8까지 같이 속음(전이 성공) {fool_both_single.sum().item()}/{max(fool16_single.sum().item(),1)} "
          f"= {fool_both_single.sum().item()/max(fool16_single.sum().item(),1):.3f}")

    # ── (B) joint attack: P16+P8 동시 공격 ──────────────────────────
    print(f"\n[Joint attack] P16+P8 동시 공격 생성 중 (총 {args.num_samples}개, {args.chunk}개씩)...")
    adv_joint_chunks = []
    true_idx_chunks = []
    for s in range(0, args.num_samples, args.chunk):
        e = min(s + args.chunk, args.num_samples)
        print(f"  [{s}:{e}] 처리 중...")
        c, ti = joint_patch_fool_attack(
            model16, model8, images[s:e], labels[s:e], device,
            attn_layer_idx=args.attn_layer_idx, num_patch=1, train_attack_iters=250)
        adv_joint_chunks.append(c)
        true_idx_chunks.append(ti)
        torch.cuda.empty_cache()
    adv_joint = torch.cat(adv_joint_chunks, dim=0)
    true_idx = torch.cat(true_idx_chunks, dim=0)

    with torch.no_grad():
        pred16_joint = model16(adv_joint).argmax(dim=1)
        pred8_joint = model8(adv_joint).argmax(dim=1)

    fool16_joint = both_orig_correct & (pred16_joint != labels)
    fool8_joint = both_orig_correct & (pred8_joint != labels)
    fool_both_joint = both_orig_correct & (pred16_joint != labels) & (pred8_joint != labels)
    n_base = both_orig_correct.sum().item()
    print(f"\n=== Joint attack 결과 (원래 둘 다 맞춘 {n_base}개 기준) ===")
    print(f"  P16만 속음(P8은 여전히 맞음): {(fool16_joint & ~fool8_joint).sum().item()}/{n_base}")
    print(f"  P8만 속음(P16은 여전히 맞음): {(fool8_joint & ~fool16_joint).sum().item()}/{n_base}")
    print(f"  둘 다 속음(방어 완전 무력화): {fool_both_joint.sum().item()}/{n_base} "
          f"= {fool_both_joint.sum().item()/max(n_base,1):.3f}")
    print(f"  (참고: P16만 공격했을 때 전이로 둘 다 속은 비율은 위에서 "
          f"{fool_both_single.sum().item()}/{max(fool16_single.sum().item(),1)}였음)")

    # ── (C) 탐지기: 기존 calibration 임계값으로 joint attack도 잡히는지 ──
    lw_joint = collect_layer_attn(model16, adv_joint)
    v_joint = raw_at_layer(lw_joint, args.detect_layer)
    score_joint = top4_mass(v_joint).cpu().numpy()
    flagged_joint = score_joint > CALIBRATED_THRESHOLD
    sorted_joint = v_joint.argsort(dim=1, descending=True)

    print(f"\n=== 탐지기 (기존 calibration 임계값 {CALIBRATED_THRESHOLD} 그대로 적용) ===")
    print(f"  joint attack {args.num_samples}개 중 flag된 비율: {flagged_joint.mean():.3f} "
          f"({flagged_joint.sum()}/{args.num_samples})")

    flagged_joint_t = torch.tensor(flagged_joint, device=device)
    caught_both_fooled = fool_both_joint & flagged_joint_t
    print(f"  '둘 다 속은'(방어 무력화) 사례 중 그래도 탐지는 된 것: "
          f"{caught_both_fooled.sum().item()}/{max(fool_both_joint.sum().item(),1)}")
    print(f"  (탐지는 됐지만 P8로 보내도 이미 P8도 뚫린 상태라 소용없음에 유의)")

    hit1 = (sorted_joint[:, :1] == true_idx.unsqueeze(1)).any(dim=1).float().mean().item()
    hit4 = (sorted_joint[:, :4] == true_idx.unsqueeze(1)).any(dim=1).float().mean().item()
    top1 = sorted_joint[:, 0]
    true_r, true_c = true_idx // ppl, true_idx % ppl
    top1_r, top1_c = top1 // ppl, top1 % ppl
    cheby = torch.maximum((true_r - top1_r).abs(), (true_c - top1_c).abs()).float()
    print(f"\n=== Localization (joint attack에 대해서도 여전히 위치를 잘 찾는지) ===")
    print(f"  recall@1={hit1:.3f}  recall@4={hit4:.3f}  grid_dist_mean={cheby.mean().item():.2f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'07_joint_attack_test_n{args.num_samples}.npz'),
             threshold=CALIBRATED_THRESHOLD, score_joint=score_joint,
             pred16_clean=pred16_clean.cpu().numpy(), pred8_clean=pred8_clean.cpu().numpy(),
             pred16_single=pred16_single.cpu().numpy(), pred8_single=pred8_single.cpu().numpy(),
             pred16_joint=pred16_joint.cpu().numpy(), pred8_joint=pred8_joint.cpu().numpy(),
             labels=labels.cpu().numpy(), true_idx=true_idx.cpu().numpy(),
             sorted_joint=sorted_joint.cpu().numpy())
    print(f"\nSaved: {os.path.join(out_dir, f'07_joint_attack_test_n{args.num_samples}.npz')}")


if __name__ == '__main__':
    main()
