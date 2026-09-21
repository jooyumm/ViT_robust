"""
probes/adaptive_evasion_full_test.py — [탐색적 검증, 롤백 가능] 완전판 adaptive
attack(P16+P8 동시 공격 + 탐지 회피 제약) stress test. 지금 방어의 진짜 worst-case.

배경
----
§7(joint_attack_test.py)의 adaptive attacker는 방어 구조(P16+P8)는 알지만 탐지기
존재는 몰랐다 -> 무력화율 18.4%, 그 중에서도 20%는 탐지기가 여전히 flag했음. 이 스크립트는
patch_fool_joint_evasive.joint_patch_fool_attack_evasive로 탐지기 존재까지 아는 공격자를
흉내낸다: P16+P8을 속이면서 동시에 L=12 top-4 mass를 clean 범위 안으로 강제하는 제약 최적화
(STRAP-ViT류 설계).

target_bound 두 가지를 다 시험한다(둘 다 probes/results/final_validation_n200.npz의 실제
calibration 결과에서 가져온 값, 하드코딩 아님):
  - clean_max (0.6728): 사용자가 요청한 원래 STRAP-ViT식 설계 — 관측된 clean 최댓값 아래
  - calibrated_threshold (0.5567): 실제 배포된 탐지기가 쓰는 임계값 (Youden's J, calibration
    100개로 도출) — clean_max보다 낮아서 더 엄격한 제약. 공격이 clean_max 밑으로는 내려가도
    실제 배포된 탐지기(threshold 기준)에는 여전히 flag될 수 있으므로, "진짜 회피 성공"은
    이 threshold 기준으로 판정해야 정확함.

이미지 집합은 §7(joint_attack_test.py)과 동일(seed=123, n=50, eval_utils의
load_paired_batch로 로드)이라 §7의 18.4%/flagged 20% 결과와 직접(같은 표본) 비교 가능.

주의: src/attacks/patch_fool.py, patch_fool_joint_evasive.py는 import만(수정 없음).
이 파일 지우면 원상복구.

사용법:
  python probes/adaptive_evasion_full_test.py --seed 123 --num_samples 50 --chunk 20
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
from patch_fool_joint_evasive import joint_patch_fool_attack_evasive
from eval_utils import load_paired_batch, wilson_ci, both_correct_mask

# probes/results/final_validation_n200.npz에서 그대로 가져온 값 (calibration 100개로 도출,
# patch_switch_defense/README.md §6). 여기서 새로 계산 안 함 -- 재현하려면:
#   d = np.load('probes/results/final_validation_n200.npz'); d['threshold'], d['score_clean'].max()
CALIBRATED_THRESHOLD = 0.5567247867584229
CLEAN_MAX = 0.6727843284606934

# §7(joint_attack_test.py, 같은 seed=123/n=50)의 참고값
PREV_FOOL_BOTH = 7 / 38       # 0.184
PREV_FLAGGED_RATE = 0.20      # README §7: "joint attack의 20%만 flag"


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


def run_one_condition(name, target_bound, model16, model8, images, labels, both_orig_correct,
                       device, args):
    print(f"\n{'='*70}\n[{name}] target_bound={target_bound:.4f}\n{'='*70}")
    adv_chunks, true_idx_chunks, score_hist_last = [], [], []
    for s in range(0, args.num_samples, args.chunk):
        e = min(s + args.chunk, args.num_samples)
        print(f"  [{s}:{e}] 처리 중...")
        adv, ti, score_hist, lam_hist = joint_patch_fool_attack_evasive(
            model16, model8, images[s:e], labels[s:e], device,
            attn_layer_idx=args.attn_layer_idx, num_patch=1,
            train_attack_iters=args.iters, target_bound=target_bound,
            lambda0=args.lambda0, lambda_growth=args.lambda_growth,
            lambda_max=args.lambda_max, lambda_step_every=args.lambda_step_every,
            detect_layer=args.detect_layer)
        adv_chunks.append(adv)
        true_idx_chunks.append(ti)
        score_hist_last.append(score_hist[-1])   # 이 청크의 마지막 iter 학습 중 score (참고용)
        print(f"    최종 lambda={lam_hist[-1]:.1f}, 최종 iter 학습중 score 평균={score_hist[-1].mean():.4f}")
        torch.cuda.empty_cache()
    adv = torch.cat(adv_chunks, dim=0)
    true_idx = torch.cat(true_idx_chunks, dim=0)

    with torch.no_grad():
        pred16_adv = model16(adv).argmax(dim=1)
        pred8_adv = model8(adv).argmax(dim=1)
    fool16 = both_orig_correct & (pred16_adv != labels)
    fool8 = both_orig_correct & (pred8_adv != labels)
    fool_both = both_orig_correct & (pred16_adv != labels) & (pred8_adv != labels)
    n_base = both_orig_correct.sum().item()
    rate_both = fool_both.sum().item() / max(n_base, 1)

    # 학습에 쓴 미분가능 score가 아니라, 배포된 탐지기와 100% 동일한 방식(post-hoc, no_grad)으로
    # 독립적으로 다시 측정 -- "정말 회피했는지"를 자체 채점하지 않고 별도로 검증
    lw_adv = collect_layer_attn(model16, adv)
    v_adv = raw_at_layer(lw_adv, args.detect_layer)
    score_final = top4_mass(v_adv).cpu().numpy()
    flagged_by_threshold = score_final > CALIBRATED_THRESHOLD
    flagged_by_cleanmax = score_final > CLEAN_MAX

    print(f"\n  --- 결과 ({n_base}개 기준, 둘 다 원래 맞춘 표본) ---")
    print(f"  P16만 속음: {(fool16 & ~fool8).sum().item()}/{n_base}")
    print(f"  P8만 속음: {(fool8 & ~fool16).sum().item()}/{n_base}")
    print(f"  둘 다 속음(방어 완전 무력화): {fool_both.sum().item()}/{n_base} = {rate_both:.3f}")
    print(f"  (참고: §7 탐지 회피 없는 joint attack은 {PREV_FOOL_BOTH:.3f}였음)")
    print(f"\n  실제 배포 임계값({CALIBRATED_THRESHOLD:.4f}) 기준 flag 비율: "
          f"{flagged_by_threshold.mean():.3f} ({flagged_by_threshold.sum()}/{args.num_samples})")
    print(f"  clean 최댓값({CLEAN_MAX:.4f}) 기준 flag 비율: "
          f"{flagged_by_cleanmax.mean():.3f} ({flagged_by_cleanmax.sum()}/{args.num_samples})")
    print(f"  (참고: §7 탐지 회피 없는 joint attack의 flag 비율은 {PREV_FLAGGED_RATE:.2f}였음)")

    fool_both_t = fool_both
    caught_both_fooled = fool_both_t & torch.tensor(flagged_by_threshold, device=device)
    n_worst = caught_both_fooled.sum().item()
    n_complete_defeat = fool_both_t.sum().item() - n_worst
    print(f"\n  *** 진짜 최악(완전 무력화 + 탐지도 회피) ***: "
          f"{n_complete_defeat}/{n_base} = {n_complete_defeat/max(n_base,1):.3f}")
    print(f"      (둘 다 속았지만 임계값 기준 여전히 flag되는 것: {n_worst}/{max(fool_both_t.sum().item(),1)})")

    return dict(name=name, target_bound=target_bound, adv=adv, true_idx=true_idx,
                pred16_adv=pred16_adv.cpu().numpy(), pred8_adv=pred8_adv.cpu().numpy(),
                score_final=score_final, rate_both=rate_both,
                flagged_by_threshold=flagged_by_threshold, flagged_by_cleanmax=flagged_by_cleanmax,
                n_complete_defeat=n_complete_defeat, n_base=n_base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123)  # §7과 동일 -> 직접 비교
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    parser.add_argument('--detect_layer', type=int, default=12)
    parser.add_argument('--iters', type=int, default=250)
    parser.add_argument('--chunk', type=int, default=20)
    parser.add_argument('--lambda0', type=float, default=5.0)
    parser.add_argument('--lambda_growth', type=float, default=1.5)
    parser.add_argument('--lambda_max', type=float, default=500.0)
    parser.add_argument('--lambda_step_every', type=int, default=25)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()

    images, labels, _ = load_paired_batch(args.seed, args.num_samples)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        pred16_clean = model16(images).argmax(dim=1)
        pred8_clean = model8(images).argmax(dim=1)
    both_orig_correct = both_correct_mask(pred16_clean, pred8_clean, labels)
    print(f"[참고] P16 clean acc: {(pred16_clean==labels).float().mean().item():.3f}  "
          f"P8 clean acc: {(pred8_clean==labels).float().mean().item():.3f}  "
          f"둘 다 원래 맞춘 이미지: {both_orig_correct.sum().item()}/{args.num_samples} "
          f"(§7과 동일 seed=123이라 같은 38개여야 함)")

    results = {}
    for name, bound in [('clean_max', CLEAN_MAX), ('calibrated_threshold', CALIBRATED_THRESHOLD)]:
        results[name] = run_one_condition(name, bound, model16, model8, images, labels,
                                           both_orig_correct, device, args)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'14_adaptive_evasion_full_n{args.num_samples}_seed{args.seed}.npz')
    save_dict = {'both_orig_correct': both_orig_correct.cpu().numpy(),
                 'labels': labels.cpu().numpy()}
    for name, r in results.items():
        save_dict[f'{name}_rate_both'] = r['rate_both']
        save_dict[f'{name}_score_final'] = r['score_final']
        save_dict[f'{name}_flagged_by_threshold'] = r['flagged_by_threshold']
        save_dict[f'{name}_flagged_by_cleanmax'] = r['flagged_by_cleanmax']
        save_dict[f'{name}_pred16_adv'] = r['pred16_adv']
        save_dict[f'{name}_pred8_adv'] = r['pred8_adv']
        save_dict[f'{name}_n_complete_defeat'] = r['n_complete_defeat']
        save_dict[f'{name}_n_base'] = r['n_base']
    np.savez(save_path, **save_dict)

    print(f"\n\n{'='*70}\n요약\n{'='*70}")
    print(f"{'조건':<22}{'둘 다 속음':>12}{'threshold flag':>16}{'진짜 최악(무력화+미탐지)':>26}")
    print(f"{'(참고) §7 탐지회피無':<22}{PREV_FOOL_BOTH:>12.3f}{PREV_FLAGGED_RATE:>16.3f}{'N/A':>26}")
    for name, r in results.items():
        worst = r['n_complete_defeat'] / max(r['n_base'], 1)
        print(f"{name:<22}{r['rate_both']:>12.3f}{r['flagged_by_threshold'].mean():>16.3f}{worst:>26.3f}")
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()
