"""
§18. [탐색적, 롤백 가능] §16(국소 토큰 세분화)에 §14급 완전판 adaptive evasion 테스트.

배경
----
§17이 local-swap 경로에 나이브 joint attack(방어 구조는 알지만 탐지기 존재는 모름)을
걸어서 16.7%(§7의 18.4%와 근접, §8의 74.4% 함정 회피 확인)를 얻었다. 그런데 §6(전체
재분류)류 설계는 §7뿐 아니라 §14(탐지기 존재까지 아는 완전판 adaptive attacker, STRAP-ViT류
제약 최적화)까지 통과했다 — 공정하게 비교하려면 local-swap도 같은 급의 검증이 필요하다.

이 실험은 §14(patch_fool_joint_evasive.py)와 완전히 같은 구조를 쓰되, 두 번째 타깃만
model8 전체에서 §17의 local_swap_logits(local-swap 경로)로 바꾼다. loss = CE16 + CE_local
- lambda * relu(score - target_bound)^2, score는 model16의 L=12 raw attention top-4 mass
(§14와 동일하게 미분 가능한 형태로 hook에서 계산).

target_bound 두 가지(§14와 동일, 하드코딩 아니고 §6 calibration 산출물에서 그대로 가져옴):
  - clean_max (0.6728): 관측된 clean 최댓값 아래
  - calibrated_threshold (0.5567): 실제 배포된 탐지기 임계값(더 엄격)

이미지 집합은 §7/§8/§17과 동일(seed=123, n=50)이라 직접 비교 가능. 어댑터는 §16/§17과
동일한 입력 레벨 아핀 변환, calibration 100장(seed=42, 공격 표본과 분리)으로 피팅.

주의: src/attacks/patch_fool.py는 import만(수정 없음). 이 파일 지우면 원상복구.

사용법:
  python local_swap_adaptive_evasion_full_test.py --num_samples 50 --seed 123 --chunk 20
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
import torch.nn as nn
import torch.nn.functional as F

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.patch_fool import _collect_attn, _select_patch_attn, _build_mask

# §6 calibration 산출물에서 그대로 가져온 값 (README §14와 동일, 새로 계산 안 함)
CALIBRATED_THRESHOLD = 0.5567247867584229
CLEAN_MAX = 0.6727843284606934

# §17(같은 seed=123/n=50)의 참고값 — README §17
PREV_FOOL_BOTH = 7 / 42        # 0.167, 탐지 회피 없는 local-swap joint attack
PREV_FLAGGED_RATE = 0.04       # README §17


class AffineAdapter(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim), requires_grad=False)
        self.b = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def forward(self, x):
        return x @ self.W + self.b

    @torch.no_grad()
    def fit(self, x_src, y_tgt):
        C = x_src.shape[1]
        ones = torch.ones(x_src.shape[0], 1, dtype=x_src.dtype)
        x_aug = torch.cat([x_src, ones], dim=1)
        sol = torch.linalg.lstsq(x_aug, y_tgt).solution
        self.W.copy_(sol[:C])
        self.b.copy_(sol[C])


def patch_embed_with_pos(model, images):
    x = model.patch_embed(images)
    return x + model.pos_embed[:, 1:, :]


def p16_to_p8_subpatch_indices(idx16, ppl16=14, ppl8=28):
    r16, c16 = idx16 // ppl16, idx16 % ppl16
    r8, c8 = r16 * 2, c16 * 2
    return [r8 * ppl8 + c8, r8 * ppl8 + c8 + 1, (r8 + 1) * ppl8 + c8, (r8 + 1) * ppl8 + c8 + 1]


def local_swap_logits(model16, model8, adapter, images, flag_idx, ppl16=14):
    """§16/§17과 동일 — 미분 가능(torch.no_grad 없음), joint attack 최적화에 씀."""
    B = images.shape[0]
    p16_full = patch_embed_with_pos(model16, images)
    p8_full = patch_embed_with_pos(model8, images)
    p8_adapted = adapter(p8_full.reshape(-1, p8_full.shape[-1])).reshape(p8_full.shape)
    cls = model16.cls_token.expand(B, -1, -1) + model16.pos_embed[:, :1, :]
    seqs = []
    for i in range(B):
        idx16 = int(flag_idx[i])
        keep_mask = torch.ones(p16_full.shape[1], dtype=torch.bool)
        keep_mask[idx16] = False
        kept = p16_full[i, keep_mask]
        sub_idx = p16_to_p8_subpatch_indices(idx16, ppl16=ppl16)
        replacement = p8_adapted[i, sub_idx]
        seqs.append(torch.cat([cls[i], kept, replacement], dim=0))
    x = torch.stack(seqs, dim=0)
    x = model16.pos_drop(x)
    for blk in model16.blocks:
        x = blk(x)
    x = model16.norm(x)
    x = x[:, 0]
    x = model16.fc_norm(x)
    return model16.head(x)


def _make_diff_attn_hook(store, key='attn'):
    """§14와 동일 — grad를 안 끊는 관측 hook (delta까지 역전파 필요)."""
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        store[key] = attn.mean(dim=1)
    return hook


def detection_score(attn):
    cls_to_patch = attn[:, 0, 1:]
    v = cls_to_patch / cls_to_patch.sum(dim=1, keepdim=True)
    return v.topk(4, dim=1).values.sum(dim=1)


def joint_local_swap_attack_evasive(
    model16, model8, adapter, images, labels, device,
    attn_layer_idx=4, num_patch=1,
    train_attack_iters=250, attack_lr=0.22,
    step_size=10, gamma=0.95,
    detect_layer=12, target_bound=0.5567,
    lambda0=5.0, lambda_growth=1.5, lambda_max=500.0,
    lambda_step_every=25, penalty_eps=1e-3,
):
    """§14의 joint_patch_fool_attack_evasive와 동일 구조 — 두 번째 타깃만 model8 전체 대신
    local_swap_logits(§16/§17 경로)로 교체."""
    model16.eval(); model8.eval()
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    B, C, H, W = images.shape

    attn_weights, hooks = _collect_attn(model16)
    with torch.no_grad():
        model16(images)
    for h in hooks:
        h.remove()
    max_patch_index = _select_patch_attn(attn_weights, attn_layer_idx, num_patch, device)
    flag_idx = max_patch_index[:, 0]

    mask = _build_mask(max_patch_index, B, H, 16, device)

    delta = torch.randn_like(images) * 0.1
    original_images = images.clone()
    delta = delta.to(device).requires_grad_(True)

    opt = torch.optim.Adam([delta], lr=attack_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    detect_block = model16.blocks[detect_layer - 1]
    store = {}
    diff_hook = detect_block.attn.register_forward_hook(_make_diff_attn_hook(store))

    lam = lambda0
    score_history, lambda_history = [], []

    try:
        for it in range(train_attack_iters):
            model16.zero_grad(); model8.zero_grad(); opt.zero_grad()
            perturbed = original_images + torch.mul(delta, mask)
            out16 = model16(perturbed)             # store['attn'] 채워짐 (grad 유지)
            out_local = local_swap_logits(model16, model8, adapter, perturbed, flag_idx)
            ce = criterion(out16, labels) + criterion(out_local, labels)

            score = detection_score(store['attn'])
            penalty = F.relu(score - target_bound).pow(2).mean()

            combined = ce - lam * penalty
            combined.backward()
            if delta.grad is not None:
                delta.grad = -delta.grad
            opt.step()
            scheduler.step()

            score_history.append(score.detach().cpu().numpy())
            lambda_history.append(lam)
            if (it + 1) % lambda_step_every == 0:
                mean_violation = F.relu(score.detach() - target_bound).mean().item()
                if mean_violation > penalty_eps:
                    lam = min(lam * lambda_growth, lambda_max)
    finally:
        diff_hook.remove()

    with torch.no_grad():
        adv_images = (original_images + torch.mul(delta, mask)).detach()
    return adv_images, flag_idx, score_history, lambda_history


def collect_layer_attn(model, images):
    weights = []

    def hook(module, inp, out):
        x = inp[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        weights.append(attn.mean(dim=1).detach())

    hooks = [blk.attn.register_forward_hook(hook) for blk in model.blocks]
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()
    return weights


def raw_top4_mass(model16, images, L=12):
    lw = collect_layer_attn(model16, images)
    attn = lw[L - 1]
    v = attn[:, 0, 1:]
    v = v / v.sum(dim=1, keepdim=True)
    return v.topk(4, dim=1).values.sum(dim=1)


def run_one_condition(name, target_bound, model16, model8, adapter, images, labels,
                      both_orig_correct, device, args):
    print(f"\n{'='*70}\n[{name}] target_bound={target_bound:.4f}\n{'='*70}")
    adv_chunks, flag_idx_chunks = [], []
    for s in range(0, args.num_samples, args.chunk):
        e = min(s + args.chunk, args.num_samples)
        print(f"  [{s}:{e}] 처리 중...")
        adv, fi, score_hist, lam_hist = joint_local_swap_attack_evasive(
            model16, model8, adapter, images[s:e], labels[s:e], device,
            attn_layer_idx=args.attn_layer_idx, num_patch=1,
            train_attack_iters=args.iters, target_bound=target_bound,
            lambda0=args.lambda0, lambda_growth=args.lambda_growth,
            lambda_max=args.lambda_max, lambda_step_every=args.lambda_step_every,
            detect_layer=args.detect_layer)
        adv_chunks.append(adv)
        flag_idx_chunks.append(fi)
        print(f"    최종 lambda={lam_hist[-1]:.1f}, 최종 iter 학습중 score 평균={score_hist[-1].mean():.4f}")
    adv = torch.cat(adv_chunks, dim=0)
    flag_idx = torch.cat(flag_idx_chunks, dim=0)

    with torch.no_grad():
        pred16_adv = model16(adv).argmax(dim=1)
        pred_local_adv = local_swap_logits(model16, model8, adapter, adv, flag_idx).argmax(dim=1)
    fool16 = both_orig_correct & (pred16_adv != labels)
    fool_local = both_orig_correct & (pred_local_adv != labels)
    fool_both = fool16 & fool_local
    n_base = int(both_orig_correct.sum().item())
    rate_both = fool_both.sum().item() / max(n_base, 1)

    score_final = raw_top4_mass(model16, adv, args.detect_layer).cpu().numpy()
    flagged_by_threshold = score_final > CALIBRATED_THRESHOLD
    flagged_by_cleanmax = score_final > CLEAN_MAX

    print(f"\n  --- 결과 ({n_base}개 기준) ---")
    print(f"  P16만 속음: {(fool16 & ~fool_local).sum().item()}/{n_base}")
    print(f"  local-swap만 속음: {(fool_local & ~fool16).sum().item()}/{n_base}")
    print(f"  둘 다 속음(방어 완전 무력화): {int(fool_both.sum().item())}/{n_base} = {rate_both:.3f}")
    print(f"  (참고: §17 탐지회피無 local-swap joint attack은 {PREV_FOOL_BOTH:.3f}였음)")
    print(f"\n  실제 배포 임계값({CALIBRATED_THRESHOLD:.4f}) 기준 flag 비율: "
          f"{flagged_by_threshold.mean():.3f}")
    print(f"  clean 최댓값({CLEAN_MAX:.4f}) 기준 flag 비율: {flagged_by_cleanmax.mean():.3f}")
    print(f"  (참고: §17 탐지회피無 local-swap joint attack의 flag 비율은 {PREV_FLAGGED_RATE:.2f}였음)")

    caught_both_fooled = fool_both & torch.tensor(flagged_by_threshold, device=device)
    n_worst = int(caught_both_fooled.sum().item())
    n_complete_defeat = int(fool_both.sum().item()) - n_worst
    print(f"\n  *** 진짜 최악(완전 무력화 + 탐지도 회피) ***: "
          f"{n_complete_defeat}/{n_base} = {n_complete_defeat/max(n_base,1):.3f}")

    return dict(name=name, target_bound=target_bound,
                pred16_adv=pred16_adv.cpu().numpy(), pred_local_adv=pred_local_adv.cpu().numpy(),
                score_final=score_final, rate_both=rate_both,
                flagged_by_threshold=flagged_by_threshold, flagged_by_cleanmax=flagged_by_cleanmax,
                n_complete_defeat=n_complete_defeat, n_base=n_base)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123, help='§7/§8/§17과 동일 -> 직접 비교')
    parser.add_argument('--num_calib', type=int, default=100)
    parser.add_argument('--calib_seed', type=int, default=42)
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

    # ── 어댑터 피팅 (calibration, 공격 표본과 분리된 seed) ──
    calib_loader, _ = get_dataloader(batch_size=args.num_calib, num_samples=args.num_calib, seed=args.calib_seed)
    calib_images, _ = next(iter(calib_loader))
    calib_images = calib_images.to(device)
    with torch.no_grad():
        p16_full = patch_embed_with_pos(model16, calib_images)
        p8_full = patch_embed_with_pos(model8, calib_images)
    src_list, tgt_list = [], []
    for idx16 in range(p16_full.shape[1]):
        for si in p16_to_p8_subpatch_indices(idx16):
            src_list.append(p8_full[:, si, :])
            tgt_list.append(p16_full[:, idx16, :])
    adapter = AffineAdapter(dim=p16_full.shape[-1])
    adapter.fit(torch.cat(src_list, dim=0).cpu(), torch.cat(tgt_list, dim=0).cpu())
    adapter = adapter.to(device)
    print(f"어댑터 피팅 완료 (calibration seed={args.calib_seed}, n={args.num_calib})")

    # ── 공격 표본 (§7/§8/§17과 동일: seed=123, n=50) ──
    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        pred16_clean = model16(images).argmax(dim=1)
        attn_w, hooks = _collect_attn(model16)
        model16(images)
        for h in hooks:
            h.remove()
        gt_idx = _select_patch_attn(attn_w, args.attn_layer_idx, 1, device)[:, 0]
        pred_local_clean = local_swap_logits(model16, model8, adapter, images, gt_idx).argmax(dim=1)
    both_orig_correct = (pred16_clean == labels) & (pred_local_clean == labels)
    n_base_check = int(both_orig_correct.sum().item())
    print(f"[참고] P16 clean acc={(pred16_clean==labels).float().mean().item():.3f}  "
          f"local-swap clean acc={(pred_local_clean==labels).float().mean().item():.3f}  "
          f"둘 다 원래 맞춘 이미지: {n_base_check}/{args.num_samples} "
          f"(§17과 동일 seed=123이면 42개 근처여야 함)")

    results = {}
    for name, bound in [('clean_max', CLEAN_MAX), ('calibrated_threshold', CALIBRATED_THRESHOLD)]:
        results[name] = run_one_condition(name, bound, model16, model8, adapter, images, labels,
                                          both_orig_correct, device, args)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'18_local_swap_adaptive_evasion_full_n{args.num_samples}_seed{args.seed}.npz')
    save_dict = {'both_orig_correct': both_orig_correct.cpu().numpy(), 'labels': labels.cpu().numpy()}
    for name, r in results.items():
        save_dict[f'{name}_rate_both'] = r['rate_both']
        save_dict[f'{name}_score_final'] = r['score_final']
        save_dict[f'{name}_flagged_by_threshold'] = r['flagged_by_threshold']
        save_dict[f'{name}_flagged_by_cleanmax'] = r['flagged_by_cleanmax']
        save_dict[f'{name}_n_complete_defeat'] = r['n_complete_defeat']
        save_dict[f'{name}_n_base'] = r['n_base']
    np.savez(save_path, **save_dict)

    print(f"\n\n{'='*70}\n요약\n{'='*70}")
    print(f"{'조건':<22}{'둘 다 속음':>12}{'threshold flag':>16}{'진짜 최악(무력화+미탐지)':>26}")
    print(f"{'(참고) §17 탐지회피無':<22}{PREV_FOOL_BOTH:>12.3f}{PREV_FLAGGED_RATE:>16.3f}{'N/A':>26}")
    for name, r in results.items():
        worst = r['n_complete_defeat'] / max(r['n_base'], 1)
        print(f"{name:<22}{r['rate_both']:>12.3f}{r['flagged_by_threshold'].mean():>16.3f}{worst:>26.3f}")
    print(f"\nSaved: {save_path}")


if __name__ == '__main__':
    main()
