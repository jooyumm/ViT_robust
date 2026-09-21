"""
§17. [탐색적, 롤백 가능] §16(국소 토큰 세분화)에 joint attack stress test.

배경
----
§16은 P16 12개 레이어를 안 건드리고, 의심 패치 1개만 P8 서브패치(선형 어댑터로 보정)로
in-place 교체하는 방어다(나이브 PatchFool 공격 기준 복원율 84.6%). 그런데 §8(diversity
diagnostic)이 이미 경고했다 — "두 표현을 정렬(공유 좌표계)시키면 joint attack에 오히려
더 취약해진다"(같은 patch size·다른 학습 74.4% 무력화 vs 다른 patch size 18.4%). §16의
선형 어댑터는 정확히 "P8을 P16 좌표계에 맞추는" 정렬 작업이라, 이 함정에 빠질 후보다.

이 실험은 §7(joint_attack_test.py)과 완전히 같은 세팅(seed=123, n=50, 250 iter,
attn_layer_idx=4)으로, P8 전체가 아니라 **§16의 local-swap 경로**를 두 번째 타깃으로 놓고
같은 joint attack을 건다. §7/§8 수치(나이브 전이 2.9%/18.4%/6배, diversity 18.4%/74.4%)와
바로 비교 가능하게 만드는 게 핵심.

방법
----
- (A) 나이브 baseline: P16만 공격(patch_fool_attack, 기존과 동일) → local-swap 경로까지
  같이 속는 비율 ("전이" 비율, §16의 84.6% 복원율의 반대 표현이지만 이번엔 seed=123
  50장으로 다시 재서 §7/§8과 동일 표본 기준으로 비교 가능하게 함 — §8이 처음 겪었던
  "다른 표본끼리 비교" 실수를 반복하지 않기 위함)
- (B) joint attack: loss = CE(P16(perturbed)) + CE(local_swap_forward(perturbed)) 를
  하나의 delta로 동시 최적화(§7의 joint_patch_fool_attack과 동일 구조, 두 번째 항만 교체)
- 어댑터는 §16과 동일한 입력 레벨 아핀 변환, calibration 100장(seed=42, 공격 표본과
  완전히 다른 seed라 안 겹침)으로 닫힌 형태 피팅
- 공격 위치는 §7과 동일하게 attn_layer_idx=4 clean attention 기준 ground-truth 패치
  (탐지 회피는 다루지 않음 — 그건 §14급 후속 작업)

주의: src/attacks/patch_fool.py는 import만(수정 없음). 이 파일 지우면 원상복구.

사용법:
  python local_swap_joint_attack_test.py --num_samples 50 --seed 123 --num_calib 100
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

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.patch_fool import patch_fool_attack, _collect_attn, _select_patch_attn, _build_mask

CALIBRATED_THRESHOLD = 0.5567  # §6/§7과 동일 — 새로 안 정하고 그대로 재사용


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
    """§16의 local-swap 경로를 처음부터 끝까지 미분 가능하게 계산 (joint attack 최적화용,
    torch.no_grad 없음). images에 대해 flag_idx[i] 위치의 P16 패치를 대응 P8 서브패치
    4개(어댑터 보정)로 in-place 교체하고, P16의 12개 레이어(가중치 불변)를 그대로 통과."""
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
        seq = torch.cat([cls[i], kept, replacement], dim=0)
        seqs.append(seq)
    x = torch.stack(seqs, dim=0)

    x = model16.pos_drop(x)
    for blk in model16.blocks:
        x = blk(x)
    x = model16.norm(x)
    x = x[:, 0]
    x = model16.fc_norm(x)
    return model16.head(x)


def joint_patch_fool_local_swap_attack(model16, model8, adapter, images, labels, device,
                                       attn_layer_idx=4, num_patch=1,
                                       train_attack_iters=250, attack_lr=0.22,
                                       step_size=10, gamma=0.95):
    """§7의 joint_patch_fool_attack과 동일 구조 — 두 번째 타깃만 model8 전체 대신
    local_swap_logits(§16 경로)로 교체."""
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

    for it in range(train_attack_iters):
        model16.zero_grad(); model8.zero_grad(); opt.zero_grad()
        perturbed = original_images + torch.mul(delta, mask)
        out16 = model16(perturbed)
        out_local = local_swap_logits(model16, model8, adapter, perturbed, flag_idx)
        loss = criterion(out16, labels) + criterion(out_local, labels)
        loss.backward()
        if delta.grad is not None:
            delta.grad = -delta.grad
        opt.step()
        scheduler.step()

    with torch.no_grad():
        adv_images = (original_images + torch.mul(delta, mask)).detach()
    return adv_images, flag_idx


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


def raw_top4_mass_l12(model16, images, L=12):
    lw = collect_layer_attn(model16, images)
    attn = lw[L - 1]
    v = attn[:, 0, 1:]
    v = v / v.sum(dim=1, keepdim=True)
    return v.topk(4, dim=1).values.sum(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=123, help='§7/§8과 동일 표본 (calibration/eval seed=42와 다름)')
    parser.add_argument('--num_calib', type=int, default=100, help='어댑터 피팅용, 별도 seed')
    parser.add_argument('--calib_seed', type=int, default=42)
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()

    # ── 어댑터 피팅 (calibration, 공격 표본과 완전히 다른 seed) ──
    calib_loader, _ = get_dataloader(batch_size=args.num_calib, num_samples=args.num_calib, seed=args.calib_seed)
    calib_images, _ = next(iter(calib_loader))
    calib_images = calib_images.to(device)
    with torch.no_grad():
        p16_full = patch_embed_with_pos(model16, calib_images)
        p8_full = patch_embed_with_pos(model8, calib_images)
    n16 = p16_full.shape[1]
    src_list, tgt_list = [], []
    for idx16 in range(n16):
        for si in p16_to_p8_subpatch_indices(idx16):
            src_list.append(p8_full[:, si, :])
            tgt_list.append(p16_full[:, idx16, :])
    adapter = AffineAdapter(dim=p16_full.shape[-1])
    adapter.fit(torch.cat(src_list, dim=0).cpu(), torch.cat(tgt_list, dim=0).cpu())
    adapter = adapter.to(device)
    print(f"어댑터 피팅 완료 (calibration seed={args.calib_seed}, n={args.num_calib}, "
          f"공격 표본 seed={args.seed}과 분리됨)")

    # ── 공격 표본 (§7/§8과 동일: seed=123, n=50) ──
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
    print(f"[참고] P16 clean acc={(pred16_clean==labels).float().mean().item():.3f}  "
          f"local-swap(위치=clean attention top-1) clean acc={(pred_local_clean==labels).float().mean().item():.3f}  "
          f"둘 다 원래 맞춘 이미지: {both_orig_correct.sum().item()}/{args.num_samples}")
    n_base = int(both_orig_correct.sum().item())

    # ── (A) 나이브 baseline: P16만 공격, local-swap 경로까지 같이 속는지(전이) ──
    print(f"\n[A. 나이브 baseline] P16 단독 공격 생성 중...")
    adv_single, _ = patch_fool_attack(
        model16, images, labels, device, patch_size_model=16,
        attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn',
        attn_layer_idx=args.attn_layer_idx)
    with torch.no_grad():
        pred16_single = model16(adv_single).argmax(dim=1)
        pred_local_single = local_swap_logits(model16, model8, adapter, adv_single, gt_idx).argmax(dim=1)
    fool16_single = both_orig_correct & (pred16_single != labels)
    fool_both_single = fool16_single & (pred_local_single != labels)
    n_fool16_single = int(fool16_single.sum().item())
    naive_transfer_rate = fool_both_single.sum().item() / max(n_fool16_single, 1)
    print(f"  P16만 공격: P16 속음 {n_fool16_single}/{n_base}, "
          f"그 중 local-swap까지 같이 속음(전이) {int(fool_both_single.sum().item())}/{max(n_fool16_single,1)} "
          f"= {naive_transfer_rate:.3f}")

    # ── (B) joint attack: P16 + local-swap 경로 동시 공격 ──
    print(f"\n[B. Joint attack] P16+local-swap 동시 공격 생성 중...")
    adv_joint, flag_idx_joint = joint_patch_fool_local_swap_attack(
        model16, model8, adapter, images, labels, device,
        attn_layer_idx=args.attn_layer_idx, num_patch=1, train_attack_iters=250)
    with torch.no_grad():
        pred16_joint = model16(adv_joint).argmax(dim=1)
        pred_local_joint = local_swap_logits(model16, model8, adapter, adv_joint, flag_idx_joint).argmax(dim=1)

    fool16_joint = both_orig_correct & (pred16_joint != labels)
    fool_local_joint = both_orig_correct & (pred_local_joint != labels)
    fool_both_joint = fool16_joint & fool_local_joint
    joint_defeat_rate = fool_both_joint.sum().item() / max(n_base, 1)
    print(f"\n=== Joint attack 결과 (원래 둘 다 맞춘 {n_base}개 기준) ===")
    print(f"  P16만 속음(local-swap은 여전히 맞음): {(fool16_joint & ~fool_local_joint).sum().item()}/{n_base}")
    print(f"  local-swap만 속음(P16은 여전히 맞음): {(fool_local_joint & ~fool16_joint).sum().item()}/{n_base}")
    print(f"  둘 다 속음(방어 완전 무력화): {int(fool_both_joint.sum().item())}/{n_base} = {joint_defeat_rate:.3f}")
    print(f"  (참고, §7: P16+P8 전체 joint = 18.4%, §8: 정렬된 P16-A/B joint = 74.4%(페어링))")
    print(f"  (참고, 나이브 전이 위에서: {naive_transfer_rate:.3f})")

    # ── 탐지기 (기존 calibration 임계값 재사용) ──
    score_joint = raw_top4_mass_l12(model16, adv_joint).cpu().numpy()
    flagged_joint = score_joint > CALIBRATED_THRESHOLD
    flagged_joint_t = torch.tensor(flagged_joint, device=device)
    caught_both_fooled = fool_both_joint & flagged_joint_t
    print(f"\n=== 탐지기 (calibration 임계값 {CALIBRATED_THRESHOLD} 그대로) ===")
    print(f"  joint attack {args.num_samples}개 중 flag 비율: {flagged_joint.mean():.3f}")
    print(f"  '둘 다 속은'(완전 무력화) 중 그래도 탐지는 된 것: "
          f"{int(caught_both_fooled.sum().item())}/{max(int(fool_both_joint.sum().item()),1)}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'17_local_swap_joint_attack_n{args.num_samples}.npz'),
             n_base=n_base, n_fool16_single=n_fool16_single,
             naive_transfer_rate=naive_transfer_rate,
             n_fool_both_joint=int(fool_both_joint.sum().item()),
             joint_defeat_rate=joint_defeat_rate,
             flagged_joint_frac=float(flagged_joint.mean()),
             n_caught_both_fooled=int(caught_both_fooled.sum().item()))
    print(f"\nSaved: {os.path.join(out_dir, f'17_local_swap_joint_attack_n{args.num_samples}.npz')}")


if __name__ == '__main__':
    main()
