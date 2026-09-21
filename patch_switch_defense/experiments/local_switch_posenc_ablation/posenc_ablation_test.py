"""
experiments/local_switch_posenc_ablation/posenc_ablation_test.py — [탐색적, 옵션 A]
local_switch의 4개 P8 서브패치가 지금 "quadrant(어느 위치의 서브패치인지)"를 구분하는 신호
없이 브릿지되는 게 아닌지 확인하기 위한 빠른 ablation.

배경
----
`defense/local_switch.py`의 `patch_embed_with_pos`는 P8 서브패치에 P16의 위치정보를
전혀 섞지 않고 P8 자신의 native pos_embed만 쓴다(자세한 설명은 대화 기록 참고). 어댑터
(`AffineAdapter`)는 4개 서브패치 전부에 대해 **하나의 공유 아핀 변환**을 학습하는데, 이때
학습 타깃이 4개 다 동일(그 부모 P16 패치 하나)이라, "이게 4개 중 몇 번째 서브패치인지"를
구분하는 명시적 신호가 브릿지 안에 없다. APT(arXiv 2510.18091)의
`TokenizedZeroConvPatchAttn`이 여러 sub-patch를 하나로 합칠 때 위치 구분 벡터를 쓰는 걸
참고해서, 재학습 없이(closed-form 어댑터는 그대로 두고) 4개의 **고정된(학습 안 된) quadrant
구분 벡터**를 브릿지 입력에 더해보고 복원율이 89.4%(system_comparison, n=250/seed=42)에서
개선되는지 본다.

방법
----
system_comparison과 완전히 동일한 n=250(seed=42), calibration 100/eval 150 분리를 쓴다.
같은 실행 안에서 두 조건을 전부 계산한다(PatchFool 공격은 한 번만 생성해서 공유 — 별도
실행으로 나누면 공격 생성의 미세한 비결정성이 비교에 섞일 수 있음):
  (A) baseline: `defense.local_switch.fit_adapter`/`apply_local_switch` 그대로(기존 89.4%
      재현 확인용)
  (B) +quadrant posenc: 4개의 고정 orthogonal 벡터(학습 안 됨, seed 고정)를 어댑터 피팅 전에
      P8 서브패치 임베딩에 더한 뒤(피팅 시점과 적용 시점 모두 동일하게), 나머지는 (A)와 동일

quadrant 벡터의 크기(scale)는 임의로 정하지 않고, calibration의 실제 patch embedding 평균
노름의 --posenc_scale_frac(기본 0.1) 배로 데이터 기반으로 정한다.

주의: 이 파일은 `defense/local_switch.py`를 수정하지 않는다(아직 검증 안 된 실험이라 프로젝트
관례상 defense/에 반영하지 않음) — 필요한 함수(`patch_embed_with_pos`,
`p16_to_p8_subpatch_indices`, `AffineAdapter`, `full_forward_from_tokens`)만 그대로
import해서 쓰고, quadrant 벡터를 더하는 부분만 이 폴더 안에서 새로 감싼다.

사용법:
  python posenc_ablation_test.py --num_samples 250 --cal_frac 0.4 --seed 42 --chunk 20 \
    --posenc_scale_frac 0.1

결과 (2026-09-20, job 2293640/2293693, RTX 4090, n_attacked=112)
----
scale=0.1(평균 노름의 10%)과 scale=1.0(100%) 둘 다에서 **완전히 동일한 결과**
(101/112=90.18%, delta=0.0000 — 단 하나의 예측도 안 바뀜)가 나왔다. 스케일을 10배 키워도
전혀 변화가 없다는 건 "스케일이 작아서"가 아니라 **이 방식 자체가 원리적으로 효과가 없다**는
뜻으로 해석된다: 하나의 공유 아핀 변환 W,b가 모든 4개 서브패치의 (같은 quadrant면 항상 같은)
고정 벡터를 입력에 받으면, 최소제곱 피팅이 이 고정 성분을 그냥 하나의 상수 이동으로 흡수해서
W,b를 재조정해버릴 뿐, quadrant별로 다른 보정을 학습하게 만들지 못한다 — 애초에 "공유된 하나의
선형 변환"이라는 설계 자체가 이런 종류의 위치 구분 신호를 살릴 수 없는 구조다. 따라서 옵션 B
(APT류 conv 집약 + zero-init, 위치별로 실제로 다르게 반응할 수 있는 비선형/conv 메커니즘)로
가지 않는 한 이 경로로는 개선 여지가 없어 보인다 — 대화 기록 참고, 옵션 B는 사용자 확인 후 진행.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(ROOT, 'experiments')):
    ROOT = os.path.dirname(ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
SHARED_SRC_ROOT = os.path.dirname(ROOT)  # ViT_robust -- shared src/ (models/dataset/attacks)
DETECT_ROOT = os.path.join(SHARED_SRC_ROOT, 'patch_attack_detector')  # for `from detector.topk_mass_v1 import ...`
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, DETECT_ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np
import torch

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.patch_fool import patch_fool_attack
from detector.topk_mass_v1 import localize_top1
from defense.local_switch import (
    AffineAdapter, patch_embed_with_pos, p16_to_p8_subpatch_indices,
    full_forward_from_tokens, fit_adapter as fit_adapter_baseline,
    apply_local_switch as apply_local_switch_baseline,
)
from eval_utils import calibration_eval_split, wilson_ci


def build_quadrant_lookup(ppl16=14, ppl8=28, device='cpu'):
    """P8의 784개 flat index 각각이 자기 부모 P16 패치 안에서 몇 번째(0~3) 서브패치인지."""
    lookup = torch.zeros(ppl8 * ppl8, dtype=torch.long)
    for idx16 in range(ppl16 * ppl16):
        sub_idx = p16_to_p8_subpatch_indices(idx16, ppl16=ppl16, ppl8=ppl8)
        for q, si in enumerate(sub_idx):
            lookup[si] = q
    return lookup.to(device)


def fixed_orthogonal_quadrant_vectors(dim, seed=0, device='cpu'):
    """4개의 고정(학습 안 된) orthogonal 벡터. 단위 노름, 서로 직교."""
    g = torch.Generator().manual_seed(seed)
    m = torch.empty(4, dim)
    torch.nn.init.orthogonal_(m, generator=g)
    return m.to(device)


@torch.no_grad()
def fit_adapter_variant(model16, model8, calib_images, quadrant_add=None, ppl16=14, ppl8=28):
    """defense.local_switch.fit_adapter와 동일 로직 + quadrant_add(옵션) 벡터 추가."""
    p16_full = patch_embed_with_pos(model16, calib_images)
    p8_full = patch_embed_with_pos(model8, calib_images)
    if quadrant_add is not None:
        p8_full = p8_full + quadrant_add
    N, n16, D = p16_full.shape
    src_list, tgt_list = [], []
    for idx16 in range(n16):
        sub_idx = p16_to_p8_subpatch_indices(idx16, ppl16=ppl16, ppl8=ppl8)
        for si in sub_idx:
            src_list.append(p8_full[:, si, :])
            tgt_list.append(p16_full[:, idx16, :])
    x_src = torch.cat(src_list, dim=0)
    y_tgt = torch.cat(tgt_list, dim=0)
    adapter = AffineAdapter(dim=D)
    adapter.fit(x_src.cpu(), y_tgt.cpu())
    return adapter.to(calib_images.device)


@torch.no_grad()
def apply_local_switch_variant(model16, model8, adapter, images, flag_idx, quadrant_add=None,
                                ppl16=14, ppl8=28):
    """defense.local_switch.apply_local_switch와 동일 로직 + quadrant_add(옵션) 벡터 추가."""
    B = images.shape[0]
    p16_full = patch_embed_with_pos(model16, images)
    p8_full = patch_embed_with_pos(model8, images)
    if quadrant_add is not None:
        p8_full = p8_full + quadrant_add
    p8_adapted = adapter(p8_full.reshape(-1, p8_full.shape[-1])).reshape(p8_full.shape)
    cls = model16.cls_token.expand(B, -1, -1) + model16.pos_embed[:, :1, :]
    seqs = []
    for i in range(B):
        idx16 = int(flag_idx[i])
        keep_mask = torch.ones(p16_full.shape[1], dtype=torch.bool)
        keep_mask[idx16] = False
        kept = p16_full[i, keep_mask]
        sub_idx = p16_to_p8_subpatch_indices(idx16, ppl16=ppl16, ppl8=ppl8)
        replacement = p8_adapted[i, sub_idx]
        seq = torch.cat([cls[i], kept, replacement], dim=0)
        seqs.append(seq)
    seq = torch.stack(seqs, dim=0)
    return full_forward_from_tokens(model16, seq)


def chunked_patch_fool(model16, images, labels, device, attn_layer_idx, chunk):
    adv_chunks = []
    for s in range(0, images.shape[0], chunk):
        e = min(s + chunk, images.shape[0])
        adv_chunk, _ = patch_fool_attack(
            model16, images[s:e], labels[s:e], device, patch_size_model=16,
            attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn',
            attn_layer_idx=attn_layer_idx)
        adv_chunks.append(adv_chunk)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return torch.cat(adv_chunks, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=250,
                        help='system_comparison과 동일(calibration+evaluation 합계)')
    parser.add_argument('--cal_frac', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    parser.add_argument('--detect_layer', type=int, default=12)
    parser.add_argument('--chunk', type=int, default=20)
    parser.add_argument('--posenc_seed', type=int, default=0, help='quadrant 벡터 고정 seed')
    parser.add_argument('--posenc_scale_frac', type=float, default=0.1,
                        help='quadrant 벡터 노름 = calibration patch embedding 평균 노름 * 이 값')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cal, ev = calibration_eval_split(args.num_samples, frac=args.cal_frac)
    n_cal, n_eval = cal.stop - cal.start, ev.stop - ev.start
    print(f"calibration: {n_cal}개, evaluation: {n_eval}개 (system_comparison과 동일 split)")

    device = get_device()
    model16 = load_vit_model(16, device); model16.eval()
    model8 = load_vit_model(8, device); model8.eval()

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    calib_images = images[cal]
    eval_images, eval_labels = images[ev], labels[ev]

    with torch.no_grad():
        pred16_clean = model16(images).argmax(dim=1)

    # ── quadrant 벡터 스케일: calibration patch embedding의 실제 평균 노름 기준 ──
    with torch.no_grad():
        p8_calib = patch_embed_with_pos(model8, calib_images)
    mean_norm = p8_calib.reshape(-1, p8_calib.shape[-1]).norm(dim=1).mean().item()
    D = p8_calib.shape[-1]
    quadrant_lookup = build_quadrant_lookup(device=device)
    quad_vecs = fixed_orthogonal_quadrant_vectors(D, seed=args.posenc_seed, device=device)
    quad_vecs = quad_vecs * (mean_norm * args.posenc_scale_frac)
    print(f"[posenc] calibration patch embedding 평균 노름={mean_norm:.3f}, "
          f"quadrant 벡터 노름={quad_vecs[0].norm().item():.3f} "
          f"(= 평균 노름의 {args.posenc_scale_frac:.0%})")
    quadrant_add_calib = quad_vecs[build_quadrant_lookup(device=device)]  # (784, D), calib용과 동일 lookup
    quadrant_add_eval = quad_vecs[quadrant_lookup]  # (784, D)

    # ── 어댑터 2종 피팅: baseline(기존 코드 그대로) vs +quadrant posenc ──
    adapter_base = fit_adapter_baseline(model16, model8, calib_images)
    adapter_pos = fit_adapter_variant(model16, model8, calib_images, quadrant_add=quadrant_add_calib)
    print("어댑터 2종 피팅 완료 (baseline / +quadrant posenc)")

    # ── PatchFool 공격: eval에 한 번만 생성, 두 조건이 공유 ──
    print(f"\n[PatchFool on P16] 공격 생성 중 (eval {n_eval}개, {args.chunk}개씩)...")
    adv_eval = chunked_patch_fool(model16, eval_images, eval_labels, device, args.attn_layer_idx, args.chunk)
    with torch.no_grad():
        pred16_adv_eval = model16(adv_eval).argmax(dim=1)

    flag_idx_eval = localize_top1(model16, adv_eval, detect_layer=args.detect_layer)

    # ── 두 조건을 같은 공격 이미지에 적용 ──
    pred_base, pred_pos = [], []
    for s in range(0, n_eval, args.chunk):
        e = min(s + args.chunk, n_eval)
        pred_base.append(apply_local_switch_baseline(
            model16, model8, adapter_base, adv_eval[s:e], flag_idx_eval[s:e]).argmax(dim=1))
        pred_pos.append(apply_local_switch_variant(
            model16, model8, adapter_pos, adv_eval[s:e], flag_idx_eval[s:e],
            quadrant_add=quadrant_add_eval).argmax(dim=1))
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    pred_base = torch.cat(pred_base, dim=0)
    pred_pos = torch.cat(pred_pos, dim=0)

    pred16_clean_ev = pred16_clean[ev]

    def recovery_stats(pred_defended):
        attack_succeeded = (pred16_clean_ev == eval_labels) & (pred16_adv_eval != eval_labels)
        n_attacked = int(attack_succeeded.sum().item())
        recovered = attack_succeeded & (pred_defended == eval_labels)
        n_recovered = int(recovered.sum().item())
        rate = n_recovered / max(n_attacked, 1)
        return n_attacked, n_recovered, rate, wilson_ci(n_recovered, max(n_attacked, 1))

    n_att_b, n_rec_b, rate_b, ci_b = recovery_stats(pred_base)
    n_att_p, n_rec_p, rate_p, ci_p = recovery_stats(pred_pos)
    assert n_att_b == n_att_p, "두 조건이 같은 공격-성공 집합을 써야 함"

    print(f"\n=== 복원율 (공격 성공 {n_att_b}/{n_eval} 기준, 같은 공격 이미지) ===")
    print(f"  baseline (posenc 없음)   : {n_rec_b}/{n_att_b} = {rate_b:.4f}  "
          f"95% CI [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
    print(f"  +quadrant posenc (옵션 A) : {n_rec_p}/{n_att_p} = {rate_p:.4f}  "
          f"95% CI [{ci_p[0]:.4f}, {ci_p[1]:.4f}]")
    print(f"  delta: {rate_p - rate_b:+.4f}")
    print(f"\n[참고] system_comparison(n=250, seed=42)의 기존 baseline 복원율: 89.4% (101/113, "
          f"95% CI [0.8235, 0.9382]) — 이번 실행의 baseline({rate_b:.4f})과 비교해 재현성 확인용")

    out_dir = HERE
    os.makedirs(out_dir, exist_ok=True)
    out_name = f'posenc_ablation_scale{args.posenc_scale_frac}.npz'
    np.savez(
        os.path.join(out_dir, out_name),
        n_cal=n_cal, n_eval=n_eval, n_attacked=n_att_b,
        mean_norm=mean_norm, posenc_scale_frac=args.posenc_scale_frac,
        rate_base=rate_b, ci_base=ci_b, n_recovered_base=n_rec_b,
        rate_posenc=rate_p, ci_posenc=ci_p, n_recovered_posenc=n_rec_p,
    )
    print(f"\nSaved: {os.path.join(out_dir, out_name)}")


if __name__ == '__main__':
    main()
