"""
experiments/system_comparison/system_comparison_test.py — all_switch와 local_switch를
**같은 이미지, 같은 탐지 결과**로 직접 비교하는 단일 실험.

배경
----
기존 §6(all_switch/final_validation, seed=42/n=200, 100 calibration+100 eval)과
§16(local_switch/recovery_test, seed=42지만 num_calib/num_eval이 달라 경계가 다름)은
get_dataloader가 seed 고정 permutation의 prefix를 자르는 방식이라(src/dataset.py) 둘 다
seed=42로 겹치는 인덱스를 뽑긴 했지만, calibration/eval 경계가 서로 달라 §16의 eval
슬라이스가 §6의 calibration 구간과 절반 겹치는 등 완전히 같은 조건이 아니었다. 그래서
"97.1%(§6) vs 89.8%(§16)"이라는 헤드라인 비교는 엄밀히 말해 서로 다른 이미지·다른 탐지
결과에서 나온 숫자였다. §6/§16은 제거됐다(git 히스토리에서 복구 가능) — 이 실험이 그 역할을
완전히 대체한다: 탐지 임계값 calibration, local_switch 어댑터 calibration, 평가용 held-out
전부 **하나의 calibration/eval 분리 표본**에서 나오고, "flag 여부"는 한 번만 계산돼 두
방어 메커니즘에 동일하게 적용된다.

방법
----
1) num_samples장을 calibration/eval로 분리(calibration_eval_split, 임계값용과 어댑터용을
   같은 calibration 슬라이스에서 따로 씀 — 둘 다 eval과는 겹치지 않음이 핵심).
2) calibration 이미지(clean)로 local_switch 어댑터를 피팅(defense.local_switch.fit_adapter).
3) sanity: patch_embed_with_pos+full_forward_from_tokens로 재구성한 순수 P16 forward가
   model16(images)와 완전히 같은지 확인(배선 검증, §16의 sanity 조건과 동일).
4) calibration+eval 전체 이미지에 PatchFool 공격 생성(attn_layer_idx=4, §6/§16과 동일).
5) 공유 탐지기(patch_attack_detector 프로젝트의 detector.py)로 L=12 top4_mass 점수를 계산하고,
   calibration에서만 Youden 임계값을 정한다(순환평가 방지, §6와 동일 원칙).
6) eval 슬라이스에서: 그 임계값으로 flag 여부를 **한 번** 계산 -> all_switch/local_switch
   둘 다 이 flag를 그대로 쓴다(공통된 탐지 결과 보장).
7) all_switch/local_switch 각각에 대해: (a) 무조건 복원율(공격 성공 표본에 방어를 적용했을 때,
   flag 여부와 무관하게 복원되는 비율 — 메커니즘 자체의 능력), (b) 시스템 정확도(evaluation
   전체에서, flag된 것만 방어 적용 + 나머지는 P16 그대로 — 실배포 조건), (c) clean 오탐 비용
   (clean eval 이미지에 방어를 적용했을 때 정확도 변화)을 계산한다.

주의: src/attacks/patch_fool.py는 import만(수정 없음). defense/ 두 모듈(all_switch/
local_switch)과 patch_attack_detector의 detector.py도 import만.

사용법:
  python system_comparison_test.py --num_samples 250 --cal_frac 0.4 --seed 42 --chunk 20
  (기본값: calibration 100 + eval 150)
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
from detector.topk_mass_v1 import collect_layer_attn, raw_at_layer, top4_mass, localize_top1
from defense.all_switch import apply_all_switch
from defense.local_switch import fit_adapter, apply_local_switch, patch_embed_with_pos, full_forward_from_tokens
from eval_utils import calibration_eval_split, youden_threshold, wilson_ci


def detection_scores(model, images, detect_layer):
    lw = collect_layer_attn(model, images)
    v = raw_at_layer(lw, detect_layer)
    return top4_mass(v).cpu().numpy()


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
    parser.add_argument('--num_samples', type=int, default=250, help='calibration+evaluation 합계')
    parser.add_argument('--cal_frac', type=float, default=0.4, help='calibration 비율 (기본 100/150)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    parser.add_argument('--detect_layer', type=int, default=12)
    parser.add_argument('--chunk', type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cal, ev = calibration_eval_split(args.num_samples, frac=args.cal_frac)
    n_cal = cal.stop - cal.start
    n_eval = ev.stop - ev.start
    print(f"calibration: {n_cal}개 (인덱스 0~{n_cal-1}), evaluation: {n_eval}개 "
          f"(인덱스 {n_cal}~{n_cal+n_eval-1})")

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
        pred8_clean_all = model8(images).argmax(dim=1)
    print(f"[참고] P16 clean acc: {(pred16_clean == labels).float().mean().item():.3f}  "
          f"P8 clean acc: {(pred8_clean_all == labels).float().mean().item():.3f}")

    # ── local_switch 어댑터: calibration 이미지(clean)로만 피팅 ──────────
    adapter = fit_adapter(model16, model8, calib_images)
    print(f"어댑터 피팅 완료 (calibration {n_cal}개 clean 이미지 기준)")

    # ── sanity: 배선 검증 (교체 없이 재구성한 forward == model16 forward) ──
    with torch.no_grad():
        cls0 = model16.cls_token.expand(eval_images.shape[0], -1, -1) + model16.pos_embed[:, :1, :]
        seq0 = torch.cat([cls0, patch_embed_with_pos(model16, eval_images)], dim=1)
        out_sanity = full_forward_from_tokens(model16, seq0)
        out_ref = model16(eval_images)
    max_diff = (out_sanity - out_ref).abs().max().item()
    print(f"[sanity] 배선 검증 max diff = {max_diff:.6f} (0에 가까워야 정상)")
    assert max_diff < 1e-4, "local_switch 배선 검증 실패 — patch_embed_with_pos/full_forward_from_tokens 확인 필요"

    # ── PatchFool 공격: calibration+eval 전체 (calibration은 임계값 fitting용) ──
    print(f"\n[PatchFool on P16] 공격 생성 중 (총 {args.num_samples}개, {args.chunk}개씩 나눠서)...")
    adv = chunked_patch_fool(model16, images, labels, device, args.attn_layer_idx, args.chunk)

    with torch.no_grad():
        pred16_adv = model16(adv).argmax(dim=1)

    # ── 탐지 점수 (공유 탐지기, 두 메커니즘이 그대로 재사용) ──────────────
    score_clean = detection_scores(model16, images, args.detect_layer)
    score_adv = detection_scores(model16, adv, args.detect_layer)

    # ── 임계값: calibration에서만 ────────────────────────────────────
    thr, j = youden_threshold(score_adv[cal], score_clean[cal])
    print(f"\n=== 임계값 (calibration {n_cal}개로만 도출) === threshold={thr:.4f}  J={j:.3f}")

    # ── eval에서: flag 여부를 한 번만 계산, 두 방어 모두 그대로 사용 ───────
    flagged_clean_ev = score_clean[ev] > thr
    flagged_adv_ev = score_adv[ev] > thr
    fpr = flagged_clean_ev.mean()
    recall = flagged_adv_ev.mean()
    fpr_ci = wilson_ci(int(flagged_clean_ev.sum()), n_eval)
    recall_ci = wilson_ci(int(flagged_adv_ev.sum()), n_eval)
    print(f"\n=== 탐지 (evaluation {n_eval}개, held-out, 두 방어 공통) ===")
    print(f"  FPR={fpr:.3f} ({flagged_clean_ev.sum()}/{n_eval})  95% CI [{fpr_ci[0]:.3f}, {fpr_ci[1]:.3f}]")
    print(f"  recall={recall:.3f} ({flagged_adv_ev.sum()}/{n_eval})  95% CI [{recall_ci[0]:.3f}, {recall_ci[1]:.3f}]")

    # ── 국소 교체용 위치특정: adv eval 이미지 기준 top-1 (실배포와 동일) ────
    flag_idx_ev = localize_top1(model16, eval_images_adv := adv[ev], detect_layer=args.detect_layer)

    # ── 두 방어를 eval 전체에 무조건 적용 (flag 여부와 무관 — 메커니즘 자체 능력 측정) ──
    pred_all_ev, pred_local_ev = [], []
    for s in range(0, n_eval, args.chunk):
        e = min(s + args.chunk, n_eval)
        with torch.no_grad():
            pred_all_ev.append(apply_all_switch(model8, eval_images_adv[s:e]).argmax(dim=1))
            pred_local_ev.append(apply_local_switch(
                model16, model8, adapter, eval_images_adv[s:e], flag_idx_ev[s:e]).argmax(dim=1))
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    pred_all_ev = torch.cat(pred_all_ev, dim=0)
    pred_local_ev = torch.cat(pred_local_ev, dim=0)

    pred16_clean_ev = pred16_clean[ev]
    pred16_adv_ev = pred16_adv[ev]

    def recovery_stats(pred_defended):
        attack_succeeded = (pred16_clean_ev == eval_labels) & (pred16_adv_ev != eval_labels)
        n_attacked = int(attack_succeeded.sum().item())
        recovered = attack_succeeded & (pred_defended == eval_labels)
        n_recovered = int(recovered.sum().item())
        rate = n_recovered / max(n_attacked, 1)
        return n_attacked, n_recovered, rate, wilson_ci(n_recovered, max(n_attacked, 1))

    n_att_all, n_rec_all, rate_all, ci_all = recovery_stats(pred_all_ev)
    n_att_loc, n_rec_loc, rate_loc, ci_loc = recovery_stats(pred_local_ev)
    assert n_att_all == n_att_loc, "두 방어가 같은 eval 표본을 써야 함 — 공격 성공 집합이 달라짐"
    print(f"\n=== (a) 무조건 복원율 (flag 여부 무관, 공격 성공 {n_att_all}/{n_eval} 기준) ===")
    print(f"  all_switch  : {n_rec_all}/{n_att_all} = {rate_all:.3f}  95% CI [{ci_all[0]:.3f}, {ci_all[1]:.3f}]")
    print(f"  local_switch: {n_rec_loc}/{n_att_loc} = {rate_loc:.3f}  95% CI [{ci_loc[0]:.3f}, {ci_loc[1]:.3f}]")

    # ── (b) 시스템 정확도: flag된 것만 방어, 나머지는 P16 그대로 (실배포 조건) ──
    flagged_t = torch.tensor(flagged_adv_ev, device=device)
    sys_pred_all = torch.where(flagged_t, pred_all_ev, pred16_adv_ev)
    sys_pred_local = torch.where(flagged_t, pred_local_ev, pred16_adv_ev)
    p16_only_acc = (pred16_adv_ev == eval_labels).float().mean().item()
    sys_acc_all = (sys_pred_all == eval_labels).float().mean().item()
    sys_acc_local = (sys_pred_local == eval_labels).float().mean().item()
    print(f"\n=== (b) 시스템 정확도 (evaluation {n_eval}개 전체, 공격 이미지 기준) ===")
    print(f"  P16 단독(무방어): {p16_only_acc:.3f}")
    print(f"  all_switch 시스템  : {sys_acc_all:.3f}")
    print(f"  local_switch 시스템: {sys_acc_local:.3f}")

    # ── (c) clean 오탐 비용: clean eval 이미지에 무조건 방어 적용 ───────────
    pred_all_clean_ev, pred_local_clean_ev = [], []
    clean_flag_idx_ev = localize_top1(model16, eval_images, detect_layer=args.detect_layer)
    for s in range(0, n_eval, args.chunk):
        e = min(s + args.chunk, n_eval)
        with torch.no_grad():
            pred_all_clean_ev.append(apply_all_switch(model8, eval_images[s:e]).argmax(dim=1))
            pred_local_clean_ev.append(apply_local_switch(
                model16, model8, adapter, eval_images[s:e], clean_flag_idx_ev[s:e]).argmax(dim=1))
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    pred_all_clean_ev = torch.cat(pred_all_clean_ev, dim=0)
    pred_local_clean_ev = torch.cat(pred_local_clean_ev, dim=0)
    clean_p16_acc = (pred16_clean_ev == eval_labels).float().mean().item()
    clean_all_acc = (pred_all_clean_ev == eval_labels).float().mean().item()
    clean_local_acc = (pred_local_clean_ev == eval_labels).float().mean().item()
    print(f"\n=== (c) clean 오탐 비용 (무조건 방어 적용 시 정확도 변화) ===")
    print(f"  P16 clean acc          : {clean_p16_acc:.3f}")
    print(f"  all_switch 적용 시     : {clean_all_acc:.3f}  (delta {clean_all_acc - clean_p16_acc:+.3f})")
    print(f"  local_switch 적용 시   : {clean_local_acc:.3f}  (delta {clean_local_acc - clean_p16_acc:+.3f})")

    out_dir = HERE
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, f'system_comparison_n{args.num_samples}.npz'),
        n_cal=n_cal, n_eval=n_eval, threshold=thr,
        fpr=fpr, recall=recall, fpr_ci=fpr_ci, recall_ci=recall_ci,
        n_attacked=n_att_all,
        rate_all=rate_all, ci_all=ci_all, n_recovered_all=n_rec_all,
        rate_local=rate_loc, ci_local=ci_loc, n_recovered_local=n_rec_loc,
        p16_only_acc=p16_only_acc, sys_acc_all=sys_acc_all, sys_acc_local=sys_acc_local,
        clean_p16_acc=clean_p16_acc, clean_all_acc=clean_all_acc, clean_local_acc=clean_local_acc,
        sanity_max_diff=max_diff,
    )
    print(f"\nSaved: {os.path.join(out_dir, f'system_comparison_n{args.num_samples}.npz')}")


if __name__ == '__main__':
    main()
