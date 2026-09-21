"""
probes/failure_case.py — [탐색적 검증, 롤백 가능] localization 실패 샘플(cheby>1) 진단.

localization.py(n=30, P=16, attn_layer_idx=4)에서 top-1 attention이 실제 공격 토큰과
어긋난 샘플이 1개 있었다. 이 샘플에 대해:
  1) PatchFool 공격 자체가 성공했는지(clean pred -> adv pred가 바뀌었는지, 참 라벨과의 관계)
  2) 공격이 성공했는데도 attention이 딴 데를 가리켰다면, 그 이미지의 특징(객체 크기 프록시로
     원본/공격 이미지, 실제 공격 위치와 attention이 가리킨 위치를 bbox로 표시해서 저장)
을 확인한다. 재현성을 위해 torch 전역 시드도 고정한다(이전 probe들은 dataloader 시드만 고정
했었음 — 이번엔 공격 최적화 초기화까지 고정해서 동일한 실패 샘플이 재현되도록 함).

주의: src/ 코드는 import만 하고 수정하지 않음. 이 파일도 지우면 그만.

사용법:
  python probes/failure_case.py --patch_size 16 --num_samples 30 --seed 42
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
import matplotlib.patches as mpatches

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from src.attacks.patch_fool import patch_fool_attack, _collect_attn, _select_patch_attn

try:
    from timm.data import ImageNetInfo
    _INFO = ImageNetInfo()
    def class_name(idx):
        return _INFO.index_to_description(idx).split(',')[0]
except Exception:
    def class_name(idx):
        return str(idx)

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]


def denorm(t):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


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


def draw_patch_box(ax, idx, ppl, patch_px, color, label):
    r, c = idx // ppl, idx % ppl
    ax.add_patch(mpatches.Rectangle((c * patch_px, r * patch_px), patch_px, patch_px,
                                    fill=False, edgecolor=color, linewidth=3))
    ax.text(c * patch_px, r * patch_px - 4, label, color=color, fontsize=11, fontweight='bold')


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
    patch_px = args.patch_size

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples, seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        pred_clean = model(images).argmax(dim=1)

    attn_weights, hooks = _collect_attn(model)
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()
    true_idx = _select_patch_attn(attn_weights, args.attn_layer_idx, 1, device)[:, 0]

    print("[PatchFool] 공격 생성 중...")
    adv_pf, _ = patch_fool_attack(
        model, images, labels, device, patch_size_model=args.patch_size,
        attack_mode='CE_loss', train_attack_iters=250, num_patch=1, patch_select='Attn',
        attn_layer_idx=args.attn_layer_idx)

    with torch.no_grad():
        pred_adv = model(adv_pf).argmax(dim=1)

    lw_pf = collect_layer_attn(model, adv_pf)
    v_pf = raw_at_layer(lw_pf, args.detect_layer)
    top1 = v_pf.argmax(dim=1)

    true_r, true_c = true_idx // ppl, true_idx % ppl
    top1_r, top1_c = top1 // ppl, top1 % ppl
    cheby = torch.maximum((true_r - top1_r).abs(), (true_c - top1_c).abs()).float()

    fail_mask = cheby > 1
    fail_idx = fail_mask.nonzero(as_tuple=True)[0].cpu().tolist()
    print(f"\n실패 샘플(그리드 거리 > 1칸): {fail_idx} (총 {len(fail_idx)}개 / {args.num_samples})")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    for i in range(args.num_samples):
        atk_success = (pred_adv[i] != pred_clean[i]).item()
        was_correct = (pred_clean[i] == labels[i]).item()
        still_correct = (pred_adv[i] == labels[i]).item()
        tag = "*** 실패(위치 못맞춤) ***" if i in fail_idx else ""
        print(f"  [{i:2d}] cheby={cheby[i].item():5.1f}  true=({true_r[i].item():2d},{true_c[i].item():2d}) "
              f"top1=({top1_r[i].item():2d},{top1_c[i].item():2d})  "
              f"clean_correct={was_correct}  pred_changed={atk_success}  still_correct={still_correct}  "
              f"label={class_name(labels[i].item())}  {tag}")

    # 실패 샘플들 상세 진단 + 이미지 저장
    for i in fail_idx:
        atk_success = (pred_adv[i] != pred_clean[i]).item()
        was_correct = (pred_clean[i] == labels[i]).item()
        print(f"\n=== 샘플 {i} 상세 ===")
        print(f"  참 라벨        : {labels[i].item()} ({class_name(labels[i].item())})")
        print(f"  clean 예측      : {pred_clean[i].item()} ({class_name(pred_clean[i].item())}) "
              f"{'[원래 정답 맞춤]' if was_correct else '[원래부터 틀림]'}")
        print(f"  adv(PatchFool) 예측: {pred_adv[i].item()} ({class_name(pred_adv[i].item())})")
        print(f"  공격으로 예측이 바뀌었는가: {atk_success}")
        print(f"  실제 공격 토큰 위치: ({true_r[i].item()},{true_c[i].item()})  "
              f"attention이 가리킨 위치: ({top1_r[i].item()},{top1_c[i].item()})  "
              f"거리(칸): {cheby[i].item():.1f}")

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        for ax, img_t, title in [(axes[0], images[i], 'Clean'), (axes[1], adv_pf[i], 'PatchFool adv')]:
            ax.imshow(denorm(img_t))
            draw_patch_box(ax, true_idx[i].item(), ppl, patch_px, '#ff1744', 'true attack')
            draw_patch_box(ax, top1[i].item(), ppl, patch_px, '#00e5ff', 'attn top-1')
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f"Sample {i} | label={class_name(labels[i].item())} | "
                    f"clean->adv: {class_name(pred_clean[i].item())} -> {class_name(pred_adv[i].item())} | "
                    f"pred_changed={atk_success}")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'03_failure_case_sample{i}.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  저장: {out_path}")

    if not fail_idx:
        print("\n실패 샘플이 이번 실행에서는 재현되지 않았습니다 "
              "(공격 최적화의 난수 초기화 차이로 결과가 미세하게 달라질 수 있음).")


if __name__ == '__main__':
    main()
