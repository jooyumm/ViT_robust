"""공격 성공(오분류)/실패(여전히 정답) 이미지를 하나씩 뽑아 attention을 비교한다 —
"attention이 튀는 게 실제 오분류와 같이 가는가"를 직접 본다. 00과 같은 seed로 공격을
다시 만들고(attack_gen.py 공유), 뽑힌 두 장에 대해서만 attention/GT를 계산한다.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_ROOT = HERE
while not os.path.isdir(os.path.join(DETECT_ROOT, 'experiments')):
    DETECT_ROOT = os.path.dirname(DETECT_ROOT)
SHARED_SRC_ROOT = os.path.dirname(DETECT_ROOT)
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, DETECT_ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(DETECT_ROOT, 'ground_truth'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from attention_lib import collect_full_layer_attn, full_row
from attack_gen import chunked_patch_fool
from coverage import gt_coverage, gt_patch_summary


def predict(model, images, chunk=20):
    preds = []
    for s in range(0, images.shape[0], chunk):
        e = min(s + chunk, images.shape[0])
        with torch.no_grad():
            out = model(images[s:e])
        preds.append(out.argmax(dim=-1).cpu())
    return torch.cat(preds, dim=0)


def attn_row_at_layer(model, image, layer):
    """이미지 한 장, 레이어 하나의 CLS row(197차원)."""
    layer_weights = collect_full_layer_attn(model, image.unsqueeze(0))
    attn_L_avg = layer_weights[layer - 1].mean(dim=1)
    return full_row(attn_L_avg)[0].numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk', type=int, default=20)
    parser.add_argument('--attn_layer_idx', type=int, default=4)
    parser.add_argument('--layer', type=int, default=12)
    parser.add_argument('--success_rank', type=int, default=0, help='nth success candidate')
    parser.add_argument('--fail_rank', type=int, default=0, help='nth failure candidate')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    model = load_vit_model(16, device)
    model.eval()

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples,
                                seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    print(f"[PatchFool] generating attack (n={args.num_samples})...")
    adv = chunked_patch_fool(model, images, labels, device, args.attn_layer_idx, args.chunk)

    clean_pred = predict(model, images, args.chunk)
    adv_pred = predict(model, adv, args.chunk)
    labels_cpu = labels.cpu()

    clean_correct = (clean_pred == labels_cpu)
    adv_correct = (adv_pred == labels_cpu)
    success_idx = torch.where(clean_correct & ~adv_correct)[0]   # correct -> misclassified
    fail_idx = torch.where(clean_correct & adv_correct)[0]       # correct -> still correct

    print(f"clean accuracy: {clean_correct.float().mean():.1%}")
    print(f"attack-success candidates: {len(success_idx)}, attack-failure candidates: {len(fail_idx)}")

    assert len(success_idx) > args.success_rank, "not enough attack-success candidates"
    assert len(fail_idx) > args.fail_rank, "not enough attack-failure candidates"
    i_success = int(success_idx[args.success_rank])
    i_fail = int(fail_idx[args.fail_rank])
    print(f"selected: success = image {i_success}, failure = image {i_fail}")

    out_dir = HERE.replace('/experiments/', '/results/', 1)
    os.makedirs(out_dir, exist_ok=True)

    cases = [('Attack succeeded (misclassified)', i_success),
             ('Attack failed (still correct)', i_fail)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, (title, idx) in zip(axes, cases):
        clean_row = attn_row_at_layer(model, images[idx], args.layer)
        adv_row = attn_row_at_layer(model, adv[idx], args.layer)
        cov = gt_coverage(images[idx].detach().cpu(), adv[idx].detach().cpu())
        gt_patches = gt_patch_summary(cov)
        gt_top = gt_patches[0][0] if gt_patches else None

        x = np.arange(len(clean_row))
        ax.plot(x, clean_row, color='tab:blue', label='Clean', lw=1.3)
        ax.plot(x, adv_row, color='tab:red', label='PatchFool', lw=1.3)
        if gt_top is not None:
            ax.axvline(gt_top + 1, color='gray', ls=':', lw=1, label=f'GT patch {gt_top}')
        ax.set_xlim(0, len(clean_row) - 1)
        ax.set_xlabel('token index (0=CLS, 1-196=patch)')
        ax.set_ylabel('attention')
        ax.set_title(f'{title} — image {idx}\n'
                     f'true={int(labels_cpu[idx])} clean_pred={int(clean_pred[idx])} '
                     f'adv_pred={int(adv_pred[idx])}', fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(f'Layer {args.layer} — attack success vs failure, attention comparison')
    plt.tight_layout()
    p = os.path.join(out_dir, f'03_outcome_compare_L{args.layer}.png')
    plt.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    # 어떤 이미지를 골랐는지는 seed/파라미터가 바뀌면 달라질 수 있어서, 실행할 때마다
    # 여기 다시 저장한다(덮어씀) -- 로그를 뒤지지 않아도 항상 최신 선택이 파일로 남는다.
    selection = {
        'seed': args.seed, 'num_samples': args.num_samples, 'layer': args.layer,
        'clean_accuracy': clean_correct.float().mean().item(),
        'success': {'image': i_success, 'true': int(labels_cpu[i_success]),
                    'clean_pred': int(clean_pred[i_success]), 'adv_pred': int(adv_pred[i_success])},
        'failure': {'image': i_fail, 'true': int(labels_cpu[i_fail]),
                    'clean_pred': int(clean_pred[i_fail]), 'adv_pred': int(adv_pred[i_fail])},
    }
    sel_path = os.path.join(out_dir, '03_outcome_compare_selection.json')
    with open(sel_path, 'w') as f:
        json.dump(selection, f, indent=2)
    print(f"Saved: {sel_path}")


if __name__ == '__main__':
    main()
