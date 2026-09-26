"""탐지기를 아는 공격자(adaptive attack) 평가: 새 탐지기(LSW)와 v1이 각각 자기를 겨냥한 공격에서 얼마나 버티는지.

공격은 PatchFool과 같은 설정(patch 1개 = clean CLS attention 5번 블록 1등 위치, 16x16, 픽셀 제한 없음, Adam
lr 0.22, StepLR(10, 0.95), 250 iteration)이고 손실만 다르다.
  - ce        : 교차 엔트로피만 올린다(attention loss 없음). 탐지기를 모르는 공격의 기준선.
  - adapt_v1  : CE - W * relu(v1 top-4 mass - tau)   (v1 점수가 임계값 tau 아래로 가게)
  - adapt_lsw : CE - W * relu(LSW surge - tau)       (LSW 점수가 임계값 tau 아래로 가게)
tau는 정상 이미지 점수의 95번째 백분위수 x 0.9로, 공격자가 임계값을 안다고 가정한 강한 설정이다.
평가: 생성된 공격 이미지 각각에 대해 (성공 여부, 각 탐지기의 경보 여부). "evasive success" = 성공하면서 어떤
탐지기에도 안 걸린 비율. 탐지기별 임계값은 같은 clean 이미지의 95번째 백분위수(FPR 5%).
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1] / 'detector'))
sys.path.insert(0, str(HERE))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare02 = load_module('compare02', HERE.parents[0] / 'characterization' / '02_compare_attention_success_failure.py')
cmp = load_module('cmp', HERE / 'compare_detectors.py')
import emergent_window_v2 as lsw   # noqa: E402

PATCH = 16
GRID = 14


def rows_with_grad(model, images):
    """미분 가능한 블록별 CLS->patch attention 행 (B,12,196)과 logits."""
    rows = []

    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
        q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = ((module.q_norm(q[:, :, :1]) * module.scale) @ module.k_norm(k).transpose(-2, -1)).softmax(-1)
        rows.append(attn[:, :, 0].mean(1))

    hooks = [blk.attn.register_forward_hook(hook) for blk in model.blocks]
    try:
        logits = model(images)
    finally:
        for h in hooks:
            h.remove()
    out = torch.stack(rows, dim=1)[:, :, 1:]
    return out / out.sum(-1, keepdim=True), logits


def pick_patch(model, images):
    with torch.no_grad():
        rows, _ = rows_with_grad(model, images)
    return rows[:, 4].argmax(-1)          # 5번 블록(0-based 4)에서 CLS가 가장 많이 보는 patch


def patch_mask(index, size, device):
    mask = torch.zeros(len(index), 1, size, size, device=device)
    for i, p in enumerate(index.tolist()):
        r, c = (p // GRID) * PATCH, (p % GRID) * PATCH
        mask[i, :, r:r + PATCH, c:c + PATCH] = 1
    return mask


def attack(model, images, labels, mode, tau, weight=100., iters=250, seed=0, clip_valid=False):
    torch.manual_seed(seed)
    mask = patch_mask(pick_patch(model, images), images.shape[-1], images.device)
    delta = (torch.randn_like(images) * 0.1).requires_grad_(True)
    optimizer = torch.optim.Adam([delta], lr=0.22)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    for _ in range(iters):
        rows, logits = rows_with_grad(model, images + delta * mask)
        loss = F.cross_entropy(logits, labels, reduction='sum')          # 올려야 하는 값
        if mode == 'adapt_v1':
            loss = loss - weight * F.relu(rows[:, 11].topk(4, 1).values.sum(1) - tau).sum()
        elif mode == 'adapt_lsw':
            loss = loss - weight * F.relu(lsw.surge_score(rows) - tau).sum()
        optimizer.zero_grad()
        (-loss).backward()
        optimizer.step()
        scheduler.step()
        if clip_valid:
            delta.data = (images + delta.data).clamp(-1, 1) - images
    return (images + delta * mask).detach()


@torch.no_grad()
def detector_scores(model, images, device):
    """{탐지기: 점수} — compare_detectors와 같은 정의."""
    out = {'v1 top-4 mass @12': [], 'LSW (new)': [], 'ARMOR-scan (layer max)': [], 'Occlusion (36 masks)': []}
    for s in range(0, len(images), cmp.BATCH):
        x = images[s:s + cmp.BATCH].to(device)
        rows = lsw.cls_patch_rows(model, x)
        out['v1 top-4 mass @12'].append(rows[:, 11].topk(4, 1).values.sum(1).cpu())
        out['LSW (new)'].append(lsw.surge_score(rows).cpu())
        out['ARMOR-scan (layer max)'].append(cmp.ratio_top1_top2(cmp.column_means(model, x)).max(1).values.cpu())
        out['Occlusion (36 masks)'].append(cmp.occlusion_disagreement(model, x).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='정상 이미지 점수로 임계값을 정하고, 공격 대상 이미지를 뽑는 run')
    parser.add_argument('--n', type=int, default=100, help='공격할 clean 정답 이미지 수')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--clip_valid', action='store_true', help='패치 픽셀을 유효 이미지 범위 [-1,1]로 제한')
    args = parser.parse_args()
    manifest, data = compare02.load_paired_run(args.run)
    device = torch.device('cuda')
    from src.models import load_vit_model
    model = load_vit_model(16, device)
    model.eval()
    model.requires_grad_(False)

    clean_all = torch.from_numpy(data['clean__images'])
    labels_all = torch.from_numpy(data['label']).long()
    correct = torch.from_numpy(data['clean__prediction'] == data['label'])
    clean_scores = detector_scores(model, clean_all, device)
    thresholds = {k: float(np.quantile(v, .95)) for k, v in clean_scores.items()}
    print('thresholds (95th pct of clean):', {k: round(v, 4) for k, v in thresholds.items()}, flush=True)

    chosen = torch.nonzero(correct).flatten()[:args.n]
    images, labels = clean_all[chosen], labels_all[chosen]
    results = {}
    for mode in ('ce', 'adapt_v1', 'adapt_lsw'):
        tau = {'ce': 0., 'adapt_v1': 0.9 * thresholds['v1 top-4 mass @12'], 'adapt_lsw': 0.9 * thresholds['LSW (new)']}[mode]
        adv = []
        for s in range(0, len(images), 20):
            adv.append(attack(model, images[s:s + 20].to(device), labels[s:s + 20].to(device), mode, tau, clip_valid=args.clip_valid).cpu())
            print(f'{mode}: {min(s + 20, len(images))}/{len(images)}', flush=True)
        adv = torch.cat(adv)
        with torch.no_grad():
            pred = torch.cat([model(adv[s:s + 50].to(device)).argmax(-1).cpu() for s in range(0, len(adv), 50)])
        success = (pred != labels).numpy()
        scores = detector_scores(model, adv, device)
        flagged = {k: scores[k] > thresholds[k] for k in scores}
        results[mode] = dict(success=success, flagged=flagged)
        print(f'{mode}: attack success {success.mean():.2f}; ' +
              '; '.join(f'{k}: detected {flagged[k][success].mean():.2f} of successes, evasive success {(success & ~flagged[k]).mean():.2f}'
                        for k in flagged), flush=True)

    modes = list(results)
    detectors = list(thresholds)
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout='constrained')
    width = .8 / len(detectors)
    for j, detector in enumerate(detectors):
        detected = [results[m]['flagged'][detector][results[m]['success']].mean() if results[m]['success'].any() else np.nan for m in modes]
        evasive = [(results[m]['success'] & ~results[m]['flagged'][detector]).mean() for m in modes]
        axes[0].bar(np.arange(len(modes)) + j * width, detected, width, label=detector)
        axes[1].bar(np.arange(len(modes)) + j * width, evasive, width, label=detector)
    for axis, title in zip(axes, ['Detected among successful attacks (higher = better detector)',
                                  'Evasive success = attack succeeds AND is not flagged (lower = better detector)']):
        axis.set_xticks(np.arange(len(modes)) + width * (len(detectors) - 1) / 2)
        axis.set_xticklabels(['CE only\n(detector-unaware)', 'adaptive vs v1', 'adaptive vs LSW'])
        axis.set_ylim(0, 1)
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=.2, axis='y')
    axes[0].legend(fontsize=8)
    figure.suptitle(f'Adaptive attacks (n={len(images)} clean-correct images, {"valid-range" if args.clip_valid else "unbounded"} 16x16 patch, thresholds known to attacker)')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, bbox_inches='tight')
    plt.close(figure)


if __name__ == '__main__':
    main()
