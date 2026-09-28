"""배포용 탐지기(v1, detector/topk_mass_v1.py)와 v3(detector/lsw_v3.py)를 같은 이미지·같은 프로토콜로 비교한다.

입력: characterization/01_extract_paired_attention.py가 만든 run(같은 clean/공격 쌍, 예측, GT). 02의 검증
함수(load_paired_run)를 그대로 써서 해시·표본 순서·예측·GT를 확인한 뒤 쓴다.

비교 대상(전부 이 저장소에서 다시 구현, 학습 없음)
- v1 top-4 mass @12  : detector/topk_mass_v1.py (02_patch_switch_defense가 쓰는 배포본)
- v1 top-1 mass @12  : 같은 행의 최댓값
- ARMOR-avg          : 모든 query의 열(column) 평균 attention을 레이어·헤드 평균한 뒤 1등/2등 비(Liu et al. 2023,
                       Appendix D의 abnormality ratio). 탐지 점수만 구현했고 마스킹 복원은 하지 않는다.
- ARMOR-scan         : 레이어별 열 평균의 1등/2등 비를 구해 레이어 최댓값(논문의 layer-wise scanning의 점수판).
- Occlusion          : PatchCleanser/Minority Reports 계열의 1라운드 마스킹 불일치 — 64px 정사각 마스크를 32px
                       간격으로 36번 씌워 원래 예측과 달라지는 비율. forward 37번이 필요하다.
- LSW (v3)           : detector/lsw_v3.py — value 이상치 경로는 새 시드에서 이득이 재현되지 않아 채택하지 않았다
                       (README "탐지기 v3" 절 참고).
- Pixel max local TV : 모델 없이 16x16 창의 국소 변화량 최댓값(픽셀 수준 기준선 — 공격이 유효 픽셀 범위를 벗어나거나
                       고주파 잡음이면 이것만으로도 잡힌다)

프로토콜: 이미지 index 앞 절반/뒤 절반으로 나눠 교차 보정한다(한쪽의 clean 이미지로 임계값 = 95번째
백분위수를 정하고 다른 쪽에서 평가, 두 번 하고 합침). AUROC는 임계값이 필요 없어서 전체로 계산하고
표본(이미지) 단위 부트스트랩 95% 신뢰구간을 붙인다. 공격 이미지는 오분류에 성공한 것만 본다.
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1] / 'detector'))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare02 = load_module('compare02', HERE.parents[0] / 'characterization' / '02_compare_attention_success_failure.py')
import topk_mass_v1 as v1   # noqa: E402
import lsw_v3 as lsw        # noqa: E402

BATCH = 25


def stored_rows(attention):
    """저장된 CLS 행(N,12,heads,197) -> (N,12,196): head 평균 후 patch 위에서 재정규화(v1과 같은 정의)."""
    a = attention.mean(2)[:, :, 1:]
    return torch.from_numpy((a / a.sum(-1, keepdims=True)).astype(np.float32))


def _column_hook(store):
    def hook(module, input, output):
        with torch.no_grad():
            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, C // module.num_heads)
            q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attn = ((module.q_norm(q) * module.scale) @ module.k_norm(k).transpose(-2, -1)).softmax(-1)
            store.append(attn.mean((1, 2))[:, 1:])          # 헤드·query 평균 -> patch key (B,196)
    return hook


@torch.no_grad()
def column_means(model, images):
    """(B,3,H,W) -> (B,12,196): 블록별 열 평균 attention(모든 query, 모든 head)."""
    store = []
    hooks = [blk.attn.register_forward_hook(_column_hook(store)) for blk in model.blocks]
    try:
        model(images)
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(store, dim=1)


def ratio_top1_top2(x):
    top = x.topk(2, dim=-1).values
    return top[..., 0] / top[..., 1].clamp_min(1e-12)


@torch.no_grad()
def occlusion_disagreement(model, images, mask=4, stride=2, patch=16):
    """마스크(0 = 정규화 공간의 회색)를 슬라이딩해서 원래 예측과 달라지는 비율. forward 1 + 36번."""
    base = model(images).argmax(-1)
    size = images.shape[-1]
    starts = list(range(0, size // patch - mask + 1, stride))
    if (size // patch - mask) % stride:
        starts.append(size // patch - mask)
    changed = torch.zeros(len(images), device=images.device)
    for r in starts:
        for c in starts:
            masked = images.clone()
            masked[:, :, r * patch:(r + mask) * patch, c * patch:(c + mask) * patch] = 0
            changed += (model(masked).argmax(-1) != base).float()
    return changed / (len(starts) ** 2)


def max_patch_tv(images):
    x = torch.from_numpy(images) if isinstance(images, np.ndarray) else images
    tv = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean(1)[:, :-1, :] + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean(1)[:, :, :-1]
    return F.avg_pool2d(tv.unsqueeze(1), 16, stride=8).flatten(1).max(1).values.numpy()


def batched(fn, images, device):
    out = []
    for s in range(0, len(images), BATCH):
        out.append(fn(images[s:s + BATCH].to(device)).cpu())
    return torch.cat(out).numpy()


def all_scores(model, data, device, attacks):
    """{탐지기: {'clean'|'patchfool'|'lavan': 점수 (N,)}}"""
    groups = ('clean',) + tuple(attacks)
    scores = {}
    rows = {g: stored_rows(data[g + '__attention']) for g in groups}
    scores['v1 top-4 mass @12'] = {g: rows[g][:, 11].topk(4, 1).values.sum(1).numpy() for g in groups}
    scores['v1 top-1 mass @12'] = {g: rows[g][:, 11].max(1).values.numpy() for g in groups}
    scores['LSW (v3)'] = {g: lsw.surge_score(rows[g]).numpy() for g in groups}
    armor_avg, armor_scan, occlusion, pixel = {}, {}, {}, {}
    for g in groups:
        images = torch.from_numpy(data[g + '__images'])
        cols = torch.from_numpy(batched(lambda x: column_means(model, x), images, device))
        armor_avg[g] = ratio_top1_top2(cols.mean(1)).numpy()
        armor_scan[g] = ratio_top1_top2(cols).max(1).values.numpy()
        occlusion[g] = batched(lambda x: occlusion_disagreement(model, x), images, device)
        pixel[g] = max_patch_tv(data[g + '__images'])
        print(f'  {g}: column/occlusion/pixel scores done', flush=True)
    scores['ARMOR-avg (column ratio)'] = armor_avg
    scores['ARMOR-scan (layer max)'] = armor_scan
    scores['Occlusion (36 masks)'] = occlusion
    scores['Pixel max local TV (no model)'] = pixel
    return scores


def sanity_check(model, data, scores, device):
    """저장된 attention에서 계산한 점수가 배포 코드 경로(detection_score)와 같은지 앞 20장으로 확인."""
    images = torch.from_numpy(data['clean__images'][:20]).to(device)
    ref_v1 = v1.detection_score(model, images).cpu().numpy()
    ref_lsw = lsw.detection_score(model, images).cpu().numpy()
    np.testing.assert_allclose(ref_v1, scores['v1 top-4 mass @12']['clean'][:20], atol=2e-4)
    np.testing.assert_allclose(ref_lsw, scores['LSW (v3)']['clean'][:20], atol=2e-4)


def auroc(negative, positive):
    ranks = rankdata(np.concatenate([negative, positive]))
    n1, n2 = len(negative), len(positive)
    return (ranks[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2)


def crossfit_tpr(clean, attacked, success, fpr=.05):
    """앞/뒤 절반 교차 보정: 한쪽 clean의 (1-fpr) 분위수를 임계값으로 다른 쪽을 평가해 합친다."""
    half = len(clean) // 2
    folds = [(np.arange(half), np.arange(half, len(clean))), (np.arange(half, len(clean)), np.arange(half))]
    hits, total, false_alarms, clean_total = 0, 0, 0, 0
    for calib, test in folds:
        threshold = np.quantile(clean[calib], 1 - fpr)
        chosen = test[success[test]]
        hits += (attacked[chosen] > threshold).sum()
        total += len(chosen)
        false_alarms += (clean[test] > threshold).sum()
        clean_total += len(test)
    return hits / total, false_alarms / clean_total


def bootstrap(fn, n, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    values = [fn(rng.integers(0, n, n)) for _ in range(reps)]
    return np.percentile(values, [2.5, 97.5])


def evaluate(scores, data, attacks):
    successes = {a: data[a + '__outcome'] == 1 for a in attacks}
    table = {}
    for name, per_group in scores.items():
        for attack in attacks:
            clean, attacked, success = per_group['clean'], per_group[attack], successes[attack]
            point = auroc(clean, attacked[success])

            def boot(idx, clean=clean, attacked=attacked, success=success):
                keep = success[idx]
                return auroc(clean[idx], attacked[idx][keep]) if keep.any() else np.nan

            low, high = bootstrap(boot, len(clean))
            tpr, fpr = crossfit_tpr(clean, attacked, success)
            table[(name, attack)] = dict(auroc=point, low=low, high=high, tpr=tpr, fpr=fpr,
                                         n_success=int(success.sum()))
    return table


def timing(model, data, device, n=100):
    images = torch.from_numpy(data['clean__images'][:n]).to(device)
    fns = {'v1 top-4 mass @12': lambda x: v1.detection_score(model, x),
           'v1 top-1 mass @12': lambda x: v1.detection_score(model, x),
           'LSW (v3)': lambda x: lsw.detection_score(model, x),
           'ARMOR-avg (column ratio)': lambda x: ratio_top1_top2(column_means(model, x).mean(1)),
           'ARMOR-scan (layer max)': lambda x: ratio_top1_top2(column_means(model, x)).max(1).values,
           'Occlusion (36 masks)': lambda x: occlusion_disagreement(model, x),
           'Pixel max local TV (no model)': lambda x: torch.from_numpy(max_patch_tv(x.cpu()))}
    out = {}
    for name, fn in fns.items():
        fn(images[:BATCH])
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        for s in range(0, n, BATCH):
            fn(images[s:s + BATCH])
        torch.cuda.synchronize() if device.type == 'cuda' else None
        out[name] = (time.perf_counter() - start) / n * 1000
    return out


def draw(table, latency, names, output, title, attacks):
    figure, axes = plt.subplots(1, len(attacks) + 1, figsize=(5.6 * (len(attacks) + 1), 6.6), layout='constrained')
    colors = {'LSW (v3)': '#d03b3b', 'Pixel max local TV (no model)': '#3b6fd0'}
    y = np.arange(len(names))
    for axis, attack in zip(axes[:-1], attacks):
        for i, name in enumerate(names):
            row = table[(name, attack)]
            axis.barh(i, row['auroc'], color=colors.get(name, '#8a8a8a'))
            axis.errorbar(row['auroc'], i, xerr=[[row['auroc'] - row['low']], [row['high'] - row['auroc']]],
                          color='black', capsize=3, lw=1)
            axis.text(1.005, i, f"{row['auroc']:.2f}  TPR {row['tpr']:.0%}", va='center', fontsize=9)
        axis.axvline(.5, color='#555', lw=.8, ls=':')
        axis.set_yticks(y)
        axis.set_yticklabels(names if attack == attacks[0] else [])
        axis.invert_yaxis()
        axis.set_xlim(.2, 1.0)
        axis.set_xlabel('AUROC (clean vs. successful attack), 95% bootstrap CI')
        axis.set_title(f"{attack} (n={table[(names[0], attack)]['n_success']} successful)  |  TPR at 5% FPR")
    for i, name in enumerate(names):
        axes[-1].barh(i, latency[name], color=colors.get(name, '#8a8a8a'))
        axes[-1].text(latency[name], i, f' {latency[name]:.1f}', va='center', fontsize=9)
    axes[-1].set_yticks(y)
    axes[-1].set_yticklabels([])
    axes[-1].invert_yaxis()
    axes[-1].set_xscale('log')
    axes[-1].set_xlabel('ms per image (log scale, batch 25, same GPU)')
    axes[-1].set_title('Cost')
    figure.suptitle(title, fontsize=13)
    figure.savefig(output, dpi=160, bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='결과 PNG 경로')
    args = parser.parse_args()
    manifest, data = compare02.load_paired_run(args.run)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from src.models import load_vit_model
    model = load_vit_model(16, device)
    model.eval()
    print(f'{len(data["label"])} images, device={device}', flush=True)

    attacks = [a for a in ('patchfool', 'lavan') if a in manifest['attacks']]
    scores = all_scores(model, data, device, attacks)
    sanity_check(model, data, scores, device)
    table = evaluate(scores, data, attacks)
    latency = timing(model, data, device)
    names = list(scores)
    for name in names:
        for attack in attacks:
            r = table[(name, attack)]
            print(f'{name:26s} {attack:9s} AUROC {r["auroc"]:.3f} [{r["low"]:.3f},{r["high"]:.3f}]  '
                  f'TPR@5%FPR {r["tpr"]:.3f} (observed FPR {r["fpr"]:.3f})  n={r["n_success"]}')
        print(f'{name:26s} latency {latency[name]:.2f} ms/img')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw(table, latency, names, args.output,
         f'Detector comparison ({len(data["label"])} clean/attack pairs, seed {manifest["config"]["seed"]}, '
         f'lavan ratio {manifest["config"]["lavan_patch_ratio"]}, PatchFool select layer {manifest["config"]["attn_layer_idx"]}); '
         'thresholds cross-fitted on clean halves', attacks)


if __name__ == '__main__':
    main()
