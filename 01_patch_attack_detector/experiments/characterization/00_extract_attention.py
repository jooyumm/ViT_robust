"""Clean/PatchFool/LaVAN의 레이어x헤드x전체토큰 raw attention을 뽑아 npz 하나로 저장.
GPU 필요한 유일한 단계 — 01/03/05가 이 npz만 읽는다.

저장 필드: top1/entropy/gini, row/column(헤드평균/헤드별) 12개 배열, argmax_row/col_avg
(위치), row_full_avg/full_avg(대표 이미지의 CLS row/전체 197x197 행렬), repr_images(원본
픽셀), gt_coverage(공격이 실제로 건드린 patch, ground_truth/gt_lib.py 참고).

대표 이미지는 --repr_start부터 n_repr장(기본 0부터 5장) — 다른 예시를 보려면 이 값을
바꿔서 재실행(같은 --chunk 안이어야 함).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_ROOT = HERE
while not os.path.isdir(os.path.join(DETECT_ROOT, 'experiments')):
    DETECT_ROOT = os.path.dirname(DETECT_ROOT)
SHARED_SRC_ROOT = os.path.dirname(DETECT_ROOT)  # ViT_robust -- shared src/ (models/dataset/attacks)
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, DETECT_ROOT)
sys.path.insert(0, HERE)   # for `import attention_lib`
sys.path.insert(0, os.path.join(DETECT_ROOT, 'ground_truth'))   # for `import gt_lib`

import numpy as np
import torch

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from attention_lib import (
    collect_full_layer_attn, row_distribution, col_distribution, full_row,
    top1_mass, normalized_entropy, gini_coefficient,
)
from gt_lib import gt_coverage
from attack_gen import chunked_patch_fool, chunked_lavan

GROUPS = ('clean', 'patchfool', 'lavan')


def process_group(model, images, chunk, n_layers=12, n_repr=2, repr_start=0):
    """레이어별 attention 지표 계산 + 대표 이미지(repr_start부터 n_repr장, 같은 청크 안)의
    원본 데이터를 뽑는다."""
    B = images.shape[0]
    keys_avg = ['top1_row_avg', 'ent_row_avg', 'gini_row_avg',
                'top1_col_avg', 'ent_col_avg', 'gini_col_avg',
                'argmax_row_avg', 'argmax_col_avg']
    keys_ph = ['top1_row_ph', 'ent_row_ph', 'gini_row_ph',
               'top1_col_ph', 'ent_col_ph', 'gini_col_ph']
    acc_avg = {k: [] for k in keys_avg}
    acc_ph = {k: [] for k in keys_ph}
    repr_row_avg, repr_col_avg = [], []
    repr_row_ph, repr_col_ph = [], []
    repr_row_full_avg = []
    repr_full_avg = []

    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        layer_weights = collect_full_layer_attn(model, images[s:e])   # 12 x (b,heads,197,197)

        chunk_avg = {k: [] for k in keys_avg}
        chunk_ph = {k: [] for k in keys_ph}
        chunk_row_avg_layers, chunk_col_avg_layers = [], []
        chunk_row_ph_layers, chunk_col_ph_layers = [], []
        chunk_row_full_avg_layers = []
        chunk_full_avg_layers = []

        for L in range(n_layers):
            attn_L = layer_weights[L]                # (b, heads, 197, 197)
            attn_L_avg = attn_L.mean(dim=1)           # (b, 197, 197) — 헤드 평균

            row_ph = row_distribution(attn_L)         # (b, heads, 196)
            col_ph = col_distribution(attn_L)
            row_avg = row_distribution(attn_L_avg)    # (b, 196)
            col_avg = col_distribution(attn_L_avg)

            chunk_avg['top1_row_avg'].append(top1_mass(row_avg))
            chunk_avg['ent_row_avg'].append(normalized_entropy(row_avg))
            chunk_avg['gini_row_avg'].append(gini_coefficient(row_avg))
            chunk_avg['top1_col_avg'].append(top1_mass(col_avg))
            chunk_avg['ent_col_avg'].append(normalized_entropy(col_avg))
            chunk_avg['gini_col_avg'].append(gini_coefficient(col_avg))
            chunk_avg['argmax_row_avg'].append(row_avg.argmax(dim=-1))
            chunk_avg['argmax_col_avg'].append(col_avg.argmax(dim=-1))

            chunk_ph['top1_row_ph'].append(top1_mass(row_ph))
            chunk_ph['ent_row_ph'].append(normalized_entropy(row_ph))
            chunk_ph['gini_row_ph'].append(gini_coefficient(row_ph))
            chunk_ph['top1_col_ph'].append(top1_mass(col_ph))
            chunk_ph['ent_col_ph'].append(normalized_entropy(col_ph))
            chunk_ph['gini_col_ph'].append(gini_coefficient(col_ph))

            chunk_row_avg_layers.append(row_avg)
            chunk_col_avg_layers.append(col_avg)
            chunk_row_ph_layers.append(row_ph)
            chunk_col_ph_layers.append(col_ph)
            chunk_row_full_avg_layers.append(full_row(attn_L_avg))   # (b, 197) -- CLS 포함, 03_token_bars.py용
            chunk_full_avg_layers.append(attn_L_avg)                 # (b, 197, 197) -- 04_column_check.py용

        for k in keys_avg:
            acc_avg[k].append(torch.stack(chunk_avg[k], dim=1).numpy())          # (b, n_layers)
        for k in keys_ph:
            acc_ph[k].append(torch.stack(chunk_ph[k], dim=1).numpy())            # (b, n_layers, heads)

        if s <= repr_start < e:
            local_start = repr_start - s
            local_end = min(local_start + n_repr, e - s)
            repr_row_avg = torch.stack(chunk_row_avg_layers, dim=1)[local_start:local_end].numpy()
            repr_col_avg = torch.stack(chunk_col_avg_layers, dim=1)[local_start:local_end].numpy()
            repr_row_ph = torch.stack(chunk_row_ph_layers, dim=1)[local_start:local_end].numpy()
            repr_col_ph = torch.stack(chunk_col_ph_layers, dim=1)[local_start:local_end].numpy()
            repr_row_full_avg = torch.stack(chunk_row_full_avg_layers, dim=1)[local_start:local_end].numpy()
            repr_full_avg = torch.stack(chunk_full_avg_layers, dim=1)[local_start:local_end].numpy()

        del layer_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = {}
    metrics.update({k: np.concatenate(v, axis=0) for k, v in acc_avg.items()})
    metrics.update({k: np.concatenate(v, axis=0) for k, v in acc_ph.items()})
    return (metrics, repr_row_avg, repr_col_avg, repr_row_ph, repr_col_ph,
            repr_row_full_avg, repr_full_avg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk', type=int, default=20)
    parser.add_argument('--attn_layer_idx', type=int, default=4, help='PatchFool 타겟 레이어')
    parser.add_argument('--lavan_patch_ratio', type=float, default=0.02)
    parser.add_argument('--lavan_steps', type=int, default=40)
    parser.add_argument('--n_repr', type=int, default=5, help='대표 이미지 수')
    parser.add_argument('--repr_start', type=int, default=0,
                        help='대표 이미지 시작 인덱스 (--chunk 안에 들어와야 함)')
    args = parser.parse_args()
    assert (args.repr_start // args.chunk) == ((args.repr_start + args.n_repr - 1) // args.chunk), \
        f"--repr_start {args.repr_start} ~ +{args.n_repr}가 --chunk {args.chunk} 경계를 " \
        "넘어감 -- 대표 이미지 구간은 한 청크 안에 있어야 함"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    model16 = load_vit_model(16, device)
    model16.eval()

    loader, _ = get_dataloader(batch_size=args.num_samples, num_samples=args.num_samples,
                                seed=args.seed)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    print(f"\n[PatchFool] 공격 생성 중 (n={args.num_samples}, attn_layer_idx={args.attn_layer_idx})...")
    adv_pf = chunked_patch_fool(model16, images, labels, device, args.attn_layer_idx, args.chunk)

    print(f"[LaVAN] 공격 생성 중 (n={args.num_samples}, patch_ratio={args.lavan_patch_ratio}, "
          f"steps={args.lavan_steps})...")
    adv_lavan = chunked_lavan(model16, images, labels, device, args.lavan_patch_ratio,
                               args.lavan_steps, args.chunk)

    images_by_group = {'clean': images, 'patchfool': adv_pf, 'lavan': adv_lavan}
    n_layers = 12

    out_dir = HERE.replace('/experiments/', '/results/', 1)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f'00_attention_data_n{args.num_samples}.npz')

    npz_payload = {'n_samples': args.num_samples, 'n_layers': n_layers}
    for name in GROUPS:
        print(f"\n[{name}] 전체 레이어x헤드 attention 수집 및 지표 계산 중...")
        metrics, row_avg, col_avg, row_ph, col_ph, row_full_avg, full_avg = process_group(
            model16, images_by_group[name], args.chunk, n_layers=n_layers, n_repr=args.n_repr,
            repr_start=args.repr_start)
        for k, v in metrics.items():
            npz_payload[f'{name}__{k}'] = v
        npz_payload[f'{name}__row_avg_repr'] = row_avg
        npz_payload[f'{name}__col_avg_repr'] = col_avg
        npz_payload[f'{name}__row_ph_repr'] = row_ph
        npz_payload[f'{name}__col_ph_repr'] = col_ph
        npz_payload[f'{name}__row_full_avg_repr'] = row_full_avg
        npz_payload[f'{name}__full_avg_repr'] = full_avg

        repr_end = min(args.repr_start + args.n_repr, images_by_group[name].shape[0])
        npz_payload[f'{name}__repr_images'] = \
            images_by_group[name][args.repr_start:repr_end].detach().cpu().numpy()

        if name != 'clean':
            # 공격이 실제로 어느 patch를 얼마나 건드렸는지 clean과 diff해서 GT를 만든다
            # (추측/눈대중 아님 -- gt_lib.gt_coverage 참고)
            gt = np.stack([
                gt_coverage(images[i].detach().cpu(), images_by_group[name][i].detach().cpu())
                for i in range(args.repr_start, repr_end)
            ])
            npz_payload[f'{name}__gt_coverage_repr'] = gt

        med = np.median(metrics['top1_row_avg'], axis=0)
        print(f"  top1_row_avg median by layer: " + ", ".join(f"{v:.3f}" for v in med))

    np.savez(npz_path, **npz_payload)
    print(f"\nSaved: {npz_path}")
    print("Next: run 01_concentration.py / 02_sink_position.py on this npz "
          "(no GPU needed, they only read this file).")


if __name__ == '__main__':
    main()
