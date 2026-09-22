"""
experiments/characterization/00_extract_attention.py — Clean/PatchFool/LaVAN 세 그룹의
레이어x헤드x전체토큰 raw attention에서 필요한 모든 걸 뽑아 **하나의 npz**로 저장한다. 이
파일이 유일하게 GPU가 필요한/느린 단계다 — 01_concentration.py/02_sink_position.py/
03_token_bars.py 셋 다 이 npz만 읽어서 각자 다른 질문에 답한다(공격을 다시 생성하지 않음).

성격 규정
--------
탐지 임계값/AUROC/flag 없음 — attention_lib.py의 함수로 분포·집중도만 계산해서 저장한다.
detector/topk_mass_v1.py의 CLS-row·헤드평균 전용 함수는 쓰지 않는다(헤드별/column까지
봐야 하므로).

저장하는 것
----------
그룹(clean/patchfool/lavan)마다:
  - top1/entropy/gini, row/column, 헤드평균("_avg")/헤드별("_ph") 조합 12개 배열 — 전부
    (n_samples, 12) 또는 (n_samples, 12, heads)
  - argmax_row_avg/argmax_col_avg: (n_samples, 12) — CLS-row/column 분포의 argmax 패치
    위치(196개 중 인덱스). 02_sink_position.py가 이걸로 위치 집중도/일치율을 계산한다.
  - row_avg/col_avg/row_ph/col_ph의 대표 이미지 n_repr장분(전체 196차원 분포, 히트맵/
    정렬-질량 그림용)
  - row_full_avg의 대표 이미지 n_repr장분(197차원, CLS 포함·정규화 안 됨 — 03_token_bars.py
    가 "토큰 197개 중 어디가 튀는가"를 CLS까지 포함해서 그리는 데 씀)
  - full_avg의 대표 이미지 n_repr장분(197x197 전체 헤드평균 attention 행렬 그대로, 레이어별
    — 04_column_check.py가 "CLS 말고 다른 토큰들도 특정 patch를 보는가"를 확인하려고
    임의의 key 열(column)을 뽑아 쓴다. row_full_avg는 이 행렬의 0번째 행과 같다)
  - repr_images: 대표 이미지 n_repr장분의 실제 (3,224,224) 픽셀 텐서 그대로(clean/patchfool/
    lavan 각각) — "원본 vs 공격 당한 이미지"를 눈으로/코드로 직접 볼 수 있는 GT 재료
  - gt_coverage(patchfool/lavan만): 대표 이미지 n_repr장분, (n_repr, 196) — 공격이 실제로
    "어느 patch를 얼마나 건드렸는지"를 clean/adv 픽셀을 diff해서 계산한 ground truth(추측
    아님, `../../ground_truth/gt_lib.py` 참고 — 탐지기 테스트 등에서도 재사용하려고
    characterization 밖의 프로젝트 top-level로 뺐다). `../../ground_truth/visualize_gt.py`
    가 이 GT를 그림/표로 뽑아준다.

대표 이미지는 항상 --repr_start(기본 0)부터 n_repr장 -- --repr_start를 안 바꾸면 매번
images[0:n_repr]로 고정돼서 "예시 이미지"가 절대 안 바뀐다. 다른 예시 세트를 보고
싶으면 --repr_start를 바꿔서 다시 실행할 것(같은 --chunk 안에 들어와야 함).

사용법:
  python 00_extract_attention.py --num_samples 100 --seed 42 --chunk 20 --repr_start 0
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
from src.attacks.patch_fool import patch_fool_attack
from src.attacks.lavan import lavan_attack
from attention_lib import (
    collect_full_layer_attn, row_distribution, col_distribution, full_row,
    top1_mass, normalized_entropy, gini_coefficient,
)
from gt_lib import gt_coverage

GROUPS = ('clean', 'patchfool', 'lavan')


def chunked_patch_fool(model16, images, labels, device, attn_layer_idx, chunk):
    adv_chunks = []
    for s in range(0, images.shape[0], chunk):
        e = min(s + chunk, images.shape[0])
        # attack_mode='Attention' (patch_fool_attack의 실제 기본값) -- CE loss뿐 아니라
        # attention 손실(PCGrad로 CE와 결합)까지 같이 최적화하는 PatchFool 논문의 원래
        # 공격 방식. 예전엔 여기서 'CE_loss'로 덮어써서 attention 손실 없이 순수 분류
        # 손실만 쓰고 있었다 -- 지금까지 관찰한 attention 왜곡이 patch_select='Attn'
        # (공격 위치를 attention 기준으로 고르는 것)만으로 생긴 건지, attack_mode='Attention'
        # (최적화 자체가 attention을 직접 왜곡하는 것)까지 필요한 건지 구분하려고 기본값으로
        # 되돌렸다.
        adv_chunk, _ = patch_fool_attack(
            model16, images[s:e], labels[s:e], device, patch_size_model=16,
            attack_mode='Attention', train_attack_iters=250, num_patch=1, patch_select='Attn',
            attn_layer_idx=attn_layer_idx)
        adv_chunks.append(adv_chunk)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return torch.cat(adv_chunks, dim=0)


def chunked_lavan(model16, images, labels, device, patch_ratio, steps, chunk):
    adv_chunks = []
    for s in range(0, images.shape[0], chunk):
        e = min(s + chunk, images.shape[0])
        adv_chunk = lavan_attack(
            model16, images[s:e], labels[s:e], device, patch_ratio=patch_ratio, steps=steps)
        adv_chunks.append(adv_chunk)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return torch.cat(adv_chunks, dim=0)


def process_group(model, images, chunk, n_layers=12, n_repr=2, repr_start=0):
    """반환: metrics(dict of ndarray), repr_row_avg, repr_col_avg, repr_row_ph, repr_col_ph,
    repr_row_full_avg, repr_full_avg. docstring 상세는 모듈 docstring의 "저장하는 것" 참고.
    repr_start: 대표 이미지로 뽑을 구간의 시작 인덱스(전역, 0-based). 항상 0(첫 청크의
    맨 앞)만 쓰면 "대표 이미지"가 매번 똑같은 5장(images[0:5])으로 고정돼서 절대 안
    바뀐다 — 다른 예시가 필요하면 이 값을 바꿔서 다시 추출해야 한다. repr_start와
    repr_start+n_repr는 반드시 같은 청크 안에 있어야 한다(청크 경계를 넘어가는 구간은
    지원 안 함)."""
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
    parser.add_argument('--n_repr', type=int, default=5,
                        help='시각화/GT 확인용 대표 이미지 수 (05_gt_check.py가 여러 장에 '
                             '걸쳐 GT-vs-관측 일치율을 보려면 2장으론 부족해서 5로 늘림)')
    parser.add_argument('--repr_start', type=int, default=0,
                        help='대표 이미지로 쓸 구간의 시작 인덱스(0-based, 전체 배치 기준). '
                             '항상 0이면 "대표 이미지"가 매번 images[0:n_repr]로 고정돼서 '
                             '절대 안 바뀐다 -- 다른 예시 세트가 필요하면 이 값을 바꿀 것. '
                             'repr_start + n_repr는 --chunk를 넘으면 안 됨(청크 경계 안 지원).')
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
