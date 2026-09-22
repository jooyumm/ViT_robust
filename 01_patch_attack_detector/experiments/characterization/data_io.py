"""
experiments/characterization/data_io.py — 00_extract_attention.py가 저장한 npz를 다시
all_metrics/repr_by_group 딕셔너리로 복원한다. 01_concentration.py/02_sink_position.py
둘 다 이 함수 하나로 데이터를 불러온다.
"""
import numpy as np

GROUPS = ('clean', 'patchfool', 'lavan')

METRIC_KEYS_AVG = ['top1_row_avg', 'ent_row_avg', 'gini_row_avg',
                   'top1_col_avg', 'ent_col_avg', 'gini_col_avg',
                   'argmax_row_avg', 'argmax_col_avg']
METRIC_KEYS_PH = ['top1_row_ph', 'ent_row_ph', 'gini_row_ph',
                  'top1_col_ph', 'ent_col_ph', 'gini_col_ph']


def load_attention_data(npz_path):
    """반환: (all_metrics, repr_by_group, n_layers)
    all_metrics[group][metric_key] -> (n_samples, n_layers) 또는 (n_samples, n_layers, heads)
    repr_by_group[group]['row_avg'|'col_avg'|'row_ph'|'col_ph'|'row_full_avg'|'full_avg'|
    'images'|'gt_coverage'] -> 대표 이미지 분포 배열. row_full_avg는 197차원(CLS 포함,
    정규화 안 됨), full_avg는 (n_repr, n_layers, 197, 197) 전체 헤드평균 attention 행렬
    그대로(row_full_avg는 이 행렬의 0번째 행과 같음) — 나머지는 196차원(patch만, 정규화됨).
    images는 (n_repr, 3, 224, 224) 원본 픽셀 텐서, gt_coverage는 (n_repr, 196) 공격이 실제로
    건드린 patch 비율(clean에는 없음 — 05_gt_check.py 참고). 이 필드들은 전부 구버전 npz
    (00_extract_attention.py가 해당 필드를 추가하기 전에 생성됨)에는 없을 수 있어 있을
    때만 채운다.
    """
    data = np.load(npz_path)
    n_layers = int(data['n_layers'])

    all_metrics = {}
    repr_by_group = {}
    for g in GROUPS:
        all_metrics[g] = {}
        for k in METRIC_KEYS_AVG + METRIC_KEYS_PH:
            all_metrics[g][k] = data[f'{g}__{k}']
        repr_by_group[g] = dict(
            row_avg=data[f'{g}__row_avg_repr'],
            col_avg=data[f'{g}__col_avg_repr'],
            row_ph=data[f'{g}__row_ph_repr'],
            col_ph=data[f'{g}__col_ph_repr'],
        )
        full_key = f'{g}__row_full_avg_repr'
        if full_key in data.files:
            repr_by_group[g]['row_full_avg'] = data[full_key]
        full_mat_key = f'{g}__full_avg_repr'
        if full_mat_key in data.files:
            repr_by_group[g]['full_avg'] = data[full_mat_key]
        images_key = f'{g}__repr_images'
        if images_key in data.files:
            repr_by_group[g]['images'] = data[images_key]
        gt_key = f'{g}__gt_coverage_repr'
        if gt_key in data.files:
            repr_by_group[g]['gt_coverage'] = data[gt_key]
    return all_metrics, repr_by_group, n_layers
