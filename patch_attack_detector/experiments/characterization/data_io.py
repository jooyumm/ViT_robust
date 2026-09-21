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
    repr_by_group[group]['row_avg'|'col_avg'|'row_ph'|'col_ph'] -> 대표 이미지 분포 배열
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
    return all_metrics, repr_by_group, n_layers
