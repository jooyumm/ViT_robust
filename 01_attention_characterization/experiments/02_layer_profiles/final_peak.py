"""입력별 final-layer 최대 패치의 레이어 경향을 정상·공격 집단에서 비교한다. 각 조건(clean/
patchfool)이 자기 자신의 마지막 레이어 최대 patch를 독립적으로 고른다 -- 원본(clean) 없이도
그 이미지 하나만으로 계산 가능한 정의라서, paired clean이 있어야만 계산되는 위치 고정(이전
clean_max 방식)은 쓰지 않는다. 레이어 그래프와 예시 그림, 중첩 비교 그림을 저장한다.
JSON·마크다운 결과는 생성하지 않는다. raw attention은 patch 합으로 재정규화하지 않는다.
"""
import argparse
import importlib
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
from src.data.paired import load_paired_dataset
from src.cohort import DEFAULT_DATASET
from src.visualization.colors import GROUP_COLORS
plotting = importlib.import_module('experiments.02_layer_profiles.plotting')
GROUPS = ('clean', 'patchfool')  # PatchFool-only research scope; LaVAN dropped
METHOD = 'final_max'


def extract(patch, indices):
    return np.take_along_axis(patch, indices[:, None, None], axis=-1)[..., 0]


def most_divergent(series, groups):
    """같은 이미지의 clean/patchfool 궤적 모양(정규화된 shape) 차이가 가장 큰 이미지 1개를 고른다."""
    shapes = {g: series[g] / np.maximum(series[g].max(axis=1, keepdims=True), 1e-12) for g in groups}
    divergence = np.abs(shapes[groups[0]] - shapes[groups[1]]).mean(axis=1)
    idx = int(np.argmax(divergence))
    return [(idx, float(divergence[idx]), 1)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--output',type=Path,default=HERE.parents[1]/'results'/HERE.name)
    args=parser.parse_args()
    print('공통 원본·공격 500장 검증 및 읽기',flush=True)
    manifest,data=load_paired_dataset(args.run)
    patches={g:data[g+'__attention'].mean(axis=2).astype(np.float64)[...,1:] for g in GROUPS}
    shape=patches['clean'].shape
    if shape[1]!=12 or any(v.shape!=shape for v in patches.values()):raise ValueError('동일한 12개 레이어 데이터가 필요합니다.')
    indices={g:patches[g][:,-1].argmax(1) for g in GROUPS}
    series={g:extract(patches[g],indices[g]) for g in GROUPS}
    # clean/patchfool 궤적 모양이 가장 많이 벌어지는 이미지 1개만 뽑는다(clean 통계만으로 고르던 것 대체).
    selected=most_divergent(series, GROUPS)
    sample_rows=[s[0] for s in selected]
    # 비교 그림들은 각자 따로 스케일되지 않도록 전체 범위를 미리 구해서 공유한다.
    population_ylim=(0,max(series[g].max() for g in GROUPS)*100*1.05+1e-8)
    ylim_max=max(series[g][i].max()*100 for g in GROUPS for i in sample_rows)
    map_max=max(patches[g][sample_rows][:,[0,5,8,10,11]].max()*100 for g in GROUPS)
    args.output.mkdir(parents=True,exist_ok=True)
    for group in GROUPS:
        plotting.draw_population(series[group],args.output/f'{group}_layer.png',group,GROUP_COLORS[group],METHOD,population_ylim)
    plotting.draw_examples(data,manifest,patches,indices,series,selected,args.output/'examples.png',
        GROUPS,GROUP_COLORS,METHOD,ylim_max,map_max)
    plotting.draw_comparison(series, args.output/'comparison.png', METHOD)
    print('예시 원본 ID:',data['sample_index'][sample_rows].tolist(),flush=True)
    print('개별 그림과 중첩 비교 그림 저장 완료:',args.output,flush=True)

if __name__=='__main__':main()
