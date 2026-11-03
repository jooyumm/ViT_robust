"""공통 최종 JSON 선택을 사용해 fig의 seed별 지표를 읽는다."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from result_store import read_results

RESULTS_DIR=str(ROOT/'results')
METRICS_DIR=str(ROOT/'results/metrics')
FIG_DIR=str(ROOT/'results/figs')
ATK_LABEL={'pgd':'PGD','lavan':'LaVAN','patch_fool':'PatchFool'}
ATK_ORDER=['pgd','lavan','patch_fool']
P_ORDER=[8,16]


def collect_results(result_dir=None, verbose=True, tag=''):
    data={}
    selected=read_results(result_dir or METRICS_DIR,{tag})
    for (attack,p,_,seed),(_,metrics) in sorted(selected.items()):
        bucket=data.setdefault(attack,{}).setdefault(p,dict(CA=[],RA=[],ASR=[],seeds=[]))
        for key in ('CA','RA','ASR'):
            bucket[key].append(metrics[key])
        bucket['seeds'].append(seed)
    if verbose:
        print(f'최종 JSON {len(selected)}개 사용 (tag={tag!r}, 조건·모델·seed별 최신)')
    return data
