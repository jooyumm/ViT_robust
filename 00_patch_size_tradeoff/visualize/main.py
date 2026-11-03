"""00 fig 단일 진입점. 저장된 최종 수치만 읽으며 공격을 다시 실행하지 않는다."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fig_common.results_io import collect_results, FIG_DIR
from fig_common.plotting import plot_ca, plot_results, plot_condition_comparison

# 실험 7: LaVAN 위치 고정
LAVAN_FIXEDLOC = [
    ('P=8\nRandom loc',  'lavan', 8,  '',        '...', 0.55),
    ('P=8\nFixed loc',   'lavan', 8,  'fixedloc', '...', 0.9),
    ('P=16\nRandom loc', 'lavan', 16, '',        '',    0.7),
    ('P=16\nFixed loc',  'lavan', 16, 'fixedloc', '',    1.0),
]
LAVAN_FIXEDLOC_TITLE = 'LaVAN: Random vs. Fixed (Center) Attack Location'

# 실험 4+5: PatchFool 면적 통제 (scattered vs contiguous)
# 기본(baseline, tag='') P=8/P=16은 01_baseline과 완전히 같은 hatch/alpha(P_HATCH/P_ALPHA)를
# 써서 "이게 그 baseline이다"가 바로 보이게 하고, 비교 대상(ablation, tag 있음)은 더 연한
# alpha + baseline과 다른 hatch를 줘서 "이건 변형이다"가 눈에 띄게 한다.
PATCHFOOL_AREAMATCH = [
    ('P=8\n1 token (64 px²)',        'patch_fool', 8,  '',                  '...',   0.85),
    ('P=8\n4 tokens, scattered',   'patch_fool', 8,  'areamatch16',       '\\\\\\', 0.4),
    ('P=8\n4 tokens, contiguous',  'patch_fool', 8,  'areamatch16contig', 'xxx',   0.4),
    ('P=16\n1 token (256 px²)',      'patch_fool', 16, '',                  '',      0.9),
]
PATCHFOOL_AREAMATCH_TITLE = 'PatchFool: Area match (scattered/contiguous)'

# 실험 6: PatchFool 토큰 선택 방식 (Attn vs Rand)
PATCHFOOL_RANDSEL = [
    ('P=8\nAttn-select',  'patch_fool', 8,  '',        '...',   0.85),
    ('P=8\nRand-select',  'patch_fool', 8,  'randsel', '\\\\\\', 0.4),
    ('P=16\nAttn-select', 'patch_fool', 16, '',        '',      0.9),
    ('P=16\nRand-select', 'patch_fool', 16, 'randsel', '\\\\\\', 0.4),
]
PATCHFOOL_RANDSEL_TITLE = 'PatchFool: Attention-Guided vs. Random Token Selection'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fig', choices=['all','ca','baseline','fixedloc','areamatch','randsel'], default='all')
    parser.add_argument('--output-dir', type=Path, default=Path(FIG_DIR))
    parser.add_argument('--attacks', nargs='+', choices=['pgd','lavan','patch_fool'], default=['lavan','patch_fool'], help='baseline에 표시할 공격')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.fig in ('all','ca','baseline'):
        data=collect_results()
        if args.fig in ('all','ca'):
            plot_ca(data,args.output_dir/'00_ca.png',' (ImageNet-1k val)')
        if args.fig in ('all','baseline'):
            selected={k:v for k,v in data.items() if k in args.attacks}
            name='01_baseline_all.png' if 'pgd' in args.attacks else '01_baseline.png'
            plot_results(selected,args.output_dir/name,' (ImageNet-1k val)')
    for key, conditions, title, name in [
        ('fixedloc',LAVAN_FIXEDLOC,LAVAN_FIXEDLOC_TITLE,'02_lavan_fixedloc.png'),
        ('areamatch',PATCHFOOL_AREAMATCH,PATCHFOOL_AREAMATCH_TITLE,'03_patchfool_areamatch.png'),
        ('randsel',PATCHFOOL_RANDSEL,PATCHFOOL_RANDSEL_TITLE,'05_patchfool_randsel.png'),
    ]:
        if args.fig in ('all',key):
            plot_condition_comparison(conditions,args.output_dir/name,title)


if __name__ == '__main__':
    main()
