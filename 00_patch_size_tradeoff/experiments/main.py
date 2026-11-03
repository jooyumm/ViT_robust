"""00 실험 1~5의 단일 진입점. 실험 조건은 아래 주석별로 정의한다.

사용: python 00_patch_size_tradeoff/experiments/main.py --experiment 1 --seed 42
실제 공통 실행·로그·지표는 프로젝트 루트 runner.py, 공유 공격은 src에 둔다.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from runner import run_case

# 01. Baseline: PGD / LaVAN / PatchFool × P8/P16
EXPERIMENTS = {
    1: dict(attacks=['pgd', 'lavan', 'patch_fool'], patch_sizes=[8,16]),

    # 02. LaVAN 중앙 고정: baseline의 랜덤 위치와 비교
    2: dict(attacks=['lavan'], patch_sizes=[8,16], lavan_loc='center', tag='fixedloc'),

    # 03. 동일 면적·분산: P8 attention 상위 4토큰 독립 선택
    3: dict(attacks=['patch_fool'], patch_sizes=[8], pf_num_patch=4,
            pf_patch_select='Attn', tag='areamatch16'),

    # 04. 동일 면적·인접: P8 2×2 토큰, P16 한 토큰과 같은 16×16px
    4: dict(attacks=['patch_fool'], patch_sizes=[8], pf_num_patch=4,
            pf_patch_select='Contiguous', tag='areamatch16contig'),

    # 05. 랜덤 선택: baseline의 attention 위치 선택과 비교
    5: dict(attacks=['patch_fool'], patch_sizes=[8,16], pf_patch_select='Rand', tag='randsel'),
}


def get_args(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--experiment", type=int, choices=range(1, 6), default=1)
    selected, _ = selector.parse_known_args(argv)
    parser = argparse.ArgumentParser(description='ViT Adversarial Patch Experiment')

    parser.add_argument('--experiment', type=int, choices=range(1, 6), default=1, help='실험 번호 1~5')
    parser.add_argument('--attacks', nargs='+', default=['patch_fool'],
                        choices=['patch_fool', 'pgd', 'lavan'])
    parser.add_argument('--patch_sizes', nargs='+', type=int, choices=[8, 16], default=[8, 16])
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--seed',        type=int, default=42)

    # patch_fool — 기본값 None: 지정 안 하면 src/attacks/patch_fool.py의 실제 기본값을 씀
    parser.add_argument('--pf_attack_mode',  default=None, choices=['Attention', 'CE_loss'])
    parser.add_argument('--pf_iters',        type=int,   default=None)
    parser.add_argument('--pf_num_patch',    type=int,   default=None)
    parser.add_argument('--pf_patch_select', default=None, choices=['Attn', 'Rand', 'Saliency', 'Contiguous'])
    parser.add_argument('--pf_mild_l_inf',   type=float, default=None)
    parser.add_argument('--pf_sparse_pixel_num', type=int, default=None)
    parser.add_argument('--pf_clip_valid_range', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--pf_attention_source', choices=['incoming', 'cls'], default=None)
    parser.add_argument('--pf_initialization', choices=['replacement', 'delta'], default=None)

    # pgd — 기본값 None: 지정 안 하면 src/attacks/pgd.py의 실제 기본값을 씀
    parser.add_argument('--pgd_epsilon', type=float, default=None)
    parser.add_argument('--pgd_alpha',   type=float, default=None)
    parser.add_argument('--pgd_steps',   type=int,   default=None)

    # lavan — 기본값 None: 지정 안 하면 src/attacks/lavan.py의 실제 기본값을 씀
    parser.add_argument('--lavan_patch_ratio', type=float, default=None)
    parser.add_argument('--lavan_steps',       type=int,   default=None)
    parser.add_argument('--lavan_alpha',       type=float, default=None)
    parser.add_argument('--lavan_loc', default=None, choices=['center'],
                        help="지정 안 하면(기본) 랜덤 위치. 'center'면 매번 이미지 중앙 "
                             "고정 위치를 공격(위치 변동성 제거) — lavan_attack()의 "
                             "fixed_loc 인자로 그대로 전달됨")

    parser.add_argument('--result-dir', dest='result_dir', default=os.path.join(ROOT, 'results', 'metrics'), help='최종 JSON 저장 폴더 (진행 로그는 저장하지 않음)')
    parser.add_argument('--tag', default='',
                        help='결과 파일명에 붙는 태그. 같은 (attack, P, seed)라도 설정이 다른 '
                             '실험(예: 면적 통제 ablation)을 baseline과 구분해서 따로 집계하고 싶을 때 사용')

    parser.set_defaults(num_samples=1000, batch_size=None, **EXPERIMENTS[selected.experiment])
    return parser.parse_args(argv)



def iter_cases(args):
    """정의된 조건을 실제 (모델, 공격) 조합으로 펼친다."""
    if args.num_samples < 1 or (args.batch_size is not None and args.batch_size < 1):
        raise ValueError('num_samples와 batch_size는 양수여야 합니다.')
    for patch in args.patch_sizes:
        for attack in args.attacks:
            case = deepcopy(args)
            case.patch_sizes, case.attacks = [patch], [attack]
            if case.batch_size is None:
                case.batch_size = (8 if attack == 'patch_fool' else 16) if patch == 8 else 64
            yield case


def main(argv=None):
    for case in iter_cases(get_args(argv)):
        run_case(case)


if __name__ == '__main__':
    main()
