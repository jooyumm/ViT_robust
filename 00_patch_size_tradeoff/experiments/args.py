"""CLI 인자 정의 + 공격별 kwargs 매핑 (main.py에서 분리).

공격별 세부 옵션(--pf_*/--pgd_*/--lavan_*)은 전부 기본값을 `None`으로 둔다 — CLI에서
지정하지 않으면 build_attack_kwargs()가 그 키를 아예 안 넣고, 그러면 src/attacks/*.py의
실제 함수 시그니처에 있는 기본값이 그대로 적용된다. 예전엔 여기 있는 기본값(예:
--pf_iters의 250)이 patch_fool_attack()의 train_attack_iters=250과 완전히 같은 숫자를
따로 하드코딩하고 있었다 — 함수 쪽 기본값이 바뀌면 여기도 같이 안 바꾸는 한 조용히
어긋날 수 있는 진짜 중복이었다(실제로 지금까지 scripts/*.sh 어디도 이 플래그들을 직접
쓴 적이 없어서, 이 중복은 한 번도 실제로 값을 오버라이드하는 데 쓰인 적이 없었다). 이
파일이 "진짜 기본값"을 갖지 않고 공격 함수 쪽에만 두면, 두 곳에 같은 숫자가 존재하지
않으니 어긋날 일도 없다.
"""
import argparse
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args():
    parser = argparse.ArgumentParser(description='ViT Adversarial Patch Experiment')

    parser.add_argument('--attacks', nargs='+', default=['patch_fool'],
                        choices=['patch_fool', 'pgd', 'lavan'])
    parser.add_argument('--patch_sizes', nargs='+', type=int, default=[16, 32])
    parser.add_argument('--num_samples', type=int, default=1024)
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--seed',        type=int, default=42)

    # patch_fool — 기본값 None: 지정 안 하면 src/attacks/patch_fool.py의 실제 기본값을 씀
    parser.add_argument('--pf_attack_mode',  default=None, choices=['Attention', 'CE_loss'])
    parser.add_argument('--pf_iters',        type=int,   default=None)
    parser.add_argument('--pf_num_patch',    type=int,   default=None)
    parser.add_argument('--pf_patch_select', default=None, choices=['Attn', 'Rand', 'Contiguous'])
    parser.add_argument('--pf_mild_l_inf',   type=float, default=None)

    # pgd — 기본값 None: 지정 안 하면 00_patch_size_tradeoff/pgd.py의 실제 기본값을 씀
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

    parser.add_argument('--log_dir',   default=os.path.join(ROOT, 'results', 'logs'))
    parser.add_argument('--tag', default='',
                        help='로그 파일명에 붙는 태그. 같은 (attack, P, seed)라도 설정이 다른 '
                             '실험(예: 면적 통제 ablation)을 baseline과 구분해서 따로 집계하고 싶을 때 사용')

    return parser.parse_args()


def build_attack_kwargs(args, attack_name):
    """args의 값 중 None이 아닌 것만 kwargs에 넣는다 — None으로 남은(CLI에서 지정 안 한)
    항목은 아예 안 넣어서, 호출되는 공격 함수 자신의 기본값이 적용되게 한다."""
    if attack_name == 'patch_fool':
        raw = {
            'attack_mode':        args.pf_attack_mode,
            'train_attack_iters': args.pf_iters,
            'num_patch':          args.pf_num_patch,
            'patch_select':       args.pf_patch_select,
            'mild_l_inf':         args.pf_mild_l_inf,
        }
    elif attack_name == 'pgd':
        raw = {'epsilon': args.pgd_epsilon, 'alpha': args.pgd_alpha, 'steps': args.pgd_steps}
    elif attack_name == 'lavan':
        raw = {'patch_ratio': args.lavan_patch_ratio, 'steps': args.lavan_steps,
               'alpha': args.lavan_alpha, 'fixed_loc': args.lavan_loc}
    else:
        raw = {}
    return {k: v for k, v in raw.items() if v is not None}