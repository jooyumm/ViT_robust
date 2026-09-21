"""CLI 인자 정의 + 공격별 kwargs 매핑 (main.py에서 분리)."""
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

    # patch_fool
    parser.add_argument('--pf_attack_mode',  default='Attention',
                        choices=['Attention', 'CE_loss'])
    parser.add_argument('--pf_iters',        type=int,   default=250)
    parser.add_argument('--pf_num_patch',    type=int,   default=1)
    parser.add_argument('--pf_patch_select', default='Attn',
                        choices=['Attn', 'Rand', 'Contiguous'])
    parser.add_argument('--pf_mild_l_inf',   type=float, default=0.0)

    # pgd
    parser.add_argument('--pgd_epsilon', type=float, default=8/255)
    parser.add_argument('--pgd_alpha',   type=float, default=2/255)
    parser.add_argument('--pgd_steps',   type=int,   default=40)

    # lavan
    parser.add_argument('--lavan_patch_ratio', type=float, default=0.02)
    parser.add_argument('--lavan_steps',       type=int,   default=40)
    parser.add_argument('--lavan_alpha',       type=float, default=2/255)
    parser.add_argument('--lavan_loc', default='random', choices=['random', 'center'],
                        help="'center'면 매번 이미지 중앙 고정 위치를 공격 (위치 변동성 제거)")

    parser.add_argument('--log_dir',   default=os.path.join(ROOT, 'results', 'logs'))
    parser.add_argument('--tag', default='',
                        help='로그 파일명에 붙는 태그. 같은 (attack, P, seed)라도 설정이 다른 '
                             '실험(예: 면적 통제 ablation)을 baseline과 구분해서 따로 집계하고 싶을 때 사용')

    return parser.parse_args()


def build_attack_kwargs(args, attack_name):
    if attack_name == 'patch_fool':
        return {
            'attack_mode':        args.pf_attack_mode,
            'train_attack_iters': args.pf_iters,
            'num_patch':          args.pf_num_patch,
            'patch_select':       args.pf_patch_select,
            'mild_l_inf':         args.pf_mild_l_inf,
        }
    elif attack_name == 'pgd':
        return {'epsilon': args.pgd_epsilon, 'alpha': args.pgd_alpha,
                'steps':   args.pgd_steps}
    elif attack_name == 'lavan':
        return {'patch_ratio': args.lavan_patch_ratio,
                'steps':       args.lavan_steps,
                'alpha':       args.lavan_alpha,
                'fixed_loc':   args.lavan_loc if args.lavan_loc != 'random' else None}
    return {}