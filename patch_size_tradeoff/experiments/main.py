"""
ViT 적대적 패치 공격 실험 실행기

  python experiments/main.py --attacks pgd lavan patch_fool --patch_sizes 8 16 32
  python experiments/main.py --attacks patch_fool --patch_sizes 8 16 32 --num_samples 5000 --pf_iters 250

CLI 인자 정의는 args.py, 로그 파일 저장은 logfile.py, 배치 단위 공격 실행은
attack_experiment_runner.py에 있다 — 이 파일은 그것들을 순서대로 호출하는 얇은 진입점만 담당한다.
"""

import torch
import numpy as np
import os
import sys
from itertools import product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED_SRC_ROOT = os.path.dirname(ROOT)  # ViT_robust -- shared src/ (models/dataset/attacks)
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, ROOT)

from src.models import get_device, load_vit_model
from src.dataset import get_dataloader
from experiments.args import get_args, build_attack_kwargs
from experiments.logfile import Tee, make_log_path
from experiments.attack_experiment_runner import run_experiment


def print_header(args, device, num_batches):
    print("=" * 65)
    print("  ViT Adversarial Robustness Experiment")
    print("=" * 65)
    print(f"  Device      : {device}")
    print(f"  Seed        : {args.seed}")
    if args.tag:
        print(f"  Tag         : {args.tag}")
    print(f"  Attacks     : {args.attacks}")
    print(f"  Patch sizes : P = {args.patch_sizes}")
    print(f"  Samples     : {num_batches * args.batch_size} "
          f"({args.batch_size} x {num_batches} batches)")
    print(f"  Metrics     : CA / RA / ASR  (see metrics.py)")
    print(f"    A = 맞음→맞음   B = 맞음→틀림   C = 틀림→맞음   D = 틀림→틀림")
    print(f"    CA  = Clean Accuracy       = (A+B) / N")
    print(f"    RA  = Robust Accuracy      = (A+C) / N")
    print(f"    ASR = Attack Success Rate  = B / (A+B)")
    print(f"    Δ   = CA - RA  (accuracy drop due to attack)")
    print()


def print_result_block(overall):
    print(f"\n  ── Overall ──────────────────────────────────────")
    print(f"  CA  (Clean Accuracy)   : {overall['CA']:.2f}%")
    print(f"  RA  (Robust Accuracy)  : {overall['RA']:.2f}%")
    print(f"  ASR (Attack Success)   : {overall['ASR']:.2f}%")
    print(f"  Δ   (CA - RA)          : {overall['acc_drop']:.2f}%p")
    print(f"  Sample breakdown       : "
          f"A={overall['A']} B={overall['B']} "
          f"C={overall['C']} D={overall['D']} "
          f"(valid={overall['valid']})")


def print_summary(results, attacks, patch_sizes):
    print(f"\n\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")

    col_w = 10
    for metric in ['CA', 'RA', 'ASR', 'acc_drop']:
        label = {'CA': 'Clean Acc', 'RA': 'Robust Acc',
                'ASR': 'ASR', 'acc_drop': 'Δ (CA-RA)'}[metric]
        header = f"  {label:<14}" + "".join(
            f"{'P='+str(P):>{col_w}}" for P in patch_sizes)
        print(header)
        print("  " + "─" * (14 + col_w * len(patch_sizes)))

        for attack_name in attacks:
            row = f"  {attack_name:<14}"
            for P in patch_sizes:
                r = results[attack_name].get(P)
                row += f"{'N/A':>{col_w}}" if r is None else f"{r[metric]:>{col_w-1}.2f}%"
            print(row)
        print()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    log_path    = make_log_path(args)
    num_batches = args.num_samples // args.batch_size
    device      = get_device()
    orig_stdout = sys.stdout

    with open(log_path, 'w', encoding='utf-8') as f:
        sys.stdout = Tee(f)
        try:
            print_header(args, device, num_batches)

            loader, _ = get_dataloader(
                batch_size=args.batch_size,
                num_samples=num_batches * args.batch_size,
                seed=args.seed,
            )

            results = {atk: {} for atk in args.attacks}

            for P, attack_name in product(args.patch_sizes, args.attacks):
                print(f"\n{'─'*65}")
                print(f"  Attack: {attack_name:12s} | P={P} | "
                      f"Tokens: {(224 // P)**2}")
                print(f"{'─'*65}")

                model = load_vit_model(P, device)
                if model is None:
                    print(f"  [SKIP] P={P} 모델 로드 실패")
                    results[attack_name][P] = None
                    continue

                overall = run_experiment(
                    model, loader, device,
                    attack_name=attack_name,
                    patch_size_model=P,
                    num_batches=num_batches,
                    attack_kwargs=build_attack_kwargs(args, attack_name),
                )
                results[attack_name][P] = overall
                print_result_block(overall)

            print_summary(results, args.attacks, args.patch_sizes)
        finally:
            sys.stdout = orig_stdout

    print(f"\nLog saved: {log_path}")


if __name__ == '__main__':
    main()