"""P8/P16 각각에 직접 PatchFool을 적용하고 P16 region switch와 비교한다.

Clean은 방어 없는 원본 모델의 정확도다. 동일 500장에서 P8 1토큰(8×8px), P8 인접 4토큰(16×16px),
P16 1토큰(16×16px)을 동일 반복 예산으로 비교한다.
두 모델 모두 같은 현재 공격 구현으로 재공격한다. 방어를 목표로 최적화하는
적응형 공격 평가는 아니다. 결과는 PNG, 수치와 설정은 stdout에 기록한다.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '02_detection_defense')]
from src.models import load_vit_model
from src.attacks import patch_fool_attack, patch_fool_config
from src.cohort import DEFAULT_DATASET, validate_default_manifest
from region_switch.core.inference import predict_defended
from region_switch.core.attention import PatchAttentionDetector
from region_switch.core.switch import converted_patch_kernel
from src.provenance import model_hash, sha256
from region_switch.experiments.model_support import CASES, PROTOCOL, source_hashes, reusable_results, plot_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--attack-iters', type=int, default=250)
    parser.add_argument('--reuse-log', type=Path, help='이전 동일 조건 실행의 model_comparison.log')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.suffix.lower() != '.png' or min(args.batch_size, args.attack_iters) < 1:
        raise ValueError('PNG 경로와 양의 batch-size/attack-iters가 필요합니다.')
    manifest = json.loads((DEFAULT_DATASET / 'manifest.json').read_text())
    validate_default_manifest(manifest)
    torch.set_num_threads(4)
    started = time.monotonic()
    sources = source_hashes(ROOT)
    manifest_hash = sha256(DEFAULT_DATASET/'manifest.json')
    protocols = {case: patch_fool_config(patch_size_model=patch, num_patch=count, patch_select=selection,
                                          train_attack_iters=args.attack_iters)
                 for case, patch, count, selection in CASES}
    results = reusable_results(args.reuse_log, attack_iters=args.attack_iters,
                               batch_size=args.batch_size, seed=manifest['config']['seed'],
                               manifest_hash=manifest_hash, sources=sources,
                               n=len(manifest['sample_indices'])) if args.reuse_log else {}
    for case, patch, num_patch, selection in CASES:
        if case in results:
            continue
        model = load_vit_model(patch, args.device, strict=True).requires_grad_(False)
        np.testing.assert_allclose(model.pretrained_cfg['mean'], manifest['mean'])
        np.testing.assert_allclose(model.pretrained_cfg['std'], manifest['std'])
        assert model.pretrained_cfg['input_size'] == (3, 224, 224)
        original_hash = model_hash(model)
        if patch == 16:
            assert original_hash == manifest['model_sha256']
        kernel8 = converted_patch_kernel(model) if patch == 16 else None
        correct = dict(clean=0, attack=0)
        if patch == 16:
            correct['defense'] = 0
        n = 0
        for chunk in manifest['chunks']:
            path = DEFAULT_DATASET / chunk['path']
            assert sha256(path) == chunk['sha256']
            with np.load(path) as z:
                for start in range(0, chunk['n'], args.batch_size):
                    x = torch.from_numpy(z['clean__images'][start:start + args.batch_size]).to(args.device)
                    labels = torch.from_numpy(z['label'][start:start + args.batch_size]).to(args.device)
                    with torch.inference_mode():
                        pred = model(x).argmax(-1)
                    correct['clean'] += int((pred == labels).sum())
                    if patch == 16:
                        np.testing.assert_array_equal(pred.cpu().numpy(), z['clean__prediction'][start:start + len(x)])
                    seed = int(manifest['config']['seed']) + 1009 * (n + start)
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    # P8 1토큰=64px; P8 인접4토큰/P16 1토큰=256px.
                    attacked, weights = patch_fool_attack(model, x, labels, args.device, **protocols[case])
                    del weights
                    changed = (attacked-x).abs().sum(1) > 1e-6
                    assert bool((changed.sum((1, 2)) <= num_patch*patch*patch).all())
                    if patch == 16:
                        base, defended, _ = predict_defended(model, kernel8, attacked, PatchAttentionDetector())
                        correct['attack'] += int((base.argmax(-1) == labels).sum())
                        correct['defense'] += int((defended.argmax(-1) == labels).sum())
                    else:
                        with torch.inference_mode():
                            correct['attack'] += int((model(attacked).argmax(-1) == labels).sum())
                n += chunk['n']
            print(f'{case} direct attack {n}/500, counts={correct}, elapsed={time.monotonic()-started:.1f}s', flush=True)
        assert model_hash(model) == original_hash
        results[case] = dict(n=n, correct=correct, accuracy={k:v/n for k,v in correct.items()}, model_sha256=original_hash)
        del model, kernel8, x, attacked
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    plot_results(results, args.output, args.attack_iters)
    print(json.dumps(dict(results=results, clean_definition='original model without defense',
                         attack_protocol=PROTOCOL, patchfool_protocols=protocols, source_sha256=sources, source_manifest_sha256=manifest_hash,
                         attack_iters=args.attack_iters, batch_size=args.batch_size, seed=manifest['config']['seed'],
                         elapsed_seconds=time.monotonic()-started), indent=2), flush=True)


if __name__ == '__main__':
    main()
