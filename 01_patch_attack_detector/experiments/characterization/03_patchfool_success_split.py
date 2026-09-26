"""03 분석(PatchFool 전용): 공격에 성공한 이미지를 "공격 패치로 CLS attention이 실제로 몰린 애"와
"안 몰린 애"로 나눠 레이어별로 비교한다. 02와 같은 실행 폴더를 입력으로 쓰고, 02의 검증/맵 함수를
그대로 가져다 쓴다. 기존 결과 파일은 건드리지 않고 새 PNG 3장(focused/unfocused 12개 블록, focused 6·9·12 블록)만 만든다.

몰림의 정의(임계값 없음): 마지막 블록(12)에서 공격 패치가 CLS attention(헤드 평균)의 196개 patch 중
1등이다. 배포용 Top-K 탐지기가 의심 위치를 argmax로 잡는 것과 같은 기준이다.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('compare02', HERE / '02_compare_attention_success_failure.py')
compare02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare02)

ATTACK = 'patchfool'
LAYERS_OF_INTEREST = (7, 8)   # 논문(ARMOR)의 예시 그림에서 "activated" 단계로 나온 구간 근처


def attacked_patch_series(metrics, gt_patch):
    """공격 패치가 CLS attention에서 차지하는 값(%)과 순위(1=196개 중 최대)를 레이어별로."""
    n = len(gt_patch)
    attacked = metrics['attacked'][:, :, 1:]
    clean = metrics['clean'][:, :, 1:]
    attacked_at = attacked[np.arange(n), :, gt_patch]
    clean_at = clean[np.arange(n), :, gt_patch]
    rank = (attacked > attacked_at[:, :, None]).sum(-1) + 1
    return attacked_at * 100, clean_at * 100, rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='manifest.json이 있는 동일 실행 폴더')
    parser.add_argument('--output', type=Path, help='기본값: results/characterization/patchfool/')
    args = parser.parse_args()
    manifest, data = compare02.load_paired_run(args.run)
    if ATTACK not in manifest['attacks']:
        raise SystemExit(f'{ATTACK} 결과가 이 실행에 없습니다.')
    output = args.output or args.run.resolve().parent.parent / ATTACK
    output.mkdir(parents=True, exist_ok=True)

    coverage = data[ATTACK + '__gt_coverage']
    if not ((coverage > 0).sum(axis=1) == 1).all():
        raise ValueError('PatchFool은 표본마다 공격 patch가 정확히 1개여야 합니다.')
    gt_patch = coverage.argmax(axis=1)
    metrics = compare02.compute_metrics(data['clean__attention'], data[ATTACK + '__attention'], coverage)
    outcome = data[ATTACK + '__outcome']
    success = outcome == 1

    attacked_pct, clean_pct, rank = attacked_patch_series(metrics, gt_patch)
    last = rank.shape[1] - 1
    focused = success & (rank[:, last] == 1)
    unfocused = success & ~focused
    if not focused.any() or not unfocused.any():
        raise ValueError(f'집단이 비었습니다: focused={focused.sum()}, not focused={unfocused.sum()}')

    # 대표 예시: focused는 마지막 블록 값이 집단 중앙값에 가장 가까운 표본, not focused는 전 레이어에서
    # 공격 패치 attention의 최댓값이 가장 작은 표본(= 공격 패치를 가장 안 본 표본).
    scales = compare02.map_scales(metrics, outcome)
    focused_idx = np.flatnonzero(focused)
    pick_focused = int(focused_idx[np.argmin(np.abs(attacked_pct[focused_idx, last]
                                                    - np.median(attacked_pct[focused_idx, last])))])
    unfocused_idx = np.flatnonzero(unfocused)
    pick_unfocused = int(unfocused_idx[np.argmin(attacked_pct[unfocused_idx].max(axis=1))])
    layers = list(range(manifest['n_layers']))
    for name, sample_i, label in [
            ('focused', pick_focused, f'Attack succeeded · FOCUSED (top-1 at block {last + 1})'),
            ('unfocused', pick_unfocused, f'Attack succeeded · NOT focused (rank {int(rank[pick_unfocused, last])} at block {last + 1})')]:
        compare02.draw_maps(data, manifest, ATTACK, sample_i, metrics, layers, scales, label,
                            output / f'attack_success_{name}.png')
    # focused 예시를 6·9·12 블록만 (12개 전부는 세로로 너무 길어서)
    compare02.draw_maps(data, manifest, ATTACK, pick_focused, metrics, [5, 8, 11], scales,
                        f'Attack succeeded · FOCUSED (top-1 at block {last + 1})',
                        output / 'attack_success_focused_L6_L9_L12.png')
    print(f'focused example: sample {int(data["sample_index"][pick_focused])}, '
          f'not-focused example: sample {int(data["sample_index"][pick_unfocused])}')

    for block in LAYERS_OF_INTEREST:
        b = block - 1
        print(f'block {block}: attacked-patch attention median % — focused {np.median(attacked_pct[focused, b]):.2f}, '
              f'not focused {np.median(attacked_pct[unfocused, b]):.2f}, clean {np.median(clean_pct[success, b]):.2f} | '
              f'median rank — focused {np.median(rank[focused, b]):.0f}, not focused {np.median(rank[unfocused, b]):.0f} | '
              f'in top-3: focused {(rank[focused, b] <= 3).sum()}/{focused.sum()}, '
              f'not focused {(rank[unfocused, b] <= 3).sum()}/{unfocused.sum()}')
    print(f'focused n={focused.sum()}, not focused n={unfocused.sum()}, saved to {output}')


if __name__ == '__main__':
    main()
