"""02 분석: 동일 실행의 clean/attack attention과 GT로 성공·실패를 비교한다.

입력은 01_extract_paired_attention.py가 만든 실행 폴더이다.
예측/표본 ID가 없는 과거 NPZ와 독립적으로 재생성한 공격 결과는 합치지 않는다.
그림은 PNG만, 수치는 CSV로 저장한다. 별도 로그나 PDF는 생성하지 않는다.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

MASS_ATOL = 1e-4
OUTCOMES = {'success': 1, 'failure': 0}
COLORS = {'success': '#c44e36', 'failure': '#367faa'}
METRICS = {
    'tv': 'Total variation',
    'gt_delta': 'Attention change in GT region',
}


def sha256(path):
    """추출 때 기록한 파일 지문과 대조한다."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def coverage_from_images(clean, attacked, patch_size, threshold):
    """GT가 현재 저장된 이미지 쌍에서 나온 값인지 다시 계산해 검증한다."""
    changed = np.abs(attacked - clean).sum(axis=1) > threshold
    batch, height, width = changed.shape
    cells = changed.reshape(batch, height // patch_size, patch_size, width // patch_size, patch_size)
    return cells.mean(axis=(2, 4)).reshape(batch, -1)


def load_paired_run(run):
    """완료된 동일 실행만 읽고 파일·표본 순서·예측·GT·확률 질량을 검증한다."""
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest.get('schema_version') != 2 or not manifest.get('complete'):
        raise ValueError('완료된 schema_version=2 추출 결과만 분석할 수 있습니다.')
    if not manifest['chunks']:
        raise ValueError('저장된 청크가 없습니다.')
    pieces = {}
    expected_keys = None
    for chunk in manifest['chunks']:
        path = (run / chunk['path']).resolve()
        if run.resolve() not in path.parents or sha256(path) != chunk['sha256']:
            raise ValueError(f'청크 경로 또는 지문 불일치: {path}')
        with np.load(path, allow_pickle=False) as archive:
            if archive['run_id'].item() != manifest['run_id']:
                raise ValueError('다른 실행의 데이터가 섞여 있습니다.')
            if len(archive['sample_index']) != chunk['n']:
                raise ValueError('청크 표본 수가 일치하지 않습니다.')
            keys = set(archive.files) - {'run_id'}
            if expected_keys is not None and keys != expected_keys:
                raise ValueError('청크마다 저장 필드가 다릅니다.')
            expected_keys = keys
            for key in keys:
                pieces.setdefault(key, []).append(archive[key])
    data = {key: np.concatenate(values) for key, values in pieces.items()}
    if not np.array_equal(data['sample_index'], manifest['sample_indices']):
        raise ValueError('표본 ID 또는 저장 순서가 manifest와 다릅니다.')
    if len(np.unique(data['sample_index'])) != len(data['sample_index']):
        raise ValueError('중복 표본 ID가 있습니다.')
    expected_shape = None
    for group in ['clean', *manifest['attacks']]:
        attention = data[group + '__attention']
        logits = data[group + '__logits']
        images = data[group + '__images']
        if not all(np.isfinite(array).all() for array in (attention, logits, images)):
            raise ValueError(f'{group}: NaN 또는 Inf가 있습니다.')
        if attention.ndim != 4 or attention.shape[:3] != (
                len(data['label']), manifest['n_layers'], manifest['n_heads']):
            raise ValueError(f'{group}: attention 차원이 올바르지 않습니다.')
        if expected_shape is not None and attention.shape != expected_shape:
            raise ValueError('clean/attack attention의 차원이 다릅니다.')
        expected_shape = attention.shape
        if (attention < 0).any():
            raise ValueError('음수 attention이 있습니다.')
        np.testing.assert_allclose(attention.sum(-1), 1, atol=MASS_ATOL, rtol=0)
        np.testing.assert_array_equal(logits.argmax(-1), data[group + '__prediction'])
        if group == 'clean':
            continue
        if images.shape != data['clean__images'].shape:
            raise ValueError('clean/attack 이미지 차원이 다릅니다.')
        coverage = coverage_from_images(data['clean__images'], images,
                                        manifest['patch_size'], manifest['gt_threshold'])
        np.testing.assert_allclose(coverage, data[group + '__gt_coverage'], atol=1e-7, rtol=0)
        if attention.shape[-1] != coverage.shape[-1] + 1:
            raise ValueError('GT와 attention 토큰 수가 다릅니다.')
        correct = data['clean__prediction'] == data['label']
        outcome = np.where(correct, (data[group + '__prediction'] != data['label']).astype(np.int8), -1)
        np.testing.assert_array_equal(outcome, data[group + '__outcome'])
    return manifest, data


def compute_metrics(clean_heads, attacked_heads, coverage):
    """헤드 평균 raw 분포로 같은 위치의 차이를 계산한다. CLS는 재정규화하지 않는다."""
    clean = clean_heads.mean(axis=2)
    attacked = attacked_heads.mean(axis=2)
    delta = attacked - clean
    support = coverage > 0
    attacked_delta = (delta[..., 1:] * support[:, None]).sum(-1)
    other_delta = (delta[..., 1:] * ~support[:, None]).sum(-1)
    cls_delta = delta[..., 0]
    np.testing.assert_allclose(attacked_delta + other_delta + cls_delta, 0, atol=MASS_ATOL, rtol=0)
    head_delta = attacked_heads - clean_heads
    return {
        'clean': clean, 'attacked': attacked, 'delta': delta,
        'tv': .5 * np.abs(delta).sum(-1),
        'gt_delta': attacked_delta, 'other_delta': other_delta, 'cls_delta': cls_delta,
        # 부분 공격 토큰을 분수 가중한 값은 별도 보조값이며 픽셀별 중요도를 의미하지 않는다.
        'coverage_weighted_delta': (delta[..., 1:] * coverage[:, None]).sum(-1),
        'gt_clean': (clean[..., 1:] * support[:, None]).sum(-1),
        'gt_attacked': (attacked[..., 1:] * support[:, None]).sum(-1),
        'head_gt_delta': (head_delta[..., 1:] * support[:, None, None]).sum(-1),
        'head_tv': .5 * np.abs(head_delta).sum(-1),
    }


def save_csv(path, rows, header):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def save_figure(figure, path):
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def draw_gt(axis, coverage, side):
    """부분적으로라도 변경된 토큰에 테두리를 표시한다."""
    for patch_i in np.flatnonzero(coverage > 0):
        row, column = divmod(int(patch_i), side)
        axis.add_patch(Rectangle((column - .5, row - .5), 1, 1,
                                fill=False, edgecolor='#15c7c4', linewidth=1.2))


def draw_maps(data, manifest, attack, sample_i, metrics, layers, scales, output):
    """사진은 첫 행만, 이후에는 사진 중첩 없는 clean/attack/차이 맵을 그린다."""
    sample_id = int(data['sample_index'][sample_i])
    outcome = int(data[attack + '__outcome'][sample_i])
    status = {1: 'success', 0: 'failure', -1: 'clean_incorrect'}[outcome]
    coverage = data[attack + '__gt_coverage'][sample_i]
    side = int(np.sqrt(len(coverage)))
    if side * side != len(coverage):
        raise ValueError('현재 맵은 정사각 patch 격자를 지원합니다.')
    figure, axes = plt.subplots(len(layers) + 1, 3,
                               figsize=(11, 2.7 * (len(layers) + 1)), layout='constrained')
    mean = np.asarray(manifest['mean'])[:, None, None]
    std = np.asarray(manifest['std'])[:, None, None]
    for column, group in enumerate(['clean', attack]):
        image = data[group + '__images'][sample_i] * std + mean
        image = np.clip(image.transpose(1, 2, 0), 0, 1)
        axes[0, column].imshow(image, extent=[-.5, side-.5, side-.5, -.5])
        draw_gt(axes[0, column], coverage, side)
        axes[0, column].set_title(group)
        axes[0, column].set_axis_off()
    axes[0, 2].set_axis_off()
    mass_max, delta_max = scales
    for row, layer_i in enumerate(layers, 1):
        panels = [('clean', 'Clean'), ('attacked', 'Attack'), ('delta', 'Attack − clean')]
        for column, (key, title) in enumerate(panels):
            axis = axes[row, column]
            values = metrics[key][sample_i, layer_i, 1:].reshape(side, side) * 100
            if key == 'delta':
                image = axis.imshow(values, cmap='RdBu_r', interpolation='nearest',
                                    norm=TwoSlopeNorm(vmin=-delta_max, vcenter=0, vmax=delta_max))
                label = 'Change (percentage points)'
            else:
                image = axis.imshow(values, cmap='magma', interpolation='nearest', vmin=0, vmax=mass_max)
                label = 'Raw attention (%)'
            draw_gt(axis, coverage, side)
            axis.set_title(f'L{layer_i + 1} · {title}')
            axis.set_xticks([])
            axis.set_yticks([])
            if key == 'delta':
                gt_gain = metrics['gt_delta'][sample_i, layer_i] * 100
                cls_gain = metrics['cls_delta'][sample_i, layer_i] * 100
                axis.set_xlabel(f'GT {gt_gain:+.2f} pp | CLS {cls_gain:+.2f} pp', fontsize=8)
            if row == len(layers):
                figure.colorbar(image, ax=axis, shrink=.75, label=label)
    label = int(data['label'][sample_i])
    clean_pred = int(data['clean__prediction'][sample_i])
    attacked_pred = int(data[attack + '__prediction'][sample_i])
    figure.suptitle(f'{attack} · sample {sample_id} · {status}\n'
                   f'label {label} | prediction {clean_pred} → {attacked_pred}', fontsize=13)
    save_figure(figure, output)


def draw_outcomes(attack, metrics, outcome, output):
    """모든 clean 정답 표본을 집계하며 빈 집단과 작은 표본 수도 그대로 표시한다."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    layers = np.arange(1, metrics['tv'].shape[1] + 1)
    for axis, (key, title) in zip(axes, METRICS.items()):
        for status, code in OUTCOMES.items():
            values = metrics[key][outcome == code] * 100
            if not len(values):
                axis.plot([], [], color=COLORS[status], label=f'{status}: n=0')
                continue
            # 중앙값과 IQR로 집단의 위치와 표본 간 퍼짐을 함께 표시한다.
            lower, median, upper = np.quantile(values, [.25, .5, .75], axis=0)
            axis.plot(layers, median, 'o-', color=COLORS[status], label=f'{status}: n={len(values)}')
            axis.fill_between(layers, lower, upper, color=COLORS[status], alpha=.18)
        axis.axhline(0, color='#777', linewidth=.6)
        axis.set(title=title, xlabel='Block', ylabel='Percentage points' if key == 'gt_delta' else 'TV × 100')
        axis.set_xticks(layers)
        axis.grid(alpha=.15)
        axis.legend()
    figure.suptitle(f'{attack} · success vs failure · median and IQR (not confidence intervals)')
    save_figure(figure, output)


def rank_layers(metrics, outcome):
    """특이 레이어 후보를 재현 가능한 기술통계 기준으로 고른다. 검정이나 인과 판정이 아니다."""
    valid = outcome >= 0
    if not valid.any():
        return [], []
    median_tv = np.median(metrics['tv'][valid], axis=0)
    # 첫 레이어는 이전 값이 없으므로 급증 후보에서 제외한다.
    jump = np.diff(median_tv, prepend=median_tv[0])
    rows = []
    reasons = {}
    reasons.setdefault(int(np.argmax(median_tv)), []).append('largest_median_tv')
    if len(jump) > 1 and jump[1:].max() > 0:
        reasons.setdefault(int(np.argmax(jump)), []).append('largest_positive_tv_jump')
    gap = np.full(len(median_tv), np.nan)
    if (outcome == 1).any() and (outcome == 0).any():
        gap = np.abs(np.median(metrics['gt_delta'][outcome == 1], axis=0)
                     - np.median(metrics['gt_delta'][outcome == 0], axis=0))
        reasons.setdefault(int(np.argmax(gap)), []).append('largest_success_failure_gt_gap')
    for layer_i in range(len(median_tv)):
        rows.append([layer_i + 1, median_tv[layer_i], jump[layer_i], gap[layer_i],
                     ';'.join(reasons.get(layer_i, []))])
    return rows, sorted(reasons)


def select_examples(metrics, outcome):
    """집단 중앙에 가까운 사례, 큰 변화 사례, 반례를 고정된 규칙으로 선택한다."""
    selected = {}
    score = metrics['tv'].mean(axis=1)
    for status, code in OUTCOMES.items():
        indices = np.flatnonzero(outcome == code)
        if not len(indices):
            continue
        median = np.median(score[indices])
        representative = int(indices[np.argmin(abs(score[indices] - median))])
        largest = int(indices[np.argmax(score[indices])])
        selected.setdefault(representative, []).append(status + '_median_tv')
        selected.setdefault(largest, []).append(status + '_largest_tv')
    # 마지막 레이어의 공격 영역 집중 증가 없이 성공 / 증가했지만 실패한 사례도 확인한다.
    for code, condition, reason in [
            (1, metrics['gt_delta'][:, -1] <= 0, 'success_without_last_gt_gain'),
            (0, metrics['gt_delta'][:, -1] > 0, 'failure_with_last_gt_gain')]:
        indices = np.flatnonzero((outcome == code) & condition)
        if len(indices):
            selected.setdefault(int(indices[0]), []).append(reason)
    return selected


def draw_head_details(attack, metrics, sample_i, sample_id, outcome, scale, output):
    """레이어×헤드 지도로 헤드 평균에 가려진 공격 영역 증감을 확인한다."""
    values = metrics['head_gt_delta'][sample_i] * 100
    figure, axis = plt.subplots(figsize=(8, 6), layout='constrained')
    image = axis.imshow(values, cmap='RdBu_r', aspect='auto',
                        norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale))
    axis.set_xticks(range(values.shape[1]), range(1, values.shape[1] + 1))
    axis.set_yticks(range(values.shape[0]), range(1, values.shape[0] + 1))
    axis.set(xlabel='Head', ylabel='Block', title=f'{attack} · sample {sample_id} · outcome {outcome}')
    figure.colorbar(image, ax=axis, label='Raw attention change in GT region (pp)')
    save_figure(figure, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='manifest.json이 있는 동일 실행 폴더')
    parser.add_argument('--output', type=Path, help='기본값: 실행 폴더 안 analysis. 기존 폴더 덮어쓰기 금지')
    parser.add_argument('--all_maps', action='store_true', help='선택 사례 외 모든 clean 정답 표본의 12레이어 맵도 저장')
    args = parser.parse_args()
    manifest, data = load_paired_run(args.run)
    output = args.output or args.run / 'analysis'
    output.mkdir(parents=True, exist_ok=False)
    maps_dir = output / 'maps'
    maps_dir.mkdir()
    details_dir = output / 'layer_details'
    details_dir.mkdir()
    results = {}
    for attack in manifest['attacks']:
        results[attack] = compute_metrics(data['clean__attention'], data[attack + '__attention'],
                                          data[attack + '__gt_coverage'])
    # 모든 공격·성공/실패·표본·레이어에 공통 범위를 적용한다. 표본별 자동 확대 금지.
    mass_max = max(max(values['clean'][..., 1:].max(), values['attacked'][..., 1:].max())
                   for values in results.values()) * 100
    delta_max = max(np.abs(values['delta'][..., 1:]).max() for values in results.values()) * 100
    head_max = max(np.abs(values['head_gt_delta']).max() for values in results.values()) * 100
    scales = max(float(mass_max), 1e-8), max(float(delta_max), 1e-8)
    head_max = max(float(head_max), 1e-8)
    sample_rows, layer_rows, head_rows, summary_rows, selections, candidates = [], [], [], [], [], []
    n_layers = manifest['n_layers']
    for attack, metrics in results.items():
        outcome = data[attack + '__outcome']
        draw_outcomes(attack, metrics, outcome, output / f'{attack}_success_failure_by_layer.png')
        ranking, selected_layers = rank_layers(metrics, outcome)
        candidates.extend([[attack, *row] for row in ranking])
        chosen = select_examples(metrics, outcome)
        selected_samples = np.flatnonzero(outcome >= 0) if args.all_maps else sorted(chosen)
        for sample_i in selected_samples:
            sample_i = int(sample_i)
            sample_id = int(data['sample_index'][sample_i])
            draw_maps(data, manifest, attack, sample_i, metrics, list(range(n_layers)), scales,
                      maps_dir / f'{attack}_sample{sample_id}_all_layers.png')
            selections.append([attack, sample_id, int(outcome[sample_i]),
                               ';'.join(chosen.get(sample_i, ['all_maps']))])
        for layer_i in selected_layers:
            neighbors = list(range(max(0, layer_i - 1), min(n_layers, layer_i + 2)))
            # 후보 레이어에서 각 집단의 가장 큰 변화 사례를 검사하고, 앞뒤 레이어도 보여준다.
            for status, code in OUTCOMES.items():
                indices = np.flatnonzero(outcome == code)
                if not len(indices):
                    continue
                sample_i = int(indices[np.argmax(metrics['tv'][indices, layer_i])])
                sample_id = int(data['sample_index'][sample_i])
                stem = f'{attack}_L{layer_i+1}_{status}_sample{sample_id}'
                draw_maps(data, manifest, attack, sample_i, metrics, neighbors, scales,
                          details_dir / (stem + '_neighbors.png'))
                draw_head_details(attack, metrics, sample_i, sample_id, code, head_max,
                                  details_dir / (stem + '_heads.png'))
                selections.append([attack, sample_id, code, f'candidate_L{layer_i+1}_largest_tv_{status}'])
        for sample_i, sample_id in enumerate(data['sample_index']):
            status = {1: 'success', 0: 'failure', -1: 'clean_incorrect'}[int(outcome[sample_i])]
            coverage = data[attack + '__gt_coverage'][sample_i]
            sample_rows.append([attack, sample_id, manifest['sample_paths'][sample_i], data['label'][sample_i],
                                data['clean__prediction'][sample_i], data[attack + '__prediction'][sample_i],
                                status, int((coverage > 0).sum()), float(coverage.mean())])
            for layer_i in range(n_layers):
                keys = ['tv', 'gt_clean', 'gt_attacked', 'gt_delta', 'other_delta', 'cls_delta', 'coverage_weighted_delta']
                layer_rows.append([attack, sample_id, status, layer_i + 1,
                                   *[metrics[key][sample_i, layer_i] for key in keys]])
                for head_i in range(manifest['n_heads']):
                    head_rows.append([attack, sample_id, status, layer_i + 1, head_i + 1,
                                      metrics['head_gt_delta'][sample_i, layer_i, head_i],
                                      metrics['head_tv'][sample_i, layer_i, head_i]])
        for status, code in OUTCOMES.items():
            for layer_i in range(n_layers):
                for key in METRICS:
                    values = metrics[key][outcome == code, layer_i]
                    quantiles = np.quantile(values, [.25, .5, .75]) if len(values) else [np.nan] * 3
                    summary_rows.append([attack, status, layer_i + 1, key, len(values), *quantiles])
    save_csv(output / 'samples.csv', sample_rows,
             ['attack', 'sample_id', 'path', 'label', 'clean_prediction', 'attack_prediction', 'outcome',
              'touched_tokens', 'changed_pixel_fraction'])
    save_csv(output / 'per_sample_layer.csv', layer_rows,
             ['attack', 'sample_id', 'outcome', 'layer', 'tv', 'gt_clean', 'gt_attack', 'gt_delta',
              'other_delta', 'cls_delta', 'coverage_weighted_delta'])
    save_csv(output / 'per_sample_head.csv', head_rows,
             ['attack', 'sample_id', 'outcome', 'layer', 'head', 'gt_delta', 'tv'])
    save_csv(output / 'success_failure_summary.csv', summary_rows,
             ['attack', 'outcome', 'layer', 'metric', 'n', 'q25', 'median', 'q75'])
    save_csv(output / 'layer_candidates.csv', candidates,
             ['attack', 'layer', 'median_tv', 'tv_jump', 'success_failure_gt_gap', 'selection_reason'])
    save_csv(output / 'selected_examples.csv', selections, ['attack', 'sample_id', 'outcome_code', 'reason'])
    settings = {'run_id': manifest['run_id'], 'manifest_sha256': sha256(args.run / 'manifest.json'),
                'analysis_source_sha256': sha256(Path(__file__)),
                'mass_scale_percent': scales[0], 'delta_scale_pp': scales[1], 'head_scale_pp': head_max,
                'all_maps': args.all_maps}
    (output / 'analysis_settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# 동일 쌍의 공격 전후 attention과 성공/실패 비교\n\n'
        '- 입력의 파일 지문, 실행 ID, 표본 순서, 예측, GT를 검증한 후 분석했습니다.\n'
        '- 성공: clean 정답 → 공격 후 오답. 실패: clean 정답 → 공격 후 정답. 원래 오답은 집단 비교에서 제외합니다.\n'
        '- CLS query의 softmax attention이며 CLS key를 보존합니다. 헤드 평균 후 patch 격자에 표시합니다.\n'
        '- GT는 실제 변경 픽셀에서 계산합니다. clean에도 같은 GT를 적용합니다. 청록색은 변경 토큰입니다.\n'
        '- GT 영역 + 나머지 patch + CLS 변화량 합은 0입니다. 부분 공격 토큰은 토큰 전체를 포함합니다.\n'
        '- coverage 가중치는 별도 보조값입니다. 픽셀별 attention이라는 의미는 아닙니다.\n'
        '- maps: 선택 사례의 모든 레이어, layer_details: 후보 레이어와 앞뒤 맵 및 헤드별 증감.\n'
        '- 모든 공간 맵은 동일한 선형 색 범위입니다. 차이 맵은 0 중심 대칭 범위입니다. 사진 중첩은 없습니다.\n'
        '- 집단 비교 그림은 중앙값/IQR입니다. 신뢰구간이나 유의성 검정이 아닙니다.\n'
        '- layer_candidates는 변화량 최대/급증/성공실패 GT 차이 최대에 따른 탐색 후보입니다. 이상 판정이 아닙니다.\n'
        '- selected_examples에 사례 선정 이유를 기록했습니다. 전체 표본 수치는 CSV에 있습니다.\n'
        '- 모든 레이어를 평균한 값은 사례 선택에만 사용하며, 레이어별 분석을 대체하지 않습니다.\n'
        '- CSV 값은 0~1 단위 attention 또는 그 차이이며 그림에는 100을 곱합니다.\n'
        '- 과거 독립 실행 결과는 사용하지 않습니다. 원본 데이터, PNG, CSV를 보존합니다.\n')
    print(f'분석 완료: {output}')


if __name__ == '__main__':
    main()
