"""02 분석: 동일 실행의 clean/attack attention과 GT로 공격 성공·실패를 비교한다.

입력은 01_extract_paired_attention.py가 만든 실행 폴더이다.
예측/표본 ID가 없는 과거 NPZ와 독립적으로 재생성한 공격 결과는 합치지 않는다.
출력은 공격별·결과별 폴더에 그림 몇 장과 README뿐이다. CSV는 만들지 않는다 —
숫자는 README의 레이어 요약표로 충분하고, 나머지는 표본 수가 많아 훑어보기 어렵다.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle

MASS_ATOL = 1e-4
# 결과 이름 자체가 "무엇이 성공/실패했는지"를 말하게 한다 -- "failure"만 보면 공격이
# 실패했다는 건지 분류가 실패(오분류)했다는 건지 구분이 안 된다는 지적 반영.
OUTCOMES = {'attack_success': (1, 'Attack succeeded (misclassified)'),
            'attack_fail': (0, 'Attack failed (still correct)')}
# 방어 관점에서 공격 성공=critical, 실패=good인 상태 색 사용 -- 이름의 의미와 색이 서로 확인해준다.
OUTCOME_COLORS = {'attack_success': '#d03b3b', 'attack_fail': '#0ca30c'}
GT_COLOR = '#15c7c4'
# 순차 색상: 옅은 하늘색(0에 가까움, 흰 배경과도 검은 셀과도 확실히 구분) -> 짙은 남색(최댓값).
# 기존 magma+선형 스케일은 0에 가까운 값이 전부 거의 검게 뭉개져서 비교가 안 됐다.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    'seq_blue', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#1c5cab', '#0d366b'])
DIVERGING = 'RdBu_r'
METRICS = {'tv': 'Total variation', 'gt_delta': 'Attention change in GT region (pp)'}


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
    return {
        'clean': clean, 'attacked': attacked, 'delta': delta,
        'tv': .5 * np.abs(delta).sum(-1),
        'gt_delta': attacked_delta, 'other_delta': other_delta, 'cls_delta': cls_delta,
    }


def save_figure(figure, path):
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def draw_gt(axis, coverage, side):
    """부분적으로라도 변경된 토큰에 테두리를 표시한다."""
    for patch_i in np.flatnonzero(coverage > 0):
        row, column = divmod(int(patch_i), side)
        axis.add_patch(Rectangle((column - .5, row - .5), 1, 1,
                                fill=False, edgecolor=GT_COLOR, linewidth=1.2))


def robust_scale(values, percentile=99):
    """단일 극단값이 색 범위를 지배하지 않도록 백분위수로 vmax를 잡는다.
    이보다 큰 값은 색이 잘리며(colorbar의 extend로 표시), 실제 값이 바뀌지는 않는다."""
    finite = np.abs(values[np.isfinite(values)])
    return float(np.percentile(finite, percentile)) if finite.size else 1e-8


def draw_maps(data, manifest, attack, sample_i, metrics, layers, scales, status_label, output):
    """clean/attack 사진 + GT coverage를 첫 행에, 이후 지정한 레이어만 clean/attack/차이로 그린다.
    모든 행(사진 포함)이 같은 3열 grid를 쓰고, 컬러바는 열 전체(사진 행 포함)에서 폭을
    빌려 배치한다 -- 그래야 컬러바가 붙는 마지막 레이어 행만 폭이 줄어 다른 행과 어긋나
    보이는 문제가 생기지 않는다."""
    sample_id = int(data['sample_index'][sample_i])
    coverage = data[attack + '__gt_coverage'][sample_i]
    side = int(np.sqrt(len(coverage)))
    if side * side != len(coverage):
        raise ValueError('현재 맵은 정사각 patch 격자를 지원합니다.')
    n_rows = len(layers) + 1
    figure, axes = plt.subplots(n_rows, 3, figsize=(11, 2.9 * n_rows), layout='constrained')
    mean = np.asarray(manifest['mean'])[:, None, None]
    std = np.asarray(manifest['std'])[:, None, None]
    extent = [-.5, side - .5, side - .5, -.5]

    clean_img = np.clip((data['clean__images'][sample_i] * std + mean).transpose(1, 2, 0), 0, 1)
    attack_img = np.clip((data[attack + '__images'][sample_i] * std + mean).transpose(1, 2, 0), 0, 1)
    coverage_pct = coverage.reshape(side, side) * 100

    top_panels = [(0, clean_img, 'Clean', False), (1, attack_img, attack, False),
                 (2, coverage_pct, 'GT: % of patch changed', True)]
    for column, image, title, is_coverage in top_panels:
        axis = axes[0, column]
        if is_coverage:
            axis.imshow(image, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=100, extent=extent)
        else:
            axis.imshow(image, extent=extent)
        draw_gt(axis, coverage, side)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])

    mass_max, delta_max = scales
    mass_norm = PowerNorm(gamma=.45, vmin=0, vmax=mass_max)
    delta_norm = TwoSlopeNorm(vmin=-delta_max, vcenter=0, vmax=delta_max)
    column_image = {}
    for row, layer_i in enumerate(layers, 1):
        panels = [('clean', 'Clean'), ('attacked', attack), ('delta', 'Attack − clean')]
        for column, (key, title) in enumerate(panels):
            axis = axes[row, column]
            values = metrics[key][sample_i, layer_i, 1:].reshape(side, side) * 100
            if key == 'delta':
                image = axis.imshow(values, cmap=DIVERGING, norm=delta_norm, interpolation='nearest')
            else:
                image = axis.imshow(values, cmap=SEQUENTIAL_BLUE, norm=mass_norm, interpolation='nearest')
            column_image[column] = image
            draw_gt(axis, coverage, side)
            axis.set_title(f'L{layer_i + 1} · {title}')
            axis.set_xticks([])
            axis.set_yticks([])
            if key == 'delta':
                gt_gain = metrics['gt_delta'][sample_i, layer_i] * 100
                cls_gain = metrics['cls_delta'][sample_i, layer_i] * 100
                axis.set_xlabel(f'GT {gt_gain:+.2f} pp | CLS {cls_gain:+.2f} pp', fontsize=8)
    # 사진 행을 포함한 열 전체에서 공간을 빌려야 모든 행의 폭이 똑같이 유지된다.
    for column, label, extend in [(0, 'Raw attention (%)', 'max'),
                                   (1, 'Raw attention (%)', 'max'),
                                   (2, 'Attack − clean (pp)', 'both')]:
        figure.colorbar(column_image[column], ax=[axes[r, column] for r in range(n_rows)],
                        shrink=.85, extend=extend, label=label)
    label_value = int(data['label'][sample_i])
    clean_pred = int(data['clean__prediction'][sample_i])
    attacked_pred = int(data[attack + '__prediction'][sample_i])
    figure.suptitle(f'{attack} · sample {sample_id} · {status_label}\n'
                   f'label {label_value} | prediction {clean_pred} → {attacked_pred}', fontsize=13)
    save_figure(figure, output)


def draw_layer_comparison(attack, metrics, outcome, output):
    """모든 clean 정답 표본을 집계하며 빈 집단과 작은 표본 수도 그대로 표시한다."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    layers = np.arange(1, metrics['tv'].shape[1] + 1)
    for axis, (key, title) in zip(axes, METRICS.items()):
        for status, (code, status_label) in OUTCOMES.items():
            values = metrics[key][outcome == code] * 100
            color = OUTCOME_COLORS[status]
            if not len(values):
                axis.plot([], [], color=color, label=f'{status_label}: n=0')
                continue
            lower, median, upper = np.quantile(values, [.25, .5, .75], axis=0)
            axis.plot(layers, median, 'o-', color=color, label=f'{status_label}: n={len(values)}')
            axis.fill_between(layers, lower, upper, color=color, alpha=.18)
        axis.axhline(0, color='#777', linewidth=.6)
        axis.set(title=title, xlabel='Block', ylabel='Percentage points' if key == 'gt_delta' else 'TV × 100')
        axis.set_xticks(layers)
        axis.grid(alpha=.15)
        axis.legend(fontsize=8)
    figure.suptitle(f'{attack} · attack succeeded vs failed · median and IQR (not confidence intervals)')
    save_figure(figure, output)


def flag_layers(metrics, outcome):
    """특이 레이어 후보를 재현 가능한 기술통계 기준으로 고른다. 검정이나 인과 판정이 아니다."""
    valid = outcome >= 0
    median_tv = np.median(metrics['tv'][valid], axis=0) if valid.any() else np.zeros(metrics['tv'].shape[1])
    median_gt_delta = {code: (np.median(metrics['gt_delta'][outcome == code], axis=0)
                              if (outcome == code).any() else None)
                       for _, (code, _) in OUTCOMES.items()}
    jump = np.diff(median_tv, prepend=median_tv[0])
    flagged = {int(np.argmax(median_tv))}
    if len(jump) > 1 and jump[1:].max() > 0:
        flagged.add(int(np.argmax(jump)))
    success_gt = median_gt_delta.get(1)
    fail_gt = median_gt_delta.get(0)
    if success_gt is not None and fail_gt is not None:
        flagged.add(int(np.argmax(np.abs(success_gt - fail_gt))))
    return median_tv, median_gt_delta, sorted(flagged)


def pick_representative(metrics, outcome, code):
    """집단 평균 TV가 중앙값에 가장 가까운 표본을 고정된 규칙으로 고른다."""
    indices = np.flatnonzero(outcome == code)
    if not len(indices):
        return None
    score = metrics['tv'].mean(axis=1)[indices]
    return int(indices[np.argmin(np.abs(score - np.median(score)))])


def layer_table(n_layers, median_tv, median_gt_delta, flagged):
    """README에 넣을 레이어별 요약표 -- 별도 CSV 대신 이 표 하나로 충분하다."""
    lines = ['| Block | TV (median) | GT Δ success (pp) | GT Δ fail (pp) | flagged |',
             '|---|---|---|---|---|']
    for layer_i in range(n_layers):
        success = median_gt_delta.get(1)
        fail = median_gt_delta.get(0)
        success_cell = f'{success[layer_i] * 100:+.2f}' if success is not None else 'n/a'
        fail_cell = f'{fail[layer_i] * 100:+.2f}' if fail is not None else 'n/a'
        mark = '⚠' if layer_i in flagged else ''
        lines.append(f'| {layer_i + 1} | {median_tv[layer_i] * 100:.2f} | {success_cell} | {fail_cell} | {mark} |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='manifest.json이 있는 동일 실행 폴더')
    parser.add_argument('--output', type=Path, help='기본값: 실행 폴더 안 analysis. 기존 폴더 덮어쓰기 금지')
    args = parser.parse_args()
    manifest, data = load_paired_run(args.run)
    output = args.output or args.run / 'analysis'
    output.mkdir(parents=True, exist_ok=False)
    n_layers = manifest['n_layers']

    results = {}
    for attack in manifest['attacks']:
        results[attack] = compute_metrics(data['clean__attention'], data[attack + '__attention'],
                                          data[attack + '__gt_coverage'])

    readme_sections = []
    for attack, metrics in results.items():
        outcome = data[attack + '__outcome']
        attack_dir = output / attack
        attack_dir.mkdir()
        draw_layer_comparison(attack, metrics, outcome, attack_dir / 'layer_comparison.png')

        median_tv, median_gt_delta, flagged = flag_layers(metrics, outcome)
        # 레이어 전부(clean 대비 attack이 레이어를 타고 어떻게 바뀌는지 그 자체가 보고 싶은
        # 것)를 보여준다 -- flag_layers는 README 표에 표시할 후보를 고르는 용도로만 쓴다.
        example_layers = list(range(n_layers))

        # 표본별 자동 확대는 금지한다 -- 같은 공격의 모든 clean 정답 표본에서 공통 범위를 잡는다.
        # 초반 레이어는 진짜로 거의 변화가 없고 후반 레이어에서 커지는 것 자체가 보여주려는
        # 내용이라, 레이어 전체를 합쳐 하나의 범위로 잡는다 -- percentile을 쓰는 이유는
        # 극단적인 표본 하나가 전체 스케일을 지배하지 않게 하기 위해서일 뿐, 레이어 간 진짜
        # 차이(초반 옅음 -> 후반 진함)는 그대로 드러난다.
        # 맵에 그리는 값은 전부 *100(퍼센트 포인트)이라 스케일도 같은 단위로 잡아야 한다 --
        # 여기서 *100을 빼먹으면 vmax가 100배 작아져서 거의 모든 patch가 최고색으로 잘린다.
        valid = outcome >= 0
        mass_max = robust_scale(100 * np.concatenate([metrics['clean'][valid, :, 1:],
                                                        metrics['attacked'][valid, :, 1:]]),
                                percentile=99.5)
        delta_max = robust_scale(100 * metrics['delta'][valid, :, 1:], percentile=99.5)
        scales = max(mass_max, 1e-8), max(delta_max, 1e-8)

        counts = {}
        for status, (code, status_label) in OUTCOMES.items():
            n = int((outcome == code).sum())
            counts[status] = n
            status_dir = attack_dir / status
            status_dir.mkdir()
            sample_i = pick_representative(metrics, outcome, code)
            if sample_i is None:
                continue
            sample_id = int(data['sample_index'][sample_i])
            draw_maps(data, manifest, attack, sample_i, metrics, example_layers, scales,
                      status_label, status_dir / f'example_sample{sample_id}.png')

        readme_sections.append(
            f'## {attack}\n\n'
            f'- Attack succeeded (misclassified): n={counts["attack_success"]}. '
            f'Attack failed (still correct): n={counts["attack_fail"]}. '
            f'(Clean-incorrect samples are excluded from this comparison.)\n'
            f'- Flagged blocks (largest median TV / largest TV jump / largest '
            f'success-vs-fail GT gap): {", ".join(str(i + 1) for i in flagged) or "none"}.\n'
            f'- `{attack}/layer_comparison.png`: population median + IQR by block.\n'
            f'- `{attack}/attack_success/`, `{attack}/attack_fail/`: one representative example each, '
            f'clean vs. attack vs. difference at every block (1-{n_layers}) — the flagged blocks above '
            f'are where the population-level chart says to look closest, not the only blocks shown.\n\n'
            f'{layer_table(n_layers, median_tv, median_gt_delta, flagged)}\n')

    (output / 'README.md').write_text(
        '# Attention before/after attack: success vs failure, by layer\n\n'
        '- Folders are organized `<attack>/attack_success/` (attack succeeded, image '
        'misclassified) and `<attack>/attack_fail/` (attack failed, image still correct). '
        'These names describe the ATTACK\'s outcome, not a pipeline failure.\n'
        '- Maps show CLS-query attention to each patch key, head-averaged, CLS key mass kept '
        '(not renormalized away). The teal box marks GT patches (any pixel actually changed).\n'
        '- Sequential color (light blue -> dark navy) covers raw attention; it uses a '
        'power-law scale and a 99.5th-percentile cap computed only over the blocks actually '
        'shown (not all 12 — mixing near-zero early blocks with the huge late-block swings '
        'would wash out exactly the blocks being displayed). Values above the cap are drawn '
        'at the top color, marked by the colorbar\'s arrow, not silently redrawn.\n'
        '- Diverging color (blue negative / red positive) covers attack-minus-clean change, '
        'centered on zero.\n'
        '- GT Δ: attention change summed over GT patches (raw, not renormalized). '
        'TV: 0.5 × sum(|attack - clean|) over all 197 tokens including CLS.\n'
        '- The layer table uses the *median* over all clean-correct samples; flagged blocks '
        '(⚠) are exploratory candidates by descriptive statistics, not a significance test '
        'or causal claim.\n'
        '- No CSVs here by design — the table above is the numeric result worth keeping. '
        'Per-sample/per-head raw numbers are recomputable from `../data/` with '
        '`compute_metrics()` in this folder\'s script if ever needed.\n\n'
        + '\n'.join(readme_sections))
    print(f'분석 완료: {output}')


if __name__ == '__main__':
    main()
