"""02 분석: 동일 실행의 clean/attack attention과 GT로 공격 성공·실패를 비교한다.

입력은 01_extract_paired_attention.py가 만든 실행 폴더이다.
예측/표본 ID가 없는 과거 NPZ와 독립적으로 재생성한 공격 결과는 합치지 않는다.
출력은 공격별·결과별 폴더에 그림 몇 장과 README뿐이다. CSV는 만들지 않는다 —
숫자는 README의 레이어 요약표로 충분하고, 나머지는 표본 수가 많아 훑어보기 어렵다.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'ground_truth'))   # for `import draw_boxes`
from draw_boxes import draw_patch_box, CLEAN_BOX_COLOR, ATTACK_BOX_COLOR   # noqa: E402 -- GT 박스는 여기서 그대로 가져다 쓴다

MASS_ATOL = 1e-4
# 결과 이름 자체가 "무엇이 성공/실패했는지"를 말하게 한다 -- "failure"만 보면 공격이
# 실패했다는 건지 분류가 실패(오분류)했다는 건지 구분이 안 된다는 지적 반영.
OUTCOMES = {'attack_success': (1, 'Attack succeeded (misclassified)'),
            'attack_fail': (0, 'Attack failed (still correct)')}
# 방어 관점에서 공격 성공=critical, 실패=good인 상태 색 사용 -- 이름의 의미와 색이 서로 확인해준다.
OUTCOME_COLORS = {'attack_success': '#d03b3b', 'attack_fail': '#0ca30c'}
# 양 끝이 검정으로 가면 원본/공격/차이가 다 어두워져서 구분이 안 된다 -- 끝 색을 원색으로 둔다.
# raw attention: 흰색 -> 순수 파랑. 차이: 감소=연초록->밝은 초록->진한 초록(#008000), 0=흰색, 증가(attention이 더 몰림)=순수 빨강.
SEQUENTIAL = LinearSegmentedColormap.from_list('white_blue', ['#ffffff', '#0000ff'])
# 0은 흰색. 흰색에서 바로 연한 초록/연한 빨강으로 건너뛴 뒤 원색으로 진해진다 -- 선형으로
# 흰색->원색을 이으면 작은 변화가 흰색과 구분이 안 된다.
DIVERGING = LinearSegmentedColormap.from_list('green_white_red', [
    (0.0, '#008000'), (0.3, '#00ff00'), (0.4, '#99ff99'), (0.5, '#ffffff'), (0.6, '#ff9999'), (0.7, '#ff0000'), (1.0, '#800000')])


class SignedPowerNorm(Normalize):
    """0을 중심으로 부호를 유지한 채 |값|^gamma로 펼친다(gamma<1이면 작은 변화도 흰색에서
    확실히 떨어진다). 컬러바가 눈금을 되돌릴 수 있도록 inverse를 제공한다."""

    def __init__(self, vmax, gamma=.5):
        super().__init__(vmin=-vmax, vmax=vmax, clip=True)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        v = np.ma.asarray(value, dtype=float)
        scaled = np.sign(v) * (np.clip(np.abs(v) / self.vmax, 0, 1) ** self.gamma)
        return np.ma.masked_array(.5 + .5 * scaled, mask=np.ma.getmask(v))

    def inverse(self, value):
        s = 2 * np.asarray(value, dtype=float) - 1
        return np.sign(s) * (np.abs(s) ** (1 / self.gamma)) * self.vmax
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


def draw_gt(axis, coverage, side, color):
    """부분적으로라도 변경된 토큰에 테두리를 표시한다 -- ground_truth/draw_boxes.py의
    draw_patch_box를 그대로 쓴다. draw_boxes.py와 같은 규약: clean 패널은 CLEAN_BOX_COLOR(파랑),
    공격 후/차이 패널은 ATTACK_BOX_COLOR(노랑). 이 맵은 imshow 기본 extent(칸 중심이 정수
    좌표)라 patch_size=1, offset=-.5로 반 칸 보정한다."""
    for patch_i in np.flatnonzero(coverage > 0):
        draw_patch_box(axis, int(patch_i), color, ppl=side, patch_size=1, lw=1.2, offset=-.5)


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
    top_panels = [(0, clean_img, 'Clean', CLEAN_BOX_COLOR), (1, attack_img, attack, ATTACK_BOX_COLOR)]
    for column, image, title, box_color in top_panels:
        axis = axes[0, column]
        axis.imshow(image, extent=extent)
        draw_gt(axis, coverage, side, box_color)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    # 차이 열의 첫 행은 비워 둔다(칸은 남겨야 컬러바가 열 전체에서 같은 폭을 빌려 행이 어긋나지 않는다).
    axes[0, 2].set_axis_off()

    mass_max, delta_max = scales
    mass_norm = PowerNorm(gamma=.45, vmin=0, vmax=mass_max)
    delta_norm = SignedPowerNorm(delta_max)
    column_image = {}
    for row, layer_i in enumerate(layers, 1):
        panels = [('clean', 'Clean'), ('attacked', attack), ('delta', '(Attack − clean)')]
        for column, (key, title) in enumerate(panels):
            axis = axes[row, column]
            values = metrics[key][sample_i, layer_i, 1:].reshape(side, side) * 100
            if key == 'delta':
                image = axis.imshow(values, cmap=DIVERGING, norm=delta_norm, interpolation='nearest')
            else:
                image = axis.imshow(values, cmap=SEQUENTIAL, norm=mass_norm, interpolation='nearest')
            column_image[column] = image
            draw_gt(axis, coverage, side, CLEAN_BOX_COLOR if key == 'clean' else ATTACK_BOX_COLOR)
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
        bar = figure.colorbar(column_image[column], ax=[axes[r, column] for r in range(n_rows)],
                              shrink=.85, extend=extend, label=label)
        if column == 2:
            # 눈금을 정수(1,2,3...)로 두면 |값|^0.5 스케일 때문에 양 끝 눈금이 겹친다 -- 상한의
            # 비율로 잡으면 스케일 위에서 간격이 고르다.
            ticks = np.array([-1, -.5, -.25, -.05, 0, .05, .25, .5, 1]) * delta_max
            bar.set_ticks(ticks)
            bar.set_ticklabels([f'{v:.2g}' for v in ticks])
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
    parser.add_argument('--output', type=Path,
                        help='기본값: results/characterization/ (실행이 바뀌어도 같은 자리에 최신 결과만 남김)')
    args = parser.parse_args()
    manifest, data = load_paired_run(args.run)
    # attention_comparison/<run>/analysis/ 처럼 실행마다 폴더를 새로 만들면 결과 폴더가
    # 계속 깊어지고 늘어난다 -- 사람이 보는 요약은 results/characterization/<attack>/ 하나에
    # 모으고, 실행별 원본(재현에 필요한 manifest/data)만 run 폴더에 남긴다.
    output = args.output or args.run.resolve().parent.parent
    output.mkdir(parents=True, exist_ok=True)
    n_layers = manifest['n_layers']

    results = {}
    for attack in manifest['attacks']:
        results[attack] = compute_metrics(data['clean__attention'], data[attack + '__attention'],
                                          data[attack + '__gt_coverage'])

    readme_sections = []
    for attack, metrics in results.items():
        outcome = data[attack + '__outcome']
        attack_dir = output / attack
        attack_dir.mkdir(exist_ok=True)
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
            sample_i = pick_representative(metrics, outcome, code)
            if sample_i is None:
                continue
            sample_id = int(data['sample_index'][sample_i])
            # 폴더로 나누지 않고 파일명에 상태를 넣는다 -- attack_success/attack_fail
            # 폴더 두 개보다, 공격당 폴더 하나에 파일 몇 개가 있는 게 훑어보기 쉽다.
            draw_maps(data, manifest, attack, sample_i, metrics, example_layers, scales,
                      status_label, attack_dir / f'ex_{status}_{sample_id}.png')

        readme_sections.append(
            f'## {attack}\n\n'
            f'- Attack succeeded (misclassified): n={counts["attack_success"]}. '
            f'Attack failed (still correct): n={counts["attack_fail"]}. '
            f'(Clean-incorrect samples are excluded from this comparison.)\n'
            f'- Flagged blocks (largest median TV / largest TV jump / largest '
            f'success-vs-fail GT gap): {", ".join(str(i + 1) for i in flagged) or "none"}.\n'
            f'- `{attack}/layer_comparison.png`: population median + IQR by block.\n'
            f'- `{attack}/ex_attack_success_<id>.png`, '
            f'`{attack}/ex_attack_fail_<id>.png`: one representative example each, '
            f'clean vs. attack vs. difference at every block (1-{n_layers}) — the flagged blocks above '
            f'are where the population-level chart says to look closest, not the only blocks shown.\n\n'
            f'{layer_table(n_layers, median_tv, median_gt_delta, flagged)}\n')

    (output / 'README.md').write_text(
        '# Attention before/after attack: success vs failure, by layer\n\n'
        f'Generated from run `{args.run.resolve()}` — its `manifest.json` and `data/` hold the '
        'reproducible source (seeds, hashes, raw images/attention); re-running this script only '
        'overwrites the files below, not that run.\n\n'
        '- One folder per attack (`patchfool/`, `lavan/`), no outcome subfolders — the outcome is '
        'in the filename instead: `ex_attack_success_*` (attack succeeded, image '
        'misclassified) vs. `ex_attack_fail_*` (attack failed, image still correct). These '
        'names describe the ATTACK\'s outcome, not a pipeline failure.\n'
        '- Maps show CLS-query attention to each patch key, head-averaged, CLS key mass kept '
        '(not renormalized away). GT patches (any pixel actually changed) are outlined with '
        'ground_truth/draw_boxes.py\'s colors, reused directly: blue on Clean panels, yellow on '
        'attacked/difference panels.\n'
        '- Raw attention: white -> pure blue. Attack-minus-clean: pure green (attention lost) -> '
        'white (no change) -> pure red (attention concentrated more). Ends are pure colors so panels stay '
        'distinguishable at the extremes; small changes are stretched (signed power scale) so they stay visibly off-white. Both '
        'use a 99.5th-percentile cap (not the true max) shared across every block of a given '
        'attack, so a single extreme sample or patch can\'t blank out the rest, while the real '
        'block-to-block story (near-zero early on, then increasingly concentrated) still shows. '
        'Values above the cap are drawn at the top color, marked by the colorbar\'s arrow, not '
        'silently redrawn.\n'
        '- GT Δ: attention change summed over GT patches (raw, not renormalized). '
        'TV: 0.5 × sum(|attack - clean|) over all 197 tokens including CLS.\n'
        '- The layer table uses the *median* over all clean-correct samples; flagged blocks '
        '(⚠) are exploratory candidates by descriptive statistics, not a significance test '
        'or causal claim.\n'
        '- No CSVs here by design — the table above is the numeric result worth keeping. '
        f'Per-sample/per-head raw numbers are recomputable from `{args.run.resolve()}/data/` with '
        '`compute_metrics()` in this folder\'s script if ever needed.\n\n'
        + '\n'.join(readme_sections))
    print(f'분석 완료: {output}')


if __name__ == '__main__':
    main()
