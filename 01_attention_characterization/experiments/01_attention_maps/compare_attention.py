"""02 분석: 동일 실행의 clean/attack attention과 GT로 공격 성공·실패를 비교한다.

입력은 src/data/paired.py로 검증하는 공통 paired run이다.
예측/표본 ID가 없는 과거 NPZ와 독립적으로 재생성한 공격 결과는 합치지 않는다.
출력은 공격별·결과별 폴더에 그림 몇 장과 README뿐이다. CSV는 만들지 않는다 —
숫자는 README의 레이어 요약표로 충분하고, 나머지는 표본 수가 많아 훑어보기 어렵다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1]))
import importlib
_metrics = importlib.import_module("experiments.01_attention_maps.metrics")
compute_metrics = _metrics.compute_metrics
from src.data.paired import load_paired_dataset
from src.cohort import DEFAULT_DATASET
import importlib
_plotting = importlib.import_module("experiments.01_attention_maps.plotting")
draw_maps = _plotting.draw_maps
map_scales = _plotting.map_scales
from src.visualization.figs import save_fig   # noqa: E402

MASS_ATOL = 1e-4
# 결과 이름 자체가 "무엇이 성공/실패했는지"를 말하게 한다 -- "failure"만 보면 공격이
# 실패했다는 건지 분류가 실패(오분류)했다는 건지 구분이 안 된다는 지적 반영.
OUTCOMES = {'attack_success': (1, 'Attack succeeded (misclassified)'),
            'attack_fail': (0, 'Attack failed (still correct)')}
# 방어 관점에서 공격 성공=critical, 실패=good인 상태 색 사용 -- 이름의 의미와 색이 서로 확인해준다.
OUTCOME_COLORS = {'attack_success': '#d03b3b', 'attack_fail': '#0ca30c'}

METRICS = {'tv': 'Overall attention shift across all tokens (Total Variation)',
           'gt_delta': 'Attention gained or lost at the attacked patches (GT Δ)'}
DISPLAY_NAMES = {'patchfool': 'PatchFool', 'lavan': 'LaVAN'}


def draw_layer_comparison(attack, metrics, outcome, output):
    """모든 clean 정답 표본을 집계하며 빈 집단과 작은 표본 수도 그대로 표시한다."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    layers = np.arange(1, metrics['tv'].shape[1] + 1)
    bounds = []
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
            bounds += [float(lower.min()), float(upper.max())]
        axis.axhline(0, color='#777', linewidth=.6)
        axis.set(title=title, xlabel='Layer', ylabel='Attention (pp of total mass)')
        axis.set_xticks(layers)
        axis.grid(alpha=.15)
        axis.legend(fontsize=8)
    # TV와 GT Δ 둘 다 attention 질량(합 100%)에 대한 percentage-point 단위라 같은 축을 쓸 수 있다.
    # TV는 절댓값 합이라 항상 0 이상이고 GT Δ는 부호가 있어 음수로도 내려간다 -- 둘의 실제 범위를
    # 모두 덮는 공통 축을 쓴다(TV 쪽 0 아래는 비는 게 정상이다, GT Δ 전용 영역이다).
    low, high = min(0.0, min(bounds)), max(bounds)
    margin = (high - low) * .05
    for axis in axes:
        axis.set_ylim(low - margin, high + margin)
    fig.get_layout_engine().set(h_pad=0.08)
    display = DISPLAY_NAMES.get(attack, attack.title())
    fig.suptitle(f'{display} Success vs Fail', fontsize=15, fontweight='bold')
    save_fig(fig, output)


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
    parser.add_argument('--run', type=Path, default=DEFAULT_DATASET, help='기본: 공통 500장 run')
    parser.add_argument('--output', type=Path,
                        help='기본값: results/01_attention_maps/')
    args = parser.parse_args()
    manifest, data = load_paired_dataset(args.run)
    # attention_comparison/<run>/analysis/ 처럼 실행마다 폴더를 새로 만들면 결과 폴더가
    # 계속 깊어지고 늘어난다 -- 사람이 보는 요약은 results/01_attention_maps/<attack>/ 하나에
    # 모으고, 실행별 원본(재현에 필요한 manifest/data)만 run 폴더에 남긴다.
    output = args.output or HERE.parents[1] / 'results' / HERE.name
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
        scales = map_scales(metrics, outcome)

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
                      status_label, attack_dir / f'{status}.png')

        readme_sections.append(
            f'## {attack}\n\n'
            f'- Attack succeeded (misclassified): n={counts["attack_success"]}. '
            f'Attack failed (still correct): n={counts["attack_fail"]}. '
            f'(Clean-incorrect samples are excluded from this comparison.)\n'
            f'- Flagged blocks (largest median TV / largest TV jump / largest '
            f'success-vs-fail GT gap): {", ".join(str(i + 1) for i in flagged) or "none"}.\n'
            f'- `{attack}/layer_comparison.png`: population median + IQR by block.\n'
            f'- `{attack}/attack_success.png`, '
            f'`{attack}/attack_fail.png`: one representative example each (sample id is in the fig title), '
            f'clean vs. attack vs. difference at every block (1-{n_layers}) — the flagged blocks above '
            f'are where the population-level chart says to look closest, not the only blocks shown.\n\n'
            f'{layer_table(n_layers, median_tv, median_gt_delta, flagged)}\n')

    (output / 'README.md').write_text(
        '# Attention before/after attack: success vs failure, by layer\n\n'
        f'Generated from run `{args.run.resolve()}` — its `manifest.json` and `data/` hold the '
        'reproducible source (seeds, hashes, raw images/attention); re-running this script only '
        'overwrites the files below, not that run.\n\n'
        '- One folder per attack (`patchfool/`, `lavan/`), no outcome subfolders — the outcome is '
        'in the filename instead: `attack_success*` (attack succeeded, image '
        'misclassified) vs. `attack_fail*` (attack failed, image still correct). These '
        'names describe the ATTACK\'s outcome, not a pipeline failure.\n'
        '- Maps show CLS-query attention to each patch key, head-averaged, CLS key mass kept '
        '(not renormalized away). GT patches (any pixel actually changed) are outlined with '
        'src/ground_truth/draw_boxes.py\'s colors, reused directly: blue on Clean panels, attack-specific red/green on '
        'attacked/difference panels.\n'
        '- Raw attention: white -> pure blue. Attack-minus-clean: pure green (attention lost) -> '
        'white (no change) -> pure red (attention concentrated more). Ends are pure colors so panels stay '
        'distinguishable at the extremes; the color ramps jump quickly off white so small values stay visible, while the colorbars are linear with evenly spaced ticks. Both '
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
