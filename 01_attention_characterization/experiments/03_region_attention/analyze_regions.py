"""clean/PatchFool의 2×2 영역 attention 시각화."""
import argparse
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from region_metrics import RUNS
from flow_io import open_flows

def tick_positions(n_regions, max_ticks=8):
    """영역 축의 눈금 위치를 반환한다."""
    step = max(1, -(-n_regions // max_ticks))
    return list(range(0, n_regions, step))

def heatmap(flow_clean, flow_attack, attack, path, area, n_regions):
    """이미지·헤드 평균 흐름과 공격−clean의 원래 값 차이를 표시한다."""
    import matplotlib.pyplot as plt
    from src.visualization.figs import save_fig
    c = flow_clean.mean((0, 2))
    a = flow_attack.mean((0, 2))
    diff = a - c
    (fig, axes) = plt.subplots(3, 12, figsize=(25, 8.5), layout='constrained')
    vmax = max(c.max(), a.max())
    dmax = max(abs(diff).max(), 1e-08)
    ticks = tick_positions(n_regions)
    tick_labels = [f'Q{i + 1}' for i in ticks] if n_regions <= 4 else [str(i + 1) for i in ticks]
    for (row, (arr, label)) in enumerate([(c, 'Clean'), (a, attack), (diff, 'Raw difference (attack - clean)')]):
        for (l, ax) in enumerate(axes[row]):
            im = ax.imshow(arr[l], cmap='RdBu_r' if row == 2 else 'viridis', vmin=-dmax if row == 2 else 0, vmax=dmax if row == 2 else vmax)
            ax.set_xticks(ticks, tick_labels, fontsize=6)
            ax.set_yticks(ticks, tick_labels, fontsize=6)
            ax.set_title(f'L{l + 1}', fontsize=8)
            if l == 0:
                ax.set_ylabel(label + '\nquery region')
            if row == 2:
                ax.set_xlabel('key region', fontsize=7)
        cb_label = 'Raw probability mass' if row < 2 else 'Difference (separate scale)'
        fig.colorbar(im, ax=axes[row].tolist(), shrink=0.7, label=cb_label)
    fig.suptitle(f'Region attention, averaged over {len(flow_clean)} images (clean vs {attack})', fontsize=13, fontweight='bold')
    save_fig(fig, path)

def get_example(group, index):
    """RUNS에서 index번째 이미지를 원본 npz 청크에서 찾아 image/GT coverage/manifest를 반환."""
    import json
    run = RUNS[0]
    meta = json.loads((run / 'manifest.json').read_text())
    offset = 0
    for chunk in meta['chunks']:
        if offset <= index < offset + chunk['n']:
            with np.load(run / chunk['path'], allow_pickle=False) as z:
                j = index - offset
                return (z[group + '__images'][j], None if group == 'clean' else z[group + '__gt_coverage'][j], meta)
        offset += chunk['n']
    raise IndexError(index)

def examples(attack, metrics_a, clean_metrics, area, outcome, folder, n_regions, block, per_category=1):
    """성공/실패/원본 오답 사례의 이미지, incoming 및 CLS 흐름을 표시한다."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from src.visualization.figs import save_fig, unnormalize
    from src.visualization.boxes import draw_attack_region
    codes = [(1, 'Attack succeeded'), (0, 'Attack failed'), (-1, 'Clean already wrong')]
    picks = [(int(i), label) for (code, label) in codes for i in np.flatnonzero(outcome == code)[:per_category]]
    if not picks:
        return
    labels = [f'Q{i + 1}' for i in range(n_regions)] if n_regions <= 4 else [str(i + 1) for i in range(n_regions)]
    ticks = tick_positions(n_regions)
    tick_labels = [labels[i] for i in ticks]
    regionpix = block * 16
    side = round(n_regions ** 0.5)
    # 오른쪽 열(CLS line chart) y축을 6개 패널 전부 동일하게 고정하기 위해 먼저 전체 범위를 구한다.
    cls_values = [source['cls'][i].mean(1) for (i, _) in picks for source in (clean_metrics, metrics_a)]
    cls_ymin, cls_ymax = min(v.min() for v in cls_values), max(v.max() for v in cls_values)
    pad = (cls_ymax - cls_ymin) * 0.05 or 1e-3
    cls_ylim = (cls_ymin - pad, cls_ymax + pad)
    (fig, axes) = plt.subplots(2 * len(picks), 3, figsize=(13, 4 * 2 * len(picks)), squeeze=False, layout='constrained')
    for (pair, (i, label)) in enumerate(picks):
        (atk_img, cov, meta) = get_example(attack, i)
        (clean_img, _, _) = get_example('clean', i)
        for (sub, (image, cov_i, source, name)) in enumerate([(clean_img, None, clean_metrics, 'Clean'), (atk_img, cov, metrics_a, attack)]):
            row = 2 * pair + sub
            ax = axes[row, 0]
            ax.imshow(unnormalize(image, meta['mean'], meta['std']))
            for k in range(regionpix, 224, regionpix):
                ax.axhline(k - 0.5, color='white', lw=0.6, alpha=0.7)
                ax.axvline(k - 0.5, color='white', lw=0.6, alpha=0.7)
            if n_regions <= 8:
                # region이 적을 때는 사진만 보고도 몇 번인지 바로 알 수 있게 칸마다 가운데에 번호를 적는다.
                for r in range(n_regions):
                    row_block, col_block = divmod(r, side)
                    cx, cy = col_block * regionpix + regionpix / 2, row_block * regionpix + regionpix / 2
                    ax.text(cx, cy, str(r + 1), color='white', fontsize=11, fontweight='bold',
                            ha='center', va='center',
                            path_effects=[pe.withStroke(linewidth=2, foreground='black')])
            gt_text = ','.join(str(r) for r in (np.flatnonzero(area[i] > 0) + 1).tolist())
            if cov_i is not None:
                draw_attack_region(ax, cov_i, attack)
            title = (f'{label}\n' if sub == 0 else '') + name + ('' if cov_i is None else f', GT region {gt_text}')
            ax.set_title(title, fontsize=15, fontweight='bold')
            ax.axis('off')
            ax = axes[row, 1]
            im = ax.imshow(source['incoming'][i].mean(1).T, aspect='auto', vmin=0, vmax=1, cmap='viridis')
            ax.set(title='Incoming from patch queries', xlabel='Layer', ylabel='Region')
            ax.set_xticks(range(12), range(1, 13))
            ax.set_yticks(ticks, tick_labels, fontsize=6)
            fig.colorbar(im, ax=ax)
            ax = axes[row, 2]
            cls = source['cls'][i].mean(1)
            for q in range(n_regions):
                ax.plot(np.arange(1, 13), cls[:, q], label=labels[q])
            ax.set(title='Separate CLS-query incoming', xlabel='Layer', ylabel='Raw probability mass', ylim=cls_ylim)
            ax.legend(fontsize=7)
    fig.suptitle('How much does each region attend to others? Clean vs Attacked', fontsize=15, fontweight='bold')
    save_fig(fig, folder / '01_summary_examples.png')

def run_analysis(data, output):
    """2×2 영역의 attention만 시각화한다."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('결과 덮어쓰기 금지: ' + str(output))
    with open_flows(data) as d:
        summaries = {g: dict(incoming=d[g + '__flow'].mean(-2), cls=d[g + '__cls_flow'])
                     for g in ('clean', 'patchfool')}
        folder = output / 'patchfool'
        heatmap(d['clean__flow'], d['patchfool__flow'], 'patchfool',
                folder / '02_layer_flow_heatmaps.png', d['patchfool__attack_area'], 4)
        examples('patchfool', summaries['patchfool'], summaries['clean'],
                 d['patchfool__attack_area'], d['patchfool__outcome'], folder, 4, 7)
    print('완료:', output, flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'extracted/flow.npz')
    parser.add_argument('--output', type=Path, default=HERE.parents[1] / 'results/03_region_attention')
    args = parser.parse_args()
    run_analysis(args.data, args.output)

if __name__ == '__main__':
    main()
