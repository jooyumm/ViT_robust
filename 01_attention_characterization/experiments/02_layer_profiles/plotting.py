"""레이어 실험 전용 그림 구성. 공통 표시 도구는 vis에서 가져온다."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.visualization.boxes import draw_patch_box, draw_attack_region, draw_patch_grid
from src.visualization.colors import TRACKED_PATCH_COLOR, attack_color
from src.visualization.figs import unnormalize

def draw_population(values, output, group="clean", color="#2465a0", method="final_max", ylim=None):
    """ylim을 넘기면 clean/patchfool 두 그림이 같은 y축을 공유한다(각자 따로 자동 스케일되지 않게)."""
    x = np.arange(1, values.shape[1] + 1)
    lo, q25, median, q75, hi = np.quantile(values * 100, [.05, .25, .5, .75, .95], axis=0)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    ax.plot(x, values.T * 100, color=color, alpha=.035, lw=.6)
    ax.fill_between(x, lo, hi, color=color, alpha=.12, label='Pointwise 5–95%')
    ax.fill_between(x, q25, q75, color=color, alpha=.25, label='Pointwise 25–75%')
    ax.plot(x, median, 'o-', color=color, lw=2, label=f'{group} median (n={len(values)})')
    ax.set(xlabel='Layer', ylabel='Raw CLS attention to fixed patch (%)',
           title=f'{method} | {group} | fixed patch across layers')
    ax.set_xticks(x)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=.15)
    ax.legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def draw_examples(data, manifest, patches, peaks, values, selected, output, groups, colors, method, ylim_max=None, map_max=None):
    """selected(pick)마다 group(clean/patchfool)별로 한 행씩 묶어 공용 그림 하나에 쌓는다 --
    같은 이미지의 clean/공격 행이 바로 위아래로 붙어서 직접 비교된다. pick이 1개면 행 2개(clean/공격)만 나온다.
    ylim_max를 넘기면 모든 행이 같은 y축을 공유한다 -- 행마다 따로 스케일되면 높이를 직접 비교할 수 없기 때문이다."""
    n = values[groups[0]].shape[0]
    layers = values[groups[0]].shape[1]
    tokens = patches[groups[0]].shape[-1]
    grid = int(np.sqrt(tokens))
    if grid * grid != tokens:
        raise ValueError('Example maps require a square patch grid.')
    show_layers = sorted(set([0, min(5, layers - 1), min(8, layers - 1), max(0, layers - 2), layers - 1]))
    rows = [(idx, distance, rank, group) for idx, distance, rank in selected for group in groups]
    fig = plt.figure(figsize=(18, 3.2 * len(rows)), layout='constrained')
    gs = fig.add_gridspec(len(rows), 3 + len(show_layers))
    vmax = map_max if map_max is not None else max(
        float(patches[g][[s[0] for s in selected]][:, show_layers].max() * 100) for g in groups)
    medians = {g: np.median(values[g], axis=0) * 100 for g in groups}
    heat_axes = []
    for row, (idx, distance, rank, group) in enumerate(rows):
        patch, peak, color = patches[group], peaks[group], colors[group]
        ax = fig.add_subplot(gs[row, 0])
        picture = unnormalize(data[group + '__images'][idx], manifest['mean'], manifest['std'])
        ax.imshow(picture)
        cell_h, cell_w = picture.shape[0] / grid, picture.shape[1] / grid
        draw_patch_grid(ax, ppl=grid, patch_size=cell_w)
        draw_patch_box(ax, int(peak[idx]), TRACKED_PATCH_COLOR, ppl=grid, patch_size=cell_w)
        if group != 'clean':
            draw_attack_region(ax, data[group + '__gt_coverage'][idx], group, ppl=grid, patch_size=cell_w)
        correct = data[group + '__prediction'][idx] == data['label'][idx]
        ax.set_title(f'{group}-{"correct" if correct else "incorrect"}\nmost divergent (of {n})', fontsize=9)
        ax.axis('off')
        ax = fig.add_subplot(gs[row, 1:3])
        ax.plot(np.arange(1, layers + 1), medians[group], '--', color='#888', label='Population median')
        ax.plot(np.arange(1, layers + 1), values[group][idx] * 100, 'o-', color=color, label=group + ' image')
        ax.set(xlabel='Layer', ylabel='Raw attention (%)')
        ax.set_xticks(np.arange(1, layers + 1))
        ax.set_ylim(0, (ylim_max if ylim_max is not None else values[group][idx].max() * 100) * 1.05 + 1e-8)
        ax.grid(alpha=.15)
        ax.legend(fontsize=7)
        for column, layer in enumerate(show_layers, start=3):
            ax = fig.add_subplot(gs[row, column])
            im = ax.imshow(patch[idx, layer].reshape(grid, grid) * 100,
                           cmap='Blues', vmin=0, vmax=vmax, interpolation='nearest')
            draw_patch_box(ax, int(peak[idx]), TRACKED_PATCH_COLOR, ppl=grid, patch_size=1, lw=1.8)
            if group != 'clean':
                draw_attack_region(ax, data[group + '__gt_coverage'][idx], group, ppl=grid, patch_size=1)
            ax.set_xlabel(f'Layer {layer + 1}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            heat_axes.append(ax)
    fig.colorbar(im, ax=heat_axes, shrink=.6, label='Raw CLS-to-patch attention (%)')
    legend_handles = [Patch(edgecolor=TRACKED_PATCH_COLOR, facecolor='none', lw=1.8, label='Tracked patch (max attention)')]
    for group in groups:
        if group != 'clean':
            legend_handles.append(Patch(edgecolor=attack_color(group), facecolor='none', lw=1.8, label=f'{group} GT region'))
    fig.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=.9)
    fig.suptitle('Does attention to the tracked patch change under attack? Most clean-vs-attack divergent example',
                 fontsize=15, fontweight='bold')
    fig.savefig(output, dpi=150)
    plt.close(fig)



def draw_comparison(series, output, method):
    """같은 축에서 세 조건의 중앙값과 사분위 구간 비교."""
    from src.visualization.colors import GROUP_COLORS
    from src.visualization.figs import save_fig
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    for group, values in series.items():
        x = np.arange(1, values.shape[1] + 1)
        q25, median, q75 = np.quantile(values * 100, [.25, .5, .75], axis=0)
        ax.plot(x, median, 'o-', color=GROUP_COLORS[group], label=f'{group} (n={len(values)})')
        ax.fill_between(x, q25, q75, color=GROUP_COLORS[group], alpha=.13)
    selection = 'independently selected final maximum' if method == 'final_max' else 'clean final-maximum position shared across conditions'
    ax.set(title=f'{method}: {selection}\nMedian and IQR',
           xlabel='Layer', ylabel='Raw CLS attention to selected patch (%)')
    ax.set_xticks(x)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.15)
    ax.legend()
    save_fig(fig, output)
