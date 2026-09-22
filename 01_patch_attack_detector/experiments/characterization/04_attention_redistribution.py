"""Where does attention move after attack? Reuse the existing 00 NPZ only.

100-sample argmax changes and stored-example spatial distributions have different
sample coverage. Raw attention retains CLS so regional deltas sum to zero.
Positive/negative deltas describe redistribution, not causal token-to-token flow.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / 'results' / 'characterization'
GROUPS = ('patchfool', 'lavan')
COLORS = {'patchfool': '#bb4a32', 'lavan': '#2879a5'}
REGIONS = ('Attacked patches', 'Other patches', 'CLS')
REGION_COLORS = ('#d45b34', '#367f9f', '#80718f')


def redistribution(clean, attacked, coverage):
    """B,L,197 distributions -> raw delta, regional delta, total variation.

    A touched token belongs entirely to the attacked region, including partially
    covered LaVAN tokens. This is token-level mass, not pixel-level attribution.
    """
    np.testing.assert_allclose(clean.sum(-1), 1, atol=2e-5)
    np.testing.assert_allclose(attacked.sum(-1), 1, atol=2e-5)
    delta = attacked - clean
    support = coverage > 0
    region = np.stack(((delta[..., 1:] * support[:, None]).sum(-1),
                       (delta[..., 1:] * ~support[:, None]).sum(-1), delta[..., 0]), -1)
    np.testing.assert_allclose(region.sum(-1), 0, atol=2e-5)
    tv = .5 * np.abs(delta).sum(-1)
    return delta, region, tv


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def overview(data, results, output):
    n = len(data['clean__argmax_row_avg'])
    nr, nl = results[('patchfool', 'CLS')][0].shape[:2]
    x = np.arange(1, nl + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
    for row, (query, suffix) in enumerate([('CLS', 'row'), ('All queries', 'col')]):
        for group in GROUPS:
            moved = data[group + '__argmax_' + suffix + '_avg'] != data['clean__argmax_' + suffix + '_avg']
            axes[row, 0].plot(x, moved.mean(0) * 100, 'o-', label=group, color=COLORS[group])
            _, _, tv = results[(group, query)]
            axes[row, 2].plot(x, tv.mean(0) * 100, 'o-', label=group, color=COLORS[group])
            axes[row, 2].fill_between(x, tv.min(0) * 100, tv.max(0) * 100,
                                      color=COLORS[group], alpha=.12)
        for group, style in [('patchfool', '-'), ('lavan', '--')]:
            _, region, _ = results[(group, query)]
            for j, (label, color) in enumerate(zip(REGIONS, REGION_COLORS)):
                axes[row, 1].plot(x, region[..., j].mean(0) * 100, style,
                                 color=color, label=f'{group}: {label}')
        axes[row, 0].set_title(f'{query}: top patch changed (n={n})')
        axes[row, 0].set_ylabel('Images with a different argmax (%)')
        axes[row, 0].set_ylim(0, 100)
        axes[row, 1].set_title(f'{query}: where mass is gained / lost (n={nr})')
        axes[row, 1].set_ylabel('Mean attention change (percentage points)')
        axes[row, 1].axhline(0, color='#555', lw=.8)
        axes[row, 2].set_title(f'{query}: amount redistributed (n={nr})')
        axes[row, 2].set_ylabel('Total variation × 100 (mean and min–max)')
        for ax in axes[row]:
            ax.set_xlabel('Transformer block')
            ax.set_xticks(x)
            ax.grid(alpha=.18)
            ax.legend(fontsize=7)
    fig.suptitle('Attention redistribution after attack\n'
                 'Full sample: location changes only • Stored examples: regional and spatial analysis', fontsize=15)
    save_figure(fig, output / 'attention_redistribution_overview')


def rgb(image):
    return np.clip(image.transpose(1, 2, 0) * .5 + .5, 0, 1)


def boxes(ax, coverage, side):
    for idx in np.flatnonzero(coverage > 0):
        row, col = divmod(int(idx), side)
        ax.add_patch(Rectangle((col-.5, row-.5), 1, 1, fill=False,
                               edgecolor='#21d9d0', lw=1.5))


def spatial_panel(data, group, index, layers, output):
    clean = data['clean__row_full_avg_repr'][index]
    attacked = data[group + '__row_full_avg_repr'][index]
    cov = data[group + '__gt_coverage_repr'][index]
    side = int(np.sqrt(len(cov)))
    if side * side != len(cov):
        raise ValueError('Patch tokens must form a square grid')
    original = rgb(data['clean__repr_images'][index])
    changed = rgb(data[group + '__repr_images'][index])
    layer_ids = np.asarray(layers) - 1
    delta = attacked - clean
    mass_max = max(clean[layer_ids, 1:].max(), attacked[layer_ids, 1:].max(), 1e-8) * 100
    delta_max = max(np.abs(delta[layer_ids, 1:]).max(), 1e-8) * 100
    fig, axes = plt.subplots(len(layers)+1, 3, figsize=(12, 3.4*(len(layers)+1)), layout='constrained')
    extent = [-.5, side-.5, side-.5, -.5]
    axes[0, 0].imshow(original, extent=extent)
    boxes(axes[0, 0], cov, side)
    axes[0, 0].set_title('Original image')
    axes[0, 1].imshow(changed, extent=extent)
    boxes(axes[0, 1], cov, side)
    axes[0, 1].set_title('Attacked image (display clipped to RGB range)')
    axes[0, 2].axis('off')
    for ax in axes[0, :2]:
        ax.set_axis_off()
    extent = [-.5, side-.5, side-.5, -.5]
    for r, layer in enumerate(layers, 1):
        l = layer-1
        for col, (values, title) in enumerate([(clean[l, 1:], 'Clean'), (attacked[l, 1:], 'Attack'),
                                               (delta[l, 1:], 'Attack − clean')]):
            ax = axes[r, col]
            if col < 2:
                im = ax.imshow(values.reshape(side, side)*100, cmap='magma',
                               vmin=0, vmax=mass_max, interpolation='nearest')
                label = 'Raw CLS attention (%)'
            else:
                im = ax.imshow(values.reshape(side, side)*100, cmap='RdBu_r',
                               norm=TwoSlopeNorm(vmin=-delta_max, vcenter=0, vmax=delta_max),
                               interpolation='nearest')
                label = 'Attention change (percentage points)'
            boxes(ax, cov, side)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Block {layer} · {title}')
            if col == 2:
                region_gain = delta[l, 1:][cov > 0].sum()*100
                ax.set_xlabel(f'Attack region: {region_gain:+.2f} pp | CLS: {delta[l, 0]*100:+.2f} pp', fontsize=9)
            if r == len(layers):
                fig.colorbar(im, ax=ax, shrink=.8, label=label)
    sample = match_representatives(data)[index]
    fig.suptitle(f'{group} · sample {sample} · CLS attention', fontsize=17)
    save_figure(fig, output / f'{group}_example{index}_spatial_attention_shift')


def match_representatives(data):
    """Recover archive sample offsets only when all groups match uniquely."""
    matches = []
    count = len(data['clean__repr_images'])
    for i in range(count):
        candidates = None
        for group in ('clean', *GROUPS):
            metric = data[group + '__row_avg_repr'][i].max(-1)
            same = np.max(np.abs(data[group + '__top1_row_avg'] - metric), axis=1) < 1e-7
            candidates = same if candidates is None else candidates & same
        idx = np.flatnonzero(candidates)
        matches.append(int(idx[0]) if len(idx) == 1 else 'unknown')
    return matches


def verify_model(data, output):
    """Recompute cached attention using pretrained model, on allocated GPU."""
    import sys
    sys.path.insert(0, str(HERE.parents[2]))
    import torch
    from src.models import load_vit_model
    from attention_lib import collect_full_layer_attn
    if not torch.cuda.is_available():
        raise RuntimeError('--verify_model requires a GPU allocation')
    model = load_vit_model(16, torch.device('cuda'))
    if model is None:
        raise RuntimeError('Pretrained model unavailable')
    records = []
    for group in ('clean', *GROUPS):
        images = data[group + '__repr_images']
        for i in range(len(images)):
            weights = collect_full_layer_attn(model, torch.from_numpy(images[i:i+1]).cuda())
            actual = torch.stack([a.mean(1)[0] for a in weights]).numpy()
            stored = data[group + '__full_avg_repr'][i]
            error = abs(actual - stored)
            records.append([group, i, float(error.max()), float(error.mean()),
                            bool(np.allclose(actual, stored, atol=1e-5, rtol=1e-4)),
                            float(np.mean(abs(images[i]) > 1))])
    with (output / 'gpu_attention_verification.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'stored_example', 'max_abs_error', 'mean_abs_error',
                         'matches_atol1e-5_rtol1e-4', 'pixels_outside_normalized_range'])
        writer.writerows(records)
    if not all(r[4] for r in records):
        raise RuntimeError('Cached attention differs from model recomputation; see verification CSV')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=RESULTS / '00_attention_data_n100.npz')
    parser.add_argument('--output', type=Path, default=RESULTS / '04_attention_redistribution')
    parser.add_argument('--layers', type=int, nargs='+', default=[1, 6, 12], help='1-based blocks for spatial maps')
    parser.add_argument('--verify_model', action='store_true', help='Verify saved attention by GPU forward passes')
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    nl = data['clean__row_full_avg_repr'].shape[1]
    if not args.layers or any(l < 1 or l > nl for l in args.layers):
        parser.error(f'--layers must be between 1 and {nl}')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.verify_model:
        verify_model(data, args.output)
    samples = match_representatives(data)
    results, region_rows, location_rows, token_rows = {}, [], [], []
    nr = len(data['clean__repr_images'])
    for group in GROUPS:
        cov = data[group + '__gt_coverage_repr']
        for query, suffix in [('CLS', 'row'), ('All queries', 'col')]:
            if query == 'CLS':
                clean, attacked = (data[g + '__row_full_avg_repr'] for g in ('clean', group))
            else:
                # Average raw attention over ALL queries, retaining CLS as a key.
                clean, attacked = (data[g + '__full_avg_repr'].mean(axis=-2) for g in ('clean', group))
            delta, region, tv = redistribution(clean, attacked, cov)
            results[(group, query)] = delta, region, tv
            moved = data[group + '__argmax_' + suffix + '_avg'] != data['clean__argmax_' + suffix + '_avg']
            for l in range(nl):
                location_rows.append([group, query, l+1, len(moved), int(moved[:, l].sum()), moved[:, l].mean()])
                for i in range(nr):
                    region_rows.append([group, query, i, l+1, *region[i, l], tv[i, l]])
                    for key in range(delta.shape[-1]):
                        token_rows.append([group, query, i, l+1, key, key-1 if key else '',
                                           bool(cov[i, key-1] > 0) if key else False,
                                           clean[i, l, key], attacked[i, l, key], delta[i, l, key]])
        for index in range(nr):
            spatial_panel(data, group, index, args.layers, args.output)
    overview(data, results, args.output)
    for filename, header, rows in [
        ('argmax_location_changes.csv', ['attack', 'query', 'block', 'n', 'changed_count', 'changed_fraction'], location_rows),
        ('regional_attention_changes.csv', ['attack', 'query', 'stored_example', 'block', 'attacked_delta',
                                           'other_delta', 'cls_delta', 'total_variation'], region_rows),
        ('per_token_attention_changes.csv', ['attack', 'query', 'stored_example', 'block', 'token_index',
                                            'patch_index', 'is_attacked', 'clean', 'attack', 'delta'], token_rows)]:
        with (args.output / filename).open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    (args.output / 'README.md').write_text(
        '# 04 — Attention redistribution\n\n'
        f'Input: `{args.input.resolve()}`. No new attacks/model inference.\n\n'
        f'- Argmax location change: n={len(data["clean__argmax_row_avg"])} paired images. '
        'Argmax is among patch tokens; CLS excluded.\n'
        f'- Spatial maps and regional analysis: {nr} stored examples only, not the full sample.\n'
        '- Raw head-averaged attention retains CLS; attacked + other + CLS deltas sum to zero.\n'
        '- All queries means the raw attention averaged over CLS and patch queries.\n'
        '- Positive delta is gained attention, negative is lost attention; this is not causal flow.\n'
        '- Total variation = 0.5 * sum(abs(attack - clean)), including CLS.\n'
        '- GT support includes any token with coverage > 0; partial tokens are counted as whole tokens.\n'
        '- Overview bands show example min–max, not confidence intervals. CSV values are fractions.\n'
        f'- Stored example IDs map to full-array sample offsets {samples} by unique metric matching across all groups. '
        'These are not ImageNet dataset IDs. Source has no labels or reliable run metadata; '
        'do not join with independently regenerated 03 results or infer success/failure.\n'
        '- Figures are saved as PNG only. Original inputs are not modified.\n')
    print(f'Saved figures and numeric tables: {args.output}')


if __name__ == '__main__':
    main()
