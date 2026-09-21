"""
experiments/characterization/plotting.py — 01_concentration.py와 02_sink_position.py가
공유하는 그림 함수. 둘 다 00_extract_attention.py가 저장한 npz만 읽고 다시 attention을
계산하지 않으므로, 여기 함수들도 전부 이미 계산된 배열(분포/지표)만 받는다.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attention_lib import UNIFORM, PPL

GROUP_COLORS = {'clean': 'tab:blue', 'patchfool': 'tab:red', 'lavan': 'tab:green'}
GROUP_LABELS = {'clean': 'Clean', 'patchfool': 'PatchFool', 'lavan': 'LaVAN'}
GROUPS = ('clean', 'patchfool', 'lavan')


def plot_heatmap_grid(repr_by_group, img_idx, n_layers, out_path):
    fig, axes = plt.subplots(len(GROUPS), n_layers,
                              figsize=(1.7 * n_layers, 1.7 * len(GROUPS) + 0.5))
    for gi, g in enumerate(GROUPS):
        row_avg = repr_by_group[g]['row_avg'][img_idx]   # (n_layers, 196)
        for L in range(n_layers):
            ax = axes[gi, L]
            grid = row_avg[L].reshape(PPL, PPL)
            ax.imshow(grid, cmap='viridis')
            ax.set_xticks([]); ax.set_yticks([])
            if gi == 0:
                ax.set_title(f'L={L+1}', fontsize=9)
            if L == 0:
                ax.set_ylabel(GROUP_LABELS[g], fontsize=10)
    fig.suptitle(f'Head-averaged CLS->patch attention, image {img_idx} '
                 f'(clean vs PatchFool vs LaVAN, all layers)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sorted_mass_by_layer(repr_by_group, img_idx, n_layers, out_path):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for L in range(n_layers):
        ax = axes[L // 4, L % 4]
        for g in GROUPS:
            dist = repr_by_group[g]['row_avg'][img_idx, L]
            sorted_dist = np.sort(dist)[::-1]
            ax.plot(np.arange(1, len(sorted_dist) + 1), sorted_dist,
                    color=GROUP_COLORS[g], label=GROUP_LABELS[g], lw=1.5)
        ax.axhline(UNIFORM, color='gray', ls=':', lw=1)
        ax.set_title(f'L={L+1}', fontsize=10)
        ax.set_xlabel('patch rank (sorted desc.)')
        ax.set_ylabel('attention mass')
        if L == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f'Sorted CLS->patch attention mass per layer, image {img_idx} '
                 '(concentrated attack = tall left bar + flat tail)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_head_variability(repr_by_group, img_idx, layers, out_path):
    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 4.5))
    if len(layers) == 1:
        axes = [axes]
    for ai, L in enumerate(layers):
        ax = axes[ai]
        n_heads = repr_by_group['clean']['row_ph'].shape[2]
        x = np.arange(n_heads)
        width = 0.25
        for gi, g in enumerate(GROUPS):
            dist_ph = repr_by_group[g]['row_ph'][img_idx, L - 1]     # (heads, 196)
            top1_ph = dist_ph.max(axis=-1)
            ax.bar(x + (gi - 1) * width, top1_ph, width=width, color=GROUP_COLORS[g],
                   label=GROUP_LABELS[g])
        ax.axhline(UNIFORM, color='gray', ls=':', lw=1, label='uniform (1/196)')
        ax.set_xlabel('head')
        ax.set_ylabel('top-1 mass (CLS row)')
        ax.set_title(f'L={L}')
        ax.set_xticks(x)
        if ai == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f'Per-head top-1 mass, image {img_idx} (does concentration hide in specific heads?)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_histograms(all_metrics, metric_key, n_layers, title, out_path):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for L in range(n_layers):
        ax = axes[L // 4, L % 4]
        for g in GROUPS:
            vals = all_metrics[g][metric_key][:, L]
            ax.hist(vals, bins=20, alpha=0.5, color=GROUP_COLORS[g], label=GROUP_LABELS[g], density=True)
        ax.set_title(f'L={L+1}', fontsize=10)
        if L == 0:
            ax.legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_layer_comparison(all_metrics, metric_key, n_layers, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    layers = np.arange(1, n_layers + 1)
    for g in GROUPS:
        vals = all_metrics[g][metric_key]                    # (B, n_layers)
        med = np.median(vals, axis=0)
        q1 = np.percentile(vals, 25, axis=0)
        q3 = np.percentile(vals, 75, axis=0)
        ax.plot(layers, med, marker='o', color=GROUP_COLORS[g], label=GROUP_LABELS[g])
        ax.fill_between(layers, q1, q3, color=GROUP_COLORS[g], alpha=0.15)
    ax.axhline(UNIFORM, color='gray', ls=':', lw=1, label='uniform (1/196)')
    ax.set_xlabel('Layer L')
    ax.set_ylabel(ylabel)
    ax.set_xticks(layers)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
