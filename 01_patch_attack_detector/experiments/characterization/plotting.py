"""01_concentration.py/03_token_bars.py가 쓰는 그림 함수들."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from attention_lib import UNIFORM

GROUP_COLORS = {'clean': 'tab:blue', 'patchfool': 'tab:red', 'lavan': 'tab:green'}
GROUP_LABELS = {'clean': 'Clean', 'patchfool': 'PatchFool', 'lavan': 'LaVAN'}
GROUPS = ('clean', 'patchfool', 'lavan')


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


def plot_token_attention_line(repr_by_group, img_idx, layer, out_path, spike_threshold=0.3):
    """CLS가 197개 토큰에 주는 attention을 레이어 1개/이미지 1개로 그린다. 최댓값이
    spike_threshold(절대값)를 넘으면 그 지점에 토큰 번호를 라벨로 붙인다."""
    if 'row_full_avg' not in repr_by_group['clean']:
        return False
    n_tokens = repr_by_group['clean']['row_full_avg'].shape[-1]   # 197
    x = np.arange(n_tokens)

    fig, ax = plt.subplots(figsize=(9, 5))
    for g in GROUPS:
        vals = repr_by_group[g]['row_full_avg'][img_idx, layer - 1]   # (197,)
        ax.plot(x, vals, color=GROUP_COLORS[g], label=GROUP_LABELS[g], lw=1.3)

        peak_idx = int(np.argmax(vals))
        if vals[peak_idx] > spike_threshold:
            label = 'CLS' if peak_idx == 0 else f'patch {peak_idx}'
            ax.annotate(label, xy=(peak_idx, vals[peak_idx]), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=9,
                        color=GROUP_COLORS[g], fontweight='bold')

    ax.set_xlim(0, n_tokens - 1)
    ax.set_xlabel('token index (0=CLS, 1-196=patch)')
    ax.set_ylabel('attention')
    ax.legend()
    ax.set_title(f'Layer {layer} — attention by token')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True
