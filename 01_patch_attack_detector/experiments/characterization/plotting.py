"""
experiments/characterization/plotting.py — 01_concentration.py/02_sink_position.py/
03_token_bars.py가 공유하는 그림 함수. 셋 다 00_extract_attention.py가 저장한 npz만 읽고
다시 attention을 계산하지 않으므로, 여기 함수들도 전부 이미 계산된 배열(분포/지표)만 받는다.
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


def plot_token_attention_line(repr_by_group, img_idx, layer, out_path, spike_threshold=0.3):
    """CLS가 197개 토큰(0=CLS 자기 자신, 1~196=patch) 각각에 주는 raw attention을 **레이어
    하나, 이미지 하나**에 대해 선 그래프로 그린다 — "정말 튀는 토큰이 있는가"를 자연
    순서(정렬 안 함)로 확인하는 예시 한 장. layer는 1-indexed(예: 12). x축은 0(CLS)부터
    196(마지막 patch)까지 딱 맞춘다(196개 patch + CLS 1개 = 197토큰, 인덱스 0~196).
    각 곡선의 최댓값이 spike_threshold(절대값, 기본 0.3)를 넘으면 "튀는 값"으로 보고 그
    지점 꼭지에 몇 번 토큰인지(patch 번호, CLS면 'CLS') 적는다 — 안 튀는 곡선은 라벨을 안
    단다. 중앙값 대비 배수가 아니라 절대 기준을 쓰는 이유: 나머지 196개가 거의 0에 가까운
    분포라(baseline이 원래 평평함) 상대 기준(예: 중앙값의 n배)을 쓰면 그냥 그 곡선에서
    "제일 높은 점"마다 다 튀는 값으로 잡혀버린다(실제로 clean의 자연스러운 국소 최댓값도
    중앙값의 수십 배가 나옴). 0.3은 이 실험에서 실제로 PatchFool급 왜곡(예: 0.63)과
    clean/LaVAN의 평범한 국소 최댓값(0.1~0.15대)을 가르는 데 쓴 값이다.
    plot_sorted_mass_by_layer(정렬함, patch 196개만)와는 상호보완적: 여기서는 (a) 순서를
    유지해서 특정 인덱스가 튀는지 그대로 보고, (b) CLS 자기 자신도 197번째 지점으로 포함한다.
    repr_by_group[g]['row_full_avg']가 없으면(구버전 npz) 조용히 건너뛴다."""
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


def plot_column_attention_line(repr_by_group, img_idx, layer, target_token, out_path):
    """plot_token_attention_line을 뒤집은 그림: "CLS가 어디를 보는가"가 아니라 "**전체
    197개 쿼리 토큰 각각이** target_token 하나를 얼마나 보는가"를 그린다. x축은 쿼리
    토큰 인덱스(0=CLS, 1~196=patch), y축은 그 쿼리가 target_token에게 주는 raw attention.
    03_token_bars.py가 CLS 관점에서 찾은 스파이크(예: L=12에서 patch 17)가 진짜 "다들 보는
    sink"라면, 이 그림에서도 대부분의 쿼리 곡선이 target_token 지점에서 높아야 한다 — 반대로
    CLS 혼자만 튀고 나머지 196개 쿼리는 낮으면, "sink"가 아니라 "CLS만의 특이 현상"이라는
    뜻. repr_by_group[g]['full_avg']가 없으면(구버전 npz) 조용히 건너뛴다."""
    if 'full_avg' not in repr_by_group['clean']:
        return False
    full_avg = repr_by_group['clean']['full_avg']
    n_tokens = full_avg.shape[-1]   # 197
    x = np.arange(n_tokens)

    fig, ax = plt.subplots(figsize=(9, 5))
    for g in GROUPS:
        mat = repr_by_group[g]['full_avg'][img_idx, layer - 1]   # (197, 197) query x key
        vals = mat[:, target_token]                               # (197,) -- 모든 쿼리 -> target_token
        ax.plot(x, vals, color=GROUP_COLORS[g], label=GROUP_LABELS[g], lw=1.3)

    label = 'CLS' if target_token == 0 else f'patch {target_token}'
    ax.set_xlim(0, n_tokens - 1)
    ax.set_xlabel('query token index (0=CLS, 1-196=patch)')
    ax.set_ylabel(f'attention paid to {label}')
    ax.legend()
    ax.set_title(f'Layer {layer} — who attends to {label}?')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return True
