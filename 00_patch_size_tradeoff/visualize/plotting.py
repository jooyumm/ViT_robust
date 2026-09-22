"""
plotting.py — 그래프 그리기 공용 함수
파이썬 모듈명은 숫자로 시작할 수 없어서(01_fig_baseline처럼 import 불가) 다른 시각화
스크립트들이 공유해서 쓰는 함수는 번호 없는 이 파일에 모아둔다.

- 세 그래프(CA/RA/ASR)가 모두 같은 y축 범위를 쓰도록 통일
- 공격별 고유 색상, P=16 단색 / P=32 빗금 / P=8 점무늬
- 막대(mean) 위에 개별 시드 값을 점으로 겹쳐 그려서 다중 시드 스프레드가 직접 보이게 함
"""
import matplotlib
matplotlib.rcParams['hatch.linewidth'] = 1.5
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

from results_io import ATK_LABEL, ATK_ORDER, collect_all_logs

# ── 폰트 크기 설정 ──────────────────────────────────────────────
FS_TITLE_MAIN  = 18   # 전체(결합본) 제목
FS_TITLE_SUB   = 16   # 결합본 서브플롯 제목
FS_AXIS_LABEL  = 12   # 축 레이블 ylabel
FS_TICK        = 11   # 축 눈금
FS_BAR_VALUE   = 12   # 막대 위 숫자
FS_LEGEND      = 11   # legend
FS_XTICK       = 14   # x축 레이블
# ─────────────────────────────────────────────────────────────────

ATK_COLOR = {
    'pgd':        '#2563EB',
    'lavan':      '#16A34A',
    'patch_fool': '#E94B3C',
}

P_HATCH = {8: '...', 16: '', 32: '///'}
P_ALPHA = {8: 0.85, 16: 0.9, 32: 0.7}

SEED_JITTER = 0.028  # 막대 폭 대비 점 좌우 흩뿌림 정도


def mean_std(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def seed_dot_kw(scale=1.0):
    return dict(marker='o', s=34 * scale, facecolor='white', edgecolor='#111827',
               linewidth=1.1 * scale, zorder=5)


def scatter_seeds(ax, x_center, values, rng, scale=1.0):
    """막대 위에 개별 시드 값을 점으로 겹쳐 그린다 (n=1이면 생략 — 막대 자체가 그 값)."""
    if len(values) < 2:
        return
    xs = x_center + rng.uniform(-SEED_JITTER, SEED_JITTER, size=len(values))
    ax.scatter(xs, values, **seed_dot_kw(scale))


def shared_y_max(data, attacks, p_values):
    all_tops = []
    for atk in attacks:
        for p in p_values:
            for metric in ('CA', 'RA', 'ASR'):
                vals = data.get(atk, {}).get(p, {}).get(metric)
                if vals:
                    all_tops.append(max(vals))
    y_max = (max(all_tops) if all_tops else 100) * 1.18
    return min(y_max, 118)  # 100%를 크게 넘는 표시 여유는 불필요


def draw_ca_panel(ax, data, p_values, ref_atk, y_max, rng):
    for pi, p in enumerate(p_values):
        vals = data[ref_atk].get(p, {}).get('CA', [])
        if not vals:
            continue
        m, s = mean_std(vals)
        ax.bar(pi, m, yerr=s if len(vals) > 1 else None, capsize=5,
              color='#64748B',
              hatch=P_HATCH.get(p, ''),
              alpha=P_ALPHA.get(p, 0.85),
              edgecolor='white', linewidth=1.2,
              error_kw=dict(elinewidth=1.3, ecolor='#1F2937'))
        scatter_seeds(ax, pi, vals, rng)
        ax.text(pi, max(vals + [m + s]) + y_max*0.015, f'{m:.1f}%',
               ha='center', va='bottom',
               fontsize=FS_BAR_VALUE, fontweight='bold', color='#374151')
    ax.set_title('Clean Accuracy (baseline)\n(higher is better)',
                 fontsize=FS_TITLE_SUB, fontweight='bold', pad=12)
    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels([f'P={p}\n(tokens={(224//p)**2})' for p in p_values], fontsize=FS_XTICK)
    ax.set_ylabel('%', fontsize=FS_AXIS_LABEL)
    ax.set_ylim(0, y_max)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def draw_metric_panel(ax, data, attacks, p_values, metric, title, direction, y_max, rng):
    x   = np.arange(len(attacks))
    w   = 0.75 / len(p_values)
    off = np.linspace(-(len(p_values)-1)/2, (len(p_values)-1)/2, len(p_values)) * w
    # P값이 3개 이상이면 막대끼리 너무 붙어있어서 "P=" 라벨을 넣으면 서로 겹친다
    # (legend에 이미 hatch로 P가 표시되니) — 2개(예: 01_baseline_p8p16)일 때만 라벨을 얹는다
    show_p_label = len(p_values) <= 2

    for pi, p in enumerate(p_values):
        for ai, atk in enumerate(attacks):
            vals = data.get(atk, {}).get(p, {}).get(metric, [])
            if not vals:
                continue
            m, s = mean_std(vals)
            color = ATK_COLOR.get(atk, '#888')
            xpos = x[ai] + off[pi]
            ax.bar(xpos, m,
                  yerr=s if len(vals) > 1 else None, capsize=4,
                  width=w*0.88,
                  color=color,
                  hatch=P_HATCH.get(p, ''),
                  alpha=P_ALPHA.get(p, 0.85),
                  edgecolor='white', linewidth=1.0,
                  error_kw=dict(elinewidth=1.1, ecolor='#1F2937'))
            scatter_seeds(ax, xpos, vals, rng)
            if m > 0:
                label = f'P={p}\n{m:.1f}' if show_p_label else f'{m:.1f}'
                ax.text(xpos, max(vals + [m + s]) + y_max*0.012,
                       label, ha='center', va='bottom',
                       fontsize=FS_BAR_VALUE, fontweight='bold', color=color,
                       linespacing=1.3)
    ax.set_title(f'{title}\n({direction} is better)',
                fontsize=FS_TITLE_SUB, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([ATK_LABEL.get(a, a) for a in attacks], fontsize=FS_XTICK)
    ax.set_ylabel('%', fontsize=FS_AXIS_LABEL)
    ax.set_ylim(0, y_max)
    ax.tick_params(axis='y', labelsize=FS_TICK)
    ax.axhline(100, color='gray', lw=0.8, ls='--', alpha=0.4)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def legend_handles(attacks, p_values, dot_scale=1.0, include_attacks=True):
    handles = []
    if include_attacks:
        handles += [mpatches.Patch(color=ATK_COLOR.get(a, '#888'), alpha=0.9,
                    label=ATK_LABEL.get(a, a)) for a in attacks]
    handles += [mpatches.Patch(facecolor='#999', hatch=P_HATCH.get(p, ''),
                alpha=P_ALPHA.get(p, 0.85), edgecolor='white',
                label=f'P={p} (tokens={(224//p)**2})') for p in p_values]
    dkw = seed_dot_kw(dot_scale)
    handles += [mlines.Line2D([], [], linestyle='None', label='individual seed',
                marker=dkw['marker'], markersize=7 * dot_scale,
                markerfacecolor=dkw['facecolor'],
                markeredgecolor=dkw['edgecolor'],
                markeredgewidth=dkw['linewidth'])]
    return handles


def plot_results(data, out_path, title_suffix=''):
    """RA/ASR 두 그래프를 한 장에 합친 결합본 (CA는 공격과 무관해서 plot_ca()로 따로 뺌)."""
    attacks  = [a for a in ATK_ORDER if a in data]
    p_values = sorted({p for a in data.values() for p in a.keys()})
    if not attacks or not p_values:
        print("No data"); return

    rng = np.random.default_rng(0)  # 점 흩뿌림용 (실행마다 그림이 안 흔들리게 고정 seed)
    y_max = shared_y_max(data, attacks, p_values)

    fig, (ax_ra, ax_asr) = plt.subplots(1, 2, figsize=(13, 6.5))
    fig.suptitle(f'ViT Adversarial Robustness{title_suffix}',
                fontsize=FS_TITLE_MAIN, fontweight='bold', y=1.02)

    draw_metric_panel(ax_ra,  data, attacks, p_values, 'RA',  'Robust Accuracy',     'higher', y_max, rng)
    draw_metric_panel(ax_asr, data, attacks, p_values, 'ASR', 'Attack Success Rate', 'lower',  y_max, rng)

    fig.legend(handles=legend_handles(attacks, p_values),
              loc='upper right', fontsize=FS_LEGEND,
              bbox_to_anchor=(1.0, 1.0), framealpha=0.92, ncol=1)

    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_ca(data, out_path, title_suffix=''):
    """Clean Accuracy만 단독으로 그린다 — 공격과 무관하게 P(patch size)에만 의존하는 값이라
    모든 실험이 공유하므로 실험 1~3의 결합 그래프에서 분리해서 한 번만 뽑는다."""
    attacks  = [a for a in ATK_ORDER if a in data]
    p_values = sorted({p for a in data.values() for p in a.keys()})
    if not attacks or not p_values:
        print("No data"); return

    rng = np.random.default_rng(0)
    y_max = shared_y_max(data, attacks, p_values)
    ref_atk = attacks[0]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    fig.suptitle(f'Clean Accuracy{title_suffix}',
                fontsize=FS_TITLE_MAIN, fontweight='bold', y=1.0)
    draw_ca_panel(ax, data, p_values, ref_atk, y_max, rng)

    # 나머지 그래프들과 동일한 방식: legend는 그래프 안쪽이 아니라 오른쪽에 전용 여백을
    # 예약해서(tight_layout rect) 그 안에 둔다
    dkw = seed_dot_kw(1.0)
    seed_handle = mlines.Line2D([], [], linestyle='None', label='individual seed',
                                marker=dkw['marker'], markersize=8,
                                markerfacecolor=dkw['facecolor'],
                                markeredgecolor=dkw['edgecolor'],
                                markeredgewidth=dkw['linewidth'])
    fig.legend(handles=[seed_handle], loc='upper right', fontsize=FS_LEGEND,
              bbox_to_anchor=(1.0, 1.0), framealpha=0.92, ncol=1)

    plt.tight_layout(rect=[0, 0, 0.78, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_condition_comparison(conditions, out_path, suptitle, color=None, rotate_xlabels=False):
    """
    baseline vs ablation처럼, (공격, P, tag) 조합 몇 개를 나란히 비교하는 RA/ASR 막대그래프.
    ablation 시각화 스크립트들이 이 함수 하나만 재사용하면 되도록 만든 공용 플로터.

    conditions: [(label, attack, P, tag, hatch, alpha), ...]
                각 조건은 results_io.collect_all_logs(tag=tag)[attack][P]에서 데이터를 가져온다.
    color:      막대 색상 (조건 간 구분은 hatch/alpha로, 색은 통일 — "같은 공격의 변형들"이라는 뜻).
                None이면 conditions의 첫 공격을 기준으로 ATK_COLOR에서 자동 결정
                (01_baseline 그래프와 같은 공격은 항상 같은 색을 쓰도록).
    rotate_xlabels: x축 라벨이 길어서(예: "4 tokens, scattered") 가로로 겹칠 때 True로 주면
                45도 대각선으로 기울여서 표시 (03처럼 라벨이 긴 경우용, 기본은 기존 그대로 가로).
    """
    if color is None:
        color = ATK_COLOR.get(conditions[0][1], '#888')

    tags_needed = sorted({tag for _, _, _, tag, _, _ in conditions})
    sources = {tag: collect_all_logs(tag=tag, verbose=False) for tag in tags_needed}

    rows = []
    for label, attack, p, tag, hatch, alpha in conditions:
        bucket = sources[tag].get(attack, {}).get(p, {})
        ra  = bucket.get('RA', [])
        asr = bucket.get('ASR', [])
        if not ra:
            print(f"[경고] 데이터 없음: {label!r} (attack={attack}, P={p}, tag={tag!r})")
            continue
        rows.append((label, ra, asr, hatch, alpha))
        print(f"{label!r}: RA={ra} ASR={asr}")

    if not rows:
        print("표시할 데이터가 없습니다."); return

    # P=8/P=16 baseline(tag='')이 있으면 그 값을 "비교 기준선"으로 각 패널에 수평선으로 그어서,
    # 나머지 막대들이 그 기준보다 높은지/낮은지 한눈에 보이게 한다.
    # P=8은 점선(막대 hatch가 점무늬인 것과 맞춤), P=16은 대시선(막대가 무지인 것과 맞춤)
    p_refs = {}  # {8: {'RA':.., 'ASR':..}, 16: {...}}
    for target_p in (8, 16):
        for label, attack, p, tag, _, _ in conditions:
            if p == target_p and tag == '':
                bucket = sources[tag].get(attack, {}).get(p, {})
                if bucket.get('RA'):
                    p_refs[target_p] = {'RA': mean_std(bucket['RA'])[0], 'ASR': mean_std(bucket['ASR'])[0]}
                break
    P_REF_LINESTYLE = {8: ':', 16: '--'}

    rng = np.random.default_rng(0)
    fig, (ax_ra, ax_asr) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(suptitle, fontsize=FS_TITLE_MAIN, fontweight='bold', y=1.02)

    all_vals = [v for _, ra, asr, _, _ in rows for v in ra + asr]
    y_max = max(all_vals) * 1.25 if all_vals else 100

    def draw(ax, idx, title, direction):
        metric = 'RA' if idx == 0 else 'ASR'
        for i, (label, ra, asr, hatch, alpha) in enumerate(rows):
            vals = ra if idx == 0 else asr
            m, s = mean_std(vals)
            ax.bar(i, m, yerr=s if len(vals) > 1 else None, capsize=5,
                  color=color, hatch=hatch, alpha=alpha,
                  edgecolor='white', linewidth=1.3,
                  error_kw=dict(elinewidth=1.3, ecolor='#1F2937'))
            scatter_seeds(ax, i, vals, rng, scale=1.3)
            ax.text(i, max(vals + [m + s]) + y_max * 0.02, f'{m:.1f}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold', color=color)
        for ref_p, ref_vals in p_refs.items():
            ax.axhline(ref_vals[metric], color=color, linestyle=P_REF_LINESTYLE[ref_p],
                      linewidth=1.3, alpha=0.55, zorder=1)
        ax.set_title(f'{title}\n({direction} is better)', fontsize=16, fontweight='bold', pad=12)
        ax.set_xticks(range(len(rows)))
        if rotate_xlabels:
            ax.set_xticklabels([r[0] for r in rows], fontsize=13,
                               rotation=45, ha='right', rotation_mode='anchor')
        else:
            ax.set_xticklabels([r[0] for r in rows], fontsize=13)
        ax.set_ylabel('%', fontsize=13)
        ax.set_ylim(0, y_max)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    draw(ax_ra,  0, 'Robust Accuracy',    'higher')
    draw(ax_asr, 1, 'Attack Success Rate', 'lower')

    dkw = seed_dot_kw(1.3)
    seed_handle = mlines.Line2D([], [], linestyle='None', label='individual seed',
                                marker=dkw['marker'], markersize=9,
                                markerfacecolor=dkw['facecolor'],
                                markeredgecolor=dkw['edgecolor'],
                                markeredgewidth=dkw['linewidth'])
    # 01(plot_results)과 동일한 방식: legend를 그래프 안쪽에 끼워넣지 않고, 오른쪽에
    # 전용 여백을 예약해서(tight_layout rect) 그 안에 둔다 — 제목과도 안 겹치고 막대와도 안 겹침
    fig.legend(handles=[seed_handle], loc='upper right', fontsize=FS_LEGEND,
              bbox_to_anchor=(1.0, 1.0), framealpha=0.92, ncol=1)

    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def _shade(hex_color, factor):
    """factor>0: 밝게, factor<0: 어둡게 (대략 -1~1 범위)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    if factor >= 0:
        r, g, b = (int(c + (255 - c) * factor) for c in (r, g, b))
    else:
        r, g, b = (int(c * (1 + factor)) for c in (r, g, b))
    return f'#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}'


def p_color_shades(attack, p_values):
    """
    공격 하나(예: lavan)의 기본색을 P별로 밝기만 다르게 만든다
    (작은 P=밝게, 큰 P=어둡게) — 같은 공격의 P별 변형이라는 걸 색으로도 드러냄.
    """
    base = ATK_COLOR.get(attack, '#888888')
    p_values = sorted(p_values)
    n = len(p_values)
    factors = np.linspace(0.55, -0.35, n) if n > 1 else [0.0]
    return {p: _shade(base, f) for p, f in zip(p_values, factors)}


def plot_sweep(x_values, data_by_p, out_path, suptitle, xlabel, attack='lavan'):
    """
    x축이 연속값(예: 공격 면적 비율)인 스윕 결과를 P별 선 그래프로 그린다 (RA/ASR 두 패널).

    x_values  : 정렬된 x축 값 리스트 (예: [0.5, 1, 2, 5, 10])
    data_by_p : {P: {'RA': [[seed값들] per x], 'ASR': [[seed값들] per x]}}
                x_values와 같은 길이 순서로 각 x값에서의 시드별 값 리스트를 담는다.
                해당 x에 데이터가 없으면 빈 리스트.
    """
    p_values = sorted(data_by_p.keys())
    colors = p_color_shades(attack, p_values)

    fig, (ax_ra, ax_asr) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle(suptitle, fontsize=FS_TITLE_MAIN, fontweight='bold', y=0.99)

    def draw(ax, metric, title, direction):
        for p in p_values:
            xs, means, stds = [], [], []
            for x, vals in zip(x_values, data_by_p[p][metric]):
                if not vals:
                    continue
                m, s = mean_std(vals)
                xs.append(x); means.append(m); stds.append(s)
            if not xs:
                continue
            means = np.array(means); stds = np.array(stds)
            ax.plot(xs, means, marker='o', color=colors[p], linewidth=2.2,
                   markersize=7, label=f'P={p}')
            ax.fill_between(xs, means - stds, means + stds, color=colors[p], alpha=0.15)
        ax.set_title(f'{title}\n({direction} is better)', fontsize=FS_TITLE_SUB, fontweight='bold', pad=12)
        ax.set_xlabel(xlabel, fontsize=FS_AXIS_LABEL)
        ax.set_ylabel('%', fontsize=FS_AXIS_LABEL)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=FS_LEGEND, framealpha=0.92, loc='upper right')

    draw(ax_ra,  'RA',  'Robust Accuracy',    'higher')
    draw(ax_asr, 'ASR', 'Attack Success Rate', 'lower')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()