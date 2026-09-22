"""
tables.py — 표 생성 공용 함수 (01_tbl_baseline.py 등에서 분리)
파이썬 모듈명은 숫자로 시작할 수 없어서(01_tbl_baseline처럼 import 불가) 표를 만드는
시각화 스크립트들이 공유하는 함수는 번호 없는 이 파일에 모아둔다.

PNG(흑백 minimal 스타일, 논문 그림으로 바로 캡처해도 되는 형태)만 만든다 — LaTeX(.tex)는
당장 필요 없어서 뺐다. 필요해지면 make_latex류 함수를 다시 추가하면 된다.
"""
import matplotlib.pyplot as plt

from plotting import mean_std
from results_io import collect_all_logs, ATK_LABEL, ATK_ORDER, P_ORDER


def fmt(values, decimals=1, show_n=True):
    if not values:
        return 'N/A'
    m, s = mean_std(values)
    n = len(values)
    base = f'{m:.{decimals}f}±{s:.{decimals}f}' if n > 1 else f'{m:.{decimals}f}'
    return f'{base} (n={n})' if show_n else base


def _render_table(row_labels, col_labels, cell_text, out_path, title, footer=None):
    """행/열 레이블과 셀 텍스트만 넘기면 흑백 minimal 스타일로 그려서 저장한다."""
    n_col = len(col_labels)
    fig_w = 1.8 + 1.7 * n_col
    fig_h = 1.6 + 0.65 * len(row_labels) + (0.3 if footer else 0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.04)

    tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.2)

    # 흑백 minimal: 헤더(행/열 모두)만 옅은 회색, 나머지는 흰 배경 + 검은 글씨 + 얇은 테두리
    for j in range(n_col):
        cell = tbl[(0, j)]
        cell.set_facecolor('#E5E7EB'); cell.set_edgecolor('black'); cell.set_linewidth(0.6)
        cell.set_text_props(color='black', fontweight='bold')
    for i in range(len(row_labels)):
        cell = tbl[(i+1, -1)]
        cell.set_facecolor('#E5E7EB'); cell.set_edgecolor('black'); cell.set_linewidth(0.6)
        cell.set_text_props(color='black', fontweight='bold')
    for i in range(len(row_labels)):
        for j in range(n_col):
            c = tbl[(i+1, j)]
            c.set_facecolor('white'); c.set_edgecolor('black'); c.set_linewidth(0.6)

    if footer:
        fig.text(0.5, 0.02, footer, ha='center', fontsize=10,
                 color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


def make_baseline_table(data, out_path, title_suffix=''):
    """attack x P 그리드 표 (01_fig_baseline.py와 같은 데이터)."""
    attacks  = [a for a in ATK_ORDER if a in data]
    p_values = [p for p in P_ORDER if any(p in data.get(a, {}) for a in attacks)]
    if not attacks or not p_values:
        print("No data"); return

    col_labels = [f'RA\nP={p}' for p in p_values] + [f'ASR\nP={p}' for p in p_values]
    col_info   = [('RA', p) for p in p_values] + [('ASR', p) for p in p_values]
    row_labels = [ATK_LABEL.get(a, a) for a in attacks]
    cell_text  = [[fmt(data.get(atk, {}).get(p, {}).get(metric, []))
                  for metric, p in col_info] for atk in attacks]

    ref_atk = attacks[0]
    ca_str = '   '.join(f'P={p}: {fmt(data[ref_atk].get(p, {}).get("CA", []))}%' for p in p_values)
    footer = f'Clean Accuracy (attack-independent, mean±std, n=seed count) — {ca_str}'

    _render_table(row_labels, col_labels, cell_text, out_path,
                 f'ViT Adversarial Robustness — Results{title_suffix}', footer)


def make_condition_table(conditions, out_path, title):
    """
    baseline vs ablation처럼 (공격, P, tag) 조합 몇 개를 나란히 비교하는 표.
    plot_condition_comparison()과 같은 CONDITIONS 포맷을 그대로 받는다 —
    ablation 그래프 스크립트 옆에 표 버전을 몇 줄만 추가하면 되도록.

    conditions: [(label, attack, P, tag, hatch, alpha), ...] (hatch/alpha는 그래프 전용, 표에서는 무시)
    """
    tags_needed = sorted({tag for _, _, _, tag, _, _ in conditions})
    sources = {tag: collect_all_logs(tag=tag, verbose=False) for tag in tags_needed}

    row_labels, cell_text = [], []
    for label, attack, p, tag, _, _ in conditions:
        bucket = sources[tag].get(attack, {}).get(p, {})
        ra, asr = bucket.get('RA', []), bucket.get('ASR', [])
        if not ra:
            print(f"[경고] 데이터 없음: {label!r} (attack={attack}, P={p}, tag={tag!r})")
            continue
        row_labels.append(label.replace('\n', ' '))
        cell_text.append([fmt(ra), fmt(asr)])

    if not row_labels:
        print("표시할 데이터가 없습니다."); return

    _render_table(row_labels, ['RA (%)', 'ASR (%)'], cell_text, out_path, title)


def make_sweep_table(x_values, x_labels, data_by_p, out_path, title):
    """
    plot_sweep()과 같은 데이터(P x 연속 x값)를 그리드 표로 보여준다.

    x_values  : 정렬된 x축 값 리스트
    x_labels  : x_values와 같은 길이의 표시용 라벨 (예: ['0.5%', '1%', ...])
    data_by_p : {P: {'RA': [[seed값들] per x], 'ASR': [[seed값들] per x]}}
    """
    p_values = sorted(data_by_p.keys())
    if not p_values:
        print("No data"); return

    col_labels = [f'RA\n{x}' for x in x_labels] + [f'ASR\n{x}' for x in x_labels]
    row_labels = [f'P={p}' for p in p_values]
    cell_text  = []
    for p in p_values:
        row = [fmt(vals) for vals in data_by_p[p]['RA']] + \
              [fmt(vals) for vals in data_by_p[p]['ASR']]
        cell_text.append(row)

    _render_table(row_labels, col_labels, cell_text, out_path, title)