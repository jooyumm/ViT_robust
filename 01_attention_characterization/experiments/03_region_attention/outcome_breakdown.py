"""고정 500장에서 PatchFool 공격 결과로 나올 수 있는 모든 경우의 수를 2x2 표로 만든다.

clean 정답 여부 x 공격 후 정답 여부로 배타적 4개 case를 나눈다. 숫자는
data/gt/imagenet500.json이 고정한 같은 500장에서 label·clean__prediction·
patchfool__prediction만으로 계산하며, attention이나 탐지기 출력은 쓰지 않는다.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cohort import DEFAULT_DATASET, validate_default_manifest


def load_outcomes():
    """고정 500장 전체의 label/clean 예측/공격 후 예측을 읽는다."""
    manifest = json.loads((DEFAULT_DATASET / 'manifest.json').read_text())
    validate_default_manifest(manifest)
    n = len(manifest['sample_indices'])
    label = np.empty(n, dtype=np.int64)
    clean_pred = np.empty(n, dtype=np.int64)
    attack_pred = np.empty(n, dtype=np.int64)
    offset = 0
    for chunk in manifest['chunks']:
        with np.load(DEFAULT_DATASET / chunk['path']) as z:
            count = chunk['n']
            label[offset:offset + count] = z['label']
            clean_pred[offset:offset + count] = z['clean__prediction']
            attack_pred[offset:offset + count] = z['patchfool__prediction']
        offset += count
    return label, clean_pred, attack_pred, n


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE.parents[1]/'results/03_region_attention/patchfool/00_outcome_breakdown.png')
    out_path=parser.parse_args().output
    if out_path.exists():
        raise FileExistsError(out_path)
    label, clean_pred, attack_pred, n = load_outcomes()
    clean_ok = clean_pred == label
    attack_ok = attack_pred == label

    # (clean, attack) x (correct, fail) -- 4개 배타적 case, 행으로 나열
    cases = [
        (1, True, True),
        (2, True, False),
        (3, False, True),
        (4, False, False),
    ]
    counts = {(c_ok, a_ok): int(((clean_ok == c_ok) & (attack_ok == a_ok)).sum()) for _, c_ok, a_ok in cases}
    assert sum(counts.values()) == n

    row_labels = [f'Case {case}' for case, _, _ in cases]
    # "Attack: Fail"은 "공격 시도가 실패했다"로 오독되기 쉬워, 모델 예측 기준임을 분명히 하는 라벨로 바꾼다.
    col_labels = ['Clean prediction', 'Under-attack prediction', 'Count']
    cell_text = [
        ['Correct' if c_ok else 'Wrong', 'Correct' if a_ok else 'Wrong', str(counts[(c_ok, a_ok)])]
        for _, c_ok, a_ok in cases
    ]

    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    ax.axis('off')
    ax.set_title(f'PatchFool attack outcome, fixed n={n}', fontsize=13, fontweight='bold', pad=14)
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                    loc='center', cellLoc='center', colWidths=[0.3, 0.37, 0.2])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.4)

    n_col = len(col_labels)
    for j in range(n_col):
        cell = tbl[(0, j)]
        cell.set_facecolor('#E5E7EB'); cell.set_edgecolor('black'); cell.set_linewidth(0.6)
        cell.set_text_props(color='black', fontweight='bold')
    for i in range(len(row_labels)):
        cell = tbl[(i + 1, -1)]
        cell.set_facecolor('#E5E7EB'); cell.set_edgecolor('black'); cell.set_linewidth(0.6)
        cell.set_text_props(color='black', fontweight='bold')
    for i in range(len(row_labels)):
        for j in range(n_col):
            c = tbl[(i + 1, j)]
            c.set_facecolor('white'); c.set_edgecolor('black'); c.set_linewidth(0.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('완료:', out_path, flush=True)
    for case, c_ok, a_ok in cases:
        print(f'Case {case}: Clean={"Correct" if c_ok else "Wrong"}, Under-attack={"Correct" if a_ok else "Wrong"} -> {counts[(c_ok, a_ok)]}', flush=True)


if __name__ == '__main__':
    main()
