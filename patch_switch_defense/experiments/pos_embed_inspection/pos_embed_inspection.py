"""
experiments/pos_embed_inspection/pos_embed_inspection.py — [탐색적] P8의 position embedding
테이블이 실제로 존재하는지, 진짜 숫자로 직접 확인한다. GPU 계산 없음 — 모델 파라미터를
읽기만 해서 CPU에서도 바로 끝난다.

"embedding dim"이 뭔지
----
patch_embed 레이어는 이미지의 각 위치(패치)를 **768개의 숫자로 이루어진 리스트(벡터)** 하나로
바꾼다. 이 768이라는 길이가 "embedding dim"이다 — 공간적인 의미는 전혀 없고, 그냥 "그 벡터
안의 몇 번째 숫자인가"라는 인덱스일 뿐이다. pos_embed 테이블은 위치마다(P8은 784개 위치) 이
768개짜리 벡터를 하나씩 따로 갖고 있는, "위치 개수 x 768"짜리 큰 표다.

사용법:
  python pos_embed_inspection.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, 'experiments')):
    ROOT = os.path.dirname(ROOT)
RESULTS = HERE
SHARED_SRC_ROOT = os.path.dirname(ROOT)  # ViT_robust -- shared src/ (models/dataset/attacks) across sibling projects
sys.path.insert(0, SHARED_SRC_ROOT)
sys.path.insert(0, ROOT)

from src.models import get_device, load_vit_model


def main():
    device = get_device()
    model8 = load_vit_model(8, device); model8.eval()

    pos8_full = model8.pos_embed[0].detach().cpu().numpy()  # (785, 768): CLS + 784 patch positions
    pos8 = pos8_full[1:]  # (784, 768) — patch position 0~783만 (CLS 제외)

    print(f"model8.pos_embed 전체 shape: {tuple(model8.pos_embed.shape)}")
    print(f"  = (배치축 1개, CLS 1개 + 패치 위치 784개, embedding dim 768개)")
    print(f"\n표로 치면: 행(row) = 784개의 '위치', 열(column) = 그 위치를 나타내는 768개의 숫자.\n")

    # 진짜 숫자로 된 작은 표 — 처음 10개 위치 x 처음 10개 dim만 그대로 출력
    n_rows, n_cols = 10, 10
    sub = pos8[:n_rows, :n_cols]
    print(f"실제 표 일부 (patch position 0~{n_rows-1}, embedding dim 0~{n_cols-1}):\n")
    header = "position | " + " ".join(f"dim{j:<5}" for j in range(n_cols))
    print(header)
    print("-" * len(header))
    for i in range(n_rows):
        row = " ".join(f"{sub[i, j]:+.3f}" for j in range(n_cols))
        print(f"pos {i:<4} | {row}")

    # 실제 숫자가 박힌 표 이미지 (히트맵 아님 -- 진짜 grid + 셀 안에 숫자)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis('off')
    col_labels = [f'dim{j}' for j in range(n_cols)]
    row_labels = [f'pos{i}' for i in range(n_rows)]
    cell_text = [[f'{sub[i, j]:+.3f}' for j in range(n_cols)] for i in range(n_rows)]
    tbl = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#E5E7EB')
    ax.set_title(f'model8.pos_embed -- actual values, patch position 0-{n_rows-1} x '
                 f'embedding dim 0-{n_cols-1}\n(out of the full 784 x 768 table)',
                 fontsize=12, fontweight='bold', pad=14)
    fig.tight_layout()

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, 'pos_embed_exists.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f"\nSaved: {out_path}")

    np.savez(os.path.join(RESULTS, 'pos_embed_inspection.npz'), pos8=pos8)
    print(f"Saved: {os.path.join(RESULTS, 'pos_embed_inspection.npz')}")


if __name__ == '__main__':
    main()
