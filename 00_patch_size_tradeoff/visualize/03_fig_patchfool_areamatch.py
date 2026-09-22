"""
03_fig_patchfool_areamatch.py — 실험 4+5 시각화: PatchFool 면적 통제(area-matched) ablation 비교
scripts/03_patchfool_areamatch_scattered.sh(실험4) + 04_patchfool_areamatch_contiguous.sh(실험5)
결과를 함께 보여준다. 조건 정의는 conditions.py에 있음
(같은 데이터를 표로 보고 싶으면 03_tbl_patchfool_areamatch.py).

실제 그래프 로직은 plotting.py의 plot_condition_comparison()을 재사용한다.

사용법:
  python visualize/03_fig_patchfool_areamatch.py
"""
import os
from results_io import FIG_DIR
from plotting import plot_condition_comparison
from conditions import PATCHFOOL_AREAMATCH, PATCHFOOL_AREAMATCH_TITLE


def main():
    out = os.path.join(FIG_DIR, '03_patchfool_areamatch.png')
    plot_condition_comparison(PATCHFOOL_AREAMATCH, out, PATCHFOOL_AREAMATCH_TITLE,
                               rotate_xlabels=True)


if __name__ == '__main__':
    main()