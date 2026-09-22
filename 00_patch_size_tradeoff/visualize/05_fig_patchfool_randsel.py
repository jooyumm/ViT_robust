"""
05_fig_patchfool_randsel.py — 실험 6 시각화: PatchFool 토큰 선택 방식(attention vs random) ablation 비교
scripts/05_patchfool_randsel.sh 결과를 시각화한다. 조건 정의는 conditions.py에 있음
(같은 데이터를 표로 보고 싶으면 05_tbl_patchfool_randsel.py).

실제 그래프 로직은 plotting.py의 plot_condition_comparison()을 재사용한다.

사용법:
  python visualize/05_fig_patchfool_randsel.py
"""
import os
from results_io import FIG_DIR
from plotting import plot_condition_comparison
from conditions import PATCHFOOL_RANDSEL, PATCHFOOL_RANDSEL_TITLE


def main():
    out = os.path.join(FIG_DIR, '05_patchfool_randsel.png')
    plot_condition_comparison(PATCHFOOL_RANDSEL, out, PATCHFOOL_RANDSEL_TITLE)


if __name__ == '__main__':
    main()