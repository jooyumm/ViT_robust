"""
02_fig_lavan_fixedloc.py — 실험 7 시각화: LaVAN 공격 위치(랜덤 vs 고정) ablation 비교
scripts/02_lavan_fixedloc.sh 결과를 시각화한다. 조건 정의는 conditions.py에 있음
(같은 데이터를 표로 보고 싶으면 02_tbl_lavan_fixedloc.py).

실제 그래프 로직은 plotting.py의 plot_condition_comparison()을 재사용한다.

사용법:
  python visualize/02_fig_lavan_fixedloc.py
"""
import os
from results_io import FIG_DIR
from plotting import plot_condition_comparison
from conditions import LAVAN_FIXEDLOC, LAVAN_FIXEDLOC_TITLE


def main():
    out = os.path.join(FIG_DIR, '02_lavan_fixedloc.png')
    plot_condition_comparison(LAVAN_FIXEDLOC, out, LAVAN_FIXEDLOC_TITLE)


if __name__ == '__main__':
    main()