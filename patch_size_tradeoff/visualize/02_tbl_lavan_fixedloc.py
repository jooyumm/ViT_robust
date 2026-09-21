"""
02_tbl_lavan_fixedloc.py — 실험 7 결과 표: LaVAN 공격 위치(랜덤 vs 고정) ablation
02_fig_lavan_fixedloc.py와 같은 데이터를 표로 보여준다.

사용법:
  python visualize/02_tbl_lavan_fixedloc.py
"""
import os
from results_io import TABLE_DIR
from tables import make_condition_table
from conditions import LAVAN_FIXEDLOC, LAVAN_FIXEDLOC_TITLE


def main():
    out = os.path.join(TABLE_DIR, '02_lavan_fixedloc_table.png')
    make_condition_table(LAVAN_FIXEDLOC, out, LAVAN_FIXEDLOC_TITLE)


if __name__ == '__main__':
    main()