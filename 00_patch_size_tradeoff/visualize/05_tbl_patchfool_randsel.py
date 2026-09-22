"""
05_tbl_patchfool_randsel.py — 실험 6 결과 표: PatchFool 토큰 선택 방식 ablation
05_fig_patchfool_randsel.py와 같은 데이터를 표로 보여준다.

사용법:
  python visualize/05_tbl_patchfool_randsel.py
"""
import os
from results_io import TABLE_DIR
from tables import make_condition_table
from conditions import PATCHFOOL_RANDSEL, PATCHFOOL_RANDSEL_TITLE


def main():
    out = os.path.join(TABLE_DIR, '05_patchfool_randsel_table.png')
    make_condition_table(PATCHFOOL_RANDSEL, out, PATCHFOOL_RANDSEL_TITLE)


if __name__ == '__main__':
    main()