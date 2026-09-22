"""
03_tbl_patchfool_areamatch.py — 실험 4+5 결과 표: PatchFool 면적 통제(area-matched) ablation
03_fig_patchfool_areamatch.py와 같은 데이터를 표로 보여준다.

사용법:
  python visualize/03_tbl_patchfool_areamatch.py
"""
import os
from results_io import TABLE_DIR
from tables import make_condition_table
from conditions import PATCHFOOL_AREAMATCH, PATCHFOOL_AREAMATCH_TITLE


def main():
    out = os.path.join(TABLE_DIR, '03_patchfool_areamatch_table.png')
    make_condition_table(PATCHFOOL_AREAMATCH, out, PATCHFOOL_AREAMATCH_TITLE)


if __name__ == '__main__':
    main()