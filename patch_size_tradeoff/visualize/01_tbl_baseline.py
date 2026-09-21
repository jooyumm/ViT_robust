"""
01_tbl_baseline.py — 실험 1~3(baseline) 결과 표: CA / RA / ASR, 다중 시드 mean±std, 논문용 스타일
- 01_fig_baseline.py와 같은 데이터(scripts/01_baseline.sh 로그)를 표 형태로 보여줌
- 표 렌더링 로직은 tables.py 공용 모듈 사용

사용법:
  python visualize/01_tbl_baseline.py
"""
import os
from results_io import collect_all_logs, TABLE_DIR
from tables import make_baseline_table


def main():
    data = collect_all_logs()
    print("수집된 데이터 (attack, P, metric -> seed별 값):")
    for atk, pdict in data.items():
        for p, m in sorted(pdict.items()):
            n_seeds = len(m['CA'])
            print(f"  {atk} P={p}: seeds={m['seeds']} (n={n_seeds})  "
                  f"CA={m['CA']}  RA={m['RA']}  ASR={m['ASR']}")

    make_baseline_table(data, os.path.join(TABLE_DIR, '01_baseline_table.png'),
                        ' (ImageNet-1k val)')
    print("Done.")


if __name__ == '__main__':
    main()