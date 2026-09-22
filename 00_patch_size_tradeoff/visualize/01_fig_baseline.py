"""
01_fig_baseline.py — 실험 1~3(baseline) 시각화: RA / ASR, 다중 시드 mean±std
- CA는 공격과 무관해서 00_fig_ca.py로 따로 뺐음
- 로그 수집/파싱은 results_io.py, 그래프 함수는 plotting.py 공용 모듈 사용
- scripts/01_baseline.sh가 만든 로그를 씀

사용법:
  python visualize/01_fig_baseline.py
"""
import os
from results_io import collect_all_logs, FIG_DIR
from plotting import plot_results


def main():
    data = collect_all_logs()
    print("수집된 데이터 (attack, P, metric -> seed별 값):")
    for atk, pdict in data.items():
        for p, m in sorted(pdict.items()):
            n_seeds = len(m['CA'])
            print(f"  {atk} P={p}: seeds={m['seeds']} (n={n_seeds})  "
                  f"CA={m['CA']}  RA={m['RA']}  ASR={m['ASR']}")

    suffix = ' (ImageNet-1k val)'
    plot_results(data, os.path.join(FIG_DIR, '01_baseline_all.png'), suffix)
    print("Done.")


if __name__ == '__main__':
    main()