"""
01_fig_baseline_p8p16.py — 01_fig_baseline.py의 축약판: PGD와 P=32를 뺀 버전.
LaVAN/PatchFool x P=8/16만 남긴다. 데이터는 01_fig_baseline.py와 동일하게
scripts/01_baseline.sh가 만든 기존 로그(results/logs/)를 그대로 재사용하고,
plot_results에 넘기기 전에 데이터만 필터링한다 — 재실험 불필요.

사용법:
  python visualize/01_fig_baseline_p8p16.py
"""
import os
from results_io import collect_all_logs, FIG_DIR
from plotting import plot_results


def main():
    data = collect_all_logs()

    # PGD, P=32 제거
    data = {atk: {p: m for p, m in pdict.items() if p != 32}
            for atk, pdict in data.items() if atk != 'pgd'}

    print("필터링된 데이터 (attack, P -> seed별 값):")
    for atk, pdict in data.items():
        for p, m in sorted(pdict.items()):
            n_seeds = len(m['CA'])
            print(f"  {atk} P={p}: seeds={m['seeds']} (n={n_seeds})  "
                  f"CA={m['CA']}  RA={m['RA']}  ASR={m['ASR']}")

    suffix = ' (ImageNet-1k val)'
    out_path = os.path.join(FIG_DIR, '01_baseline.png')
    plot_results(data, out_path, suffix)
    print(f"Done. Saved: {out_path}")


if __name__ == '__main__':
    main()
