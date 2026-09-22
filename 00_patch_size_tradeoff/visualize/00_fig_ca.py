"""
00_fig_ca.py — Clean Accuracy (실험 1~3 로그에서 뽑음, 공격과 무관)
CA는 patch size(모델)에만 의존하고 공격 종류와는 무관한 값이라 모든 실험이 공유한다.
그래서 특정 실험 번호에 묶지 않고 00번(attack_demo와 같은 급)으로 독립적으로 뺐다.

사용법:
  python visualize/00_fig_ca.py
"""
import os
from results_io import collect_all_logs, FIG_DIR
from plotting import plot_ca


def main():
    data = collect_all_logs()
    plot_ca(data, os.path.join(FIG_DIR, '00_ca.png'), ' (ImageNet-1k val)')


if __name__ == '__main__':
    main()