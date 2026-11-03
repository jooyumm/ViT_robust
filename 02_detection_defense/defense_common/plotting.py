"""02 내부 방어 비교 그림. clean은 항상 방어 없는 원본 모델이다."""
import numpy as np


def plot_comparison(report, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = np.array([report[k] for k in ('clean_accuracy','attack_accuracy','defended_attack_accuracy')])*100
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    bars = ax.bar(['Clean\nViT-B/16', 'Attacked\nPatchFool', 'Defended\nRegion switch'], values,
                  color=['#4682b4', '#c95b54', '#4e9970'])
    ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=4)
    ax.set(ylim=(0, 105), ylabel='Top-1 accuracy (%)', title=title)
    fig.savefig(output, dpi=160)
    plt.close(fig)
