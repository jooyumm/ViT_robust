"""P8/P16 비교 실험 전용: 조건/소스 검증, 재사용, Clean→Attacked→Defended 그림."""
import json
import numpy as np
from src.provenance import sha256

CASES = [('p8_one', 8, 1, 'Attn'), ('p8_four', 8, 4, 'Contiguous'), ('p16_one', 16, 1, 'Contiguous')]
PROTOCOL = 'model-specific direct PatchFool, incoming/replacement; P8 1 token, P8 adjacent 4 tokens, P16 1 token; non-adaptive'


def source_hashes(root):
    files = ['src/attacks/__init__.py', 'src/models.py', 'src/attacks/patch_fool.py', 'src/attack_utils.py',
             'src/attention.py', 'src/cohort.py', 'src/provenance.py', 'src/patches.py',
             'data/gt/imagenet500.json', '02_detection_defense/region_switch/core/attention.py',
             '02_detection_defense/region_switch/core/switch.py',
             '02_detection_defense/region_switch/core/inference.py',
             '02_detection_defense/region_switch/run.py',
             '02_detection_defense/region_switch/experiments/compare_models.py',
             '02_detection_defense/region_switch/experiments/model_support.py']
    return {file: sha256(root/file) for file in files}


def reusable_results(path, *, attack_iters, batch_size, seed, manifest_hash, sources, n):
    text = path.read_text()
    marker = text.rfind('\n{\n  "results":')
    if marker < 0:
        raise ValueError('완료된 비교 report가 없는 로그입니다.')
    previous = json.loads(text[marker+1:])
    for key, expected in dict(attack_iters=attack_iters, batch_size=batch_size, seed=seed,
                              attack_protocol=PROTOCOL, source_manifest_sha256=manifest_hash,
                              source_sha256=sources).items():
        if previous.get(key) != expected:
            raise ValueError(f'{key}가 달라 결과를 자동 재사용할 수 없습니다. 이전 로그는 보존하고 새로 실행하세요.')
    results = previous['results']
    if any(k not in {case[0] for case in CASES} or v['n'] != n for k,v in results.items()):
        raise ValueError('비교 조건 또는 표본 수 불일치')
    return results


def plot_results(results, output, attack_iters):
    if results['p8_one']['correct']['clean'] != results['p8_four']['correct']['clean']:
        raise ValueError('P8 조건 사이 clean 정답 수가 다릅니다.')
    values = np.array([results['p8_one']['accuracy']['clean'], results['p16_one']['accuracy']['clean'],
                       results['p8_one']['accuracy']['attack'], results['p8_four']['accuracy']['attack'],
                       results['p16_one']['accuracy']['attack'], results['p16_one']['accuracy']['defense']])*100
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
    positions = [0, 1, 3, 4, 5, 7]
    bars = ax.bar(positions, values, color=['#4682b4', '#4682b4', '#c95b54', '#c95b54', '#c95b54', '#4e9970'])
    ax.set_xticks(positions, ['ViT-B/8', 'ViT-B/16', 'P8: 1 token\n8x8 px',
                             'P8: 4 adjacent\n16x16 px', 'P16: 1 token\n16x16 px', 'Region switch\nP16 attack'])
    ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=4)
    for left, right, center, title in [(-.5,1.5,.5,'Clean'),(2.5,5.5,4,'Attacked'),(6.5,7.5,7,'Defended')]:
        ax.axvspan(left, right, color='gray', alpha=.045, zorder=0)
        ax.text(center, 101, title, ha='center', fontsize=12, weight='bold')
    ax.set(ylim=(0, 110), ylabel='Top-1 accuracy (%)',
           title=f'ViT-B baselines and region switch (n={results["p16_one"]["n"]})')
    fig.supxlabel(f'Model-specific direct PatchFool; {attack_iters} steps; non-adaptive', fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
