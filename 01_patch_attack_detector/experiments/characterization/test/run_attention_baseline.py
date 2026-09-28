"""PatchFool 전용: 동일 이미지에서 CLS/column 순위 간격 탐지를 비교하고 PNG만 저장한다."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ranked_gaps(scores, top_k):
    """상위 후보 간 비율만 사용한다. GT는 탐지 계산에 들어가지 않는다."""
    order = np.argsort(-scores, axis=-1, kind='stable')[..., :top_k]
    values = np.take_along_axis(scores, order, axis=-1)
    ratios = values[..., :-1] / np.maximum(values[..., 1:], 1e-12)
    return order[..., :-1], ratios


def detect(scores, top_k, threshold):
    positions, ratios = ranked_gaps(scores, top_k)
    selected = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(selected, positions, ratios > threshold, axis=-1)
    return selected


def calibrate(scores, top_k, fpr):
    # 레이어별 임계값이 아니라 전체 레이어 합집합의 정상 이미지 오탐을 보정한다.
    maxima = ranked_gaps(scores, top_k)[1].max(axis=(1, 2))
    allowed = int(np.floor(len(maxima) * fpr))
    return float(np.sort(maxima)[len(maxima) - allowed - 1])


def observe(model, images):
    import torch
    cls, column, handles = [], [], []

    def hook(module, inputs, output):
        tokens = inputs[0]
        batch, count, channels = tokens.shape
        qkv = module.qkv(tokens).reshape(batch, count, 3, module.num_heads,
                                         channels // module.num_heads)
        query, key, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = ((module.q_norm(query) * module.scale)
                   @ module.k_norm(key).transpose(-2, -1)).softmax(-1)
        cls.append(weights[:, :, 0].mean(1).cpu())
        # 모든 query(CLS 포함)를 평균한다. 탐지 후보 key에서만 CLS를 제외한다.
        column.append(weights.mean((1, 2))[:, 1:].cpu())

    try:
        for block in model.blocks:
            handles.append(block.attn.register_forward_hook(hook))
        with torch.no_grad():
            logits = model(images).cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    return logits, torch.stack(cls, 1).numpy(), torch.stack(column, 1).numpy()


def extract(args, manifest):
    import torch
    from src.models import load_vit_model
    torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다. GPU 노드에서 실행하세요.')
    model = load_vit_model(manifest['patch_size'], args.device)
    if model is None:
        raise RuntimeError('모델 로딩 실패')
    model.eval().requires_grad_(False)
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if digest.hexdigest() != manifest['model_sha256']:
        raise ValueError('원본 실행과 모델 가중치가 다릅니다.')
    collected, checks = {}, []

    def append(key, value):
        collected.setdefault(key, []).append(value)

    for entry in manifest['chunks']:
        path = (args.run / entry['path']).resolve()
        if not path.is_relative_to(args.run.resolve()) or file_hash(path) != entry['sha256']:
            raise ValueError(f'청크 무결성 확인 실패: {path}')
        with np.load(path, allow_pickle=False) as data:
            if str(data['run_id'].item()) != manifest['run_id']:
                raise ValueError('다른 실행 청크가 섞였습니다.')
            for key in ('sample_index', 'label'):
                append(key, data[key])
            for group in ('clean', 'patchfool'):
                logits, cls, column = observe(model, torch.from_numpy(data[group + '__images']).to(args.device))
                old_cls = data[group + '__attention'].mean(axis=2)
                np.testing.assert_allclose(cls, old_cls, atol=1e-4, rtol=1e-3)
                prediction = logits.argmax(-1)
                np.testing.assert_array_equal(prediction, data[group + '__prediction'])
                if not np.isfinite(cls).all() or not np.isfinite(column).all():
                    raise ValueError('attention에 비유한 값이 있습니다.')
                checks.append({'chunk': entry['path'], 'group': group,
                               'max_cls_error': float(np.abs(cls - old_cls).max())})
                append(group + '__cls', cls[..., 1:])
                append(group + '__column', column)
                if group != 'clean':
                    changed = np.abs(data[group + '__images'] - data['clean__images']).sum(1) > manifest['gt_threshold']
                    batch, height, width = changed.shape
                    patch = manifest['patch_size']
                    coverage = changed.reshape(batch, height // patch, patch, width // patch, patch).mean((2, 4)).reshape(batch, -1)
                    np.testing.assert_allclose(coverage, data[group + '__gt_coverage'], atol=1e-7)
                    outcome = np.where(data['clean__prediction'] != data['label'], -1,
                                       (prediction != data['label']).astype(int))
                    np.testing.assert_array_equal(outcome, data[group + '__outcome'])
                    append(group + '__gt', coverage > 0)
                    append(group + '__outcome', outcome)
        print(f'확인 완료: {entry["path"]}', flush=True)
    arrays = {key: np.concatenate(values) for key, values in collected.items()}
    np.testing.assert_array_equal(arrays['sample_index'], manifest['sample_indices'])
    return arrays, checks


def evaluate(args, manifest, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count = len(data['sample_index'])
    permutation = np.random.default_rng(args.seed).permutation(count)
    split = count // 2
    if split < 2:
        raise ValueError('보정/평가 분리에 최소 4개 이미지가 필요합니다.')
    calibration, evaluation = permutation[:split], permutation[split:]
    summaries, samples, layer_rows, thresholds = [], [], [], {}
    for method in ('cls', 'column'):
        clean = data['clean__' + method]
        threshold = calibrate(clean[calibration], args.top_k, args.fpr)
        thresholds[method] = threshold
        clean_flags = detect(clean, args.top_k, threshold).any((1, 2))
        summaries.append(dict(method=method, attack='clean', outcome='all', n=len(evaluation),
                              detection_rate=float(clean_flags[evaluation].mean()),
                              gt_hit_rate='', gt_recall='', precision='', benign_selected=''))
        for attack in ('patchfool',):
            selected = detect(data[attack + '__' + method], args.top_k, threshold)[evaluation]
            gt = data[attack + '__gt'][evaluation]
            outcomes = data[attack + '__outcome'][evaluation]
            union = selected.any(1)
            hits = selected & gt[:, None, :]
            for index, source_index in enumerate(evaluation):
                first = np.flatnonzero(hits[index].any(-1))
                samples.append(dict(method=method, attack=attack,
                    sample_index=int(data['sample_index'][source_index]), outcome=int(outcomes[index]),
                    detected=int(union[index].any()), gt_hit=int(hits[index].any()),
                    first_gt_layer=int(first[0] + 1) if len(first) else '',
                    gt_recall=float((union[index] & gt[index]).sum() / max(gt[index].sum(), 1)),
                    precision=float((union[index] & gt[index]).sum() / max(union[index].sum(), 1)),
                    benign_selected=int((union[index] & ~gt[index]).sum())))
            for outcome, name in ((1, 'success'), (0, 'failure')):
                subset = outcomes == outcome
                n = int(subset.sum())
                rows = [row for row in samples if row['method'] == method and row['attack'] == attack and row['outcome'] == outcome]
                def average(key):
                    return float(np.mean([row[key] for row in rows])) if n else ''
                summaries.append(dict(method=method, attack=attack, outcome=name, n=n,
                    detection_rate=average('detected'), gt_hit_rate=average('gt_hit'),
                    gt_recall=average('gt_recall'), precision=average('precision'),
                    benign_selected=average('benign_selected')))
                for layer_i in range(selected.shape[1]):
                    layer_rows.append(dict(method=method, attack=attack, outcome=name, n=n,
                        layer=layer_i + 1,
                        gt_hit_rate=float(hits[subset, layer_i].any(-1).mean()) if n else '',
                        cumulative_gt_hit_rate=float(hits[subset, :layer_i + 1].any((1, 2)).mean()) if n else ''))
    split_info = dict(thresholds=thresholds, target_calibration_fpr=args.fpr,
                      calibration_ids=data['sample_index'][calibration].tolist(),
                      evaluation_ids=data['sample_index'][evaluation].tolist(), seed=args.seed,
                      top_k=args.top_k)
    # 별도 JSON 대신 PNG 메타데이터에 분할과 임계값을 보존한다.
    metadata = {'Description': json.dumps(dict(
        experiment='PatchFool CLS vs column rank-gap baseline',
        source_run=str(args.run.resolve()), run_id=manifest['run_id'],
        source_manifest_sha256=file_hash(args.run / 'manifest.json'),
        script_sha256=file_hash(__file__), calibration=split_info), ensure_ascii=False)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    categories = [(row['attack'], row['outcome']) for row in summaries if row['method'] == 'cls']
    positions = np.arange(len(categories))
    for method_i, method in enumerate(('cls', 'column')):
        rows = [row for row in summaries if row['method'] == method]
        for axis, key in zip(axes, ('detection_rate', 'gt_hit_rate')):
            values = [row[key] if row[key] != '' else np.nan for row in rows]
            axis.bar(positions + (method_i - .5) * .35, values, .35, label=method.upper())
            axis.set_xticks(positions, [f'{a}\n{o}\nn={r["n"]}' for (a, o), r in zip(categories, rows)], fontsize=8)
            axis.set_ylim(0, 1.05)
            axis.set_title('Image alarm rate (clean = FPR)' if key == 'detection_rate' else 'GT localization hit rate')
            axis.legend()
    fig.savefig(args.output / 'detection_and_localization.png', dpi=180, metadata=metadata)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False, constrained_layout=True)
    for attack_i, attack in enumerate(('patchfool',)):
        for outcome_i, outcome in enumerate(('success', 'failure')):
            axis = axes[attack_i, outcome_i]
            for method in ('cls', 'column'):
                rows = [r for r in layer_rows if r['attack'] == attack and r['outcome'] == outcome and r['method'] == method]
                for key, style, label in (('gt_hit_rate', '--', 'layer'), ('cumulative_gt_hit_rate', '-', 'cumulative')):
                    axis.plot([r['layer'] for r in rows], [r[key] if r[key] != '' else np.nan for r in rows], style,
                              color='tab:blue' if method == 'cls' else 'tab:orange', label=f'{method}: {label}')
            axis.set(title=f'{attack} / {outcome} (n={rows[0]["n"]})', xlabel='Layer', ylabel='GT hit rate', ylim=(0, 1.05))
            axis.legend()
    fig.savefig(args.output / 'layer_gt_localization.png', dpi=180, metadata=metadata)
    plt.close(fig)
    # 표도 PNG로 남겨 그림만으로 수치와 평가 조건을 확인할 수 있게 한다.
    table_rows = []
    for row in summaries:
        values = [row['method'].upper(), row['attack'], row['outcome'], str(row['n'])]
        for key in ('detection_rate', 'gt_hit_rate', 'gt_recall', 'precision'):
            values.append(f"{100 * row[key]:.1f}%" if row[key] != '' else 'N/A')
        table_rows.append(values)
    fig, axis = plt.subplots(figsize=(14, 4.5), layout='constrained')
    axis.axis('off')
    axis.set_title('PatchFool: CLS vs column attention detection', fontsize=14)
    table = axis.table(cellText=table_rows, colLabels=[
        'Method', 'Input', 'Attack outcome', 'n', 'Alarm rate', 'GT hit',
        'GT recall', 'Precision'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    axis.text(.5, .06,
              f'Clean calibration: {len(calibration)} | Held-out evaluation: {len(evaluation)} | '
              f'Target calibration FPR: {args.fpr:.0%} | Top-k: {args.top_k}\n'
              f'Thresholds: CLS={thresholds["cls"]:.4f}, column={thresholds["column"]:.4f} | '
              'Attack groups include initially correct images only.\n'
              'Precision is averaged over all images; no selection counts as zero. N/A = not applicable.',
              ha='center', va='center', transform=axis.transAxes, fontsize=9)
    fig.savefig(args.output / 'summary.png', dpi=180, metadata=metadata)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--fpr', type=float, default=.05)
    args = parser.parse_args()
    if not 0 <= args.fpr < 1 or not 2 <= args.top_k <= 196:
        parser.error('fpr는 [0,1), top-k는 2~196이어야 합니다.')
    manifest = json.loads((args.run / 'manifest.json').read_text())
    if manifest.get('schema_version') != 2 or not manifest.get('complete'):
        raise ValueError('완료된 schema 2 실행만 사용할 수 있습니다.')
    if 'patchfool' not in manifest['attacks']:
        raise ValueError('원본 실행에 PatchFool 결과가 없습니다.')
    args.output.mkdir(parents=True, exist_ok=False)
    data, checks = extract(args, manifest)
    evaluate(args, manifest, data)
    print(f'검증 최대 CLS 오차: {max(item["max_cls_error"] for item in checks):.3g}', flush=True)
    print(f'완료: {args.output}', flush=True)


if __name__ == '__main__':
    main()
