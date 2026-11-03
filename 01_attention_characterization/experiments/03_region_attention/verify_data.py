"""기존 파일을 변경하지 않는 스트리밍 데이터 감사. 결과는 stdout에만 출력한다."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parents[1]))
from src.provenance import sha256
from src.patches import coverage_from_images
from src.cohort import DEFAULT_DATASET, validate_default_manifest
from region_metrics import patch_region_index, RUNS

def token_quadrants():
    """사분면(2x2) 버전의 patch->region 매핑 -- common.patch_region_index(7)과 같다."""
    return patch_region_index(7)

def audit():
    """원본 clean/PatchFool 데이터와 모델·좌표 규약의 정합성을 확인한다."""
    base = None
    clean_hashes = {}
    reports = {}
    for run in RUNS:
        m = json.loads((run / 'manifest.json').read_text())
        if run == DEFAULT_DATASET:
            validate_default_manifest(m)
            base = m
        assert m['complete'] and m['schema_version'] == 2
        for key in ['model_sha256', 'sample_indices', 'sample_paths', 'mean', 'std']:
            assert m[key] == base[key], key
        print('실행', run.name, '모델', m['model'], '체크포인트', m['model_sha256'], flush=True)
        ids = []
        stats = {}
        offset = 0
        expected_keys = None
        for chunk in m['chunks']:
            path = (run / chunk['path']).resolve()
            assert run.resolve() in path.parents and sha256(path) == chunk['sha256']
            with np.load(path, allow_pickle=False) as z:
                assert z['run_id'].item() == m['run_id']
                keys = set(z.files)
                if expected_keys is None:
                    expected_keys = keys
                assert keys == expected_keys
                ids.extend(z['sample_index'].tolist())
                n = chunk['n']
                labels = z['label']
                assert len(labels) == n
                clean = z['clean__images']
                clean_pred = z['clean__prediction']
                for (i, sid) in enumerate(z['sample_index']):
                    h = hashlib.sha256(clean[i].tobytes()).hexdigest()
                    if sid in clean_hashes:
                        assert clean_hashes[sid] == h
                    else:
                        clean_hashes[sid] = h
                for group in ['clean', 'patchfool']:
                    a = z[group + '__attention']
                    x = z[group + '__images']
                    logits = z[group + '__logits']
                    pred = z[group + '__prediction']
                    assert a.shape == (n, 12, 12, 197) and x.shape == (n, 3, 224, 224)
                    assert np.isfinite(a).all() and np.isfinite(x).all() and np.isfinite(logits).all()
                    assert a.min() >= 0 and a.max() <= 1 + 1e-06
                    err = float(np.abs(a.sum(-1) - 1).max())
                    assert err < 0.0001
                    np.testing.assert_array_equal(pred, logits.argmax(-1))
                    pixels = x * np.array(m['std'])[None, :, None, None] + np.array(m['mean'])[None, :, None, None]
                    assert pixels.min() >= -1e-05 and pixels.max() <= 1 + 1e-05
                    s = stats.setdefault(group, dict(n=0, row_error=0, min=1.0, max=0.0, outcomes=[0, 0, 0], single=0, multi=0, areas=[], patches=[]))
                    s['n'] += n
                    s['row_error'] = max(s['row_error'], err)
                    s['min'] = min(s['min'], float(a.min()))
                    s['max'] = max(s['max'], float(a.max()))
                    if group != 'clean':
                        cov = coverage_from_images(clean, x, 16, m['gt_threshold'])
                        np.testing.assert_allclose(cov, z[group + '__gt_coverage'], atol=1e-07, rtol=0)
                        changed = np.abs(x - clean).sum(1) > m['gt_threshold']
                        area = changed.reshape(n, 2, 112, 2, 112).sum((2, 4)).reshape(n, 4)
                        assert (area.sum(1) > 0).all()
                        count = (area > 0).sum(1)
                        s['single'] += int((count == 1).sum())
                        s['multi'] += int((count > 1).sum())
                        s['areas'].extend(changed.sum((1, 2)).tolist())
                        s['patches'].extend((cov > 0).sum(1).tolist())
                        out = np.where(clean_pred == labels, (pred != labels).astype(np.int8), -1)
                        np.testing.assert_array_equal(out, z[group + '__outcome'])
                        for code in [-1, 0, 1]:
                            s['outcomes'][code + 1] += int((out == code).sum())
                offset += n
        assert ids == m['sample_indices'] and len(set(ids)) == len(ids) == 500
        print('필드', sorted(expected_keys), flush=True)
        print('hidden 저장 필드', [k for k in expected_keys if 'hidden' in k or 'feature' in k], flush=True)
        for (g, s) in stats.items():
            if s['areas']:
                s['area_range'] = [min(s['areas']), max(s['areas'])]
                s['token_range'] = [min(s['patches']), max(s['patches'])]
            s.pop('areas')
            s.pop('patches')
            print(g, s, flush=True)
        reports[run.name] = stats
    import torch
    torch.set_num_threads(2)
    from src.models import load_vit_model
    model = load_vit_model(16, torch.device('cpu'))
    assert model is not None
    from src.provenance import model_hash
    assert model_hash(model) == base['model_sha256']
    assert model.patch_embed.grid_size == (14, 14) and model.patch_embed.patch_size == (16, 16)
    assert model.num_prefix_tokens == 1 and len(model.blocks) == 12
    assert model.embed_dim == 768 and all((b.attn.num_heads == 12 for b in model.blocks))
    print('모델 구조 검증: ViT-B/16, 12 blocks, 12 heads, D=768, head D=64, CLS 1, patch 196', flush=True)
    print('patch embedding', model.patch_embed.proj, 'MLP hidden', model.blocks[0].mlp.fc1.out_features, flush=True)
    x = torch.arange(224 * 224).reshape(1, 1, 224, 224).float()
    patches = torch.nn.functional.unfold(x, kernel_size=16, stride=16)
    for k in [0, 6, 7, 13, 84, 91, 98, 105, 195]:
        (r, c) = divmod(k, 14)
        assert patches[0, 0, k] == x[0, 0, r * 16, c * 16]
    assert all(((token_quadrants() == q).sum() == 49 for q in range(4)))
    print('좌표: token=1+14*r+c; Q1 r<7,c<7; Q2 r<7,c>=7; Q3 r>=7,c<7; Q4 나머지', flush=True)
    try:
        from PIL import Image
        from src.dataset import _TRANSFORM
        img = _TRANSFORM(Image.open(base['sample_paths'][0]).convert('RGB')).numpy()
        with np.load(DEFAULT_DATASET / base['chunks'][0]['path']) as z:
            np.testing.assert_allclose(img, z['clean__images'][0], atol=1e-06, rtol=0)
        print('원본 JPEG 재전처리: 첫 표본 저장 픽셀과 일치; 전체 JPEG 재검증은 하지 않음', flush=True)
    except (FileNotFoundError, PermissionError) as e:
        print('원본 JPEG 재전처리 확인 불가:', str(e), flush=True)
    print('기존 실제 변경 mask는 픽셀 차이로 복원 가능; 최적화에 허용한 원래 mask/좌표는 별도 저장 안 됨.', flush=True)
    print('감사 완료; 신규 forward 일치와 full attention/hidden 검증은 추출 단계에서 수행.', flush=True)
if __name__ == '__main__':
    audit()
