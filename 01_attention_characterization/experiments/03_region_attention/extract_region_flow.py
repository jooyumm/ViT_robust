"""clean/PatchFool을 한 번 추론하여 2×2 영역 attention 흐름을 추출한다."""
import argparse
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from region_metrics import BLOCKS, GRID, RUNS, patch_region_index

class Capture:
    """forward 1번에 12개 block의 attention을 전부 가로채 block 크기 region 단위로 즉시 집계하는
    집계기. 원시 attention은 timm 공식 추출기에서 전달받는다."""

    def __init__(self, model, block, pilot=False):
        """하나의 해상도에 대한 집계 행렬과 선택적 hidden 검증 hook을 준비한다."""
        import torch
        self.flow = []
        self.to_cls = []
        self.cls_flow = []
        self.cls_self = []
        self.cls_rows = []
        self.full = []
        self.hidden = []
        self.handles = []
        self.pilot = pilot
        self.error = 0.0
        self.verified = 0
        self.block = block
        self.n_regions = (GRID // block) ** 2
        self.q = torch.tensor(patch_region_index(block), device=next(model.parameters()).device)
        self.membership = torch.nn.functional.one_hot(self.q, num_classes=self.n_regions).float()
        if pilot:
            self.handles.append(model.blocks[0].register_forward_pre_hook(lambda m, i: self.hidden.append(i[0][:1].detach().cpu())))
            for b in model.blocks:
                self.handles.append(b.register_forward_hook(lambda m, i, o: self.hidden.append(o[:1].detach().cpu())))

    def attention(self, a):
        """실제 softmax 확률을 영역별로 집계하고 질량 보존을 확인한다."""
        import torch
        if a.shape[1:] != (12, 197, 197):
            raise ValueError('full attention 차원 불일치')
        assert torch.isfinite(a).all() and a.min() >= 0 and (a.max() <= 1 + 1e-06)
        err = float((a.sum(-1) - 1).abs().max())
        self.error = max(self.error, err)
        assert err < 0.0001
        pp = a[:, :, 1:, 1:]
        member = self.membership.to(pp.dtype)
        per_region = self.block * self.block
        f = torch.einsum('pi,bhpq,qj->bhij', member, pp, member) / per_region
        to_cls = torch.einsum('pi,bhp->bhi', member, a[:, :, 1:, 0]) / per_region
        cls = torch.einsum('bhq,qj->bhj', a[:, :, 0, 1:], member)
        torch.testing.assert_close(f.sum(-1) + to_cls, torch.ones_like(to_cls), atol=1e-05, rtol=0)
        torch.testing.assert_close(cls.sum(-1) + a[:, :, 0, 0], torch.ones_like(a[:, :, 0, 0]), atol=1e-05, rtol=0)
        if self.pilot:
            reference = torch.stack([torch.stack([pp[:1, :, self.q == i][:, :, :, self.q == j].sum(-1).mean(-1) for j in range(self.n_regions)], -1) for i in range(self.n_regions)], -2)
            torch.testing.assert_close(reference, f[:1], atol=1e-06, rtol=0.0001)
            self.full.append(a[:1].detach().cpu())
        for (name, v) in [('flow', f), ('to_cls', to_cls), ('cls_flow', cls), ('cls_self', a[:, :, 0, 0]), ('cls_rows', a[:, :, 0])]:
            getattr(self, name).append(v.detach())

    def close(self):
        """등록해둔 hook을 전부 해제한다(한 그룹 forward가 끝날 때마다 호출)."""
        for h in self.handles:
            h.remove()

    def arrays(self):
        """쌓아둔 block별 리스트를 (이미지, block, ...) 모양의 numpy 배열로 합친다."""
        import torch
        return {k: torch.stack(getattr(self, k), 1).cpu().numpy() for k in ['flow', 'to_cls', 'cls_flow', 'cls_self', 'cls_rows']}

from src.provenance import model_hash

def region_attack_area(mask, side, regionpix, n_regions):
    """픽셀 단위 변경 mask(공격이 바꾼 곳)를 region 단위 면적으로 집계한다. 흐름(F) 계산에는
    전혀 쓰이지 않고, 나중에 GT 위치 평가(4단계)에서만 쓰인다."""
    return mask.reshape(len(mask), side, regionpix, side, regionpix).sum((2, 4)).reshape(-1, n_regions)

def extract(resolutions, output_root=None, device='cuda', source_file=None, save_pilot=False):
    """원본 데이터 검증 후 2×2 영역 흐름을 집계하고 디스크 배열에 청크별 기록한다."""
    import hashlib, json
    import torch
    from src.provenance import sha256
    from src.patches import coverage_from_images
    from src.models import load_vit_model
    from src.attention import extract_attention
    outputs = {name: Path(output_root) / 'extracted' if output_root else HERE / 'extracted' for name in resolutions}
    for output in outputs.values():
        if output.exists():
            raise FileExistsError('기존 출력 덮어쓰기 금지: ' + str(output))
    if device == 'cuda' and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA 장치 없음')
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    manifests = [json.loads((r / 'manifest.json').read_text()) for r in RUNS]
    base = manifests[0]
    model = load_vit_model(16, torch.device(device))
    assert model is not None
    model.eval().requires_grad_(False)
    assert model_hash(model) == base['model_sha256']
    assert model.num_prefix_tokens == 1 and model.embed_dim == 768
    with np.load(RUNS[0] / base['chunks'][0]['path']) as z:
        probe = torch.tensor(z['clean__images'][:1], device=device)
    with torch.inference_mode():
        expected = model.patch_embed.norm(model.patch_embed.proj(probe).flatten(2).transpose(1, 2))
        torch.testing.assert_close(model.patch_embed(probe), expected)
        fused_logits = model(probe)
    for b in model.blocks:
        b.attn.fused_attn = False
    with torch.inference_mode():
        torch.testing.assert_close(model(probe), fused_logits, atol=0.0001, rtol=0.0001)
    for output in outputs.values():
        output.mkdir(parents=True)
    pieces = {name: {} for name in resolutions}
    clean_hash = {}
    audit = []
    with torch.inference_mode():
        for (run, m) in zip(RUNS, manifests):
            assert m['complete']
            for key in ['model_sha256', 'sample_indices', 'sample_paths', 'mean', 'std']:
                assert m[key] == base[key]
            offset = 0
            for chunk in m['chunks']:
                path = (run / chunk['path']).resolve()
                assert run.resolve() in path.parents
                assert sha256(path) == chunk['sha256']
                with np.load(path, allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['sample_index'], m['sample_indices'][offset:offset + chunk['n']])
                    clean = z['clean__images']
                    labels = z['label']
                    for (i, sid) in enumerate(z['sample_index']):
                        h = hashlib.sha256(clean[i].tobytes()).hexdigest()
                        if sid in clean_hash:
                            assert clean_hash[sid] == h
                        else:
                            clean_hash[sid] = h
                    groups = ['clean', 'patchfool']
                    for group in groups:
                        x = z[group + '__images']
                        pilot = offset == 0 and group == 'clean'
                        # 공식 attention 추출 결과를 받을 2×2 영역 집계기를 준비한다.
                        captures = {name: Capture(model, BLOCKS[name], pilot) for name in resolutions}
                        try:
                            logits, attention = extract_attention(model, torch.from_numpy(x).to(device))
                            logits = logits.cpu().numpy()
                            for capture in captures.values():
                                for a in attention:
                                    capture.attention(a)
                            del attention
                        finally:
                            for capture in captures.values():
                                capture.close()
                        for (name, cap) in captures.items():
                            output = outputs[name]
                            block = BLOCKS[name]
                            side = GRID // block
                            n_regions = side * side
                            regionpix = block * 16
                            arrays = cap.arrays()
                            np.testing.assert_allclose(logits, z[group + '__logits'], atol=0.003, rtol=0.001)
                            np.testing.assert_array_equal(logits.argmax(-1), z[group + '__prediction'])
                            err = float(np.abs(arrays['cls_rows'] - z[group + '__attention']).max())
                            np.testing.assert_allclose(arrays['cls_rows'], z[group + '__attention'], atol=0.0001, rtol=0.0001)
                            arrays.pop('cls_rows')
                            if pilot and save_pilot:
                                full = torch.stack(cap.full, 1).numpy()
                                hidden = torch.stack(cap.hidden, 1)[:1].numpy()
                                assert hidden.shape == (1, 13, 197, 768) and np.isfinite(hidden).all()
                                np.savez_compressed(output / f'pilot_{group}.npz', attention=full, hidden=hidden, sample_index=z['sample_index'][:1], image=x[:1])
                            # 공격 위치는 흐름 계산이 끝난 뒤 평가용 GT로만 기록한다.
                            if group != 'clean':
                                cov = coverage_from_images(clean, x, 16, m['gt_threshold'])
                                np.testing.assert_allclose(cov, z[group + '__gt_coverage'], atol=1e-07, rtol=0)
                                mask = np.abs(x - clean).sum(1) > m['gt_threshold']
                                areas = region_attack_area(mask, side, regionpix, n_regions)
                                arrays['attack_area'] = areas
                                arrays['attack_fraction'] = areas / areas.sum(1, keepdims=True)
                                arrays['outcome'] = z[group + '__outcome']
                            for (key, value) in arrays.items():
                                field = group + '__' + key
                                if field not in pieces[name]:
                                    # 전체 청크를 RAM에서 이어 붙이지 않고 디스크 배열의 지정 위치에 기록한다.
                                    pieces[name][field] = np.lib.format.open_memmap(output / (field + '.npy'), mode='w+', dtype=value.dtype, shape=(len(base['sample_indices']), *value.shape[1:]))
                                pieces[name][field][offset:offset + len(value)] = value
                            audit.append((name, group, offset, cap.error, err, float(np.abs(logits - z[group + '__logits']).max())))
                    offset += chunk['n']
                    print(run.name, offset, '/', len(base['sample_indices']), 'forward 및 확률/GT 검증', flush=True)
    for (name, output) in outputs.items():
        out = dict(pieces[name])
        out.update(sample_index=np.asarray(base['sample_indices']), sample_paths=np.asarray(base['sample_paths']), model_sha256=np.array(base['model_sha256']), block=np.array(BLOCKS[name]), source_manifest_sha256=np.asarray([sha256(r / 'manifest.json') for r in RUNS]), source_script_sha256=np.array(sha256(source_file) if source_file else ''), torch_version=np.array(torch.__version__))
        temporary = output / 'flow.tmp'
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **out)
        temporary.replace(output / 'flow.npz')
        for (field, array) in pieces[name].items():
            array.flush()
            array._mmap.close()
            (output / (field + '.npy')).unlink()
        print(name, '추출 완료:', output / 'flow.npz', flush=True)
    for (name, group, offset, row_error, attention_error, logit_error) in audit:
        if offset == 0:
            print(name, group, '첫 청크 검증 오차:', row_error, attention_error, logit_error)

def main():
    """clean/PatchFool의 2×2 영역 흐름을 추출한다."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resolution', choices=['2x2'], default='2x2', help='2x2=4 regions')
    p.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--output-root', type=Path, default=None, help='extracted를 생성할 상위 폴더')
    p.add_argument('--save-pilot', action='store_true', help='첫 clean 표본의 원시 tensor도 저장')
    args = p.parse_args()
    resolutions = [args.resolution]
    extract(resolutions, args.output_root, args.device, Path(__file__), args.save_pilot)
if __name__ == '__main__':
    main()
