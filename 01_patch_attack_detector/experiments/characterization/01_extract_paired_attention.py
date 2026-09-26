"""01 추출: 동일 clean–attack 쌍의 이미지·예측·헤드별 attention·GT를 함께 저장한다.

실행은 사용자가 요청할 때만 수행한다. 과거 독립 실행 결과를 합치지 않는다.
분석은 02_compare_attention_success_failure.py에서 같은 실행 폴더만 읽어 수행한다.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    """서로 다른 실행의 청크가 섞이거나 저장 후 변경되는 것을 확인한다."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def observe(model, images):
    """한 번의 forward에서 logits와 모든 레이어의 CLS attention을 함께 얻는다."""
    import torch
    rows = []
    handles = []

    def attention_hook(module, inputs, output):
        tokens = inputs[0]
        batch, count, channels = tokens.shape
        qkv = module.qkv(tokens).reshape(
            batch, count, 3, module.num_heads, channels // module.num_heads)
        query, key, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query = module.q_norm(query)
        key = module.k_norm(key)
        # CLS query만 필요하다. CLS key를 포함해 softmax하므로 확률 질량이 보존된다.
        attention = ((query[:, :, :1] * module.scale) @ key.transpose(-2, -1)).softmax(-1)
        rows.append(attention[:, :, 0].detach().cpu())

    try:
        for block in model.blocks:
            handles.append(block.attn.register_forward_hook(attention_hook))
        with torch.no_grad():
            logits = model(images).detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return logits.numpy(), torch.stack(rows, dim=1).numpy()


def gt_coverage(clean, attacked, patch_size):
    """공격 생성 시의 동일 텐서 쌍에서 실제 변경 픽셀 비율을 계산한다."""
    mask = (attacked - clean).abs().sum(dim=1) > 1e-6
    cells = mask.float().unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    return cells.mean(dim=(-1, -2)).flatten(1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--attacks', nargs='+', choices=['patchfool', 'lavan'], default=['patchfool', 'lavan'])
    parser.add_argument('--pf_iters', type=int, default=250)
    parser.add_argument('--attn_layer_idx', type=int, default=4, help='PatchFool 위치 선택 레이어의 0-based 인덱스')
    parser.add_argument('--pf_clip_valid', action='store_true', help='PatchFool 패치 픽셀을 유효 이미지 범위로 제한')
    parser.add_argument('--lavan_steps', type=int, default=40)
    parser.add_argument('--lavan_patch_ratio', type=float, default=.02)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--output', type=Path, help='새 실행 폴더. 기존 폴더 덮어쓰기 금지')
    args = parser.parse_args()
    if min(args.num_samples, args.batch_size, args.pf_iters, args.lavan_steps) <= 0:
        parser.error('표본 수, 배치 크기, 반복 횟수는 양수여야 합니다.')
    if not 0 <= args.attn_layer_idx < 12:
        parser.error('attn_layer_idx는 0~11이어야 합니다.')
    if not 1 / 224**2 <= args.lavan_patch_ratio < 1:
        parser.error('LaVAN 패치 크기는 1픽셀 이상, 전체 이미지 미만이어야 합니다.')
    if len(set(args.attacks)) != len(args.attacks):
        parser.error('공격 이름을 중복 지정할 수 없습니다.')

    import numpy as np
    import torch
    import timm
    from src.models import load_vit_model, MODEL_NAMES
    from src.dataset import get_dataloader
    from src.attacks.patch_fool import patch_fool_attack
    from src.attacks.lavan import lavan_attack

    def seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    seed_all(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU가 없습니다. GPU 노드에서 실행하거나 --device cpu를 지정하세요.')
    model = load_vit_model(16, torch.device(args.device))
    if model is None:
        raise RuntimeError('모델을 불러오지 못했습니다.')
    model.eval()
    model.requires_grad_(False)
    # 공유 공격 구현의 q/k 재구성과 맞지 않는 모델을 조용히 사용하지 않는다.
    for block in model.blocks:
        if not isinstance(block.attn.q_norm, torch.nn.Identity) or not isinstance(block.attn.k_norm, torch.nn.Identity):
            raise ValueError('현재 공유 PatchFool은 q/k 정규화가 없는 모델만 지원합니다.')
    patch_size = model.patch_embed.patch_size[0]
    loader, dataset = get_dataloader(batch_size=args.batch_size, num_samples=args.num_samples, seed=args.seed)
    indices = list(loader.dataset.indices)
    run_id = uuid.uuid4().hex
    output = args.output or HERE.parents[1] / 'results/characterization/attention_comparison' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + run_id[:8])
    output.mkdir(parents=True, exist_ok=False)
    chunks_dir = output / 'data'
    chunks_dir.mkdir()
    # 실제 모델 파라미터의 지문으로 체크포인트 이름만 같은 경우도 구분한다.
    model_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        model_digest.update(name.encode())
        model_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    metadata = {
        'schema_version': 2, 'run_id': run_id, 'complete': False,
        'model': MODEL_NAMES[16], 'model_sha256': model_digest.hexdigest(),
        'torch': torch.__version__, 'timm': timm.__version__, 'numpy': np.__version__,
        'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'attacks': args.attacks, 'patch_size': patch_size,
        'n_layers': len(model.blocks), 'n_heads': model.blocks[0].attn.num_heads,
        'mean': list(model.pretrained_cfg['mean']), 'std': list(model.pretrained_cfg['std']),
        'preprocessing': 'src/dataset.py의 transform',
        'sample_indices': indices,
        'sample_paths': [dataset.samples[index][0] for index in indices],
        'source_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in [
            Path(__file__).resolve(), ROOT / 'src/models.py', ROOT / 'src/dataset.py',
            ROOT / 'src/attacks/patch_fool.py', ROOT / 'src/attacks/lavan.py']},
        'gt_threshold': 1e-6,
        'attention': 'CLS query, 모든 key 포함, softmax 이후, 헤드별 저장; 재정규화 없음',
        'seed_policy': 'seed + 1009 * 표본 시작 offset + 공격별 고정 offset(PF=0, LaVAN=1)',
        'limitations': ['PatchFool은 공유 구현의 unbounded 픽셀 범위를 유지함.',
                       'LaVAN은 배치 내 위치를 공유함. 재현 시 배치 크기를 고정할 것.',
                       '두 공격의 면적/최적화 예산은 같지 않음. 공격별 성공/실패를 비교할 것.'],
        'chunks': [],
    }

    def save_manifest():
        temporary = output / 'manifest.tmp'
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(output / 'manifest.json')

    save_manifest()
    offset = 0
    for images, labels in loader:
        images = images.to(args.device)
        labels = labels.to(args.device)
        logits, attention = observe(model, images)
        payload = {
            'run_id': np.array(run_id), 'sample_index': np.asarray(indices[offset:offset + len(images)]),
            'label': labels.cpu().numpy(), 'clean__images': images.cpu().numpy(),
            'clean__logits': logits, 'clean__prediction': logits.argmax(-1), 'clean__attention': attention,
        }
        for attack in args.attacks:
            seed_all(args.seed + 1009 * offset + {'patchfool': 0, 'lavan': 1}[attack])
            if attack == 'patchfool':
                attacked, _ = patch_fool_attack(
                    model, images, labels, torch.device(args.device), patch_size_model=16,
                    num_patch=1, attack_mode='Attention', patch_select='Attn',
                    attn_layer_idx=args.attn_layer_idx, train_attack_iters=args.pf_iters,
                    clip_valid_range=args.pf_clip_valid)
            else:
                attacked = lavan_attack(model, images, labels, torch.device(args.device),
                                       patch_ratio=args.lavan_patch_ratio, steps=args.lavan_steps)
            attacked_logits, attacked_attention = observe(model, attacked)
            clean_correct = logits.argmax(-1) == payload['label']
            # -1: clean부터 오답, 0: 공격 실패, 1: 공격 성공
            outcome = np.where(clean_correct, (attacked_logits.argmax(-1) != payload['label']).astype(np.int8), -1)
            fields = {'images': attacked.cpu().numpy(), 'logits': attacked_logits,
                      'prediction': attacked_logits.argmax(-1), 'attention': attacked_attention,
                      'gt_coverage': gt_coverage(images, attacked, patch_size), 'outcome': outcome}
            payload.update({attack + '__' + key: value for key, value in fields.items()})
        path = chunks_dir / f'chunk_{offset:06d}.npz'
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(path)
        metadata['chunks'].append({'path': str(path.relative_to(output)), 'sha256': file_hash(path), 'n': len(images)})
        offset += len(images)
        save_manifest()
        print(f'{offset}/{len(indices)}개 저장 완료', flush=True)
    metadata['complete'] = True
    save_manifest()
    print(f'추출 완료: {output}')


if __name__ == '__main__':
    main()
