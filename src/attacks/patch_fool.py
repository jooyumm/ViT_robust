"""Patch-Fool 기본 공격의 timm ViT 적용판.

공식 구현: https://github.com/GATECH-EIC/Patch-Fool/blob/main/main.py
기본: incoming attention 위치 선택 → 유효 범위의 무작위 패치로 교체 → Adam(CE+attention).
모델은 src.models에서 별도로 로드한다. 공격이 모델 가중치를 학습하지 않는다.

선택 가능한 변형은 아래 함수 파라미터만 바꾼다. Contiguous/고정 Sparse와 CLS 선택은
이 저장소의 확장이다. Sparse의 학습형 mask·Mild L2는 구현하지 않았다. 공식 PCGrad의
기본 PCGrad는 기존 norm²을 유지한다. pcgrad_mode="official"이면 공식 norm 분모를 사용한다.
공식 DeiT 대신 timm ViT를 쓰므로 모델·전처리까지 포함한 공식 실험 전체 재현은 아니다.
"""
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from ..attack_utils import attack_context, attention_terms


def _pcgrad(atten_grad_flat, ce_grad_flat, cos_sim, original_shape, mode="projected"):
    """CE와 충돌하는 attention gradient 성분만 제거한다."""
    projected = atten_grad_flat.clone()
    conflict = cos_sim < 0
    a, c = projected[conflict], ce_grad_flat[conflict]
    if mode not in ("projected", "official"):
        raise ValueError("PCGrad mode는 projected 또는 official이어야 합니다.")
    denominator = c.norm(dim=-1, keepdim=True).clamp_min(1e-8) if mode == "official" else c.square().sum(-1, keepdim=True).clamp_min(1e-12)
    projected[conflict] = a - (a*c).sum(-1, keepdim=True) / denominator * c
    return projected.view(original_shape)


def _attention(q, k, scale):
    return ((q * scale) @ k.transpose(-2, -1)).softmax(-1)


def _select_patches(attention, logits, images, labels, patch_size, count, method, source):
    """위치 선택만 담당한다. 공격 최적화는 모든 방법에서 동일하다."""
    batch, _, height, _ = images.shape
    side = height // patch_size
    if method == 'Rand':
        return torch.from_numpy(np.random.randint(0, side*side, (batch, count))).to(images.device)
    if method == 'Saliency':
        grad = torch.autograd.grad(F.cross_entropy(logits, labels), images)[0]
        scores = F.avg_pool2d(grad.abs().sum(1, keepdim=True), patch_size, patch_size).flatten(1)
    else:
        a = attention.mean(1)
        scores = a.mean(-2)[:, 1:] if source == 'incoming' else a[:, 0, 1:]
    if method != 'Contiguous':
        return scores.argsort(descending=True)[:, :count]
    block = 16 // patch_size  # P16 1개와 같은 면적: P8이면 2×2 토큰
    if block < 1 or 16 % patch_size or side % block:
        raise ValueError('Contiguous는 16px 영역을 모델 patch 격자로 나눌 수 있어야 합니다.')
    sums = scores.reshape(batch, side, side).unfold(1, block, block).unfold(2, block, block).sum((-1, -2))
    index = sums.flatten(1).argmax(-1); blocks_per_side = side // block
    dr, dc = torch.meshgrid(torch.arange(block, device=images.device), torch.arange(block, device=images.device), indexing='ij')
    return ((index // blocks_per_side)[:, None]*block + dr.flatten())*side + (index % blocks_per_side)[:, None]*block + dc.flatten()


def patch_fool_attack(
    model, images, labels, device,
    patch_size_model=16,       # 공격 대상 모델의 patch 크기: 8 / 16
    num_patch=1,              # 변경할 token 수. Contiguous에서는 16×16px 면적으로 결정
    attack_mode='Attention',  # Attention: CE+attention / CE_loss: CE만
    patch_select='Attn',      # Attn / Rand / Saliency / Contiguous(로컬 확장)
    attn_layer_idx=4,         # 위치 선택용 0-based 레이어: 4는 다섯 번째
    train_attack_iters=250,   # 입력 이미지 최적화 반복 수
    attack_lr=0.22, step_size=10, gamma=0.95, atten_loss_weight=0.002,
    mild_l_inf=0.0,           # 0: 제한 없음. 양수: 정규화 공간 L∞ (std=.5이면 픽셀 예산의 2배)
    clip_valid_range=True,    # 기본은 유효 픽셀 범위 [-1,1]. False: 로컬 unbounded 실험
    sparse_pixel_num=0,       # 0: token 전체. 양수: 허용 영역 내 고정 무작위 RGB 픽셀 수
    *, attention_source='incoming',  # incoming: 공식 위치 점수 / cls: 기존 로컬 CLS 점수
    initialization='replacement',   # replacement: 무작위 픽셀 교체 / delta: 기존 작은 Gaussian 섭동
    pcgrad_mode='projected',        # projected: 기존 norm² / official: 공식 저장소의 norm 분모
    classification_forward=None,   # 적응형 평가 전용: 실제 방어 logits의 미분 가능한 forward
    initial_delta=None,             # 적응형 평가 전용: 기존 공격본에서 시작할 섭동
    candidate_callback=None,        # 후보 이미지와 목적 모델 logits를 기록 (기본 공격에는 영향 없음)
    selected_patches=None,          # 적응형 재최적화: 원래 공격자가 선택했던 위치 고정 (B,num_patch)
):
    """반환: (공격 이미지, 공격 전 full attention 목록). 모델 모드와 기존 grad는 보존한다.

    기본: patch_fool_attack(model, images, labels, device)
    무작위: patch_select='Rand' / gradient 위치: patch_select='Saliency'
    다중 패치: num_patch=4 / CE만: attack_mode='CE_loss'
    기존 로컬 기본 재현: attention_source='cls', initialization='delta', clip_valid_range=False
    Mild/Sparse 단위·공식 코드와의 차이는 src/README.md 참고.
    """
    with attack_context(model, device=device):
        x = images.detach().clone().to(device)
        y = labels.detach().to(device=device, dtype=torch.long)
        if (x.ndim != 4 or not len(x) or x.shape[-2] != x.shape[-1]
                or not torch.isfinite(x).all()):
            raise ValueError('유한하고 비어 있지 않은 정사각 BCHW 입력이 필요합니다.')
        batch, _, height, _ = x.shape
        size = model.patch_embed.patch_size
        size = (size, size) if isinstance(size, int) else tuple(size)
        if (not isinstance(patch_size_model, int) or patch_size_model < 1
                or size != (patch_size_model,)*2 or height % patch_size_model
                or getattr(model, 'num_prefix_tokens', 1) != 1):
            raise ValueError('모델 patch 크기/정사각 격자/CLS prefix 1개를 확인하세요.')
        patches = (height // patch_size_model)**2
        if (y.shape != (batch,) or patch_select not in ('Attn','Rand','Saliency','Contiguous')
                or attack_mode not in ('Attention','CE_loss')):
            raise ValueError('labels 또는 위치/공격 모드가 올바르지 않습니다.')
        if pcgrad_mode not in ('projected','official'):
            raise ValueError('pcgrad_mode는 projected 또는 official이어야 합니다.')
        if attention_source not in ('incoming','cls') or initialization not in ('replacement','delta'):
            raise ValueError('attention_source/initialization이 올바르지 않습니다.')
        if (not isinstance(attn_layer_idx, int) or not 0 <= attn_layer_idx < len(model.blocks)
                or not isinstance(num_patch, int) or not 1 <= num_patch <= patches):
            raise ValueError('레이어 인덱스와 num_patch를 확인하세요.')
        if (not isinstance(train_attack_iters, int) or train_attack_iters < 1 or not attack_lr > 0
                or not isinstance(step_size, int) or step_size < 1 or not gamma > 0
                or mild_l_inf < 0 or not isinstance(sparse_pixel_num, int) or sparse_pixel_num < 0):
            raise ValueError('반복 수/스케줄/예산 파라미터가 올바르지 않습니다.')
        if clip_valid_range and (x.min() < -1.00001 or x.max() > 1.00001):
            raise ValueError('유효 범위 공격은 mean/std=.5로 정규화된 [-1,1] 입력이 필요합니다.')
        selection_input = x.detach().requires_grad_(patch_select == 'Saliency')
        with torch.set_grad_enabled(patch_select == 'Saliency'), attention_terms(model, _attention) as terms:
            selection_logits = model(selection_input)
        weights = [value.detach().cpu() for value in terms.values()]
        selected = _select_patches(weights[attn_layer_idx].to(device), selection_logits, selection_input, y,
                                   patch_size_model, num_patch, patch_select, attention_source)
        if selected_patches is not None:
            selected=torch.as_tensor(selected_patches,device=device)
            if (selected.shape!=(batch,num_patch) or selected.dtype!=torch.long or
                    not bool(((selected>=0)&(selected<patches)).all()) or
                    any(len(row.unique())!=num_patch for row in selected)):
                raise ValueError('selected_patches는 중복 없는 유효한 B×num_patch long 인덱스여야 합니다.')
        grid = F.one_hot(selected, patches).sum(1).clamp_max(1).reshape(batch,1,height//patch_size_model,height//patch_size_model)
        mask = grid.repeat_interleave(patch_size_model,2).repeat_interleave(patch_size_model,3).to(x.dtype)
        if sparse_pixel_num:
            sparse = torch.zeros_like(mask)
            for i in range(batch):
                allowed = mask[i].flatten().nonzero().flatten()
                if sparse_pixel_num > len(allowed): raise ValueError('sparse_pixel_num이 허용 영역보다 큽니다.')
                sparse[i].view(-1)[allowed[torch.randperm(len(allowed),device=device)[:sparse_pixel_num]]] = 1
            mask = sparse
        if initial_delta is not None:
            delta=initial_delta.detach().clone().to(x)
            if delta.shape!=x.shape or not bool(torch.isfinite(delta).all()):
                raise ValueError('initial_delta는 입력과 같은 모양의 유한한 tensor여야 합니다.')
            if mild_l_inf:delta.clamp_(-mild_l_inf,mild_l_inf)
        elif mild_l_inf:
            delta = (2*torch.rand_like(x)-1)*mild_l_inf
        else:
            delta = torch.rand_like(x)*2-1-x if initialization == 'replacement' else torch.randn_like(x)*.1
        if clip_valid_range: delta = (x+delta).clamp(-1,1)-x
        delta.requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=attack_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
        target = (selected[:,0]+1)[:,None].expand(batch,patches+1).reshape(-1)
        for _ in range(train_attack_iters):
            optimizer.zero_grad()
            context = attention_terms(model,_attention) if attack_mode == 'Attention' else nullcontext({})
            with context as current:
                logits = model(x+delta*mask)
            if classification_forward is not None:
                logits = classification_forward(x+delta*mask)
            if candidate_callback is not None:
                candidate_callback((x+delta*mask).detach(),logits.detach())
            ce = F.cross_entropy(logits,y)
            grad = torch.autograd.grad(ce,delta,retain_graph=bool(current))[0]
            ce_grad = grad.flatten(1).detach().clone()
            for layer in range(1,len(model.blocks)//2) if current else ():
                a = current[layer].mean(1).reshape(-1,patches+1)
                loss = a.clamp_min(1e-8).log().gather(1,target[:,None]).mean()
                attention_grad = torch.autograd.grad(loss,delta,retain_graph=True)[0].flatten(1)
                similarity = F.cosine_similarity(attention_grad,ce_grad,dim=1)
                grad = grad + atten_loss_weight*_pcgrad(attention_grad,ce_grad,similarity,delta.shape,pcgrad_mode)
            delta.grad = -grad
            optimizer.step(); scheduler.step()
            with torch.no_grad():
                if mild_l_inf: delta.clamp_(-mild_l_inf,mild_l_inf)
                if clip_valid_range: delta.copy_((x+delta).clamp(-1,1)-x)
        result=(x+delta*mask).detach()
        if candidate_callback is not None:
            with torch.no_grad():
                candidate_callback(result,(classification_forward or model)(result))
        return result, weights


def patch_fool_config(**overrides):
    """실제로 쓰는 기본값까지 펼친 실행 설정. 새 결과의 재현 정보에 사용한다."""
    import inspect
    config = {name: parameter.default for name, parameter in inspect.signature(patch_fool_attack).parameters.items()
              if parameter.default is not inspect.Parameter.empty and
              name not in ('classification_forward','initial_delta','candidate_callback','selected_patches')}
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(f'알 수 없는 PatchFool 옵션: {sorted(unknown)}')
    config.update(overrides)
    return config
