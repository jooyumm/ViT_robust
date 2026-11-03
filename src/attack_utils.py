"""공유 패치 좌표·정규화·최적화 엔진. `attention_fool.py`가 이 엔진
(`projected_attack`)을 쓴다 — `patch_fool.py`/`lavan.py`는 자체 최적화 루프를 쓰므로 해당 없음.

새 공격의 예산은 원래 [0,1] 픽셀 단위다(정규화 공간 단위가 아님). `eps`/`alpha`도 마찬가지로
픽셀 단위이며, 함수 내부에서 mean/std로 정규화 공간에 매핑한다. 기본 `mean=std=(.5,.5,.5)`
(우리 timm 체크포인트 기준) — 다른 정규화를 쓰는 모델은 두 인자를 명시해야 한다.
"""
from contextlib import contextmanager, nullcontext
import math
import torch
import torch.nn.functional as F


@contextmanager
def attack_context(model, seed=None, device='cpu'):
    """모델 모드를 복원하고 입력 gradient를 활성화한다.

    seed가 있으면 RNG를 격리해 재현하고, None이면 전역 RNG를 정상적으로 진행한다.
    모델 파라미터의 기존 grad는 건드리지 않는다. CPU/CUDA RNG를 지원한다.
    """
    modes = [(module, module.training) for module in model.modules()]
    device = torch.device(device)
    if seed is not None and device.type not in ('cpu', 'cuda'):
        raise ValueError('seed를 지정한 RNG 격리는 현재 CPU/CUDA만 지원합니다.')
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    rng_context = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
    with rng_context:
        if seed is not None:
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
        model.eval()
        try:
            with torch.inference_mode(False), torch.enable_grad():
                yield
        finally:
            for module, mode in modes:
                module.training = mode


def prepare(images, labels, device, mean, std):
    images = images.detach().clone().to(device)
    labels = labels.detach().to(device=device, dtype=torch.long)
    if images.ndim != 4 or images.shape[0] == 0 or labels.shape != (images.shape[0],):
        raise ValueError('images는 비어 있지 않은 BCHW, labels는 B여야 합니다.')
    mean = images.new_tensor(mean).reshape(1, -1, 1, 1)
    std = images.new_tensor(std).reshape(1, -1, 1, 1)
    if mean.shape[1] not in (1, images.shape[1]) or std.shape[1] not in (1, images.shape[1]) or (std <= 0).any():
        raise ValueError('mean/std 채널 수와 양수 std를 확인하세요.')
    pixels = images * std + mean
    if not torch.isfinite(pixels).all() or pixels.min() < -1e-5 or pixels.max() > 1+1e-5:
        raise ValueError('정규화 해제한 이미지가 [0,1] 밖입니다. mean/std와 원본 범위를 확인하세요.')
    return images, labels, pixels.clamp(0, 1), mean, std


def make_patch_mask(images, patch_ratio=.02, patch_size=None, fixed_loc=None, mask=None):
    """B1HW bool 마스크. fixed_loc=(top,left), 'center', None(표본별 무작위)."""
    batch, _, height, width = images.shape
    if mask is not None:
        result = torch.as_tensor(mask, device=images.device)
        if result.ndim == 2:
            result = result[None, None]
        elif result.ndim == 3:
            result = result[:, None]
        if result.shape not in ((1, 1, height, width), (batch, 1, height, width)):
            raise ValueError('mask는 HW, BHW, B1HW 또는 11HW여야 합니다.')
        if not ((result == 0) | (result == 1)).all():
            raise ValueError('mask는 0/1이어야 합니다.')
        result = result.bool().expand(batch, -1, -1, -1).clone()
        if not result.flatten(1).any(1).all():
            raise ValueError('표본마다 최소 한 픽셀의 마스크가 필요합니다.')
        return result
    if patch_size is None:
        if not 0 < patch_ratio <= 1:
            raise ValueError('patch_ratio는 (0,1]이어야 합니다.')
        side = max(1, int(math.sqrt(height * width * patch_ratio)))
        ph, pw = side, side
    else:
        ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    if not (1 <= ph <= height and 1 <= pw <= width):
        raise ValueError('패치 크기가 이미지 밖입니다.')
    result = torch.zeros(batch, 1, height, width, device=images.device, dtype=torch.bool)
    for sample_i in range(batch):
        if fixed_loc is None:
            top = int(torch.randint(height-ph+1, ()).item())
            left = int(torch.randint(width-pw+1, ()).item())
        elif fixed_loc == 'center':
            top, left = (height-ph)//2, (width-pw)//2
        else:
            top, left = fixed_loc
        if not (0 <= top <= height-ph and 0 <= left <= width-pw):
            raise ValueError('패치 위치가 이미지 밖입니다.')
        result[sample_i, :, top:top+ph, left:left+pw] = True
    return result


def token_support(model, mask):
    """실제 patch embedding 크기로 변경 영역과 겹치는 image token을 구한다."""
    size = model.patch_embed.patch_size
    size = (size, size) if isinstance(size, int) else size
    if mask.shape[-2] % size[0] or mask.shape[-1] % size[1]:
        raise ValueError('패치 격자로 나눠지는 고정 크기 입력만 지원합니다.')
    return F.max_pool2d(mask.float(), size, stride=size).flatten(1).bool()


def classification_loss(logits, labels, loss='ce', targeted=False, target_labels=None):
    if targeted:
        if target_labels is None or target_labels.shape != labels.shape:
            raise ValueError('targeted=True에는 B 크기 target_labels가 필요합니다.')
        if loss != 'ce':
            raise ValueError('targeted 모드는 CE만 지원합니다.')
        return -F.cross_entropy(logits, target_labels, reduction='none')
    if loss == 'ce':
        return F.cross_entropy(logits, labels, reduction='none')
    if loss == 'dlr':
        if logits.shape[1] < 3:
            raise ValueError('untargeted DLR에는 최소 3개 클래스가 필요합니다.')
        sorted_logits, order = logits.sort(-1)
        true = logits.gather(1, labels[:, None]).squeeze(1)
        other = torch.where(order[:, -1] == labels, sorted_logits[:, -2], sorted_logits[:, -1])
        return -(true-other)/(sorted_logits[:, -1]-sorted_logits[:, -3]+1e-12)
    raise ValueError('loss는 ce 또는 dlr이어야 합니다.')


def projected_attack(model, images, labels, device, objective, *, mask=None,
                     patch_ratio=.02, patch_size=None, fixed_loc=None, steps=100,
                     alpha=8/255, eps=1., restarts=1, targeted=False, target_labels=None,
                     mean=(.5,.5,.5), std=(.5,.5,.5), seed=None, return_info=False,
                     cosine=False):
    """목적함수는 (점수[B], logits[B,C], 추가성공조건[B]|None)를 반환한다.

    - `eps=1.0`, `alpha=8/255`는 **원래 [0,1] 픽셀 공간** 단위다(정규화 공간이 아님).
    - `steps=100`, `restarts=1`, `seed=None`. 첫 시작은 원본이고 추가 restart는 예산 내 무작위
      시작이다.
    - `mask=`로 HW/BHW/B1HW/11HW 0/1 마스크를 직접 전달하면 해당 영역만 수정한다(다중 영역 가능).
      mask가 없으면 `patch_ratio=.02`로 정사각형 크기를 정하거나 `patch_size=32`/`(h,w)`로
      직접 지정한다.
    - `fixed_loc=None`: 표본별 무작위 위치. `'center'`: 중앙. `(top,left)`: 지정한 픽셀 위치.
      restart 간 위치는 고정된다.
    - mask 밖은 정규화된 입력과 **정확히 같은 값**으로 유지한다.
    - `targeted=True, target_labels=...`는 CE 모드에서 지원한다(기본은 untargeted).
    - 각 표본에서 성공 후보를 우선 보존하고, 동일 성공 상태에서는 목적함수가 가장 큰 후보를
      반환한다 — 모든 표본의 성공을 보장하지 않는다(`objective`의 최종 iterate가 아니라
      과정 중 최선의 후보가 반환될 수 있음).
    """
    if steps < 1 or restarts < 1 or alpha <= 0 or not 0 < eps <= 1:
        raise ValueError('steps/restarts/alpha는 양수, eps는 (0,1]이어야 합니다.')
    with attack_context(model, seed, device):
        original, labels, pixels, mu, sigma = prepare(images, labels, device, mean, std)
        support = make_patch_mask(original, patch_ratio, patch_size, fixed_loc, mask)
        targets = None if target_labels is None else target_labels.to(device=device, dtype=torch.long)
        if targeted and (targets is None or targets.shape != labels.shape):
            raise ValueError('표본별 target_labels가 필요합니다.')
        lower, upper = (pixels-eps).clamp(0,1), (pixels+eps).clamp(0,1)
        best = pixels.clone()
        best_score = pixels.new_full((len(pixels),), -float('inf'))
        best_success = torch.zeros(len(pixels), device=device, dtype=torch.bool)
        for restart in range(restarts):
            current = pixels.clone() if restart == 0 else lower + torch.rand_like(pixels)*(upper-lower)
            current = torch.where(support, current, pixels)
            for step in range(steps+1):
                current = current.detach().requires_grad_(True)
                normalized = torch.where(support, (current-mu)/sigma, original)
                score, logits, extra_success = objective(normalized, labels, targets, support)
                if score.shape != labels.shape or not torch.isfinite(score).all():
                    raise ValueError('공격 목적함수는 유한한 표본별 B 점수를 반환해야 합니다.')
                success = logits.argmax(-1).eq(targets) if targeted else logits.argmax(-1).ne(labels)
                if extra_success is not None:
                    success = success & extra_success
                improve = (success & ~best_success) | ((success == best_success) & (score.detach() > best_score))
                best[improve] = current.detach()[improve]
                best_score[improve] = score.detach()[improve]
                best_success[improve] = success[improve]
                if step == steps:
                    break
                grad = torch.autograd.grad(score.sum(), current)[0]
                if not torch.isfinite(grad).all():
                    raise ValueError('비유한 공격 gradient가 발생했습니다.')
                rate = alpha * (.5*(1+math.cos(math.pi*step/steps)) if cosine else 1.)
                current = torch.minimum(torch.maximum(current.detach()+rate*grad.sign(), lower), upper)
                current = torch.where(support, current, pixels)
        adv = torch.where(support, (best-mu)/sigma, original).detach()
        info = {'mask': support.detach(), 'success': best_success.detach(), 'objective': best_score.detach()}
        return (adv, info) if return_info else adv


@contextmanager
def attention_terms(model, reducer, layers=None):
    """block별 forward hook으로 pre-softmax Q/K(q_norm/k_norm 적용, softmax 이전)를
    `reducer(q, k, scale)`에 넘겨 미분 가능한 값을 모은다. `layers=None`은 전체 블록,
    직접 지정하면 **0-based** 인덱스다. CLS prefix가 정확히 하나인 일반 timm ViT용 hook이며
    Swin/DETR/register token/동적 patch 격자는 지원하지 않는다."""
    if getattr(model, 'num_prefix_tokens', 1) != 1:
        raise ValueError('현재 공격은 CLS prefix가 하나인 ViT만 지원합니다.')
    layers = tuple(range(len(model.blocks))) if layers is None else tuple(layers)
    if not layers or len(set(layers)) != len(layers) or any(i < 0 or i >= len(model.blocks) for i in layers):
        raise ValueError('layers는 중복 없는 유효한 0-based block 인덱스여야 합니다.')
    terms, handles = {}, []
    def make_hook(layer_i):
        def hook(module, inputs, output):
            x = inputs[0]
            batch, count, channels = x.shape
            qkv = module.qkv(x).reshape(batch, count, 3, module.num_heads, channels//module.num_heads)
            q, k, _ = qkv.permute(2,0,3,1,4).unbind(0)
            q = module.q_norm(q) if hasattr(module, 'q_norm') else q
            k = module.k_norm(k) if hasattr(module, 'k_norm') else k
            terms[layer_i] = reducer(q, k, module.scale)
        return hook
    try:
        for layer_i in layers:
            handles.append(model.blocks[layer_i].attn.register_forward_hook(make_hook(layer_i)))
        yield terms
    finally:
        for handle in handles:
            handle.remove()
