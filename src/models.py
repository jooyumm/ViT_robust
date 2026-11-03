"""실험에서 사용하는 사전학습 ViT 체크포인트와 모델 로더.

각 patch 크기는 서로 다른 체크포인트다. patch 크기만 바꾼 동일 가중치 모델이 아니다.
모델은 eval 모드로 반환하지만 공격 입력 gradient 계산을 위해 parameter를 freeze하지 않는다.
"""
import timm
import torch

MODEL_NAMES = {
    8: 'vit_base_patch8_224.augreg2_in21k_ft_in1k',
    16: 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
}


def get_device():
    """기존 선택 순서(MPS → CUDA → CPU)를 유지한다."""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('  Device: Apple MPS')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'  Device: CUDA ({torch.cuda.get_device_name(0)})')
    else:
        device = torch.device('cpu')
        print('  Device: CPU')
    return device


def load_vit_model(patch_size, device, *, strict=False):
    """P8/P16 사전학습 모델을 로드한다.

    기본은 기존 API처럼 실패 메시지와 None을 반환한다. strict=True이면 원래 예외를
    전달하므로 실패 원인을 추적하기 쉽다. 네트워크 없는 실행은 HF_HUB_OFFLINE=1을 쓴다.
    """
    try:
        if patch_size not in MODEL_NAMES:
            raise ValueError(f'지원 patch 크기: {tuple(MODEL_NAMES)}; 요청: {patch_size}')
        name = MODEL_NAMES[patch_size]
        model = timm.create_model(name, pretrained=True)
        actual = model.patch_embed.patch_size
        actual = (actual, actual) if isinstance(actual, int) else tuple(actual)
        if actual != (patch_size, patch_size):
            raise ValueError(f'patch size 불일치: 요청={patch_size}, 실제={actual}')
        model = model.to(device)
        model.eval()
        print(f'로드 성공: {name} (P={patch_size}, tokens={model.patch_embed.num_patches}, '
              f'params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M)')
        return model
    except Exception as error:
        if strict:
            raise
        print(f'모델 로드 실패: {type(error).__name__}: {error}')
        return None
