"""
src/models.py — patch_size_tradeoff/patch_switch_defense/patch_attack_detector가 공유하는 모델 로딩 코드.

2026-09-21 이전에는 세 프로젝트 각자가 이 파일(과 dataset.py, attacks/patch_fool.py,
attacks/lavan.py)의 사본을 갖고 있었다("형제 프로젝트에 의존하지 않는다"는 원칙 —
patch_switch_defense/README.md, patch_attack_detector/README.md 참고). 세 사본이 완전히 같은 내용이라
편집 시 셋 다 따로 고쳐야 하는 게 비효율적이라 이 위치(ViT_robust/src/)로 합쳤다. 각
프로젝트는 이제 이 파일을 직접 import한다(각자의 sys.path 설정에서 ViT_robust를 추가).

MODEL_NAMES에 P32가 남아있는 건 patch_size_tradeoff가 P8/P16/P32를 다 쓰기 때문 —
patch_switch_defense/patch_attack_detector는 P8/P16만 쓰고 P32는 그냥 안 부른다(코드에서 막아둔 게 아니라
그 프로젝트들의 실험 스크립트가 32를 요청하지 않는 것뿐).
"""
import torch
import timm

MODEL_NAMES = {
    8:  'vit_base_patch8_224.augreg2_in21k_ft_in1k',
    16: 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
    32: 'vit_base_patch32_224.augreg_in1k',
}

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"  Device: Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"  Device: CPU")
    return device


def load_vit_model(patch_size, device):
    name = MODEL_NAMES.get(patch_size, f'vit_base_patch{patch_size}_224')

    try:
        print(f"  시도: {name}")
        model = timm.create_model(name, pretrained=True)

        # patch_size 검증 — timm 모델이 요청한 P와 실제로 다를 경우 스킵
        actual_p = getattr(model.patch_embed, 'patch_size', None)
        if actual_p is not None:
            # timm은 (H, W) tuple 또는 int 둘 다 사용
            actual_p = actual_p[0] if isinstance(actual_p, (tuple, list)) else actual_p
            if actual_p != patch_size:
                print(f"  ✗ patch_size 불일치: 요청={patch_size}, 실제={actual_p} → 스킵")
                return None

        model = model.to(device)
        model.eval()
        num_tokens = (224 // patch_size) ** 2
        print(f"  ✓ 로드 성공: {name}  "
              f"(P={patch_size}, tokens={num_tokens}, "
              f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M)")
        return model

    except Exception as e:
        print(f"  ✗ 실패: {e}")
        return None
