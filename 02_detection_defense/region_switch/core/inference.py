"""이미지 전용 탐지·국소 P8 추론 및 고정 변환 검증. CLI와 데이터 로딩을 포함하지 않는다."""
import numpy as np
import torch
import torch.nn.functional as F
from .attention import AttentionCapture
from .switch import (build_region_swap_batch,forward_with_replacement_mass,
                     full_forward_from_tokens,patch_embed_with_pos)


@torch.inference_mode()
def predict_defended(model, kernel8, images, detector):
    """입력은 이미지뿐이다. 공격 여부, 정상 짝, label, GT를 받지 않는다."""
    return forward_defended(model,kernel8,images,detector)


def forward_defended(model, kernel8, images, detector):
    """실제 hard routing. 적응형 공격은 선택된 경로의 입력 gradient만 미분한다.

    TV/attention 기반 이산 flag/region 자체에는 surrogate gradient를 만들지 않는다.
    탐지 결정은 매 forward마다 실제 입력으로 다시 계산한다.
    """
    capture = AttentionCapture(model)
    baseline_logits = capture(images)
    decision = detector.predict(images,capture.arrays())
    logits = baseline_logits.clone()
    active = np.flatnonzero(decision['flag'])
    if len(active):
        sequence = build_region_swap_batch(model,kernel8,images[active],
                    torch.as_tensor(decision['region'][active],device=images.device))
        logits[active] = forward_with_replacement_mass(model,sequence,key_weight=.25)
    return baseline_logits,logits,decision


@torch.inference_mode()
def verify_transform(model,kernel8,image):
    # 변환은 학습이 아니라 원래 Conv16와 bilinear 확대의 고정 행렬 곱이다.
    probe = image[:,:,:8,:8]
    reference = model.patch_embed.proj(F.interpolate(probe,size=(16,16),mode='bilinear',align_corners=False))
    transformed = F.conv2d(probe,kernel8,model.patch_embed.proj.bias)
    torch.testing.assert_close(reference,transformed,atol=2e-5,rtol=2e-4)
    patch = patch_embed_with_pos(model,image)
    cls = model.cls_token+model.pos_embed[:,:1]
    original = full_forward_from_tokens(model,torch.cat([cls,patch],dim=1))
    for region in range(4):
        row,col=divmod(region,2)
        keep=torch.ones(14,14,dtype=torch.bool,device=image.device)
        keep[row*7:(row+1)*7,col*7:(col+1)*7]=False
        replicated=torch.cat([cls,patch[:,keep.flatten()],patch[:,~keep.flatten()].repeat_interleave(4,dim=1)],dim=1)
        corrected=forward_with_replacement_mass(model,replicated,key_weight=.25)
        torch.testing.assert_close(original,corrected,atol=2e-4,rtol=2e-4)
        actual=build_region_swap_batch(model,kernel8,image,torch.tensor([region],device=image.device))
        assert actual.shape==(1,344,768)
        torch.testing.assert_close(actual[:,1:148],patch[:,keep.flatten()])
    print('검증: 고정 커널 변환, 네 사분면 질량 보존, 미교체 P16 토큰 일치',flush=True)


