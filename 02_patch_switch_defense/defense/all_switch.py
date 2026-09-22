"""
defense/all_switch.py — "전체 전환" 방어 메커니즘.

메커니즘은 의도적으로 이만큼 단순하다: 탐지기가 flag한 이미지는 P8 모델로 통째 다시
분류한다. 별도 정렬/접합 메커니즘이 없다 — local_switch(defense/local_switch.py)와의
대비를 위해 존재하는 비교 기준선(README 참고).

이 함수는 탐지 여부를 스스로 판단하지 않는다 — "이 이미지가 flag됐다"는 판단은 호출부
(탐지기, 현재는 01_patch_attack_detector/detector/topk_mass_v1.py)가 이미 끝낸 뒤,
flag된 이미지만 여기 images로 넘어온다고 가정한다.
"""
import torch


@torch.no_grad()
def apply_all_switch(model8, images):
    """flag된 이미지를 P8로 통째 재분류.

    Args:
        model8: P8 ViT 모델 (eval 모드로 이미 전환돼 있다고 가정).
        images: (B, 3, H, W) — 이미 flag된 이미지만 넘길 것(호출부 책임).

    Returns:
        (B, num_classes) logits — 후처리(softmax/argmax) 없이 그대로 반환한다.
    """
    return model8(images)
