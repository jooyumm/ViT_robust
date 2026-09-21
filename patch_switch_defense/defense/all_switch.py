"""
defense/all_switch.py — "전체 전환" 방어 메커니즘.

메커니즘은 의도적으로 이만큼 단순하다: 탐지기가 flag한 이미지는 P8 모델로 통째 다시
분류한다. 별도 정렬/접합 메커니즘이 없다 — local_switch(defense/local_switch.py)와의
대비를 위해 존재하는 비교 기준선(NARRATIVE_16_17_18.md, README 참고).
"""
import torch


@torch.no_grad()
def apply_all_switch(model8, images):
    """flag된 이미지를 P8로 통째 재분류. (B,) 예측 logits — 그 이상도 이하도 아님."""
    return model8(images)
