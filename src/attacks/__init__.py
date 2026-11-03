"""공유 공격 함수 모음.

모든 함수의 첫 네 인자는 `model, images, labels, device`다. `images`는 모델에 넣는 정규화된
BCHW 텐서, `labels`는 B 정답 라벨이다. 함수 호출만으로 파일·PDF·로그를 생성하지 않는다.
정확한 GT(실제 변경 영역)는 항상 원본과 공격 이미지의 픽셀 차이로 계산한다 — mask는
"변경이 허용된 영역"일 뿐 실제로 다 바뀌었다는 보장은 아니다. 정상부터 오답인 표본은
호출자가 별도로 분리해야 한다.

| 함수 | 역할 | 기본 반환값 |
|---|---|---|
| `pgd_attack` | 전역 L∞ PGD | `adv_images` |
| `patch_fool_attack` | PatchFool 일반·Mild·Sparse·다중 옵션 (patch_fool.py 참고) | `(adv_images, attention_list)` |
| `lavan_attack` | LaVAN, 고정 픽셀 면적 패치 (lavan.py 참고) | `adv_images` |
| `attention_fool_attack` | pre-softmax QK 목적함수 + 분류 손실 (attention_fool.py 참고) | `adv_images` |

Attention-Fool은 `return_info=True`를 주면 `(adv_images, info)`를 반환한다
(공통 엔진은 `attack_utils.projected_attack` — 그 안의 기본값/의미론 참고).
`info['mask']`는 B1HW bool 텐서(변경 허용 영역), `info['success']`는 표본별 성공 여부다.
PatchFool은 기존 `(adv_images, attention_list)` 반환을 유지하며 `return_info`는 지원하지 않는다.

공격은 입력만 최적화한다. 모델 가중치·기존 parameter gradient·호출 전 train/eval 모드를
보존하며, Attention-Fool 공격에서 seed를 명시하면 CPU/CUDA RNG도 격리한다.
예산 단위와 구현상 차이는 src/README.md를 참고한다.
"""
from .patch_fool import patch_fool_attack, patch_fool_config
from .pgd import pgd_attack
from .lavan import lavan_attack
from .attention_fool import attention_fool_attack

__all__ = [
    'pgd_attack', 'patch_fool_attack', 'patch_fool_config', 'lavan_attack',
    'attention_fool_attack',
]
