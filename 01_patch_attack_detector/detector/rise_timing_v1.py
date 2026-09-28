"""
detector/rise_timing_v1.py — "언제 튀는지"(타이밍) 기반 탐지기. LSW(lsw_v3.py, "얼마나 튀는지")와는
다른 종류의 신호라서 보조 신호로 쓴다. 단독 성능은 LSW보다 낮다 — 자세한 수치는 아래 "실측 성능".

## 아이디어와 근거

정상 이미지에서 CLS가 최종적으로(12번 블록) 가장 많이 보는 patch는 이미 7~8번 블록까지 자기 최종
attention의 절반에 도달한다(정상 500장 중 495장). 공격받은 patch는 그 자리를 훨씬 늦게(중앙값
10~11번 블록) 차지한다 — 정상 500장 중 8번을 넘는 경우는 5장뿐이다. "몰리는 크기"가 아니라
"몰리는 시점" 자체가 정상에서는 거의 안 보이는 값이라는 관찰(개발 논의 중 지도교수 제안)에서 출발했다.

## 실측 성능과 한계 (반드시 읽을 것)

GT(실제 공격 patch)를 안다고 가정하고 그 patch만 봤을 때는 AUROC가 PatchFool 0.90, LaVAN 0.75로
LSW(0.88 / 0.85)와 비슷했다. 하지만 **실제 탐지기는 어느 patch를 볼지 스스로 찾아야 한다**(GT 없이).
자동으로 "12번 블록에서 attention이 가장 큰 patch 하나"를 고르면:

- PatchFool: AUROC 0.80 (그 patch가 실제 공격 patch와 일치하는 비율 62%)
- LaVAN: AUROC 0.60 (여러 patch에 퍼지는 공격이라 patch 하나만 보면 놓친다)

즉 **혼자 쓰면 LSW보다 약하다**(LSW: 0.88 / 0.85). 3x3 창으로 합쳐 보거나 LSW가 고른 창을 그대로
재사용해도 나아지지 않았다(LSW가 고른 창은 "후반에 급증"이라는 같은 기준으로 뽑혔기 때문에, 그 창의
타이밍을 다시 확인하면 정상 이미지에서도 이미 편향되어 있어 구분이 안 된다 — 탐지 기준과 확인 기준이
같으면 안 된다는 교훈).

그래서 이 파일은 **LSW 단독의 보조용**으로만 쓴다. 두 점수를 독립적으로 계산해 OR로 묶으면(각자
5% FPR로 보정), seed 21(500쌍) 기준 PatchFool 탐지가 71.9%→75.2%로(LSW가 놓친 92장 중 8장 추가
탐지), LaVAN은 67.9%→68.7%로(198장 중 2장) 아주 조금 오른다. 결합 FPR은 4.8%→5.0%.

## 사용

```python
timing = RiseTimingDetector(model); timing.fit(clean_images, fpr=0.05)
lsw_result = lsw_detector.detect(images); timing_result = timing.detect(images)
alarm = lsw_result['alarm'] | timing_result['alarm']   # 실측한 조합이 바로 이 OR
```
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lsw_v3 import cls_patch_rows   # noqa: E402 -- CLS 행 추출 hook은 LSW와 완전히 같다, 중복 구현하지 않는다


def rise_layer(rows):
    """rows (B,12,196) -> (B,) half-rise 레이어(1~12). 12번 블록에서 attention이 가장 큰 patch 하나를
    고르고, 그 patch의 궤적이 자기 최댓값의 절반에 처음 도달하는 블록(1-indexed)."""
    top_patch = rows[:, 11, :].argmax(-1)
    trajectory = torch.stack([rows[b, :, top_patch[b]] for b in range(len(rows))])
    peak = trajectory.max(-1, keepdim=True).values
    reached = trajectory >= 0.5 * peak
    return reached.float().argmax(-1) + 1


class RiseTimingDetector:
    def __init__(self, model, batch_size=50):
        self.model = model.eval()
        self.batch_size = batch_size
        self.threshold = None

    @torch.no_grad()
    def _rows(self, images):
        device = next(self.model.parameters()).device
        for start in range(0, len(images), self.batch_size):
            yield cls_patch_rows(self.model, images[start:start + self.batch_size].to(device))

    @torch.no_grad()
    def score(self, images):
        return torch.cat([rise_layer(rows).cpu() for rows in self._rows(images)])

    def fit(self, clean_images, fpr=0.05):
        scores = self.score(clean_images)
        self.threshold = float(torch.quantile(scores.float(), 1 - fpr))
        return self.threshold

    def detect(self, images):
        if self.threshold is None:
            raise RuntimeError('fit()으로 임계값을 먼저 정해야 합니다.')
        scores = self.score(images)
        return {'alarm': scores > self.threshold, 'score': scores}
