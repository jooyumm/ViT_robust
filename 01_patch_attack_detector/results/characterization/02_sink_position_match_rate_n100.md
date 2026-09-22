# Clean -> attacked argmax position match rate, per image (n=100)

같은 이미지에서 clean일 때의 top-1 위치와 공격 적용 후 top-1 위치가 같은 patch인지 이미지별로 비교한 비율. 높으면 "기존 sink 증폭", 낮으면 "공격이 새 위치를 만듦"에 가깝다.

| L | match rate (clean vs PatchFool) | match rate (clean vs LaVAN) |
|---|---|---|
| 1 | 0.96 | 1.00 |
| 2 | 0.45 | 1.00 |
| 3 | 0.21 | 0.84 |
| 4 | 0.23 | 0.75 |
| 5 | 0.80 | 0.85 |
| 6 | 0.25 | 0.99 |
| 7 | 0.16 | 1.00 |
| 8 | 0.19 | 1.00 |
| 9 | 0.26 | 0.99 |
| 10 | 0.45 | 0.98 |
| 11 | 0.62 | 0.68 |
| 12 | 0.76 | 0.72 |
