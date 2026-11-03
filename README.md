# ViT_robust — ViT 패치 공격 강건성 연구 (프로젝트 모음)

패치 크기 비교 실험과 attention 기반 공격 탐지·국소 방어 코드를 관리한다.

## 프로젝트 구성

```
00_patch_size_tradeoff/     패치 크기별 강건성 비교 실험. (00_patch_size_tradeoff/README.md)
01_attention_characterization/   attention 분석 및 공격 탐지 탐색. (01_attention_characterization/README.md)
02_detection_defense/    패치 집중도 탐지, 국소 P8 방어. (02_detection_defense/README.md)
```

## 공유 `src/` (모델/데이터셋/공격)

파일별 역할, 공격 예산 단위, 수정한 오류와 남은 한계는 [src/README.md](src/README.md)에 정리했다.

`models.py`(timm ViT 로드), `dataset.py`(ImageNet-1k val 로더), `attacks/{patch_fool.py,
lavan.py, attention_fool.py}`, `attack_utils.py`(공격 공통 엔진)는
세 프로젝트에서 실제로 사용하는 공통 코드다. 각 프로젝트는 `sys.path`에 `ViT_robust`(이 폴더)를
추가해 `from src.models import ...` 식으로 직접 import한다. 공격 함수 각각의 사용법·옵션은
`src/attacks/__init__.py`와 각 공격 파일의 모듈 docstring 참고.

공격 함수는 모두 `src/attacks/` 한 곳에서 구현하고 `from src.attacks import ...`로 가져온다. PGD·PatchFool·LaVAN·Attention-Fool의 별도 프로젝트 사본은 두지 않는다. 실험마다 선택하는 공격·면적·토큰 수 등 파라미터는 다를 수 있다. `00_patch_size_tradeoff/metrics.py`는 00 전용 CA/RA/ASR 집계라 해당 프로젝트에 둔다.

## 탐지와 국소 방어

`02_detection_defense/region_switch/`에 탐지기, P16→P8 영역 교체,
세 조건 비교 실행 코드와 `results/comparison.png`를 모았다. 실행 방법과 결과는
[설명 문서](02_detection_defense/README.md)를 참고한다.

고정 500장 metadata는 `data/gt/imagenet500.json`, 공유 cohort·해시·패치 좌표는 `src/{cohort,provenance,patches}.py`에 있다. 01의 attention 집계·그림은 01에, 두 방어의 공통 평가는 `02_detection_defense/defense_common/`에 둔다. [평가 조건·수정 사항](docs/evaluation_audit.md)에서 논문 비교와 검증 범위를 확인할 수 있다.

공통 데이터는 `data/{clean,attacks,gt,attention}/`에 역할별로 구분한다. 공통 GT 코드는 `src/ground_truth/`, 데이터 읽기·준비는 `src/data/`에 두며 02는 저장된 공격본을 재사용한다. [데이터 구조](data/README.md)를 참고한다.
