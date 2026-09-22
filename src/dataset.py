import os
import timm
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.models import MODEL_NAMES

# 00_patch_size_tradeoff/02_patch_switch_defense/01_patch_attack_detector가 공유하는 데이터 로딩 코드
# (2026-09-21에 세 프로젝트 각자의 사본을 여기로 합침 — models.py 헤더 참고).
#
# 우리가 쓰는 체크포인트(vit_base_patch{8,16,32}_224 augreg 계열)는 학습 시
# 표준 ImageNet 정규화가 아니라 mean=std=0.5(inception 스타일)를 썼다.
# timm의 pretrained_cfg에서 직접 읽어와서 하드코딩 오차를 없앤다.
# (P=8/16/32 세 체크포인트 모두 동일한 설정을 공유함을 확인함)
_CFG = timm.get_pretrained_cfg(MODEL_NAMES[16])
_MEAN, _STD = _CFG.mean, _CFG.std
_RESIZE = int(224 / _CFG.crop_pct)   # crop_pct=0.9 -> 248
assert _CFG.interpolation == 'bicubic'

_TRANSFORM = transforms.Compose([
    transforms.Resize(_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

# ImageNet val 경로 
_IMAGENET_VAL = '/dataset/imagenet/val'

# 최대 샘플 수 제한 (ImageNet val은 50,000개), 단일 seed
def get_dataloader(batch_size=32, num_samples=2000, seed=42):

    assert os.path.exists(_IMAGENET_VAL), \
        f"ImageNet val 경로 없음: {_IMAGENET_VAL}"

    # ImageFolder가 synset 폴더명 기준으로 자동 매핑, remapping 불필요
    dataset = torchvision.datasets.ImageFolder(
        root=_IMAGENET_VAL,
        transform=_TRANSFORM
    )

    # 재현 가능한 균등 샘플링 (클래스당 동일 비율)
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm    = torch.randperm(len(dataset), generator=rng).tolist()
    indices = perm[:min(num_samples, len(dataset))]

    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 클래스 수 확인
    n_classes = len(dataset.classes)
    print(f"  Dataset: ImageNet-1k val | "
          f"samples={len(subset)} | classes={n_classes}")
    return loader, dataset