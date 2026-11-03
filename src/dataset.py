"""ImageNet validation 전처리와 재현 가능한 이미지 단위 무작위 표본 선택.

P8/P16 실험은 동일 전처리와 같은 seed의 같은 표본을 사용한다. 클래스별 균등
표본 추출이 아니다. 기존 _TRANSFORM과 get_dataloader의 반환 형식은 유지한다.
"""
from pathlib import Path
import timm
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from src.models import MODEL_NAMES

_CFG = timm.get_pretrained_cfg(MODEL_NAMES[16])
_MEAN, _STD = _CFG.mean, _CFG.std
_RESIZE = int(224 / _CFG.crop_pct)
if _CFG.interpolation != 'bicubic':
    raise ValueError('고정 표본의 기존 전처리는 bicubic 체크포인트 설정을 요구합니다.')

_TRANSFORM = transforms.Compose([
    transforms.Resize(_RESIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])
_IMAGENET_VAL = '/dataset/imagenet/val'


def get_dataloader(batch_size=32, num_samples=2000, seed=42, *,
                   dataset_root=None, num_workers=4, pin_memory=True):
    """정규화된 이미지 loader와 전체 ImageFolder를 반환한다.

    표본은 torch.randperm의 앞 num_samples개이며 데이터 수를 넘으면 전체를 사용한다.
    shuffle=False로 고정 순서를 유지한다. 경로와 worker 옵션은 이식성을 위한 keyword 인자다.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size는 양의 정수여야 합니다.')
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError('num_samples는 양의 정수여야 합니다.')
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError('num_workers는 0 이상의 정수여야 합니다.')
    root = Path(_IMAGENET_VAL if dataset_root is None else dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f'ImageNet val 경로 없음: {root}')
    dataset = ImageFolder(root=root, transform=_TRANSFORM)
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng).tolist()[:min(num_samples, len(dataset))]
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)
    print(f'  Dataset: ImageNet-1k val | samples={len(indices)} | classes={len(dataset.classes)}')
    return loader, dataset
