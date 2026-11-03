"""01/02 실험이 공유하는 고정 500장 선택(imagenet500.json)과 기본 run 경로(DEFAULT_DATASET)를 정의한다."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT / 'data/gt/imagenet500.json'
COHORT = json.loads(SAMPLE_FILE.read_text())
from src.data.paths import PAIRED_ATTENTION
DEFAULT_DATASET = PAIRED_ATTENTION

def validate_default_manifest(manifest):
    """이 run이 imagenet500.json과 같은 ID·순서·경로·모델로 만들어졌는지 확인한다."""
    expected = [s['index'] for s in COHORT['samples']]
    paths = [s['path'] for s in COHORT['samples']]
    if (manifest['sample_indices'] != expected or manifest['sample_paths'] != paths
            or manifest['model_sha256'] != COHORT['model_sha256']):
        raise ValueError('공통 500장 ID·순서·경로·모델이 다릅니다.')
    from src.data.paths import PATCHFOOL
    from src.provenance import sha256
    from src.data.protocol import vanilla_p16_config
    if (manifest.get('source_manifest_sha256')!=sha256(PATCHFOOL/'manifest.json')
            or manifest.get('patchfool_protocol')!=vanilla_p16_config()
            or manifest.get('attack_generation') is not False
            or manifest.get('attacks')!=['patchfool']):
        raise ValueError('01 어텐션은 02와 동일한 저장된 공식 PatchFool 공격에서 추출해야 합니다.')


def fixed_dataloader(batch_size=10, *, num_workers=4):
    """로컬 원본 500장을 저장된 순서·원래 ImageNet 정답으로 읽는다."""
    import csv
    import torch
    from torch.utils.data import DataLoader
    from src.dataset import _TRANSFORM
    from src.data.paths import GT
    from src.data.images import FixedImageDataset
    with (GT/'labels.csv').open() as stream:
        labels={int(r['sample_index']):int(r['label']) for r in csv.DictReader(stream)}
    dataset=FixedImageDataset(COHORT['samples'],labels,_TRANSFORM)
    return DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,
                      pin_memory=torch.cuda.is_available()),dataset
