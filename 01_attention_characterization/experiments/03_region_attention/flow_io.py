"""영역 흐름 NPZ에서 필요한 배열만 읽고, 큰 배열은 임시 파일에 풀어 메모리 사용을 제한한다."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import zipfile
import numpy as np

@contextmanager
def open_flows(path, groups=('clean', 'patchfool')):
    """필요한 조건만 읽고 흐름 행렬은 디스크 매핑한다. 종료 시 매핑과 임시 파일을 정리한다."""
    mapped = []
    with tempfile.TemporaryDirectory(prefix='region_flow_') as temporary:
        try:
            with zipfile.ZipFile(path) as archive, np.load(path, allow_pickle=False) as small:
                data = {'sample_index': small['sample_index']}
                for group in groups:
                    key = group + '__flow'
                    destination = Path(temporary) / (key + '.npy')
                    # 압축 NPZ는 직접 부분 매핑할 수 없어 필요한 멤버만 임시 NPY로 순차 해제한다.
                    with archive.open(key + '.npy') as source, destination.open('wb') as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    array = np.load(destination, mmap_mode='r', allow_pickle=False)
                    mapped.append(array)
                    data[key] = array
                    for suffix in ('to_cls', 'cls_flow'):
                        data[group + '__' + suffix] = small[group + '__' + suffix]
                    if group != 'clean':
                        for suffix in ('attack_area', 'outcome'):
                            data[group + '__' + suffix] = small[group + '__' + suffix]
                yield data
        finally:
            for array in mapped:
                array._mmap.close()
