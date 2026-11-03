"""00의 최종 수치 JSON 저장·읽기. 실행 로그와 과거 텍스트 형식은 지원하지 않는다."""
from pathlib import Path
import json
import re


def read_results(folder, tags=None):
    """(attack, patch, tag, seed) -> (JSON path, metrics)."""
    results={}
    for path in sorted(Path(folder).glob('*.json')):
        record=json.loads(path.read_text())
        if tags is not None and record['tag'] not in tags:continue
        key=(record['attack'],record['patch_size'],record['tag'],record['seed'])
        if key in results:raise ValueError(f'중복 실험 결과: {key}')
        results[key]=(path,record['metrics'])
    return results


def save_result(folder, record):
    """조건·모델·seed별 JSON 하나를 성공한 실행의 최종 수치로 교체한다."""
    if not re.fullmatch(r'[A-Za-z0-9_-]*',record['tag']):
        raise ValueError('tag에는 영문·숫자·밑줄·하이픈을 사용하세요.')
    content=json.dumps(record,indent=2,allow_nan=False)+'\n'
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"{record['attack']}_P{record['patch_size']}_tag-{record['tag'] or 'baseline'}_seed{record['seed']}.json"
    pending=path.with_suffix('.pending')
    try:
        pending.write_text(content)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    return path
