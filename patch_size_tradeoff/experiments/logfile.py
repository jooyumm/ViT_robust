"""실험 로그 파일 저장 관련 유틸 (main.py에서 분리)."""
import os
import sys
from datetime import datetime


class Tee:
    """stdout에 쓰는 걸 파일에도 동시에 쓴다."""
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def make_log_path(args):
    """
    results/logs/result_{attack}_P{patch_sizes}[_tag-{tag}]_seed{seed}_{timestamp}.txt
    seed를 파일명에 넣어야 다중 시드 결과를 구분해서 mean±std로 집계할 수 있다
    (visualize/results_io.py가 이 이름 규칙을 파싱함).
    """
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    attack_str = '_'.join(args.attacks)
    patch_str  = 'P' + '_'.join(str(p) for p in sorted(args.patch_sizes))
    tag_str    = f'_tag-{args.tag}' if args.tag else ''
    fname = f'result_{attack_str}_{patch_str}{tag_str}_seed{args.seed}_{timestamp}.txt'
    return os.path.join(args.log_dir, fname)