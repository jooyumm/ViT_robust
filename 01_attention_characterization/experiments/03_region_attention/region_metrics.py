"""03 전용 해상도 설정. 공유 패치 좌표는 src.patches에서 가져온다."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from src.cohort import DEFAULT_DATASET
from src.patches import patch_region_index

GRID = 14
BLOCKS = {'2x2': 7}
RUNS = [DEFAULT_DATASET]
