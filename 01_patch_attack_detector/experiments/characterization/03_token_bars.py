"""
experiments/characterization/03_token_bars.py — "공격받았을 때 어떤 레이어에서 어떤 토큰의
attention이 튀는가"를 자연 순서(정렬 안 함) 선 그래프로 직접 확인한다. x축: 토큰 인덱스
(0=CLS 자기 자신, 1~196=patch), y축: CLS가 그 토큰에 주는 raw attention. GPU 필요 없음 —
00_extract_attention.py가 저장한 npz만 읽는다.

**예시 한 장만 그린다** — 대표 이미지 1장 x 레이어 1개(기본 L=12, 앞선 sweep에서 신호가
가장 뚜렷했던 레이어). 12개 레이어를 전부 그리드로 늘어놓지 않는다: 이 스크립트의 목적은
"정말 튀는 값이 있는지"를 눈으로 확인하는 예시 하나면 충분하고, 레이어별 전수 비교는
이미 01_concentration.py(히스토그램)와 02_sink_position.py(정렬된 곡선)가 담당한다.
다른 레이어/이미지를 보고 싶으면 --layer/--img_idx로 바꿔서 다시 실행할 것.

02_sink_position.py의 sorted_mass_by_layer 그림과 다른 점: 그건 값을 내림차순으로 **정렬**해서
"몇 등까지 튀는가"의 모양(급경사 vs 완만)만 보여준다 — 어느 토큰인지는 사라진다. 여기서는
정렬하지 않고 원래 토큰 순서를 유지해서, 정말 특정 위치 하나가 튀는지 눈으로 그대로 확인하고,
CLS 자기 자신에게 주는 attention도 197번째 지점으로 포함한다(sorted_mass_by_layer는 patch
196개만 다룸 — detector.py의 실배포 경로가 CLS 자기 자신은 제외하기 때문).

범위 제한: 탐지 임계값/AUROC/flag 없음. 대표 이미지 1장에 대한 정성적 확인용이지, n=100
전체의 통계적 결론이 아니다(공격이 만드는 스파이크 위치는 이미지마다 다르므로, 여러
이미지를 평균 내면 스파이크가 씻겨나가 오히려 안 보인다 — 그래서 대표 이미지 단위로 본다).

사용법:
  python 03_token_bars.py --num_samples 100 --layer 12 --img_idx 0
  (같은 폴더에 00_extract_attention.py로 만든 00_attention_data_n100.npz가 있어야 하고,
   그 npz는 row_full_avg_repr 필드를 포함하는 버전이어야 함 — 없으면 00_extract_attention.py를
   다시 실행할 것)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # for `import data_io/plotting`

from data_io import load_attention_data
from plotting import plot_token_attention_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--layer', type=int, default=12, help='그릴 레이어 1개 (1-indexed)')
    parser.add_argument('--img_idx', type=int, default=0, help='그릴 대표 이미지 1장의 인덱스')
    args = parser.parse_args()

    out_dir = HERE.replace('/experiments/', '/results/', 1)
    npz_path = os.path.join(out_dir, f'00_attention_data_n{args.num_samples}.npz')
    assert os.path.exists(npz_path), \
        f"{npz_path} 없음 — 먼저 00_extract_attention.py --num_samples {args.num_samples} 실행할 것"
    all_metrics, repr_by_group, n_layers = load_attention_data(npz_path)

    if 'row_full_avg' not in repr_by_group['clean']:
        raise SystemExit(
            "이 npz는 row_full_avg_repr 필드가 없는 구버전입니다 — "
            "00_extract_attention.py를 다시 실행해서 npz를 새로 만드세요.")

    p = os.path.join(out_dir, f'03_token_bars_L{args.layer}_img{args.img_idx}.png')
    plot_token_attention_line(repr_by_group, args.img_idx, args.layer, p)
    print(f"Saved: {p}")


if __name__ == '__main__':
    main()
