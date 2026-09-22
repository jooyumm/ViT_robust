"""공격받았을 때 정말 토큰 하나가 튀는지, 대표 이미지 1장 x 레이어 1개로 직접 확인.
x축: 토큰 인덱스(0=CLS, 1~196=patch), y축: CLS가 그 토큰에 주는 raw attention.
GPU 필요 없음, npz만 읽는다. --layer/--img_idx로 다른 예시도 볼 수 있음.
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
