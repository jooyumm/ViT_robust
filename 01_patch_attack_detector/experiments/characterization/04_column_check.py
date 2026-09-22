"""
experiments/characterization/04_column_check.py — 03_token_bars.py가 CLS 관점에서 찾은
스파이크(예: L=12, image 0에서 patch 17로 0.63)가 "진짜 sink"인지 검증한다. PatchFool
논문 주장대로라면, CLS뿐 아니라 다른 토큰들도 공격받은 patch를 보고 있어야 한다 — 즉
patch 17이 "다들 보는 곳"이지 "CLS만의 특이 현상"이 아니어야 한다.

x축: 쿼리 토큰 인덱스(0=CLS, 1~196=patch, 즉 "누가 보는가"),
y축: 그 쿼리가 target_token 하나에게 주는 raw attention.
03_token_bars.py와 정반대 방향(그건 "CLS 하나가 어디를 보는가" = 고정 쿼리, 가변 키;
이건 "고정 키, 가변 쿼리"). GPU 필요 없음 — 00_extract_attention.py가 저장한
full_avg_repr(전체 197x197 행렬)만 읽는다.

사용법:
  python 04_column_check.py --num_samples 100 --layer 12 --img_idx 0 --target_token 17
  (target_token 기본값 없음 — 03_token_bars.py 결과에서 실제로 튄 토큰 번호를 넣을 것)
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # for `import data_io/plotting`

from data_io import load_attention_data
from plotting import plot_column_attention_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--layer', type=int, default=12, help='볼 레이어 1개 (1-indexed)')
    parser.add_argument('--img_idx', type=int, default=0, help='볼 대표 이미지 1장의 인덱스')
    parser.add_argument('--target_token', type=int, required=True,
                        help='"다들 이 토큰을 보는가"를 확인할 토큰 번호 (0=CLS, 1~196=patch). '
                             '03_token_bars.py에서 실제로 튄 토큰 번호를 넣을 것')
    args = parser.parse_args()

    out_dir = HERE.replace('/experiments/', '/results/', 1)
    npz_path = os.path.join(out_dir, f'00_attention_data_n{args.num_samples}.npz')
    assert os.path.exists(npz_path), \
        f"{npz_path} 없음 — 먼저 00_extract_attention.py --num_samples {args.num_samples} 실행할 것"
    all_metrics, repr_by_group, n_layers = load_attention_data(npz_path)

    if 'full_avg' not in repr_by_group['clean']:
        raise SystemExit(
            "이 npz는 full_avg_repr 필드가 없는 구버전입니다 — "
            "00_extract_attention.py를 다시 실행해서 npz를 새로 만드세요.")

    p = os.path.join(out_dir, f'04_column_check_L{args.layer}_img{args.img_idx}_'
                               f'tok{args.target_token}.png')
    plot_column_attention_line(repr_by_group, args.img_idx, args.layer, args.target_token, p)
    print(f"Saved: {p}")


if __name__ == '__main__':
    main()
