"""
results_io.py — 시각화 스크립트 공용 로그 파싱 모듈

results/logs/result_{attack}_P{...}[_tag-{tag}]_seed{n}_{timestamp}.txt 를 읽어
(attack, P) -> {'CA':[...], 'RA':[...], 'ASR':[...], 'seeds':[...]} 형태로 모은다.

주의 1: SLURM job이 실패해서 재시도되면 같은 (attack, patch_sizes, tag, seed) 조합에 대해
로그 파일이 여러 개(타임스탬프만 다름) 쌓일 수 있다. 이때 전부 다 읽으면 같은 시드가
중복 집계된다 — 그래서 조합별로 **가장 최신 로그 하나만** 사용한다.

주의 2: baseline과 다른 설정(예: 면적 통제 ablation처럼 num_patch를 바꾼 실험)은
main.py --tag 옵션으로 구분해서 저장한다. collect_all_logs()는 기본적으로
tag='' (태그 없는 baseline)만 모으므로, ablation 로그가 baseline 집계에 섞여
들어가지 않는다. ablation 결과를 보려면 tag=원하는태그 로 명시적으로 불러온다.
"""
import re, os, glob

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'results')
LOG_DIR   = os.path.join(RESULTS_DIR, 'logs')       # 실험 원본 텍스트 로그
FIG_DIR   = os.path.join(RESULTS_DIR, 'figures')    # 그래프 전부 (파일명 번호로 구분)
TABLE_DIR = os.path.join(RESULTS_DIR, 'tables')     # 표 전부 (파일명 번호로 구분)

for _d in (FIG_DIR, TABLE_DIR):
    os.makedirs(_d, exist_ok=True)

ATK_LABEL = {'pgd': 'PGD', 'lavan': 'LaVAN', 'patch_fool': 'PatchFool'}
ATK_ORDER = ['pgd', 'lavan', 'patch_fool']
P_ORDER   = [8, 16, 32]

# 파일명 패턴: result_{attack}_P{nums}[_tag-{tag}]_seed{n}_{timestamp}.txt
FNAME_PAT = re.compile(
    r'^result_(.+?)_P([\d_]+)(?:_tag-([A-Za-z0-9]+))?_seed(\d+)_(\d{8}_\d{6})\.txt$')


def parse_filename(fname):
    m = FNAME_PAT.match(fname)
    if not m:
        return None, [], '', None, None
    attack    = m.group(1)
    patches   = [int(p) for p in m.group(2).split('_') if p]
    tag       = m.group(3) or ''
    seed      = int(m.group(4))
    timestamp = m.group(5)
    return attack, patches, tag, seed, timestamp


def parse_log_content(path, attack, p_values):
    with open(path) as f:
        content = f.read()
    results = {}
    for p in p_values:
        pat = (rf"Attack:\s*{re.escape(attack)}\s*\|\s*P={p}.*?"
               rf"── Overall.*?"
               rf"CA\s+\(Clean Accuracy\)\s*:\s*([\d.]+)%\s*"
               rf"RA\s+\(Robust Accuracy\)\s*:\s*([\d.]+)%\s*"
               rf"ASR\s+\(Attack Success\)\s*:\s*([\d.]+)%")
        m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        if m:
            results[p] = {
                'CA':  float(m.group(1)),
                'RA':  float(m.group(2)),
                'ASR': float(m.group(3)),
            }
    return results


def collect_all_logs(log_dir=None, verbose=True, tag=''):
    """
    (attack, P) -> {'CA': [...], 'RA': [...], 'ASR': [...], 'seeds': [...]}
    tag='' (기본) 이면 baseline(태그 없는) 로그만 모은다.
    특정 ablation을 보고 싶으면 tag='areamatch16' 처럼 명시한다.

    dedup은 (attack, patch_sizes 조합, seed) 단위가 아니라 (attack, 개별 P, seed) 단위로 한다 —
    예전에 --patch_sizes 16 32로 한 번에 돌린 파일과, 이후 --patch_sizes 16만 단독으로 돌린 파일은
    "패치 조합"이 달라서 서로 다른 키가 되어 버리는데, 그러면 두 파일 모두 P=16 결과를
    (중복으로) 기여하게 된다. 개별 P마다 가장 최신 로그 하나만 쓰도록 해야 이걸 막을 수 있다.
    """
    log_dir = log_dir or LOG_DIR
    all_logs = sorted(glob.glob(os.path.join(log_dir, 'result_*.txt')))

    latest_by_key = {}  # (attack, P, seed) -> (path, timestamp)
    n_matched = 0
    for path in all_logs:
        fname = os.path.basename(path)
        attack, p_values, file_tag, seed, timestamp = parse_filename(fname)
        if attack is None:
            if verbose:
                print(f"  [스킵] 파일명 패턴 불일치 (구버전 로그로 추정): {fname}")
            continue
        n_matched += 1
        if file_tag != tag:
            continue
        for p in p_values:
            key = (attack, p, seed)
            prev = latest_by_key.get(key)
            if prev is None or timestamp > prev[1]:
                latest_by_key[key] = (path, timestamp)

    n_dup = sum(1 for path in all_logs
               if FNAME_PAT.match(os.path.basename(path))
               and parse_filename(os.path.basename(path))[2] == tag
               for _ in parse_filename(os.path.basename(path))[1]) - len(latest_by_key)
    if verbose and n_dup > 0:
        print(f"  [중복 제거] 재시도/구성 변경으로 쌓인 구버전 로그 {n_dup}개는 제외하고 "
              f"(attack, P, seed)별 최신 로그만 사용 (tag='{tag}')")

    # path -> 이 파일에서 실제로 채택된 (attack, P, seed) 목록
    by_path = {}
    for (attack, p, seed), (path, _ts) in latest_by_key.items():
        by_path.setdefault(path, []).append((attack, p, seed))

    data = {}
    for path, entries in by_path.items():
        attack  = entries[0][0]
        p_list  = [p for (_, p, _) in entries]
        metrics = parse_log_content(path, attack, p_list)
        for (_, p, seed) in entries:
            if p not in metrics:
                continue
            m = metrics[p]
            bucket = data.setdefault(attack, {}).setdefault(
                p, {'CA': [], 'RA': [], 'ASR': [], 'seeds': []})
            for k in ('CA', 'RA', 'ASR'):
                bucket[k].append(m[k])
            bucket['seeds'].append(seed)
    return data


def gather_sweep_data(attack, p_values, x_to_tag):
    """
    연속값(x) 스윕 결과를 plot_sweep()/make_sweep_table()이 바로 쓸 수 있는 형태로 모은다.

    x_to_tag  : [(x값, tag), ...] — x값 오름차순으로 정렬해서 넘길 것
    반환      : {P: {'RA': [[x별 seed값들]], 'ASR': [[x별 seed값들]]}}
                (x_to_tag와 같은 순서, 해당 x에 데이터 없으면 빈 리스트)
    """
    tags_needed = {tag for _, tag in x_to_tag}
    sources = {tag: collect_all_logs(tag=tag, verbose=False) for tag in tags_needed}

    data_by_p = {p: {'RA': [], 'ASR': []} for p in p_values}
    for _, tag in x_to_tag:
        for p in p_values:
            bucket = sources[tag].get(attack, {}).get(p, {})
            data_by_p[p]['RA'].append(bucket.get('RA', []))
            data_by_p[p]['ASR'].append(bucket.get('ASR', []))
    return data_by_p