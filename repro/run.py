"""재현 파이프라인 러너.

    python3 repro/run.py --list                 단계와 상태를 본다
    python3 repro/run.py --plan safety          그 단계까지 무엇이 필요한지
    python3 repro/run.py safety                 그 단계를 (필요하면 상류부터) 돈다
    python3 repro/run.py --from 5               tier 5 이상을 다시 돈다

기본은 **이미 있는 산출물을 다시 만들지 않는다**.  상류가 더 새것이면 낡은
것으로 표시하고, --force 가 없으면 돌지 않고 알려만 준다.  캐시 만드는 데
세 시간 넘게 걸리므로 자동 재실행은 위험하다.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, HERE)
from stages import STAGES, EXPLORATORY          # noqa: E402

BY_ID = {s['id']: s for s in STAGES}


def path_of(p):
    return p if os.path.isabs(p) else os.path.join(ANALYSIS, p)


def newest(paths):
    """존재하는 것 중 가장 최근 mtime.  하나도 없으면 None."""
    ts = []
    for p in paths:
        q = path_of(p)
        if os.path.isdir(q):
            for root, _, fs in os.walk(q):
                ts += [os.path.getmtime(os.path.join(root, f)) for f in fs]
        elif os.path.exists(q):
            ts.append(os.path.getmtime(q))
    return max(ts) if ts else None


def oldest_output(s):
    ts = []
    for p in s['outputs']:
        q = path_of(p)
        if os.path.isdir(q):
            fs = [os.path.join(r, f) for r, _, g in os.walk(q) for f in g]
            if not fs:
                return None
            ts.append(min(os.path.getmtime(f) for f in fs))
        elif os.path.exists(q):
            ts.append(os.path.getmtime(q))
        else:
            return None
    return min(ts) if ts else None


def status(s):
    """주의: 'stale' 은 mtime 비교일 뿐이다.

    이 저장소의 단계들은 셀 하나씩 순서대로 파일을 쓰므로, 두 단계를
    번갈아 돌리면 상류의 마지막 셀이 하류의 첫 셀보다 새것이 된다 —
    내용은 같은데 낡음으로 뜬다 (측정: temp_factor 는 다시 지어도
    바이트 단위로 동일했다).  그래서 'stale' 은 "다시 돌려라" 가 아니라
    "내용을 확인해 볼 것" 으로 읽어야 한다.
    """
    out = oldest_output(s)
    if out is None:
        return 'missing'
    inp = newest(s['inputs'])
    if inp is not None and inp > out + 1.0:
        return 'stale'
    return 'ok'


def upstream(target):
    """target 이 필요로 하는 단계들을 tier 순서로."""
    want = {target}
    changed = True
    while changed:
        changed = False
        for s in STAGES:
            if s['id'] not in want:
                continue
            for i in s['inputs']:
                for t in STAGES:
                    if t['id'] in want:
                        continue
                    if any(os.path.normpath(o).startswith(os.path.normpath(i))
                           or os.path.normpath(i).startswith(os.path.normpath(o))
                           for o in t['outputs']):
                        want.add(t['id'])
                        changed = True
    return [s for s in STAGES if s['id'] in want]


def show(ss):
    mark = {'ok': 'OK  ', 'stale': 'mtime', 'missing': '없음'}
    print(f"  {'단계':<14}{'tier':>5}{'상태':>7}{'분':>7}   설명", flush=True)
    print('  ' + '-' * 76, flush=True)
    tot = 0
    for s in ss:
        st = status(s)
        if st != 'ok':
            tot += s['minutes']
        b = ' (보드)' if s.get('board') else ''
        m = f"{s['minutes']}" + ('' if s.get('measured') else '~')
        print(f"  {s['id']:<14}{s['tier']:>5}{mark[st]:>7}{m:>7}{b}   "
              f"{s['why'].splitlines()[0][:44]}", flush=True)
    if tot:
        print(f"\n  다시 돌릴 것 합계 약 {tot} 분 "
              f"({tot/60:.1f} 시간).  ~ 는 재본 적 없는 추정.", flush=True)
        print("  'mtime' 은 상류가 더 새것이라는 뜻일 뿐 내용이 바뀌었다는 "
              "뜻이 아니다 — status() 의 주석을 볼 것.", flush=True)
    return tot


def run_one(s, dry):
    cmd = s['cmd'].replace('{py}', sys.executable)
    print(f"\n  == {s['id']}  ({s['minutes']} 분 예상)\n     {cmd}", flush=True)
    if dry:
        return True
    if s.get('board'):
        print('     보드가 필요한 단계다 — 수동으로 실행할 것', flush=True)
        return True
    t0 = time.time()
    rc = subprocess.call(cmd, shell=True, cwd=ANALYSIS)
    dt = (time.time() - t0) / 60
    print(f"     {'완료' if rc == 0 else '실패'}  실제 {dt:.1f} 분 "
          f"(기록 {s['minutes']} 분)", flush=True)
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('target', nargs='?')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--plan', metavar='STAGE')
    ap.add_argument('--from', dest='from_tier', type=int)
    ap.add_argument('--force', action='store_true',
                    help='상태가 ok 인 단계도 다시 돈다')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--exploratory', action='store_true',
                    help='임계 경로가 아닌 스크립트와 그 이유를 본다')
    a = ap.parse_args()

    if a.exploratory:
        print('  임계 경로가 아닌 것들 — 재현에 필요하지 않다\n')
        for k, v in EXPLORATORY.items():
            print(f'  {k}')
            for line in [v[i:i + 66] for i in range(0, len(v), 66)]:
                print(f'      {line}')
            print()
        return 0

    if a.list or (not a.target and not a.plan and a.from_tier is None):
        show(STAGES)
        print('\n  python3 repro/run.py <단계>      그 단계까지 돈다', flush=True)
        print('  python3 repro/run.py --plan <단계>  무엇이 필요한지만 본다',
              flush=True)
        return 0

    if a.plan:
        if a.plan not in BY_ID:
            print(f'  모르는 단계: {a.plan}')
            return 1
        show(upstream(a.plan))
        return 0

    if a.from_tier is not None:
        todo = [s for s in STAGES if s['tier'] >= a.from_tier]
    else:
        if a.target not in BY_ID:
            print(f'  모르는 단계: {a.target}.  --list 로 확인할 것')
            return 1
        todo = upstream(a.target)

    todo = [s for s in todo if a.force or status(s) != 'ok']
    if not todo:
        print('  전부 최신이다.  --force 로 강제할 수 있다.')
        return 0
    show(todo)
    for s in todo:
        if not run_one(s, a.dry_run):
            print(f'\n  {s["id"]} 에서 멈춘다.', flush=True)
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
