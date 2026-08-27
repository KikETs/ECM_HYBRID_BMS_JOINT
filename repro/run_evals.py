"""모든 트림 판을 SOP 반전에 통과시킨다 — 논문 표의 원천.

행렬:  방향 2 x 트림 판 5 x SOH 입력 (정답 / 추정)

SOH 추정판은 채택 트림(A8)과 비교군 A3 에만 돌린다.  문헌 비교군(직접/축소/
RLS)까지 추정 SOH 로 돌릴 이유는 없다 — 그 판들의 결론은 정답 SOH 에서 이미
정해지고, 추정 SOH 는 그 위에 같은 방향으로 얹힌다.

analysis/ 에서 실행한다.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(os.path.dirname(HERE), 'analysis')
OUT = os.path.join(ANALYSIS, 'results', 'eval')
SOH_PRED = os.path.join(ANALYSIS, 'results', 'soh_pred.npz')

# (이름, 방전 트림 디렉터리, 충전 트림 디렉터리, 추정 SOH 도 돌릴지)
TRIMS = [
    ('a8',     'runs_trim_a8',      'runs_trim_a8_chg',      True),
    ('a3',     'runs_trim_v2',      'runs_trim_chg_v2',      True),
    ('direct', 'runs_trim_direct',  'runs_trim_direct_chg',  False),
    ('shrink', 'runs_trim_shrink',  'runs_trim_shrink_chg',  False),
    ('rls',    'runs_trim_rls',     'runs_trim_rls_chg',     False),
]

# 채택 평가 구성.  31.1 에서 16 절의 lambda 를 재현하는 것으로 확인했다.
#   방전  --trim-agg max,  허용 0.0 A  -> lambda 0.679 / 0.462  (A3)
#   충전  --trim-agg max,  허용 0.5 A  -> lambda 0.567 / 0.544  (A3)
AGG = 'max'


def jobs():
    out = []
    for name, dis, chg, do_est in TRIMS:
        for direction, trim in (('discharge', dis), ('charge', chg)):
            out.append((f'{name}_{direction[:4]}_oracle', direction, trim, None))
            if do_est:
                out.append((f'{name}_{direction[:4]}_est', direction, trim,
                            SOH_PRED))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=7,
                    help='동시 실행 수.  각 평가가 코어 하나를 쓴다.')
    ap.add_argument('--only', default=None, help='이름에 이 문자열이 든 것만')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    todo = [j for j in jobs() if not a.only or a.only in j[0]]
    print(f'  평가 {len(todo)} 개, 동시 {a.jobs} 개', flush=True)
    running = []
    failed = []

    def reap(block):
        while running and (block or len(running) >= a.jobs):
            nm, p = running.pop(0)
            rc = p.wait()
            print(f'    {"OK " if rc == 0 else "실패"} {nm}', flush=True)
            if rc:
                failed.append(nm)

    for nm, direction, trim, soh in todo:
        if not os.path.isdir(os.path.join(ANALYSIS, trim)):
            print(f'    건너뜀 {nm} — {trim} 없음', flush=True)
            continue
        cmd = [sys.executable, 'eval_sop_amps.py', '--direction', direction,
               '--trim', trim, '--trim-agg', AGG,
               '--out', os.path.join(OUT, f'{nm}.csv')]
        if soh:
            cmd += ['--soh-est', soh]
        reap(False)
        running.append((nm, subprocess.Popen(
            cmd, cwd=ANALYSIS, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)))
    reap(True)
    if failed:
        print(f'\n  실패 {len(failed)} 개: {", ".join(failed)}', flush=True)
        return 1
    print(f'\n  -> {OUT}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
