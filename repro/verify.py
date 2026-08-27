"""논문에 인용된 수치가 지금 파이프라인에서 나오는지 확인한다.

    python3 repro/verify.py              전부 확인
    python3 repro/verify.py --only sop   이름에 sop 가 든 것만

expected.json 에 (값, 허용오차, 출처 절) 을 적어 둔다.  허용오차는 0 이
기본이다 — 이 파이프라인에 난수를 쓰는 곳은 팩 모사(씨앗 고정)와 트림
학습(씨앗 고정) 뿐이라 원칙적으로 재현이 정확해야 한다.  허용오차가 0 이
아닌 항목은 왜 그런지 이유를 함께 적는다.

**이 스크립트가 실패하면 논문의 그 수치를 고쳐야 한다.**  반대가 아니다.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')
EXPECTED = os.path.join(HERE, 'expected.json')


def read(name):
    p = os.path.join(TABLES, name)
    if not os.path.exists(p):
        return None
    return list(csv.DictReader(open(p, encoding='utf-8')))


def pick(rows, where, col):
    """where 의 모든 열이 일치하는 유일한 행에서 col 을 꺼낸다."""
    hit = [r for r in rows
           if all(str(r.get(k, '')) == str(v) for k, v in where.items())]
    if len(hit) != 1:
        raise LookupError(f'{len(hit)} 행이 맞았다 (1 이어야 함): {where}')
    return float(hit[0][col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--update', action='store_true',
                    help='현재 값으로 expected.json 을 다시 쓴다. '
                         '값이 바뀐 이유를 알 때만 쓸 것.')
    a = ap.parse_args()

    spec = json.load(open(EXPECTED, encoding='utf-8'))
    checks = [c for c in spec['checks']
              if not a.only or a.only in c['id']]

    cache = {}
    ok = bad = skip = 0
    out = []
    print(f"  {'항목':<34}{'기대':>10}{'실제':>10}{'허용':>8}  출처", flush=True)
    print('  ' + '-' * 78, flush=True)
    for c in checks:
        rows = cache.get(c['table'])
        if rows is None:
            rows = cache[c['table']] = read(c['table'])
        if rows is None:
            print(f"  {c['id']:<34}{'':>10}{'표 없음':>10}{'':>8}  "
                  f"{c['source']}", flush=True)
            skip += 1
            out.append(c)
            continue
        try:
            got = pick(rows, c['where'], c['column'])
        except LookupError as e:
            print(f"  {c['id']:<34}  {e}", flush=True)
            bad += 1
            out.append(c)
            continue
        tol = c.get('tol', 0.0)
        good = abs(got - c['value']) <= tol
        mark = '' if good else '   <-- 불일치'
        print(f"  {c['id']:<34}{c['value']:>10.3f}{got:>10.3f}{tol:>8.3f}  "
              f"{c['source']}{mark}", flush=True)
        ok += good
        bad += not good
        d = dict(c)
        if a.update:
            d['value'] = round(got, 4)
        out.append(d)

    if a.update:
        spec['checks'] = out
        json.dump(spec, open(EXPECTED, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        print(f'\n  expected.json 을 갱신했다 ({len(out)} 항목)', flush=True)
        return 0

    print(f'\n  일치 {ok}   불일치 {bad}   건너뜀 {skip}', flush=True)
    if skip:
        print('  건너뛴 것은 해당 단계를 아직 안 돌린 것이다 — '
              'repro/run.py 를 볼 것', flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
