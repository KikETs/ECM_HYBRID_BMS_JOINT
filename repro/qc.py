"""초안 전 QC — 문서에 남은 낡은 수치와 철회된 주장을 찾는다.

verify.py 는 표에 있는 38 개를 확인한다.  문서에는 그보다 훨씬 많은 수치가
있고, **이번 세션에 채택 구성이 A3 -> A8 로 바뀌면서 그중 일부가 낡았다.**
자동으로 훑어 사람이 판단할 목록을 만든다.

세 가지를 본다.

  (1) 낡은 값   A3 시절 수치가 그대로 남아 있는 곳.  A8 값과 나란히 보인다.
  (2) 철회      이번 세션에 반박·철회한 주장이 여전히 단정형으로 적힌 곳.
  (3) 고아      표에 없어서 verify.py 가 못 보는 수치.

**판단은 하지 않는다.** 어디를 봐야 하는지만 낸다 — 낡은 값이 비교군으로
일부러 남아 있을 수도 있기 때문이다.

    python3 repro/qc.py
"""
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, 'docs')
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')

# (낡은 값, 지금 값, 무엇, 일부러 남아 있을 수 있는가)
STALE = [
    ('0.679', '0.683', '방전 lambda(10s)  A3 -> A8', True),
    ('0.462', '0.470', '방전 lambda(2s)   A3 -> A8', True),
    ('0.567', '0.586', '충전 lambda(10s)  A3 -> A8', True),
    ('0.544', '0.560', '충전 lambda(2s)   A3 -> A8', True),
    ('12.42', '5.99', '특징 갱신 us  A3 -> A8', True),
    ('217', '214.8', '주기 합계 us  A3 -> A8', True),
    ('0.0128', '0.0135', 'SOH RMSE  결함 포함 -> 제외', True),
    ('+0.0010', '+0.0001', 'SOH 편향  결함 포함 -> 제외', True),
    ('0.594', '0.657', '추정 SOH 방전 lambda', True),
    ('3.11 -> 3.35', '2.05 -> 2.17', 'SOC 추정 SOH 대가 (순환 벤치)', True),
]

# 이번 세션에 반박·철회한 주장.  (정규식, 무엇을, 어디에 정정이 있나)
RETRACTED = [
    (r'편향이?\s*\+0\.0010\s*으?로\s*하필\s*위험한\s*쪽',
     'SOH 팔 편향이 위험한 쪽이라는 주장', '30.12 에서 철회'),
    (r'R_volt\s*를?\s*작게\s*잡아.*과신',
     'R_volt 스케줄이 추정 SOH 대가의 원인이라는 주장', '30.11 에서 반박 (대가 0.00)'),
    (r'N=1\s*부터\s*192\s*까지\s*전부\s*초과\s*0',
     '28.4 의 팩 무초과 주장', '31.2 에서 조건부로 재확인 (충전은 허용 0.5A 기준)'),
    (r'전압\s*RMSE\s*는?\s*SOP\s*순위를\s*(안|못)\s*(준다|줌)',
     '전압이 순위를 전혀 안 준다는 주장', '32.6 — 방전 10s 는 순위 보존 (rho=1.00)'),
]

# 표에서 나와야 하는데 문서에만 있는 수치 (verify.py 가 못 보는 것)
ORPHAN_HINTS = [
    (r'0\.19\b|0\.16\s*~\s*0\.24', '전달비 alpha — 표에 없다 (32.3)'),
    (r'-0\.385|-0\.411|-0\.400|-0\.587', '28.3 의 상관 — 표에 없다'),
    (r'1,?872\s*B|142,?060|143,?932', '배치 빌드 크기 — 표에 없다 (33.6)'),
    (r'0\.98\s*배|1\.00\s*배', '33.5 의 저항 비 — 표에 없다'),
]


def docs():
    for f in sorted(os.listdir(DOCS)):
        if f.endswith('.md'):
            p = os.path.join(DOCS, f)
            yield f, open(p, encoding='utf-8').read().splitlines()


def scan():
    stale_hits, retr_hits, orph_hits = [], [], []
    for fn, lines in docs():
        for i, ln in enumerate(lines, 1):
            for old, new, what, ok in STALE:
                if old in ln:
                    stale_hits.append((fn, i, old, new, what, ln.strip()[:72]))
            for pat, what, where in RETRACTED:
                if not re.search(pat, ln):
                    continue
                # 정정이 이미 붙은 자리는 걸러낸다.  안 그러면 같은 셋이
                # 계속 떠서 새로 생긴 진짜 문제를 가린다.
                #   ~~...~~   취소선
                #   [철회 / [갱신 / 라고 냈고  -> 인용해서 반박하는 문맥
                ctx = '\n'.join(lines[max(0, i - 4):i + 6])
                if ('~~' in ln or '[철회' in ctx or '[갱신' in ctx
                        or '라고 냈고' in ctx or '라고 적었' in ctx
                        or '지목했다' in ctx):
                    continue
                retr_hits.append((fn, i, what, where, ln.strip()[:72]))
            for pat, what in ORPHAN_HINTS:
                if re.search(pat, ln):
                    orph_hits.append((fn, i, what, ln.strip()[:60]))
    return stale_hits, retr_hits, orph_hits


def table_numbers():
    """표에 있는 값들 — 문서 수치가 여기 있으면 verify 가 잡을 수 있다."""
    vals = set()
    for f in sorted(os.listdir(TABLES)) if os.path.isdir(TABLES) else []:
        for r in csv.DictReader(open(os.path.join(TABLES, f), encoding='utf-8')):
            for v in r.values():
                try:
                    vals.add(round(float(v), 3))
                except (TypeError, ValueError):
                    pass
    return vals


def main():
    st, rt, orp = scan()

    print(f"  == (1) 낡았을 수 있는 값  {len(st)} 곳\n", flush=True)
    print(f"  {'파일':<22}{'행':>6}  {'낡은값':>9} -> {'지금':<9} 무엇", flush=True)
    print('  ' + '-' * 88, flush=True)
    for fn, i, old, new, what, txt in st:
        print(f"  {fn:<22}{i:>6}  {old:>9} -> {new:<9} {what}", flush=True)
    if not st:
        print('    없음', flush=True)

    print(f"\n  == (2) 철회·반박된 주장이 남아 있는가  {len(rt)} 곳\n", flush=True)
    for fn, i, what, where, txt in rt:
        print(f"  {fn}:{i}", flush=True)
        print(f"    주장: {what}", flush=True)
        print(f"    정정: {where}", flush=True)
        print(f"    본문: {txt}", flush=True)
    if not rt:
        print('    없음 — 정정이 이미 반영됐거나 단정형으로 안 남아 있다',
              flush=True)

    print(f"\n  == (3) 표에 없어 verify 가 못 보는 수치  {len(orp)} 곳\n",
          flush=True)
    seen = set()
    for fn, i, what, txt in orp:
        if what in seen:
            continue
        seen.add(what)
        print(f"  {what}", flush=True)
        print(f"    처음 나오는 곳: {fn}:{i}", flush=True)

    tv = table_numbers()
    print(f"\n  참고: 표에 있는 고유 수치 {len(tv)} 개.  "
          f"verify.py 가 확인하는 것은 그중 38 개.", flush=True)
    print("\n  이 목록은 '봐야 할 곳' 이지 '고쳐야 할 곳' 이 아니다 — "
          "비교군으로 일부러 남긴 값도 걸린다.", flush=True)


if __name__ == '__main__':
    main()
