"""QC 가 '표에 없다' 고 지적한 수치들을 표로 만든다.

문서에만 있고 표에 없으면 verify.py 가 못 본다.  이번 세션의 오류가
정확히 그런 자리에서 나왔으므로 표로 뺀다.

  전달비 alpha        32.3 — 저전류 잔차 기울기가 고전류로 얼마나 전이되나
  약함-낙관 상관      28.3 — 약한 셀일수록 더 낙관적으로 추정되나
  결함 사이클 저항비   33.5 — 0 C 로 기록된 HPPC 가 정말 0 C 였나
  배치 빌드 크기      33.6 — A8 전용 빌드가 얼마나 작나

    python3 repro/run_extras.py
"""
import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
TABLES = os.path.join(ANALYSIS, 'results', 'tables')
sys.path.insert(0, ANALYSIS)
sys.path.insert(0, HERE)

CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST',
         'CC', 'CC_CELL2']


def alphas():
    """32.3 의 전달비 — 셀 하나씩 빼고 맞춘 alpha_f, alpha_s."""
    from sop_baseline_fill import fit_alpha
    import sop_trim
    rows = []
    for data, direction in (('cache/trim', 'discharge'),
                            ('cache/trim_chg', 'charge')):
        cells = sop_trim.load_cells(os.path.join(ANALYSIS, data))
        names = sorted(cells)
        for c in names:
            af, asl = fit_alpha([cells[o] for o in names if o != c])
            rows.append([direction, c, f'{af:.3f}', f'{asl:.3f}'])
    return (['direction', 'holdout', 'alpha_fast', 'alpha_slow'], rows)


def correlations():
    """28.3 — 참 |I*| 와 예측/참 의 상관.  약한 셀이 더 낙관적인가."""
    from run_safety import load, keep
    rows = []
    for direction, path in (('discharge', 'a8_disc_oracle'),
                            ('charge', 'a8_char_oracle')):
        d = load(os.path.join(ANALYSIS, 'results', 'eval', f'{path}.csv'))
        for tau in (10.0, 2.0):
            m = keep(d, tau)
            if m.sum() < 10:
                continue
            meas, pred = d['meas'][m], d['hyb'][m]
            r = float(np.corrcoef(meas, pred / meas)[0, 1])
            rows.append([direction, f'{tau:.0f}', int(m.sum()), f'{r:+.3f}'])
    return (['direction', 'tau_s', 'n', 'corr_meas_vs_ratio'], rows)


def cold_ratio():
    """33.5 — 0 C 로 기록된 HPPC 의 저항이 이웃 대비 몇 배인가.

    0 C 였다면 3~5 배여야 한다.  1 배면 온도 기록만 깨진 것이다.
    """
    rows = []
    for cell, tgt in (('BOOST', 1462), ('CC', 1500)):
        p = f'/tmp/r_all_{cell}.csv'
        if not os.path.exists(p):
            continue
        r = list(csv.DictReader(open(p, encoding='utf-8')))
        cy = np.array([int(float(x['cycle'])) for x in r])
        R = np.array([float(x['R_mOhm']) for x in r])
        tau = np.array([float(x['tau_s']) for x in r])
        dr = np.array([x['direction'] for x in r])
        m = (dr == 'discharge') & (np.round(tau, 1) == 10.0)
        cl = sorted(set(cy[m]))
        if tgt not in cl:
            continue
        i = cl.index(tgt)
        neigh = [c for c in (cl[i - 1], cl[i + 1]) if c != tgt]
        med = float(np.median(R[m & (cy == tgt)]))
        nm = float(np.median([np.median(R[m & (cy == c)]) for c in neigh]))
        rows.append([cell, tgt, f'{med:.2f}', f'{nm:.2f}', f'{med / nm:.2f}'])
    return (['cell', 'cycle', 'R_mOhm', 'neighbour_mOhm', 'ratio'], rows)


def build_size():
    """33.6 — 비교용 빌드와 배치용(A8 전용) 빌드의 크기."""
    p = os.path.join(ROOT, 'mcu', 'fw_sop', 'Build', 'nmc_dst_cc', 'size.txt')
    if not os.path.exists(p):
        return (['build', 'text_B', 'bss_B'], [])
    ln = open(p, encoding='utf-8').read().splitlines()
    if len(ln) < 2:
        return (['build', 'text_B', 'bss_B'], [])
    f = ln[1].split()
    # 현재 빌드가 어느 판인지 심볼로 판정한다 (파일명만으로는 알 수 없다)
    elf = os.path.join(os.path.dirname(p), 'sop_bench.elf')
    kind = 'unknown'
    if os.path.exists(elf):
        import subprocess
        nmbin = ('/opt/st/stm32cubeide_2.2.0/plugins/com.st.stm32cube.ide.mcu'
                 '.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100'
                 '.202602081740/tools/bin/arm-none-eabi-nm')
        if os.path.exists(nmbin):
            out = subprocess.run([nmbin, elf], capture_output=True,
                                 text=True).stdout
            kind = ('both' if ' T sop_feat_update\n' in out else 'a8_only')
    return (['build', 'text_B', 'bss_B'], [[kind, f[0], f[2]]])


def main():
    os.makedirs(TABLES, exist_ok=True)
    for name, fn in (('alpha', alphas), ('correlation', correlations),
                     ('cold_ratio', cold_ratio), ('build_size', build_size)):
        try:
            hdr, rows = fn()
        except Exception as e:                            # noqa: BLE001
            print(f'  건너뜀 {name}: {type(e).__name__} {e}', flush=True)
            continue
        if not rows:
            print(f'  건너뜀 {name}: 입력이 없다', flush=True)
            continue
        out = os.path.join(TABLES, f'{name}.csv')
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(rows)
        print(f'  -> {name}.csv  ({len(rows)} 행)', flush=True)
        for r in rows[:4]:
            print(f'       {r}', flush=True)


if __name__ == '__main__':
    main()
