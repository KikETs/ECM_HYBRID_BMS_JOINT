"""Turn the numbers QC flagged as 'not in a table' into tables.

A number that lives only in the documents is invisible to verify.py.  This
session's errors came from exactly such places, so they are pulled into
tables.

  transfer ratio alpha   32.3 — how much of the low-current residual slope
                                transfers to high current
  weak-optimistic corr   28.3 — are weaker cells estimated more optimistically
  defect cycle R ratio   33.5 — was an HPPC logged at 0 C really at 0 C
  deployment build size  33.6 — how much smaller is the A8-only build

    python3 repro/run_extras.py
"""
import csv
import os
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
    """32.3's transfer ratio — alpha_f, alpha_s fitted leaving one cell out."""
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
    """28.3 — correlation of true |I*| with predicted/true.  Are weak cells
    more optimistic."""
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
    """33.5 — how many times the neighbours' resistance an HPPC logged at
    0 C shows.

    At a real 0 C it would be 3-5x.  A ratio of 1x means only the temperature
    log broke.

    Input provenance: results/cold_check/r_all_<cell>.csv is
    uypydj_hppc_resistance.py run **with the temperature-defect filter
    disabled** (the defect cycle is the thing being measured, so the normal
    filter would remove it), reduced to the discharge / tau = 10 s rows this
    check uses.  It is committed rather than rebuilt because rebuilding needs
    the 24 GB raw data.  These used to be read from /tmp, which a
    reproduction package must not depend on — the same defect build_soc_runs
    was created to remove.
    """
    rows = []
    for cell, tgt in (('BOOST', 1462), ('CC', 1500)):
        p = os.path.join(ANALYSIS, 'results', 'cold_check',
                         f'r_all_{cell}.csv')
        if not os.path.exists(p):
            print(f'    cold_ratio: missing {os.path.relpath(p, ROOT)}',
                  flush=True)
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


def _one_build(d):
    """(kind, text_B, bss_B) for one Build/<id> directory, or None."""
    p = os.path.join(d, 'size.txt')
    if not os.path.exists(p):
        return None
    ln = open(p, encoding='utf-8').read().splitlines()
    if len(ln) < 2:
        return None
    f = ln[1].split()
    # Decide which version this build is from its symbols (the directory
    # name alone does not say)
    elf = os.path.join(d, 'sop_bench.elf')
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
    return [kind, f[0], f[2]]


def build_size():
    """33.6 — size of the comparison build vs the deployment (A8-only) build.

    Both are reported.  The deployment figure is quoted in the text and in
    README, so leaving it out of the tables put a published number beyond the
    verifier's reach -- which is what qc.py flagged.
    """
    base = os.path.join(ROOT, 'mcu', 'fw_sop', 'Build')
    rows = []
    for sub in ('nmc_dst_cc', 'a8_only'):
        r = _one_build(os.path.join(base, sub))
        if r:
            rows.append(r)
    return (['build', 'text_B', 'bss_B'], rows)


def main():
    os.makedirs(TABLES, exist_ok=True)
    for name, fn in (('alpha', alphas), ('correlation', correlations),
                     ('cold_ratio', cold_ratio), ('build_size', build_size)):
        try:
            hdr, rows = fn()
        except Exception as e:                            # noqa: BLE001
            print(f'  skipped {name}: {type(e).__name__} {e}', flush=True)
            continue
        if not rows:
            print(f'  skipped {name}: no input', flush=True)
            continue
        out = os.path.join(TABLES, f'{name}.csv')
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(rows)
        print(f'  -> {name}.csv  ({len(rows)} rows)', flush=True)
        for r in rows[:4]:
            print(f'       {r}', flush=True)


if __name__ == '__main__':
    main()
