"""Pre-draft QC — find stale numbers and retracted claims left in the docs.

verify.py checks the 38 numbers that live in tables.  The docs carry far more
numbers than that, and **some went stale this session when the adopted
configuration changed from A3 to A8.**  This sweeps them automatically and
produces a list for a human to judge.

It looks at three things.

  (1) stale     places where an A3-era number is still written.  The A8 value
                is shown beside it.
  (2) retracted places where a claim refuted or withdrawn this session is
                still written as an assertion.
  (3) orphan    numbers that are in no table, so verify.py cannot see them.

**It does not judge.**  It only says where to look — a stale value may be left
in deliberately as a comparison group.

    python3 repro/qc.py
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, 'docs')
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')

# (stale value, current value, what, may be deliberate)
STALE = [
    ('0.679', '0.683', 'discharge lambda(10s)  A3 -> A8', True),
    ('0.462', '0.470', 'discharge lambda(2s)   A3 -> A8', True),
    ('0.567', '0.586', 'charge lambda(10s)     A3 -> A8', True),
    ('0.544', '0.560', 'charge lambda(2s)      A3 -> A8', True),
    ('12.42', '5.99', 'feature update us      A3 -> A8', True),
    ('217', '214.8', 'period total us        A3 -> A8', True),
    ('0.0128', '0.0135', 'SOH RMSE   defects included -> excluded', True),
    ('+0.0010', '+0.0001', 'SOH bias   defects included -> excluded', True),
    ('0.594', '0.657', 'estimated-SOH discharge lambda', True),
    ('3.11 -> 3.35', '2.05 -> 2.17', 'SOC estimated-SOH price (circular bench)',
     True),
]

# Claims refuted or withdrawn this session.  (regex, what, where the
# correction is)
RETRACTED = [
    (r'of all things,?\s*on the dangerous',
     "the claim that the SOH arm's bias is on the dangerous side",
     'retracted in 30.12'),
    (r'R_volt small,?\s*raises the Kalman',
     'the claim that the R_volt schedule causes the estimated-SOH price',
     'refuted in 30.11 (the price is 0.00)'),
    # Narrowed, not withdrawn: 31.2 reconfirmed it AT stated lambda values.
    # So the assertion is only wrong when the condition is missing, and the
    # fourth element says what makes it right.  This beats another marker
    # word, because it names the specific qualifier rather than a tone.
    (r'exceedances are zero for every N from 1 to 192',
     "28.4's claim of no pack exceedance",
     'reconfirmed conditionally in 31.2 (charge on the 0.5 A tolerance)',
     r'lambda values|zero-exceedance criterion|0\.5 A tolerance|at the '
     r'lambda|λ'),
    # Some of these are guards against a claim being re-introduced rather
    # than matches against text that exists today.  A pattern with zero hits
    # is therefore not automatically a broken regex - but check which it is
    # before assuming, because a regex that never matches is exactly how the
    # Korean-era patterns went dead without anyone noticing.
    (r'voltage RMSE\s+(gives|says)\s+nothing\s+about\s+(the\s+)?SOP\s+rank',
     'the claim that voltage gives no rank at all',
     '32.6 — discharge 10 s preserves rank exactly (rho = 1.00)'),
    # --- withdrawn by the 2026-08-27 external audit -----------------------
    (r'calibrated leave-one-cell-out to zero exceedance',
     'the claim that lambda is calibrated to zero exceedance',
     '34.1 — the pooled median lets the evaluated cell set its own lambda; '
     'strict per-cell calibration leaves 1-2 exceedances'),
    (r'number to use in the paper is not 1\.51 %p but \*\*2\.05',
     'the claim that 2.05 %p is the disturbance average',
     '34.2 — 2.05 is the mean over seven rows including the undisturbed one; '
     'the six-disturbance mean is 2.14'),
    (r'physics[- ]aware CNN|residual[- ]aware CNN',
     'calling the SOH model physics- or residual-aware',
     '34.3 — soh_cnn.py is a plain two-layer 1D CNN'),
    (r'directly measured SOP label',
     'calling the UYPYDJ targets directly measured',
     '34.5 — 78 % of discharge labels extrapolate past 1.5x; they are '
     'pulse-derived'),
    # The negative lookbehinds matter: the corrected text says "not a pack
    # validation" and "never pack validation", and a bare match flagged those
    # as the very claim they retract.  Both prefixes are six characters, so a
    # fixed-width lookbehind is enough.
    (r'(?<!not a )(?<!never )pack validation\b',
     'calling the resampling simulation a pack validation',
     '34.10 — there is no pack hardware; it is a simulation sensitivity'),

    # --- third review round, 2026-08-31 ---------------------------------
    # These four phrasings were each removed for a stated reason, and each is
    # the kind that creeps back when a sentence gets rewritten.  The negative
    # lookbehinds let the correction blocks quote the retracted wording,
    # which is how this repository records what fell.
    (r'(?<!not )(?<!never )deployment-efficient equivalence',
     'claiming equivalence between the trim and the baselines',
     '37.3 — no margin was pre-specified and no noninferiority test was run; '
     'A8 places 3rd, 3rd, 2nd and 5th of six'),
    # Must name a universal claim.  "A8 beats A0 at every extrapolation
    # ceiling" is true and specific; the retracted claim was "all baselines".
    (r'A8 (?:outperform\w*|beats|is superior to) (?:all|every|the other) '
     r'baselines|outperform\w* all baselines',
     'claiming the trim outperforms the baselines',
     '37.3 — method_comparison.csv: 3 of 20 intervals separate, and all '
     'three are FFRLS'),
    (r'(?<!not a )(?<!never a )(?<!ever "a )four-parameter model',
     'calling the deployed trim a four-parameter model',
     '37.3 — the header ships 50 floats; say "four effective deployed '
     'coefficients"'),
    (r'corrupts the (?:onboard )?(?:safety |admission )?filter\b'
     r'|onboard admission filter fail',
     'describing the SOC-dependent evaluation inclusion rule as an onboard '
     'filter failure',
     '37.2 — no onboard filter was implemented or tested; the measured thing '
     'is the offline inclusion rule'),
    (r'for a production BMS|production-ready',
     'the production framing',
     '35.8 / 37.1 — withdrawn; oracle-state validation scores a row set the '
     'vehicle could not have selected'),
]

# Numbers that should come from a table but live only in the docs (invisible
# to verify.py)
ORPHAN_HINTS = [
    (r'0\.19\b|0\.16\s*[-–~]\s*0\.24', 'transfer ratio alpha — not in a table (32.3)'),
    (r'-0\.385|-0\.411|-0\.400|-0\.587|−0\.385|−0\.411|−0\.400|−0\.587',
     "28.3's correlation — not in a table"),
    # RESOLVED 2026-08-31: build_size.csv now carries both builds and
    # verify.py checks both (mcu.build.both, mcu.build.a8_only), so 142,060 is
    # table-backed.  143,932 and -1,872 B remain in the text only as the
    # superseded pre-rebuild values, explicitly marked as such in 35.4, and
    # flagging those is noise.  Kept as a comment so the check is not
    # silently dropped:
    #   (r'1,?872\s*B|142,?060|143,?932',
    #    'deployment build size — not in a table (33.6)'),
    (r'0\.98[-–]?1\.00×|0\.98\s*[-–~]\s*1\.00\s*×|0\.98×|1\.00×',
     "33.5's resistance ratio — not in a table"),
]

# A correction already sitting next to a hit.  Without these the same three
# places surface every run and hide anything genuinely new.
#   ~~...~~                strikethrough
#   [Retracted / [Updated  an explicit correction marker
#   wrote / pointed at /   the claim is being quoted in order to refute it
#   section's logic is
# Superseded by CORRECTION_MARKERS and DISCUSSION_MARKERS below, which are
# applied at block scope instead of over a +-4 line window.  The window was
# the defect: it had to be widened every time a correction paragraph wrapped
# differently, and it let an exemption in one block reach the next.  Kept
# only as the record of what it used to hold.
#   FILTERS = ('[Retracted', '[Updated', '[Added', '[Corrected', '[Audited',
#              'wrote', 'pointed at', "section's logic is", 'never be written',
#              'must not be called', 'overstates them', 'never pack validation')


def docs():
    """Every prose file a reader might take a claim from.

    This used to scan docs/ only, so README.md - the file most people
    actually read, and where the headline table lives - was never checked.
    A retracted claim sitting in the README would have passed silently.
    """
    for f in sorted(os.listdir(DOCS)):
        if f.endswith('.md'):
            yield f, open(os.path.join(DOCS, f),
                          encoding='utf-8').read().splitlines()
    for f in ('README.md', 'REPRODUCE.md', 'DATA.md'):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            yield f, open(p, encoding='utf-8').read().splitlines()


def retraction_hits(lines, blks, line_no, line):
    """Retracted claims asserted on one line.  The single code path.

    tests/test_qc_corpus.py calls this, not a copy of it.  The corpus used to
    reimplement the loop, so a mutation to the real one -- disabling the
    per-entry `unless`, for instance -- left every corpus case green.  A test
    that exercises its own reimplementation tests nothing.
    """
    out = []
    for entry in RETRACTED:
        pat, what, where = entry[0], entry[1], entry[2]
        unless = entry[3] if len(entry) > 3 else None
        m = re.search(pat, line)
        if not m:
            continue
        btxt, boff = block_for(line_no, blks)
        if unless and re.search(unless, btxt, re.I):
            # The claim was narrowed, not withdrawn, and this block carries
            # the qualifier that makes it true.
            continue
        if not is_assertion(line, m, btxt, boff):
            continue
        out.append((line_no, what, where, line.strip()[:72]))
    return out


def scan():
    stale_hits, retr_hits, orph_hits = [], [], []
    for fn, lines in docs():
        blks = blocks(lines)
        for i, ln in enumerate(lines, 1):
            for old, new, what, ok in STALE:
                if old not in ln:
                    continue
                blk = block_for(i, blks)[0].lower()
                recorded = (any(c in blk for c in CORRECTION_MARKERS)
                            or '~~' in ln
                            or all(x.strip().startswith('>')
                                   for x in blk.split('\n') if x.strip()))
                stale_hits.append((fn, i, old, new, what,
                                   ln.strip()[:72], recorded))
            for i2, what, where, txt in retraction_hits(lines, blks, i, ln):
                retr_hits.append((fn, i2, what, where, txt))
            for pat, what in ORPHAN_HINTS:
                if re.search(pat, ln):
                    orph_hits.append((fn, i, what, ln.strip()[:60]))
    return stale_hits, retr_hits, orph_hits


# A document that records retractions has to be able to quote them.  The
# question is not "does this phrase appear" but "is this line ASSERTING it".
#
# That is decided structurally, on the markdown block -- one contiguous run of
# non-blank lines, which is what a paragraph, a list item, a blockquote or a
# table row is.  A block is exempt when it is visibly a correction; otherwise
# a quoted phrase is exempt only if the block also discusses the claim.  The
# earlier version compared against a sliding window of one, then two, then
# three lines, which had to be widened every time a correction paragraph
# wrapped differently.  Block scope removes the tuning knob.
#
# tests/test_qc_corpus.py pins both directions: every shape that must be
# flagged, and every shape that must not.

CORRECTION_MARKERS = (
    '[retracted', '[updated', '[added', '[corrected', '[audited',
    '[superseded', '[narrowed', '[note —', '[note -', '[wrong',
    '[i was wrong', '[both earlier accounts were wrong',
)

# Words that mark a sentence as talking ABOUT a claim rather than making it.
DISCUSSION_MARKERS = (
    'claimed', 'claim', 'called', 'said', 'wrote', 'reported', 'offered',
    'withdrawn', 'withdraw', 'retracted', 'superseded', 'replaced',
    'no longer', 'not available', 'never', 'was wrong', 'is false',
    'corrected', 'narrowed', 'forbidden', 'must not', 'stopped',
    'fails if', 'left standing', 'stale', 'marked', 'read ',
    # migrated from the old FILTERS tuple, now applied at block scope
    'wrote', 'pointed at', "section's logic is", 'never be written',
    'must not be called', 'overstates them', 'never pack validation',
)


def blocks(lines):
    """(start_index, [lines]) for each contiguous run of non-blank lines."""
    out, cur, start = [], [], 0
    for i, ln in enumerate(lines):
        if ln.strip():
            if not cur:
                start = i
            cur.append(ln)
        elif cur:
            out.append((start, cur))
            cur = []
    if cur:
        out.append((start, cur))
    return out


def block_for(line_no, blks):
    """(text, offset of that line's start) for the block holding line_no.

    The text keeps its case; callers lowercase what they compare.  The offset
    is needed because quotedness has to be judged over the whole block: a
    correction routinely opens a quote on one line and closes it on the next,
    and a line-local test calls that unquoted and flags it.
    """
    for start, ls in blks:
        if start < line_no <= start + len(ls):
            j = line_no - start - 1
            return '\n'.join(ls), sum(len(x) + 1 for x in ls[:j])
    return '', 0


def is_assertion(line, match, block_text, line_offset=0):
    """False when this occurrence is a record of the claim, not a use of it.

    Four ways a block can be a record, all decidable without guessing:
      * it carries an audit marker such as [Corrected] or [SUPERSEDED]
      * it is a blockquote, which is how this repository writes corrections
      * it is about qc.py, which has to be able to name its own guards
      * the phrase is struck through
    Failing those, a phrase in double quotes is a record only if the block
    also carries a word that discusses the claim.  A bare assertion is never
    exempt, however many marker words are nearby.
    """
    block = block_text.lower()
    if '~~' in line:
        return False
    if any(m in block for m in CORRECTION_MARKERS):
        return False
    if all(x.strip().startswith('>')
           for x in block_text.split('\n') if x.strip()):
        return False
    if 'qc.py' in block:
        return False
    # Quotedness over the block, not the line: corrections wrap.
    at = line_offset + match.start()
    before, after = block_text[:at], block_text[at + (match.end() -
                                                     match.start()):]
    quoted = before.count('"') % 2 == 1 and '"' in after
    if quoted and any(m in block for m in DISCUSSION_MARKERS):
        return False
    return True


def table_numbers():
    """Values that live in tables — a doc number found here is verifiable."""
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

    standing = [h for h in st if not h[6]]
    recorded = [h for h in st if h[6]]
    print(f"  == (1) possibly stale values  {len(standing)} standing, "
          f"{len(recorded)} inside a correction block\n", flush=True)
    print(f"  {'file':<22}{'line':>6}  {'stale':>9} -> {'current':<9} what",
          flush=True)
    print('  ' + '-' * 88, flush=True)
    for fn, i, old, new, what, txt, _rec in standing:
        print(f"  {fn:<22}{i:>6}  {old:>9} -> {new:<9} {what}", flush=True)
    if recorded:
        print(f"\n  {len(recorded)} more sit inside a [Corrected] block, a "
              f"blockquote or a strikethrough — those are the document doing "
              f"its job, not drift.  Run with --all to list them.", flush=True)
    if not st:
        print('    none', flush=True)

    print(f"\n  == (2) retracted or refuted claims still standing  {len(rt)}"
          f" places\n", flush=True)
    for fn, i, what, where, txt in rt:
        print(f"  {fn}:{i}", flush=True)
        print(f"    claim:      {what}", flush=True)
        print(f"    correction: {where}", flush=True)
        print(f"    text:       {txt}", flush=True)
    if not rt:
        print('    none — the corrections are in place, or the claim is no '
              'longer written as an assertion', flush=True)

    print(f"\n  == (3) numbers verify.py cannot see (not in any table)  "
          f"{len(orp)} places\n", flush=True)
    seen = set()
    for fn, i, what, txt in orp:
        if what in seen:
            continue
        seen.add(what)
        print(f"  {what}", flush=True)
        print(f"    first occurrence: {fn}:{i}", flush=True)

    tv = table_numbers()
    print(f"\n  For reference: {len(tv)} distinct values live in tables.  "
          f"verify.py checks 38 of them.", flush=True)
    print("\n  This list is 'where to look', not 'what to fix' — values left "
          "in deliberately as comparison groups also show up.", flush=True)


if __name__ == '__main__':
    main()
