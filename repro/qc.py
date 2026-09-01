"""Pre-draft QC — find stale numbers and retracted claims left in the docs.

verify.py checks the numbers that live in tables.  The docs carry far more
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
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, 'docs')
TABLES = os.path.join(ROOT, 'analysis', 'results', 'tables')

# (stale value, current value, what, may be deliberate)
# (old value, description, where the current value lives).  The CURRENT side
# is looked up in the tables, never written here: this list previously carried
# it as a literal and went stale itself -- it was telling readers to replace
# 217 with 214.8 after 214.8 had become 227.79, and 0.0128 with 0.0135 after
# the SOH arm became ridge at 0.0094.  A checker that hands out stale numbers
# is worse than no checker.
STALE_SPEC = [
    ('0.679',  'discharge lambda(10s)  A3 -> A8',
     ('safety_strict_oracle.csv', {'direction': 'discharge', 'tau_s': '10.0',
                                   'soh_arm': 'oracle'},
      'lambda_pooled_shipped', 3)),
    ('0.462',  'discharge lambda(2s)   A3 -> A8',
     ('safety_strict_oracle.csv', {'direction': 'discharge', 'tau_s': '2.0',
                                   'soh_arm': 'oracle'},
      'lambda_pooled_shipped', 3)),
    ('0.567',  'charge lambda(10s)     A3 -> A8',
     ('safety_strict_oracle.csv', {'direction': 'charge', 'tau_s': '10.0',
                                   'soh_arm': 'oracle'},
      'lambda_pooled_shipped', 3)),
    ('0.544',  'charge lambda(2s)      A3 -> A8',
     ('safety_strict_oracle.csv', {'direction': 'charge', 'tau_s': '2.0',
                                   'soh_arm': 'oracle'},
      'lambda_pooled_shipped', 3)),
    ('12.42',  'A8 feature update us',
     ('mcu.csv', {'stage': 'FEAT_A8'}, 'median_us', 2)),
    ('217',    'per-cycle total us',
     ('mcu_cycle.csv', {'case': 'median'}, 'cycle_total_us', 2)),
    ('0.0128', 'SOH RMSE',
     ('soh.csv', {'cell': 'ALL'}, 'rmse', 4)),
    ('+0.0010', 'SOH bias',
     ('soh.csv', {'cell': 'ALL'}, 'bias', 4)),
    ('0.594',  'estimated-SOH discharge lambda',
     ('safety_strict_est.csv', {'direction': 'discharge', 'tau_s': '10.0',
                                'soh_arm': 'est'},
      'lambda_pooled_shipped', 3)),
]


def _lookup(table, where, column, places):
    """The current value of a published number, read from its table."""
    p = os.path.join(TABLES, table)
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            if all(r.get(k) == v for k, v in where.items()):
                try:
                    return f'{float(r[column]):.{places}f}'
                except (KeyError, ValueError):
                    return None
    return None


def stale_pairs():
    """(old, current, description) with the current side resolved now."""
    out = []
    for old, what, spec in STALE_SPEC:
        cur = _lookup(*spec)
        if cur is None:
            continue
        if old.lstrip('+-').rstrip('0').rstrip('.') == \
                cur.lstrip('+-').rstrip('0').rstrip('.'):
            continue           # no longer a change; nothing to flag
        out.append((old, cur, what, True))
    return out


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
    # RESOLVED: alpha.csv now carries alpha_fast / alpha_slow per cell and
    # verify.py checks two of them.  The pattern only fires in the design-era
    # documents, where the value is a record rather than an unverifiable
    # claim, so keeping it is noise.  Left as a comment, not deleted:
    #   (r'0\.19\b|0\.16\s*[-–~]\s*0\.24',
    #    'transfer ratio alpha — not in a table (32.3)'),
    # RESOLVED: correlation.csv carries all four, checked by corr.disc.10s.
    #   (r'-0\.385|-0\.411|-0\.400|-0\.587|−0\.385|−0\.411|−0\.400|−0\.587',
    #    "28.3's correlation — not in a table"),
    # RESOLVED 2026-08-31: build_size.csv now carries both builds and
    # verify.py checks both (mcu.build.both, mcu.build.a8_only), so 142,060 is
    # table-backed.  143,932 and -1,872 B remain in the text only as the
    # superseded pre-rebuild values, explicitly marked as such in 35.4, and
    # flagging those is noise.  Kept as a comment so the check is not
    # silently dropped:
    #   (r'1,?872\s*B|142,?060|143,?932',
    #    'deployment build size — not in a table (33.6)'),
    # RESOLVED: cold_ratio.csv carries both cells, checked by
    # cold.BOOST.ratio and cold.CC.ratio.
    #   (r'0\.98[-–]?1\.00×|0\.98\s*[-–~]\s*1\.00\s*×|0\.98×|1\.00×',
    #    "33.5's resistance ratio — not in a table"),
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


# --- scope rules: a claim that needs its qualifier in the same breath -------
#
# RETRACTED is a blacklist, and a blacklist only catches the wording someone
# already used.  "Pack validation" is banned, so a rewrite says "confirmed on
# a 192-cell string" and sails through.  These rules invert it: certain
# CLAIMS may appear only if the block carrying them also carries the fact
# that bounds them.  The unit is the block, same as is_assertion, so the
# qualifier has to be adjacent prose rather than a footnote three screens
# down where a reader quoting the sentence would lose it.
#
# (trigger, required, what, why).  Both regexes run over the whole block.
SCOPED = [
    # "pack" as the SYSTEM UNDER TEST, not as a unit.  Proximity matching was
    # the first attempt and it flagged "10-14 %p of pack peak current", which
    # is a cell-level result quoted in pack terms and claims nothing about a
    # pack.  The trigger is therefore a list of phrasings that put a pack on
    # the stand: an exceedance rate FOR a pack, a result AT pack level, a
    # string of N cells.  Paraphrase is what this is for -- the blacklist
    # already owns the literal phrase "pack validation".
    (r"pack(?:'s)? (?:exceedance|min\b|safe|behaviou?r)"
     r"|(?:exceedance|usability|lambda|margin) (?:rate )?(?:on|for|at) (?:a |the )?pack"
     r"|at pack[- ]level|on a pack\b|pack[- ]level (?:result|number|safety|"
     r"validat|exceedance)|N *= *192|192[- ]cell",
     r'simulat|resampl|no pack hardware|Monte Carlo|sensitivity|'
     r'not a pack|never .{0,16}pack|no pack\b|future work|revalidat',
     'a pack-level result stated without saying it is the resampling '
     'simulation',
     'there is no pack, module or HIL bench in this work; sop_pack2.py '
     'resamples single-cell rows, so a pack sentence must carry that in the '
     'same block'),

    # Likewise, "1.06x across cells" is a measured spread, not a claim about
    # cells in general.  Only generalisation verbs and part-number statements
    # trigger.
    (r'generali[sz]\w*.{0,60}\bcells?\b|\bcells?\b.{0,40}generali[sz]'
     r'|(?:holds|works|transfers?|applies) (?:for|to|across) (?:all |any |'
     r'other )?cells|INR21700-30T (?:cells )?(?:in general|as a part)'
     r'|cell[- ]to[- ]cell (?:variation|spread) (?:is|was) (?:covered|'
     r'characteri[sz]ed)',
     r'six cells|leave-one-cell-out|one model|not a manufacturing|'
     r'PARTIAL|one order|one external cell',
     'a cell-generalisation claim without the population it rests on',
     'six cells from one order of one model is not a manufacturing '
     'population; generalization_scope.yaml holds axes.cell at PARTIAL'),

    # An exact-binomial bound over rows of one cell reads as a risk figure
    # unless the sentence says otherwise.  Trigger on the bound being quoted
    # as a percentage next to an exceedance/upper-bound word; require the
    # block to say what the denominator is.
    (r'(?:upper bound|ub95|one-sided|Clopper)\D{0,40}\d+(?:\.\d+)? *%'
     r'|\d+(?:\.\d+)? *% *(?:upper bound|one-sided)'
     # The phrasing a paper actually reaches for.  The corpus caught that the
     # first two alternatives miss "bounded at 9.2 %", which is the sentence
     # most likely to be written and the one that reads hardest as a risk.
     r'|bounded at \d+(?:\.\d+)? *%'
     r'|\d+(?:\.\d+)? *% *(?:with )?(?:at )?\d+ *% confidence',
     r'row|conditional|one (?:physical )?cell|single external cell|'
     r'this grid|not .{0,30}(?:risk|population)|in-hull|sample size|'
     r'per-cell|points\b|\d+ */ *\d+',
     'a binomial upper bound quoted without its denominator',
     'the bound is over rows of one physical cell, conditional on the tested '
     'grid; cell- or population-level risk is not estimable from it'),

    (r'\bWCET\b|worst[- ]case execution',
     r'sum of|per-stage|not a measured|derived|budget|stage maxima',
     'WCET stated as if measured',
     'no integrated loop was ever timed end to end on the board; the figure '
     'adds per-stage maxima measured separately'),
]


def section_preamble(lines, line_no, max_lines=18):
    """The heading of this line's section plus what opens it.

    A long section states its standing caveat once, at the top, and then
    argues for forty paragraphs.  Block scope alone would demand the word
    "simulation" in every one of those paragraphs, which is how a checker
    teaches people to ignore it.  A caveat in the section's opening lines
    counts for the whole section -- and only there, because a qualifier
    buried in the middle is one a reader quoting the section would miss.
    """
    head = 0
    for i, ln in enumerate(lines, 1):
        if i > line_no:
            break
        if re.match(r'#+ ', ln):
            head = i
    if not head:
        return ''
    return '\n'.join(lines[head - 1:head - 1 + max_lines])


def scope_hits(lines, blks, line_no, line):
    """Claims whose bounding fact is missing from their own block."""
    out = []
    for trig, need, what, why in SCOPED:
        m = re.search(trig, line, re.I)
        if not m:
            continue
        btxt, boff = block_for(line_no, blks)
        if re.search(need, btxt, re.I):
            continue
        if re.search(need, section_preamble(lines, line_no), re.I):
            continue
        if not is_assertion(line, m, btxt, boff):
            continue
        out.append((line_no, what, why, line.strip()[:72]))
    return out


def scan():
    stale_hits, retr_hits, orph_hits, scope_h = [], [], [], []
    stale = stale_pairs()
    for fn, lines in docs():
        blks = blocks(lines)
        for i, ln in enumerate(lines, 1):
            for old, new, what, ok in stale:
                # Token match, not substring: '217' otherwise fires inside
                # the cell part number INR21700-30T, which is how three of
                # these sat in README.md and DATA.md looking like drift.
                if not re.search(rf'(?<![\d.]){re.escape(old)}(?![\d])', ln):
                    continue
                blk = block_for(i, blks)[0].lower()
                recorded = (any(c in blk for c in CORRECTION_MARKERS)
                            or '~~' in ln
                            or all(x.strip().startswith('>')
                                   for x in blk.split('\n') if x.strip()))
                stale_hits.append((fn, i, old, new, what,
                                   ln.strip()[:72], recorded,
                                   section_of(fn, i, lines)))
            for i2, what, where, txt in retraction_hits(lines, blks, i, ln):
                retr_hits.append((fn, i2, what, where, txt))
            for i2, what, why, txt in scope_hits(lines, blks, i, ln):
                scope_h.append((fn, i2, what, why, txt))
            for pat, what in ORPHAN_HINTS:
                if re.search(pat, ln):
                    orph_hits.append((fn, i, what, ln.strip()[:60]))
    return stale_hits, retr_hits, orph_hits, scope_h


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


def section_of(fn, line_no, lines):
    """The numbered section a line falls in, or 0.

    Reported because it decides what a hit means.  Sections 34 and up are the
    audit's own record of current results; anything below is a dated account
    of an earlier configuration, and an old value there is the document doing
    its job.  Without the split, 71 historical records drowned the three that
    were real -- and those three turned out to be '217' matching inside the
    cell part number INR21700-30T.
    """
    import re as _re
    cur = 0
    for i, ln in enumerate(lines, 1):
        if i > line_no:
            break
        m = _re.match(r'#+ (\d+)\.', ln)
        if m:
            cur = int(m.group(1))
    return cur


def _n_checks():
    """How many values verify.py actually checks, read from expected.json.

    This was the literal 38 in two places and stayed there while the file grew
    to 80 -- the same defect as the STALE list carrying its own current
    values.  A count in prose is a number someone has to remember to update.
    """
    import json
    p = os.path.join(ROOT, 'repro', 'expected.json')
    try:
        return len(json.load(open(p, encoding='utf-8'))['checks'])
    except Exception:                                      # noqa: BLE001
        return '?'


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
    # Sections 1 and 4 are advisory by design: a stale-looking value may be a
    # deliberate comparison group, and an orphan number may be prose.  Those
    # need a person.  Sections 2 and 3 do not -- a retracted claim standing as
    # an assertion, or a claim without the fact that bounds it, is a defect
    # under a rule with a regression corpus behind it.  --fail-on-current
    # gates on exactly those two, plus stale values in CURRENT sections, so
    # repro/gate.py can refuse a push without turning the advisory lists into
    # noise that has to be silenced.
    gate = '--fail-on-current' in sys.argv
    st, rt, orp, scp = scan()

    CURRENT_FROM = 34
    live = [h for h in st if not h[6] and (h[7] == 0 or h[7] >= CURRENT_FROM)]
    historical = [h for h in st if not h[6] and 0 < h[7] < CURRENT_FROM]
    recorded = [h for h in st if h[6]]
    print(f"  == (1) possibly stale values  {len(live)} in current text, "
          f"{len(historical)} in sections below {CURRENT_FROM}, "
          f"{len(recorded)} inside a correction block\n", flush=True)
    print(f"  {'file':<22}{'line':>6}  {'stale':>9} -> {'current':<9} what",
          flush=True)
    print('  ' + '-' * 88, flush=True)
    for fn, i, old, new, what, txt, _rec, sec in live:
        print(f"  {fn:<22}{i:>6}  {old:>9} -> {new:<9} {what}", flush=True)
    if historical:
        print(f"\n  {len(historical)} more are in sections 1-{CURRENT_FROM - 1}, "
              f"which record what earlier configurations measured.  An old "
              f"value there is the document working, not drift.", flush=True)
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

    print(f"\n  == (3) claims stated without the fact that bounds them  "
          f"{len(scp)} places\n", flush=True)
    for fn, i, what, why, txt in scp:
        print(f"  {fn}:{i}", flush=True)
        print(f"    claim: {what}", flush=True)
        print(f"    why:   {why}", flush=True)
        print(f"    text:  {txt}", flush=True)
    if not scp:
        print('    none — every such claim carries its qualifier in the same '
              'block', flush=True)

    print(f"\n  == (4) numbers verify.py cannot see (not in any table)  "
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
          f"verify.py checks {_n_checks()} of them.", flush=True)
    print("\n  This list is 'where to look', not 'what to fix' — values left "
          "in deliberately as comparison groups also show up.", flush=True)

    if gate:
        live_now = [h for h in st if not h[6]
                    and (h[7] == 0 or h[7] >= CURRENT_FROM)]
        hard = len(live_now) + len(rt) + len(scp)
        if hard:
            print(f'\n  --fail-on-current: {len(live_now)} stale values in '
                  f'current text, {len(rt)} standing retracted claims, '
                  f'{len(scp)} unbounded claims.', flush=True)
            return 1
        print('\n  --fail-on-current: sections 1 (current text), 2 and 3 are '
              'clean.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
