"""The stage graph of the reproduction pipeline.

What is written here is the path that produces every number in the paper.
analysis/ holds 82 scripts, but most of them are exploratory; the critical
path is the stages below.  The rest are listed in `EXPLORATORY` with the
reason — deleting them would not make reproduction work, but without an
answer to "which of these 82 do I run?" reproduction is impossible.

Each stage carries:
    id        short name.  What run.py takes as a target.
    tier      0 raw -> 6 board.  Lower tiers run first.
    cmd       the command that actually runs.  Executed from analysis/.
    inputs    paths that must exist (relative to analysis/)
    outputs   paths that get created
    minutes   how long it took (measured=True if timed, not estimated)
    why       what this stage settles

{py} inside a command is substituted with the interpreter.
"""

RAW_DOI = {
    'UYPYDJ': ('10.5683/SP3/UYPYDJ',
               'Kollmeyer et al., Samsung INR21700-30T ageing cycling'),
    'RPCWBY': ('10.5683/SP3/RPCWBY',
               'Chen et al., 30T SOP measured directly (temperature axis)'),
    'Mendeley': ('10.17632/9xyvy2njj3.2',
                 'Kollmeyer & Skells, Samsung INR21700 30T 3Ah battery data '
                 '(temperature axis + drive cycles).  The earlier '
                 '10.17632/cp3473x7xv.3 in this slot is the LG 18650HG2 '
                 'dataset and was wrong.'),
}

CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST',
         'CC', 'CC_CELL2']

STAGES = [
    # ---- tier 1  raw -> cache -------------------------------------------
    dict(id='cache', tier=1, minutes=210, measured=False,
         # TWO invocations.  --part defaults to Fifteen_Drive_Cycles, so a
         # single call writes six of the twelve declared outputs and the six
         # *_HPPC.npz are never built.  The stage still exited 0, and run.py
         # reported it done - found by the 2026-08-27 raw-to-result rebuild.
         cmd='{py} build_uypydj_cache.py --raw ../raw/UYPYDJ --cache cache_t '
             '--part Fifteen_Drive_Cycles && '
             '{py} build_uypydj_cache.py --raw ../raw/UYPYDJ --cache cache_t '
             '--part HPPC',
         force_flag='--force',
         inputs=['../raw/UYPYDJ'],
         outputs=[f'cache_t/uypydj_{c}_{p}.npz'
                  for c in CELLS for p in ('Fifteen_Drive_Cycles', 'HPPC')],
         why='Collect the 5,020 raw .mat files into cell x protocol arrays. '
             'Every later stage reads only this.'),

    dict(id='temp_audit', tier=1, minutes=35, measured=True,
         cmd='{py} temp_audit_all.py --out temp_audit_all.csv',
         inputs=['../raw/UYPYDJ'],
         outputs=['temp_audit_all.csv'],
         why='Exhaustive audit of the temperature channel.  It found 6 HPPC '
             '/ 2 OCV / 10 drive defects, and every later dataset excludes '
             'them through temp_defects.py.  Skip this and 18 defects flow '
             'downstream.'),

    # ---- tier 2  characterisation tables --------------------------------
    dict(id='ocv', tier=2, minutes=12, measured=False,
         cmd='{py} uypydj_ocv.py --raw ../raw/UYPYDJ --out uypydj_ocv.csv',
         inputs=['../raw/UYPYDJ', 'temp_audit_all.csv'],
         outputs=['uypydj_ocv.csv'],
         why='OCV over the SOC axis.  The EKF measurement model and the '
             'reference point of the SOP inversion.'),

    dict(id='hppc_r', tier=2, minutes=25, measured=False,
         cmd='{py} uypydj_hppc_resistance.py --raw ../raw/UYPYDJ '
             '--out uypydj_hppc_resistance.csv',
         inputs=['../raw/UYPYDJ', 'temp_audit_all.csv'],
         outputs=['uypydj_hppc_resistance.csv'],
         why='Equivalent resistance for every HPPC pulse.  The source of '
             'the SOP labels.'),

    dict(id='ecm', tier=2, minutes=40, measured=False,
         cmd='{py} uypydj_ecm.py --raw ../raw/UYPYDJ --out uypydj_ecm.csv',
         inputs=['../raw/UYPYDJ', 'temp_audit_all.csv'],
         outputs=['uypydj_ecm.csv'],
         why='Fit the 2RC parameters (R0, R1, tau1, R2, tau2) per pulse.'),

    dict(id='mendeley_ecm', tier=2, minutes=6, measured=False,
         cmd='{py} mendeley_ecm.py',
         inputs=['../raw/Mendeley'],
         outputs=['mendeley_ecm.csv'],
         why='2RC parameters over the Mendeley temperature sweep (-20 to '
             '40 C).  This stage was missing from the graph, so the raw '
             'archive had no declared path to the CSV the temperature '
             'factor is built from.'),

    dict(id='rpcwby_ecm', tier=2, minutes=20, measured=False,
         cmd='{py} rpcwby_to_ecm.py',
         inputs=['../raw/RPCWBY'],
         outputs=['rpcwby_ecm.csv'],
         why='2RC parameters from the RPCWBY external dataset.  ORPHANED: '
             'nothing in the repository reads rpcwby_ecm.csv, and TWO '
             'scripts write it - rpcwby_to_ecm.py (2,865 rows) and '
             'rpcwby_resistance.py (3,451 rows, the committed file).  They '
             'filter differently and disagree.  The stage runs the former, '
             'so a rebuild replaces the committed file with a smaller one; '
             'no published number moves because none depends on it.  Pick '
             'one producer or delete the artifact.'),

    dict(id='temp_factor', tier=2, minutes=3, measured=False,
         cmd='{py} ecm_temp_factor.py --out ecm_temp_factor.csv',
         inputs=['mendeley_ecm.csv'],
         outputs=['ecm_temp_factor.csv'],
         why='Temperature correction factor.  UYPYDJ is 25 C only, so it '
             'comes from the Mendeley temperature sweep.  The graph used to '
             'declare rpcwby_ecm.csv here while ecm_temp_factor.py actually '
             'opens mendeley_ecm.csv — the wrong upstream.'),

    dict(id='pool', tier=2, minutes=8, measured=False,
         cmd='{py} ecm_pool.py --outdir cache/pool',
         inputs=['uypydj_ecm.csv', 'uypydj_ocv.csv'],
         outputs=[f'cache/pool/ecm_pool_{c}.csv' for c in CELLS],
         why='Pooled surfaces for per-cell holdout.  Each surface is built '
             'without its holdout cell, so the evaluation never saw it.'),

    # ---- tier 3  labels and training data -------------------------------
    dict(id='label_dis', tier=3, minutes=6, measured=False,
         cmd='{py} sop_label.py --direction discharge',
         inputs=['uypydj_hppc_resistance.csv'],
         outputs=['sop_label_measured.csv'],
         why='Discharge SOP labels.  Only extrap <= 1.5 counts as a '
             'trustworthy label.'),

    dict(id='label_chg', tier=3, minutes=6, measured=False,
         cmd='{py} sop_label.py --direction charge',
         inputs=['uypydj_hppc_resistance.csv'],
         outputs=['sop_label_charge.csv'],
         why='Charge SOP labels.  Seven times as many as discharge '
             '(section 25).'),

    dict(id='trim_data_dis', tier=3, minutes=130, measured=True,
         cmd='{py} sop_trim_dataset.py --direction discharge --out cache/trim',
         inputs=['cache_t', 'cache/pool', 'sop_label_measured.csv',
                 'uypydj_ecm.csv', 'uypydj_hppc_resistance.csv'],
         outputs=[f'cache/trim/trim_{c}.npz' for c in CELLS],
         why='Pair each HPPC pulse with the 12 preceding drive-history '
             'windows and extract 12 EW features from them.  The trim\'s '
             'input.'),

    dict(id='trim_data_chg', tier=3, minutes=110, measured=False,
         cmd='{py} sop_trim_dataset.py --direction charge --out cache/trim_chg',
         inputs=['cache_t', 'cache/pool', 'sop_label_charge.csv'],
         outputs=[f'cache/trim_chg/trim_{c}.npz' for c in CELLS],
         why='The same thing in the charge direction.'),

    dict(id='soh_data', tier=3, minutes=45, measured=False,
         cmd='{py} soh_charge_dataset.py --raw ../raw/UYPYDJ '
             '--out cache/soh_charge.npz',
         inputs=['../raw/UYPYDJ', 'temp_audit_all.csv'],
         outputs=['cache/soh_charge.npz'],
         why='Dataset for predicting SOH from a partial charge segment.  '
             'Given the full curve, integrating hands over the capacity, so '
             'only a segment is given.'),

    # ---- tier 4  training -----------------------------------------------
    dict(id='trim_dis', tier=4, minutes=9, measured=True,
         cmd='{py} sop_trim.py --rung A8 --data cache/trim '
             '--out runs_trim_a8 --save-pred',
         inputs=[f'cache/trim/trim_{c}.npz' for c in CELLS],
         outputs=[f'runs_trim_a8/pred_A8_{c}.npz' for c in CELLS],
         why='The adopted trim (discharge).  Two multipliers from dR_fast '
             'alone.  4 parameters.'),

    dict(id='trim_chg', tier=4, minutes=8, measured=True,
         cmd='{py} sop_trim.py --rung A8 --data cache/trim_chg '
             '--out runs_trim_a8_chg --save-pred',
         inputs=[f'cache/trim_chg/trim_{c}.npz' for c in CELLS],
         outputs=[f'runs_trim_a8_chg/pred_A8_{c}.npz' for c in CELLS],
         why='The adopted trim (charge).  It beat A3 in 32.7.'),

    dict(id='trim_a3_dis', tier=4, minutes=11, measured=True,
         cmd='{py} sop_trim.py --rung A3 --data cache/trim '
             '--out runs_trim_v2 --save-pred',
         inputs=[f'cache/trim/trim_{c}.npz' for c in CELLS],
         outputs=[f'runs_trim_v2/pred_A3_{c}.npz' for c in CELLS],
         why='Comparison group (12 features, 26 parameters).  Needed for '
             'the paper\'s ladder.'),

    dict(id='trim_a3_chg', tier=4, minutes=10, measured=False,
         cmd='{py} sop_trim.py --rung A3 --data cache/trim_chg '
             '--out runs_trim_chg_v2 --save-pred',
         inputs=[f'cache/trim_chg/trim_{c}.npz' for c in CELLS],
         outputs=[f'runs_trim_chg_v2/pred_A3_{c}.npz' for c in CELLS],
         why='Comparison group (charge).'),

    dict(id='soh', tier=4, minutes=6, measured=True,
         cmd='{py} soh_cnn.py --save-model runs_soh_cnn '
             '--save-pred results/soh_pred.npz',
         inputs=['cache/soh_charge.npz'],
         outputs=['runs_soh_cnn/soh_CC.pt', 'results/soh_pred.npz'],
         why='The SOH arm.  10,945 parameters, cell-holdout RMSE 0.0135, '
             'bias +0.0001 (after excluding the 8 temperature-defect curves '
             '— 30.12).  Every estimated-SOH version uses these '
             'predictions.'),

    # ---- tier 5  evaluation ---------------------------------------------
    dict(id='baselines', tier=5, minutes=4, measured=True,
         cmd='{py} sop_baseline_fill.py --data cache/trim --suffix "" && '
             '{py} sop_baseline_fill.py --data cache/trim_chg --suffix _chg',
         inputs=[f'cache/trim/trim_{c}.npz' for c in CELLS],
         outputs=['runs_trim_direct', 'runs_trim_shrink', 'runs_trim_rls',
                  'runs_trim_direct_chg', 'runs_trim_shrink_chg',
                  'runs_trim_rls_chg'],
         why='Export the three literature comparisons (direct plug-in / '
             'shrinkage coefficient / HPPC-RLS) in the trim directory '
             'format.  Section 32.'),

    dict(id='eval', tier=5, minutes=22, measured=True,
         cmd='{py} ../repro/run_evals.py',
         inputs=['runs_trim_a8', 'runs_trim_a8_chg', 'runs_trim_v2',
                 'runs_trim_chg_v2', 'runs_trim_direct', 'runs_trim_shrink',
                 'runs_trim_rls', 'results/soh_pred.npz'],
         outputs=['results/eval/'],
         why='Run every trim version x both directions x (oracle SOH, '
             'estimated SOH) through the SOP inversion.  The source of '
             'sections 16 / 25 / 29 / 32.'),

    dict(id='voltage', tier=5, minutes=1, measured=True,
         cmd='{py} ../repro/run_voltage.py',
         inputs=['runs_trim_a8', 'runs_trim_v2', 'runs_trim_direct',
                 'runs_trim_shrink', 'runs_trim_rls'],
         outputs=['results/tables/voltage.csv'],
         why='The voltage RMSE table.  These used to be twelve constants '
             'inside the figure file, so the figure did not follow the table '
             'and there was no way to check it.'),

    dict(id='safety', tier=5, minutes=3, measured=True,
         cmd='{py} ../repro/run_safety.py',
         inputs=['results/eval/'],
         outputs=['results/tables/safety.csv', 'results/tables/ladder.csv',
                  'results/tables/soh_cost.csv'],
         why='Set the safety factor lambda leaving one cell out, and report '
             'optimism / worst overshoot / usable current.  The values that '
             'decide deployment.'),

    dict(id='soh_table', tier=5, minutes=1, measured=True,
         cmd='{py} ../repro/run_soh_table.py',
         inputs=['results/soh_pred.npz'],
         outputs=['results/tables/soh.csv'],
         why='Per-cell SOH error table.  soh.csv carried two published '
             'numbers with no producer in the repository at all; this is '
             'that producer.  It also surfaces the worst cell, which the '
             'pooled RMSE hides.'),

    dict(id='safety_strict', tier=5, minutes=2, measured=True,
         cmd='{py} ../repro/run_safety_strict.py --arm oracle && '
             '{py} ../repro/run_safety_strict.py --arm est',
         inputs=['results/eval/'],
         outputs=['results/tables/safety_strict_oracle.csv',
                  'results/tables/safety_strict_percell_oracle.csv',
                  'results/tables/safety_strict_tolsens_oracle.csv',
                  'results/tables/safety_strict_est.csv',
                  'results/tables/safety_strict_percell_est.csv',
                  'results/tables/safety_strict_tolsens_est.csv'],
         why='Safety factor calibrated strictly per held-out cell.  The '
             'shipped safety.csv pools six LOCO lambdas into their median '
             'and applies it to every cell, so the evaluated cell helps set '
             'its own lambda.  This stage removes that and reports per-cell '
             'lambda, worst cell, a Clopper-Pearson upper bound and a '
             'cell-cluster bootstrap interval.'),

    dict(id='pack', tier=5, minutes=5, measured=True,
         cmd='{py} sop_pack2.py',
         inputs=['results/eval/'],
         outputs=['results/tables/pack.csv'],
         why='Pack level.  Does the margin survive the min over N cells.  '
             'Section 31.'),

    dict(id='soc_runs', tier=5, minutes=4, measured=True,
         cmd='{py} ../repro/build_soc_runs.py',
         # ECMSurface opens uypydj_ecm.csv, uypydj_ocv.csv and
         # ecm_temp_factor.csv directly; those were undeclared, so a changed
         # characterisation table did not mark the SOC runs stale.
         inputs=['cache_t', 'cache/pool', 'uypydj_ecm.csv', 'uypydj_ocv.csv',
                 'ecm_temp_factor.csv'],
         outputs=['results/soc_runs.pkl'],
         why='Build the 36 runs the SOC benchmark uses.  They used to live '
             'in /tmp, and a reproduction package must not depend on /tmp, so '
             'this became an explicit stage.  The picking rule is np.linspace '
             'and therefore deterministic; --check-against confirmed it '
             'matches the previous bundle.'),

    dict(id='soc', tier=5, minutes=55, measured=True,
         cmd='{py} soc_perturb_bench.py && {py} soc_est_soh.py',
         inputs=['results/soc_runs.pkl', 'results/soh_pred.npz'],
         outputs=['results/tables/soc_perturb.csv'],
         why='The SOC benchmark with the circularity broken.  It separates '
             'the current that made the label from the current the filter '
             'sees.  Section 30.'),

    dict(id='figures', tier=5, minutes=3, measured=True,
         cmd='{py} ../repro/fig_ladder.py && '
             '{py} ../repro/fig_soc_traj.py && '
             '{py} ../repro/fig_soh_traj.py && '
             '{py} ../repro/fig_sop_traj.py --direction discharge && '
             '{py} ../repro/fig_sop_traj.py --direction charge && '
             '{py} ../repro/fig_usable_ci.py',
         inputs=['results/tables/ladder.csv', 'results/tables/voltage.csv',
                 'results/tables/safety_strict_oracle.csv',
                 'results/tables/soc_perturb.csv',
                 'results/soh_pred.npz', 'results/soc_runs.pkl',
                 'results/eval/'],
         outputs=['../results_fig_ladder.png', '../results_fig_soc_traj.png',
                  '../results_fig_soh_traj.png',
                  '../results_fig_sop_traj_discharge.png',
                  '../results_fig_sop_traj_charge.png',
                  '../results_fig_usable_ci.png'],
         why='The paper\'s five figures.  All of them read straight from '
             'the tables in results/, so a changed table changes the figure '
             '— numbers typed into a figure by hand drift from the table.'),

    dict(id='extras', tier=5, minutes=2, measured=True,
         cmd='{py} ../repro/run_extras.py',
         inputs=['results/eval/', 'cache/trim', 'cache/trim_chg',
                 'results/cold_check/'],
         outputs=['results/tables/alpha.csv',
                  'results/tables/correlation.csv',
                  'results/tables/cold_ratio.csv',
                  'results/tables/build_size.csv'],
         why='Pull numbers that lived only in the documents into tables — '
             'the transfer ratio alpha, the weak-optimistic correlation, the '
             'defect-cycle resistance ratio, and the deployment build size.  '
             'What is not in a table is invisible to verify.py, and this '
             'session\'s errors came from exactly such places.'),

    # ---- tier 6  board --------------------------------------------------
    dict(id='mcu_export', tier=6, minutes=2, measured=True,
         cmd='{py} export_mcu_tables.py --rung A8 --out ../mcu/sop_tables.h && '
             '{py} export_soh_mcu.py --out ../mcu/soh_tables.h',
         inputs=['runs_trim_a8', 'runs_soh_cnn'],
         outputs=['../mcu/sop_tables.h', '../mcu/soh_tables.h'],
         why='The 32x16 grid and the trim weights as C headers.  A8, so '
             'there are 2 EW states.'),

    dict(id='mcu_table', tier=6, minutes=1, measured=True,
         cmd='{py} ../repro/run_mcu_table.py',
         inputs=['../mcu/sop_mcu_bench.csv'],
         outputs=['results/tables/mcu.csv'],
         why='Reduce the board benchmark to the published timing table.  '
             'bench_sop.py wrote only its own per-sample CSV; nothing wrote '
             'results/tables/mcu.csv, so four published board numbers had '
             'no producer.'),

    dict(id='mcu_measure', tier=6, minutes=25, measured=True, board=True,
         cmd='cd ../mcu/fw_sop && make && '
             'STM32_Programmer_CLI -c port=SWD -w Build/nmc_dst_cc/'
             'sop_bench.elf -v -rst && cd .. && {py} bench_sop.py --n 500 && '
             'cd ../analysis && {py} ../repro/run_extras.py',
         inputs=['../mcu/sop_tables.h', '../mcu/soh_tables.h'],
         outputs=['../mcu/sop_mcu_bench.csv'],
         why='Measured on a NUCLEO-H563ZI with the DWT cycle counter.  us '
             'per period, Flash, stack.  Without the board this is skipped '
             'and the table stays empty.'),
]

# Off the critical path — and why each is kept
EXPLORATORY = {
    'lstm_voltage.py / eval_voltage.py':
        'Reproduction of the reference paper (Chen et al.).  The grounds '
        'for section 11.4\'s "the reference LSTM cannot run the SOP bisection '
        'search".  Cited in the paper, but the numbers are fixed so it needs '
        'no re-run.',
    'rpcwby_*.py':
        'External cell validation (temperature axis).  Section 26.  Needs '
        'the RPCWBY raw data to run.',
    'runs_trim_q*, runs_trim_cur*, runs_trim_w':
        'Exploratory traces of the quantile-loss and current-weighted '
        'variants.  They do not appear in the text.',
    'soc_robust_sweep.py, soc_robust_loco.py, soc_ibias_*.py':
        'Section 30\'s failed path — the rule that grows R with the '
        'residual.  The conclusion was "discard", so it is not a reproduction '
        'target, but 30.2\'s account quotes these scripts\' output.',
    'dekf_soh.py, sop_from_ekf.py, ecm_refine.py':
        'Paths tried and abandoned.  Kept for the record.',
}
