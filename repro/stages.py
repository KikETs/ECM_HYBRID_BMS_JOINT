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

    # The two fits that actually ship.  These had no stage at all, which
    # meant the header on the board was produced by a command that existed
    # only in a shell history -- the same class of gap as a published number
    # with no producer.
    dict(id='trim_deploy_dis', tier=4, minutes=9, measured=True,
         cmd='{py} sop_trim.py --rung A8 --data cache/trim '
             '--out runs_trim_a8_deploy --deployment --save-pred',
         inputs=['cache/trim/trim_BOOST.npz',
                 'cache/trim/trim_BOOST_NEGPULSE.npz',
                 'cache/trim/trim_BOOST_NEGPULSE_1S.npz',
                 'cache/trim/trim_BOOST_REST.npz',
                 'cache/trim/trim_CC.npz',
                 'cache/trim/trim_CC_CELL2.npz'],
         outputs=['runs_trim_a8_deploy/model_A8_ALL.pt'],
         why='The all-cell discharge fit that goes on the board.  It is the '
             'artifact and it is never evaluated: there is no seventh cell to '
             'hold out, so the leave-one-cell-out folds are the honest '
             'estimate of what it does on a cell it has not seen.'),

    dict(id='trim_deploy_chg', tier=4, minutes=8, measured=True,
         cmd='{py} sop_trim.py --rung A8 --data cache/trim_chg '
             '--out runs_trim_a8_chg_deploy --deployment --save-pred',
         inputs=['cache/trim_chg/trim_BOOST.npz',
                 'cache/trim_chg/trim_BOOST_NEGPULSE.npz',
                 'cache/trim_chg/trim_BOOST_NEGPULSE_1S.npz',
                 'cache/trim_chg/trim_BOOST_REST.npz',
                 'cache/trim_chg/trim_CC.npz',
                 'cache/trim_chg/trim_CC_CELL2.npz'],
         outputs=['runs_trim_a8_chg_deploy/model_A8_ALL.pt'],
         why='The charge twin of trim_deploy_dis.'),

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

    dict(id='soh_select', tier=4, minutes=3, measured=True,
         cmd='{py} ../repro/run_soh_nested.py',
         inputs=['cache/soh_charge.npz'],
         outputs=['results/tables/soh_nested.csv',
                  'results/tables/soh_nested_summary.csv'],
         why='Which SOH model family an honest procedure picks.  Every '
             'candidate is scored by leave-one-cell-out over the five '
             'TRAINING cells before the held-out cell is touched.  The CNN '
             'is scored too, and comes last on every fold — it would never '
             'have been selected by anything that had not already seen the '
             'test cells (36.1).'),

    dict(id='soh', tier=4, minutes=1, measured=True,
         cmd='{py} soh_ridge.py --save-model runs_soh_ridge '
             '--save-pred results/soh_pred.npz --deployment',
         inputs=['cache/soh_charge.npz'],
         outputs=['runs_soh_ridge/soh_CC.npz', 'runs_soh_ridge/soh_ALL.npz',
                  'results/soh_pred.npz'],
         why='The SOH arm.  65 coefficients, cell-holdout RMSE 0.0094, worst '
             'cell 0.0130 (after excluding the 8 temperature-defect curves '
             '— 30.12).  Replaced the 1D CNN in the second audit round: '
             'better on every cell, 2,991x faster on the board, and half the '
             'firmware.  Every estimated-SOH version uses these predictions.  '
             'The alpha is never chosen on the held-out cell.'),

    dict(id='soh_cnn_reference', tier=4, minutes=6, measured=True,
         optional=True,
         cmd='{py} soh_cnn.py --save-model runs_soh_cnn '
             '--save-pred results/soh_pred_cnn.npz',
         inputs=['cache/soh_charge.npz'],
         outputs=['runs_soh_cnn/soh_CC.pt', 'results/soh_pred_cnn.npz'],
         why='The superseded CNN, kept runnable so the comparison in 36 can '
             'be reproduced rather than taken on trust.  It writes to a '
             'separate prediction file and feeds nothing.'),

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
             '{py} ../repro/run_safety_strict.py --arm est && '
             '{py} ../repro/run_safety_strict.py --arm oracle --method a3 && '
             '{py} ../repro/run_safety_strict.py --arm oracle --method lstm && '
             '{py} ../repro/run_safety_strict.py --arm oracle --method gru && '
             '{py} ../repro/run_safety_strict.py --arm oracle --method ffrls && '
             '{py} ../repro/run_safety_strict.py --arm oracle --method shrink',
         inputs=['results/eval/'],
         outputs=[
'results/tables/safety_strict_a3_oracle.csv',
                  'results/tables/safety_strict_est.csv',
                  'results/tables/safety_strict_ffrls_oracle.csv',
                  'results/tables/safety_strict_gru_oracle.csv',
                  'results/tables/safety_strict_lstm_oracle.csv',
                  'results/tables/safety_strict_oracle.csv',
                  'results/tables/safety_strict_percell_a3_oracle.csv',
                  'results/tables/safety_strict_percell_est.csv',
                  'results/tables/safety_strict_percell_ffrls_oracle.csv',
                  'results/tables/safety_strict_percell_gru_oracle.csv',
                  'results/tables/safety_strict_percell_lstm_oracle.csv',
                  'results/tables/safety_strict_percell_oracle.csv',
                  'results/tables/safety_strict_percell_shrink_oracle.csv',
                  'results/tables/safety_strict_shrink_oracle.csv',
                  'results/tables/safety_strict_tolsens_a3_oracle.csv',
                  'results/tables/safety_strict_tolsens_est.csv',
                  'results/tables/safety_strict_tolsens_ffrls_oracle.csv',
                  'results/tables/safety_strict_tolsens_gru_oracle.csv',
                  'results/tables/safety_strict_tolsens_lstm_oracle.csv',
                  'results/tables/safety_strict_tolsens_oracle.csv',
                  'results/tables/safety_strict_tolsens_shrink_oracle.csv'
         ],
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
         outputs=['results/tables/soc_perturb.csv',
                  'results/tables/soc_perturb_runs.csv',
                  'results/soc_perturb.npz'],
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
         cmd='{py} export_mcu_tables.py --rung A8 --deployment '
             '--trim runs_trim_a8_deploy --trim-chg runs_trim_a8_chg_deploy '
             '--out ../mcu/sop_tables.h && '
             '{py} export_soh_ridge.py --out ../mcu/soh_tables.h',
         inputs=['runs_trim_a8_deploy', 'runs_trim_a8_chg_deploy',
                 'runs_soh_ridge'],
         outputs=['../mcu/sop_tables.h', '../mcu/soh_tables.h'],
         why='The 32x16 grid and the trim weights as C headers.  A8, so '
             'there are 2 EW states.  --deployment because a header that '
             'ships must hold the all-cell fit: exporting a leave-one-cell-out '
             'fold would put a model trained without one of the six cells on '
             'the board and still call it the product.'),

    # ---- tier 5b  analyses that were reachable only by hand ------------
    # These produce 21 of the 68 verified numbers and had no stage, so a clean
    # clone could not rebuild them and `run.py --list` said the pipeline was
    # complete when it was not.  Adding them is what makes "raw to every
    # published number" a claim the graph can actually support.

    dict(id='soh_baselines', tier=5, minutes=2, measured=True,
         cmd='{py} ../repro/run_soh_baselines.py',
         inputs=['cache/soh_charge.npz'],
         outputs=['results/tables/soh_baselines.csv'],
         why='Five SOH comparison groups on the adopted splits, each with its '
             'hyperparameter chosen on the training cells only.'),

    dict(id='soh_ablations', tier=5, minutes=3, measured=True,
         cmd='{py} ../repro/run_soh_ablations.py',
         inputs=['cache/soh_charge.npz'],
         outputs=['results/tables/soh_ablations.csv'],
         why='What the SOH input representation is actually worth: dQ/dV '
             'against time-per-bin, and the voltage window swept down.'),

    dict(id='soc_baselines', tier=5, minutes=4, measured=True,
         cmd='{py} ../repro/run_soc_baselines.py',
         inputs=['results/soc_runs.pkl'],
         outputs=['results/tables/soc_baselines.csv'],
         why='Coulomb counting, 1RC-EKF, adaptive-R, dual EKF and a UKF on the '
             'same six disturbances.  Leave-one-cell-out since 37.24.'),

    dict(id='soc_headline', tier=5, minutes=3, measured=True,
         cmd='{py} ../repro/run_soc_headline.py',
         inputs=['results/soc_runs.pkl'],
         outputs=['results/tables/soc_headline.csv'],
         why='The SOC headline recomputed from the runs, after the original '
             'averaged seven rows while describing six.'),

    dict(id='integration_cost', tier=5, minutes=1, measured=True,
         cmd='{py} ../repro/run_integration_cost.py',
         inputs=['../mcu/sop_mcu_bench.csv'],
         outputs=['results/tables/integration_cost.csv'],
         why='mcu_cycle.csv adds per-stage maxima and 37.10 refuses to call '
             'that a WCET.  SOP_CMD_FULL times the trim and the solve in one '
             'window, so the summation error can be measured on the largest '
             'pair the firmware integrates.'),

    dict(id='usable_reference', tier=5, minutes=1, measured=True,
         cmd='{py} ../repro/run_usable_reference.py',
         inputs=['results/eval/'],
         outputs=['results/tables/usable_reference.csv'],
         why='"Usable current" was a percentage of the cell own capability, '
             'which no single-lambda policy can reach: lambda is set by the '
             'worst row of the most demanding cell.  This adds the two '
             'reference points that were missing - the all-cells lambda '
             'ceiling, and a per-cell oracle lambda.'),

    dict(id='soc_soh_selection', tier=5, minutes=50, measured=True,
         cmd='{py} ../repro/run_soc_soh_selection.py',
         inputs=['results/soc_runs.pkl', 'results/soh_pred.npz'],
         outputs=['results/tables/soc_soh_selection.csv'],
         why='The SOC filter was selected on a benchmark that hands every '
             'configuration its cell TRUE SOH.  This re-runs the whole '
             'comparison under the ridge estimate and under a deliberate '
             '+-0.02 bias, so the choice can be checked against the '
             'condition it deploys in.'),

    dict(id='soc_loco', tier=5, minutes=12, measured=True,
         cmd='{py} ../repro/run_soc_loco.py',
         inputs=['results/soc_runs.pkl', 'uypydj_ecm.csv', 'uypydj_ocv.csv'],
         outputs=['results/tables/soc_loco.csv'],
         why='The SOC arm reads the evaluated cell own surface, which makes '
             'it a per-cell calibrated deployment until 37.24.  This runs '
             'the identical '
             'benchmark on the leave-one-cell-out pooled surface the SOP and '
             'SOH arms use, so the cost of not seeing the cell is measured '
             'rather than left as a caveat.'),

    dict(id='soc_percell', tier=5, minutes=1, measured=True,
         cmd='{py} ../repro/run_soc_percell.py',
         inputs=['results/tables/soc_perturb_runs.csv',
                 'results/tables/soc_headline.csv'],
         outputs=['results/tables/soc_percell.csv'],
         why='The SOC arm published one pooled mean and no spread, while the '
             'SOP and SOH arms both report a per-cell breakdown, a worst '
             'cell and an interval.  Reads the per-run errors from the '
             'committed per-run table, so it cannot disagree with the '
             'headline and a clean clone can recompute it.'),

    dict(id='label_quality', tier=5, minutes=2, measured=True,
         cmd='{py} ../repro/run_label_quality.py',
         inputs=['sop_label_measured.csv', 'sop_label_charge.csv'],
         outputs=['results/tables/label_quality.csv',
                  'results/tables/label_sensitivity.csv'],
         why='How many SOP labels are measured rather than extrapolated, and '
             'how the headline moves as the trust threshold tightens.'),

    dict(id='seq_baselines', tier=5, minutes=45, measured=True,
         cmd='{py} ../repro/run_sop_seq_baselines.py',
         inputs=['cache/trim', 'cache/trim_chg'],
         outputs=['runs_trim_lstm', 'runs_trim_gru', 'runs_trim_ffrls'],
         why='LSTM, GRU and forgetting-factor RLS, trained only.  It does NOT '
             'write the safety tables: eval and safety_strict do that, and '
             'they have to run AFTER this stage.  Declaring the tables here '
             'was wrong and it hid an ordering trap - run_evals scored the '
             'previous sequence models when this ran second.'),

    dict(id='chen2026', tier=5, minutes=7, measured=True,
         cmd='{py} ../repro/run_chen2026_baseline.py && '
             '{py} ../repro/run_chen2026_baseline.py --all-surfaces',
         inputs=['cache/pool', 'sop_label_measured.csv'],
         outputs=['results/tables/chen2026_baseline.csv',
                  'results/tables/external_temp_envelope.csv',
                  'results/tables/external_temp_surfaces.csv'],
         why="Chen 2026's constant-power binary search reimplemented on this "
             'data, to its own published tolerances.  The same rows also give '
             'the external temperature envelope: Test#3 is the only external '
             'sheet with a temperature axis, and it decides how far the '
             'frozen safety factor reaches.'),

    dict(id='external_crate', tier=5, minutes=3, measured=True,
         cmd='{py} ../repro/run_external_crate.py && '
             '{py} ../repro/run_external_crate.py --all-surfaces',
         inputs=['rpcwby_sop_test8.csv', 'cache/pool'],
         outputs=['results/tables/external_crate_envelope.csv',
                  'results/tables/external_crate_surfaces.csv'],
         why='RPCWBY Test#8: the same cell at 0 C after discharging at six '
             'different rates.  0 C is where the frozen lambda has least room '
             '(margin 1.396), so if prior load moves the requirement it moves '
             'it there.  Physics layer only - Test#8 has no paired drive '
             'cycle either, so the trim cannot be scored on it (37.12).'),

    dict(id='nested_selection', tier=5, minutes=120, measured=True,
         cmd='{py} ../repro/run_nested_selection.py',
         inputs=['cache/trim', 'cache/trim_chg'],
         outputs=['results/tables/nested_selection.csv'],
         why='Outer leave-one-cell-out x inner leave-one-out on the five '
             'training cells, so the rung and the aggregation are chosen '
             'without seeing the cell they are scored on.'),

    dict(id='end_to_end', tier=5, minutes=12, measured=True,
         cmd='{py} ../repro/run_end_to_end.py',
         inputs=['results/soc_runs.pkl', 'results/soh_pred.npz',
                 'runs_trim_a8', 'runs_trim_a8_chg'],
         outputs=['results/tables/end_to_end.csv',
                  'results/tables/end_to_end_paired.csv',
                  'results/tables/end_to_end_fixed_lambda.csv',
                  'results/tables/end_to_end_drift.csv'],
         why='The four oracle/estimated corners: unpaired, paired on the rows '
             'all four keep, paired under a lambda frozen at the oracle '
             "corner's calibration, and the rows the intersection drops."),

    dict(id='external', tier=5, minutes=105, measured=True,
         cmd='{py} ../repro/run_external_a8.py && '
             '{py} ../repro/run_external_a8.py --all-surfaces',
         inputs=['rpcwby_ecm.csv', 'runs_trim_a8'],
         outputs=['results/tables/external_a8.csv',
                  'results/tables/external_a8_coverage.csv',
                  'results/tables/external_a8_safety.csv',
                  'results/tables/external_a8_surfaces.csv'],
         why='The six frozen A8 folds carried to RPCWBY Test#2 without '
             'refitting: error, in-hull coverage by operating point, and '
             'whether the frozen lambda stays conservative.'),

    dict(id='method_comparison', tier=5, minutes=1, measured=True,
         cmd='{py} ../repro/run_method_comparison.py && '
             '{py} ../repro/run_method_comparison.py --arm est',
         inputs=['results/tables/safety_strict_oracle.csv',
                 'results/tables/safety_strict_a3_oracle.csv',
                 'results/tables/safety_strict_lstm_oracle.csv',
                 'results/tables/safety_strict_gru_oracle.csv',
                 'results/tables/safety_strict_ffrls_oracle.csv',
                 'results/tables/safety_strict_shrink_oracle.csv'],
         outputs=['results/tables/method_comparison.csv',
                  'results/tables/method_comparison_est.csv'],
         why='Every method side by side with its bootstrap interval and its '
             'rank.  A8 places 3rd, 3rd, 2nd and 5th across the four '
             'conditions and only FFRLS separates from it, which is what '
             '"competitive" means and why "outperformed" and "equivalent" '
             'are both unavailable.'),

    dict(id='soh_deploy_tables', tier=6, minutes=1, measured=True,
         cmd='{py} ../repro/run_soh_deploy_tables.py',
         inputs=['results/soh_pred.npz', '../mcu/soh_mcu_bench.csv',
                 '../mcu/soh_mcu_bench_cnn.csv', '../mcu/sop_mcu_bench.csv',
                 '../mcu/sop_mcu_bench_cnn.csv',
                 '../mcu/sop_mcu_bench_noicache_ridge.csv',
                 '../mcu/sop_mcu_bench_noicache_cnn.csv'],
         outputs=['results/tables/soh_model_cost.csv',
                  'results/tables/mcu_icache.csv'],
         why='What replacing the CNN cost and bought, and the controlled '
             'instruction-cache experiment that explains why the SOP path got '
             'slower when it should not have.'),

    dict(id='mcu_table', tier=6, minutes=1, measured=True,
         cmd='{py} ../repro/run_mcu_table.py',
         inputs=['../mcu/sop_mcu_bench.csv'],
         outputs=['results/tables/mcu.csv',
                  'results/tables/mcu_cycle.csv'],
         why='Reduce the board benchmark to the published timing table.  '
             'bench_sop.py wrote only its own per-sample CSV; nothing wrote '
             'results/tables/mcu.csv, so four published board numbers had '
             'no producer.'),

    dict(id='mcu_integrated', tier=6, minutes=8, measured=True, board=True,
         cmd='cd ../mcu/fw_sop && make MODEL_ID=pack_bench '
             'EXTRA_CFLAGS=-DSOP_BENCH_PACK && STM32_Programmer_CLI -c '
             'port=SWD -w Build/pack_bench/sop_bench.elf -v -rst && cd ../.. '
             '&& {py} repro/run_mcu_integrated.py --port /dev/ttyACM0',
         inputs=['../mcu/sop_mcu_bench.csv'],
         outputs=['results/tables/mcu_integrated.csv'],
         why='The four stages of a control cycle were only ever timed apart '
             'and added.  SOP_CMD_CYCLE runs them in one DWT window on the '
             'same operating points, and SOP_CMD_PACK repeats the cycle for '
             'N cells with a state each.  Needs the NUCLEO-H563ZI.'),

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
