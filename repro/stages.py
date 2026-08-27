"""재현 파이프라인의 단계 그래프.

여기 적힌 것이 논문의 모든 수치를 만드는 경로다.  analysis/ 에는 스크립트가
82 개 있지만 그 대부분은 탐색용이고, 임계 경로는 아래 단계들이다.  나머지는
`EXPLORATORY` 에 이유와 함께 남겨 둔다 — 지웠다고 재현이 되는 것은 아니지만,
"이 82 개 중 무엇을 돌려야 하는가" 에 답이 없으면 재현이 안 된다.

각 단계는:
    id        짧은 이름.  run.py 의 대상.
    tier      0 원본 -> 6 보드.  낮은 tier 가 먼저.
    cmd       실제로 도는 명령.  analysis/ 에서 실행한다.
    inputs    있어야 하는 경로 (analysis/ 기준 상대)
    outputs   만들어지는 경로
    minutes   측정한 소요 시간 (추정이 아니라 잰 값이면 measured=True)
    why       이 단계가 무엇을 정하는가

명령 안의 {py} 는 인터프리터로 치환된다.
"""

RAW_DOI = {
    'UYPYDJ': ('10.5683/SP3/UYPYDJ',
               'Kollmeyer et al., Samsung INR21700-30T 노화 사이클링'),
    'RPCWBY': ('10.5683/SP3/RPCWBY',
               'Chen et al., 30T SOP 직접 측정 (온도 축)'),
    'Mendeley': ('10.17632/cp3473x7xv.3',
                 'Kollmeyer et al., 30T 드라이브 사이클'),
}

CELLS = ['BOOST', 'BOOST_NEGPULSE', 'BOOST_NEGPULSE_1S', 'BOOST_REST',
         'CC', 'CC_CELL2']

STAGES = [
    # ---- tier 1  원본 -> 캐시 -------------------------------------------
    dict(id='cache', tier=1, minutes=210, measured=False,
         cmd='{py} build_uypydj_cache.py --raw ../raw/UYPYDJ --cache cache_t',
         inputs=['../raw/UYPYDJ'],
         outputs=[f'cache_t/uypydj_{c}_{p}.npz'
                  for c in CELLS for p in ('Fifteen_Drive_Cycles', 'HPPC')],
         why='원본 .mat 5,020 개를 셀 x 프로토콜 배열로 모은다. '
             '이후 모든 단계가 이것만 읽는다.'),

    dict(id='temp_audit', tier=1, minutes=35, measured=True,
         cmd='{py} temp_audit_all.py --out temp_audit_all.csv',
         inputs=['../raw/UYPYDJ'],
         outputs=['temp_audit_all.csv'],
         why='온도 채널 결함 전수조사.  HPPC 6 건 / OCV 2 건 / 주행 10 건을 '
             '찾아냈고, 이후 모든 데이터셋이 temp_defects.py 로 이것을 '
             '제외한다.  이 단계를 건너뛰면 결함 18 건이 하류로 흘러간다.'),

    # ---- tier 2  특성화 표 ----------------------------------------------
    dict(id='ocv', tier=2, minutes=12, measured=False,
         cmd='{py} uypydj_ocv.py --raw ../raw/UYPYDJ --out uypydj_ocv.csv',
         inputs=['../raw/UYPYDJ', 'temp_audit_all.csv'],
         outputs=['uypydj_ocv.csv'],
         why='SOC 축 위의 OCV.  EKF 의 측정 모델과 SOP 반전의 기준점.'),

    dict(id='hppc_r', tier=2, minutes=25, measured=False,
         cmd='{py} uypydj_hppc_resistance.py --raw ../raw/UYPYDJ '
             '--out uypydj_hppc_resistance.csv',
         inputs=['../raw/UYPYDJ', 'temp_audit_all.csv'],
         outputs=['uypydj_hppc_resistance.csv'],
         why='HPPC 펄스마다의 등가저항.  SOP 라벨의 원천.'),

    dict(id='ecm', tier=2, minutes=40, measured=False,
         cmd='{py} uypydj_ecm.py --raw ../raw/UYPYDJ --out uypydj_ecm.csv',
         inputs=['../raw/UYPYDJ', 'temp_audit_all.csv'],
         outputs=['uypydj_ecm.csv'],
         why='2RC 파라미터 (R0, R1, tau1, R2, tau2) 를 펄스마다 적합.'),

    dict(id='temp_factor', tier=2, minutes=3, measured=False,
         cmd='{py} ecm_temp_factor.py --out ecm_temp_factor.csv',
         inputs=['rpcwby_ecm.csv'],
         outputs=['ecm_temp_factor.csv'],
         why='온도 보정 계수.  UYPYDJ 는 25 C 뿐이라 RPCWBY 에서 가져온다.'),

    dict(id='pool', tier=2, minutes=8, measured=False,
         cmd='{py} ecm_pool.py --outdir cache/pool',
         inputs=['uypydj_ecm.csv', 'uypydj_ocv.csv'],
         outputs=[f'cache/pool/ecm_pool_{c}.csv' for c in CELLS],
         why='셀별 홀드아웃용 통합 표면.  홀드아웃 셀을 빼고 만든 표면이라 '
             '평가가 그 셀을 본 적이 없다.'),

    # ---- tier 3  라벨과 학습 데이터 --------------------------------------
    dict(id='label_dis', tier=3, minutes=6, measured=False,
         cmd='{py} sop_label.py --direction discharge',
         inputs=['uypydj_hppc_resistance.csv'],
         outputs=['sop_label_measured.csv'],
         why='방전 SOP 라벨.  extrap <= 1.5 인 것만 신뢰 라벨로 쓴다.'),

    dict(id='label_chg', tier=3, minutes=6, measured=False,
         cmd='{py} sop_label.py --direction charge',
         inputs=['uypydj_hppc_resistance.csv'],
         outputs=['sop_label_charge.csv'],
         why='충전 SOP 라벨.  방전보다 7 배 많다 (25 절).'),

    dict(id='trim_data_dis', tier=3, minutes=130, measured=True,
         cmd='{py} sop_trim_dataset.py --direction discharge --out cache/trim',
         inputs=['cache_t', 'cache/pool', 'sop_label_measured.csv',
                 'uypydj_ecm.csv', 'uypydj_hppc_resistance.csv'],
         outputs=[f'cache/trim/trim_{c}.npz' for c in CELLS],
         why='각 HPPC 펄스를 그 앞의 주행 이력 창 12 개와 짝짓고, 그 창에서 '
             'EW 특징 12 개를 뽑는다.  트림의 입력.'),

    dict(id='trim_data_chg', tier=3, minutes=110, measured=False,
         cmd='{py} sop_trim_dataset.py --direction charge --out cache/trim_chg',
         inputs=['cache_t', 'cache/pool', 'sop_label_charge.csv'],
         outputs=[f'cache/trim_chg/trim_{c}.npz' for c in CELLS],
         why='같은 것을 충전 방향으로.'),

    dict(id='soh_data', tier=3, minutes=45, measured=False,
         cmd='{py} soh_charge_dataset.py --raw ../raw/UYPYDJ '
             '--out cache/soh_charge.npz',
         inputs=['../raw/UYPYDJ', 'temp_audit_all.csv'],
         outputs=['cache/soh_charge.npz'],
         why='부분 충전 구간에서 SOH 를 맞히는 데이터셋.  전체 곡선을 주면 '
             '적분으로 용량이 그냥 나오므로 구간만 준다.'),

    # ---- tier 4  학습 --------------------------------------------------
    dict(id='trim_dis', tier=4, minutes=9, measured=True,
         cmd='{py} sop_trim.py --rung A8 --data cache/trim '
             '--out runs_trim_a8 --save-pred',
         inputs=[f'cache/trim/trim_{c}.npz' for c in CELLS],
         outputs=[f'runs_trim_a8/pred_A8_{c}.npz' for c in CELLS],
         why='채택 트림 (방전).  dR_fast 하나에서 배수 두 개.  파라미터 4 개.'),

    dict(id='trim_chg', tier=4, minutes=8, measured=True,
         cmd='{py} sop_trim.py --rung A8 --data cache/trim_chg '
             '--out runs_trim_a8_chg --save-pred',
         inputs=[f'cache/trim_chg/trim_{c}.npz' for c in CELLS],
         outputs=[f'runs_trim_a8_chg/pred_A8_{c}.npz' for c in CELLS],
         why='채택 트림 (충전).  32.7 에서 A3 를 이겼다.'),

    dict(id='trim_a3_dis', tier=4, minutes=11, measured=True,
         cmd='{py} sop_trim.py --rung A3 --data cache/trim '
             '--out runs_trim_v2 --save-pred',
         inputs=[f'cache/trim/trim_{c}.npz' for c in CELLS],
         outputs=[f'runs_trim_v2/pred_A3_{c}.npz' for c in CELLS],
         why='비교군 (12 특징, 26 파라미터).  논문의 사다리에 필요.'),

    dict(id='trim_a3_chg', tier=4, minutes=10, measured=False,
         cmd='{py} sop_trim.py --rung A3 --data cache/trim_chg '
             '--out runs_trim_chg_v2 --save-pred',
         inputs=[f'cache/trim_chg/trim_{c}.npz' for c in CELLS],
         outputs=[f'runs_trim_chg_v2/pred_A3_{c}.npz' for c in CELLS],
         why='비교군 (충전).'),

    dict(id='soh', tier=4, minutes=6, measured=True,
         cmd='{py} soh_cnn.py --save-model runs_soh_cnn '
             '--save-pred results/soh_pred.npz',
         inputs=['cache/soh_charge.npz'],
         outputs=['runs_soh_cnn/soh_CC.pt', 'results/soh_pred.npz'],
         why='SOH 팔.  파라미터 10,945 개, 셀 홀드아웃 RMSE 0.0135, '
             '편향 +0.0001 (온도 결함 8 곡선 제외 후 — 30.12).  '
             '추정 SOH 판 전부가 이 예측을 쓴다.'),

    # ---- tier 5  평가 ---------------------------------------------------
    dict(id='baselines', tier=5, minutes=4, measured=True,
         cmd='{py} sop_baseline_fill.py --data cache/trim --suffix "" && '
             '{py} sop_baseline_fill.py --data cache/trim_chg --suffix _chg',
         inputs=[f'cache/trim/trim_{c}.npz' for c in CELLS],
         outputs=['runs_trim_direct', 'runs_trim_shrink', 'runs_trim_rls',
                  'runs_trim_direct_chg', 'runs_trim_shrink_chg',
                  'runs_trim_rls_chg'],
         why='문헌 비교군 세 판 (직접 대입 / 축소 계수 / HPPC-RLS) 을 트림 '
             '디렉터리 형식으로 내보낸다.  32 절.'),

    dict(id='eval', tier=5, minutes=22, measured=True,
         cmd='{py} ../repro/run_evals.py',
         inputs=['runs_trim_a8', 'runs_trim_a8_chg', 'runs_trim_v2',
                 'runs_trim_chg_v2', 'runs_trim_direct', 'runs_trim_shrink',
                 'runs_trim_rls', 'results/soh_pred.npz'],
         outputs=['results/eval/'],
         why='모든 트림 판 x 양방향 x (정답 SOH, 추정 SOH) 를 SOP 반전에 '
             '통과시킨다.  16 / 25 / 29 / 32 절의 원천.'),

    dict(id='voltage', tier=5, minutes=1, measured=True,
         cmd='{py} ../repro/run_voltage.py',
         inputs=['runs_trim_a8', 'runs_trim_v2', 'runs_trim_direct',
                 'runs_trim_shrink', 'runs_trim_rls'],
         outputs=['results/tables/voltage.csv'],
         why='전압 RMSE 표.  전에는 그림 파일에 상수 열두 개로 박혀 있어서 '
             '표가 바뀌어도 그림이 안 따라오고 맞는지 확인할 수도 없었다.'),

    dict(id='safety', tier=5, minutes=3, measured=True,
         cmd='{py} ../repro/run_safety.py',
         inputs=['results/eval/'],
         outputs=['results/tables/safety.csv', 'results/tables/ladder.csv'],
         why='셀 하나씩 빼고 안전계수 lambda 를 잡고, 낙관율 / 최악 초과 / '
             '쓸 수 있는 전류를 낸다.  배치를 결정하는 값.'),

    dict(id='pack', tier=5, minutes=5, measured=True,
         cmd='{py} sop_pack2.py',
         inputs=['results/eval/'],
         outputs=['results/tables/pack.csv'],
         why='팩 수준.  N 셀의 min 을 거쳤을 때 여유가 유지되는가.  31 절.'),

    dict(id='soc_runs', tier=5, minutes=4, measured=True,
         cmd='{py} ../repro/build_soc_runs.py',
         inputs=['cache_t', 'cache/pool'],
         outputs=['results/soc_runs.pkl'],
         why='SOC 벤치가 쓰는 36 런을 만든다.  원래 /tmp 에 있었는데 재현 '
             '패키지가 /tmp 에 기대면 안 되므로 명시 단계로 뺐다.  고르는 '
             '규칙이 np.linspace 라 결정적이고, --check-against 로 기존 '
             '것과 동일함을 확인했다.'),

    dict(id='soc', tier=5, minutes=55, measured=True,
         cmd='{py} soc_perturb_bench.py && {py} soc_est_soh.py',
         inputs=['results/soc_runs.pkl', 'results/soh_pred.npz'],
         outputs=['results/tables/soc_perturb.csv'],
         why='순환을 끊은 SOC 벤치.  라벨을 만든 전류와 필터가 보는 전류를 '
             '어긋나게 한다.  30 절.'),

    dict(id='figures', tier=5, minutes=3, measured=True,
         cmd='{py} ../repro/fig_ladder.py && '
             '{py} ../repro/fig_soc_traj.py && '
             '{py} ../repro/fig_soh_traj.py && '
             '{py} ../repro/fig_sop_traj.py --direction discharge && '
             '{py} ../repro/fig_sop_traj.py --direction charge',
         inputs=['results/tables/ladder.csv', 'results/tables/voltage.csv',
                 'results/tables/soc_perturb.csv',
                 'results/soh_pred.npz', 'results/soc_runs.pkl',
                 'results/eval/'],
         outputs=['../results_fig_ladder.png', '../results_fig_soc_traj.png',
                  '../results_fig_soh_traj.png',
                  '../results_fig_sop_traj_discharge.png',
                  '../results_fig_sop_traj_charge.png'],
         why='논문 그림 다섯 장.  전부 results/ 의 표에서 직접 읽으므로 '
             '표가 바뀌면 그림도 따라 바뀐다 — 그림에 숫자를 손으로 '
             '박아 두면 표와 어긋난다.'),

    dict(id='extras', tier=5, minutes=2, measured=True,
         cmd='{py} ../repro/run_extras.py',
         inputs=['results/eval/', 'cache/trim', 'cache/trim_chg'],
         outputs=['results/tables/alpha.csv',
                  'results/tables/correlation.csv'],
         why='문서에만 있고 표에 없던 수치를 표로 뺀다 — 전달비 alpha, '
             '약함-낙관 상관, 결함 사이클 저항비, 배치 빌드 크기.  '
             '표에 없으면 verify.py 가 못 보고, 이번 세션의 오류가 정확히 '
             '그런 자리에서 나왔다.'),

    # ---- tier 6  보드 ---------------------------------------------------
    dict(id='mcu_export', tier=6, minutes=2, measured=True,
         cmd='{py} export_mcu_tables.py --rung A8 --out ../mcu/sop_tables.h && '
             '{py} export_soh_mcu.py --out ../mcu/soh_tables.h',
         inputs=['runs_trim_a8', 'runs_soh_cnn'],
         outputs=['../mcu/sop_tables.h', '../mcu/soh_tables.h'],
         why='32x16 격자와 트림 가중치를 C 헤더로.  A8 이라 EW 상태가 2 개.'),

    dict(id='mcu_measure', tier=6, minutes=25, measured=True, board=True,
         cmd='cd ../mcu/fw_sop && make && '
             'STM32_Programmer_CLI -c port=SWD -w Build/nmc_dst_cc/'
             'sop_bench.elf -v -rst && cd .. && {py} bench_sop.py --n 500 && '
             'cd ../analysis && {py} ../repro/run_extras.py',
         inputs=['../mcu/sop_tables.h', '../mcu/soh_tables.h'],
         outputs=['results/tables/mcu.csv'],
         why='NUCLEO-H563ZI 에서 DWT 사이클 카운터로 잰다.  주기당 us, '
             'Flash, 스택.  보드가 없으면 건너뛰고 표는 채워지지 않는다.'),
]

# 임계 경로가 아닌 것들 — 왜 남겨 두는가
EXPLORATORY = {
    'lstm_voltage.py / eval_voltage.py':
        '참조 논문(Chen et al.) 재현.  11.4 절의 "레퍼런스 LSTM 은 SOP '
        '이분 탐색을 못 돌린다" 의 근거.  논문에 인용되지만 수치는 고정이라 '
        '재실행이 필요 없다.',
    'rpcwby_*.py':
        '외부 셀 검증 (온도 축).  26 절.  RPCWBY 원본이 있어야 돈다.',
    'runs_trim_q*, runs_trim_cur*, runs_trim_w':
        '분위수 손실 / 전류 가중 변형의 탐색 흔적.  본문에 안 들어간다.',
    'soc_robust_sweep.py, soc_robust_loco.py, soc_ibias_*.py':
        '30 절의 실패 경로 — 잔차로 R 을 키우는 규칙.  결론이 "폐기" 라서 '
        '재현 대상은 아니지만, 30.2 의 서술이 이 스크립트들의 출력을 인용한다.',
    'dekf_soh.py, sop_from_ekf.py, ecm_refine.py':
        '시도했다가 접은 경로.  기록용.',
}
