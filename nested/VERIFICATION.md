# FINAL_VERIFICATION.md — RC-SOP audit 잔여 검증 (2026-09-06)

격리 디렉터리 `~/바탕화면/DL/ETRANS_AUDIT/rverify_20260906/`. 저장소 HEAD `6cced9b`, 작업 트리 시작·종료 0 변경(`00_state/start_state.txt`, `00_state/final_state.txt`). 기준 TeX·PDF·SI·그림, `manuscript_changes.md`·`REVIEW_RESOLUTION.md`·`figure_change_spec.md`·`results_summary.md`·기존 patch 파일은 읽기만 했고 해시가 동일하다(§7). 본문 수정문·캡션·patch·수정 원고는 새로 작성하지 않았다. commit/push/PR/보드 재플래시 없음.

산출물: `final_results.csv`(640행), `sensitivity_results.csv`(224행), `solver_diagnostics.csv`(27,124행), `metric_definitions.md`(+`metrics/`), `bundle/`(재현 묶음, 비공개), `firmware/`, `solver/`, `logs/`.

---

## 1. 확인된 오류와 해결

| # | 오류 | 확인 방법 | 해결 |
|---|---|---|---|
| E1 | 이전 hull-aware solver(v1, 전 구간 120점 로그 격자)가 rank 전류(hull 경계) 근처의 근과 근접한 근 쌍을 놓쳤다. A0 10 s 방전 3행의 "무근"은 무근이 아니라 **탐색 실패**였고(근이 −20.007/−20.036/−20.291 A, 세그먼트 [10, 20.06] 끝), 충전에서 A0 15행·A8 32행은 세 근 중 작은 두 근을 놓쳐 큰 근을 반환했다 | 보간 구조를 따르는 세그먼트별 열거(`solver/exact_roots.py`: rank 전류로 정의역 분할, 세그먼트당 150점 + brentq + 접촉근 후보 탐지) → `solver/exact_roots_outer.csv` | solver v2(`rfollow_20260905/hull_solver.py`, 구 버전은 `hull_solver_v1_grid120.py`): 세그먼트 인식, 세그먼트당 200점, \|f\| 국소최소 재탐색, 상태 ok/multi/noroot/tangent/hull. 전 outer 행에서 exact 열거와 **0건 불일치**(`solver/validate_v2.csv`). 영향받는 arm·여섯 방법·민감도 전부 v2로 재계산 |
| E2 | `manuscript_changes.md` D-02 제안 Table 3의 "Worst overshoot" 열이 두 정의를 혼용(2 s 충전 0.000 = tolerance 초과분, 10 s 충전 1.267 = 최대 양의 차이) | 행별 d 계산(`metrics/row_level_d.csv`) | 정의 확정(§2). 문서는 수정하지 않음 |
| E3 | `manuscript_changes.md` E-01 "started at the reference current, as used in the firmware" — 펌웨어는 기준 전류에서 시작하지 않는다(+5 A / −12 A 고정) | `mcu/sop_core.c:139-159` | 판정만(§3) |
| E4 | E-02 "its fixed-point iteration can settle on a larger root where the voltage equation has several" — 펌웨어 응답식에서는 다중 근이 관측되지 않았다 | 펌웨어 `sop_core.c`를 링크한 호스트 진단(`firmware/fw_roots.c`, 5,170 운영점) | 판정만(§3) |
| E5 | 이전 SOH 민감도(`rfollow/60_soh_sens`)는 폐기된 nested λ 기준 | — | 최종 자산 + v2 solver로 재실행(§4) |
| E6 | 이전 수정안 G-02는 외부 frozen factor를 0.6832 하나로 설명 | `repro/run_external_a8.py:380` `LAM_UYPYDJ = {'discharge': 0.6832, 'charge': 0.5860}` | 방향별 계수·producer 정리(§5) |

---

## 2. 초과 건수와 크기의 정의 (`metric_definitions.md`, `metrics/metric_summary.csv`)

d_r = λ_c·\|I_pred,r\| − \|I_ref,r\|; exceedance = d_r > δ_d (0 A 방전, 0.5 A 충전); **maximum positive difference** = max(0, max d_r); **maximum excess beyond tolerance** = max(0, max(d_r − δ_d)). λ_c는 표의 4자리 반올림값이 아니라 inner 행에서 같은 bisection으로 재적합한 전정밀도 값(반올림값과 <6e−5 차이, 표 값과 일치 확인).

최종(level 2, v2) primary arm A8:

| 조건 | 행 | 초과 | 초과 셀 | max positive difference (A) | max excess beyond tol (A) |
|---|---|---|---|---|---|
| 10 s 방전 | 496 | 0 | — | 0.000 | 0.000 |
| 2 s 방전 | 138 | 1 | CC | 0.214 | 0.214 |
| 10 s 충전 | 2459 | 1 | BOOST_NEGPULSE_1S | 1.267 | 0.767 |
| 2 s 충전 | 2080 | 0 | — | 0.405 | 0.000 |

모든 기존 표(`worst_overshoot_A`, 출하 `worst_A`)는 maximum positive difference다(전 조건 일치 확인). 정의 불일치는 D-02 한 곳(E2).

---

## 3. 펌웨어 sop_solve 대 오프라인 solver

| 항목 | 오프라인(출하 `eval_sop_amps.solve_I`) | 오프라인 v2(이번 최종) | 펌웨어 `mcu/sop_core.c` (6cced9b) |
|---|---|---|---|
| 초기값 | 평가 기준 전류 I_ref | 없음(구간 열거) | 충전 +5 A, 방전 −12 A 고정 |
| 반복 상한 | 12 | brentq ≤200 | 24 (`SOP_MAX_ITER`) |
| 수렴 조건 | \|ΔI\| < 1 mA | xtol 1e−9 A | \|ΔI\| < 1e−3 A (`SOP_TOL`) |
| 응답식 | theta: rank별 (SOC,SOH) LinearND 보간, rank 간 \|I\| 선형, τ1·τ2 보간, hull 플래그 | 동일 r_eff | 32×16 격자의 D2/D10 이중선형 + rank 선형(4 rank, 배치용 ALL 모델 표), τ2 = 8 s 고정, τ1 없음 |
| 정의역·clamp | I ∈ [−400,−0.1] / [0.1,400], SOC·SOH hull 밖이면 nan | 동일 정의역, hull 밖 세그먼트 제외 | I ∈ [−400,−0.1] / [0.05,400]; SOC 0.1–1.0, SOH 0.7–1.0로 **clamp(hull 플래그 없음)** |
| 실패 시 반환 | 상한 도달 시 마지막 반복값, R≤0이면 nan | nan + 상태 | 상한 도달 시 마지막 반복값(플래그 없음), R≤0이면 NAN |
| 감쇠 | 0.5 | — | 0.5 |

호스트 진단(`firmware/fw_root_diagnostics.csv`): 최종 primary arm의 적격 운영점 5,170개(방전 631·충전 4,539) × {A0, A8}에 대해 펌웨어의 `sop_r_eff`로 f(I)를 0.02 A 간격으로 훑어 근을 세고 `sop_solve`를 호출했다.
- 펌웨어 응답식에서는 **모든 운영점에서 근이 정확히 1개**(오프라인에서 다중 근이던 116개 운영점 포함). 이유: 펌웨어는 τ2 고정·D2/D10 이중선형이라 R_eff(I)가 rank 간 선형이며, 오프라인 다중 근은 τ1·τ2 보간을 거친 theta 응답식의 성질이다.
- `sop_solve`는 근에서 중앙 <1e−4 A; 24회 상한에 걸린 경우(방전 43/62, 충전 167/223)는 최대 0.16–1.0 A 떨어진 값을 플래그 없이 반환한다. 이것이 펌웨어의 실제 실패 모드다.
- 펌웨어 근과 오프라인 최소근의 차이 중앙 1.4 A(최대 10.8 A)는 모델 차이(배치용 ALL 격자 vs LOCO theta)이며 solver 문제가 아니다.

판정: E-01의 "reference current로 시작 … as used in the firmware"는 **틀림**(시작값·반복수·수렴 조건·응답식 모두 다름). E-02의 "펌웨어 fixed point가 큰 근에 정착할 수 있다"는 펌웨어 응답식으로는 **지지되지 않음**(다중 근 0건). 펌웨어에 대해 근거가 있는 진술은 "상한 도달 시 미수렴 값을 플래그 없이 반환"과 "hull 밖 clamp"뿐(기존 L337). 보드는 건드리지 않았고 호스트 진단은 `mcu/sop_tables.h`(배치 ALL 모델)를 사용했다. 보드 상 동작(부동소수 정밀도 등)은 미검증.

---

## 4. SOH·tolerance 민감도 (최종 자산 level 2 + v2 solver, `sensitivity_results.csv`)

프로토콜: primary arm(추정 SOH, 기준 SOC). 균일 SOH 이동 δ는 표면 축 민감도이며 두 SOH 정의 간 보정이 아니다. 행 집합 변화는 solver 적격성(hull/무근)으로만 생긴다(라벨·외삽비 고정).

**4.1 SOH 이동, λ 고정(최종 baseline λ_c, tol_fit = δ_d)** — A8, 자체 적격 행:

| 조건 | δ | 행(+추가/−제외 vs 0) | 초과(셀) | max pos diff / excess (A) | usable / worst-cell (%) |
|---|---|---|---|---|---|
| 10 s 방전 | −0.0074 / −0.0037 / 0 / +0.0037 / +0.0074 | 468(+6/−34) / 479(+3/−20) / 496 / 508(+20/−8) / 524(+39/−11) | 0 / 0 / 0 / 2(1) / 4(2) | 0 / 0 / 0 / 0.369 / 0.897 | 63.88·60.15 / 65.02·61.04 / 65.96·61.88 / 67.11·62.79 / 68.58·64.21 |
| 2 s 방전 | 동일 | 127 / 132 / 138 / 146 / 154 | 0 / 1 / 1 / 2 / 6(2) | 0 / 0.117 / 0.214 / 1.429 / 2.910 | 57.08·55.15 / 58.26·55.79 / 59.24·56.61 / 61.22·58.63 / 62.84·59.62 |
| 10 s 충전 | 동일 | 2417 / 2432 / 2459 / 2454 / 2451 | 1 / 0 / 1 / 1 / 1 | 1.765·1.265 / 0.419·0 / 1.267·0.767 / 1.979·1.479 / 1.722·1.222 | 55.70·48.13 / 56.32·48.69 / 57.04·49.43 / 57.74·50.10 / 58.42·50.84 |
| 2 s 충전 | 동일 | 2043 / 2055 / 2080 / 2076 / 2074 | 1 / 0 / 0 / 1 / 1 | 0.779·0.279 / 0.383·0 / 0.405·0 / 0.761·0.261 / 2.009·1.509 | 55.44·46.89 / 56.15·46.73 / 57.05·47.76 / 57.73·48.20 / 58.34·48.20 |

공통 행(δ 전체 교집합: 455/127/2405/2033행; `common` 행의 added/removed 열은 own 집합의 값을 그대로 옮긴 것이며 공통 집합 자체는 δ에 따라 변하지 않는다)에서는 방전 초과가 0이고 usable 변화는 ±1 pp/0.0037이다(교집합 밖으로 밀려나는 행이 초과를 만든다는 뜻). λ 재보정 변형(`soh_refit_lambda`)은 별도 행: 충전에서 λ_min이 0.41–0.45로 떨어지며 worst-cell 39.8–44.0 %가 되는 경우가 있어(−δ), 이는 재보정 효과다.

**4.2 tolerance** — `tol_eval_only`(λ 고정, 판정 임계값만): 방전은 tol_eval 0→1.0 A에서 초과 0/1→0, usable 불변; 충전 10 s는 tol_eval 0/0.25/0.5/1.0 A에서 초과 15/6/1/1, 2 s는 17/5/0/0(λ·usable 불변). `tol_joint`(tol_fit = tol_eval, λ 재적합): 10 s 충전 tol 0 A → usable 45.12 %(λ_min 0.424), 1.0 A → 62.31 %(초과 1, max excess 2.963 A). joint 분석은 보정 정책 민감도이며 판정 임계값 민감도가 아니다.

---

## 5. 외부 평가의 계수와 계산 경로 (`repro/run_external_a8.py`, `repro/run_chen2026_baseline.py`, `repro/run_external_crate.py`)

| 결과 | producer | 모델·상태 입력 | endpoint·제약 | frozen factor | margin·exceedance |
|---|---|---|---|---|---|
| Fig 6(a) / `external_a8_surfaces.csv` (RC-SOP, Test #2) | `run_external_a8.py --all-surfaces` | 6개 **LOCO A8 fold**(`runs_trim_a8/model_A8_<cell>.pt`, 12 블록 max) × 6개 내부 **per-cell 표면** `ECMSurface(cell)`(pooled 아님) + 온도 인자; SOH = RPCWBY resistance CSV의 사이클별 값, SOC = 시트 값, T = 셀 온도 중앙값 | 정전력 이분 탐색(Chen 프레임워크), 10 s, 2.55–4.15 V, 30 A 방전 / 15 A 충전, 허용 5 mV·10 mA, 50회 | **discharge 0.6832, charge 0.5860** (`LAM_UYPYDJ`; = `safety.csv`의 oracle arm 10 s pooled-median λ 0.683/0.586) | λ_needed = min(P_meas/P_pred); margin = λ_needed/λ; exceedance = λ·P_pred − P_meas > 0 (**tolerance 0 W, 충전도**); 표는 fold 6개의 최소 margin |
| Fig 6(b) / `external_temp_surfaces.csv` (nominal, Test #3) | `run_chen2026_baseline.py --all-surfaces` | 명목 2RC(k=1), per-cell 표면, **SOH 고정 2.6/3.0**, T 매핑 | 정전력 탐색, 10 s, 2.55–4.1 V, 30 A / 5 A | **discharge 0.6832만** | 온도 그룹별 λ_needed/0.6832; warm = 0–40 °C 최소 |
| SI Table S1 / `external_crate_surfaces.csv` (nominal, Test #8, 0 °C) | `run_external_crate.py --all-surfaces` | 명목 2RC, per-cell 표면, SOH 2.6/3.0 | chen_search 동일 제약 | **0.6832** | prior rate별 λ_needed 범위, margin_min |

확정: 외부 결과는 **출하 5셀 자산의 LOCO fold와 출하 pooled λ(방전 0.6832, 충전 0.5860)** 에 대한 증거이며, 이번 nested λ_c(방전 0.619–0.717, 충전 0.507–0.597)와는 무관하다. 방향별 계수가 다르므로 "0.6832 하나"라는 설명은 불완전하다. 외부 표면은 pooled LOCO 표면이 아니라 각 셀 자체 표면이다(본문 "six internal surface realizations"). 외부 결과는 재계산하지 않았다.

---

## 6. 최종 검증 수치와 이전 대비 변화 (`final_results.csv`, variant `level2_hull_v2`, `final=1`)

| 조건 | 출하 → 이전 nested(leaky) → level 2 v1 → **level 2 v2** : 행 / 초과 / usable % / worst-cell % |
|---|---|
| 10 s 방전 | 496/1/68.89/58.94 → 496/0/68.56/57.79 → 493/0/65.95/61.88 → **496/0/65.96/61.88** (v1이 제외한 3행은 A0 근이 있음) |
| 2 s 방전 | 138/1/59.99/57.78 → 138/0/58.90/56.52 → 138/1/59.24/56.61 → **138/1/59.24/56.61** (CC) |
| 10 s 충전 | 2459/2/55.54/48.13 → 2459/1/56.16/44.11 → 2459/1/57.11/49.47 → **2459/1/57.04/49.43** |
| 2 s 충전 | 2080/3/57.75/49.96 → 2080/2/55.17/40.04 → 2080/0/57.11/48.20 → **2080/0/57.05/47.76** |

A0 대비 이득(v2): +5.27 / +2.70 / +5.98 / +4.60 pp. 여섯 방법 순위(v2): A8 2 / 5 / 4 / 3위, FFRLS만 10 s 두 조건에서 구간 분리. v1→v2 변화는 ≤0.07 pp(usable), worst-cell ≤0.44 pp, 초과 동일, 10 s 방전 행 +3. 네 arm 공통 행(v2): 427/101/2431/2057 → `final_results.csv` `arms_paired/fixed/outside_common`.

---

## 7. 재현 범위 (`bundle/`)

- 기준 커밋(공개): `6cced9b477effde33b472ce383adb3f868a4b53b`. **이 묶음은 공개하지 않았다**; 그 커밋만으로는 nested 결과가 재현되지 않는다.
- 코드 패치: `code_patch/repo.patch` — `analysis/ecm_pool.py`(surfaces→build에 CSV 전달), `analysis/sop_trim_dataset.py`(`--pool-src`). 새 스크립트는 `scripts/`(경로는 `RCSOP_REPO/RCSOP_BUNDLE/RCSOP_WORK` 환경변수).
- 실제 수행한 재현: 저장소를 **로컬 clean clone**(`clean_clone/repo`, 6cced9b, cache·cache_t·raw 없음)한 뒤 `reproduce.sh patch` → `check-pools`(30 inner 풀을 5셀 CSV에서 재생성: 묶음과 바이트 동일, 출하 풀과 상이, theta·SOP 전류가 CSV 내용에 반응 — R0 상대차 최대 0.07–0.38, \|ΔI\| 최대 2–8 A) → `score`(묶음 체크포인트로 arm·여섯 방법 채점, v2 solver) → `compare_final.py`로 최종 CSV와 대조. 결과: **27개 최종 표(arm 12 + 여섯 방법 15) 전부 동일**(`logs/clone_score.log`, `logs/clone_score_methods.log`; 수치 허용오차 1e−3, 문자열 정확 일치). 기준 커밋에 추적되지 않은 출하 LSTM/GRU/FFRLS outer 예측(`repro/run_sop_seq_baselines.py` 산출)은 `assets/outer_trims`로 묶음에 포함했다 — 그 커밋만으로는 이 여섯 방법 표를 만들 수 없다.
- 수행하지 않은 재현: 전체 재학습(`full`: inner 데이터셋 60개, A8·비교 방법 재학습, EKF SOC 오차) — 작업 사본(`rfollow_20260905/20_calib_fix`)에서 1회 수행했고 clean clone에서는 반복하지 않았다. 필요한 외부 입력은 `EXTERNAL_INPUTS.md`.
- 숨은 의존: `score` 단계는 저장소 추적 파일 + `assets/`만 읽는다(출하 풀은 추적 CSV에서 재생성). inner SOH 예측(`assets/innersoh`)은 2026-09-05 세션 산출을 재사용(표면 무관).

## 8. 남은 미검증 항목

- 보드 실행 자체(부동소수·타이밍)에서의 solver 거동: 호스트 진단만 수행. 필요 근거: 보드 로그 또는 parity 실행(`repro/run_parity.py`, 보드 불필요)으로 v2 오프라인과의 차이 정량화.
- 외부 평가를 nested λ_c로 재계산할지는 계산하지 않았다(저자 결정 사항이며 본문 수치와 무관).
- `tangent` 상태 행: §10에서 정리. 보수적 제외 사례로 남김. RC-SOP 결과 불변, 가상 근 인정 시 A0 λ 변경 사례 1건.
- 이전 세션 문서(`manuscript_changes.md` D-02/E-01/E-02/G-02, `REVIEW_RESOLUTION.md` §1.5의 "A0 3행 무근")는 이번 결과와 어긋나며 수정하지 않았다(지시).

## 9. 보호 대상 미변경 확인
`00_state/final_state.txt`: 기준 TeX/PDF/SI/그림, `manuscript_changes.md` 등 이전 문서, `FINAL_REPORT.md` 해시 시작 시점과 동일; 저장소 작업 트리 0 변경(`analysis/ecm_pool.py` 등은 clean clone 안에서만 패치됨).

---

## 10. 접촉근 후보(`tangent` 상태) 행의 처리와 영향 (`solver/tangent_rows.csv`, `solver/tangent_and_differ_check.txt`)

정의: solver v2는 유효 세그먼트 안에서 부호 변화가 없고 \|f\|의 국소최소가 5 mV 미만인 곳을 40배 세밀 재탐색한 뒤에도 부호 변화가 없으면 `tangent`로 표시하고 **근 없음(실패)** 으로 처리한다. exact 열거(150점/세그먼트, 2 mV 기준)는 이 행들에서 근 0개·후보 0개를 보고했으므로 "0건 불일치" 판정에서는 **양쪽 모두 근 0개로 일치**한 것으로 세었다(값 비교 없음).

최종 자산에서 `tangent`는 (행, 모델) 쌍 17건, 전부 10 s 방전: outer 평가 행 2건(A8, BOOST_NEGPULSE_1S cyc 1243·1431), inner 보정 행 15건(A8 4, A0 11). 2000점/세그먼트 세밀 스캔 결과 17건 모두 \|f\| 최소(0.45–4.4 mV)가 **세 번째 rank 전류(≈20.0–20.6 A) 바로 안쪽**에 있고 부호 변화는 없다. 즉 접촉근이 아니라 hull 경계 바로 밖에 근이 있는 "경계 dip"이며, 유효 정의역 안에는 근이 없다. 보수적 제외가 맞다.

집계 단위는 (행, 모델) 쌍이다(같은 행이 두 모델에서 모두 tangent인 경우는 없다). 상호배타 분류(`solver/tangent_classification.md`, `tangent_rows.csv`에서 생성):

| level | model | 분류 | (행, 모델) 쌍 수 |
|---|---|---|---|
| inner | A0 | WOULD BIND lambda | 1 |
| inner | A0 | excluded regardless (other model no root) | 2 |
| inner | A0 | inner, other model ok, not binding | 8 |
| inner | A8 | excluded regardless (other model no root) | 4 |
| outer | A8 | excluded regardless (other model no root) | 2 |
| 합계 | | | 17 |

가상의 영향(dip 위치를 근으로 인정할 경우):
- 상대 모델이 무근이라 어차피 제외되는 쌍 8건 = outer A8 2건 + inner A8 4건 + inner A0 2건 → 최종 집계·적격 행·λ·순위 영향 없음.
- inner A0 9건은 상대 모델(A8)이 정상이라 A0의 보정 집합에 들어갈 수 있었다: 8건은 (I_ref+δ)/\|I_dip\| ≥ λ_c로 구속하지 않고, **1건**(outer CC_CELL2, 셀 CC cyc 1725)은 A0의 λ_c(10 s 방전, CC_CELL2 fold)를 0.5352→0.5221로 낮췄을 것이다.
- 결론은 다음으로 한정한다: **RC-SOP(A8)의 λ·적격 행·usable·초과·방법 순위는 불변이며, 가상의 근을 인정할 때 A0(명목 모델)의 λ가 바뀌는 사례가 1건 있다.** 모든 모델에 영향이 없다는 요약은 하지 않는다.

## 11. exact 대조를 inner 보정 예측 행에도 적용 (`solver/exact_roots_inner.csv`, 67,406행)

λ_c를 결정하는 inner 행(level 2 자산: 4셀 풀·inner 트림·inner SOH, primary arm) 전부에 대해 세그먼트별 exact 열거를 실행하고 v2 결과와 대조했다. 1차 대조에서 A8 방전 171행·A0 방전 4행이 "불일치"로 나왔으나, 이는 `eval_arms_nested`가 **한 모델이라도 실패한 행의 근 값을 저장하지 않는**(상태만 저장) 구조 때문이며(그 175행은 모두 상대 모델이 무근이라 제외된 행), 해당 행의 v2 근을 다시 계산해 비교하면 **175/175 일치**(`tangent_and_differ_check.txt` (a)). 그 외 라벨 적격 inner 행(방전 7,601·충전 22,553 × 2모델)은 근 개수와 최소근이 1e−3 A 안에서 일치한다. inner의 다중 근(충전 A8 1,843행·A0 1,209행)도 exact와 동일하게 열거되었다.

## 12. 문서 버전 관계 (최종 근거의 우선순위)

| 문서 | 상태 | 최종 근거로 사용 여부 |
|---|---|---|
| `raudit_20260905/90_report/FINAL_REPORT.md`, `95_manuscript/source_revised/`, `Fig4/5_corrected.png` | 이전 감사(2026-09-05 전반). inner 표면 leaky | **사용 금지** |
| `rfollow_20260905/REVIEW_RESOLUTION.md`, `results_summary.md`, `manuscript_changes.md`, `figure_change_spec.md`, `patches/*.textonly.patch`, `comparison_results.csv` | 후속 검증(2026-09-05 후반). level 2 + solver v1(격자). §1.5 "A0 3행 무근", D-02 정의 혼용, E-01/E-02 펌웨어 서술, G-02 단일 계수는 이번 결과와 어긋남 | 방법·계보 서술의 근거로만; **수치·solver·펌웨어·외부 계수 서술은 이 문서(§1–§6)가 대체** |
| `rverify_20260906/FINAL_VERIFICATION.md`, `final_results.csv`(variant `level2_hull_v2`), `sensitivity_results.csv`, `solver_diagnostics.csv`, `metric_definitions.md`, `bundle/final/` | 2026-09-06, solver v2, clean clone 재현 | **최종 근거** |

이전 문서는 지시대로 수정하지 않았다. `rfollow_20260905/hull_solver.py`만 v2로 교체했고 구 버전은 `hull_solver_v1_grid120.py`로 보존했다(그 디렉터리의 `*_level2_hull_*` 산출물은 v1 결과, `*_level2_hull2_*`가 v2 결과).

재현 로그: `logs/clone_score.log`(첫 실행: 묶음에 `soc_err_outer_replace_fixpool.npz`·추적되지 않은 LSTM/GRU/FFRLS outer 예측이 빠져 있어 실패 → 묶음 보완), `logs/clone_score_methods.log`(arm 12 + 방법 15 표 27/27 동일).

## 13. 상태

계산 검증 완료 · 원고 미반영(TeX/PDF/SI/그림/수정안 문서 미변경). 이 상태는 검증 수행자의 보고이며, 전달 묶음 자체의 독립 검증은 별도 검토자의 몫이다.

## 14. 공개 저장소 반영과 exact 열거의 성격

- 이 디렉터리(`nested/`)에는 §7의 재현 묶음 전체(코드 패치, 스크립트, 예측 npz를 포함한 `assets/`, 최종 표, `verification/`의 exact 열거·v2 대조·펌웨어 진단 CSV, manifest, clean-clone 로그)가 들어 있다. `reproduce.sh score`는 저장소 추적 파일과 `assets/`만으로 27개 표를 다시 만든다.
- `exact_roots*.py`는 rank 전류로 정의역을 나눈 뒤 세그먼트당 150점 표본과 brentq, 2 mV 미만의 \|f\| dip 표시로 근을 세는 **수치적 대조**다. 표본 간격보다 좁은 근 쌍이나 dip이 2 mV를 넘는 접촉근은 놓칠 수 있으므로, "0건 불일치"는 두 수치 방법의 일치이지 근의 완전성에 대한 수학적 증명이 아니다.
