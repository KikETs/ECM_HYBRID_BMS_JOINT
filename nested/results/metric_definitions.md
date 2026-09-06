# metric_definitions.md — 초과 건수와 초과 크기의 정의 (행별 계산 근거: `metrics/row_level_d.csv`)

행 r, target 셀 c, 방향별 tolerance δ_d (discharge 0 A, charge 0.5 A), 셀별 안전계수 λ_c (inner 행에서 bisection으로 적합, `run_safety_strict.fit_lambda`, 최대 λ s.t. max(λ·|I_pred| − |I_ref|) ≤ δ_d on the inner rows).

| 지표 | 정의 | 코드 | 비고 |
|---|---|---|---|
| d_r | λ_c·\|I_pred,r\| − \|I_ref,r\| | `metric_definitions.py` | I_ref = HPPC 선형식 cutoff 교점(`I_star_lin4_A`) |
| exceedance | d_r > δ_d | `score()`: `over > tol` | 건수 = Σ 1[d_r > δ_d] |
| maximum positive difference | max(0, max_r d_r) | `score()['worst'] = max(over.max(), 0)` | 표 열 `worst_overshoot_A`, 출하 `worst_A` 가 이 정의 |
| maximum excess beyond tolerance | max(0, max_r (d_r − δ_d)) | `metric_definitions.py` 만 | 충전에서만 두 정의가 다르다 |

## 최종 자산(level 2, segment-aware solver v2)에서의 값 — `metrics/metric_summary.csv`

| 조건 | δ_d (A) | 행 | 초과 | 초과 셀 | max positive difference (A) | max excess beyond tolerance (A) | 기존 표 `worst_overshoot_A` |
|---|---|---|---|---|---|---|---|
| 10 s 방전 | 0 | 496 | 0 | — | 0.000 | 0.000 | 0.000 (= max positive difference) |
| 2 s 방전 | 0 | 138 | 1 | CC | 0.214 | 0.214 | 0.214 |
| 10 s 충전 | 0.5 | 2459 | 1 | BOOST_NEGPULSE_1S | 1.267 | 0.767 | 1.267 |
| 2 s 충전 | 0.5 | 2080 | 0 | — | 0.405 | 0.000 | 0.405 |

A0와 네 arm 전부의 값은 `metrics/metric_summary.csv`(32 조건행), 행별 d 는 `metrics/row_level_d.csv`(40,982행).

## 기존 표·문서와의 정의 불일치

| 위치 | 값 | 사용된 정의 | 비고 |
|---|---|---|---|
| `e2e_nested_full_*.csv` 열 `worst_overshoot_A`, `safety_strict_*_est.csv` 열 `worst_A`/`worst_overshoot_A` | 예: 2 s 충전 0.405 A | maximum positive difference | 정의는 일관됨. 이름이 "overshoot"이라 tolerance 초과량으로 읽힐 수 있음 |
| `manuscript_changes.md` D-02 제안 Table 3 "Worst overshoot (A)" 2 s 충전 = 0.000 | 0.000 | maximum excess beyond tolerance | 같은 표의 10 s 충전 1.267 은 maximum positive difference(초과분은 0.767) → **한 표 안에서 두 정의를 혼용** (검토 대상, 수정하지 않음) |
| `manuscript_changes.md` D-03 "by 0.214 A … by 1.267 A" | 1.267 | maximum positive difference | 10 s 충전은 tolerance 0.5 A 를 0.767 A 넘은 것 |
| `REVIEW_RESOLUTION.md` §3.1 "최대 초과량 … 0 (2 s 충전 최대 양의 차이 0.405 A는 tolerance 이내)" | 0 / 0.405 | 두 정의를 문장 안에서 구분함 | 일관 |
| 출하 원고 L156 "worst positive exceedance" / Table 3 註 | — | 정의 미기재 | 출하 percell `worst_A` 는 maximum positive difference |
| 외부 표 `worst_overshoot_W` (`external_*`) | W 단위 | max(0, max(λ·P_pred − P_meas)), tolerance 0 | 정의상 두 지표가 일치 |

권고 (본문 미수정): 표에는 두 지표를 각각 이름으로 구분해 싣거나(“max positive difference”, “max excess beyond tolerance”), 한 지표만 쓰되 어느 정의인지 註에 명시해야 한다. 이 문서는 정의를 확정할 뿐 본문·표 파일을 바꾸지 않는다.
