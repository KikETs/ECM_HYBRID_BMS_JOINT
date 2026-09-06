# EXTERNAL_INPUTS.md — files not in the base commit and not in this bundle

| Item | Needed for | Source / how to obtain | Size |
|---|---|---|---|
| `raw/UYPYDJ/*.zip` (fast-charging aging deposit, 6 cells) | full rebuild only (tiers 1–3) | doi 10.5683/SP3/UYPYDJ, unpack per DATA.md | ~24 GB (all three deposits) |
| `raw/RPCWBY/`, `raw/Mendeley/` | shipped external tables only (not recomputed here) | doi 10.5683/SP3/RPCWBY, doi 10.17632/9xyvy2njj3.2 | — |
| `analysis/cache_t/uypydj_<cell>_{Fifteen_Drive_Cycles,HPPC}.npz` | full rebuild: inner trim datasets, SOC-error EKF runs | `repro/run.py cache` (build_uypydj_cache.py), ~210 min | 2.5 GB |
| `analysis/cache/{trim,trim_chg}/trim_<cell>.npz` | shipped LOCO trims only (already tracked as predictions) | tiers 3 | — |
| `analysis/cache/soh_charge.npz` | inner SOH ridge (reused from previous session) | `repro/run.py soh_data` | small |
| `analysis/cache/pool/*` (shipped five-cell pools) | score stage | rebuilt automatically from the tracked `uypydj_{ecm,ocv}.csv` by the patched `ecm_pool.surfaces()` | small |
| `analysis/results/soc_runs.pkl` | not needed: the SOC-error npz in `assets/soc_err` were computed directly from `cache_t` by `build_soc_err_nested.py` | — | — |

Everything the **score** stage needs beyond the base commit is inside `assets/`. Everything the **full** stage needs beyond that is listed above.
