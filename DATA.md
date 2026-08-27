# Data

Raw data is not in this repository. Fetch the three sources into `raw/`
under the names below, then run `python3 repro/run.py cache`.

| Name | DOI | Size | Used for |
|---|---|---|---|
| UYPYDJ | `10.5683/SP3/UYPYDJ` | 22 GB | Everything: aging cycling, HPPC, OCV, drive cycles |
| RPCWBY | `10.5683/SP3/RPCWBY` | 1.7 GB | External validation; temperature axis (−20 to 40 °C) |
| Mendeley | `10.17632/cp3473x7xv.3` | 438 MB | Drive cycles (UDDS, HWFET, LA92, US06) |

UYPYDJ arrives as six `.zip` files, one per protocol. Do not unpack them;
`build_uypydj_cache.py` reads inside the archives.

## Cells

Samsung INR21700-30T, 3.0 Ah rated. Same cell, different aging protocol.

| Cell | Drive files | Cycles | Protocol |
|---|---:|---|---|
| CC | 100 | 7–1893 | constant-current fast charge |
| CC_CELL2 | 102 | 7–1899 | same protocol, second cell |
| BOOST | 100 | 7–1893 | boost charge |
| BOOST_NEGPULSE | 99 | 7–1753 | boost + negative pulse |
| BOOST_NEGPULSE_1S | 100 | 44–1899 | boost + 1 s negative pulse |
| BOOST_REST | 74 | 7–1368 | boost + rest |

Six cells, one chemistry. Every evaluation holds one cell out, so models
train on five.

## SOC axis

```
SOC(t) = 1.0 + Ah(t) / 3.0        Q_RATED = 3.0 Ah, fixed
```

Rated capacity, not measured. Matches the dataset readme. A cold or aged
cell does not reach SOC 1.0, which is intended: an axis that stretches with
cell state lets SOH and SOC cancel.

## Exclusions

`temp_audit_all.py` scans all 5,020 files and flags those whose logged
temperature falls outside 15–45 °C. 64 files are flagged.

| Kind | Flagged | Excluded by |
|---|---:|---|
| HPPC | 6 | characterisation layer (`uypydj_ecm`, `uypydj_hppc_resistance`) |
| OCV | 2 | characterisation layer (`uypydj_ocv`) |
| drive | 10 | characterisation layer |
| halfC | 20 | SOH dataset only |
| other | 20 | SOH dataset only |
| schedule | 6 | SOH dataset only |
| CAP | 0 | — |

These are logging faults, not deliberate cold tests. 28 of the 64 log
~0 °C in coherent blocks (BOOST 1458–1463, CC 1495–1501) spanning every
file kind, which resembles a designed low-temperature sequence. It is not:
HPPC resistance in those cycles sits on the aging trend line — ratio 0.98
against neighbours, where 0 °C would give 3–5×. One sensor failing for a
characterisation round explains the block.

Applying the drive exclusion to the SOP path changes 0.34 % of rows and
leaves λ, optimism and usable current identical to three decimals in both
directions. `sop_trim_dataset.py` carries the exclusion, so a rebuild picks
it up; the shipped cache predates it.

## Trustworthy labels

SOP labels come from HPPC pulses. When the measured peak current falls
short of the target, `I*` is extrapolated. `extrap = |I*| / |I_max_meas|`;
only `extrap ≤ 1.5` is used.

Discharge yields 631 trustworthy labels at τ = 10 s and 140 at τ = 2 s.
Charge yields 2,461 and 2,082. A 30 A cycler cannot measure discharge SOP
directly — the required current is 56–127 A at 25 °C.
