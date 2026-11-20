import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "repo" / "analysis"
OUT = ROOT / "reference"
OUT.mkdir(exist_ok=True)

def records(path):
    with path.open(newline="", encoding="utf-8") as source:
        yield from csv.DictReader(source)

def write(name, rows):
    with (OUT / name).open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def fit(I, V, limit):
    slope, offset = np.polyfit(I, V, 1)
    return (limit - offset) / slope if slope > 0.000001 else np.nan, slope, offset

groups = defaultdict(dict)
for r in records(ANALYSIS / "uypydj_hppc_resistance.csv"):
    groups[r["direction"], r["protocol"], r["cycle"], r["soc_group"], float(r["tau_s"])][int(r["rate_rank"])] = r
out, held_out = [], []
for direction, filename in (("discharge", "sop_label_measured.csv"), ("charge", "sop_label_charge.csv")):
    sign = -1 if direction == "discharge" else 1
    for r in records(ANALYSIS / filename):
        key = direction, r["cell"], r["cycle"], r["soc_group"], float(r["tau_s"])
        if key not in groups:
            raise ValueError(key)
        g = sorted(groups[key].values(), key=lambda q: -abs(float(q["I_A"])))
        I = np.array([float(z["I_A"]) for z in g])
        V = np.array([float(z["V_tau_V"]) for z in g])
        times = np.array([float(z["tau_actual_s"]) for z in g])
        limit = float(r["V_min_V"])
        tau = float(r["tau_s"])
        estimate, slope, offset = fit(I, V, limit)
        assert abs(estimate - float(r["I_star_lin4_A"])) <= 0.000501, (r, estimate)
        trusted_times = abs(times - tau) <= 0.15
        closest = int(np.argmin(np.where(trusted_times, abs(V-limit), np.inf)))
        bracket = bool(trusted_times.any() and min(V[trusted_times]) <= limit <= max(V[trusted_times]))
        lopo_changes, lopo_residual = [], []
        ref_key = f"{r['cell']}|{r['cycle']}|{tau:.1f}|{r['V_pre_V']}|{round(float(r['I_star_lin4_A']), 3)}"
        for n, point in enumerate(g):
            take = np.arange(len(g)) != n
            omitted, b, a = fit(I[take], V[take], limit)
            if np.isfinite(omitted) and omitted * sign > 0:
                delta = abs(omitted - estimate)
                v_err = abs((b * I[n] + a) - V[n]) * 1e3
                lopo_changes.append(delta)
                lopo_residual.append(v_err)
                held_out.append({"key": ref_key, "direction": direction, "cell": r["cell"], "tau_s": tau,
                                 "eligible_label": int(float(r["extrap"]) <= 1.5 and abs(estimate) > 0.5),
                                 "rank": point["rate_rank"], "I_A": I[n],
                                 "measured_endpoint_error_mV": abs(V[n] - limit) * 1e3 if trusted_times[n] else "",
                                 "refit_boundary_current_A": omitted,
                                 "measured_minus_refit_current_A": sign * (I[n] - omitted),
                                 "endpoint_prediction_error_mV": v_err})
        out.append({"key": ref_key, "direction": direction, "cell": r["cell"], "cycle": r["cycle"],
                    "soc_group": r["soc_group"], "tau_s": tau,
                    "I_ref_A": estimate, "extrap": float(r["extrap"]), "all_vs_high2_A": r["spread_A"],
                    "eligible_label": int(float(r["extrap"]) <= 1.5 and abs(estimate) > 0.5),
                    "bracket_at_horizon": int(bracket),
                    "closest_voltage_error_mV": abs(V[closest] - limit) * 1e3 if trusted_times.any() else "",
                    "closest_current_A": I[closest] if trusted_times.any() else "",
                    "closest_actual_time_s": times[closest] if trusted_times.any() else "",
                    "max_leave_one_rank_delta_A": max(lopo_changes, default=np.nan),
                    "max_leave_one_rank_endpoint_error_mV": max(lopo_residual, default=np.nan)})

summary = []
for direction in ("charge", "discharge"):
    for tau in (2., 10.):
        for group in ("all", "eligible", "eligible_bracketed", "eligible_unbracketed"):
            sub = [r for r in out if r["direction"] == direction and r["tau_s"] == tau and
                   (group == "all" or (r["eligible_label"] and
                   (group == "eligible" or r["bracket_at_horizon"] == (group == "eligible_bracketed"))))]
            def q(col, p):
                vals = [float(r[col]) for r in sub if r[col] != "" and np.isfinite(float(r[col]))]
                return float(np.quantile(vals, p)) if vals else ""
            summary.append(dict(direction=direction, tau_s=tau, group=group, n=len(sub),
                                n_cells=len({r["cell"] for r in sub}),
                                n_bracketed=sum(r["bracket_at_horizon"] for r in sub),
                                near_2p5mv=sum(r["closest_voltage_error_mV"] != "" and r["closest_voltage_error_mV"] <= 2.500001 for r in sub),
                                near_5mv=sum(r["closest_voltage_error_mV"] != "" and r["closest_voltage_error_mV"] <= 5.000001 for r in sub),
                                near_10mv=sum(r["closest_voltage_error_mV"] != "" and r["closest_voltage_error_mV"] <= 10.000001 for r in sub),
                                spread_median_A=q("all_vs_high2_A", 0.50), spread_p95_A=q("all_vs_high2_A", 0.95),
                                max_omitted_rank_shift_median_A=q("max_leave_one_rank_delta_A", 0.50),
                                max_omitted_rank_shift_p95_A=q("max_leave_one_rank_delta_A", 0.95)))
write("row_level.csv", out)
write("held_out_ranks.csv", held_out)
write("summary.csv", summary)
print(f"{len(out)} fitted references reproduced within published label rounding and {len(held_out)} rank holdouts.")
