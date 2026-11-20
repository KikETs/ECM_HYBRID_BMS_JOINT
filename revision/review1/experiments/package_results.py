import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np

RUN=Path(__file__).resolve().parents[1]
R=RUN/"repo"
S=RUN/"statistics"
E=RUN/"external"
I=RUN/"strict_inner"
data={}
def load(p):
    with p.open(newline="") as f:return list(csv.DictReader(f))
def md(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

required=("baseline","coupled_state","mcu","strict_all","comparators_cpu","comparators_gpu","cold","late_external","late_methods")
for step in required:
    msg=(RUN/"logs"/f"{step}.exit").read_text().strip()
    assert msg=="0",(step,msg)
assert len(list((I/"predictions").glob("*/pred_A8_*.npz")))==60
assert len(list((I/"basetrees").glob("no*/analysis/*/pred_A3_*.npz")))==300
assert md(RUN/"mcu/original_flash.bin")==md(RUN/"mcu/restored_flash.bin")
assert "27 tables compared, 0 differ" in (RUN/"logs"/"baseline.log").read_text()
assert not subprocess.check_output(["git","-C",str(RUN.parent/"repo"),"status","--porcelain"]).strip()
data["repo_commit"]=subprocess.check_output(["git","-C",str(R),"rev-parse","HEAD"],text=True).strip()
data["mcu_original_and_restored_flash_sha256"]=md(RUN/"mcu/original_flash.bin")
data["mcu"]=load(RUN/"mcu"/"coupled_points.csv")
mcu=data.pop("mcu")
good=[r for r in mcu if r["kind"]=="grid" and all(int(r["status_"+k])==0 for k in ("dis2","dis10","chg2","chg10"))]
data["mcu"]={"n_grid":sum(r["kind"]=="grid" for r in mcu),"n_all_success":len(good),
             "success_median_us":float(np.median([float(r["time_us"]) for r in good])),
             "observed_max_us":max(float(r["time_us"]) for r in mcu)}
data["r4_r5_main"]=load(S/"strict_paired.csv")
data["r4_r5_six_method"]=load(I/"methods"/"method_comparison_est.csv")
assert len(data["r4_r5_six_method"])==24
assert len(data["r4_r5_main"])==8
for r in data["r4_r5_main"]:
    x=[k for k in data["r4_r5_main"] if (k["direction"],k["tau_s"])==(r["direction"],r["tau_s"])]
    assert len(x)==2 and x[0]["n_rows"]==x[1]["n_rows"]
data["r3"]={"rows":len(load(RUN/"coupled_state"/"replay_rows.csv")),"matched_rc_sop":{}}
for d,t in (("discharge","2.0"),("discharge","10.0"),("charge","2.0"),("charge","10.0")):
    g=[r for r in load(RUN/"coupled_state"/"matched_fixed_lambda.csv")
       if r["direction"]==d and r["tau_s"]==t and r["model"]=="hyb"]
    w=[int(r["n_common"]) for r in g]
    data["r3"]["matched_rc_sop"][f"{d}_{t}"]={"n":sum(w),"n_cells":len(g),"before_exceed":sum(int(r["baseline_exceed"]) for r in g),
         "after_exceed":sum(int(r["replay_exceed"]) for r in g),
         "before_mean_pct":float(np.average([float(r["baseline_usable"]) for r in g],weights=w)),
         "after_mean_pct":float(np.average([float(r["replay_usable"]) for r in g],weights=w))}
data["r2_external"]={}
for d in ("discharge","charge"):
    for a,b in ((25,10),(10,25)):
        g=[r for r in load(E/"test2_temperature_transfer.csv") if r["model"]=="SOP_A8_W" and
           r["case"].split("|")[-1]==d and (int(r["fit_C"]),int(r["test_C"]))==(a,b)]
        assert len(g)==36
        data["r2_external"][f"{d}_{a}_to_{b}"]={k:sum(int(r[k]) for r in g) for k in ("n_test","frozen_exceed","refit_exceed")}
data["r1_external_near_cutoff"]=[r for r in load(RUN/"reference"/"mendeley_held_rank_endpoints.csv")
                                    if float(r["cutoff_distance_mV"])<=10.0001]
(RUN/"result_digest.json").write_text(json.dumps(data,indent=2)+"\n")
(RUN/"mcu"/"firmware.patch").write_bytes(
    subprocess.check_output(["git","-C",str(R),"diff","--","mcu"]))
readme="""RC-SOP Reviewer 1 experiment outputs

Source checkout 23b5c8b626637ef23145aaf80d5b723a553a35f1.
The published baseline regenerated 27 tables with zero differences.
The dated isolated checkout and checkpoints remain on the SSH server.

reference contains HPPC inverse-regression sensitivity and held-out pulse diagnostics.
external contains existing RPCWBY recordings, repeated model realizations on one cell.
coupled_state contains offline HPPC replay. Previous-available SOH, resting-OCV
initialization, estimated SOC and features from past samples at a 1 s grid.
The comparison uses original fixed factors on shared baseline pulse keys.
statistics contains the original and strict main comparison with paired cell-level bootstrap.
strict_inner/methods contains strict primary and secondary methods under the original
comparison policy of 0 A discharge and 0.5 A charge. matched_pairs.csv
reports pairwise effects from common pulses and a cell-level bootstrap.
mcu contains the firmware benchmark using a conservative hull mask and a zero-current
return on a flagged solver failure. Measurements are single-board DWT
samples with interrupts disabled. Stack watermark is cumulative. The target
board firmware was restored and the 2 MiB flash readback was byte-verified.
No new cell data or independently established WCET was produced.
"""
(RUN/"README_results.txt").write_text(readme)
include=[RUN/"result_digest.json",RUN/"README_results.txt",I/"datasets_manifest.json",
         RUN/"baseline"/"source_commit.txt",R/"mcu"/"sop_review_hull.h"]
for folder, suffix in (("reference","*.csv"),("external","*.csv"),("coupled_state","*.csv"),
                       ("statistics","*.csv"),("strict_inner/methods","*.csv"),("mcu","*.csv"),
                       ("mcu","*.patch"),("mcu","*.sha256"),("experiments","*.py"),
                       ("experiments","*.c"),("logs","*.exit"),("logs","*.log")):
    include.extend(sorted((RUN/folder).glob(suffix)))
include.extend(sorted(RUN.glob("run_*.sh")))
manifest={str(p.relative_to(RUN)):{"sha256":md(p),"bytes":p.stat().st_size} for p in include}
(RUN/"manifest_review1.json").write_text(json.dumps(manifest,indent=2)+"\n")
include.append(RUN/"manifest_review1.json")
target=Path(sys.argv[1])
with tarfile.open(target,"w:gz",compresslevel=5) as z:
    for p in include:
        z.add(p,arcname=str(p.relative_to(RUN)),recursive=False)
with tarfile.open(target,"r:gz") as z:
    for k,r in manifest.items():
        b=z.extractfile(k).read()
        assert len(b)==r["bytes"] and hashlib.sha256(b).hexdigest()==r["sha256"]
print(target, len(include), md(target))
