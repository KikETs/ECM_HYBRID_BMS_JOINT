from pathlib import Path
from collections import defaultdict
import csv, hashlib, json, sys, tarfile

R=Path(__file__).resolve().parents[1]
def read(name):
    with (R/name).open(newline="") as f:return list(csv.DictReader(f))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for step in ("wcet","test1","test3_hybrid","state"):
    assert (R/"logs"/f"{step}.exit").read_text().strip()=="0",step
assert sha(R/"wcet/before.bin")==sha(R/"wcet/after.bin")
boundary=[r for r in read("boundary/cutoff_proximity_summary.csv") if float(r["band_mV"])==10.]
assert sum(int(r["count"]) for r in boundary)==217
direct=[r for r in read("boundary/direct_observable_a0_a8.csv") if r["model"]=="hyb" and r["comparable_preconditions"]=="1"]
assert len(direct)==67
pairs=read("external/independent_cell_records.csv")
assert len({(r["test_a"],r["test_b"]) for r in pairs})==3
F=[r for r in read("external/cross_physical_cell.csv") if r["model"]=="SOP_A8_W" and r["policy"]=="temperature_matched" and float(r["min_measured_W"])==0.]
g=defaultdict(list)
for r in F:
    g[tuple(r[k] for k in ("test_cell","direction","test_temp_C"))].append(r)
external=[]
for keys,r in sorted(g.items()):
    assert len(r)==36 and int(min(x["n_cal"] for x in r))>0
    external.append(dict(test_cell=keys[0],direction=keys[1],temp_C=keys[2],n_eval=sum(int(x["n"]) for x in r),
                         exceed_original=sum(int(x["original_exceed"]) for x in r),
                         exceed_cross_cell=sum(int(x["calibrated_exceed"]) for x in r)))
assert len(external)==8
cold=read("external/test3_two_other_cell_calibration.csv")
z=defaultdict(list)
for r in cold:
    z[(r["T_set_C"],r["tau_s"])].append(r)
cold_rows=[]
for (t,h),r in sorted(z.items(),key=lambda kv:tuple(map(float,kv[0]))):
    cold_rows.append(dict(temp_C=t,tau_s=h,n_model_pulse=len(r),n_unique_pulses=len({(x["member"],x["raw_row"]) for x in r}),
                          full_12_block_model_pulses=sum(int(x["n_blocks"])==12 for x in r),
                          original_exceed=sum(int(x["exceed_frozen"]) for x in r),
                          cross_cell_exceed=sum(int(x["exceed_cross_cell"]) for x in r)))
w=read("wcet/timing_summary.csv")
assert sum(int(r["n_frames"]) for r in w)==2082
r3=read("state/estimated_state_preceding_drive.csv")
tally=defaultdict(int)
for r in r3:tally[r["paired_status"]]+=1
assert tally["ok"]>0
r3_stats={}
sc=read("state/fixed_lambda_summary.csv")
for d in ("charge","discharge"):
    for tau in ("2.0","10.0"):
        ss=[r for r in sc if (r["model"],r["direction"],r["tau_s"])==("hyb",d,tau)]
        r3_stats[d+"_"+tau]={"rows":sum(int(r["n"]) for r in ss),"exceed":sum(int(r["n_exceed"]) for r in ss)}
digest=dict(boundary=boundary,matched_electrical_current_controls={"n":67,
              "observed_beyond_cutoff":sum(r["measured_endpoint_inside"]=="0" for r in direct),
              "benchmarks_admit_violating_measured_current":sum(r["model_allows_measured_violating_current"]=="1" for r in direct)},
            cross_cell_transfer=external,cold3=cold_rows,drive_state_status=dict(tally),
            drive_state_rcsop=r3_stats,mcu=w,flash_restored_sha256=sha(R/"wcet/after.bin"))
(R/"summary.json").write_text(json.dumps(digest,indent=2)+"\n")
(R/"README.txt").write_text("""RC-SOP reviewer 1 follow-up experiments

Boundary metrics use recorded UYPYDJ fixed-current endpoint measurements with
a withheld pulse, at least three other current levels for regression and an
explicit endpoint voltage band. Current differences are relative to the
measured near-cutoff current. The paired prediction table requires matching
SOC and pre-pulse voltage and uses the strict-fold calibration from the first
run. Both tables are retrospective.

RPCWBY Test 1 and Test 2 are two physical Samsung 30T cells. The raw source
overlap record lists their different recording channels and time spans. The
Test 3 2024 campaign also overlaps both cells on another recorder. Test 3
uses the protocol-aligned final constant-power records. Test 3 lacks the
active-history blocks expected by the original model. The exploratory
full-prefix variant uses the recorded blocks preceding the pulse and has
separate counts by available history length. Its measured endpoints use the
Test 3 protocol. The 2 s variant uses factors calibrated on 10 s external
cases and is a cross-horizon probe.

State replay uses a drive run that ended before each reference
characterization, estimated SOC and SOH in drive features, and cached
estimated SOC at the HPPC decision point. The two replay segments have
separate OCV initialization. The replay used existing measured data and fixed
strict-fold factors.

Timing results use DWT cycles on the connected 250 MHz Cortex-M33. The
timing data include the actual branch and a padded path forcing four times
24 inverse-solver iterations per cell. The maximum is empirical across the
tested vectors and cache states. Each pack-size trial processes instrumented
state instances; it does not involve a physical series pack. Flash readback
matches the pre-test image across both MiB.
""")
files=[R/"README.txt",R/"summary.json",R/"firmware/sop_core.c", R/"firmware/sop_core.h",R/"firmware/fw_sop/Src/main.c",
       R/"firmware/sop_review_hull.h"]
for folder in ("boundary","external","state","test3_raw_alignment"):
    files.extend(p for p in sorted((R/folder).rglob("*")) if p.is_file() and p.suffix in {".csv",".npz",".json"})
for p in ("wcet/timing_rows.csv","wcet/timing_summary.csv","wcet/restoration.sha256"):
    files.append(R/p)
for folder,ext in (("scripts","*.py"),("logs","*.log"),("logs","*.exit"),(".","run_*.sh")):
    files.extend(sorted((R/folder).glob(ext)))
names={str(f.relative_to(R)):{'sha256':sha(f),'bytes':f.stat().st_size} for f in files}
assert len(names)==len(files)
(R/"manifest.json").write_text(json.dumps(names,indent=2)+"\n")
files.append(R/"manifest.json")
archive=Path(sys.argv[1])
with tarfile.open(archive,"w:gz",compresslevel=6) as t:
    for f in files:t.add(f,arcname=str(f.relative_to(R)),recursive=False)
with tarfile.open(archive,"r:gz") as t:
    for n,h in names.items():
        b=t.extractfile(n).read()
        assert len(b)==h["bytes"] and hashlib.sha256(b).hexdigest()==h["sha256"]
print(archive,len(files),sha(archive))
