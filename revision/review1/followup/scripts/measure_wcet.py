import csv, itertools, sys
from pathlib import Path
import numpy as np
import serial

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"firmware"))
import bench_sop as board
board.CMD["WCET"]=0x73
sizes=(1,12,48,96,192)
states=[(float(z),float(h),float(v),float(i)) for z,h,v,i in
        itertools.product(np.linspace(.28,.92,7),(.76,.84,.95),(2.88,3.25,3.67,3.98,4.11),(-12.,-.25,5.))]
pack_states=[(.34,.79,3.18,-14.),(.5,.86,3.53,-.25),(.72,.90,3.84,-11.),(.84,.93,4.04,.2),
             (.56,.79,3.47,5.),(.65,.84,3.65,-.9),(.77,.97,3.92,5.),(.92,.94,4.10,-12.)]
rows=[]
with serial.Serial("/dev/ttyACM0",921600,timeout=5,exclusive=True) as port:
    port.reset_input_buffer()
    info=board.query(port)
    assert info["clock_hz"]==250000000 and info["version"]==5
    for cache,d in (("enabled",0),("invalidated_before",0x20),("disabled",0x40)):
        for path,p in (("actual",0),("full_24_stress",0x10)):
            for k,n in enumerate(sizes):
                points=states if n==1 else pack_states
                for index,(s,h,v,i) in enumerate(points):
                    res=board.call(port,"WCET",k|p|d,s,h,v,0.,0.,i,[0.]*12)
                    code,cycles=res["iters"],res["cycles"]
                    it, failures = code>>16,code&0xFFFF
                    assert 0<cycles<2**31 and 0<=it<=24 and failures<=4*n
                    if p:
                        assert it==24 and failures==0
                    reduced=[res[key] for key in ("kf","ks","r_eff","i_star")]
                    assert all(np.isfinite(reduced)) and min(reduced)>=0 and max(reduced)<=400
                    if failures and not p:
                        assert min(reduced)==0.0
                    rows.append(dict(cache=cache,path=path,n_cells=n,point=index,SOC_init=s,SOH=h,
                        pre_V=v,measured_current_A=i,timing_cycles=cycles,time_us=cycles/250,
                        iterations_max=it,failed_inversions=failures,dis2_A=reduced[0],dis10_A=reduced[1],
                        ch2_A=reduced[2],ch10_A=reduced[3]))
                print(cache,path,n,"frames",len(points),flush=True)
out=ROOT/"wcet"
out.mkdir(exist_ok=True)
def write(name,data):
    with (out/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write("timing_rows.csv",rows)
summary=[]
for cache,path,n in sorted({(r["cache"],r["path"],r["n_cells"]) for r in rows}):
    rr=[r for r in rows if (r["cache"],r["path"],r["n_cells"])==(cache,path,n)]
    timings=[r["time_us"] for r in rr]
    full=[r["time_us"] for r in rr if r["failed_inversions"]==0]
    summary.append(dict(cache=cache,path=path,n_cells=n,n_frames=len(rr),
                        median_us=np.median(timings),max_us=max(timings),
                        full_success_frames=len(full),full_success_max_us=max(full,default=""),
                        max_iteration_count=max(r["iterations_max"] for r in rr),
                        frames_over_10ms=sum(x>10000 for x in timings)))
write("timing_summary.csv",summary)
print("Board:", info,"measured frames",len(rows))
