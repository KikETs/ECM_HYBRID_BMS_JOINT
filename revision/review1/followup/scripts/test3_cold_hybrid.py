from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import re,sys,zipfile
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent; R=BASE/"repo"
sys.path[:0]=[str(BASE/"experiments"),str(R/"analysis"),str(R/"repro")]
import strict_features as fast
from ecm_pool import surfaces
from ecm_surface import ECMSurface
from run_chen2026_baseline import chen_search, SOH_25C
from run_external_a8 import load_a8_folds, trim_k
from stages import CELLS

SRC=R/"raw/RPCWBY/3_Test_3.zip"
AUD=ROOT/"test3_raw_alignment/cp_pulses.csv"

def job(c):
    sd,sc=surfaces("CC")   # fixed history extractor shared across surface realizations
    surface=ECMSurface(c,"discharge")
    health=SOH_25C
    folds=load_a8_folds()
    def ocv(soc):
        v,_=surface.ocv(soc,health)
        h,_=surface.hyst_M(soc,health)
        return float(v[0]),h
    allrows=[]
    raw=pd.read_csv(AUD)
    filtered=raw[(raw.direction=="discharge") & raw.last_occurrence & raw.full_duration & raw.horizon_s.isin([2,10])]
    filtered=filtered[(abs(filtered["voltage_V"]-2.55)<=.040) | (filtered.current_A<=-29.90)]
    with zipfile.ZipFile(SRC) as z:
        for member,recs in filtered.groupby("member",sort=True):
            with z.open(member) as f:
                frame=pd.read_csv(f,encoding="utf-8-sig",low_memory=False)
            temp_col=next(k for k in frame if k.startswith("Aux_Temperature"))
            t,i,v,temp,s=(frame[k].to_numpy(float) for k in ("Test_Time(s)","Current(A)","Voltage(V)",temp_col,"SOC"))
            valid=np.isfinite(t+i+v+temp+s)
            tt=t[valid]
            assert (np.diff(tt)>=0).all()
            S=np.clip(s[valid],.02,1.)
            history=fast.calc(sd,sc,tt,i[valid],v[valid],temp[valid],S,np.full(len(tt),health))
            times=[]
            mark=tt[0]+600.
            while (j:=int(np.searchsorted(tt,mark)))<len(tt):
                times.append(tt[j]);mark=tt[j]+600.
            assert len(times)==len(history)
            for row in recs.itertuples():
                before=np.array(times)<float(row.prior_time_s)
                duty=history[:,5]>=.05
                X=history[before][-12:,:12]
                temp_test=int(row.filename_temp_C) if np.isfinite(row.filename_temp_C) else 25
                n=len(X)
                common=dict(surface=c,member=member,raw_row=row.first_row,prior_time_s=row.prior_time_s,
                            T_set_C=temp_test,T_meas_C=row.temperature_C,soc=row.soc,tau_s=row.horizon_s,n_blocks=n,n_active_blocks=int((before & duty).sum()),
                            measured_W=abs(row.power_W),endpoint_V=row.voltage_V,endpoint_A=row.current_A)
                nom,st0=chen_search(surface,row.soc,health,row.temperature_C,row.horizon_s,ocv)
                for fold,values in folds.items():
                    pred=np.nan
                    st="no_prior_blocks"
                    if n:
                        kf,ks=trim_k(values,X)
                        if np.isfinite(nom):
                            # Apply the same branch multipliers through the
                            # existing constant-power reference simulator.
                            from run_chen2026_baseline import simulate_power, OutOfHull
                            a,b=0.,200.
                            st="no_root"
                            for _ in range(50):
                                P=(a+b)/2
                                try:
                                    V,I=simulate_power(surface,row.soc,health,P,row.temperature_C,row.horizon_s,ocv,kf=kf,ks=ks)
                                except OutOfHull:
                                    pred,st=np.nan,"out-of-hull"
                                    break
                                if not np.isfinite(V):
                                    b=P
                                    continue
                                v_gap=V-2.55
                                i_gap=30.-abs(I)
                                # Mirrors the existing Test #3 search tolerance.
                                if abs(v_gap)<=.005 or abs(i_gap)<=.01:
                                    pred,st=P,"ok"
                                    break
                                if v_gap<0 or i_gap<0: b=P
                                else: a,pred,st=P,P,"ok"
                                if b-a<1e-4: break
                        else:
                            st="nominal_"+st0
                    allrows.append(dict(**common,fold=fold,nominal_W=nom,rcsop_W=pred,status=st))
    path=ROOT/"external"/f"Test3_all_prefix_{c}.csv"
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    print(c,len(allrows),"finite",sum(np.isfinite(r["rcsop_W"]) for r in allrows),flush=True)
if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=6) as p:
        list(p.map(job,CELLS))
