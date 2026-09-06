"""hull_solver v2 against the segment-wise exact enumeration, all outer label rows."""
import csv, multiprocessing as mp, os, sys
import numpy as np
V=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# --- paths (repository layout): modules from $RCSOP_BUNDLE/scripts, stored inputs from
#     $RCSOP_BUNDLE/verification unless a fresh copy exists in $RCSOP_WORK/verification,
#     outputs to $RCSOP_WORK/verification
_B = os.environ['RCSOP_BUNDLE']; _W = os.environ['RCSOP_WORK']
_OUTV = os.path.join(_W, 'verification'); os.makedirs(_OUTV, exist_ok=True)
def _inp(name):
    p = os.path.join(_OUTV, name)
    return p if os.path.exists(p) else os.path.join(_B, 'verification', name)
sys.path.insert(0, os.path.join(_B, 'scripts')); sys.path.insert(0, os.path.join(_B, 'scripts', 'verify'))

REPO=os.environ['RCSOP_REPO']; A=os.path.join(REPO,'analysis'); FLW=os.environ['RCSOP_WORK']
sys.path.insert(0,A)
import eval_sop_amps as EV; from ecm_pool import surfaces; import hull_solver
EX={(r['direction'],r['cell'],int(r['cycle']),float(r['tau']),round(float(r['meas']),3)):r for r in csv.DictReader(open(_inp('exact_roots_outer.csv')))}
def run(task):
    d,c=task; charge=d=='charge'; lab=EV.LABEL_CHG if charge else EV.LABEL
    rows=[r for r in csv.DictReader(open(lab,encoding='utf-8')) if r['cell']==c]
    EV.SOH_EST=dict(np.load(os.path.join(A,'results','soh_pred.npz')))
    surf=surfaces(c)[1 if charge else 0]; K=EV.trim_k_agg(c,os.path.join(A,'runs_trim_a8' if not charge else 'runs_trim_a8_chg'),'max')
    out=[]
    for r in rows:
        cyc=int(r['cycle'])
        if r.get('I_star_lin4_A') in (None,'','nan') or cyc not in K: continue
        soc=float(r['SOC']); soh=EV.soh_estimated(c,cyc,float(r['SOH'])); v_pre,v_lim,tau=float(r['V_pre_V']),float(r['V_min_V']),float(r['tau_s']); meas=float(r['I_star_lin4_A']); kf,ks=K[cyc]
        ex=EX[(d,c,cyc,tau,round(meas,3))]; rec=dict(direction=d,cell=c,cycle=cyc,tau=tau,meas=meas,extrap=float(r['extrap']))
        for key,(a_,b_) in (('A0',(1.0,1.0)),('A3',(kf,ks))):
            I,n,st=hull_solver.solve(surf,soc,soh,v_pre,v_lim,tau,a_,b_,charge,EV.r_eff)
            ne=int(ex[f'{key}_n_roots_exact']); mr=float(ex[f'{key}_exact_min_root']) if ex[f'{key}_exact_min_root'] not in ('','nan') else np.nan
            agree = (n==ne) and ((not np.isfinite(I) and ne==0) or (np.isfinite(I) and np.isfinite(mr) and abs(I-mr)<1e-3))
            rec.update({f'{key}_v2_root':I,f'{key}_v2_n':n,f'{key}_v2_status':st,f'{key}_exact_n':ne,f'{key}_exact_min':mr,f'{key}_v2_vs_exact':'agree' if agree else 'DIFFER'})
        out.append(rec)
    return out
if __name__=='__main__':
    CELLS=['BOOST','BOOST_NEGPULSE','BOOST_NEGPULSE_1S','BOOST_REST','CC','CC_CELL2']
    with mp.Pool(12) as pool: res=pool.map(run,[(d,c) for d in ('discharge','charge') for c in CELLS])
    rows=[r for rr in res for r in rr]
    with open(os.path.join(_OUTV,'validate_v2.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import collections
    for key in ('A0','A3'):
        for d in ('discharge','charge'):
            g=[r for r in rows if r['direction']==d]; e=[r for r in g if r['extrap']<=1.5 and abs(r['meas'])>0.5]
            dis=[r for r in e if r[f'{key}_v2_vs_exact']=='DIFFER']
            print(key,d,'eligible',len(e),'DIFFER',len(dis),'status',dict(collections.Counter(r[f'{key}_v2_status'] for r in e)))
            for r in dis[:6]: print('   ',r['cell'],r['cycle'],r['tau'],'v2',r[f'{key}_v2_n'],r[f'{key}_v2_status'],round(r[f'{key}_v2_root'],4) if np.isfinite(r[f'{key}_v2_root']) else None,'exact',r[f'{key}_exact_n'],r[f'{key}_exact_min'])
