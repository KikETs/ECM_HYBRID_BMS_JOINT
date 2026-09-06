"""Exact segment-wise root enumeration on the INNER calibration rows of the
final assets (level 2: pools/trims/SOH built without the outer target),
primary arm, both models; compared with the solver-v2 results stored in
arm_rows_replace_level2_hull2.pkl (the rows that determine lambda_c)."""
import csv, multiprocessing as mp, os, pickle, sys
import numpy as np
V=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO=os.environ['RCSOP_REPO']; A=os.path.join(REPO,'analysis'); FLW=os.environ['RCSOP_WORK']; X=os.path.join(FLW,'20_calib_fix')
sys.path.insert(0,A); sys.path.insert(0,FLW); sys.path.insert(0,os.path.join(V,'solver'))
import eval_sop_amps as EV, ecm_pool
from exact_roots import enumerate_roots
CELLS=['BOOST','BOOST_NEGPULSE','BOOST_NEGPULSE_1S','BOOST_REST','CC','CC_CELL2']
D=pickle.load(open(os.path.join(FLW,'20_stateinput_nested','arm_rows_replace_level2_hull2.pkl'),'rb'))
def run(task):
    c,j,d=task; charge=d=='charge'
    pd_=os.path.join(X,'innerpool',f'no{c}')
    sd,sc=ecm_pool.surfaces(j,outdir=os.path.join(pd_,'pool'),ecm_csv=os.path.join(pd_,'uypydj_ecm.csv'),ocv_csv=os.path.join(pd_,'uypydj_ocv.csv'))
    surf=sc if charge else sd
    tdir=os.path.join(X,'innertrim',f"{'trim_chg' if charge else 'trim'}_no{c}"); K=EV.trim_k_agg(j,tdir,'max')
    out=[]
    for r in D[('inner',c,j,d,'eo','replace')]:
        if r['status']=='no_k' or r['cycle'] not in K: continue
        kf,ks=K[r['cycle']]; v_pre=float(r['V_pre']); v_lim=4.2 if charge else 2.5
        rec=dict(outer=c,cell=j,direction=d,cycle=r['cycle'],tau=r['tau'],extrap=r['extrap'],meas=abs(r['I_meas']),key=r['key'],v2_status=r['status'])
        for key,(a_,b_),col,st in (('A0',(1.0,1.0),'ecm','st_ecm'),('A8',(kf,ks),'hyb','st_hyb')):
            segs,roots,tang,notes=enumerate_roots(surf,r['soc_in'],r['soh_in'],v_pre,v_lim,r['tau'],a_,b_,charge)
            rs=sorted(roots,key=abs); v2=r.get(col,np.nan); v2st=r.get(st,'')
            agree=((len(rs)==0 and not np.isfinite(v2)) or (len(rs)>0 and np.isfinite(v2) and abs(abs(rs[0])-v2)<1e-3))
            rec.update({f'{key}_exact_n':len(rs),f'{key}_exact_min':rs[0] if rs else np.nan,f'{key}_tangent':';'.join(f'{x:.3f}' for x in tang),f'{key}_notes':' '.join(notes),f'{key}_v2':v2,f'{key}_v2_status':v2st,f'{key}_agree':'agree' if agree else 'DIFFER'})
        out.append(rec)
    return out
if __name__=='__main__':
    tasks=[(c,j,d) for c in CELLS for j in CELLS if j!=c for d in ('discharge','charge')]
    with mp.Pool(12) as pool: res=pool.map(run,tasks)
    rows=[r for rr in res for r in rr]
    with open(os.path.join(V,'solver','exact_roots_inner.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import collections
    print(len(rows),'inner rows')
    for key in ('A0','A8'):
        for d in ('discharge','charge'):
            e=[r for r in rows if r['direction']==d and r['extrap']<=1.5 and r['meas']>0.5]
            print(key,d,'eligible-by-label',len(e),'DIFFER',sum(1 for r in e if r[f'{key}_agree']=='DIFFER'),'tangent cand',sum(1 for r in e if r[f'{key}_tangent']),'v2 tangent status',sum(1 for r in e if r[f'{key}_v2_status']=='tangent'))
