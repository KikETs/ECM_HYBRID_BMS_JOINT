from collections import defaultdict
import ctypes
from pathlib import Path
import re

import numpy as np
from ecm_surface import T_REF_C
from temp_defects import defective

def interp_temp(factor, soc, I, temps):
    g = np.ones(len(soc))
    if factor is None:
        return g
    change = np.abs(temps - T_REF_C) > .05
    pending = change.copy()
    sets = defaultdict(list)
    by_soc = defaultdict(list)
    for r in factor.rows:
        bounds = r["SOC_lo"], r["SOC_hi"]
        sets[bounds + (r["I_lo"], r["I_hi"])].append(r)
        by_soc[bounds].append(r)
    current = abs(I)
    def apply(mask, rows):
        if mask.any():
            rows = sorted(rows, key=lambda r: r["T_cell_C"])
            g[mask] = np.interp(temps[mask], [r["T_cell_C"] for r in rows], [r["g"] for r in rows])
    for (sl, sh, il, ih), rows in sets.items():
        mask = change & (sl <= soc) & (soc < sh) & (il <= current) & (current < ih)
        apply(mask, rows)
        pending[mask] = False
    for (sl, sh), rows in by_soc.items():
        apply(pending & (sl <= soc) & (soc < sh), rows)
    return g

def ecm_values(surface, soc, soh, I, T):
    w = np.interp(abs(I), surface.rank_I, np.arange(len(surface.rank_I)))
    lo = np.floor(w).astype(int)
    hi = np.ceil(w).astype(int)
    frac = (w-lo)[:,None]
    low = np.empty((len(soc),5))
    high = np.empty_like(low)
    for rank, field in enumerate(surface.ranks):
        low_index = lo == rank
        high_index = hi == rank
        use = low_index | high_index
        if use.any():
            q = np.empty_like(low)
            q[use] = field(soc[use], soh[use])[0]
            low[low_index] = q[low_index]
            high[high_index] = q[high_index]
    v = low*(1.-frac)+high*frac
    v[:,[0,1,3]] *= (interp_temp(surface.gfac,soc,I,T)/1000.)[:,None]
    return v

def calc(sd, sc, t, I, V, T, soc, soh):
    data = np.empty((len(t), 13), dtype=np.float64)
    data[:,:6] = np.column_stack((t,I,V,T,soc,soh))
    charge = I>0.
    for select, surface in ((charge,sc),(~charge,sd)):
        if select.any():
            s,h = soc[select],soh[select]
            data[select,6:11] = ecm_values(surface,s,h,I[select],T[select])
            data[select,11] = surface.ocv(s,h)[0]
            data[select,12] = surface.hyst_field(s,h)[0][:,0]
    buffer = np.empty((len(t),21), np.float32)
    n = call(len(t),data,buffer)
    return buffer[:n]

def build_features(cell, sd, sc, cache, max_runs=None):
    with np.load(Path(cache)/f"uypydj_{cell}_Fifteen_Drive_Cycles.npz", allow_pickle=True) as z:
        arrays = {k:z[k] for k in ("files","lens","t","I","V","T","SOC","SOH","valid")}
    offs = np.r_[0,np.cumsum(arrays["lens"])]
    bad = {n for c,n in defective("drive") if c==cell}
    out=[]
    indices = range(len(arrays["lens"])) if max_runs is None else np.linspace(0,len(arrays["lens"])-1,max_runs).astype(int)
    for k in indices:
        m = re.match(r"(\d+)\|",str(arrays["files"][k]))
        if not m or (cycle:=int(m.group(1))) in bad:
            continue
        sl = slice(offs[k],offs[k+1])
        t,I,V,T,soc,soh,valid = (arrays[key][sl] for key in ("t","I","V","T","SOC","SOH","valid"))
        good = valid & np.isfinite(V) & np.isfinite(I) & np.isfinite(T) & np.isfinite(soc)
        if good.sum()<2000:
            continue
        for row in calc(sd,sc,*(x[good] for x in (t,I,V,T,soc,soh))):
            out.append({"cell":cell,"cycle":cycle,"x":row[:12],"xe":row[12:],"exc":float(row[3])})
    return out

call = ctypes.CDLL(str(Path(__file__).with_name("strict_features_native.so"))).snapshot
call.argtypes=[ctypes.c_int, np.ctypeslib.ndpointer(np.float64,flags="C_CONTIGUOUS"),
               np.ctypeslib.ndpointer(np.float32,flags="C_CONTIGUOUS")]
call.restype=ctypes.c_int
