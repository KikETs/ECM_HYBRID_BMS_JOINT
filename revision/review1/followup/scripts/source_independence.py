from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import csv, io, json, re, sys, zipfile

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT.parent/"repo/raw/RPCWBY"

def parse(stamp):
    for fmt in ("%m-%d-%Y %H:%M:%S.%f","%m-%d-%Y %H:%M:%S","%m/%d/%Y %H:%M:%S.%f","%m/%d/%Y %H:%M:%S"):
        try:return datetime.strptime(stamp,fmt)
        except ValueError:pass
    return None

def digest(number):
    dest=[]
    with zipfile.ZipFile(RAW/f"{number}_Test_{number}.zip") as z:
        for name in z.namelist():
            if name.startswith("__MACOSX") or "/Samsung30T_old/" in name or not name.lower().endswith(".csv"):
                continue
            with z.open(name) as f:
                reader=csv.reader(io.TextIOWrapper(f,encoding="utf-8-sig",errors="replace"))
                cols=next(reader)
                k=cols.index("Date_Time")
                first=last=None
                for row in reader:
                    if len(row)>k and len(row[k])>=16:
                        if first is None:first=row[k]
                        last=row[k]
            low,high=parse(first) if first else None,parse(last) if last else None
            if low and high and high>low:
                channel=int(re.search(r"Channel_(\d+)",name,re.I)[1])
                dest.append((number,name,low,high,channel,"SOC" in cols))
    return dest

if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        D=dict(zip((1,2,3),pool.map(digest,(1,2,3))))
    rows=[]
    for j,k in ((1,2),(1,3),(2,3)):
        for n,a,t0,t1,c1,s1 in D[j]:
            for m,b,u0,u1,c2,s2 in D[k]:
                start,end=max(t0,u0),min(t1,u1)
                gap=(end-start).total_seconds()
                if gap>60:
                    rows.append(dict(test_a=j,test_b=k,member_a=a,member_b=b,channel_a=c1,channel_b=c2,
                                     soc_column_a=s1,soc_column_b=s2,overlap_start=start.isoformat(),
                                     overlap_end=end.isoformat(),overlap_hours=gap/3600.))
    for pair in ((1,2),(1,3),(2,3)):
        rr=sorted([r for r in rows if (r["test_a"],r["test_b"])==pair],key=lambda r:-r["overlap_hours"])
        assert rr,pair
        print(pair,"overlap records",len(rr),"longest",rr[0],flush=True)
    with (ROOT/"external"/"independent_cell_records.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
