from concurrent.futures import ProcessPoolExecutor
import itertools, json, os, sys
from pathlib import Path
import numpy as np

from build_strict_inner import REPO, EXPERIMENT, CELLS, DIRECTIONS
A = REPO / "analysis"
B = REPO / "nested"
ASSETS = B / "assets"
OUT = EXPERIMENT / "predictions"
OUT.mkdir(exist_ok=True)

def train_pair(pair, direction):
    import torch
    import sop_trim as trim
    from train_graph import train_graph
    torch.set_num_threads(1)
    sub = DIRECTIONS[direction]
    data = EXPERIMENT / "datasets" / "__".join(pair) / sub
    c, j = pair
    result_paths = [OUT / f"{sub}_no{outer}" / f"pred_A8_{held}.npz" for outer, held in ((c,j),(j,c))]
    if all(p.exists() and p.with_name(p.name.replace('pred_', 'model_')).with_suffix('.pt').is_file() for p in result_paths):
        return
    training = trim.load_cells(str(data))
    assert set(training) == set(CELLS) - set(pair)
    tkeys = ("X","Y","NOM","I")
    tr = {k:np.concatenate([training[t][k] for t in sorted(training)]) for k in tkeys}
    # The target j uses the original direct {c,j}-excluded pool. These cached
    # rows for j have no part in fitting or standardising the residual model.
    tests = [trim.load_cells(str(EXP / f"{sub}_no{outer}"))[held] for outer,held in ((c,j),(j,c))]
    te = {k:np.concatenate([t[k] for t in tests]) for k in tkeys}
    dev = os.environ.get("SOP_TORCH_DEVICE", "cuda")
    models = [train_graph(trim.TrimLinear, tr, te, 300, .003, seed, dev, lam=.01,
                             ablate=list(range(1,12))) for seed in (0,1,2)]
    average = [np.mean([m[n] for m in models], 0) for n in (0,1,2)]
    start = 0
    for (outer,held), t, out in zip(((c,j),(j,c)),tests,result_paths):
        out.parent.mkdir(exist_ok=True)
        final = slice(start, start+len(t["I"]))
        tmp = out.with_suffix(".partial.npz")
        np.savez(tmp, pred=average[0][final], k_f=average[1][final], k_s=average[2][final],
                 **{k:t[k] for k in ("Y","NOM","I","cycle","SOC","SOH","rank","exc")})
        tmp.replace(out)
        start = final.stop
        torch.save({"seeds": [{**m[4], "model": {k:v.cpu() for k,v in m[4]["model"].items()}} for m in models],
                    "cell":held,"rung":"A8","excluded":[outer,held]},
                   out.parent / f"model_A8_{held}.pt")
    print(direction, pair, "fit rows", len(tr["I"]), "inference rows", len(te["I"]),flush=True)

EXP = REPO.parent.parent / "rfollow_20260905" / "20_calib_fix" / "innerdata"

if __name__=="__main__":
    # Disjoint jobs may share the RTX 5090. Do not displace existing work.
    tasks = list(itertools.product(itertools.combinations(CELLS,2), DIRECTIONS))
    with ProcessPoolExecutor(max_workers=int(os.environ.get('SOP_TRAIN_WORKERS', '4'))) as pool:
        results = [pool.submit(train_pair,*job) for job in tasks]
        for result in results:
            result.result()
    assert len(list(OUT.glob("*/pred_A8_*.npz")))==60
