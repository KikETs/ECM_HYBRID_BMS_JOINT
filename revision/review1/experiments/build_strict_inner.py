import csv
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parents[1]
REPO = RUN / "repo"
EXPERIMENT = RUN / "strict_inner"
sys.path.insert(0, str(REPO / "analysis"))
from ecm_pool import surfaces
import sop_trim_dataset as ds

CELLS = ("BOOST", "BOOST_NEGPULSE", "BOOST_NEGPULSE_1S", "BOOST_REST", "CC", "CC_CELL2")
DIRECTIONS = {"discharge": "trim", "charge": "trim_chg"}

def make_pair_source(pair):
    dest = EXPERIMENT / "pools" / "__".join(pair)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("uypydj_ecm.csv", "uypydj_ocv.csv"):
        with (REPO / "analysis" / name).open(newline="") as input_f, (dest / name).open("w", newline="") as output_f:
            reader = csv.DictReader(input_f)
            writer = csv.DictWriter(output_f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(row for row in reader if row["cell"] not in pair)
    return dest

def build_task(job):
    pair, cell = job
    dest = EXPERIMENT / "pools" / "__".join(pair)
    out = EXPERIMENT / "datasets" / "__".join(pair)
    files = {direction: out / sub / f"trim_{cell}.npz" for direction, sub in DIRECTIONS.items()}
    receipt = out / f"{cell}.json"
    if receipt.exists() and all(path.is_file() for path in files.values()):
        return json.loads(receipt.read_text())
    sd, sc = surfaces(cell, outdir=str(dest / "pool"), ecm_csv=str(dest / "uypydj_ecm.csv"),
                      ocv_csv=str(dest / "uypydj_ocv.csv"))
    from strict_features import build_features
    features = build_features(cell, sd, sc, ds.CACHE_T)
    row_counts = {}
    for direction, path in files.items():
        labels, drops = ds.build_labels(cell, sc if direction == "charge" else sd, direction=direction)
        rows = ds.pair(labels, features)
        if not rows:
            raise RuntimeError(f"No training rows for {pair}, {cell}, {direction}: {drops}")
        X, Y, NOM, I, meta, XE = ds.to_arrays(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial.npz")
        np.savez(tmp, X=X, Y=Y, NOM=NOM, I=I, XE=XE,
                 names_extra=np.array(ds.NAMES_EXTRA), **{f"m_{key}": values for key, values in meta.items()})
        tmp.replace(path)
        row_counts[direction] = len(rows)
    result = {"excluded_outer_and_inner": pair, "training_cell": cell,
              "surface_cells": sorted(set(CELLS) - set(pair) - {cell}),
              "rows": row_counts, "feature_blocks": len(features)}
    receipt.write_text(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)
    return result

def main():
    pairs = list(itertools.combinations(CELLS, 2))
    for pair in pairs:
        make_pair_source(pair)
    jobs = [(pair, cell) for pair in pairs for cell in CELLS if cell not in pair]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("SOP_DATA_WORKERS", "8"))) as workers:
        results = list(workers.map(build_task, jobs))
    assert len(results) == 60
    (EXPERIMENT / "datasets_manifest.json").write_text(json.dumps(results, indent=2) + "\n")

if __name__ == "__main__":
    main()
