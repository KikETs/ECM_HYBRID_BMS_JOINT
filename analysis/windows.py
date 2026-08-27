"""Sequence windows as GPU-side views instead of a materialised array.

WHY THIS REPLACES np.stack([feats[i:i+W] for i in idx])
    A 200-sample window at 1 Hz duplicates every sample 200 times. That is
    affordable for the reproduction set (978k windows = 2.35 GB) and impossible
    for the aging campaign:

        UYPYDJ, all six protocols   ~61,000,000 samples
        materialised                61e6 x 200 x 3 x 4 B  =  146 TB
        as a series                 61e6 x 3 x 4 B        =  732 MB

    So the series is stored once and a window is built at gather time from a
    start index. The gather is one advanced-index op on the card, which is what
    the training loop was already paying for when it sliced a CPU tensor.

FILE BOUNDARIES ARE NOT OPTIONAL
    Files are concatenated into one flat array, so a naive start index could span
    the join between two files and splice two different thermal states - or two
    different CELLS - into one sequence. Valid starts are enumerated per file and
    kept in an explicit index; nothing is inferred from the flat array afterwards.

TARGET CONVENTION, matched to the existing build_windows()
    X[i] = feats[i : i+W]   ->   Y[i] = volts[i+W-1]
    The label is the LAST SAMPLE OF THE WINDOW. The reference work states the
    model "estimates the battery voltage by integrating information from both
    the present measured inputs and memory states", so input and label share an
    instant. Labelling one step past the window turns nowcasting into
    one-step-ahead forecasting, which on this data costs the whole RMS of the
    one-second voltage change - 42.2 mV averaged over the four named cycles,
    against a 21.54 mV target.

SCALING - OVER WHAT ACTUALLY APPEARS, NOT OVER THE STORED SERIES
    The stored series contains samples that no window ever reaches: the first W
    voltages of every file are never a target, and the last sample is never
    inside a window. Fitting the scaler on the raw series therefore produces a
    different range than the materialised path did - measured here as a 1.7e-2
    discrepancy in the scaled targets, which is a silent recalibration of the
    reported RMSE, not a rounding difference.

    So the feature scaler is fitted over exactly the positions some window
    covers, and the target scaler over exactly the sampled targets. Coverage is
    computed with a +1/-1 difference array and a cumulative sum, which is O(N)
    and stays exact for any stride.
"""
from __future__ import annotations

import numpy as np
import torch


class WindowSet:
    """Windows over concatenated series, addressed by start index.

    group_id lets whole cells / files / protocols be held out as a unit, which
    is what the aging study needs - see docs/soh_extension_design.md section 4.
    """

    def __init__(self, feats, volts, window, starts, groups, meta=None,
                 ctx=None, ctx_len=0, aux=None):
        self.feats = feats            # (N, F) float32
        self.volts = volts            # (N,)   float32
        self.window = window
        self.starts = starts          # (M,) int64, valid window starts
        self.groups = groups          # (M,) int64, group id per window
        self.meta = meta or []        # per-group description
        # Optional MEASURED history immediately before each window, used by the
        # context encoder in docs/soh_extension_design.md section 2. It is a
        # separate channel set (V, I, T) from the model inputs (SOC, T, P)
        # because it answers a different question - how this particular cell is
        # responding right now, rather than what it is being asked to do.
        self.ctx = ctx                # (N, C) float32 or None
        self.ctx_len = ctx_len
        # Per-sample scalars the model may be conditioned on (SOH). Read at the
        # window's TARGET position, so the label and its SOH describe the same
        # instant.
        self.aux = aux                # (N,) float32 or None
        self._ar = torch.arange(window, dtype=torch.long)
        self._arc = torch.arange(ctx_len, dtype=torch.long) if ctx_len else None

    # -- construction --------------------------------------------------------
    @classmethod
    def from_series(cls, series, window=200, stride=1, feat_keys=("SOC", "T", "P"),
                    target_key="V", ctx_keys=None, ctx_len=0, aux_key=None):
        """series: iterable of dicts with the feature keys and the target key.

        A series entry may carry its own "group" value. THAT MATTERS FOR THE
        AGING STUDY: one aging protocol is one physical CELL spread over ~100
        files, and holding out a file instead of a cell would leave the model
        having already seen that cell's individual resistance trajectory - the
        very thing findings.md section 4.1 shows differs 1.58x between two cells
        aged identically. Without an explicit "group", each entry is its own.
        """
        fe, vo, st, gr, meta, cx, ax = [], [], [], [], [], [], []
        off = 0
        gmap = {}
        for gid, s in enumerate(series):
            f = np.stack([np.asarray(s[k], dtype=np.float32) for k in feat_keys], 1)
            v = np.asarray(s[target_key], dtype=np.float32)
            n = len(v) - window + 1             # last usable start + 1
            if n <= 0:
                continue
            key = s.get("group", gid)
            g = gmap.setdefault(key, len(gmap))
            fe.append(f); vo.append(v)
            if ctx_keys:
                cx.append(np.stack([np.asarray(s[k], dtype=np.float32)
                                    for k in ctx_keys], 1))
            if aux_key:
                ax.append(np.asarray(s[aux_key], dtype=np.float32))
            # Context is read from BEFORE the window, so a start closer to the
            # file's beginning than ctx_len has no history to read and is
            # dropped rather than padded - padding would teach the model that a
            # flat run-in is a normal cell response.
            idx = np.arange(ctx_len, n, stride, dtype=np.int64)
            # A per-sample validity mask kills the WHOLE window it lands in, not
            # just the sample. Ten UYPYDJ files lost their thermocouple mid-run
            # (see build_uypydj_cache.py); a window straddling the break would
            # feed the model 200 s of temperature that is partly a dead channel.
            val = s.get("valid")
            if val is not None:
                val = np.asarray(val, dtype=bool)
                # prefix[i] = number of invalid samples before i
                bad = np.concatenate([[0], np.cumsum(~val)])
                # window [i, i+W) plus its target at i+W must be clean
                clean = (bad[idx + window] - bad[idx]) == 0
                idx = idx[clean]
                if len(idx) == 0:
                    fe.pop(); vo.pop()
                    continue
            idx = idx + off
            st.append(idx); gr.append(np.full(len(idx), g, dtype=np.int64))
            meta.append({"group": key, **s.get("meta", {})})
            off += len(v)
        if not fe:
            raise ValueError("no series long enough for the requested window")
        out = cls(torch.from_numpy(np.concatenate(fe)),
                  torch.from_numpy(np.concatenate(vo)),
                  window,
                  torch.from_numpy(np.concatenate(st)),
                  torch.from_numpy(np.concatenate(gr)),
                  meta,
                  torch.from_numpy(np.concatenate(cx)) if cx else None,
                  ctx_len,
                  torch.from_numpy(np.concatenate(ax)) if ax else None)
        out.group_names = [k for k, _ in sorted(gmap.items(), key=lambda kv: kv[1])]
        return out

    # -- placement -----------------------------------------------------------
    def to(self, device):
        self.feats = self.feats.to(device)
        self.volts = self.volts.to(device)
        self.starts = self.starts.to(device)
        self.groups = self.groups.to(device)
        self._ar = self._ar.to(device)
        if self.ctx is not None:
            self.ctx = self.ctx.to(device)
            self._arc = self._arc.to(device)
        if self.aux is not None:
            self.aux = self.aux.to(device)
        return self

    @property
    def device(self):
        return self.feats.device

    def nbytes(self):
        return (self.feats.numel() + self.volts.numel()) * 4

    # -- access --------------------------------------------------------------
    def __len__(self):
        return len(self.starts)

    def batch(self, sel, with_ctx=False, with_aux=False):
        """sel: index tensor into starts. Returns (B, W, F) and (B,).

        with_ctx additionally returns (B, ctx_len, C) read from the samples
        immediately BEFORE each window.
        """
        s = self.starts[sel]
        idx = s.unsqueeze(1) + self._ar          # (B, W)
        x, y = self.feats[idx], self.volts[s + self.window - 1]
        out = [x, y]
        if with_ctx:
            if self.ctx is None:
                raise ValueError("this WindowSet carries no context channels")
            out.append(self.ctx[(s - self.ctx_len).unsqueeze(1) + self._arc])
        if with_aux:
            if self.aux is None:
                raise ValueError("this WindowSet carries no aux channel")
            out.append(self.aux[s + self.window - 1])
        return tuple(out)

    def subset(self, mask):
        """A view restricted to selected windows; the series is NOT copied."""
        out = WindowSet(self.feats, self.volts, self.window,
                        self.starts[mask], self.groups[mask], self.meta,
                        self.ctx, self.ctx_len, self.aux)
        out._ar = self._ar
        out._arc = self._arc
        return out

    def by_group(self, keep):
        """Windows whose group id is in `keep` (held-out splits)."""
        keep_t = torch.as_tensor(sorted(keep), device=self.groups.device)
        return self.subset(torch.isin(self.groups, keep_t))

    # -- scaling -------------------------------------------------------------
    def covered_mask(self):
        """Boolean over series positions: does any window include this sample?"""
        d = torch.zeros(len(self.feats) + 1, dtype=torch.int32,
                        device=self.feats.device)
        d.index_add_(0, self.starts, torch.ones_like(self.starts,
                                                     dtype=torch.int32))
        d.index_add_(0, self.starts + self.window,
                     -torch.ones_like(self.starts, dtype=torch.int32))
        return d.cumsum(0)[:-1] > 0

    def fit_scaler(self):
        cov = self.covered_mask()
        f = self.feats[cov]
        lo, hi = f.min(0).values, f.max(0).values
        rng = torch.where(hi - lo == 0, torch.ones_like(hi), hi - lo)
        tv = self.volts[self.starts + self.window - 1]  # exactly the targets
        vlo, vhi = tv.min(), tv.max()
        vrng = vhi - vlo if vhi > vlo else torch.ones_like(vhi)
        return {"lo": lo, "rng": rng, "vlo": vlo, "vrng": vrng}

    def apply_scaler(self, sc):
        """Scale the SERIES once. Windows are views, so they come out scaled."""
        self.feats = (self.feats - sc["lo"]) / sc["rng"]
        self.volts = (self.volts - sc["vlo"]) / sc["vrng"]
        return self


def load_uypydj_cells(cache_dir, cells=None, part="Fifteen_Drive_Cycles",
                      window=200, stride=30, ctx_keys=None, ctx_len=0,
                      with_soh=False, feat_keys=("SOC", "T", "P")):
    """Build a WindowSet over cached UYPYDJ cells, grouped BY CELL.

    Each cached archive is one cell holding ~100 concatenated runs, so the file
    table and per-file lengths are used to cut the flat arrays back into runs -
    a window must not span two runs even inside one cell, since consecutive runs
    are days apart in the aging test.
    """
    import glob
    import os

    paths = sorted(glob.glob(os.path.join(cache_dir, f"uypydj_*_{part}.npz")))
    series = []
    for p in paths:
        cell = os.path.basename(p)[len("uypydj_"):-len(f"_{part}.npz")]
        if cells is not None and cell not in cells:
            continue
        # EACH z[key] REREADS THE WHOLE ARRAY. NpzFile does not cache, so
        # indexing it inside the per-run loop re-decoded ~280 MB per channel per
        # run - about 100x8 times per cell - and the process was killed. Pull
        # each channel out once, then slice the in-memory array.
        z = np.load(p)
        cols = {k: z[k] for k in ("SOC", "T", "P", "V", "I", "SOH", "valid")}
        lens, names = z["lens"], z["files"]
        z.close()
        off = 0
        for k, n in enumerate(lens):
            sl = slice(off, off + n)
            series.append({
                "SOC": cols["SOC"][sl], "T": cols["T"][sl], "P": cols["P"][sl],
                "V": cols["V"][sl], "I": cols["I"][sl],
                "SOH": cols["SOH"][sl], "valid": cols["valid"][sl],
                "group": cell,
                "meta": {"cell": cell, "run": str(names[k])},
            })
            off += n
    if not series:
        raise FileNotFoundError(f"no cached cells in {cache_dir}")
    return WindowSet.from_series(series, window=window, stride=stride,
                                 feat_keys=feat_keys,
                                 ctx_keys=ctx_keys, ctx_len=ctx_len,
                                 aux_key="SOH" if with_soh else None)
