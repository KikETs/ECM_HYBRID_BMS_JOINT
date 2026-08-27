"""Sequence-model and adaptive-ECM baselines for SOP, on identical splits.

    python3 repro/run_sop_seq_baselines.py --data cache/trim --suffix ""
    python3 repro/run_sop_seq_baselines.py --data cache/trim_chg --suffix _chg

**Why these and not others.**  The ladder in sec 32 compares A0, a direct RLS
plug-in, a two-parameter shrinkage map, A8, A3 and an HPPC-RLS upper bound.
Every one of those is a *static* map from aggregated drive features to a
resistance multiplier.  A reviewer will ask whether a sequence model, given
the same causal window, does better - and whether a properly recursive
adaptive ECM does.  Neither was in the repository.

**What is held identical to the trim, deliberately.**

    split      leave-one-cell-out, the same six cells
    input      the same 12 exponentially-weighted features from the same 12
               preceding drive blocks (600 s each).  The trim sees one block
               at a time; the sequence models see all twelve at once, which
               is exactly the advantage being tested.
    output     the same parameterisation, k = exp(span * tanh(u)), so the
               comparison is of the model and not of the output head
    target     the same measured dV at tau = 2 s and 10 s
    loss       the same Huber(delta=0.02) on dV plus the same lam*log(k)^2
    optimiser  the same Adam(lr 3e-3, wd 1e-4) and MultiStepLR schedule
    seeds      the same 3, averaged
    scoring    written in the trim directory format and scored by the same
               eval_sop_amps.py through the same SOP inversion, the same
               voltage limits and the same lambda calibration

**Hyperparameters are declared here, not tuned.**  Hidden width 32, one
layer, 300 epochs.  No hyperparameter of the sequence models is selected on
any held-out cell.  The FFRLS forgetting factor IS selected, on grouped
inner folds of the training cells only.

**Honest limit on the FFRLS.**  A textbook adaptive ECM runs RLS over raw
current/voltage samples.  cache/trim stores the exponentially-weighted
aggregates per 600 s block, not the samples, so this runs RLS at block
granularity over the 12 blocks.  Sample-level FFRLS would need the drive
cache rebuilt with per-sample retention.  Labelled as such in the output.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ANALYSIS = os.path.join(ROOT, 'analysis')
sys.path.insert(0, ANALYSIS)

KF_SPAN, KS_SPAN = 0.470, 0.588
KF_LO, KF_HI = np.exp(-KF_SPAN), np.exp(KF_SPAN)
KS_LO, KS_HI = np.exp(-KS_SPAN), np.exp(KS_SPAN)
BLOCKS = 12


def load_cells(data_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, 'trim_*.npz'))):
        cell = os.path.basename(f)[5:-4]
        z = np.load(f, allow_pickle=True)
        out[cell] = {k: z[k] for k in
                     ('X', 'Y', 'NOM', 'I', 'm_SOH', 'm_SOC', 'm_cycle',
                      'm_rank', 'm_exc')}
    return out


def to_sequences(d):
    """Regroup the flattened rows into one 12-block sequence per label.

    sop_trim_dataset.pair() emits the 12 blocks of a label contiguously and
    in time order, and Y/NOM/I are identical across them (sec 24.3 measured
    the label spread at 3e-8 V).  Both facts are asserted here rather than
    assumed.
    """
    key = list(zip(d['m_cycle'], np.round(d['m_SOC'], 6), d['m_rank']))
    starts, seen = [], set()
    for i, k in enumerate(key):
        if k not in seen:
            seen.add(k)
            starts.append(i)
    idx = []
    for s in starts:
        block = list(range(s, s + BLOCKS))
        if block[-1] >= len(key) or any(key[j] != key[s] for j in block):
            raise ValueError('a label does not occupy 12 contiguous rows; '
                             'the sequence regrouping assumption is broken')
        idx.append(block)
    idx = np.array(idx)
    for k in ('Y', 'NOM', 'I'):
        a = np.asarray(d[k], float)[idx]
        if not np.allclose(a, a[:, :1], equal_nan=True):
            raise ValueError(f'{k} is not constant within a label')
    return idx


class SeqLSTM(nn.Module):
    def __init__(self, n_in, hid=32):
        super().__init__()
        self.rnn = nn.LSTM(n_in, hid, batch_first=True)
        self.head = nn.Linear(hid, 2)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1])


class SeqGRU(nn.Module):
    def __init__(self, n_in, hid=32):
        super().__init__()
        self.rnn = nn.GRU(n_in, hid, batch_first=True)
        self.head = nn.Linear(hid, 2)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1])


def decode(u):
    return (torch.exp(KF_SPAN * torch.tanh(u[:, 0])),
            torch.exp(KS_SPAN * torch.tanh(u[:, 1])))


def dv_hat(kf, ks, nom, i):
    return torch.stack([i * (kf * nom[:, 0] + ks * nom[:, 1]),
                        i * (kf * nom[:, 2] + ks * nom[:, 3])], 1)


def train_seq(cls, tr, te, epochs, lr, seed, dev, lam=1e-2, delta=0.02):
    torch.manual_seed(seed)
    Xtr, Xte = tr['Xs'], te['Xs']
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    xtr = torch.from_numpy(((Xtr - mu) / sd).clip(-4, 4)
                           .astype(np.float32)).to(dev)
    xte = torch.from_numpy(((Xte - mu) / sd).clip(-4, 4)
                           .astype(np.float32)).to(dev)
    ytr = torch.from_numpy(tr['Yl'].astype(np.float32)).to(dev)
    ntr = torch.from_numpy(tr['NOMl'].astype(np.float32)).to(dev)
    itr = torch.from_numpy(tr['Il'].astype(np.float32)).to(dev)

    m = cls(xtr.shape[-1]).to(dev)
    npar = sum(p.numel() for p in m.parameters())
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(
        opt, [int(epochs * f) for f in (0.5, 0.75)], 0.1)
    hub = nn.HuberLoss(delta=delta)
    n = len(xtr)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(n, device=dev)
        for k in range(0, n, 256):
            b = perm[k:k + 256]
            opt.zero_grad(set_to_none=True)
            kf, ks = decode(m(xtr[b]))
            loss = hub(dv_hat(kf, ks, ntr[b], itr[b]), ytr[b])
            loss = loss + lam * ((torch.log(kf) ** 2).mean()
                                 + (torch.log(ks) ** 2).mean())
            loss.backward()
            opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        kf, ks = decode(m(xte))
        t0 = time.perf_counter()
        for _ in range(20):
            m(xte[:1].cpu() if dev == 'cpu' else xte[:1])
        infer_us = (time.perf_counter() - t0) / 20 * 1e6
    return (kf.cpu().numpy(), ks.cpu().numpy(), npar, infer_us)


def ffrls(seq, ff):
    """Block-level RLS with forgetting factor over the 12 drive blocks.

    Observation per block: dR_fast (mOhm) = X[0], regressor 1, so this is a
    forgetting-factor recursive mean of the residual slope.  The state after
    the last block is what a BMS would carry into the pulse.
    """
    n = seq.shape[0]
    out = np.zeros((n, 2))
    for i in range(n):
        for c, (col, nom_col) in enumerate(((0, 10), (1, 11))):
            theta, P = 0.0, 1e3
            for b in range(seq.shape[1]):
                z = float(seq[i, b, col])
                K = P / (ff + P)
                theta = theta + K * (z - theta)
                P = (P - K * P) / ff
            nom = float(seq[i, -1, nom_col])
            out[i, c] = theta / nom if abs(nom) > 1e-6 else 0.0
    return out


def dump(tag, cell, kf, ks, d, idx):
    """Write the trim directory format, one k per label broadcast to its 12
    rows so `--trim-agg max` is a no-op and the evaluation is unchanged."""
    os.makedirs(tag, exist_ok=True)
    n = len(d['I'])
    KF = np.ones(n, np.float32)
    KS = np.ones(n, np.float32)
    for j, block in enumerate(idx):
        KF[block] = np.clip(kf[j], KF_LO, KF_HI)
        KS[block] = np.clip(ks[j], KS_LO, KS_HI)
    np.savez(os.path.join(tag, f'pred_A3_{cell}.npz'),
             k_f=KF, k_s=KS,
             cycle=d['m_cycle'].astype(np.int64),
             SOC=d['m_SOC'].astype(float), SOH=d['m_SOH'].astype(float),
             rank=d['m_rank'].astype(str), exc=d['m_exc'].astype(float),
             I=d['I'], NOM=d['NOM'], Y=d['Y'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='cache/trim')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--ff-grid', default='0.80,0.90,0.95,0.98,1.00')
    a = ap.parse_args()

    data = a.data if os.path.isabs(a.data) else os.path.join(ANALYSIS, a.data)
    cells = load_cells(data)
    if not cells:
        print(f'  no trim_*.npz under {data}', file=sys.stderr)
        return 1
    names = sorted(cells)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  {data}   {len(names)} cells   device {dev}')

    seqs = {}
    for c in names:
        d = cells[c]
        idx = to_sequences(d)
        seqs[c] = dict(
            idx=idx,
            Xs=np.asarray(d['X'], float)[idx],
            Yl=np.asarray(d['Y'], float)[idx][:, 0],
            NOMl=np.asarray(d['NOM'], float)[idx][:, 0],
            Il=np.asarray(d['I'], float)[idx][:, 0])
        print(f'    {c:<20} {len(d["I"]):>7,} rows -> {len(idx):>6,} labels')

    def cat(exclude):
        keys = [c for c in names if c != exclude]
        return {k: np.concatenate([seqs[c][k] for c in keys], 0)
                for k in ('Xs', 'Yl', 'NOMl', 'Il')}

    # ---- sequence models -------------------------------------------------
    summary = []
    for tag, cls in (('lstm', SeqLSTM), ('gru', SeqGRU)):
        outdir = os.path.join(ANALYSIS, f'runs_trim_{tag}{a.suffix}')
        npar_all, us_all, rmse = [], [], []
        for c in names:
            tr = cat(c)
            te = seqs[c]
            ks_, kf_ = [], []
            for s in range(a.seeds):
                kf, ksv, npar, us = train_seq(cls, tr, te, a.epochs, a.lr,
                                              s, dev)
                kf_.append(kf)
                ks_.append(ksv)
                npar_all.append(npar)
                us_all.append(us)
            kf = np.mean(kf_, 0)
            ksv = np.mean(ks_, 0)
            dump(outdir, c, kf, ksv, cells[c], te['idx'])
            p = np.stack([te['Il'] * (kf * te['NOMl'][:, 0]
                                      + ksv * te['NOMl'][:, 1]),
                          te['Il'] * (kf * te['NOMl'][:, 2]
                                      + ksv * te['NOMl'][:, 3])], 1)
            r = float(np.sqrt(np.mean((p - te['Yl']) ** 2)) * 1000)
            rmse.append(r)
            print(f'    {tag:<6} holdout {c:<20} dV RMSE {r:>7.2f} mV')
        summary.append((tag, float(np.mean(rmse)), int(np.mean(npar_all)),
                        float(np.median(us_all)), outdir))

    # ---- FFRLS -----------------------------------------------------------
    grid = [float(x) for x in a.ff_grid.split(',')]
    outdir = os.path.join(ANALYSIS, f'runs_trim_ffrls{a.suffix}')
    rmse, chosen = [], {}
    for c in names:
        inner = [o for o in names if o != c]
        best, bff = np.inf, grid[0]
        for ff in grid:
            s = 0.0
            for o in inner:                    # grouped inner folds
                t = seqs[o]
                k = ffrls(t['Xs'], ff)
                kf = np.clip(1.0 + k[:, 0], KF_LO, KF_HI)
                ksv = np.clip(1.0 + k[:, 1], KS_LO, KS_HI)
                p = np.stack([t['Il'] * (kf * t['NOMl'][:, 0]
                                         + ksv * t['NOMl'][:, 1]),
                              t['Il'] * (kf * t['NOMl'][:, 2]
                                         + ksv * t['NOMl'][:, 3])], 1)
                s += float(np.sum((p - t['Yl']) ** 2))
            if s < best:
                best, bff = s, ff
        chosen[c] = bff
        t = seqs[c]
        k = ffrls(t['Xs'], bff)
        kf = np.clip(1.0 + k[:, 0], KF_LO, KF_HI)
        ksv = np.clip(1.0 + k[:, 1], KS_LO, KS_HI)
        dump(outdir, c, kf, ksv, cells[c], t['idx'])
        p = np.stack([t['Il'] * (kf * t['NOMl'][:, 0] + ksv * t['NOMl'][:, 1]),
                      t['Il'] * (kf * t['NOMl'][:, 2] + ksv * t['NOMl'][:, 3])],
                     1)
        r = float(np.sqrt(np.mean((p - t['Yl']) ** 2)) * 1000)
        rmse.append(r)
        print(f'    ffrls  holdout {c:<20} dV RMSE {r:>7.2f} mV  ff={bff}')
    summary.append(('ffrls', float(np.mean(rmse)), 1, 0.0, outdir))

    print(f"\n  {'method':<8}{'dV RMSE mV':>12}{'params':>9}"
          f"{'infer us':>10}  run dir")
    print('  ' + '-' * 74)
    for tag, r, npar, us, outdir in summary:
        print(f'  {tag:<8}{r:>12.2f}{npar:>9}{us:>10.1f}  '
              f'{os.path.relpath(outdir, ROOT)}')
    print('\n  FFRLS is block-level (12 x 600 s), not sample-level: '
          'cache/trim stores EW aggregates, not raw samples.')
    print('  Score them in current with repro/run_evals.py + '
          'repro/run_safety_strict.py.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
