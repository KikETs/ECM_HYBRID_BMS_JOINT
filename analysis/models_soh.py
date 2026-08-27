"""The four conditioning variants compared in docs/soh_extension_design.md.

    M0  no conditioning          the reference work's model, unchanged
    M1  SOH scalar               the obvious extension, expected to fail
    M2  learned context vector    z from recent measured (V, I, T)
    M3  z + SOH                   are they complementary

WHY M1 IS EXPECTED TO FAIL, AND WHY IT IS BUILT ANYWAY
    Two cells at SOH 0.75 differ by 1.58x in 10 s pulse resistance and 1.41x in
    the SOP reference itself (findings.md section 4.1, design section 6.1). A
    model handed only SOH must learn the average of cells that genuinely differ,
    so it is wrong for each of them. It is still built: the failure is evidence,
    and above SOH ~0.92 - where all six cells agree within 2-4 % - it should work
    fine. Establishing WHERE it stops working is part of the result.

THE CONSTRAINT THAT SHAPES THE ARCHITECTURE
    SOP is found by binary search over candidate powers. The reference work can
    do that with ONE forward pass per candidate because its inputs (SOC, T, P)
    are all computable in advance for a hypothetical constant-power pulse.

    Feeding past VOLTAGE into the voltage model would break that: the pulse
    portion's voltage is unknown, so the model would have to be rolled out
    autoregressively, accumulating error and opening the door to the
    teacher-forcing leak this project has already been bitten by once.

    The context encoder keeps the property. z is computed ONCE from measured
    samples before the pulse and held fixed across the whole binary search, so
    the search still costs one forward pass per candidate.

WHERE THE CONDITIONING ENTERS
    Appended to the LSTM input at every timestep, not to the head. What ages is
    the cell's DYNAMIC response - how far voltage sags for a given power, and how
    fast - so the conditioning has to reach the recurrence. Injecting at the head
    would only let it shift the final readout.
"""
from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN = 256
LSTM_LAYERS = 2
# Matches the corrected reproduction head - see lstm_voltage.py. The paper's
# Table 1 says (256, 16) but its Table 4 parameter counts say (512, 256), and
# (512, 256) reproduces all three of its counts exactly. Keeping the extension
# on a different head than the reproduction would make the two incomparable.
FC_SIZES = (512, 256)
CTX_HIDDEN = 64
Z_DIM = 8


class ContextEncoder(nn.Module):
    """Recent measured (V, I, T) -> a small cell-state vector.

    A GRU rather than an LSTM, and one layer: the job is to summarise how this
    cell responds, not to model a long dependency, and a small encoder is
    harder to overfit to the five training cells.
    """

    def __init__(self, n_in=3, hidden=CTX_HIDDEN, z_dim=Z_DIM):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, 1, batch_first=True)
        self.proj = nn.Linear(hidden, z_dim)
        self.z_dim = z_dim

    def forward(self, c):                        # c: (B, L, 3)
        _, h = self.gru(c)
        return self.proj(h[-1])                  # (B, z_dim)


class ConditionedVoltageLSTM(nn.Module):
    """The reference voltage model, optionally conditioned.

    variant:
        "M0" plain, "M1" SOH scalar, "M2" learned z, "M3" both.
    Every variant keeps the same LSTM width/depth and head so the comparison is
    about the conditioning, not about capacity.
    """

    def __init__(self, variant="M0", n_in=3, hidden=HIDDEN, layers=LSTM_LAYERS,
                 fc=None, ctx_in=3, z_dim=Z_DIM):
        super().__init__()
        if variant not in ("M0", "M1", "M2", "M3"):
            raise ValueError(f"unknown variant {variant}")
        self.variant = variant
        self.use_z = variant in ("M2", "M3")
        self.use_soh = variant in ("M1", "M3")

        # THE HEAD SCALES WITH THE RECURRENT WIDTH. The paper's head is
        # (2*hidden, hidden) = (512, 256) at hidden 256; pinning those absolute
        # numbers while shrinking the LSTM would leave a 64-unit model carrying a
        # 165k-parameter head - larger than its own recurrent part, and the
        # opposite of what shrinking is for.
        if fc is None:
            fc = (2 * hidden, hidden)
        self.encoder = ContextEncoder(ctx_in, z_dim=z_dim) if self.use_z else None
        extra = (z_dim if self.use_z else 0) + (1 if self.use_soh else 0)
        self.lstm = nn.LSTM(n_in + extra, hidden, layers, batch_first=True)
        seq, prev = [], hidden
        for h in fc:
            seq += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        seq += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*seq)

    def encode(self, ctx=None, soh=None):
        """The conditioning vector, computed ONCE per binary search."""
        parts = []
        if self.use_z:
            if ctx is None:
                raise ValueError(f"{self.variant} needs a context window")
            parts.append(self.encoder(ctx))
        if self.use_soh:
            if soh is None:
                raise ValueError(f"{self.variant} needs SOH")
            parts.append(soh.reshape(-1, 1))
        return torch.cat(parts, 1) if parts else None

    def forward(self, x, ctx=None, soh=None, cond=None):
        """x: (B, W, n_in). Pass a precomputed `cond` to reuse it across a
        binary search instead of re-encoding for every candidate power."""
        if cond is None:
            cond = self.encode(ctx, soh)
        if cond is not None:
            cond = cond.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, cond], dim=2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build(variant, **kw):
    return ConditionedVoltageLSTM(variant=variant, **kw)


if __name__ == "__main__":
    B, W, L = 7, 200, 100
    x = torch.randn(B, W, 3)
    c = torch.randn(B, L, 3)
    s = torch.rand(B)
    print(f"{'변형':<5} {'파라미터':>10} {'조건 차원':>9}  출력")
    for v in ("M0", "M1", "M2", "M3"):
        m = build(v)
        n = sum(p.numel() for p in m.parameters())
        cond = m.encode(c, s)
        y = m(x, ctx=c, soh=s)
        # the same cond reused, as the binary search would
        y2 = m(x, cond=cond)
        same = torch.allclose(y, y2, atol=1e-6)
        print(f"{v:<5} {n:>10,} {0 if cond is None else cond.shape[1]:>9}  "
              f"{tuple(y.shape)}  cond 재사용 일치: {same}")
