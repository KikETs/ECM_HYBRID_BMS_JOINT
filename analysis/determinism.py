"""Make training bit-reproducible, and fail loudly when it cannot be.

Why this exists.  sop_trim.py called torch.manual_seed and nothing else, so
two runs on the same data gave weights about 0.05 % apart.  That alone would
be tolerable; what made it a reproducibility defect is
sop_baseline_fill.py's fit_alpha, a search over np.linspace(0, 1, 51).  A
0.05 % shift in the predictions is enough to move the argmin one grid step,
0.32 -> 0.34, and that lands as a 2.5 % change in a published voltage RMSE.
Eight verified numbers moved on a rebuild for this reason and nothing else.

So the seed is not the whole job.  cuDNN picks kernels by benchmarking,
several CUDA reductions accumulate in nondeterministic order, and cuBLAS needs
a fixed workspace before it will promise the same GEMM twice.  All three are
set here, and the cuBLAS variable is set at IMPORT time because it is read
when the handle is created -- importing this module after the first CUDA
matmul is too late, so import it before torch touches the GPU.

    import determinism          # first, before torch is used
    determinism.enable(seed)    # once per fit

enable() raises rather than silently degrading: a run that cannot be
reproduced should say so, not quietly produce a number someone will try to
match later.
"""
from __future__ import annotations

import os

# Read by cuBLAS when its handle is created.  Setting it later has no effect,
# which is why this is at module scope rather than inside enable().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_ENABLED = None


def enable(seed: int = 0, *, strict: bool = True) -> None:
    """Seed every generator and force deterministic kernels.

    strict=False downgrades unsupported operations to a warning instead of an
    error.  It exists for exploratory scripts; nothing that writes a published
    number should use it.
    """
    global _ENABLED

    # Checked before torch is imported: a misconfigured workspace is a
    # configuration error, and it should be reported as one on any machine,
    # including CI where torch is deliberately absent.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG is not set to :4096:8.  Import "
            "analysis/determinism.py before anything touches CUDA, or export "
            "the variable in the shell.  Without it cuBLAS does not promise "
            "the same GEMM twice and the run is not reproducible.")

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(
            torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    _ENABLED = seed


def state() -> dict:
    """What was set, for a manifest or a checkpoint."""
    import torch
    return {
        "seed": _ENABLED,
        "torch": torch.__version__,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "tf32_matmul": bool(getattr(torch.backends.cuda.matmul,
                                    "allow_tf32", False)),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()),
    }
