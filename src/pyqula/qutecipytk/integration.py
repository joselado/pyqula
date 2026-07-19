"""Port of integration.jl: Gauss-Kronrod quadrature built on top of TCI2 +
the linear-cost sum(tt) trick.

Folds the per-node Gauss-Kronrod weight into the function TCI approximates,
so that summing the resulting tensor train over all node-index combinations
reproduces the tensorized high-dimensional quadrature rule in O(L*D^2*nodes)
instead of the naive O(nodes^L) direct summation -- as long as the integrand
is well approximated by a low-rank TT to begin with.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from pyqula.qutecipytk.gausskronrod import kronrod
from pyqula.qutecipytk.tci2 import crossinterpolate2


def integrate(
    dtype, f: Callable, a: Sequence[float], b: Sequence[float], GKorder: int = 15, **kwargs
):
    if GKorder % 2 == 0:
        raise ValueError("Gauss-Kronrod order must be odd, e.g. 15 or 61.")
    if len(a) != len(b):
        raise ValueError(
            f"Integral bounds must have the same dimensionality, but got {len(a)} lower "
            f"bounds and {len(b)} upper bounds."
        )

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    nodes1d, weights1d, _ = kronrod(GKorder // 2, -1, 1)
    nnodes = len(nodes1d)

    nodes = (b[:, None] - a[:, None]) * (nodes1d[None, :] + 1) / 2 + a[:, None]
    weights = (b[:, None] - a[:, None]) * weights1d[None, :] / 2
    normalization = GKorder ** len(a)

    localdims = [nnodes] * len(a)

    def F(indices):
        x = [nodes[n, i] for n, i in enumerate(indices)]
        w = 1.0
        for n, i in enumerate(indices):
            w *= weights[n, i]
        return w * f(x) * normalization

    tci2, ranks, errors = crossinterpolate2(dtype, F, localdims, nsearchglobalpivot=10, **kwargs)

    return tci2.sum() / normalization
