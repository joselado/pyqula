"""Port of QuadGK.jl's src/gausskronrod.jl -- unit-weight/hollow-tridiagonal
subset only (the "hollow symmetric tridiagonal" branch for a symmetric
interval), since that's all integration.py needs (``kronrod(n, -1, 1)``).

Not ported: the general arbitrary-Jacobi-matrix API, BigFloat/arbitrary
precision, the ``@generated``-function result-caching trick (a plain
``functools.lru_cache`` covers that here).

Algorithm: Dirk P. Laurie, "Calculation of Gauss-Kronrod quadrature rules,"
Mathematics of Computation, vol. 66, no. 219, pp. 1133-1145 (1997), as
implemented by QuadGK.jl (Steven G. Johnson et al., MIT license).

Internal helper functions here deliberately keep Julia's 1-based array
convention (index 0 unused/padded) -- this is one of the densest pieces of
index arithmetic in the whole port, and re-deriving every offset for
0-based arrays would be much more error-prone than translating literally
and converting to plain 0-based arrays only at the public boundary
(``kronrod``).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import scipy.linalg


def _eigpolyrat_hollow(b: np.ndarray, z: float, m: int) -> float:
    """p(z)/p'(z) for p(z) = det(zI - H), H hollow sym-tridiagonal with
    off-diagonal b[1..m-1] (1-indexed-style, b[0] unused)."""
    d1 = z
    d1deriv = 1.0
    d2 = 1.0
    d2deriv = 0.0
    for i in range(2, m + 1):
        b2 = b[i - 1] ** 2
        d = z * d1 - b2 * d2
        dderiv = d1 + z * d1deriv - b2 * d2deriv
        if dderiv == 0:
            d2, d1 = d1, d
            d2deriv, d1deriv = d1deriv, dderiv
        else:
            inv_dderiv = 1.0 / dderiv
            d2, d1 = d1 * inv_dderiv, d * inv_dderiv
            d2deriv, d1deriv = d1deriv * inv_dderiv, 1.0
    return d1 / d1deriv


def _eignewt_hollow(b: np.ndarray, n: int) -> np.ndarray:
    """The n smallest eigenvalues of the hollow sym-tridiagonal matrix whose
    off-diagonal is b[1..ev_length] (1-indexed-style array; matrix_size =
    ev_length + 1)."""
    ev_length = len(b) - 1
    matrix_size = ev_length + 1
    d = np.zeros(matrix_size)
    e = b[1:ev_length + 1]
    lam0, _ = scipy.linalg.eigh_tridiagonal(d, e, select="i", select_range=(0, n - 1))
    lam = np.array(lam0, dtype=np.float64)
    # Laurie's algorithm can hit spurious overflow in eigpolyrat's rational-function
    # evaluation for larger n; already handled below via the isfinite check (matches
    # QuadGK.jl's own tolerance of this -- see module docstring), so just silence it.
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(n):
            for _ in range(1000):
                dl = _eigpolyrat_hollow(b, lam[i], matrix_size)
                if np.isfinite(dl):
                    lam[i] -= dl
                    if abs(dl) <= 10 * np.spacing(abs(lam[i])):
                        break
                else:
                    break
    return lam


def _eigvec1_hollow(b: np.ndarray, lam: float, m: int) -> np.ndarray:
    """First component of the normalized eigenvector for eigenvalue lam of
    the m x m hollow sym-tridiagonal matrix with off-diagonal b[1..m-1]."""
    v = np.zeros(m + 1)  # 1-indexed-style, v[0] unused
    v[1] = 1.0
    if m > 1:
        s = v[1] ** 2
        v[2] = lam * v[1] / b[1]
        s += v[2] ** 2
        for i in range(3, m + 1):
            v[i] = (lam * v[i - 1] - b[i - 2] * v[i - 2]) / b[i - 1]
            s += v[i] ** 2
        v[1:m + 1] /= np.sqrt(s)
    return v


def _normalize2(s: np.ndarray, t: np.ndarray, maxabs: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Scales the first n entries of s and t by 1/maxabs in place. Returns
    (s, t) unchanged in order -- the caller swaps roles via ``t, s =
    _normalize2(s, t, ...)``, matching Julia's identical call-site idiom."""
    if maxabs != 0:
        scale = 1.0 / maxabs
        s[1:n + 1] *= scale
        t[1:n + 1] *= scale
    return s, t


def _kronrodjacobi_hollow(b_in: np.ndarray, n: int) -> np.ndarray:
    """Laurie's O(n^2) recursion, extending the Jacobi matrix (off-diagonal
    b_in, already sized/populated per _kronrod_b_hollow) to the (2n+1)-point
    Kronrod-Jacobi matrix's off-diagonal. b_in/return are 1-indexed-style
    (length 2n+1, index 0 unused)."""
    b = b_in.copy()
    s = np.zeros(n // 2 + 3)
    t = np.zeros(len(s))
    t[2] = b[n + 1]

    for m in range(0, n - 1):
        maxabs = 0.0
        u = 0.0
        for k in range((m + 1) // 2, -1, -1):
            u += b[k + n + 1] * s[k + 1] - (b[m - k] * s[k + 2] if m > k else 0.0)
            s[k + 2] = u
            maxabs = max(maxabs, abs(u))
        t, s = _normalize2(s, t, maxabs, (m + 1) // 2 + 2)

    for j in range(n // 2, -1, -1):
        s[j + 2] = s[j + 1]

    for m in range(n - 1, 2 * n - 2):
        maxabs = 0.0
        u = 0.0
        for k in range(m + 1 - n, (m - 1) // 2 + 1):
            j = n - (m - k) - 1
            u -= b[k + n + 1] * s[j + 2] - b[m - k] * s[j + 3]
            s[j + 2] = u
            maxabs = max(maxabs, abs(u))
        t, s = _normalize2(s, t, maxabs, len(s) - 1)
        k = (m + 1) // 2
        j = n - (m - k + 2)
        if 2 * k != m:
            b[k + n + 1] = t[j + 2] / t[j + 3]

    if np.any(b[1:] < 0):
        raise ValueError("real Gauss-Kronrod rule does not exist for this Jacobi matrix")
    b[1:] = np.sqrt(b[1:])
    return b


def _kronrod_half(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core computation: n+1 Kronrod points x<=0 (and their weights w),
    plus n embedded-Gauss weights wg, for the unit weight function on
    [-1, 1]. Matches QuadGK.jl's ``kronrod(Float64, n)``."""
    if n < 1:
        raise ValueError("Kronrod rules require positive order")

    b = np.zeros(2 * n + 1)  # 1-indexed-style
    for j in range(1, (3 * n + 1) // 2 + 1):
        b[j] = j ** 2 / (4 * j ** 2 - 1)

    kj = _kronrodjacobi_hollow(b, n)  # length 2n+1, 1-indexed-style
    x = _eignewt_hollow(kj, n + 1)  # n+1 smallest eigenvalues (the x <= 0 half)

    m_full = 2 * n + 1
    w = np.array([2.0 * _eigvec1_hollow(kj, lam, m_full)[1] ** 2 for lam in x])

    # Embedded n-point Gauss rule: reuse b's first n-1 slots for the (smaller) Gauss recurrence.
    bg = b.copy()
    for j in range(1, n):
        bg[j] = j / np.sqrt(4 * j ** 2 - 1)
    wg = np.array([2.0 * _eigvec1_hollow(bg, x[i - 1], n)[1] ** 2 for i in range(2, n + 2, 2)])

    return x, w, wg


@lru_cache(maxsize=None)
def _kronrod_half_cached(n: int) -> tuple[tuple, tuple, tuple]:
    x, w, wg = _kronrod_half(n)
    return tuple(x), tuple(w), tuple(wg)


def kronrod(n: int, a: float = -1.0, b: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2n+1 Kronrod points/weights (x, w) for integrating on (a, b), plus
    the embedded n-point Gauss weights wg (for the points x[1::2]), for the
    unit weight function.  ``sum(w * f(x))`` approximates the integral."""
    xh, wh, wgh = _kronrod_half_cached(n)
    x = np.array(xh)
    w = np.array(wh)
    wg = np.array(wgh)

    x = np.concatenate([x, -x[:-1][::-1]])
    w = np.concatenate([w, w[:-1][::-1]])
    wg = np.concatenate([wg, wg[: len(wg) - (n % 2)][::-1]])

    xscale = (b - a) / 2
    x = (x + 1) * xscale + a
    w = w * xscale
    wg = wg * xscale
    return x, w, wg
