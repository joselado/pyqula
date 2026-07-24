# AAA rational approximation (Nakatsukasa, Sete & Trefethen, SIAM J. Sci.
# Comput. 40, A1494 (2018), arXiv:1612.00337): given samples F=f(Z) of a
# function on a candidate point set Z, greedily builds a barycentric
# rational interpolant r(z) = sum_j w_j*f_j/(z-z_j) / sum_j w_j/(z-z_j)
# that reproduces f exactly at a small, adaptively-chosen subset of Z (the
# "support points") and approximates it elsewhere, typically to high
# accuracy with far fewer support points than |Z| whenever f is close to
# rational -- e.g. a retarded self-energy/Green's function, whose only
# real-axis structure (Andreev bound states, gap edges) is exactly the
# poles/near-poles this ansatz is built to represent, as opposed to
# quantics/qtci's dyadic-bisection grid, which has to resolve a pole's
# width bit-by-bit (see qtcitk/selfenergy_qtci.py's benchmark: that
# approach needs tens of thousands of solves for the same target this
# module fits with a few hundred; see keldyshtk/current.py for how the two
# compare on the actual Keldysh dI/dV workload).
#
# Each iteration adds the current-worst-fit candidate point as a new
# support point, then solves a small SVD least-squares problem (the
# "Loewner matrix") for the barycentric weights; the algorithm is
# essentially self-contained numerical linear algebra, no vendored
# dependency needed (unlike qutecipytk).
import numpy as np
from numba import jit


def aaa(F, Z, tol=1e-13, mmax=100):
    """Fit a barycentric rational interpolant to F=f(Z).

    `F`, `Z`: complex arrays of equal length (function values, sample
    points). `tol`: relative stopping tolerance (relative to max|F|) on
    the residual at points not yet chosen as support points. `mmax`: hard
    cap on the number of support points (safety net if `f` isn't well
    approximated by any modest-order rational function over this domain).

    Returns `(r, zj, fj, w, errvec)`: `r` is the callable interpolant
    (accepts a scalar or array of evaluation points), `zj`/`fj` the chosen
    support points/values, `w` the barycentric weights, `errvec` the
    residual after each iteration (for diagnostics)."""
    Z = np.asarray(Z, dtype=np.complex128)
    F = np.asarray(F, dtype=np.complex128)
    if Z.shape != F.shape or Z.ndim != 1:
        raise ValueError("Z and F must be 1D arrays of the same length")
    M = len(Z)
    Fmax = np.max(np.abs(F))
    if Fmax == 0.: Fmax = 1.

    J = list(range(M))                          # candidates not yet chosen
    zj = np.zeros(0, dtype=np.complex128)
    fj = np.zeros(0, dtype=np.complex128)
    w = np.zeros(0, dtype=np.complex128)
    R = np.mean(F) * np.ones(M, dtype=np.complex128)   # current approximant on Z
    errvec = []

    mmax = min(mmax, M)
    for _ in range(mmax):
        jj = int(np.argmax(np.abs(F[J] - R[J])))
        j = J.pop(jj)
        zj = np.append(zj, Z[j])
        fj = np.append(fj, F[j])

        with np.errstate(divide="ignore", invalid="ignore"):
            C = 1.0 / (Z[:, None] - zj[None, :])         # (M, m+1) Cauchy matrix
            A = F[:, None] * C - C * fj[None, :]          # Loewner matrix, (M, m+1)
        # only the not-yet-chosen rows are finite/meaningful (support-point
        # rows of C/A contain a division by zero in their own column).
        # full_matrices=False is essential, not just an optimization: the
        # default (True) computes the full (M,M) left singular-vector
        # matrix even though only Vh's last row is ever used below, an
        # O(M^3) cost instead of O(M*m^2) that dominates everything else
        # in this function once M is more than a few hundred.
        _, _, Vh = np.linalg.svd(A[J, :], full_matrices=False)
        w = Vh[-1, :].conj()

        with np.errstate(divide="ignore", invalid="ignore"):
            R = (C @ (w * fj)) / (C @ w)
        R[np.isin(Z, zj)] = np.nan  # placeholders at support points, unused below

        err = np.max(np.abs(F[J] - R[J])) if J else 0.0
        errvec.append(err)
        if err <= tol * Fmax:
            break

    return _BarycentricRational(zj, fj, w), zj, fj, w, errvec


@jit(nopython=True, cache=True)
def _eval_scalar_jit(e, zj, wf, w):
    """Compiled core of the scalar hot path below: calling this one energy
    at a time (as keldyshtk.current._cached_selfenergy does, tens of
    thousands of times per dc_current call, x16 matrix entries) pays
    per-call overhead on every single evaluation that dominates the
    handful-of-support-points arithmetic itself for the modest (tens, not
    thousands) support-point counts this module actually produces --
    measured ~7.5x faster compiled than as plain Python complex-scalar
    arithmetic (itself already ~3.5x faster than a numpy (1,m) broadcast,
    the first fix made here), for a one-time ~0.3s compile cost that's
    cached to disk (cache=True) across process runs."""
    num = 0j
    den = 0j
    n = zj.shape[0]
    for k in range(n):
        zjk = zj[k]
        if e == zjk:
            return wf[k] / w[k]
        c = 1.0 / (e - zjk)
        num += wf[k] * c
        den += w[k] * c
    return num / den


class _BarycentricRational:
    """Callable barycentric rational interpolant produced by `aaa`."""

    def __init__(self, zj, fj, w):
        self.zj, self.fj, self.w = zj, fj, w
        self.wf = w * fj

    def __call__(self, z):
        if np.isscalar(z) or (isinstance(z, np.ndarray) and z.ndim == 0):
            return _eval_scalar_jit(complex(z), self.zj, self.wf, self.w)
        z = np.asarray(z, dtype=np.complex128)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = 1.0 / (z[:, None] - self.zj[None, :])
            out = (C @ self.wf) / (C @ self.w)
        hits = np.nonzero(np.isin(z, self.zj))[0]
        for idx in hits:
            out[idx] = self.fj[np.nonzero(self.zj == z[idx])[0][0]]
        return out
