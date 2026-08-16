# Entanglement entropy and single-particle entanglement spectrum from an
# already-restricted one-particle correlation matrix C_A. See
# pyqula/entanglement.py for the formalism and the references; the two
# formulas implemented here are Peschel & Eisler, J. Phys. A 42, 504003
# (2009) [arXiv:0906.1663], Eq. (16), h = ln[(1-C)/C], and the standard
# free-fermion entropy that follows from it.
import numpy as np
from .. import algebra

# default cutoff below which a correlation eigenvalue is treated as
# numerically 0 or 1 (see occupations_from_correlation)
tol = 1e-14


def occupations_from_correlation(CA,tol=tol):
    """Eigenvalues zeta_n of the restricted correlation matrix C_A,
    clipped into [tol,1-tol].

    C_A is Hermitian and (in exact arithmetic) positive semidefinite with
    eigenvalues in [0,1], but modes deep inside or far outside the region
    give zeta that are 0 or 1 to machine precision, and roundoff pushes
    them slightly out of the interval. Clipping keeps ln(zeta) and
    ln(1-zeta) finite (so the entanglement spectrum is a plain float array
    instead of a mix of finite values and +-inf) at the price of capping
    |xi| at ln(1/tol-1) ~ 32 for the default tol -- such a level means
    "numerically empty/full", not a physically meaningful value."""
    CA = np.asarray(algebra.todense(CA))
    if CA.shape[0]==0: return np.zeros(0) # empty region, no levels at all
    zeta = algebra.eigvalsh(CA) # Hermitian, real spectrum
    zeta = np.sort(np.real(zeta)) # ascending, drop roundoff imaginary parts
    return np.clip(zeta,tol,1.-tol)


def entropy_from_occupations(zeta,has_eh=False,tol=tol):
    """von Neumann entanglement entropy

        S = -sum_n [ zeta_n ln zeta_n + (1-zeta_n) ln(1-zeta_n) ]

    from the correlation-matrix eigenvalues.

    Levels with zeta outside (tol,1-tol) are skipped rather than clipped:
    their contribution is bounded by tol*ln(1/tol) (~3e-13 for the default
    tol) but evaluating them anyway would leave a fully empty or fully
    filled region at S ~ 1e-12 instead of exactly 0.

    has_eh=True halves the result, for a Nambu/BdG correlation matrix:
    there the sum runs over a doubled (particle+hole) basis whose
    eigenvalues come in (zeta,1-zeta) pairs, so every physical mode is
    counted twice (Peschel & Eisler, Eq. (18) and the Majorana form below
    it)."""
    zeta = np.real(np.asarray(zeta))
    keep = (zeta>tol) & (zeta<1.-tol) # the rest contribute ~0, see above
    z = zeta[keep]
    S = -np.sum(z*np.log(z) + (1.-z)*np.log(1.-z))
    if has_eh: S = S/2. # undo the Nambu doubling
    return S


def spectrum_from_occupations(zeta,tol=tol):
    """Single-particle entanglement Hamiltonian eigenvalues

        xi_n = ln[ (1-zeta_n)/zeta_n ]

    i.e. the eigenvalues of h in rho_A ~ exp(-sum_n xi_n f_n^dag f_n)
    (Peschel & Eisler, Eq. (16)). Returned sorted in ascending order, so
    the "occupied" entanglement levels (zeta>1/2) come first. Values are
    bounded by +-ln(1/tol-1), see occupations_from_correlation."""
    zeta = np.clip(np.real(np.asarray(zeta)),tol,1.-tol)
    return np.sort(np.log((1.-zeta)/zeta))
