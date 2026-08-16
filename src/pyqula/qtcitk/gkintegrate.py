# Shared helpers for "qtci" (Gauss-Kronrod quadrature folded into a tensor
# cross interpolation, via the vendored qutecipy port at qutecipytk/) BZ
# integration, used by both topology.chern_qtci and this package's own
# densitymatrix_qtci.get_dm_qtci -- factored out so the two features can't
# silently drift apart on the nk-to-quadrature-order mapping or on how a
# zero-valued integrand at the default pivot is handled.
import numpy as np


def gkorder_from_nk(nk):
    """Map a requested mesh-like resolution nk to a Gauss-Kronrod order.
    Because Gauss-Kronrod quadrature converges spectrally (each extra
    order roughly doubles the number of accurate digits, much like each
    extra quantics bit doubles a grid's resolution), matching the accuracy
    of an nk-point mesh only takes an order growing logarithmically with
    nk, not linearly: bits=ceil(log2(nk)), GKorder=4*bits+1."""
    bits = max(1,int(np.ceil(np.log2(max(nk,2)))))
    return 4*bits+1


def _pivot_candidates(nnodes):
    """Order in which (i,j) Gauss-Kronrod node pairs are tried as the
    seed pivot: the diagonal first, then the off-diagonal pairs by
    increasing cyclic shift j-i. Every one of the nnodes**2 pairs appears
    exactly once, so exhausting this generator means f was evaluated at
    the whole quadrature node grid."""
    for i in range(nnodes): yield (i,i) # diagonal first
    for shift in range(1,nnodes): # then each shifted diagonal in turn
        for i in range(nnodes): yield (i,(i+shift)%nnodes)


def integrate_robust(dtype,f,GKorder,tolerance,**kwargs):
    """Integrate f (a function of k=[kx,ky] in [0,1]^2) over the BZ with
    qutecipy. TensorCI2 seeds its rank estimate from a single sample point
    (the first Gauss-Kronrod node along each axis, by default) and refuses
    to start if that one point is exactly zero ("maxsamplevalue is
    zero!"), which happens whenever the integrand is exactly zero there --
    e.g. a symmetry-protected zero (a spin-off-diagonal density-matrix
    element in a spin-conserving Hamiltonian, or Berry curvature at a
    high-symmetry point). Instead of guessing a nearby point is
    representative of the nearest grid node's value (it need not be),
    evaluate f directly at actual Gauss-Kronrod node combinations until
    one comes back nonzero, and seed crossinterpolate2 with that exact,
    already-verified-nonzero index.

    The search order is _pivot_candidates: the diagonal of the node grid
    first (which is what an integrand that is nonzero somewhere almost
    always hits immediately), then the rest of the grid. Scanning past
    the diagonal matters -- a symmetry can force f to vanish on the whole
    line kx=ky while leaving the integral nonzero: a mirror exchanging kx
    and ky makes any f obeying f(kx,ky)=-f(ky,kx) vanish there, and a
    product of two such factors (f ~ (kx-ky)**2) vanishes there while
    integrating to something finite. A diagonal-only scan reported 0 for
    exactly that case.

    If f is zero at *every* node of the grid then the tensorized
    Gauss-Kronrod sum is a weighted sum of zeros, so returning 0 is not a
    guess about f between the nodes -- it is the exact value of the
    quadrature rule being requested, obtained without ever building a
    tensor train. That full scan only runs for a genuinely
    zero-everywhere integrand (a true symmetry-protected zero); anything
    nonzero exits at its first nonzero node."""
    from ..qutecipytk import integrate
    from ..qutecipytk.gausskronrod import kronrod
    nodes1d,_,_ = kronrod(GKorder//2,-1,1)
    knode = (nodes1d+1)/2 # map nodes from [-1,1] to the [0,1] domain
    for (i,j) in _pivot_candidates(len(nodes1d)):
        if f([knode[i],knode[j]]) != 0:
            return integrate(dtype,f,[0.,0.],[1.,1.],GKorder=GKorder,
                    tolerance=tolerance,initialpivots=[(i,j)],**kwargs)
    return dtype(0.0)
