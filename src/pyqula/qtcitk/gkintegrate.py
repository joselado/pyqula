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
    evaluate f directly at actual Gauss-Kronrod node combinations (the
    diagonal of the node grid) until one comes back nonzero, and seed
    crossinterpolate2 with that exact, already-verified-nonzero index. If
    every diagonal node is exactly zero, f is almost certainly identically
    zero over the whole BZ (a true symmetry-protected zero), and the
    integral is 0 without ever building a tensor train."""
    from ..qutecipytk import integrate
    from ..qutecipytk.gausskronrod import kronrod
    nodes1d,_,_ = kronrod(GKorder//2,-1,1)
    for i in range(len(nodes1d)):
        k = [(nodes1d[i]+1)/2,(nodes1d[i]+1)/2] # map node -> [0,1] domain
        if f(k) != 0:
            return integrate(dtype,f,[0.,0.],[1.,1.],GKorder=GKorder,
                    tolerance=tolerance,initialpivots=[(i,i)],**kwargs)
    return dtype(0.0)
