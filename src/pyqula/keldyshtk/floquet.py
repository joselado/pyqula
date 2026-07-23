import numpy as np
from ..algebra import todense


# Floquet-space assembly for the Floquet-Keldysh Josephson/MAR current
# (San-Jose, Cayao, Prada, Aguado, New J. Phys. 15, 075019 (2013),
# arXiv:1301.4408, Appendix A).
#
# A bias V applied across a junction can be gauged away from the (static)
# leads and concentrated entirely into a single "weak link" hopping matrix
# v0, which becomes v0(t) = e^{-i*V*t*tauz} v0(0), with tauz the Nambu
# grading (+1 electron, -1 hole) acting on the *target* site of the bond.
# Expanding in Floquet sidebands turns this into a static problem in an
# enlarged (site x sideband) space.


def floquet_hamiltonian(hlist, link, omega, nmax, proje_target, projh_target):
    """Build the Floquet-space Hamiltonian for a 1D block-tridiagonal chain
    `hlist` (list of lists of square blocks, as returned by
    transporttk.smatrix.enlarge_hlist(ht).central_intra), where the single
    bond `link=(i0,i0+1)` carries the bias-induced AC phase.

    `proje_target`/`projh_target` are the electron/hole projectors matching
    the dimension of block `i0+1` (the bond's target block).

    Returns the dense Floquet Hamiltonian matrix, of size
    (len(hlist)*(2*nmax+1)*dim)^2, WITHOUT any self-energy or complex-energy
    term added (the caller adds those, since they depend on the quasienergy
    and on which lead is attached where).
    """
    nb = len(hlist)
    dim = todense(hlist[0][0]).shape[0]
    ns = 2*nmax+1
    ntot = nb*ns*dim
    H = np.zeros((ntot, ntot), dtype=np.complex128)

    def blk(ib, isb):
        off = (ib*ns+isb)*dim
        return slice(off, off+dim)

    i0, i1 = link
    if i1 != i0+1:
        raise ValueError("link must connect two adjacent blocks")

    # diagonal blocks: h_S,nn - n*omega
    for ib in range(nb):
        hii = todense(hlist[ib][ib])
        for isb in range(ns):
            n = isb-nmax
            s = blk(ib, isb)
            H[s, s] += hii - n*omega*np.eye(dim)

    # off-diagonal (static) bonds, and the single AC-carrying bond
    v0 = todense(hlist[i1][i0])  # hopping <target i1| H |source i0>
    ve = proje_target@v0  # picks up e^{-i*omega*t}: couples n (source) -> n+1 (target)
    vh = projh_target@v0  # picks up e^{+i*omega*t}: couples n (source) -> n-1 (target)
    for ib in range(nb-1):
        if ib == i0:
            for isb in range(ns-1):
                s_src = blk(i0, isb)
                s_tgt = blk(i1, isb+1)
                H[s_tgt, s_src] += ve
                H[s_src, s_tgt] += ve.conj().T
            for isb in range(1, ns):
                s_src = blk(i0, isb)
                s_tgt = blk(i1, isb-1)
                H[s_tgt, s_src] += vh
                H[s_src, s_tgt] += vh.conj().T
        else:
            vij = todense(hlist[ib+1][ib])
            for isb in range(ns):
                s_a = blk(ib, isb)
                s_b = blk(ib+1, isb)
                H[s_b, s_a] += vij
                H[s_a, s_b] += vij.conj().T
    return H
