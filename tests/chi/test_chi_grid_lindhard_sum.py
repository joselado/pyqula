"""The grid response of chiAB_q is the Lindhard sum on the mesh, every pair
included.

The kernels skip a pair whose contribution is negligible, for speed. They
used to skip every pair with |fac| < delta/100, fac the Lindhard factor of
chiAB.lindhard_pair. For a degenerate pair fac = i*delta*f', so that test
dropped every degenerate pair with |f'| < 1e-2 at any delta, and at
delta = 0.1 it also dropped every pair with |f_a - f_b| < 1e-3. In a metal
at T = 0.05 to 0.1 this biased the response by up to 2e-3. Here the library sum
is compared with the same sum written in numpy without any skip, on the
same mesh and the same eigenpairs, for a spinful metal, where every band
is doubly degenerate, so that q = 0 is made of degenerate pairs only."""
import numpy as np
import pytest
from scipy.special import expit

from pyqula import algebra
from pyqula import geometry
from pyqula.chitk import chiAB as chiAB_mod

NK = 12
ENERGIES = np.linspace(-0.5, 0.5, 5)


def _metal():
    """A spinful honeycomb metal"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_onsite(0.4)
    return h


def _site_projectors(h):
    """The default projectors of chiAB_q, one per site, both spins"""
    n = len(h.geometry.r)
    out = []
    for i in range(n):
        p = np.zeros((2*n, 2*n), dtype=np.complex128)
        p[2*i, 2*i] = p[2*i+1, 2*i+1] = 1.
        out.append(p)
    return np.array(out)


def _lindhard_sum(h, q, T, delta):
    """chi[w,i,j], the mesh average of the Lindhard terms of every pair,
    with the degenerate pairs of chiAB.lindhard_pair, without any skip"""
    hk = h.get_hk_gen()
    ps = _site_projectors(h)
    beta = 1./T
    out = 0.
    ks = h.geometry.get_kmesh(nk=NK)
    for k in ks:
        e1, w1 = algebra.eigh(algebra.todense(hk(k)))
        e2, w2 = algebra.eigh(algebra.todense(hk(np.array(k) + q)))
        o1, o2 = expit(-beta*e1), expit(-beta*e2)
        fac = (o1[:, None] - o2[None, :]).astype(np.complex128)
        de = e1[:, None] - e2[None, :]
        deg = np.abs(de) < chiAB_mod.DEGENERATE_TOL
        fp = np.broadcast_to((-beta*o1*(1. - o1))[:, None], fac.shape)
        fac[deg] = 1j*delta*fp[deg]
        de[deg] = 0.
        MA = np.einsum("xa,ixy,yb->iab", w1.conj(), ps, w2)
        MB = np.einsum("xb,jxy,ya->jba", w2.conj(), ps, w1)
        D = fac[None]/(de[None] - ENERGIES[:, None, None] + 1j*delta)
        out = out + np.einsum("iab,jba,wab->wij", MA, MB, D)
    return out/len(ks)


@pytest.mark.parametrize("q", [[0., 0., 0.], [0.13, 0.07, 0.]])
@pytest.mark.parametrize("T,delta", [(5e-2, 0.1), (5e-2, 1e-3)])
@pytest.mark.parametrize("kw", [dict(mode="matrix"), dict(mode="trace"),
                                dict(mode="matrix", ij_mode="accelerated")])
def test_grid_is_the_lindhard_sum_without_skip(q, T, delta, kw):
    """mode="matrix" goes through chiAB_matrix, mode="trace" through
    chiAB_jit, ij_mode="accelerated" through chiAB_full_matrix_jit. The
    delta/100 cutoff made all three differ from the sum by 3e-6 to 2e-4
    here"""
    h = _metal()
    q = np.array(q)
    ref = _lindhard_sum(h, q, T, delta)
    chi = np.array(chiAB_mod.chiAB_q(h, q=q, energies=ENERGIES, nk=NK, T=T,
                                     delta=delta, **kw)[1])
    if kw["mode"] == "trace":  # the mean of the diagonal over the sites
        ref = np.mean(np.diagonal(ref, axis1=1, axis2=2), axis=1)
    assert chi.shape == ref.shape
    assert np.max(np.abs(chi - ref)) < 1e-12
    assert np.max(np.abs(ref)) > 1e-2  # a metal, a response to compare
