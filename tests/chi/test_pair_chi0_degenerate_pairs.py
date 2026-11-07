"""The bare response of the pair basis is the one of the site basis.

On a k-mesh, a pair of states with the same energy, E_n(k) = E_m(k+q),
stands for the intraband continuum around it, and the site-basis kernels
(chitk.chiAB.lindhard_pair) give it the term i*delta*f'/(i*delta - omega):
the static Pauli term f' at omega = 0 and zero for |omega| >> delta. Every
pair is of this kind at q = 0, n = m and the spin-degenerate partners, and
at a q on the mesh the nesting pairs E(k) = E(k+q) are too. The pair basis
(chitk.pairchi) used to skip every pair with f_n - f_m = 0, so in a metal
its chi_0 at q = 0 had no Fermi-surface term at all. On the meshes below
(nk = 8, T = 0.1) the two routes then differed by 0.112 at q = 0 on the
square-lattice metal, which is its whole bare response, by 0.436 with an
exchange field, by 0.042 on the honeycomb, and by 7e-3 to 0.25 at the
nested q = (1/4, 1/4), while they agreed to 3e-16 at a generic q. With the
same rule on both routes they agree to 2e-15, static and dynamic.

The pair operators c^dag_a c_b of one site give the site-basis response of
the projectors |a><b| directly, so chiAB_q with those projectors is the
reference, with the same mesh, q, delta and an explicit T on both sides
(T=None means T=delta in chiAB_q and a step in pair_chi0).
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk.chiAB import chiAB_q
from pyqula.chitk.pairchi import pair_chi0, pair_chi_rpa

NK = 8
T = 0.1
ENERGIES = np.array([0., 0.05, 0.5])


def _metal(name, mz):
    g = (geometry.square_lattice() if name == "square"
         else geometry.honeycomb_lattice())
    h = g.get_hamiltonian()  # spinful
    h.add_onsite(0.4 if name == "square" else 0.3)  # away from half filling
    if mz != 0.: h.add_exchange([0., 0., mz])
    return h


def _both_routes(h, q, delta):
    n = h.intra.shape[0]
    pairs = [(a, b, (0, 0, 0)) for i in range(len(h.geometry.r))
             for a in (2*i, 2*i + 1) for b in (2*i, 2*i + 1)]
    pA, pB = [], []
    for (a, b, _) in pairs:
        m = np.zeros((n, n), dtype=np.complex128)
        m[a, b] = 1.
        pA.append(m)
        pB.append(m.conj().T)
    _, site = chiAB_q(h, energies=ENERGIES, q=q, nk=NK, pAs=np.array(pA),
                      pBs=np.array(pB), delta=delta, T=T)
    site = np.transpose(np.array(site), (1, 2, 0))
    pair = pair_chi0(h, pairs, q=q, energies=ENERGIES, delta=delta, nk=NK,
                     T=T)
    return site, pair


@pytest.mark.parametrize("name,mz", [("square", 0.), ("square", 0.3),
                                     ("honeycomb", 0.3)])
@pytest.mark.parametrize("q", [[0., 0., 0.], [0.137, 0.211, 0.],
                               [0.25, 0.25, 0.]])
@pytest.mark.parametrize("delta", [1e-2, 1e-1])
def test_the_two_routes_give_the_same_bare_response(name, mz, q, delta):
    """q = 0, a generic q, and a q on the mesh with nesting pairs"""
    site, pair = _both_routes(_metal(name, mz), q, delta)
    assert np.max(np.abs(site[:, :, 0])) > 0.05  # not zero against zero
    assert np.max(np.abs(site - pair)) < 1e-12


def test_the_static_uniform_response_is_the_fermi_surface_term():
    """One band per spin: at q = 0 every pair is degenerate, so the whole
    bare response is the degenerate pair term, -f' summed over the mesh in
    the static limit and suppressed as delta/omega at finite frequency.
    Skipping those pairs gave exactly zero here"""
    h = _metal("square", 0.)
    _, pair = _both_routes(h, [0., 0., 0.], 1e-2)
    up = pair[0, 0]  # <<c^dag_up c_up ; c^dag_up c_up>>
    e = np.array([np.linalg.eigvalsh(h.get_hk_gen()(k))[0]
                  for k in h.geometry.get_kmesh(nk=NK)])
    f = 0.5*(1. - np.tanh(0.5*e/T))
    pauli = np.mean(-f*(1. - f)/T)  # the mesh sum of f', per spin
    assert abs(pauli) > 0.05
    assert abs(up[0] - pauli) < 1e-12
    w = ENERGIES[2]
    assert abs(up[2] - pauli*1j*1e-2/(1j*1e-2 - w)) < 1e-12


def test_the_public_spin_response_agrees_on_both_routes_in_a_metal():
    """The bare spin response of the public entry point (site basis) and
    the pair basis with a zero interaction, at q = 0 where the Pauli term
    is everything"""
    h = _metal("square", 0.3)
    n = h.intra.shape[0]
    kw = dict(energies=ENERGIES, delta=1e-2, nk=NK, q=[0., 0., 0.], T=T)
    _, site = h.get_spinchi_full(RPA=False, **kw)
    _, pair = pair_chi_rpa(h, W={(0, 0, 0): np.zeros((n, n))}, **kw)
    assert np.max(np.abs(np.array(site)[0])) > 0.05
    assert np.max(np.abs(np.array(site) - pair)) < 1e-12
