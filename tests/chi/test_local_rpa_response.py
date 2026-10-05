"""The local RPA response, q=None, is the mesh average of the response
dressed at every q.

With q left out, chiAB averages the bare response over the q-mesh, which is
the local response, the one a probe on a single site sees. The dressed
response used to be built from that average, chi0_avg (1 - V(0) chi0_avg)^-1,
and that is the RPA of nothing: the ladder is summed at a fixed momentum
transfer, so every q has to be dressed on its own, chi0(q)(1 - V(q) chi0(q))^-1,
and the average taken afterwards. The pair basis (chitk.pairchi) always did
the second, so the two local responses of the same Hubbard U disagreed;
measured on a half-filled Neel Hubbard chain at nk=6 the old number was off
by 18 against a maximum of 19. These tests pin the construction three ways:
against the pair basis where the site vertex is exact, against the mesh
average of the per-q calls, and against the old recipe, which has to differ.
The charge channel (get_densitychi_RPA) goes through the same dressing and
is pinned the same way.
"""
import numpy as np

from pyqula import geometry
from pyqula.chitk import spinchi
from pyqula.chitk.rpa import interaction_at_q
from pyqula.meanfield import VJinteraction

NK = 6


def _neel_chain(**kw):
    g = geometry.chain().get_supercell(2)
    g.get_sublattice()
    return VJinteraction(g.get_hamiltonian(), filling=0.5, mf="antiferro",
                         nk=NK, maxerror=1e-11, mix=0.3, maxite=3000,
                         **kw).hamiltonian


def _old_recipe(bare, V):
    """The averaged bare response dressed at q=0, what q=None used to do"""
    iden = np.identity(bare[0].shape[0])
    return np.array([c@np.linalg.inv(iden - V@c) for c in bare])


def test_the_local_spin_response_is_the_average_of_the_dressed_one():
    """Full (Sx,Sy,Sz) tensor and S+/S- ladder of a half-filled Neel
    Hubbard chain, where the site vertex is exact for the transverse
    response. Measured: 3e-13 against the pair basis on every block,
    exact against the mesh average of the per-q calls, 18 away from the
    old recipe (ladder: 4e-14 and 37)."""
    h = _neel_chain(U=3.0)
    assert spinchi._is_site_local(h)
    kw = dict(energies=np.linspace(-1.5, 1.5, 13), delta=0.05, nk=NK, T=1e-3)
    qs = h.geometry.get_kmesh(nk=NK)
    assert len(qs) == NK
    # the (Sx,Sy,Sz) tensor
    _, local = h.get_spinchi_full(**kw)
    local = np.array(local)
    mean = np.mean([np.array(h.get_spinchi_full(q=q, **kw)[1]) for q in qs],
                   axis=0)
    _, pair = spinchi._pair_route_response(h, spinchi._SPIN_OPS,
                                           spinchi._SPIN_OPS, **kw)
    _, bare = h.get_spinchi_full(RPA=False, **kw)
    old = _old_recipe(np.array(bare),
                      interaction_at_q(spinchi._full_spin_U(h), h, None))
    assert np.max(np.abs(local)) > 10.
    assert np.max(np.abs(local - mean)) < 1e-12
    assert np.max(np.abs(local - pair)) < 1e-10
    assert np.max(np.abs(local - old)) > 1.
    # the S+/S- ladder
    sp = np.array([[0., 1.], [0., 0.]], dtype=complex)
    _, local = h.get_spinchi_ladder(**kw)
    local = np.array(local)
    mean = np.mean([np.array(h.get_spinchi_ladder(q=q, **kw)[1]) for q in qs],
                   axis=0)
    _, pair = spinchi._pair_route_response(h, [sp], [sp.T], **kw)
    _, bare = h.get_spinchi_ladder(RPA=False, **kw)
    old = _old_recipe(np.array(bare),
                      interaction_at_q(spinchi._transverse_spin_K(h), h, None))
    assert np.max(np.abs(local)) > 10.
    assert np.max(np.abs(local - mean)) < 1e-12
    assert np.max(np.abs(local - pair)) < 1e-10
    assert np.max(np.abs(local - old)) > 1.


def test_the_local_charge_response_is_the_average_of_the_dressed_one():
    """get_densitychi_RPA needs no mean field, so this is the bare chain
    with a V1 and a U. The dressed response at every q uses V(q), which
    for a first-neighbor V1 is 2 V1 cos(2 pi q), so the old recipe, V(0)
    on the averaged bare response, is a different number (measured 0.7
    away, against a maximum of 1.3)."""
    h = geometry.chain().get_hamiltonian()
    kw = dict(energies=np.linspace(0., 3., 7), delta=0.05, nk=8)
    V = dict(V1=0.6, U=0.5)
    qs = h.geometry.get_kmesh(nk=8)
    _, local = h.get_densitychi_RPA(**V, **kw)
    local = np.array(local)
    mean = np.mean([np.array(h.get_densitychi_RPA(q=q, **V, **kw)[1])
                    for q in qs], axis=0)
    from pyqula.chitk.densitychi import _density_v
    from pyqula.chitk.rpa import chi_AB_RPA
    h1 = h.get_multicell().get_dense()
    _, bare = chi_AB_RPA(h1, V=None, **kw)  # the averaged bare response
    old = _old_recipe(np.array(bare), interaction_at_q(
        _density_v(h1, **V), h1, None))
    assert np.max(np.abs(local)) > 0.1
    assert np.max(np.abs(local - mean)) < 1e-12
    assert np.max(np.abs(local - old)) > 1e-2*np.max(np.abs(local))
