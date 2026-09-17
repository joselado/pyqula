"""Which basis the public spin response is summed in.

The site-basis vertex has one index per site, so it is exact for an
interaction that couples each site to itself (a Hubbard U) and misses the
Fock rung of anything that couples two different sites, which lives on the
electron-hole pair index. get_spinchi_full, get_spinchi_ladder and
get_magnon_bands therefore sum any interaction between different sites in
the pair basis (chitk.pairchi). These tests pin the rule, including the
case the lattice-vector keys alone do not show (a 0D island keeps all of
its bonds under (0,0,0)), and that the two routes agree where the site
vertex is exact.

The last tests are the spin-rotation guard on the recorded channels: a
global rotation keeps an isotropic exchange exact and would put an
anisotropic one in the wrong frame, so it is refused.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk import spinchi
from pyqula.chitk.pairchi import pair_chi_rpa
from pyqula.meanfield import VJinteraction

NK = 6


def _chain(**kw):
    g = geometry.chain().get_supercell(2)
    g.get_sublattice()
    return VJinteraction(g.get_hamiltonian(), filling=0.5, mf="antiferro",
                         nk=NK, maxerror=1e-11, mix=0.3, maxite=3000,
                         **kw).hamiltonian


def test_the_two_routes_agree_where_the_site_vertex_is_exact():
    """A half-filled Neel Hubbard chain. The bare response is the same on
    both routes (measured 6e-17) and so are the dressed transverse blocks
    (5e-14), which is where the site vertex is exact. The dressed zz block
    is not: the pair kernel couples Sz to the charge fluctuations the Neel
    state mixes it with (the bare charge-Sz response is 1e-3 here), which
    a spin-only vertex has no channel for, and it moves by 2e-5."""
    h = _chain(U=3.0)
    assert spinchi._is_site_local(h)
    kw = dict(energies=np.linspace(-1.5, 1.5, 13), delta=0.05, nk=NK,
              q=[0.13, 0., 0.], T=1e-3)
    n = len(h.geometry.r)
    for RPA in (False, True):
        _, site = h.get_spinchi_full(RPA=RPA, **kw)
        if RPA:
            _, pair = spinchi._pair_route_response(h, spinchi._SPIN_OPS,
                                                   spinchi._SPIN_OPS, **kw)
        else:
            _, pair = pair_chi_rpa(h, W={(0, 0, 0): np.zeros((2*n, 2*n))},
                                   **kw)
        d = np.abs(np.array(site) - pair)
        assert np.max(np.abs(site)) > 0.1
        assert np.max(d[:, :2*n, :2*n]) < 1e-10  # the transverse blocks
        assert np.max(d) < (1e-10 if not RPA else 1e-4)


def test_an_exchange_between_sites_is_summed_in_the_pair_basis():
    h = _chain(U=1.0, J1=2.0)
    assert not spinchi._is_site_local(h)
    kw = dict(energies=np.linspace(0., 2., 7), delta=0.05, nk=NK,
              q=[0.1, 0., 0.])
    _, chi = h.get_spinchi_full(**kw)
    _, ref = pair_chi_rpa(h, T=0.05, **kw)
    assert np.max(np.abs(chi - ref)) < 1e-12


def test_a_zero_dimensional_island_with_v1_is_not_site_local():
    """All the bonds of an island sit under the (0,0,0) key, so a gate on
    the keys let this through to a site vertex that is identically zero,
    and a V1-ordered ferromagnet had no spin response dressing at all."""
    g = geometry.chain().get_supercell(6)
    g.dimensionality = 0
    h = g.get_hamiltonian().get_mean_field_hamiltonian(V1=1.5, filling=0.1,
            mf="ferro", nk=1, maxerror=1e-10, mix=0.3, maxite=3000)
    assert max(abs(h.get_vev("sz"))) > 0.1
    assert max(np.max(np.abs(m)) for m in
               spinchi._full_spin_U(h).values()) == 0.0
    assert not spinchi._is_site_local(h)
    kw = dict(energies=np.linspace(0., 1., 5), delta=0.05, nk=1)
    _, chi = h.get_spinchi_full(**kw)
    _, ref = pair_chi_rpa(h, T=0.05, q=[0., 0., 0.], **kw)
    _, bare = h.get_spinchi_full(RPA=False, **kw)
    assert np.max(np.abs(chi - ref)) < 1e-12
    assert np.max(np.abs(chi - np.array(bare))) > 1e-2  # dressed, not bare


def test_what_the_pair_basis_does_not_have_is_refused():
    h = _chain(U=1.0, J1=2.0)
    es = np.linspace(0., 1., 3)
    with pytest.raises(NotImplementedError, match="chi_cpugpu"):
        h.get_spinchi_full(energies=es, nk=NK, chi_cpugpu="GPU")
    with pytest.raises(NotImplementedError, match="imode"):
        h.get_spinchi_full(energies=es, nk=NK, imode="adaptive")
    with pytest.raises(ValueError, match="accepted"):
        h.get_spinchi_full(energies=es, nk=NK, A="sx")


def test_a_rotation_keeps_an_isotropic_or_uniaxial_exchange():
    """An isotropic exchange is invariant under any global rotation, a
    uniaxial one under a rotation about its axis, and a full turn about
    any axis changes nothing: all three keep their channels."""
    for kw, vector, angle in ((dict(J1=3.0), [1., 0.3, 0.], 0.37),
                              (dict(U=4.0, J1=1.0, J1z=0.5), [0., 0., 1.],
                               0.37),
                              (dict(U=4.0, J1=1.0, J1z=0.5), [0., 1., 0.],
                               1.0)):
        h = _chain(**kw)
        h.global_spin_rotation(vector=vector, angle=angle)
        assert h.Vchannels is not None


def test_a_rotation_of_an_anisotropic_exchange_is_refused():
    """A rotated XXZ exchange has Sx_i Sz_j-like cross terms no x/y/z
    channel can hold, so the rotation is refused before anything is
    touched."""
    h = _chain(U=4.0, J1=1.0, J1z=0.5)
    e0 = np.sort(h.get_bands(write=False)[1])
    with pytest.raises(ValueError, match="anisotropic"):
        h.global_spin_rotation(vector=[1., 0., 0.], angle=0.25)
    assert np.max(np.abs(np.sort(h.get_bands(write=False)[1]) - e0)) < 1e-12
    h.V, h.Vchannels = None, None  # the remedy the message names
    h.global_spin_rotation(vector=[1., 0., 0.], angle=0.25)
