"""With gf_mode="triangle" the energy sweeps of an embedding
(Embedding.multildos, Embedding.multidos, embedding.dos_impurity) take the
bulk selfenergy of the whole sweep from green.supercell_selfenergy_batch,
which diagonalizes the nk^3 mesh of a 3d crystal once instead of once per
energy, and then do the Dyson step energy by energy. The contract is that a
sweep answers what the single-energy calls answer: to rounding for the
tetrahedra, where the energies of the batch share their matrix products,
and exactly for every other mode, which still goes energy by energy."""
import numpy as np
import pytest
import scipy.linalg as lg

from pyqula import geometry
from pyqula import embedding
from pyqula.greentk.selfenergy import bloch_selfenergy


ES = np.array([-1.1, -0.3, 0.05, 0.4, 1.3])
DELTA, NK = 0.05, 6


def _crystal():
    """Spinful diamond with a sublattice imbalance and an exchange field,
    so that the sz map is not zero"""
    h = geometry.diamond_lattice_minimal().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.3)
    h.add_exchange([0., 0., 0.2])
    return h


def _defect(h):
    hv = h.copy()
    hv.add_onsite(lambda r: 2.0 if np.sum(np.abs(r)) < 1e-6 else 0.)
    return hv


def _site_defect(h):
    """A strong onsite potential on the first site of the cell"""
    r0 = h.geometry.r[0]
    hv = h.copy()
    hv.add_onsite(lambda r: 3.0 if np.sum(np.abs(r-r0)) < 1e-6 else 0.)
    return hv


def _triangle_embedding():
    h = _crystal()
    eb = embedding.Embedding(h, m=_defect(h))
    eb.gf_mode = "triangle"
    return eb


def _close(a, b, rtol=1e-10):
    a, b = np.array(a), np.array(b)
    return np.max(np.abs(a-b)) <= rtol*np.max(np.abs(b))


def _bond(h, seed=3):
    """A random Hermitian matrix of the cell, which couples the sites"""
    n = h.intra.shape[0]
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    return (a + a.conj().T)/2.


def test_the_triangle_multildos_matches_get_ldos():
    eb = _triangle_embedding()
    for op in [None, "sz", _bond(eb.H)]:
        (x, y, es, maps) = eb.multildos(es=ES, delta=DELTA, nk=NK,
                                        operator=op, write=False)
        ref = [eb.get_ldos(energy=e, delta=DELTA, nk=NK, operator=op,
                           write=False)[2] for e in ES]
        assert np.array(maps).shape == np.array(ref).shape
        assert np.max(np.abs(ref)) > 1e-2 # not a vacuous map
        assert np.std(np.array(ref)[:, 0]) > 1e-3 # it moves with energy
        assert _close(maps, ref)


def test_the_triangle_multidos_is_the_sum_of_the_maps():
    eb = _triangle_embedding()
    (es, ds) = eb.multidos(es=ES, delta=DELTA, nk=NK)
    ref = [eb.get_dos(energy=e, delta=DELTA, nk=NK, write=False) for e in ES]
    assert _close(ds, ref)


def test_the_triangle_dos_impurity_matches_a_dyson_loop():
    h = _crystal()
    vc = _defect(h).intra
    ds, dv = embedding.dos_impurity(h, vc=vc, energies=ES, mode="triangle",
                                    delta=DELTA, nk=NK, write=False)
    iden = np.identity(h.intra.shape[0], dtype=np.complex128)
    for (e, d0, d1) in zip(ES, ds, dv):
        g, s = bloch_selfenergy(h, energy=e, delta=DELTA, nk=NK,
                                mode="triangle")
        gv = np.linalg.inv(iden*(e+1j*DELTA) - vc - s)
        assert abs(d0 + np.trace(g).imag) <= 1e-10*abs(d0)
        assert abs(d1 + np.trace(gv).imag) <= 1e-10*abs(d1)
    assert np.max(np.abs(ds-dv)) > 1e-2 # the defect changes the DOS


def test_the_other_modes_are_unchanged():
    """A 2d embedding with the default gf_mode does not batch, so the
    sweep calls the single-energy routine and the numbers are identical"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    hv = _site_defect(h)
    eb = embedding.Embedding(h, m=hv.intra)
    for nsuper in [1, 2]:
        kw = dict(delta=0.1, nk=6, nsuper=nsuper)
        (x, y, es, maps) = eb.multildos(es=ES[:3], write=False, **kw)
        ref = [eb.get_ldos(energy=e, write=False, **kw)[2] for e in ES[:3]]
        assert len(x) == len(ref[0]) == 2*nsuper**2
        assert np.max(np.abs(ref)) > 1e-2
        assert np.max(np.abs(np.array(maps)-np.array(ref))) == 0.
        (es, ds) = eb.multidos(es=ES[:3], **kw)
        assert np.max(np.abs(ds-np.sum(ref, axis=1))) == 0.
    ds, dv = embedding.dos_impurity(h, vc=hv.intra, energies=ES[:3],
                                    mode="adaptive", delta=0.1, write=False)
    iden = np.identity(h.intra.shape[0], dtype=np.complex128)
    for (e, d1) in zip(ES[:3], dv):
        g, s = bloch_selfenergy(h, energy=e, delta=0.1, nk=50,
                                mode="adaptive")
        gv = lg.inv(iden*(e+0.1j) - hv.intra - s) # as dos_impurity does
        assert d1 == -np.trace(gv).imag
    assert np.max(np.abs(ds-dv)) > 1e-2 # the defect changes the DOS


def test_the_sweeps_carry_the_operator_in_the_other_modes():
    """The sweeps of a 2d embedding with an operator: a name, a potential,
    a bond matrix of the embedded cell and an Operator object reach every
    energy as they reach get_ldos, at nsuper=1 and in a supercell"""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_rashba(0.3)
    h.add_exchange([0.1, 0.2, 0.3])
    eb = embedding.Embedding(h, m=_site_defect(h).intra)
    for nsuper in [1, 2]:
        kw = dict(delta=0.1, nk=6, nsuper=nsuper)
        ops = ["sz", lambda r: 1.0 + 0.3*r[0] + 0.17*r[1],
               _bond(h.supercell(nsuper))]
        if nsuper == 1: ops.append(h.get_operator("sx"))
        plain = eb.multildos(es=ES[:3], write=False, **kw)[3]
        for op in ops:
            maps = eb.multildos(es=ES[:3], operator=op, write=False, **kw)[3]
            ref = [eb.get_ldos(energy=e, operator=op, write=False, **kw)[2]
                   for e in ES[:3]]
            assert np.max(np.abs(np.array(maps)-plain)) > 1e-3 # it arrived
            assert np.max(np.abs(np.array(maps)-np.array(ref))) == 0.
            (es, ds) = eb.multidos(energies=ES[:3], operator=op, **kw)
            assert _close(ds, np.sum(ref, axis=1), rtol=1e-12)


def test_multildos_takes_energies_as_multidos_does():
    """energies= is the name of the grid in h.get_multildos and in
    Embedding.multidos; Embedding.multildos took it and ignored it,
    returning its default grid of 20 energies"""
    eb = _triangle_embedding()
    kw = dict(delta=DELTA, nk=4, write=False)
    (x, y, es, maps) = eb.multildos(energies=ES[:2], **kw)
    ref = eb.multildos(es=ES[:2], **kw)[3]
    assert len(es) == 2 and np.max(np.abs(es-ES[:2])) == 0.
    assert np.array(maps).shape == np.array(ref).shape
    assert np.max(np.abs(np.array(maps)-np.array(ref))) == 0.


@pytest.mark.parametrize("which", ["multildos", "multidos", "dos_impurity"])
def test_a_triangle_sweep_diagonalizes_the_mesh_once(which, monkeypatch):
    """What the sweeps gain: one call of the tetrahedron Green's function,
    with all the energies, where a loop over the energies makes one call,
    and one diagonalization of the nk^3 mesh, per energy"""
    from pyqula.dostk import simplexdos
    calls = []
    green_triangle = simplexdos.green_triangle

    def counted(h, energies, **kwargs):
        calls.append(len(np.atleast_1d(energies)))
        return green_triangle(h, energies, **kwargs)

    monkeypatch.setattr(simplexdos, "green_triangle", counted)
    eb = _triangle_embedding()
    eb.get_ldos(energy=ES[0], delta=DELTA, nk=4, write=False)
    assert calls == [1] # the counter sees the single-energy route
    calls.clear()
    if which == "multildos":
        eb.multildos(es=ES, delta=DELTA, nk=4, write=False)
    elif which == "multidos":
        eb.multidos(es=ES, delta=DELTA, nk=4)
    else:
        embedding.dos_impurity(eb.H, vc=eb.m, energies=ES, mode="triangle",
                               delta=DELTA, nk=4, write=False)
    assert calls == [len(ES)]
