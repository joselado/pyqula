import numpy as np
import pytest

from pyqula import geometry
from pyqula import specialhopping

# The neighbor shells V1/V2/V3 and J1/J2/J3 of an interaction were built
# with the multicell builder of get_hamiltonian, which looks two cells away.
# On a cell smaller than the third shell (the one-site chain, a rectangular
# cell, a skewed basis) the pairs further away were dropped, with no
# warning, in every route that takes the shells: the mean field (exact
# diagonalization, triangle, jax, dense KPM, BdG, the exchange channels),
# the density and spin RPA, the Bethe-Salpeter kernel and the spinon mean
# field. The tail Vr/Jr sized its range from the distance and was right
# everywhere. So the oracle here is: the shell keyword must give exactly
# what Vr or Jr gives when it is the same coupling on the same shell, and
# the one-site chain must give what its cell of three sites gives, where
# the third shell is one cell away.


def shell(V, dist):
    """A coupling V on the pairs at distance dist only"""
    return lambda r1, r2: V if abs(np.linalg.norm(r1 - r2) - dist) < 1e-3 \
            else 0.


def rectangular():
    g = geometry.square_lattice()
    g.a2 = np.array([0., 5., 0.]) # the third shell is three cells along a1
    return g


def skewed():
    g = geometry.square_lattice()
    g.a2 = g.a2 + 3*g.a1 # the same lattice in a skewed basis
    return g


GEOMETRIES = {"chain": geometry.chain,
        "chain3": lambda: geometry.chain().get_supercell(3),
        "square": geometry.square_lattice,
        "honeycomb": geometry.honeycomb_lattice,
        "kagome": geometry.kagome_lattice,
        "rectangular": rectangular, "skewed": skewed}


@pytest.mark.parametrize("name", list(GEOMETRIES))
def test_shells_reach_every_cell(name):
    """The shells against the KD-tree builder of the sparse mean field,
    which reaches every cell that holds a pair: the same couplings in the
    same directions, and every pair of each shell kept (z_n per site on the
    shell n)"""
    from pyqula.scftk import sparsemeanfield
    g = GEOMETRIES[name]()
    vs = [1., 10., 100.]
    v = specialhopping.shell_interaction(g, vs)
    h = g.get_hamiltonian(has_spin=False)
    ref = sparsemeanfield.interaction(h, V1=vs[0], V2=vs[1], V3=vs[2])
    ref = {d: 2*np.array(m.todense()) for d, m in ref.items()} # halved there
    keys = {d for d in set(v) | set(ref)
            if np.max(np.abs(v.get(d, 0.))) > 0 or np.max(np.abs(ref.get(d, 0.))) > 0}
    for d in keys:
        a, b = v.get(d, 0.), ref.get(d, 0.)
        assert np.max(np.abs(a - b)) < 1e-10, (name, d)
    # the third shell is there, in every site
    third = sum(np.sum(np.abs(m - 100.) < 1e-12, axis=1) for m in v.values())
    assert np.all(third > 0), (name, third)


def chain(spin=False, n=1):
    g = geometry.chain()
    if n > 1: g = g.get_supercell(n)
    return g.get_hamiltonian(has_spin=spin)


MF = dict(filling=0.5, T=0.05, maxerror=1e-9, mix=0.5)


def energy(h, nk, **kw):
    np.random.seed(1)
    return h.get_mean_field_hamiltonian(nk=nk, return_total_energy=True,
            **MF, **kw)[1]


@pytest.mark.parametrize("spin", [False, True])
def test_third_shell_mean_field_against_the_cell_of_three_sites(spin):
    """V3 on the one-site chain is a bond three cells away; in the cell of
    three sites it is one cell away. The energy per site must agree, and
    differ from the energy without V3, which is what the chain gave
    before"""
    e1 = energy(chain(spin), 60, V3=1.0)
    e3 = energy(chain(spin, 3), 20, V3=1.0)
    e0 = energy(chain(spin), 60, V3=0.0)
    assert abs(e1 - e3/3) < 1e-8, (e1, e3/3)
    assert abs(e1 - e0) > 0.1


@pytest.mark.parametrize("route", ["ed", "triangle", "jax"])
def test_shell_keyword_is_the_tail_on_that_shell(route):
    """V3 and J3 against Vr and Jr equal to them on the third shell, in the
    exact diagonalization, triangle and jax mean fields"""
    kw = dict()
    if route == "triangle": kw = dict(integration="triangle")
    if route == "jax":
        pytest.importorskip("jax")
        kw = dict(use_jax=True)
    spin = route != "triangle"
    a = energy(chain(spin), 60, V3=1.0, **kw)
    b = energy(chain(spin), 60, Vr=shell(1.0, 3.), rcut=3.5, **kw)
    assert abs(a - b) < 1e-9, (route, a, b)
    if spin:
        a = energy(chain(True), 60, J3=1.0, **kw)
        b = energy(chain(True), 60, Jr=shell(1.0, 3.), rcut=3.5, **kw)
        assert abs(a - b) < 1e-9, (route, a, b)


def test_shell_keyword_in_the_dense_kpm_mean_field():
    """The dense KPM mean field took the shells from the same builder"""
    kw = dict(filling=0.5, T=0.05, maxerror=1e-4, mix=0.5, npol=200,
            nk=30, write=False, return_total_energy=True)
    h = chain()
    np.random.seed(1)
    a = h.get_mean_field_hamiltonian_kpm(V3=1.0, **kw)[1]
    np.random.seed(1)
    b = h.get_mean_field_hamiltonian_kpm(Vr=shell(1.0, 3.), rcut=3.5, **kw)[1]
    np.random.seed(1)
    c = h.get_mean_field_hamiltonian_kpm(V3=0.0, **kw)[1]
    assert abs(a - b) < 1e-8, (a, b)
    assert abs(a - c) > 0.1


def test_shell_keyword_in_the_pairing_mean_field():
    """An attractive V3 in a BdG mean field, against the same Vr"""
    out = []
    for kw in (dict(V3=-2.0), dict(Vr=shell(-2.0, 3.), rcut=3.5)):
        h = chain(True)
        h.add_onsite(-0.6)
        h.setup_nambu_spinor()
        np.random.seed(1)
        out.append(h.get_mean_field_hamiltonian(nk=40, mf="swave", T=0.01,
                maxerror=1e-8, mix=0.2, return_total_energy=True, **kw)[1])
    assert abs(out[0] - out[1]) < 1e-7, out


def test_shell_keyword_in_the_density_rpa():
    """The density RPA with V3 against the same Vr, at a generic q, and
    different from the bare bubble"""
    from pyqula.chitk import densitychi
    h = chain()
    h.add_onsite(0.4)
    kw = dict(q=[0.31, 0., 0.], energies=np.array([0.4, 1.3]), delta=0.1,
            T=0.05, nk=60)
    _, a = densitychi.densitychi_RPA(h, V3=0.5, **kw)
    _, b = densitychi.densitychi_RPA(h, Vr=shell(0.5, 3.), rcut=3.5, **kw)
    _, c = densitychi.densitychi_RPA(h, **kw)
    assert np.max(np.abs(a - b)) < 1e-12
    assert np.max(np.abs(a - c)) > 1e-2


def test_shell_keyword_in_the_spin_rpa():
    """A ferromagnetic chain with J3: the mean field and the transverse and
    longitudinal spin response, which goes through the pair basis since J3
    couples different sites, against the same Jr"""
    out = []
    for kw in (dict(J3=-0.5), dict(Jr=shell(-0.5, 3.), rcut=3.5)):
        h = chain(True).get_mean_field_hamiltonian(U=5.0, filling=0.2,
                mf="ferroZ", T=0.02, maxerror=1e-10, mix=0.5, nk=60, **kw)
        _, chi = h.get_spinchi_full(q=[0.31, 0., 0.], nk=60,
                energies=np.array([0.15, 0.6]), delta=0.05, T=0.02)
        out.append(chi)
    assert np.max(np.abs(out[0] - out[1])) < 1e-8


def test_shell_keyword_in_the_bse_interaction():
    """The Bethe-Salpeter density interaction with V3 against the same Vr"""
    from pyqula.bsetk import interaction
    h = chain()
    a = interaction.density_interaction(h, V3=0.7)
    b = interaction.density_interaction(h, Vr=shell(0.7, 3.), rcut=3.5)
    for d in set(a) | set(b):
        assert np.max(np.abs(a.get(d, 0.) - b.get(d, 0.))) < 1e-14, d
    assert np.max(np.abs(a[(3, 0, 0)])) > 0.


def test_shell_keyword_in_the_spinon_mean_field():
    """The bond decoupling of the spinon mean field with J3 against the
    same Jr, from the same real ansatz"""
    from pyqula.spinon import SpinonHamiltonian
    out = []
    for kw in (dict(J3=0.5), dict(Jr=shell(0.5, 3.), rcut=3.5)):
        g = geometry.chain()
        ans = g.get_hamiltonian(tij=[1., 0., 1.], has_spin=True)
        np.random.seed(0)
        out.append(SpinonHamiltonian(g).get_mean_field_hamiltonian(nk=60,
                J1=1.0, decoupling="bond", mix=0.5, maxerror=1e-6, T=0.05,
                ansatz=ans, return_total_energy=True, **kw)[1])
    assert abs(out[0] - out[1]) < 1e-8, out
