"""The distance-dependent interactions Vr and Jr of a mean-field calculation
are the interaction of each pair of sites, the same convention as the
neighbor shells V1/V2/V3 and J1/J2/J3.

The interaction dictionary v of the mean field stores half of each pair
(V1/2 on every first-neighbor entry, U/2 on each up-down entry of a site),
since the decoupling visits every pair from both of its ends. Vr and Jr
used to be stored whole, so that the mean field of Vr was the one of 2*Vr,
twice what bsetk.interaction.density_interaction and chitk.densitychi take
as Vr. The oracle is exact: a Vr equal to V1 on the first-neighbor shell and
zero beyond it is V1, and the same for Jr and J1."""
import numpy as np

from pyqula import geometry
from pyqula import meanfield


def _first_shell(value):
    """A pair interaction equal to value between first neighbors, whose
    distance is 1 in pyqula's lattices, and zero otherwise"""
    def f(r1, r2):
        r = np.linalg.norm(np.array(r1) - np.array(r2))
        return value if abs(r - 1.) < 1e-3 else 0.
    return f


def _same_scf(a, b, tol=1e-8):
    """The two SCF objects converged to the same Hamiltonian and energy"""
    assert a.converged and b.converged
    assert abs(a.total_energy - b.total_energy) < tol, \
        (a.total_energy, b.total_energy)
    da, db = a.hamiltonian.get_dict(), b.hamiltonian.get_dict()
    for key in set(da) | set(db):
        ma = np.array(da.get(key, 0.*da[(0, 0, 0)]))
        mb = np.array(db.get(key, 0.*db[(0, 0, 0)]))
        assert np.max(np.abs(ma - mb)) < tol, key


def test_vr_on_the_first_shell_is_v1_in_the_spinful_mean_field():
    """VJinteraction, the engine of get_mean_field_hamiltonian, on a
    honeycomb lattice with U and a first-neighbor V: the charge order it
    drives and its Fock bond terms have to be the same whether the V comes
    as V1 or as a Vr restricted to the first shell"""
    g = geometry.honeycomb_lattice()
    kw = dict(U=1.0, filling=0.5, mf="CDW", nk=6, maxerror=1e-11, mix=0.8,
              maxite=3000)
    a = meanfield.VJinteraction(g.get_hamiltonian(), V1=2.0, **kw)
    b = meanfield.VJinteraction(g.get_hamiltonian(), Vr=_first_shell(2.0),
                                rcut=1.5, **kw)
    _same_scf(a, b)


def test_vr_on_the_first_shell_is_v1_in_the_spinless_mean_field(tmp_path, monkeypatch):
    """The same for the spinless density-density engine, Vinteraction,
    which builds its interaction separately"""
    monkeypatch.chdir(tmp_path) # the engine writes MF.pkl
    g = geometry.honeycomb_lattice()
    kw = dict(filling=0.5, mf="CDW", nk=6, maxerror=1e-11, mix=0.8,
              maxite=3000, verbose=0)
    h = g.get_hamiltonian(has_spin=False)
    a = meanfield.Vinteraction(h, V1=2.0, **kw)
    b = meanfield.Vinteraction(h, Vr=_first_shell(2.0), rcut=1.5, **kw)
    _same_scf(a, b)


def test_vr_on_the_first_shell_is_v1_in_the_kpm_mean_field(tmp_path, monkeypatch):
    """And for the KPM engine, Vinteraction_kpm, which builds it a third
    time; its Chebyshev density matrix is deterministic, so the two runs
    must agree to the tolerance of the loop"""
    monkeypatch.chdir(tmp_path) # the engine writes MF.pkl
    g = geometry.honeycomb_lattice()
    kw = dict(filling=0.5, mf="CDW", nk=4, maxerror=1e-9, mix=0.8,
              maxite=2000, npol=60, verbose=0)
    h = g.get_hamiltonian(has_spin=False)
    a = meanfield.Vinteraction_kpm(h, V1=2.0, **kw)
    b = meanfield.Vinteraction_kpm(h, Vr=_first_shell(2.0), rcut=1.5, **kw)
    _same_scf(a, b, tol=1e-7)


def test_jr_on_the_first_shell_is_j1():
    """The exchange tail Jr follows the same convention as J1, which the
    ferromagnetic chain of VJinteraction checks in the same way"""
    g = geometry.chain()
    kw = dict(filling=0.2, mf="ferroZ", nk=40, maxerror=1e-11, mix=0.3,
              maxite=3000)
    a = meanfield.VJinteraction(g.get_hamiltonian(), U=1.0, J1=-1.5, **kw)
    b = meanfield.VJinteraction(g.get_hamiltonian(), U=1.0,
                                Jr=_first_shell(-1.5), rcut=1.5, **kw)
    _same_scf(a, b)


def test_the_jax_engine_builds_the_same_interaction():
    """The jax engine (VJinteraction(use_jax=True)) builds its interaction
    with the same _build_density_v and _build_v, so Vr and Jr on the first
    shell give it the same matrices as V1 and J1"""
    from pyqula.scftk.spinspin import _build_density_v, _build_v
    h = geometry.honeycomb_lattice().get_hamiltonian().get_multicell()
    va = _build_density_v(h, V1=2.0, U=1.0)
    vb = _build_density_v(h, Vr=_first_shell(2.0), U=1.0, rcut=1.5)
    ja = _build_v(h, J1=0.7)
    jb = _build_v(h, Jr=_first_shell(0.7), rcut=1.5)
    for (x, y) in [(va, vb), (ja, jb)]:
        for key in set(x) | set(y):
            mx = x.get(key, 0.*x[(0, 0, 0)]) ; my = y.get(key, 0.*y[(0, 0, 0)])
            assert np.max(np.abs(np.array(mx) - np.array(my))) < 1e-12, key


def _onsite(value):
    """A pair interaction equal to value on a site with itself and zero on
    every other pair"""
    def f(r1, r2):
        return value if np.linalg.norm(np.array(r1) - np.array(r2)) < 1e-6 \
            else 0.
    return f


def test_vr_and_jr_at_zero_distance_are_an_onsite_hubbard_term():
    """Vr and Jr are evaluated on every pair of sites within rcut, the pair
    of a site with itself included, and both enter as half the sum over
    ordered pairs, H = 1/2 sum_ij [Vr(r_ij) n_i n_j + Jr(r_ij) S_i.S_j],
    whose diagonal is Vr(0) n_i n_i / 2 and Jr(0) S_i.S_i / 2. Written out
    on one orbital, n_i n_i / 2 is n_up n_dn plus a one-body term and
    S_i.S_i / 2 is -3/4 n_up n_dn plus one, and the one-body terms are what
    Hartree-Fock makes of n_a n_a, nothing, so at the mean-field level
    Vr(0) is a Hubbard U = Vr(0) and Jr(0) a Hubbard U = -3 Jr(0)/4.
    Measured: Vr(0)=3 and Jr(0)=-4 each converge to the Hamiltonian of
    U=3 to 2e-16, on a ferromagnetic state that is not a vacuous match.
    tests/magnon/test_exchange_rung.py pins the same diagonal in the
    magnon kernel against a brute-force reference."""
    g = geometry.chain()
    kw = dict(filling=0.2, mf="ferroZ", nk=40, maxerror=1e-11, mix=0.3,
              maxite=3000)
    u = meanfield.VJinteraction(g.get_hamiltonian(), U=3.0, **kw)
    assert abs(u.hamiltonian.get_vev("sz")[0]) > 0.05
    v = meanfield.VJinteraction(g.get_hamiltonian(), Vr=_onsite(3.0),
                                rcut=0.5, **kw)
    _same_scf(u, v)
    j = meanfield.VJinteraction(g.get_hamiltonian(), Jr=_onsite(-4.0),
                                rcut=0.5, **kw)
    _same_scf(u, j)
    # and the interaction itself: Jr(0) lands on the same-site block of
    # every channel as Jr(0)/8 times the sign pattern of Sz_i Sz_j, the
    # entry a bond with J1 = Jr(0) carries, which the bond has at both d
    # and -d and the diagonal only once (hence the half)
    from pyqula.scftk.spinspin import _build_v
    vj = _build_v(g.get_hamiltonian().get_multicell(), Jr=_onsite(1.0),
                  rcut=0.5)
    assert list(vj) == [(0, 0, 0)]
    assert np.allclose(vj[(0, 0, 0)], np.array([[1., -1.], [-1., 1.]])/8.)
