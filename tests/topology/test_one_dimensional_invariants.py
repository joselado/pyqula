"""The winding number of chiral one-dimensional Hamiltonians and the Z2
invariant of one-dimensional time-reversal-symmetric superconductors
(topology.winding_number, topology.z2_invariant_1d).

In one dimension the package had only the Zak phase, which is a Z2 and so
cannot tell one zero mode per end from two, and which is trivial for any
time-reversal-symmetric superconductor, whose two Kramers copies add up.
Both new invariants are checked against the Zak phase where it does apply:
its parity has to be the parity of the winding, and the Z2 of a helical
wire whose spin is conserved has to be the Zak parity of one spin block,
which is a Kitaev chain."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula import topology


def _zak_parity(h):
    return (-1)**int(round(topology.berry_phase(h, nk=100, write=False)/np.pi))


def _ssh(t1, t2):
    """Two-site chain with intracell hopping t1 and intercell hopping t2"""
    g = geometry.bichain()
    x0 = g.r[0][0]

    def fun(r1, r2):
        dr = r1 - r2
        if abs(np.linalg.norm(dr) - 1.) > 1e-4:
            return 0.
        xm = (r1[0] + r2[0])/2.  # bond midpoint, intracell at x0+1/2
        return t1 if abs((xm - x0 - 0.5) % 2.) < 1e-4 else t2
    return g.get_hamiltonian(fun=fun, has_spin=False)


def _kitaev(mu):
    """One spin band isolated by a large Zeeman field, with p-wave
    pairing: the Kitaev chain, topological for |mu| < 2"""
    h = geometry.chain().get_hamiltonian()
    h.add_onsite(20. + mu)
    h.add_zeeman([0., 0., 20.])
    h.add_pairing(mode="pwave", delta=0.3, d=[1., 0., 0.])
    return h


def _xpwave(r1, r2):
    """The p-wave pairing of _kitaev, on the bonds along x only"""
    dr = r1 - r2
    if abs(np.linalg.norm(dr) - 1.) < 1e-4 and abs(dr[1]) < 1e-6:
        return dr[0]*np.array([[0., 1.], [1., 0.]], dtype=complex)
    return np.zeros((2, 2), dtype=complex)


def _kitaev_ladder(mu, tperp=0.1):
    """Two Kitaev chains coupled by a rung hopping tperp"""
    g = geometry.ladder()

    def fun(r1, r2):
        dr = r1 - r2
        if abs(np.linalg.norm(dr) - 1.) > 1e-4:
            return 0.
        return 1. if abs(dr[1]) < 1e-6 else tperp
    h = g.get_hamiltonian(fun=fun)
    h.add_onsite(20. + mu)
    h.add_zeeman([0., 0., 20.])
    h.add_pairing(mode=_xpwave, delta=0.3)
    return h


def _helical(mu, delta=0.3j, rashba=0., ds=0.):
    """Spin up and spin down paired with opposite chirality, time-reversal
    symmetric: two Kitaev chains that are Kramers partners"""
    h = geometry.chain().get_hamiltonian()
    h.add_onsite(mu)
    if rashba:
        h.add_rashba(rashba)
    h.add_pairing(mode="pwave", delta=delta, d=[1., 0., 0.])
    if ds:
        h.add_swave(ds)
    return h


@pytest.mark.parametrize("t1,t2,w", [(1., 0.5, 0), (0.5, 1., 1)])
def test_ssh_winding(t1, t2, w, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _ssh(t1, t2)
    W = h.get_winding_number()  # the sublattice operator by default
    assert abs(W) == w
    assert (-1)**W == _zak_parity(h)


@pytest.mark.parametrize("mu", [-3., -1., 0., 1.5, 3.])
def test_kitaev_chain_winding(mu, tmp_path, monkeypatch):
    """sigma_y tau_y is the chiral symmetry of the real BdG matrix"""
    monkeypatch.chdir(tmp_path)
    h = _kitaev(mu)
    W = h.get_winding_number()
    assert abs(W) == int(abs(mu) < 2.)
    assert (-1)**W == _zak_parity(h)


def test_two_kitaev_chains_wind_twice(tmp_path, monkeypatch):
    """Two Majorana zero modes at each end, which the Zak phase reads as
    none"""
    monkeypatch.chdir(tmp_path)
    h = _kitaev_ladder(0.)
    assert abs(h.get_winding_number()) == 2
    assert _zak_parity(h) == 1
    assert h.get_winding_number() == 2*_kitaev(0.).get_winding_number()
    assert _kitaev_ladder(3.).get_winding_number() == 0


def test_winding_needs_a_chiral_symmetry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _ssh(0.5, 1.)
    h.add_onsite(0.1)  # breaks the sublattice symmetry
    with pytest.raises(ValueError, match="anticommute"):
        h.get_winding_number()
    h = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError, match="chiral"):
        h.get_winding_number()  # no sublattice and not a Nambu Hamiltonian


@pytest.mark.parametrize("mu", [-3., -1., 0.5, 1.5, 3.])
def test_helical_wire_z2_is_the_kitaev_parity_of_one_spin(mu, tmp_path,
                                                         monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _helical(mu)
    assert h.has_time_reversal_symmetry()
    nu = h.get_topological_invariant()
    assert nu == topology.z2_invariant_1d(h)
    assert nu == _zak_parity(_kitaev(mu))
    assert nu == topology.z2_invariant_1d(_helical(mu, delta=0.3))  # phase
    assert _zak_parity(h) == 1  # the two Kramers copies add up


@pytest.mark.parametrize("rashba", [0., 0.2])
def test_s_wave_pairing_drives_the_wire_trivial(rashba, tmp_path,
                                                monkeypatch):
    """With Rashba coupling the spin is no longer conserved, and an s-wave
    pairing competing with the p-wave one closes the gap (near 0.97 without
    Rashba coupling and 0.89 with it) and makes the wire trivial"""
    monkeypatch.chdir(tmp_path)
    for (ds, nu) in [(0.2, -1), (0.6, -1), (1.3, 1), (1.6, 1)]:
        h = _helical(0.5, delta=0.5j, rashba=rashba, ds=ds)
        assert topology.z2_invariant_1d(h) == nu, ds


def test_z2_1d_needs_time_reversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    h = _helical(0.5)
    h.add_zeeman([0., 0., 0.1])
    with pytest.raises(ValueError, match="time-reversal"):
        topology.z2_invariant_1d(h)
    with pytest.raises(ValueError, match="Nambu"):
        topology.z2_invariant_1d(geometry.chain().get_hamiltonian())


def test_pfaffian(tmp_path, monkeypatch):
    """Pf(A)^2 = det(A), and Pf(B^T A B) = det(B) Pf(A), which fixes the
    sign as well"""
    from pyqula.topologytk.pfaffian import pfaffian
    rng = np.random.default_rng(1)
    for n in [2, 4, 8]:
        C = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
        A = C - C.T
        B = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
        assert np.isclose(pfaffian(A)**2, np.linalg.det(A))
        assert np.isclose(pfaffian(B.T@A@B), np.linalg.det(B)*pfaffian(A))
