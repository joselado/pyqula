"""BKT temperature, and the superfluid weight of a self-consistent
attractive-Hubbard superconductor."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.sctk import superfluidweight as sw


def _square_swave(mu=-0.7, delta=0.4):
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(mu)
    h.add_swave(delta)
    return h


def test_bkt_temperature_satisfies_its_defining_self_consistency():
    """T_BKT = (pi/8) D_s(T_BKT) is the definition, so the returned
    temperature must satisfy it -- evaluated with the same mesh."""
    h = _square_swave()
    nk = 12
    tb = sw.bkt_temperature(h, nk=nk)
    assert tb > 0.
    d = sw.superfluid_weight(h, nk=nk, T=tb)
    rhs = np.pi/8.*np.sqrt(np.linalg.det(d))
    assert abs(tb-rhs) < 1e-5*tb, (tb, rhs)


def test_bkt_temperature_is_below_the_zero_temperature_bound():
    """The stiffness only decreases with temperature, so the solution of
    T = (pi/8) D_s(T) must lie below (pi/8) D_s(0)."""
    h = _square_swave()
    nk = 12
    tb = sw.bkt_temperature(h, nk=nk)
    d0 = sw.superfluid_weight(h, nk=nk, T=0.)
    assert 0. < tb < np.pi/8.*np.sqrt(np.linalg.det(d0))


def test_bkt_temperature_grows_with_the_gap():
    h1 = _square_swave(delta=0.3)
    h2 = _square_swave(delta=0.6)
    assert sw.bkt_temperature(h1, nk=10) < sw.bkt_temperature(h2, nk=10)


def test_bkt_temperature_requires_two_dimensions():
    h = geometry.chain().get_hamiltonian()
    h.add_swave(0.3)
    with pytest.raises(NotImplementedError):
        sw.bkt_temperature(h, nk=10)


def test_selfconsistent_attractive_hubbard_has_a_positive_weight(
        tmp_path, monkeypatch):
    """End to end: an SCF BdG Hamiltonian must have a positive superfluid
    weight that the finite-difference oracle agrees with."""
    monkeypatch.chdir(tmp_path)   # the SCF caches MF.pkl in the cwd
    h = geometry.square_lattice().get_hamiltonian()
    h.turn_nambu()
    h2 = h.get_mean_field_hamiltonian(U=-2.0, filling=0.35, mf="swave",
                                      nk=8, mix=0.8, maxerror=1e-6)
    assert h2 is not None, "the mean field did not converge"
    delta = abs(h2.extract("swave")[0])
    assert delta > 1e-2, delta
    d = sw.superfluid_weight(h2, nk=12, T=0.)
    f = sw.superfluid_weight_finite_difference(h2, nk=12, T=0., dQ=3e-4)
    assert np.min(np.linalg.eigvalsh(d)) > 0.
    assert np.max(np.abs(d-f))/np.max(np.abs(d)) < 1e-3, (d, f)
    # one orbital per cell: the weight is entirely conventional
    out = sw.superfluid_weight_decomposition(h2, nk=12, T=0.)
    assert np.max(np.abs(out["geometric"])) < 1e-10
    assert sw.bkt_temperature(h2, nk=10) > 0.
