import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk import localspinchi, rpa, spinchi
from pyqula.spinon import SpinonHamiltonian

# chitk.localspinchi builds the response of the local spin operators from
# the amplitudes of the Bloch states on each site, instead of transforming
# one dense matrix per site and component as chitk.chiAB does. It is the
# same sum, so the two must agree to roundoff, for any spinful Hamiltonian
# and on the faster route it takes for a spin-degenerate one.

ES = np.linspace(-1., 3., 7)


def hamiltonian(spin_dependent):
    g = geometry.honeycomb_lattice().get_supercell((2, 1))
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(0.3) # away from half filling
    if spin_dependent: # a generic direction, and a term that mixes spins
        h.add_zeeman([0.2, -0.1, 0.3])
        h.add_rashba(0.2)
    return h


def test_pairs_of_zero_factor_are_left_out_on_the_real_axis():
    """With delta=0 a pair of states with the same occupation has a zero
    factor and an energy difference that a real frequency can hit, which
    gave 0/0. At q=0 every state paired with itself has de=0, so w=0 hits
    all of them; the result is finite there and is the limit w -> 0 of the
    response, since the dropped pairs contribute nothing at any w"""
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=True)
    h.add_sublattice_imbalance(0.6) # gapped at half filling
    h.add_zeeman([0.1, -0.2, 0.3]) # a generic direction
    q = [0., 0., 0.]
    with np.errstate(all="ignore"): # a wrong code gives 0/0
        out = localspinchi.bare_spin_chi(h, q=q, energies=[0., 1e-7], nk=4,
                                         delta=0., T=1e-3)[1]
    assert np.all(np.isfinite(out[0]))
    assert np.max(np.abs(out[0])) > 0.05
    assert np.max(np.abs(out[0] - out[1])) < 1e-5


@pytest.mark.parametrize("spin_dependent", [False, True])
@pytest.mark.parametrize("q", [[0.13, 0.31, 0.], [0., 0., 0.], [0.5, 0., 0.]])
@pytest.mark.parametrize("T", [None, 0.3])
def test_bare_response_agrees_with_chiAB(spin_dependent, q, T):
    h = hamiltonian(spin_dependent)
    assert localspinchi.is_spin_degenerate(h) == (not spin_dependent)
    kw = dict(q=q, energies=ES, delta=0.05, nk=6, T=T)
    ref = rpa.chi_ops_RPA(h, ops=spinchi._full_spin_operators(h), V=None,
                          **kw)[1]
    out = localspinchi.bare_spin_chi(h, **kw)[1]
    assert np.max(np.abs(ref)) > 0.1
    assert np.allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize("spin_dependent", [False, True])
def test_dressed_response_agrees_with_chi_ops_RPA(spin_dependent):
    h = hamiltonian(spin_dependent)
    n = 3*len(h.geometry.r)
    rng = np.random.default_rng(0)
    def vertex(): return 0.3*rng.normal(size=(n, n))
    v0, v1 = vertex(), vertex()
    V = {(0, 0, 0): v0 + v0.T, (1, 0, 0): v1, (-1, 0, 0): v1.T}
    kw = dict(q=[0.13, 0.31, 0.], energies=ES, delta=0.05, nk=6)
    ref = rpa.chi_ops_RPA(h, ops=spinchi._full_spin_operators(h), V=V,
                          **kw)[1]
    out = localspinchi.spin_chi(h, V=V, **kw)[1]
    assert np.allclose(out, ref, atol=1e-11)


def test_local_response_is_the_average_over_the_mesh():
    h = hamiltonian(True)
    kw = dict(energies=ES, delta=0.05, nk=3)
    out = localspinchi.bare_spin_chi(h, **kw)[1]
    ref = np.mean([localspinchi.bare_spin_chi(h, q=q, **kw)[1]
                   for q in h.geometry.get_kmesh(nk=3)], axis=0)
    assert np.allclose(out, ref, atol=1e-13)


def test_static_kernel_is_one_without_vertex_and_guards():
    h = hamiltonian(False)
    n = 3*len(h.geometry.r)
    V = {(0, 0, 0): np.zeros((n, n))}
    ev = localspinchi.static_kernel_eigenvalues(h, V, [0.2, 0.1, 0.], nk=4)
    assert np.allclose(ev, 1.)
    with pytest.raises(ValueError, match="q=None"):
        localspinchi.static_kernel_eigenvalues(h, V, None)
    hs = geometry.chain().get_hamiltonian(has_spin=False)
    with pytest.raises(ValueError):
        localspinchi.bare_spin_chi(hs, q=[0.1, 0., 0.])
    hn = geometry.chain().get_hamiltonian(has_spin=True)
    hn.setup_nambu_spinor()
    with pytest.raises(NotImplementedError, match="Nambu"):
        localspinchi.bare_spin_chi(hn, q=[0.1, 0., 0.])


def test_spinon_bare_response_takes_the_same_numbers():
    """SpinonHamiltonian.get_spinchi_full(RPA=False) uses this bubble; it
    is the response the Hamiltonian base class gives"""
    from pyqula.hamiltonians import Hamiltonian
    np.random.seed(0)
    h = SpinonHamiltonian(geometry.chain()).get_mean_field_hamiltonian(
            J1=1.0, nk=24, mix=0.3, maxerror=1e-5, maxite=2000)
    kw = dict(q=[0.2, 0., 0.], energies=np.linspace(0., 1., 5), delta=0.05,
              nk=40, RPA=False)
    out = h.get_spinchi_full(**kw)[1]
    ref = Hamiltonian.get_spinchi_full(h, **kw)[1]
    assert np.max(np.abs(ref)) > 0.1
    assert np.allclose(out, ref, atol=1e-12)
