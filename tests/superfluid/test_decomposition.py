"""Conventional/quantum-geometric decomposition of the superfluid weight.

Two things must hold: the split must add back up to the general Kubo
result (it is a decomposition, not an independent calculation), and it
must *refuse* to report a split when the assumptions that give it meaning
do not hold, rather than silently returning numbers."""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.sctk import superfluidweight as sw


def _uniform_swave_models():
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_swave(0.5)
    yield "square", h
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_onsite(0.5)
    h.add_swave(0.4)
    yield "honeycomb", h
    h = geometry.chain().get_hamiltonian()
    h.add_onsite(-0.5)
    h.add_swave(0.3)
    yield "chain", h


@pytest.mark.parametrize("T", [0.0, 0.12])
def test_conventional_plus_geometric_equals_the_kubo_weight(T):
    """The split is exact k-point by k-point, so it must reproduce the
    general formula to machine precision on the very same mesh."""
    for (name, h) in _uniform_swave_models():
        out = sw.superfluid_weight_decomposition(h, nk=10, T=T)
        tot = sw.superfluid_weight(h, nk=10, T=T)
        scale = max(np.max(np.abs(tot)), 1e-12)
        assert np.max(np.abs(out["total"]-tot))/scale < 1e-10, (name, out, tot)
        assert np.allclose(out["total"], out["conventional"]+out["geometric"])


def test_single_orbital_model_is_purely_conventional():
    """With one orbital per cell (per spin) there are no interband matrix
    elements at all, so the geometric part must vanish identically."""
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_swave(0.5)
    out = sw.superfluid_weight_decomposition(h, nk=10, T=0.)
    assert np.max(np.abs(out["geometric"])) < 1e-12, out["geometric"]
    assert out["conventional"][0, 0] > 0.1


def test_conventional_part_matches_the_liang_closed_form():
    """The implemented conventional part and Liang et al. Eq. (21) differ by
    a BZ integration by parts, so they must agree once the mesh is dense
    enough -- an independent check of the split, since the closed form uses
    neither the diamagnetic term nor any interband denominator."""
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_swave(0.5)
    for T in [0., 0.12]:
        a = sw.superfluid_weight_decomposition(h, nk=60, T=T)["conventional"]
        b = sw.superfluid_weight_conventional_closed(h, nk=60, T=T)
        assert np.max(np.abs(a-b))/np.max(np.abs(a)) < 5e-3, (T, a, b)


def test_decomposition_refuses_non_uniform_pairing():
    """A p-wave (non-local, non-uniform) gap has no conventional/geometric
    split in this sense; the general weight is still available."""
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_pairing(mode="pwave", delta=0.3, d=[0., 0., 1.])
    h.add_swave(0.0)
    with pytest.raises(ValueError, match="uniform on-site pairing"):
        sw.superfluid_weight_decomposition(h, nk=6, T=0.)
    assert sw.superfluid_weight(h, nk=6, T=0.)[0, 0] > 0.


def test_decomposition_refuses_broken_time_reversal_symmetry():
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_zeeman([0., 0., 0.3])
    h.add_swave(0.3)
    with pytest.raises(ValueError, match="time-reversal symmetry"):
        sw.superfluid_weight_decomposition(h, nk=6, T=0.)


def test_decomposition_refuses_degenerate_bands_with_interband_current():
    """Rashba coupling leaves a Kramers degeneracy at the time-reversal
    invariant momenta with a finite interband current: the interband
    denominators of the split blow up there, and that must raise rather
    than produce a huge meaningless number."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_onsite(0.4)
    h.add_rashba(0.3)
    h.add_swave(0.3)
    with pytest.raises(ValueError, match="degenerate normal-state bands"):
        sw.superfluid_weight_decomposition(h, nk=6, T=0.)


def test_public_api_dispatch():
    """h.get_superfluid_weight / superfluid.superfluid_weight wiring"""
    from pyqula import superfluid
    h = geometry.square_lattice().get_hamiltonian()
    h.add_onsite(-0.7)
    h.add_swave(0.5)
    a = h.get_superfluid_weight(nk=6, T=0.)
    b = superfluid.superfluid_weight(h, nk=6, T=0.)
    c = superfluid.superfluid_weight(h, nk=6, T=0., mode="finite_difference",
                                     dQ=3e-4)
    d = h.get_superfluid_weight(nk=6, T=0., decompose=True)
    assert np.allclose(a, b)
    assert np.max(np.abs(a-c))/np.max(np.abs(a)) < 1e-3
    assert np.allclose(d["total"], a)


def test_non_nambu_hamiltonian_is_rejected():
    h = geometry.square_lattice().get_hamiltonian()
    with pytest.raises(ValueError, match="Nambu"):
        sw.superfluid_weight(h, nk=4, T=0.)
