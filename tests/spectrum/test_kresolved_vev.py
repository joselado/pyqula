"""Expectation values of an operator applied inside the sum over kpoints.

spectrum.ev contracts an operator matrix with densitymatrix.full_dm, which
has already summed over the Brillouin zone. An operator that acts with a
different matrix at every kpoint (the unfolding projector) cannot be used
that way at all, and one defined only by its action on a wavefunction has
no matrix to contract. Those go through vev.kresolved_orbital_vev instead,
which must agree with spectrum.ev wherever both apply.
"""

import numpy as np
import pytest

from pyqula import geometry
from pyqula.operators import Operator


def blind_copy(op):
    """The same operator, with its matrix representation hidden, so it has
    to take the kpoint-resolved route"""
    m = op.get_matrix()
    out = Operator(lambda v, k=None: m @ v)
    assert out.matrix is None
    return out


@pytest.mark.parametrize("name,has_spin,nambu", [
    ("chain", False, False),
    ("chain", True, False),
    ("honeycomb_lattice", True, False),
    ("square_lattice", True, True),
])
def test_kresolved_vev_agrees_with_the_density_matrix(name, has_spin, nambu):
    g = getattr(geometry, name)()
    h = g.get_supercell(2).get_hamiltonian(has_spin=has_spin)
    if nambu: h.setup_nambu_spinor()
    if has_spin: h.add_zeeman([0., 0., 0.3])
    op = h.get_operator("sz" if has_spin else "xposition")
    direct = h.get_vev(operator=op, nk=6)
    kresolved = h.get_vev(operator=blind_copy(op), nk=6)
    assert np.allclose(direct, kresolved, atol=1e-10)


def test_get_single_vev_agrees_with_the_density_matrix():
    g = geometry.honeycomb_lattice()
    h = g.get_supercell(2).get_hamiltonian(has_spin=True)
    h.add_zeeman([0., 0., 0.4])
    op = h.get_operator("sz")
    direct = h.get_single_vev(op, nk=6)
    kresolved = h.get_single_vev(blind_copy(op), nk=6)
    assert np.allclose(direct, kresolved, atol=1e-10)


def test_get_several_vev_mixes_both_kinds_of_operator():
    g = geometry.honeycomb_lattice()
    h = g.get_supercell(2).get_hamiltonian(has_spin=True)
    h.add_zeeman([0., 0., 0.4])
    ops = [h.get_operator("sz"), h.get_operator("sx")]
    direct = h.get_several_vev(ops, nk=6)
    mixed = h.get_several_vev([ops[0], blind_copy(ops[1])], nk=6)
    assert np.allclose(np.ravel(direct), np.ravel(mixed), atol=1e-10)


def test_unfolding_vev_of_a_clean_supercell_is_uniform():
    """Every site of a defect-free supercell is equivalent, so the unfolded
    weight has to be the same on all of them. Its sum is the number of
    occupied primal Bloch states times the replica count: at half filling
    one of the two honeycomb bands is occupied at each kpoint and carries
    weight |det M| = 4."""
    g0 = geometry.honeycomb_lattice()
    g = g0.get_supercell(2, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    v = h.get_vev(operator="unfold", nk=8)
    assert len(v) == len(g.r)
    assert np.allclose(v, v[0], atol=1e-6)
    assert np.isclose(np.sum(v), 4.0, atol=1e-6)


def test_unfolding_vev_empties_the_defect_site():
    """A site pushed far above the Fermi level holds no weight, and the
    weight it lost shows up on its neighbors"""
    g0 = geometry.honeycomb_lattice()
    g = g0.get_supercell(2, store_primal=True)
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: 100.0 if np.sum((r - g.r[0]) ** 2) < 1e-2 else 0.0)
    v = h.get_vev(operator="unfold", nk=8)
    assert np.isclose(v[0], 0.0, atol=1e-2)
    assert np.max(v) > 0.6


def test_kresolved_vev_refuses_two_names_for_the_smearing():
    from pyqula.vev import kresolved_orbital_vev
    g = geometry.chain()
    h = g.get_supercell(2).get_hamiltonian(has_spin=True)
    with pytest.raises(TypeError):
        kresolved_orbital_vev(h, blind_copy(h.get_operator("sz")),
                              nk=2, T=1e-3, delta=1e-5)
