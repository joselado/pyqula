"""`h.get_supercell([2,2])` died with a bare IndexError from deep inside the
supercell builder, while `g.get_supercell([2,2])` -- which the builder itself
calls one line earlier -- accepted the same argument.  Only a *scalar* nsuper
was padded to three components.

The opposite mistake was silent: `[2,2,2]` on a 2d Hamiltonian gave a geometry
of 4 cells next to matrices sized for 8, because geometry.get_supercell reads
only the first `dimensionality` components while
multicell.supercell_hamiltonian sizes the blocks by n1*n2*n3.

The oracle throughout is the geometry: the Hamiltonian must accept what its
own geometry accepts, and the two must come out the same size."""
import numpy as np
import pytest

from pyqula import geometry


def test_a_short_sequence_is_padded_like_the_geometry_pads_it():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    for ns in [[2, 2], (2, 2), np.array([2, 2])]:
        hs = h.get_supercell(ns)
        assert len(hs.geometry.r) == len(g.get_supercell(ns).r)
        assert hs.intra.shape[0] == 2*len(hs.geometry.r)  # spinful


def test_a_sequence_and_the_equivalent_scalar_give_the_same_hamiltonian():
    """h.get_supercell(2) already worked; [2,2] has to mean the same thing."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    ha = h.get_supercell(2)
    hb = h.get_supercell([2, 2])
    assert ha.intra.shape == hb.intra.shape
    assert ha.same_hamiltonian(hb)


def test_a_one_component_sequence_works_on_a_chain():
    g = geometry.chain()
    h = g.get_hamiltonian()
    hs = h.get_supercell([3])
    assert len(hs.geometry.r) == len(g.get_supercell([3]).r) == 3
    assert hs.intra.shape[0] == 2*3


def test_repetitions_along_a_direction_the_lattice_does_not_have_are_refused():
    """[2,2,2] on a 2d lattice used to build a Hamiltonian whose matrices
    were twice the size of its own geometry, and nothing downstream noticed."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError, match="dimensional"):
        h.get_supercell([2, 2, 2])
    h1 = geometry.chain().get_hamiltonian()
    with pytest.raises(ValueError, match="dimensional"):
        h1.get_supercell([2, 2])


def test_more_than_three_components_are_refused():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    with pytest.raises(ValueError, match="three"):
        h.get_supercell([2, 2, 1, 1])


def test_a_supercell_matrix_is_refused_with_a_message_naming_the_geometry():
    """geometry.get_supercell accepts a 3x3 matrix (a non-orthogonal
    supercell); the Hamiltonian builder cannot, and used to fail with a
    TypeError about scalar array indices."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    m = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    with pytest.raises(NotImplementedError, match="get_supercell"):
        h.get_supercell(m)


def test_the_padded_supercell_reproduces_the_primitive_spectrum():
    """A supercell is a relabelling of the same model, so the set of band
    energies over the folded Brillouin zone must match the primitive one.
    Checked at the zone centre, where folding maps every k of the primitive
    cell commensurate with the supercell onto Gamma."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    hs = h.get_supercell([2, 2]).get_dense()  # the builder returns sparse
    hk0 = h.get_hk_gen()
    hks = hs.get_hk_gen()
    es = np.sort(np.linalg.eigvalsh(np.array(hks([0., 0., 0.]))))
    e0 = []
    for k in [[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.], [.5, .5, 0.]]:
        e0 += list(np.linalg.eigvalsh(np.array(hk0(k))))
    assert np.allclose(es, np.sort(e0), atol=1e-8), (es, np.sort(e0))
