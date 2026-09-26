"""Every supercell builder records which primal replica each of its atoms
came from.

unfolding.bloch_projector consumes that record. It used to be written only
by the general (matrix) builder, and it then travelled with the geometry
through Geometry.copy() without any of the diagonal builders replacing it,
so a supercell of a supercell carried a replica array describing a
different set of atoms.
"""

import numpy as np
import pytest

from pyqula import geometry
from pyqula.supercell import infer_supercell


@pytest.mark.parametrize("name,nsuper", [
    ("chain", 3),
    ("bichain", 4),
    ("honeycomb_lattice", [3, 2]),
    ("square_lattice", [2, 3]),
    ("kagome_lattice", [2, 2]),
    ("cubic_lattice", [2, 3, 2]),
    ("diamond_lattice_minimal", [2, 2, 2]),
])
def test_diagonal_builders_record_a_consistent_replica_map(name, nsuper):
    """Every atom must sit at r0[primal] + n@A0, up to the rigid shift the
    builders apply when they center the cell"""
    from pyqula.unfolding import describes_supercell
    g0 = getattr(geometry, name)()
    g = g0.get_supercell(nsuper)
    assert len(g.supercell_replica) == len(g.r)
    assert len(g.supercell_primal_index) == len(g.r)
    ns = np.array(nsuper if np.ndim(nsuper) else [nsuper])
    expected = np.diag(list(ns) + [1] * (3 - len(ns)))
    assert np.all(g.supercell_matrix == expected)
    assert describes_supercell(g, g0, g.supercell_replica,
                               g.supercell_primal_index)


def test_a_supercell_of_a_supercell_replaces_the_record():
    """The stale record from the first (matrix) supercell must not survive
    into the second one, where it describes a different atom count"""
    g0 = geometry.honeycomb_lattice()
    g1 = g0.get_supercell([[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    g2 = g1.get_supercell([2, 2, 1])
    from pyqula.unfolding import describes_supercell
    assert len(g2.supercell_replica) == len(g2.r)
    assert np.all(g2.supercell_matrix == np.diag([2, 2, 1]))
    assert describes_supercell(g2, g1, g2.supercell_replica,
                               g2.supercell_primal_index)
    # and it must not claim to describe g2 in terms of the original primal
    assert not describes_supercell(g2, g0, g2.supercell_replica,
                                   g2.supercell_primal_index)


def test_removing_atoms_keeps_the_record_in_step():
    g0 = geometry.honeycomb_lattice()
    g = g0.get_supercell([3, 3])
    kept = g.remove([2, 7, 11])
    from pyqula.unfolding import describes_supercell
    assert len(kept.supercell_replica) == len(kept.r)
    assert describes_supercell(kept, g0, kept.supercell_replica,
                               kept.supercell_primal_index)


@pytest.mark.parametrize("a", [1.0, 0.01303157231604361, 422.19041157635354])
def test_infer_supercell_rounds_rather_than_truncating(a):
    """The ratio of two norms lands one ulp below the integer often enough
    that int() would silently return n-1 (a=0.013... and a=422.19... both
    do exactly that for n=3, 6 and 12)"""
    g0 = geometry.chain()
    g0.a1 = g0.a1 * a ; g0.r = g0.r * a ; g0.r2xyz()
    for n in [2, 3, 4, 5, 6, 7, 12]:
        g = g0.get_supercell(n)
        assert infer_supercell(g, g0) == (n, 1, 1)


def test_store_primal_leaves_the_geometry_it_was_built_from_alone():
    """store_primal used to set the primal copy on the geometry the
    supercell was built from, so that every later supercell of it carried
    a primal geometry whether it asked for one or not"""
    import numpy as np
    from pyqula import geometry
    g0 = geometry.triangular_lattice()
    g = g0.get_supercell(2, store_primal=True)
    assert g0.primal_geometry is None
    assert g.primal_geometry is not None
    assert g0.get_supercell(3).primal_geometry is None
    # the float size rotates the cell, and the primal cell with it
    gs = g0.get_supercell(np.sqrt(3), store_primal=True)
    assert g0.primal_geometry is None
    assert not np.allclose(gs.primal_geometry.a1, g0.a1)
    h = gs.get_hamiltonian(has_spin=False)
    (k, e, d) = h.get_bands(operator="unfold", kpath=[[0.1, 0.2, 0.]],
                            write=False)
    assert np.isclose(np.max(d), 3.)  # a whole band unfolds with weight 3
