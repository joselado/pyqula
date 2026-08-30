"""`get_index` used to reject index 0, reporting the first site of the cell
as "not found"."""
import numpy as np

from pyqula import geometry


def test_every_site_of_the_cell_has_an_index():
    """get_index_jit signals "not here" with -1, so 0 is a valid answer.
    The caller tested `out>0`, so the first site always came back None."""
    for g in [geometry.honeycomb_lattice(), geometry.kagome_lattice(),
              geometry.chain().get_supercell(3)]:
        idx = [g.get_index(r, replicas=True) for r in g.r]
        assert idx == list(range(len(g.r))), idx
        idx = [g.get_index(r, replicas=False) for r in g.r]
        assert idx == list(range(len(g.r))), idx


def test_sublattice_resolved_pairing_covers_both_sublattices():
    """swaveA/swaveB place an s-wave amplitude on one sublattice only. They
    resolve the site through get_index, so the site at index 0 was read as
    None and skipped: swaveA (whose sublattice is exactly that site on a
    honeycomb cell) was identically zero everywhere, and swaveB covered
    only half of what it should."""
    # go through the Hamiltonian, so that pyqula.sctk.pairing is reached
    # the way the library reaches it (importing it first is circular)
    geometry.honeycomb_lattice().get_hamiltonian()
    from pyqula.sctk.pairing import swaveA, swaveB
    g = geometry.honeycomb_lattice()
    a = [np.max(np.abs(swaveA(g, r, r))) for r in g.r]
    b = [np.max(np.abs(swaveB(g, r, r))) for r in g.r]
    # each site carries exactly one of the two, and neither is empty
    assert sum(a) > 0 and sum(b) > 0
    for (ai, bi) in zip(a, b):
        assert abs(ai + bi - 1.0) < 1e-12, (a, b)
    # and they select opposite sublattices
    assert not np.allclose(a, b)
