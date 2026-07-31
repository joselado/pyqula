import numpy as np
import pytest

from pyqula import geometry
from pyqula import embedding


def test_mismatched_shape_raises_value_error():
    """Embedding used to hit a bare `raise` (RuntimeError: No active
    exception) for a mismatched onsite matrix; it must raise a real,
    informative ValueError instead."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    hs = h.supercell(2)
    with pytest.raises(ValueError):
        embedding.Embedding(h, m=hs.intra.copy())


def test_supercell_onsite_matches_hand_built_supercell():
    """Embedding(h, m=<supercell onsite matrix>, nsuper=N) is a shortcut for
    building h.supercell(N) by hand and embedding into it directly -- the two
    must agree exactly (bit for bit), not just approximately."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)

    nsuper = 2
    defect = h.intra.copy()
    defect[0, 0] = 7.0
    ms_defective = embedding.onsite_defective_central(h, defect, nsuper)

    hs = h.supercell(nsuper)
    eb_shortcut = embedding.Embedding(h, m=ms_defective, nsuper=nsuper)
    eb_by_hand = embedding.Embedding(hs, m=ms_defective)

    for outer in [1, 2]:
        d1 = eb_shortcut.get_dos(energy=0.2, delta=1e-2, nsuper=outer, nk=20)
        d2 = eb_by_hand.get_dos(energy=0.2, delta=1e-2, nsuper=outer, nk=20)
        assert d1 == d2


def test_supercell_onsite_is_physically_consistent_with_plain_defect():
    """A single-cell defect embedded through the plain (same-shape) method,
    with the periodic embedding built at nsuper=N, should describe the same
    physical system as giving that same defect pre-placed in the central
    cell of an N x N supercell onsite matrix and going through the
    nsuper=N shortcut -- so the two must converge to the same DOS as nk
    grows, even though they take different numerical code paths
    (dyson zone-folding vs. direct renormalization)."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)

    nsuper = 2
    defect = h.intra.copy()
    defect[0, 0] = 5.0
    ms_defective = embedding.onsite_defective_central(h, defect, nsuper)

    eb_ref = embedding.Embedding(h, m=defect)
    eb_new = embedding.Embedding(h, m=ms_defective, nsuper=nsuper)

    nk = 160
    dos_ref = eb_ref.get_dos(energy=0.3, delta=1e-2, nsuper=nsuper, nk=nk)
    dos_new = eb_new.get_dos(energy=0.3, delta=1e-2, nsuper=1, nk=nk * nsuper)

    assert np.isclose(dos_ref, dos_new, atol=2e-2)
