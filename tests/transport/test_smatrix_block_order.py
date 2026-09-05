import numpy as np

from pyqula import geometry, heterostructures

ENERGY = 0.35


def _asymmetric_junction():
    """A two-channel ribbon junction whose central region is asymmetric,
    so s[0][1] and s[1][0] are genuinely different matrices and swapping
    them is observable."""
    g = geometry.square_ribbon(2)
    hl = g.get_hamiltonian(has_spin=False)
    hr = g.get_hamiltonian(has_spin=False)
    c1 = g.get_hamiltonian(has_spin=False)
    c1.add_onsite(np.array([0.6, -0.3]))
    c2 = g.get_hamiltonian(has_spin=False)
    c2.add_onsite(np.array([-0.2, 0.5]))
    return heterostructures.build(hl, hr, central=[c1, c2])


def test_unitarization_preserves_the_block_order():
    """check_and_fix flattens the S-matrix with bmat and splits it back.
    The split used to read the off-diagonal blocks transposed --
    sout[0][1] got s3[n:2n,0:n], which is block [1][0] -- so get_smatrix,
    whose default is check=True, returned the two transmission blocks
    interchanged. Unitarization is a small correction, so every block must
    still match the unchecked one closely."""
    ht = _asymmetric_junction()
    checked = heterostructures.get_smatrix(ht, energy=ENERGY, check=True)
    raw = heterostructures.get_smatrix(ht, energy=ENERGY, check=False)
    # the two off-diagonal blocks must actually differ, or the test is vacuous
    asymmetry = np.max(np.abs(np.array(raw[0][1]) - np.array(raw[1][0])))
    assert asymmetry > 1e-2
    for i in range(2):
        for j in range(2):
            d = np.max(np.abs(np.array(checked[i][j]) - np.array(raw[i][j])))
            assert d < 1e-2 * asymmetry, (i, j, d)


def test_the_checked_smatrix_is_unitary():
    """What check_and_fix is for, and the property the reassembly must not
    destroy."""
    ht = _asymmetric_junction()
    s = heterostructures.get_smatrix(ht, energy=ENERGY, check=True)
    m = np.block([[np.array(s[0][0]), np.array(s[0][1])],
                  [np.array(s[1][0]), np.array(s[1][1])]])
    n = m.shape[0]
    assert np.max(np.abs(m @ np.conjugate(m).T - np.identity(n))) < 1e-10
