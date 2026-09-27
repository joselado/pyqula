"""set_filling on a non-Hermitian Hamiltonian cuts the real parts of its
complex eigenvalues, as the mean field does (spectrum.get_fermi4filling,
densitymatrix.biorthogonal_dm). It used to take the eigenvalues of a
Hermitian eigensolver, which reads only the lower triangle and the real
part of the diagonal of H: an onsite gain and loss was dropped and a
non-reciprocal hopping replaced by one of its two directions."""
import numpy as np

from pyqula import geometry


def _chain():
    """Open spinless chain with a non-reciprocal hopping, a disordered
    onsite energy and onsite gain and loss"""
    g = geometry.chain(12)
    g.dimensionality = 0

    def tij(r1, r2):
        dx = r2[0] - r1[0]
        if abs(dx - 1.) < 1e-6: return 0.6
        if abs(dx + 1.) < 1e-6: return 1.4
        return 0.
    h = g.get_hamiltonian(tij=tij, has_spin=False, non_hermitian=True)
    rng = np.random.default_rng(1)
    onsite = rng.uniform(-1., 1., 12) + 1j*rng.uniform(-0.5, 0.5, 12)
    h.add_onsite(lambda r: onsite[int(round(r[0] - g.r[0][0]))])
    return h


def test_set_filling_cuts_the_real_parts_of_the_spectrum():
    for filling in [0.25, 0.5, 0.75]:
        h = _chain()
        h.set_filling(filling, nk=1)
        es = np.sort(np.linalg.eigvals(np.array(h.intra)).real)
        n = int(round(len(es)*filling))
        # the Fermi energy, now at zero, sits halfway between the last
        # occupied and the first empty state, counted by Re E
        assert abs((es[n-1] + es[n])/2.) < 1e-10, (filling, es)
        # and the mean field's own Fermi search agrees
        assert abs(h.get_fermi4filling(filling, nk=1)) < 1e-10
