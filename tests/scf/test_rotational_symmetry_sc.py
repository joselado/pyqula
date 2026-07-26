import numpy as np

from pyqula import geometry
from pyqula import hamiltonians  # noqa: F401 -- resolve sctk.dvector's import order
from pyqula.sctk.dvector import matrix2dvector
from testutils import SCF_MAXERROR


def _mean_field_for_random_direction(h0, v):
    h1 = h0.copy()
    h1.add_exchange(v)
    h1.turn_nambu()
    return h1.get_mean_field_hamiltonian(nk=20, mf="random", V1=-2.,
                                          filling=.3,
                                          maxerror=SCF_MAXERROR,
                                          return_total_energy=True)


def _gap_for_random_direction(h0):
    v = np.random.random(3) - .5  # random Zeeman direction
    v = 4 * v / np.sqrt(v.dot(v))  # normalize
    h, etot = _mean_field_for_random_direction(h0, v)
    return h.get_gap()


def test_superconducting_gap_is_rotationally_invariant():
    """The self-consistent superconducting gap must not depend on the
    direction of the (arbitrary) Zeeman field used to seed the SCF loop."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    gaps = np.array([_gap_for_random_direction(h0) for _ in range(6)])
    diff = gaps - np.mean(gaps)
    assert np.max(np.abs(diff)) < SCF_MAXERROR * 10, \
        f"SCF gap is not rotationally invariant: {diff}"


def test_triplet_dvector_is_perpendicular_to_zeeman_direction():
    """A Zeeman field along an arbitrary direction v induces equal-spin
    triplet pairing quantized along v (Delta_ud, the Sz=0 pairing channel
    measured along v, is suppressed). In the d-vector representation
    Delta = i(d.sigma)sigma_y, this channel IS the component of d along v
    (dvector.delta2dvector sets dz = Delta_ud), so the physically correct
    signature of rotational symmetry here is that d is exactly
    PERPENDICULAR to v for any v -- not parallel to it. Checked as
    |v.d|/|d| ~ 0 at several k-points, for several random v."""
    g = geometry.bichain()
    h0 = g.get_hamiltonian()
    for _ in range(4):
        v = np.random.random(3) - .5
        v = 4 * v / np.sqrt(v.dot(v))
        h, etot = _mean_field_for_random_direction(h0, v)
        hk = h.get_hk_gen()
        for kx in np.linspace(0.05, 0.45, 5):
            m = hk([kx, 0., 0.])
            d = matrix2dvector(m) # (3, nsites, nsites) complex d-vector matrices
            dsum = np.array([d[0].sum(), d[1].sum(), d[2].sum()])
            dnorm = np.linalg.norm(dsum)
            if dnorm < 1e-8: continue # negligible triplet weight at this k
            ratio = abs(np.dot(v, dsum))/dnorm
            assert ratio < 1e-3, \
                f"d-vector not perpendicular to Zeeman direction {v} " \
                f"at k={kx}: |v.d|/|d| = {ratio}"
