import numpy as np
import pytest

from pyqula import geometry


def _zigzag_scf_magnetization(U, n=4, nk=20):
    """Self-consistent local moments of a Hubbard zigzag ribbon, as vectors
    per site together with the transverse coordinate of each site."""
    g = geometry.honeycomb_zigzag_ribbon(n)
    h = g.get_hamiltonian()
    h = h.get_mean_field_hamiltonian(U=U, nk=nk)
    return np.array(h.get_magnetization()), h.geometry.r[:, 1]


@pytest.mark.slow
def test_zigzag_ribbon_hubbard_scf_magnetises_only_the_edges(tmp_path,
                                                             monkeypatch):
    """The flat zero-energy band of a zigzag graphene ribbon lives on the
    two edges, so an arbitrarily small Hubbard U magnetises them, and the
    textbook result is that the moment is edge-localised and decays fast
    into the bulk. At U=1 the edge moment is 0.26 against a bulk maximum of
    0.053, a factor of five. At U=3 the whole ribbon orders instead (edge
    0.63 against bulk 0.46, a factor of 1.4) and at U=0.01 the moment
    collapses to 0.14.

    The old assertion was sum(e) over the band structure, which is
    sum_k Tr H(k). The Hubbard mean field shifts every site's onsite energy
    by U*n_i, and the trace is blind to how those shifts are distributed --
    it is the same for a magnetic and a non-magnetic solution, for any U and
    any ribbon width.

    The test asserts the |m| *profile* and not the direction of any moment.
    The mean field is SU(2) symmetric, so the overall direction is set by
    the random initial guess; and at U=1 the antiparallel and parallel
    arrangements of the two edge moments are close enough in energy that the
    loop lands on the parallel one about once in ten runs. The magnitude
    profile is the same for both, to three digits. Marked slow: the SCF
    convergence drives the runtime."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT and MF.pkl to cwd
    U = 1.0
    (mag, y) = _zigzag_scf_magnetization(U)
    m = np.linalg.norm(mag, axis=1)
    edge = np.abs(y) > np.max(np.abs(y)) - 0.1
    assert np.sum(edge) == 2

    # the ribbon's mirror symmetry makes the two edge moments equal in size
    assert np.isclose(m[edge][0], m[edge][1], atol=1e-4)
    assert np.min(m[edge]) > 0.2  # the edges carry a real moment
    assert np.min(m[edge]) > 4. * np.max(m[~edge])  # and the bulk does not
