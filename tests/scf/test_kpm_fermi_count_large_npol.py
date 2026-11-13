import numpy as np
from scipy.special import expit

from pyqula import geometry
from pyqula.kpmtk.densitymatrix_kpm import get_fermi4filling_kpm


def test_kpm_fermi_level_at_finite_T_converges_as_npol_grows():
    """At a finite T the KPM Fermi level must hold the requested filling
    in the exact levels of the same Hamiltonian, and get closer to it as
    npol grows, not further away. A large npol resolves each level of a
    finite ring into a step narrower than T/10; a quadrature of the
    Fermi-Dirac convolution on a grid of spacing T/10 missed a fraction
    of that step, so the count was off by 1e-3 electrons at npol=1600.
    The oracle is the Fermi-Dirac count on the exact levels, which shares
    no code with the KPM Fermi search."""
    g = geometry.chain().get_supercell(6)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_onsite(lambda r: 0.3*np.cos(r[0]))
    h.add_exchange([0., 0., 0.4])
    m = h.intra.todense() if hasattr(h.intra, "todense") else h.intra
    e = np.linalg.eigvalsh(np.asarray(m))
    T, filling = 0.3, 0.5
    error = []
    for npol in [200, 800, 1600]:
        mu = get_fermi4filling_kpm(h, filling, nk=1, npol=npol, T=T)
        nelec = np.sum(expit(-(e - mu)/T))
        error.append(abs(nelec - filling*len(e)))
    assert max(error) < 1e-4, error
    assert error[-1] <= error[0], error
