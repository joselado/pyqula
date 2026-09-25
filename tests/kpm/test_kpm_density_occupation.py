import numpy as np
import pytest

from pyqula import geometry
from pyqula.kpmtk.density import get_density


def _chain(n=400):
    g = geometry.chain().get_supercell(n)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite(lambda r: 0.3*np.cos(0.7*r[0])) # away from half filling
    m = h.intra
    return m, np.array(m.todense()) if hasattr(m, "todense") else np.array(m)


def test_density_matches_exact_occupation_at_any_fermi():
    """kpmtk.density.get_density used the Fermi energy without dividing it
    by the scale of the expansion and had the wrong constant term, so it
    was right only at fermi=0 and changed with the scale elsewhere. The
    occupation of a site below fermi must match the one from exact
    diagonalization at every fermi, and must not depend on the scale."""
    m, md = _chain()
    e, v = np.linalg.eigh(md)
    i = 200
    for fermi in [-1., -0.5, 0., 0.5, 1.]:
        exact = np.sum(np.abs(v[i, e < fermi])**2)
        ns = [get_density(m, fermi=fermi, i=i, scale=s, npol=400)
              for s in [3., 6.]]
        assert abs(ns[0] - exact) < 1e-2
        assert abs(ns[1] - exact) < 1e-2
        assert abs(ns[0] - ns[1]) < 1e-2


def test_density_default_resolution_and_kernels():
    """With the default number of polynomials, set by delta, and with any
    of the kernels the KPM profiles accept, the occupation must match the
    exact one; an unknown kernel must be refused rather than ignored."""
    m, md = _chain()
    e, v = np.linalg.eigh(md)
    i = 200
    for fermi in [-0.5, 0.5]:
        exact = np.sum(np.abs(v[i, e < fermi])**2)
        assert abs(get_density(m, fermi=fermi, i=i) - exact) < 1e-2
        for kernel in ["jackson", "lorentz", "fejer"]:
            n = get_density(m, fermi=fermi, i=i, npol=400, kernel=kernel)
            assert abs(n - exact) < 1e-2
    with pytest.raises(ValueError, match="jackson"):
        get_density(m, fermi=0., i=i, kernel="gaussian")
