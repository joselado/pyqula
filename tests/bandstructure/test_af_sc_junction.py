import numpy as np
import pytest

from pyqula import geometry
from pyqula import films
from pyqula import algebra
from testutils import temporary_attr


@pytest.mark.slow
def test_af_sc_junction_diamond_film_matches_reference(tmp_path, monkeypatch):
    """Regression check for an antiferromagnet/superconductor junction on
    a diamond-lattice film with a smooth domain wall, at a small
    thickness (nz=6 instead of 20) and coarse mesh (nk=20 instead of 100):
    the sz-resolved band energies must match the values recorded from a
    known-good run. Marked slow: stays a few seconds even at reduced
    thickness/mesh."""
    monkeypatch.chdir(tmp_path)  # writes BANDS.OUT to cwd
    with temporary_attr(algebra, "accelerate", True):
        g = geometry.diamond_lattice_minimal()
        g = films.geometry_film(g, nz=6)

        def get_hamiltonian():
            h = g.get_hamiltonian(is_multicell=True, is_sparse=False)
            def step(z, width=0.00001):
                return (-np.tanh(z / width) + 1.0) / 2.
            h.add_antiferromagnetism(lambda r: 0.5 * step(r[2]))
            h.shift_fermi(lambda r: 0.7 * (-step(r[2], width=0.0001) + 1.0))
            h.add_swave(lambda r: 0.4 * (-step(r[2]) + 1.0))
            h.add_kane_mele(0.1)
            return h

        h = get_hamiltonian()
        (k, e, c) = h.get_bands(operator="sz", nk=20)
        assert np.isclose(np.sum(e), 2.7355895326763857e-13, atol=1e-6)
        assert np.isclose(np.sum(c), 3.9035441545820504e-13, atol=1e-6)
