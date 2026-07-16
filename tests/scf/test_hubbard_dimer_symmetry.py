import numpy as np

from pyqula import geometry
from pyqula import meanfield
from testutils import SCF_MAXERROR


def test_hubbard_dimer_antiferro_moments_are_antisymmetric():
    """By the dimer's inversion symmetry, a self-consistent antiferromagnetic
    Hubbard calculation must converge to equal and opposite local moments on
    the two sites, for any interaction strength."""
    g = geometry.dimer()
    for U in (0.5, 1.0, 2.0, 3.0):
        h = g.get_hamiltonian()
        h.get_mean_field_hamiltonian(filling=0.5, U=U, mf="random",
                                    maxerror=1e-6)
        m = h.get_magnetization()
        assert np.mean(np.abs(m[0] + m[1])) < 1e-5, \
            f"Local moments not antisymmetric at U={U}: {mz}"
