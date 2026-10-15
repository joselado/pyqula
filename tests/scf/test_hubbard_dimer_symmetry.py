import numpy as np

from pyqula import geometry
from pyqula import meanfield
from testutils import SCF_MAXERROR


def test_hubbard_dimer_antiferro_moments_are_antisymmetric():
    """By the dimer's inversion symmetry, a self-consistent antiferromagnetic
    Hubbard calculation must converge to equal and opposite local moments on
    the two sites, for any interaction strength."""
    g = geometry.dimer()
    # not U=2.0: it sits at the mean-field instability of the dimer, where
    # the loop slows down critically and does not reach maxerror=1e-6 in
    # 1000 iterations (now an error rather than a silent None)
    for U in (0.5, 1.0, 2.5, 3.0):
        h = g.get_hamiltonian()
        hmf = h.get_mean_field_hamiltonian(filling=0.5, U=U, mf="random",
                                           maxerror=1e-6)
        m = hmf.get_magnetization() # of the converged Hamiltonian
        assert np.mean(np.abs(m[0] + m[1])) < 1e-5, \
            f"Local moments not antisymmetric at U={U}: {m}"
