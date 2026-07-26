import numpy as np

from pyqula import geometry
from pyqula import meanfield


def _mz(J1, filling=0.2, nk=10, maxerror=1e-6, maxite=300):
    g = geometry.chain()
    h = g.get_hamiltonian(has_spin=True)
    scf = meanfield.SzSz(h, J1=J1, mf="ferroZ", nk=nk, maxerror=maxerror,
            mix=0.3, maxite=maxite, filling=filling)
    assert scf.converged, "SCF did not converge"
    m = scf.hamiltonian.get_magnetization()
    return np.mean(np.abs(m[:, 2]))


def test_szsz_ferromagnetic_moment_grows_with_coupling_strength():
    """H = J1 Sz_i Sz_j on a chain (J1<0, the ferromagnetic sign convention
    -- same convention as the Heisenberg model, where J>0 is
    antiferromagnetic): the self-consistent uniform z moment must grow with
    |J1|. (J1>0, the antiferromagnetic sign, is not exercised here: Neel
    order cannot be represented on this chain's 1-atom unit cell, so the
    SCF loop correctly fails to converge to a translationally-invariant
    fixed point for it -- that is expected chain physics, not a bug.)"""
    mz_weak = _mz(-0.1)
    mz_strong = _mz(-2.0)
    assert mz_strong > 5*mz_weak, \
        f"expected the moment to grow with |J1|: {mz_weak} vs {mz_strong}"
    assert mz_strong > 0.05, "no sizable ferromagnetic moment at strong coupling"
