import numpy as np

from pyqula import geometry
from pyqula.scftk.densitydensity import Vinteraction
from pyqula.scftk.densitydensity_kpm import Vinteraction_kpm
from pyqula.scftk.spinspin import VJinteraction

# The mean field of a Hermitian Hamiltonian is Hermitian, mf[d] =
# mf[-d]^dagger, exactly. The KPM density matrix used to compute the two
# members of each such pair from their own Chebyshev moments, and the
# mean-field loop amplified the anti-Hermitian part that roundoff left in
# them by roughly 1.6 per iteration on the Haldane model below, until the
# Chebyshev recursion diverged. Each test starts from the exact-engine
# fixed point plus an anti-Hermitian seed of 1e-10 and checks that the
# mean field the loop returns carries none of it. The spinful loop grows it
# faster, by roughly 2.6 per iteration, and at 30 iterations it has already
# diverged to NaN, so it runs 20 to fail on a finite number.


def _antihermitian_part(mf):
    return max(np.max(np.abs(m - mf[tuple(-x for x in d)].conj().T))
            for d, m in mf.items() if tuple(-x for x in d) in mf)


def _seed(mf):
    """The same mean field plus an anti-Hermitian onsite part of 1e-10"""
    mf = {d: m.copy() for d, m in mf.items()}
    n = mf[(0,0,0)].shape[0]
    rng = np.random.RandomState(0)
    a = rng.randn(n, n) + 1j*rng.randn(n, n)
    mf[(0,0,0)] = mf[(0,0,0)] + 1e-10*(a - a.conj().T)
    assert _antihermitian_part(mf) > 1e-10
    return mf


def _haldane(has_spin):
    h = geometry.honeycomb_lattice().get_hamiltonian(has_spin=has_spin)
    h.add_haldane(0.15)
    h.add_sublattice_imbalance(0.1)
    return h


def test_vinteraction_kpm_keeps_the_mean_field_hermitian(monkeypatch,
        tmp_path):
    monkeypatch.chdir(tmp_path) # the loops save MF.pkl
    h = _haldane(False)
    kw = dict(V1=1.5, V2=0.5, nk=8, filling=0.3, load_mf=False, verbose=0)
    ed = Vinteraction(h, mf={(0,0,0): np.diag([0.3,-0.3]).astype(complex)},
            mix=0.5, maxite=600, maxerror=1e-9, **kw)
    assert ed.converged
    scf = Vinteraction_kpm(h, mf=_seed(ed.mf), mix=0.5, maxite=30,
            maxerror=1e-15, npol=300, **kw) # maxerror: never stops early
    assert _antihermitian_part(scf.mf) < 1e-12, _antihermitian_part(scf.mf)


def test_vjinteraction_kpm_keeps_the_mean_field_hermitian(monkeypatch,
        tmp_path):
    monkeypatch.chdir(tmp_path)
    h = _haldane(True)
    kw = dict(V1=1.5, V2=0.5, nk=8, filling=0.3, verbose=0)
    ed = VJinteraction(h, mf={(0,0,0): np.diag([0.3,0.3,-0.3,-0.3])
            .astype(complex)}, mix=0.5, maxite=600, maxerror=1e-9, **kw)
    assert ed.converged
    scf = VJinteraction(h, mf=_seed(ed.mf), mix=0.5, maxite=20,
            maxerror=1e-15, integration="kpm", npol=300, **kw)
    assert _antihermitian_part(scf.mf) < 1e-12, _antihermitian_part(scf.mf)
