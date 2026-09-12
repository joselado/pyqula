"""Four arguments that were accepted at the signature, and then dropped
before the code that was supposed to read them -- the failure mode no
unknown-keyword check can catch, because the name is a real parameter.

check.py, vev.py and specialhamiltonian.py have no topic directory of
their own in tests/, so their regression checks live here next to
bandstructure.lowest_bands rather than in a directory this change does
not otherwise touch.
"""

import numpy as np
import pytest

from pyqula import geometry, specialhamiltonian


def _island(n=4):
    g = geometry.chain().supercell(n)
    g.dimensionality = 0
    h = g.get_hamiltonian(has_spin=True)
    h.add_exchange([0.3, 0.5, 0.2])
    return h


def test_check_honours_its_tolerance():
    """h.check(tol=...) threads the value down to check_hermitian, which
    then compared with check.equal's own default of 1e-4 and never with
    the caller's number: a 5e-5 defect passed even at tol=1e-8, and a 5e-3
    defect failed even at tol=1.0. A tolerance that is honoured has to
    change the verdict somewhere between the two."""
    h = _island()
    n = h.intra.shape[0]
    d = 2.5e-5  # anti-Hermitian defect; |h-h^dag| is 2d = 5e-5
    h.intra = np.array(h.intra) + 1j*d*np.identity(n)
    with pytest.raises(ValueError):
        h.check(tol=1e-8)   # far below the defect: must refuse
    h.check(tol=1e-2)       # far above it: must accept


def test_check_raises_on_broken_electron_hole_symmetry():
    """The electron-hole branch printed a message and called exit(),
    terminating the caller's interpreter with status 0 instead of raising
    something that can be caught, logged or reported."""
    h = _island(2)
    h.setup_nambu_spinor()
    n = h.intra.shape[0]
    # a chemical potential added to the WHOLE Nambu space, electrons and
    # holes alike, which is precisely what breaks electron-hole symmetry
    h.intra = np.array(h.intra) + 0.3*np.identity(n)
    with pytest.raises(ValueError):
        h.check()


def test_get_dm_vev_forwards_its_keywords():
    """get_dm_vev took **kwargs and called H.get_density_matrix() with no
    arguments, so every knob of the density matrix was dropped: the vev at
    T=2 was the T=0 number to the last digit."""
    h = _island()
    A = h.get_operator("sz").get_matrix()
    hot = h.get_dm_vev(A, T=2.0)
    cold = h.get_dm_vev(A)
    # the density matrix itself, which the method is a contraction of
    ref = np.trace(A@np.transpose(h.get_density_matrix(T=2.0)))
    assert abs(hot - ref) < 1e-8
    assert abs(hot - cold) > 1e-3  # and T really does change the answer


def test_lowest_bands_honours_nkpoints(tmp_path, monkeypatch):
    """lowest_bands(nkpoints=...) never referenced the argument: with no
    kpath given it took klist.default(h.geometry), whose path length is
    fixed, so every call cost the same and produced the same file."""
    monkeypatch.chdir(tmp_path)
    from pyqula.bandstructure import lowest_bands
    h = geometry.chain().supercell(4).get_hamiltonian(has_spin=False)
    h.add_onsite(0.37)  # keep the arpack shift-invert away from a zero mode
    for nk in [7, 13]:
        lowest_bands(h, nkpoints=nk, nbands=2)
        # one line per band and kpoint (the file's k column is a vector,
        # so it is the line count and not genfromtxt that reads it)
        assert len(open("BANDS.OUT").readlines()) == nk*2


def test_soc_tmdc_honours_soc():
    """SOC_TMDC built its phase with a literal phi=.5 whatever soc was, so
    SOC_TMDC(soc=0.0) and SOC_TMDC(soc=0.9) were bit-identical -- and the
    default of 0.0 reads as 'no SOC', which is not what it gave. soc is
    the strength here (its internal caller TMDC_MX2 has always multiplied
    the result by it), so the Hamiltonian must be linear in it."""
    unit = specialhamiltonian.SOC_TMDC()
    half = specialhamiltonian.SOC_TMDC(soc=0.5)
    off = specialhamiltonian.SOC_TMDC(soc=0.0)
    from pyqula import algebra
    def mats(h):
        return [algebra.todense(h.intra)] + [algebra.todense(t.m)
                                                for t in h.hopping]
    for (mu, mh, mo) in zip(mats(unit), mats(half), mats(off)):
        assert np.max(np.abs(mh - 0.5*mu)) < 1e-12
        assert np.max(np.abs(mo)) < 1e-12
    # the unit SOC is not zero (it lives in the intercell hoppings)
    assert max([np.max(np.abs(m)) for m in mats(unit)]) > 1e-6
