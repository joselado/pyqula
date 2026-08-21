"""The two magnon routes must agree where both are valid.

For a plain onsite Hubbard U the site-basis spin RPA of chitk/spinchi.py
is exact -- the transverse ladder rung lives on a single site, which is
precisely what a site-separable vertex can represent -- so its acoustic
magnon has to be the same number the pair-basis TDHF of bsetk/spinflip.py
returns. The two share no code below the Hamiltonian: one scans a
frequency grid for zeros of 1-V*chi(omega) built from site-resolved
response functions, the other diagonalizes a Casida matrix in the
electron-hole pair basis. Agreement is therefore a real cross-check of
both, and the one place where the newer route can be validated against
something other than a symmetry argument.

They stop agreeing as soon as the interaction reaches beyond a site, and
that is not a disagreement to fix: the RPA vertex is zero there (see
future_development/magnons_tdhf.md), so it has no magnon to compare.
"""
import numpy as np
import pytest

from pyqula import geometry
from pyqula.chitk.rpa import build_ops_projectors, rpa_kernel_poles_ops
from pyqula.chitk.spinchi import _full_spin_U, _full_spin_operators

NK = 6


def _rpa_acoustic_magnon(h, q, nk=NK, delta=5e-3):
    """The lowest sharp pole of the site-basis spin RPA kernel at q"""
    Ss = _full_spin_operators(h)
    U = _full_spin_U(h)
    pAs, pBs = build_ops_projectors(h, Ss)
    poles = rpa_kernel_poles_ops(h, V=U, pAs=pAs, pBs=pBs, q=q,
                                  energies=np.linspace(0.005, 2.0, 400),
                                  delta=delta, nk=nk)
    sharp = [p[0] for p in poles if abs(p[1]) < 0.05]
    assert len(sharp) > 0, "the RPA found no sharp collective mode at all"
    return min(sharp)


@pytest.mark.slow
def test_tdhf_and_rpa_magnons_agree_for_an_onsite_hubbard_antiferromagnet():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian()
    hmf = h.get_mean_field_hamiltonian(U=3.0, filling=0.5, mf="antiferro",
                                        nk=NK, maxerror=1e-10)
    for q in ([0.1, 0., 0.], [0.2, 0., 0.]):
        tdhf = hmf.get_magnon_energies(nk=NK, Q=q, n=1)[0].real
        rpa = _rpa_acoustic_magnon(hmf, q)
        assert abs(tdhf - rpa) < 1e-3, f"q={q}: TDHF {tdhf} vs RPA {rpa}"
        assert tdhf > 0.1  # and it is a real dispersing mode, not the zero one
