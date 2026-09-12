import numpy as np
import pytest

from pyqula import geometry


def _gap(T, nk):
    """Superconducting gap of an attractive-Hubbard Nambu chain"""
    g = geometry.chain()
    h = g.get_hamiltonian()
    h.turn_nambu()
    h = h.get_mean_field_hamiltonian(U=-.6, nk=nk, T=T, mf="random",
                                     maxerror=1e-6)
    return h.get_gap() / 2.


def _curve(nk):
    """Delta(0) and the gap at T/Delta(0) = 1/4, 1/2, 3/4"""
    d0 = _gap(0., nk)
    return d0, np.array([_gap(f*d0, nk) for f in (0.25, 0.5, 0.75)])


@pytest.mark.slow
def test_sc_gap_vs_temperature_is_bcs_like_and_mesh_independent(tmp_path,
                                                                monkeypatch):
    """Delta(0) itself is NOT a property of the superconducting state here:
    on a 1d chain it is a k-mesh artifact, 0.0390 at nk=20 and 0.0055 at
    nk=200 -- a factor of 7 -- so pinning it (which this test used to do)
    records the discretization, not the physics, and any change to the
    default k-mesh handling has to re-record it rather than being checked
    by it.

    What IS mesh-independent is the shape of Delta(T) in reduced units
    T/Delta(0), so that is what is asserted here, at two meshes whose
    Delta(0) differ by more than a factor of two:

    - the gap is exponentially FLAT as T -> 0: it has lost only ~4% by
      T = Delta(0)/4 (measured 0.9576, 0.9576, 0.9577 of Delta(0) at
      nk = 20, 60, 200). An order parameter that fell linearly from
      Delta(0) at T=0 to zero at the BCS Tc = Delta(0)/1.764 would
      already have lost 44% there, so this discriminates the BCS shape
      rather than merely "the gap decreases".
    - the gap is still finite at T = Delta(0)/2 and destroyed by
      T = 3*Delta(0)/4, which brackets the transition temperature as
      Delta(0)/2 < Tc < 3*Delta(0)/4, i.e. 1.33 < Delta(0)/Tc < 2.0 --
      a bracket the published BCS universal ratio Delta(0)/k_B Tc = 1.764
      lies inside. (It is only a bracket: a 1d van Hove density of states
      is not the weak-coupling flat-DOS limit in which 1.764 is exact, so
      asserting the number itself would not be honest.)"""
    monkeypatch.chdir(tmp_path)
    d0_coarse, gs_coarse = _curve(20)
    d0_fine, gs_fine = _curve(60)
    # the premise: the absolute gap really does depend on the mesh
    assert d0_coarse > 2.*d0_fine, (d0_coarse, d0_fine)
    assert d0_fine > 0.
    for (d0, gs) in [(d0_coarse, gs_coarse), (d0_fine, gs_fine)]:
        r = gs/d0 # the gap in reduced units
        assert np.all(np.diff(np.concatenate([[1.], r])) < 0.) # monotonic
        assert r[0] > 0.9   # flat as T -> 0
        assert r[1] > 0.05  # still superconducting at Delta(0)/2
        assert r[2] < 0.01  # gone by 3*Delta(0)/4
    # and the reduced curve itself does not move with the mesh
    assert abs(gs_coarse[0]/d0_coarse - gs_fine[0]/d0_fine) < 1e-2
