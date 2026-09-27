"""The default initial guess of get_wannier_hamiltonian (no trial_vectors
passed, no disentanglement, not Nambu) is deterministic: h's own orbitals
where the selected bands live, picked by the SCDM column selection
(Damle, Lin and Ying, arXiv:1507.03354) on the wannierization mesh.

It used to be a fresh random matrix on every call, and for the gapped
honeycomb valence band at nk=12 the spread minimization stopped in a local
minimum in 3 calls out of 12: a total spread of about 2.51 instead of 0.3206,
with the Wannier centre away from the site of lower onsite energy. The
returned band was right on the mesh either way (a single band is gauge
invariant there), but the Wannier function, its centre and its spread were
not, and they changed from call to call.
"""
import numpy as np

from pyqula import geometry
from pyqula.wanniertk import wannierize


def _gapped_honeycomb():
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    h.add_onsite([0.8, -0.8]) # the valence band lives on the second site
    return h


def test_repeated_default_calls_give_the_same_minimal_spread_function():
    """Every call lands on the same maximally localized function: spread
    0.3206, centred on the site with the lower onsite energy, with the same
    amplitudes. The old random default fails this: the amplitudes differ by a
    random phase from call to call, and the spread is 2.5 in about a quarter
    of the calls."""
    h = _gapped_honeycomb()
    site = h.geometry.r[1]
    results = [h.get_wannier_hamiltonian(bands=[0, 0], nk=12) for _ in range(4)]
    for hw in results:
        assert abs(hw.wannier_spread_total - 0.3206) < 1e-3
        assert np.linalg.norm(hw.wannier_centres[0] - site) < 1e-6
    first = results[0].wannier_functions
    for hw in results[1:]:
        assert hw.wannier_functions.keys() == first.keys()
        for R, m in first.items():
            assert np.max(np.abs(hw.wannier_functions[R] - m)) < 1e-12


def test_default_trial_orbitals_are_where_the_selected_bands_live():
    """The picked orbitals: the second (lower onsite energy) site for the
    valence band, the first for the conduction band, and every orbital, in
    order, for the full manifold."""
    h = _gapped_honeycomb()
    hk_gen = h.get_hk_gen()
    kpt_latt = wannierize._monkhorst_pack(wannierize._mp_grid(h, 6))
    def pick(bands, num_wann):
        trial = wannierize._default_trial_vectors(hk_gen, kpt_latt, bands, 2, num_wann)
        return [int(np.argmax(np.abs(trial[:, j]))) for j in range(num_wann)]
    assert pick([0], 1) == [1]
    assert pick([1], 1) == [0]
    assert pick([0, 1], 2) == [0, 1]


def test_trial_vectors_passed_by_the_user_are_used(monkeypatch):
    """A trial matrix passed by the user seeds the minimization as given:
    the default is not even computed."""
    def not_called(*args, **kwargs):
        raise AssertionError("the default was computed although trial_vectors was passed")
    monkeypatch.setattr(wannierize, "_default_trial_vectors", not_called)
    h = _gapped_honeycomb()
    hw = h.get_wannier_hamiltonian(bands=[0, 0], nk=12,
                                   trial_vectors=np.array([[0.], [1.]], dtype=complex))
    assert abs(hw.wannier_spread_total - 0.3206) < 1e-3
