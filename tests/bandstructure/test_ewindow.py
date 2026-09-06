import numpy as np

from pyqula import geometry

# get_bands' `ewindow` filter lived inside the branch of getek taken when
# an operator was given, so h.get_bands(ewindow=...) with no operator --
# and the batched fast path, which is the common case -- returned every
# band with no warning.

WINDOW = 0.5


def _window(e):
    return abs(e) < WINDOW


def _bands(h, **kwargs):
    return h.get_bands(nk=20, write=False, **kwargs)


def test_energy_window_is_applied_without_an_operator():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_zeeman([0., 0., 0.2])
    full = _bands(h)
    inside = _bands(h, ewindow=_window)
    assert np.max(np.abs(full[1])) > WINDOW  # the filter has something to do
    assert np.max(np.abs(inside[1])) < WINDOW
    assert inside.shape[1] < full.shape[1]


def test_the_two_branches_keep_the_same_bands():
    """The operator branch already honoured the window; both must now keep
    exactly the same set of energies."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_zeeman([0., 0., 0.2])
    plain = _bands(h, ewindow=_window)
    withop = _bands(h, operator="sz", ewindow=_window)
    assert plain.shape[1] == withop.shape[1]
    assert np.max(np.abs(np.sort(plain[1]) - np.sort(withop[1]))) < 1e-8


def test_energy_window_on_the_sparse_arpack_path():
    h = geometry.honeycomb_lattice().get_hamiltonian()
    h.add_zeeman([0., 0., 0.2])
    h.turn_sparse()
    out = _bands(h, num_bands=4, ewindow=_window)
    assert out.shape[1] > 0
    assert np.max(np.abs(out[1])) < WINDOW


def test_callback_still_sees_every_band():
    """The window filters the returned rows, not what the callback is
    handed -- the operator branch always worked that way."""
    h = geometry.honeycomb_lattice().get_hamiltonian()
    seen = []
    _bands(h, ewindow=_window, callback=lambda k, es, *a: seen.append(len(es)))
    assert set(seen) == {h.intra.shape[0]}
