import numpy as np

from pyqula import geometry


def test_compute_vev_accepts_positional_operator():
    """Regression check: Hamiltonian.compute_vev used to only accept
    **kwargs, so calling it positionally (as susceptibility.py's getrow
    does internally: hi.compute_vev("sz", delta=...)) raised a TypeError.
    compute_vev must forward a positional operator to get_vev."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=True)
    h.add_zeeman([0., 0., 0.3])
    out_positional = h.compute_vev("sz")
    out_keyword = h.get_vev(operator="sz")
    assert np.allclose(out_positional, out_keyword)


def test_get_operator_accepts_raw_matrix():
    """Regression check: get_operator only special-cased Operator,
    Hamiltonian, Potential/callable and None inputs; any other object
    (e.g. a raw operator matrix, as returned by operators.get_valley)
    fell through to the string-comparison chain, where `name=="None"`
    on a numpy array raised "truth value of an array is ambiguous"."""
    g = geometry.honeycomb_lattice()
    h = g.get_hamiltonian(has_spin=False)
    raw_matrix = np.eye(h.intra.shape[0], dtype=np.complex128)
    op = h.get_operator(raw_matrix)
    assert op is not None
    assert np.allclose(op.get_matrix(), raw_matrix)
