import numpy as np

from pyqula import geometry, kdos, green, algebra

ES = np.linspace(-0.4, 0.4, 5)
DELTA = 0.05


def _chain():
    """A chain with an in-plane exchange field, so sx is off-diagonal and
    its projected surface DOS is nonzero."""
    h = geometry.chain().get_hamiltonian(has_spin=True)
    h.add_exchange([0.4, 0., 0.])
    return h


def _reference(h, name):
    op = np.array(h.get_operator(name).get_matrix().todense())
    out = []
    for e in ES:
        gs, sf = green.green_renormalization(h.intra, h.inter, energy=e,
                                             delta=DELTA)
        out.append(-algebra.trace(sf @ op).imag)
    return np.array(out)


def _surface_dos(h, operator, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kdos.write_surface_1d(h, energies=ES, delta=DELTA, operator=operator)
    return np.genfromtxt("SURFACE_DOS.OUT").T[1]


def test_offdiagonal_operator_is_a_matrix_product(tmp_path, monkeypatch):
    """gs and sf are plain ndarrays, so `algebra.trace(gs*op)` was an
    elementwise product and the projected DOS was sum_i g[i,i]*op[i,i] --
    identically zero for any off-diagonal operator. sz, being diagonal,
    happened to come out right."""
    h = _chain()
    out = _surface_dos(h, h.get_operator("sx"), tmp_path, monkeypatch)
    assert np.max(np.abs(out - _reference(h, "sx"))) < 1e-8
    assert np.max(np.abs(out)) > 1e-3  # and it is not identically zero


def test_named_operator_does_not_raise(tmp_path, monkeypatch):
    """`elif callable(operator): op = callable(op)` referenced an unbound
    `op`, and an Operator is callable, so every operator obtained from
    h.get_operator raised UnboundLocalError."""
    h = _chain()
    by_name = _surface_dos(h, "sx", tmp_path, monkeypatch)
    assert np.max(np.abs(by_name - _reference(h, "sx"))) < 1e-8


def test_raw_matrix_still_accepted(tmp_path, monkeypatch):
    h = _chain()
    m = np.array(h.get_operator("sx").get_matrix().todense())
    out = _surface_dos(h, m, tmp_path, monkeypatch)
    assert np.max(np.abs(out - _reference(h, "sx"))) < 1e-8
